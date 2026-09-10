/*
 * Wireless feed/change button device for the baby monitor project.
 *
 * Two momentary push-buttons (feed, diaper change). On press, connects to
 * WiFi (if not already connected) and publishes an MQTT message to the
 * Pi's local broker. Stays on the local network only -- never talks to
 * the internet or any cloud service (see ../../HARDWARE.md "Network
 * architecture").
 *
 * WiFi provisioning: credentials are NOT compiled in. WiFiManager handles
 * connecting to a previously-saved network, or -- if none is saved/
 * reachable -- puts up a temporary "BabyMonitor-Setup" access point with a
 * captive portal so a phone can be used to pick a network and enter its
 * password, no USB/re-flashing required. This is what makes it practical
 * to move the whole setup (monitor + buttons) to a different WiFi network
 * later -- e.g. a relative's house, or a new home router -- see
 * WIFI_PROVISIONING.md. Holding both buttons at boot forces the portal
 * back open even when a network is already saved, so re-provisioning
 * doesn't require erasing anything first.
 *
 * Offline queue: if WiFi/the broker is unreachable when a button is
 * pressed, the press is held in a small RAM queue instead of being
 * dropped, and retried both on the next button press AND proactively
 * every RETRY_INTERVAL_MS (config.h) so a queued press doesn't sit
 * waiting on someone pressing another button before it's delivered.
 * Queue is RAM-only -- lost if the ESP32 itself resets/loses power while
 * events are queued (a real but narrow edge case: an outage AND a device
 * reset, both overlapping an unflushed press). Not persisted to flash,
 * to avoid flash-wear complexity for that narrow a case.
 *
 * Each queued/sent message includes "seconds_ago": how long ago the
 * button was actually pressed, computed from millis() (a free-running
 * counter since boot -- no wall-clock/NTP sync needed, so this works
 * with no internet dependency). The Pi's buttons_mqtt.py subtracts this
 * from its own current time to log the ORIGINAL press time, not the
 * delivery time -- matters for anything queued during a longer outage.
 *
 * Config (Pi hostname/IP, GPIO pins, queue sizing) lives in config.h, NOT
 * in this file -- see config.h.example. WiFi credentials live in
 * WiFiManager's own flash storage instead, set up via the portal.
 *
 * Hardware: any ESP32 dev board. Each button wired between its GPIO pin
 * and GND -- INPUT_PULLUP is used, so "pressed" = pin reads LOW, no
 * external resistor needed.
 *
 * Required Arduino libraries (install via Library Manager):
 *   - WiFiManager (by tzapu) -- WiFi credential storage + captive-portal
 *     provisioning
 *   - PubSubClient (by Nick O'Leary) -- MQTT client
 *   - ArduinoJson (by Benoit Blanchon) -- builds the JSON payload
 * Board support: install "esp32" via Boards Manager (Espressif Systems).
 */

#include <WiFi.h>
#include <WiFiManager.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>

#include "config.h"

WiFiManager wifiManager;
WiFiClient wifiClient;
PubSubClient mqttClient(wifiClient);

// Debounce: ignore a second trigger on the same pin within this window.
const unsigned long DEBOUNCE_MS = 300;
unsigned long lastFeedPressMs = 0;
unsigned long lastChangePressMs = 0;

unsigned long lastRetryAttemptMs = 0;

struct PendingEvent {
  const char* type;         // "feed" or "change" -- static strings, not owned/freed
  unsigned long pressMillis; // millis() at the moment the button was pressed
};

PendingEvent queue[MAX_QUEUE_SIZE];
int queueCount = 0;

// Reconnects to whatever network WiFiManager has saved. Does NOT open the
// setup portal -- that only happens from runSetupPortal() (first boot, or
// an intentional reset), so a routine outage just retries the known
// network rather than surprising anyone with an open access point.
void connectWiFi() {
  if (WiFi.status() == WL_CONNECTED) return;
  if (WiFi.SSID().length() == 0) return; // nothing saved yet -- portal handles this at boot

  WiFi.mode(WIFI_STA);
  WiFi.begin();

  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < WIFI_CONNECT_TIMEOUT_MS) {
    delay(250);
    Serial.print(".");
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\nWiFi connected, IP: %s\n", WiFi.localIP().toString().c_str());
  } else {
    Serial.println("\nWiFi connection failed (will retry)");
  }
}

// Opens the "BabyMonitor-Setup" captive portal so a phone can pick a
// network and enter its password -- used on first-ever boot (nothing
// saved yet) and whenever forced via a long-press at boot (moving to a
// new network). Blocks until someone configures it or WIFI_PORTAL_TIMEOUT_S
// elapses, at which point it gives up and continues with whatever's saved
// (possibly nothing, in which case connectWiFi() will just keep retrying
// and enqueueing presses until the portal is run again).
void runSetupPortal() {
  Serial.println("Opening WiFi setup portal \"" WIFI_AP_NAME "\" -- "
                  "connect to it from a phone to choose a network");
  wifiManager.setConfigPortalTimeout(WIFI_PORTAL_TIMEOUT_S);
  if (wifiManager.startConfigPortal(WIFI_AP_NAME)) {
    Serial.printf("WiFi configured, connected to \"%s\"\n", WiFi.SSID().c_str());
  } else {
    Serial.println("Setup portal timed out/failed -- will keep retrying any saved network");
  }
}

void connectMQTT() {
  if (mqttClient.connected()) return;
  if (WiFi.status() != WL_CONNECTED) return;

  mqttClient.setServer(MQTT_BROKER_HOST, MQTT_BROKER_PORT);
  Serial.printf("Connecting to MQTT broker %s:%d...\n", MQTT_BROKER_HOST, MQTT_BROKER_PORT);

  if (mqttClient.connect(DEVICE_ID)) {
    Serial.println("MQTT connected");
  } else {
    Serial.printf("MQTT connection failed, rc=%d (will retry)\n", mqttClient.state());
  }
}

// Attempts to publish a single event. Returns true on success. Does NOT
// attempt to (re)connect itself -- callers are expected to have already
// called connectWiFi()/connectMQTT() before this.
bool publishEvent(const char* eventType, unsigned long pressMillis) {
  if (!mqttClient.connected()) return false;

  unsigned long secondsAgo = (millis() - pressMillis) / 1000;

  JsonDocument doc;
  doc["type"] = eventType;
  doc["device_id"] = DEVICE_ID;
  doc["seconds_ago"] = secondsAgo;

  char payload[160];
  size_t len = serializeJson(doc, payload);

  bool ok = mqttClient.publish(MQTT_BUTTON_TOPIC, payload, len);
  Serial.printf("Publish %s (seconds_ago=%lu): %s (%s)\n",
                eventType, secondsAgo, payload, ok ? "ok" : "FAILED");
  return ok;
}

// Adds an event to the offline queue. If the queue is already full, the
// OLDEST queued event is dropped to make room -- under sustained loss of
// connectivity, keeping the most recent presses is more useful than
// keeping the oldest (a "feed" logged very late is not very actionable
// as history anyway once several more presses have piled up behind it).
void enqueueEvent(const char* eventType, unsigned long pressMillis) {
  if (queueCount >= MAX_QUEUE_SIZE) {
    Serial.println("Queue full -- dropping oldest queued event to make room");
    for (int i = 1; i < queueCount; i++) {
      queue[i - 1] = queue[i];
    }
    queueCount--;
  }
  queue[queueCount].type = eventType;
  queue[queueCount].pressMillis = pressMillis;
  queueCount++;
  Serial.printf("Queued %s event (queue size now %d)\n", eventType, queueCount);
}

// Attempts to send every queued event, in order. Stops (leaving the rest
// queued) on the first failure -- if the connection drops mid-flush,
// don't skip ahead and lose ordering/events.
void flushQueue() {
  if (queueCount == 0) return;
  if (!mqttClient.connected()) return;

  int sent = 0;
  while (sent < queueCount) {
    if (!publishEvent(queue[sent].type, queue[sent].pressMillis)) {
      break;
    }
    sent++;
  }

  if (sent > 0) {
    // shift remaining (unsent) events to the front of the queue
    for (int i = sent; i < queueCount; i++) {
      queue[i - sent] = queue[i];
    }
    queueCount -= sent;
    Serial.printf("Flushed %d queued event(s), %d remaining\n", sent, queueCount);
  }
}

void handleButtonEvent(const char* eventType) {
  unsigned long pressMillis = millis();

  connectWiFi();
  connectMQTT();

  bool delivered = mqttClient.connected() && publishEvent(eventType, pressMillis);
  if (!delivered) {
    enqueueEvent(eventType, pressMillis);
  } else {
    // connection is up -- also take the opportunity to flush anything
    // already queued from an earlier outage, oldest first
    flushQueue();
  }
}

void setup() {
  Serial.begin(115200);
  delay(200);

  pinMode(FEED_BUTTON_PIN, INPUT_PULLUP);
  pinMode(CHANGE_BUTTON_PIN, INPUT_PULLUP);

  // Both buttons held at power-on forces the setup portal open even though
  // a network is already saved -- this is how you move the device to a
  // new WiFi network (a relative's house, a new router) without USB/
  // re-flashing. A brief hold is required (not just "pressed at the
  // instant of boot") so a device that happens to power up with a stuck/
  // just-pressed button doesn't accidentally drop into setup mode.
  if (digitalRead(FEED_BUTTON_PIN) == LOW && digitalRead(CHANGE_BUTTON_PIN) == LOW) {
    Serial.println("Both buttons held at boot -- hold for 3s to reset WiFi and reconfigure...");
    unsigned long holdStart = millis();
    bool stillHeld = true;
    while (millis() - holdStart < 3000) {
      if (digitalRead(FEED_BUTTON_PIN) == HIGH || digitalRead(CHANGE_BUTTON_PIN) == HIGH) {
        stillHeld = false;
        break;
      }
      delay(50);
    }
    if (stillHeld) {
      runSetupPortal();
    } else {
      Serial.println("Released early -- continuing with saved WiFi network");
    }
  }

  // First-ever boot (or after the setup portal above failed/was skipped
  // with nothing saved): no point retrying a network that's never been
  // configured, so open the portal automatically rather than making
  // someone discover the button-hold gesture on a brand new device.
  if (WiFi.SSID().length() == 0) {
    runSetupPortal();
  }

  connectWiFi();
  connectMQTT();

  Serial.println("Ready. Waiting for button presses.");
}

void loop() {
  unsigned long now = millis();

  // Buttons read LOW when pressed (INPUT_PULLUP). Simple debounce: only
  // fire once per DEBOUNCE_MS window per button, no interrupt needed --
  // this device does nothing else, so a polling loop is fine.
  if (digitalRead(FEED_BUTTON_PIN) == LOW && (now - lastFeedPressMs) > DEBOUNCE_MS) {
    lastFeedPressMs = now;
    Serial.println("Feed button pressed");
    handleButtonEvent("feed");
  }

  if (digitalRead(CHANGE_BUTTON_PIN) == LOW && (now - lastChangePressMs) > DEBOUNCE_MS) {
    lastChangePressMs = now;
    Serial.println("Change button pressed");
    handleButtonEvent("change");
  }

  // Proactive retry: independent of button presses, periodically attempt
  // to (re)connect and flush anything still queued, so a press made
  // during an outage doesn't sit waiting on someone pressing another
  // button before it gets delivered. Note: connectWiFi() blocks for up
  // to WIFI_CONNECT_TIMEOUT_MS while disconnected, so button presses
  // aren't read during that window -- same blocking behavior as a fresh
  // button press already had, not a new limitation, but worth knowing:
  // don't set WIFI_CONNECT_TIMEOUT_MS very high if that matters to you.
  if (queueCount > 0 && (now - lastRetryAttemptMs) >= RETRY_INTERVAL_MS) {
    lastRetryAttemptMs = now;
    connectWiFi();
    connectMQTT();
    flushQueue();
  }

  // Keep the MQTT client's connection alive (handles pings, incoming
  // messages -- we don't subscribe to anything, but loop() must still be
  // called regularly per the PubSubClient library's requirements).
  if (mqttClient.connected()) {
    mqttClient.loop();
  }

  delay(20);
}
