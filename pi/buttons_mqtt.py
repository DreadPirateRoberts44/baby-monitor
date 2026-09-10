"""Wireless button listener: subscribes to MQTT messages from the ESP32
button device (see firmware/button_device/) and logs a care_events entry
for each feed/change button press.

Replaces an earlier direct-GPIO design (wired buttons on the Pi itself)
with a wireless one, so the button device can be placed wherever is most
convenient (e.g. a changing table across the room) rather than needing to
be wired to the Pi. The tradeoff is real added complexity -- a second
physical device with its own firmware, a message broker to run, and a
network dependency (local WiFi, not internet -- see HARDWARE.md's
"Network architecture" section) -- but was chosen deliberately for
placement flexibility.

Requires an MQTT broker running locally on the Pi (Mosquitto -- see
README.md's Setup section for install instructions). This subscriber
connects to that LOCAL broker only; nothing here talks to the internet.

Expected message: topic settings.MQTT_BUTTON_TOPIC, JSON payload
    {"type": "feed", "device_id": "...", "seconds_ago": 0}
matching firmware/button_device's publish format. "seconds_ago" is how
long ago the button was actually pressed (computed on the ESP32 from
millis(), a free-running counter -- no wall-clock/NTP sync needed). For
a press delivered immediately this is ~0; for one that was queued during
a WiFi/broker outage and delivered once connectivity returned (see the
firmware's offline-queue logic), it reflects the actual press time, not
the delivery time. This subscriber subtracts it from its own current
time so care_events.py logs the original press, not when the message
happened to arrive.
"""
import datetime
import json

import paho.mqtt.client as mqtt

import settings
from care_events import log_event, FEED, CHANGE

# MQTT payload "type" values -> care_events event types.
_TYPE_MAP = {"feed": FEED, "change": CHANGE}


def _on_connect(client, userdata, flags, reason_code, properties=None):
    if reason_code == 0:
        print(f"[buttons_mqtt] connected to broker at "
              f"{settings.MQTT_BROKER_HOST}:{settings.MQTT_BROKER_PORT}")
        client.subscribe(settings.MQTT_BUTTON_TOPIC)
        print(f"[buttons_mqtt] subscribed to {settings.MQTT_BUTTON_TOPIC}")
    else:
        print(f"[buttons_mqtt] connection failed, reason code {reason_code}")


def _on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"[buttons_mqtt] ignoring malformed message on {msg.topic}: {e}")
        return

    event_type = _TYPE_MAP.get(payload.get("type"))
    if event_type is None:
        print(f"[buttons_mqtt] ignoring message with unknown type: {payload}")
        return

    seconds_ago = payload.get("seconds_ago", 0)
    try:
        seconds_ago = max(0, float(seconds_ago))
    except (TypeError, ValueError):
        seconds_ago = 0
    press_time = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=seconds_ago)

    event_id = log_event(event_type, source="button", timestamp=press_time)
    delay_note = f", {seconds_ago:.0f}s delayed (queued during an outage)" if seconds_ago >= 5 else ""
    print(f"[buttons_mqtt] logged {event_type} event ({event_id}) "
          f"from device {payload.get('device_id', 'unknown')}{delay_note}")


def run_button_listener():
    """Blocks forever, listening for button-press messages over MQTT."""
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = _on_connect
    client.on_message = _on_message

    client.connect(settings.MQTT_BROKER_HOST, settings.MQTT_BROKER_PORT)
    client.loop_forever()


if __name__ == "__main__":
    run_button_listener()
