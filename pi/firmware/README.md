# ESP32 button device firmware

Firmware for the wireless feed/diaper-change button device (see
`../HARDWARE.md` for the physical parts list and network architecture).
This is Arduino C++, a completely separate toolchain from the rest of
this repo (which is Python) -- flashed once onto the ESP32 board, not run
alongside the Pi's Python code.

## What it does

Two buttons wired to GPIO pins. On press, connects to WiFi (if not
already connected) and publishes an MQTT message to the Pi's local
broker: `{"type": "feed", "device_id": "...", "seconds_ago": 0}` or the
same with `"type": "change"`. The Pi's `buttons_mqtt.py` subscribes to
this and logs the event. No internet connection is used or required --
everything stays on the local WiFi network.

**WiFi setup is phone-based, not compiled in:** the device doesn't have
a WiFi network baked into its firmware. On first boot (or after an
intentional reset) it opens its own "BabyMonitor-Setup" WiFi network
with a captive portal for picking the real network from a phone. This
also means moving the whole setup to a different network later (a
relative's house, a new router) doesn't need a reflash -- see
[button_device/WIFI_PROVISIONING.md](button_device/WIFI_PROVISIONING.md)
for the full setup and network-switching steps.

**Offline queue:** if WiFi or the broker is unreachable when a button is
pressed, the press isn't dropped — it's held in a small in-memory queue
and retried both on the next button press and proactively every
`RETRY_INTERVAL_MS` (`config.h`, default 10s). `seconds_ago` in the
published message is how long ago the button was actually pressed
(tracked via `millis()`, a free-running counter since boot — no
wall-clock/internet time sync needed), so a press delivered late once
connectivity returns still gets logged with its real press time, not the
time it happened to arrive. The queue is RAM-only: it's lost if the
ESP32 itself resets or loses power while something's still queued (not
persisted to flash, to avoid flash-wear complexity for what's a fairly
narrow edge case).

**One device or two:** the default build wires both buttons to one
ESP32. Nothing here assumes that — each GPIO pin is read and published
independently, and the Pi side doesn't know or care how many physical
devices are sending messages, only what topic they land on and what
`type` each message says. If feed and change ever need to live in two
different rooms, splitting into two single-button devices needs no
firmware/Python changes, just a second board + a distinct `DEVICE_ID`
per device — see [docs/ASSEMBLY.md](../../docs/ASSEMBLY.md#splitting-into-two-separate-button-devices)
for the full walkthrough.

## One-time setup (do this before the device ships anywhere)

### 1. Install the Arduino IDE

Download from [arduino.cc](https://www.arduino.cc/en/software) (free,
available for Windows/Mac/Linux). This is the tool used to write firmware
to the ESP32 -- you don't need to know C++ deeply to follow these steps,
just to fill in a few configuration values.

### 2. Add ESP32 board support

1. Open the Arduino IDE.
2. Go to `File > Preferences` (Windows/Linux) or `Arduino > Settings` (Mac).
3. In "Additional Boards Manager URLs", add:
   ```
   https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
   ```
4. Go to `Tools > Board > Boards Manager`, search for "esp32", install
   the package by Espressif Systems.

### 3. Install required libraries

Go to `Tools > Manage Libraries`, install:
- **WiFiManager** (by tzapu) — WiFi credential storage + captive-portal
  setup (this is what lets you configure/change the WiFi network from a
  phone instead of compiling it in)
- **PubSubClient** (by Nick O'Leary) — MQTT client
- **ArduinoJson** (by Benoit Blanchon) — builds the message payload

### 4. Configure the Pi address

1. In `pi/firmware/button_device/`, copy `config.h.example` to `config.h`.
2. Open `config.h` in the Arduino IDE (or any text editor) and fill in:
   - `MQTT_BROKER_HOST` — try `"raspberrypi.local"` first (works
     automatically if your router supports mDNS and the Pi has `avahi`
     running, which is the Raspberry Pi OS default). If that's
     unreliable, set a static IP for the Pi in your router's settings
     and use that IP instead (e.g. `"192.168.1.42"`) -- though note a
     hardcoded IP is specific to one network and won't work if the whole
     setup later moves elsewhere (see WIFI_PROVISIONING.md), so mDNS is
     preferred if it's reliable on your router.
3. `config.h` is gitignored — it's local setup, not meant to be
   committed or shared. There's no WiFi password in it anymore; that's
   entered later via the phone-based setup portal (next section).

### 5. Flash the firmware

1. Open `button_device.ino` in the Arduino IDE (it will open the whole
   `button_device/` folder, including `config.h`).
2. Connect the ESP32 to your computer via USB.
3. `Tools > Board`, select your specific ESP32 board model.
4. `Tools > Port`, select the port the board shows up as.
5. Click Upload (the arrow icon). Watch the Serial Monitor
   (`Tools > Serial Monitor`, 115200 baud) for status — since no WiFi
   network is saved yet, it should print that it opened the
   "BabyMonitor-Setup" portal.

### 6. Join the device to your WiFi network

See [button_device/WIFI_PROVISIONING.md](button_device/WIFI_PROVISIONING.md)
for the full phone-based setup steps (join "BabyMonitor-Setup" from a
phone, pick your real network, enter its password).

You should only need to reflash (steps 2-5) once per device, unless you
want to change GPIO pins or firmware logic later — reflashing requires
physically connecting the ESP32 to a computer via USB again (this is not
remotely updatable the way the Pi's Python code is via Tailscale/SSH —
see the repo root's remote-access setup for that distinction). **Changing
or adding a WiFi network does NOT require reflashing** — that's exactly
what the setup portal (step 6 / WIFI_PROVISIONING.md) is for.

## Wiring

Each button goes between its configured GPIO pin (`FEED_BUTTON_PIN` /
`CHANGE_BUTTON_PIN` in `config.h`, default GPIO4 and GPIO16) and GND.
`INPUT_PULLUP` is set in firmware, so no external resistor is needed —
the pin reads LOW when the button is pressed.

## Testing without the Pi's MQTT broker running yet

The Serial Monitor (115200 baud) prints connection status and every
button press/publish/queue event, including failures. Useful for
confirming the WiFi/button-reading side works even before
`pi/buttons_mqtt.py` and the Mosquitto broker are set up on the Pi — see
the repo root `pi/README.md` Setup section for that side.

## Testing the offline queue

With the Pi's Mosquitto broker set up and `buttons_mqtt.py` running:
1. Turn off the Pi's WiFi (or stop Mosquitto: `sudo systemctl stop
   mosquitto`) to simulate an outage.
2. Press a button on the ESP32. The Serial Monitor should show a
   "Queued ... event" message rather than a publish failure with nothing
   further happening.
3. Wait a bit (or press again), then restore the Pi's WiFi/restart
   Mosquitto (`sudo systemctl start mosquitto`).
4. Within `RETRY_INTERVAL_MS`, the Serial Monitor should show the queued
   event being flushed. Check `buttons_mqtt.py`'s output on the Pi — it
   should show the event logged with a "delayed" note if the gap was
   more than a few seconds, and `care_events.time_since()` for that
   event type should reflect roughly when it was actually pressed, not
   when Mosquitto came back.
