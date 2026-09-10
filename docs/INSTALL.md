# Software installation guide

Written for a first-time setup — no prior Raspberry Pi/Linux/Arduino
experience assumed. Do this after (or interleaved with)
[ASSEMBLY.md](ASSEMBLY.md). By the end, both physical units will be
running their software and started automatically, and you'll be able to
update either one remotely without physical access.

**Do this whole guide while the device is still with you** — several
steps (enabling SSH, setting up Tailscale, testing remote access) are
much easier before it ships anywhere you can't reach it physically.

## Overview of what you're setting up

1. **Part 1 — the monitor (Raspberry Pi)**: write the OS to the SD card,
   get this repo's code and trained models onto it, install dependencies,
   confirm it runs.
2. **Part 2 — the button device (ESP32)**: install the Arduino IDE,
   configure your WiFi/network settings, flash the firmware.
3. **Part 3 — always-on operation**: set up services so both devices
   start automatically on power-up and recover from crashes on their own.
4. **Part 4 — remote access**: set up Tailscale so you can SSH into the
   Pi from anywhere to check logs or install updates later, without
   needing to physically visit the device.

---

## Part 1: The monitor (Raspberry Pi)

### 1.1 Write the OS to the SD card

1. On your own computer (not the Pi), download **Raspberry Pi Imager**
   from [raspberrypi.com/software](https://www.raspberrypi.com/software/).
2. Insert the microSD card into your computer (using a USB SD card
   reader/adapter if your computer doesn't have a built-in slot).
3. Open Raspberry Pi Imager:
   - **Device**: select your Pi model (Pi 4 or 5).
   - **Operating System**: choose "Raspberry Pi OS (64-bit)" — the
     standard recommended option, not "Lite" (Lite has no desktop, which
     is fine functionally but makes the very first setup slightly more
     manual; the standard version is simpler for a first build).
   - **Storage**: select your SD card. Double-check you've selected the
     SD card and not another drive — this step erases whatever's on the
     selected storage.
4. Click the gear/settings icon (or you'll be prompted) to pre-configure:
   - **Hostname**: leave as `raspberrypi` (the firmware defaults to
     looking for `raspberrypi.local`).
   - **Enable SSH**: turn this on now, using password authentication —
     saves a step later.
   - **Username/password**: set something you'll remember; write it
     down.
   - **Configure WiFi**: enter your home WiFi network name and password
     here so the Pi connects automatically on first boot (skip if you're
     using Ethernet instead).
5. Write the image (this takes several minutes). Once done, insert the
   card into the Pi.

### 1.2 First boot

1. With the SD card inserted, power on the Pi (plug in the power
   supply). Give it 1-2 minutes for the first boot (it's doing some
   one-time setup).
2. From your own computer, on the same WiFi network, try:
   ```
   ssh <username>@raspberrypi.local
   ```
   using the username you set in step 1.1. If that hostname doesn't
   resolve, find the Pi's IP address instead — check your router's
   connected-devices list, or (if you have a monitor/keyboard plugged
   directly into the Pi) run `hostname -I` on the Pi itself — and use
   `ssh <username>@<that-ip-address>` instead.
3. Accept the host key prompt (type `yes`) and enter the password you
   set. You should now have a terminal on the Pi. Everything from here
   is typed into this SSH session (or directly on the Pi if you have a
   keyboard/monitor attached — either works identically).

### 1.3 Install git and clone this repo

```
sudo apt update
sudo apt install -y git
git clone <this-repo's-URL> baby-monitor
cd baby-monitor
```
(Replace `<this-repo's-URL>` with wherever this repository actually
lives — e.g. its GitHub URL.)

### 1.4 Get the trained models onto the Pi

The Pi needs the exported TFLite models, which are generated on your
regular computer (not the Pi — that process needs the full training
dependency stack) and then copied over.

**On your own computer**, from this repo (with the training Python
environment set up per the root [README.md](../README.md)'s Setup
section):
```
python export_tflite.py
python export_yamnet_embedding_tflite.py
python pi/deploy_models.py
```
This populates `pi/models/` on your computer with 5 files (3 `.tflite`
model files, 2 `.txt` label files).

**Copy `pi/models/` to the Pi.** Simplest way, from your own computer's
terminal:
```
scp -r pi/models <username>@raspberrypi.local:~/baby-monitor/pi/
```
(same username/hostname as your SSH connection above). This copies the
whole `models` folder over.

### 1.5 Install Python dependencies

**Back on the Pi** (SSH session):
```
cd ~/baby-monitor/pi
pip install -r requirements.txt
```
This may take a few minutes — it's installing `tflite-runtime`, audio
libraries, and the MQTT client.

### 1.6 Install Mosquitto (the local MQTT broker)

Still on the Pi:
```
sudo apt install -y mosquitto mosquitto-clients
sudo systemctl enable --now mosquitto
```
This is what the button device and the notification system both talk
through — entirely on the local network, no internet/cloud broker
involved. Sanity check it's running:
```
mosquitto_sub -t babymonitor/button
```
This should just hang, waiting for messages (that's correct — nothing's
been published yet). `Ctrl+C` to stop it.

### 1.7 Confirm the microphone is detected

```
python capture.py
```
This lists available audio input devices. Confirm your USB microphone
shows up in the list. If you need to select a specific device (usually
not necessary — the default works if only one mic is plugged in), set
`AUDIO_DEVICE` in `pi/settings.py` to the device's index or name.

### 1.8 Run the monitor manually to confirm it works

```
python monitor.py
```
You should see `Loading models...` then `Ready. Capturing...`, followed
by a line printed every ~4 seconds showing what it currently hears
(e.g. `silence (conf 0.98, ...)`). Try making a sound, or (carefully)
playing a recording of a baby crying near the mic, and confirm the
output changes accordingly. `Ctrl+C` to stop it once confirmed working.

Don't move on until this works — everything else builds on this running
correctly.

---

## Part 2: The button device (ESP32)

This section is a condensed walkthrough; see
[pi/firmware/README.md](../pi/firmware/README.md) for full detail if
anything here needs more context.

### 2.1 Install the Arduino IDE

Download from [arduino.cc/en/software](https://www.arduino.cc/en/software)
on your own computer (not the Pi) — free, available for
Windows/Mac/Linux.

### 2.2 Add ESP32 board support

1. Open the Arduino IDE → `File > Preferences` (Windows/Linux) or
   `Arduino > Settings` (Mac).
2. Under "Additional Boards Manager URLs," add:
   ```
   https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
   ```
3. `Tools > Board > Boards Manager`, search "esp32," install the package
   by Espressif Systems.

### 2.3 Install required libraries

`Tools > Manage Libraries`, install:
- **WiFiManager** (by tzapu) — handles WiFi credential storage and the
  phone-based setup portal (see 2.5 below); this is what lets the
  device be pointed at a WiFi network without compiling it in
- **PubSubClient** (by Nick O'Leary)
- **ArduinoJson** (by Benoit Blanchon)

### 2.4 Configure Pi connection settings

1. In this repo, go to `pi/firmware/button_device/`.
2. Copy `config.h.example` to a new file named `config.h` (same folder).
3. Open `config.h` in any text editor and fill in:
   - `MQTT_BROKER_HOST` — try leaving this as `"raspberrypi.local"`
     first; it works automatically if your router supports mDNS (usually
     does). If the button device has trouble connecting later, set a
     static/reserved IP for the Pi in your router's settings and put
     that IP here instead (e.g. `"192.168.1.42"`).
   - Leave everything else at its default unless you have a specific
     reason to change it (e.g. if you rewired the buttons to different
     GPIO pins than [ASSEMBLY.md](ASSEMBLY.md)'s defaults).

There's no WiFi password to fill in here — that's entered later through
a phone-based setup screen (2.6). `config.h` is still gitignored since
it's local, per-build setup, just without a secret in it now.

### 2.5 Flash the firmware

1. Open `pi/firmware/button_device/button_device.ino` in the Arduino
   IDE.
2. Connect the ESP32 to your computer via USB.
3. `Tools > Board`, select your specific ESP32 board model.
4. `Tools > Port`, select the port the board shows up as.
5. Click **Upload** (the arrow icon in the toolbar).
6. Open `Tools > Serial Monitor` (set baud rate to 115200 in the
   dropdown). Since no WiFi network is saved on a freshly-flashed
   device, you should see it print that it opened the
   "BabyMonitor-Setup" access point.

### 2.6 Join the device to your WiFi network

1. On a phone, open WiFi settings and join the "BabyMonitor-Setup"
   network (no password needed to join it).
2. A "sign in to network" prompt should appear automatically; if not,
   open a browser and go to `192.168.4.1`.
3. Tap "Configure WiFi," select your home network, enter its password,
   and submit.
4. The device saves this and reboots onto your network. The Serial
   Monitor should then show it connecting to WiFi, then attempting the
   MQTT broker — if Mosquitto is already running (Part 1.6), it should
   show "MQTT connected."

This is a one-time step per network — the device reconnects
automatically on every future boot. See
[pi/firmware/button_device/WIFI_PROVISIONING.md](../pi/firmware/button_device/WIFI_PROVISIONING.md)
for how to repeat this later if the device (and Pi) ever move to a
different WiFi network.

### 2.7 Test a button press

With the Serial Monitor still open and Mosquitto running on the Pi
(from Part 1.6), press each button. The Serial Monitor should show a
publish confirmation. On the Pi, in a separate SSH session, run:
```
mosquitto_sub -t babymonitor/button
```
and press a button again — you should see the JSON message appear.

This confirms the button device is fully working. You only need to
flash it once — it's not remotely updatable the way the Pi's code is;
reflashing later requires reconnecting it to a computer via USB.

---

## Part 3: Always-on operation (auto-start, auto-restart)

Right now, `monitor.py` and the button listener only run while you have
a terminal session open and typed the command manually. This section
sets up **systemd** (already built into Raspberry Pi OS — nothing new to
install) so everything starts automatically on boot and restarts on its
own if something crashes.

**On the Pi:**

1. Find the full path to the repo — if you followed Part 1.3, it's
   `/home/<username>/baby-monitor/pi`.

2. Create the monitor service:
   ```
   sudo nano /etc/systemd/system/baby-monitor.service
   ```
   Paste in (replace `/home/pi/baby-monitor/pi` and `pi` with your
   actual path/username):
   ```ini
   [Unit]
   Description=Baby monitor: audio capture and cry detection
   After=network-online.target
   Wants=network-online.target

   [Service]
   Type=simple
   User=pi
   WorkingDirectory=/home/pi/baby-monitor/pi
   ExecStart=/usr/bin/python3 /home/pi/baby-monitor/pi/monitor.py
   Restart=always
   RestartSec=5

   [Install]
   WantedBy=multi-user.target
   ```
   Save and exit (`Ctrl+O`, Enter, `Ctrl+X`).

3. Create the button-listener service:
   ```
   sudo nano /etc/systemd/system/baby-monitor-buttons.service
   ```
   ```ini
   [Unit]
   Description=Baby monitor: wireless button MQTT listener
   After=network-online.target mosquitto.service
   Wants=network-online.target
   Requires=mosquitto.service

   [Service]
   Type=simple
   User=pi
   WorkingDirectory=/home/pi/baby-monitor/pi
   ExecStart=/usr/bin/python3 /home/pi/baby-monitor/pi/buttons_mqtt.py
   Restart=always
   RestartSec=5

   [Install]
   WantedBy=multi-user.target
   ```

4. Create the app-sync-API service:
   ```
   sudo nano /etc/systemd/system/baby-monitor-sync.service
   ```
   ```ini
   [Unit]
   Description=Baby monitor: local app sync API
   After=network-online.target
   Wants=network-online.target

   [Service]
   Type=simple
   User=pi
   WorkingDirectory=/home/pi/baby-monitor/pi
   ExecStart=/usr/bin/python3 /home/pi/baby-monitor/pi/sync_api.py
   Restart=always
   RestartSec=5

   [Install]
   WantedBy=multi-user.target
   ```

5. Enable and start all three:
   ```
   sudo systemctl daemon-reload
   sudo systemctl enable --now baby-monitor.service
   sudo systemctl enable --now baby-monitor-buttons.service
   sudo systemctl enable --now baby-monitor-sync.service
   ```

6. Check they're running:
   ```
   sudo systemctl status baby-monitor.service
   sudo systemctl status baby-monitor-buttons.service
   sudo systemctl status baby-monitor-sync.service
   ```
   Look for `active (running)` in green (`q` to exit each status view).

7. View live logs any time (useful for debugging):
   ```
   sudo journalctl -u baby-monitor.service -f
   ```
   (`-f` follows live; `Ctrl+C` to stop watching — doesn't stop the
   service.)

### Set up the shutdown button

If using a separate physical shutdown button (not the Argon ONE V2's
built-in one — see [ASSEMBLY.md](ASSEMBLY.md)):
```
sudo nano /boot/firmware/config.txt
```
(on older Raspberry Pi OS versions, the path is `/boot/config.txt`
instead). Add this line at the end:
```
dtoverlay=gpio-shutdown
```
Save, then `sudo reboot`. Pressing the button now triggers a clean
shutdown — wait ~15-20 seconds (until the Pi's activity LED stops
blinking) before cutting power at the switch/smart plug.

If using the **Argon ONE V2**'s built-in power button instead, its setup
is a separate script — see the case's own included instructions
(typically a one-line install command from Argon40) rather than the
overlay above.

### Test that it actually recovers on its own

Before considering this done:
```
sudo reboot
```
Wait a minute or two, SSH back in, and check
`sudo systemctl status baby-monitor.service` again — it should already
show `active (running)` without you doing anything after the reboot
finished.

---

## Part 4: Remote access (Tailscale)

This lets you SSH into the Pi from anywhere — useful once the device is
somewhere you can't physically reach (e.g. shipped to family). **Set
this up now, while the Pi is still with you.**

1. **Create a Tailscale account**: go to
   [tailscale.com](https://tailscale.com), sign up (a Google/Microsoft/
   GitHub account works, no separate password).

2. **Install Tailscale on your own computer**: download from
   [tailscale.com/download](https://tailscale.com/download), sign in
   with the account above.

3. **Install Tailscale on the Pi** (while it's still on your network):
   ```
   curl -fsSL https://tailscale.com/install.sh | sh
   sudo tailscale up
   ```
   The second command prints a URL — open it in a browser and sign in
   with the same account to approve the Pi joining your network.

4. **Find the Pi's Tailscale address:**
   ```
   tailscale ip -4
   ```
   Prints something like `100.x.y.z` — write this down. It reaches the
   Pi from any of your Tailscale-connected devices, anywhere, regardless
   of what network the Pi is actually on.

5. **Test SSH via Tailscale, before the Pi leaves your hands** — from
   your own computer (not the Pi):
   ```
   ssh <username>@<tailscale-ip>
   ```
   If this logs you in, this exact command is what you'll use for remote
   access once the device is at its final location.

6. Confirm Tailscale starts automatically on boot (usually already the
   case):
   ```
   sudo systemctl enable --now tailscaled
   ```

**After the device ships**: as long as it has any internet connection
(WiFi is enough — no port forwarding or router configuration needed at
the far end), `ssh <username>@<tailscale-ip>` reaches it from anywhere.
Nobody at the house needs to do anything for this to keep working.

### Deploying a code update later

```
ssh <username>@<tailscale-ip>
cd ~/baby-monitor
git pull
sudo systemctl restart baby-monitor.service
sudo systemctl restart baby-monitor-buttons.service
sudo systemctl restart baby-monitor-sync.service
```
That's the full remote-update workflow once everything above is set up.

---

## Done

At this point:
- The monitor unit boots, connects to WiFi, and starts detecting cries
  automatically.
- The button device is flashed and publishing button presses.
- Both recover automatically from a crash or power cycle.
- You can SSH in from anywhere via Tailscale to check on it or push an
  update.

For what to do if something isn't behaving as expected, or to understand
*why* the software is built the way it is, see [pi/README.md](../pi/README.md)
(module-by-module reference) — this install guide intentionally doesn't
explain the reasoning behind each design choice, just the steps to get
it running.
