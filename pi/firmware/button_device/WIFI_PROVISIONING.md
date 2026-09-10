# WiFi setup and moving to a new network

The button device (ESP32) no longer has WiFi credentials compiled into
its firmware. Instead it uses [WiFiManager](https://github.com/tzapu/WiFiManager)
to store the network it's joined to on its own flash, and to offer a
phone-based setup flow whenever it needs a new one. This means moving
the whole setup (Pi + button device) to a different WiFi network --
dropping the baby off at a relative's house, replacing your home router,
etc. -- doesn't require reflashing anything.

## First-time setup

1. Flash `button_device.ino` (see `README.md` in this folder) with no
   `config.h` WiFi fields to fill in -- there are none anymore.
2. Power on the device. Since no network is saved yet, it automatically
   opens a temporary WiFi network named **"BabyMonitor-Setup"**.
3. On a phone (or laptop), open WiFi settings and join
   "BabyMonitor-Setup". No password is needed to join it.
4. A "sign in to network" prompt should pop up automatically (a captive
   portal, the same mechanism hotel/airport WiFi uses); if it doesn't
   pop up on its own, open a browser and go to `192.168.4.1`.
5. Tap "Configure WiFi", pick your home network from the list (or type
   its name if it doesn't show up), enter its password, and submit.
6. The device saves those credentials to its flash, reboots, and joins
   that network. The Serial Monitor (if still connected via USB) will
   print `WiFi configured, connected to "..."` to confirm.

From here on, the device automatically reconnects to that same network
on every boot -- steps 2-6 don't need repeating unless you want to
switch networks (see below) or it forgets its credentials.

## Moving to a new network (e.g. a relative's house)

This assumes the **Pi (monitor unit) also moves with it** and joins the
same new network -- see `docs/INSTALL.md` for adding a second known WiFi
network to the Pi (Raspberry Pi OS supports saving multiple networks and
will auto-connect to whichever one it finds, so the Pi side typically
needs no changes if the new network was set up on it ahead of time).

The button device only stores **one** network at a time, so switching
requires re-running the setup portal:

1. Power on the button device (or reset it) while **holding both
   buttons down**.
2. Keep holding for about 3 seconds -- the Serial Monitor (if connected)
   will print a countdown/confirmation, and the device re-opens the
   "BabyMonitor-Setup" access point, exactly like first-time setup.
3. Follow steps 3-6 above, this time picking the new location's network.

Releasing the buttons early (before the 3-second hold completes) skips
this and boots normally using whichever network was already saved --
this makes it hard to trigger by accident.

If the portal times out (default 3 minutes, `WIFI_PORTAL_TIMEOUT_S` in
`config.h`) without anyone completing setup, the device just goes back
to retrying whatever network it had saved before -- nothing is erased
until a new network is actually submitted through the portal.

## Moving back home

Same steps as above -- hold both buttons at boot, join the portal, pick
the home network again. The device only remembers the most recently
configured network, not a list, so switching back always means running
the portal again. (If switching back and forth between the same two
places often becomes a hassle, storing a small list of known networks
instead of just one would remove the need to re-provision each time --
not implemented here since it wasn't the need this was built for, but
worth revisiting if that becomes the common case.)

## Troubleshooting

- **Portal doesn't show a "sign in" prompt**: manually browse to
  `192.168.4.1` from a device connected to "BabyMonitor-Setup".
- **Forgot to hold both buttons long enough**: just power-cycle and try
  again; nothing is changed by an aborted attempt.
- **Wrong password entered in the portal**: the device will fail to
  join and fall back to retrying -- power-cycle while holding both
  buttons again to reopen the portal and re-enter it correctly.
- **MQTT still can't reach the Pi after switching networks**: confirm
  the Pi actually joined the same new network (check its IP/`hostname
  -I`), and that `MQTT_BROKER_HOST` in `config.h` is set to
  `"raspberrypi.local"` (mDNS) rather than a hardcoded IP from the old
  network -- a static IP from one location won't resolve on another.
