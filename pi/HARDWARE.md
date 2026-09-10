# Hardware design reasoning

Two physical units: the **monitor** (Raspberry Pi + microphone, does the
cry detection) and the **button device** (ESP32 + 2 buttons, logs
feed/diaper-change events). They talk over the home WiFi network only —
no internet round-trip for button presses (see [Network architecture](#network-architecture)).

**Looking for a shopping list or build steps?** This doc explains *why*
the hardware is designed this way — for what to actually buy, see
[docs/HARDWARE_CATALOG.md](../docs/HARDWARE_CATALOG.md); for physical
assembly steps, see [docs/ASSEMBLY.md](../docs/ASSEMBLY.md).

## Microphone placement

Do **not** mount the microphone in or attached to the crib — this
matches standard crib-safety guidance against any cord, camera, or
object in/on the crib (strangulation/choking hazard), and matches how
commercial audio baby monitors are actually placed in practice. Place
the whole monitor unit on furniture near the crib (a dresser, shelf, or
table — typically a few feet away). Infant crying is loud (60-90 dB at
close range) and this project's own training data was recorded at
normal room distance, not close-miked — placing the mic very close to
the baby would actually be a mismatch with what the model was trained
on, not just a safety issue.

## Off switch

To let the Pi be powered off when the baby is away (saving power/heat —
both end up near 0W once off, so a clean shutdown and a hard power cut
save the same amount) without risking database/SD-card corruption from
yanking power while something's mid-write, this uses Raspberry Pi OS's
built-in shutdown-button support (the `dtoverlay=gpio-shutdown` kernel
overlay, wired to a momentary push-button between GPIO3 and GND) rather
than a bare power switch. Pressing the button triggers a clean shutdown
— same as running `sudo shutdown -h now` — no custom application code
needed, this is a hardware/OS-config change. For the actual wiring steps
see [docs/ASSEMBLY.md](../docs/ASSEMBLY.md); for the one-line config
change and testing it, see [docs/INSTALL.md](../docs/INSTALL.md).

`care_events.sqlite` (SQLite) is already reasonably crash-tolerant on
its own via its journal, but a clean shutdown removes essentially all of
the residual risk that a raw power cut carries.

## Network architecture

```
        Home WiFi router (no internet dependency for this part)
              |                    |                    |
       [Button device]       [Monitor unit]      [Caregiver phones]
        ESP32, WiFi         Raspberry Pi,          app, WiFi (each
                             WiFi/Ethernet          family member)
              |                    |                    |
              +-- MQTT publish --->|                    |
                (local network    Mosquitto broker       |
                 only)            (on the Pi)            |
                                    |    |                |
                              buttons_    notify.py        |
                              mqtt.py     (publishes        |
                              subscriber  babymonitor/       |
                                    |     notify) ---MQTT---> app subscribes,
                              care_events.py           |      alerts locally
                              cry_history.py (SQLite)  |      (see README.md's
                                    ^                   |      "Notifications")
                                    |                            |
                              sync_api.py (HTTP, port 8081)        |
                                    |<--- GET/POST/DELETE --------+
                                          (only reachable when the
                                           phone is on this same
                                           WiFi network)
```

- The ESP32 connects to the same home WiFi network as the Pi and
  publishes an MQTT message directly to the Pi's local IP address when a
  button is pressed. The Pi runs the MQTT broker itself (Mosquitto) — no
  cloud/external broker involved, so this works even if the house's
  internet connection is down.
- **`notify.py` publishes cry alerts/updates to the same broker, on a
  separate topic** (`babymonitor/notify`), for the app to subscribe to.
  Deliberately not a push-notification service (APNs/FCM) — no internet
  or cloud account needed for this either; the app is expected to stay
  connected to the broker (foregrounded, or kept alive via a
  platform-appropriate background mechanism) and alert locally. See
  `pi/README.md`'s "Notifications" section for the full reasoning.
- **The app sync API (`sync_api.py`) is also local-network-only** — same
  as the MQTT paths, no internet/cloud component. A caregiver's app is
  expected to sync opportunistically whenever it detects it's on this
  same WiFi network, not push in real time from anywhere — see
  `pi/README.md`'s "App sync" section for the reasoning and what this
  means for grandparents/remote family specifically.
- **Remote access (Tailscale, for you to SSH in and update software) is
  a separate, independent system** — it does need the internet to reach
  Tailscale's coordination servers and establish the tunnel. If the
  internet is down, button presses, cry detection, and app sync (as long
  as the phone's on the home WiFi) keep working; only remote SSH access
  would be unavailable until it's back.
- The Pi's local IP can change if it's not set to a static/reserved
  address on the router — see `pi/firmware/README.md` for how the
  ESP32 finds the Pi (mDNS hostname, so a fixed IP isn't required). The
  app would need the same consideration once it's built.

## Shopping list and cost

See [docs/HARDWARE_CATALOG.md](../docs/HARDWARE_CATALOG.md) for the full
parts list with specific product recommendations and an itemized cost
estimate.
