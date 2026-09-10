# Physical assembly guide

Written for a first-time build — no prior electronics experience
assumed. Do this after parts from [HARDWARE_CATALOG.md](HARDWARE_CATALOG.md)
have arrived and before [INSTALL.md](INSTALL.md) (though you can also
interleave them — e.g. install the software on the Pi's SD card before
it's in its case).

Two separate builds: the **monitor unit** (Part 1) and the **button
device** (Part 2). They don't depend on each other — build in either
order, or in parallel if you have a second set of hands.

## Safety notes before starting

- Always assemble/wire with the Pi and ESP32 **unplugged/unpowered**.
  Only plug in power once wiring is double-checked.
- If soldering for the first time: work in a ventilated area, use a
  helping-hands tool or tape to hold small parts steady, and don't touch
  the tip (it's ~350°C/650°F). A cheap iron and 60/40 rosin-core solder
  are fine for this project — nothing here requires precision soldering.
- A momentary push-button has no "polarity" (no positive/negative side)
  — it doesn't matter which of its two terminals connects to which wire.

---

## Part 1: Monitor unit

### 1.1 Prep the Pi

1. Unbox the Raspberry Pi board. Don't insert the SD card yet if you're
   going to follow [INSTALL.md](INSTALL.md)'s SD card setup first — it's
   easier to write the OS image to the card using another computer
   before it goes in the Pi. If you've already done that, go ahead and
   insert the card into the Pi's microSD slot (on the underside of the
   board, spring-loaded — push until it clicks).

### 1.2 Wire the shutdown button (skip this section entirely if using the Argon ONE V2 case, which has a built-in power button — go to 1.3)

1. Identify the Pi's GPIO pin header (the 2-row, 40-pin header along one
   edge of the board). Pins are numbered left-to-right, top-to-bottom,
   starting at pin 1 nearest the corner marked on the board (usually
   near the USB-C power port).
2. You need **physical pin 5 (GPIO3)** and **physical pin 6 (GND)** —
   these are the 3rd and 4th pins down on the row closer to the edge of
   the board (pin 1 top-left, pin 2 top-right, so pin 5 is the 3rd pin
   down on the left column, pin 6 the 3rd pin down on the right column).
   If unsure, search "Raspberry Pi 4 GPIO pinout" for a labeled diagram
   — GPIO3/pin 5 and any GND pin (there are several; pin 6 is simplest)
   are clearly marked on any pinout reference.
3. Solder a short wire to each terminal of your momentary push-button.
4. Connect one wire to physical pin 5 (GPIO3), the other to physical pin
   6 (GND) — using female-to-female jumper wires (if your button has pin
   leads) or by soldering directly, whichever you have. Order of which
   wire goes to which pin doesn't matter.
5. Mount the button through a hole in your case if it has one, or leave
   it as a small pigtail hanging out through a cable-exit gap in the
   case for now (it just needs to be reachable).

### 1.3 Assemble the case

**If using the Argon ONE V2:**
1. Follow the included instructions — briefly: the Pi's board slots into
   the base, with its HDMI/USB/power ports connecting to the case's
   internal riser board that re-routes them to the back of the case.
2. Attach the top cover (contains the heatsink plate that touches the
   Pi's CPU via a thermal pad, and the fan).
3. The case's built-in power button is already wired correctly to
   trigger a clean shutdown once you enable it in software — see
   [INSTALL.md](INSTALL.md)'s off-switch setup step. No additional
   wiring needed for the button itself.

**If using a basic case + separate shutdown button (from step 1.2):**
1. Route the two shutdown-button wires out through the case's cable
   cutout before closing it.
2. Seat the Pi board into the case per its included instructions
   (usually: align the port cutouts, snap or screw the halves together).
3. Mount the button in the case if it has a spot for one, or let it sit
   just outside as a pigtail — functionally fine either way, just less
   tidy.

### 1.4 Connect the microphone

1. Plug the USB microphone into any USB port on the Pi (USB 3.0 ports,
   usually blue-lined, are fine but not required — a mic doesn't need
   the extra bandwidth).
2. Position the mic (and the whole monitor unit) on furniture near the
   crib — a dresser, shelf, or table, typically a few feet away. **Do
   not** place the microphone in, on, or attached to the crib itself —
   see [pi/HARDWARE.md](../pi/HARDWARE.md#microphone-placement) for why
   (crib-safety guidance against any object/cord in or on the crib, and
   the training data itself was recorded at normal room distance, not
   close-miked).
3. If the mic has a flexible gooseneck, aim it generally toward the crib
   rather than away from the room.

### 1.5 Power and network

1. Connect the Ethernet cable now if using one — plug into the Pi and
   into a wall jack or router port.
2. Connect the inline power switch (or plug into the smart plug) between
   the Pi's power supply and the wall outlet, so you have a way to cut
   power later without unplugging cables directly.
3. **Don't power on yet** if you haven't set up the SD card's software
   yet — go do [INSTALL.md](INSTALL.md)'s Pi setup section first, then
   come back and power on.

Monitor unit assembly is done once the case is closed, the mic is
plugged in and positioned, and (if wired manually) the shutdown button's
wires are connected to GPIO3/GND.

---

## Part 2: Button device

This builds **one device with both buttons together** (feed + change in
a single enclosure) — the default, lower-cost build. Both buttons don't
have to be in the same room to work, but placing them together keeps
this a one-ESP32 build; if you already know you want them in two
separate locations (e.g. a changing table and a kitchen), it's worth
reading [Splitting into two separate button devices](#splitting-into-two-separate-button-devices)
before buying parts, since that path needs a second ESP32 + enclosure.
Otherwise, build the single device now — it's straightforward to split
into two later if it turns out that's actually wanted (see that section
for what changes).

### 2.1 Prep the two buttons

1. Solder a short wire (a few inches, enough to reach across the
   enclosure to the ESP32 board once assembled) to each of the two
   terminals on **both** push-buttons — 4 wires total, 2 per button.
2. If your buttons came in two different colors (recommended — see
   [HARDWARE_CATALOG.md](HARDWARE_CATALOG.md)), keep track of which is
   "feed" and which is "change" as you go; it won't matter for the
   physical wiring (both buttons wire up identically), only for
   remembering which GPIO pin each one ends up on in
   `pi/firmware/button_device/config.h` later (see
   [INSTALL.md](INSTALL.md)).

### 2.2 Drill the enclosure

1. Mark two spots on the lid of your project box, spaced far enough
   apart to comfortably press each button separately (a couple inches),
   and far enough from the edges that the mounting nut has clearance.
2. Drill a 12mm (or whatever size matches your specific buttons — check
   the listing) hole at each mark. Start with a small pilot hole, then
   step up to the full size, or use a step bit if you have one — safer
   and cleaner than drilling straight to 12mm in one pass on thin
   plastic, which can crack it.
3. Deburr the hole edges (a round file, sandpaper, or even a slightly
   larger drill bit spun by hand removes the rough plastic lip).
4. Test-fit each button through its hole before wiring anything —
   confirm it seats flush and the retaining nut threads on from inside.

### 2.3 Wire the buttons to the ESP32

The ESP32's default pin assignment (set in
`pi/firmware/button_device/config.h.example`) is:
- **Feed button** → GPIO4
- **Change button** → GPIO16
- Both buttons' other terminal → any GND pin on the board

(These defaults can be changed later in software if needed — see
[INSTALL.md](INSTALL.md) — but there's no need to unless you have a
reason to.)

1. Identify GPIO4, GPIO16, and a GND pin on your specific ESP32 board —
   the pin labels are usually printed directly on the board's silkscreen
   next to each pin; if not, search "[your board name] pinout" (e.g.
   "ESP32-WROOM-32 DevKit pinout") for a labeled diagram.
2. Connect the feed button's two wires: one to GPIO4, one to any GND
   pin.
3. Connect the change button's two wires: one to GPIO16, one to any GND
   pin (can share the same GND pin as the feed button, or use a
   different one — all GND pins on the board are electrically the same).
4. No resistors needed — the firmware configures these pins with an
   internal pull-up resistor.
5. Double check with a multimeter (continuity/beep mode) if you have
   one: each button should show continuity between its two wires only
   when pressed, and no continuity to anything else when not pressed.

### 2.4 Mount and close up

1. Feed the button wires through into the box body (through a small
   cable-management notch, or just coiled loosely inside if the box is
   roomy enough) and set the buttons into their drilled holes, securing
   each with its retaining nut from inside.
2. Place the ESP32 board inside the box, positioned so its USB port
   lines up with a cutout or the box's opening — mark and cut/drill a
   small notch for the USB cable to exit if the box doesn't already have
   one.
3. **Don't close the lid permanently yet** — leave it easy to reopen
   until after the firmware is flashed and tested (see
   [INSTALL.md](INSTALL.md)), in case you need to re-check wiring.
4. Once everything tests out working, close the lid for good (screws if
   the box has them, or a snap-fit depending on the box style).

### 2.5 Placement

Once tested and working, mount or place the button device wherever is
most convenient for actual use — a changing table, nearby shelf, etc.
Being wireless (WiFi, not wired to the Pi) is exactly what allows this
placement flexibility — see [pi/HARDWARE.md](../pi/HARDWARE.md) for the
reasoning.

---

## Splitting into two separate button devices

The default build above puts both buttons in one enclosure, wired to one
ESP32 — the lower-cost option, and the right starting point if it's not
yet clear whether separate locations will actually be wanted. If it
later turns out the feed and change buttons should live in two different
rooms (e.g. a changing table and a kitchen), nothing about the initial
build has to be undone — this is purely an *addition*, not a rework.

**Why this works without any code changes**: the firmware
(`pi/firmware/button_device/button_device.ino`) already reads its two
button pins independently and publishes whichever one was pressed —
it has no built-in assumption that both buttons share a board. The Pi
side (`buttons_mqtt.py`) subscribes to one shared MQTT topic and reads
which button was pressed from the message itself, not from which
physical device sent it — it doesn't know or care how many button
devices exist. So "two devices" is a hardware/config change, not a
software one.

### What it takes, concretely

**Buy** (see [HARDWARE_CATALOG.md](HARDWARE_CATALOG.md) for the same
parts, just doubled where noted):
- One more ESP32 DevKit board
- One more push-button (you'll have 3 total instead of 2 — the existing
  device keeps one button, the new device gets the other, with your
  original second button now spare/unused — or just buy one more button
  and repurpose the original two-button device's now-unused pin/button
  as a spare)
- One more small project box

**Assemble** (repeat a smaller version of Part 2 above):
- Wire a single button to the new ESP32 — either GPIO pin from
  `config.h` works (`FEED_BUTTON_PIN` or `CHANGE_BUTTON_PIN`, whichever
  matches which event this device should send), same wiring technique as
  2.3. Only one button is wired on this board; the other configured pin
  is simply left unconnected (harmless — the firmware just never sees it
  go low).
- Drill/mount one hole instead of two in the new enclosure (Part 2.2,
  just once).
- Decide which existing button (feed or change) moves to the new device,
  physically remove it from the original enclosure's now-unused hole (or
  just leave it there, unwired, if removing it is more hassle than it's
  worth), and unwire it from the original ESP32's GPIO pin.

**Configure and flash** (repeat [INSTALL.md](INSTALL.md)'s Part 2, once
per device):
- Each device needs its own `config.h`, copied from
  `config.h.example` same as before, with:
  - The same `MQTT_BROKER_HOST` (both devices talk to the same Pi).
    WiFi credentials aren't set in `config.h` at all — each device gets
    joined to the network separately via its own phone-based setup
    portal after flashing (same steps as the first device — see
    [pi/firmware/button_device/WIFI_PROVISIONING.md](../pi/firmware/button_device/WIFI_PROVISIONING.md)).
  - A **different `DEVICE_ID`** per device (e.g.
    `"babymonitor-feed-01"` and `"babymonitor-change-01"`) — this field
    exists specifically for this case; it's sent with every message and
    used as the MQTT client ID, so two devices with the same ID would
    conflict.
  - Only the one relevant `#define` matters for each device's actual
    wiring (`FEED_BUTTON_PIN` on the feed device, `CHANGE_BUTTON_PIN` on
    the change device) — leaving both defined in `config.h` on both
    devices is fine, since only the pin that's actually wired to a
    button will ever trigger.
- Flash each board separately via the Arduino IDE, same steps as
  [INSTALL.md](INSTALL.md)'s Part 2.5 — connect it via USB, select the
  board/port, Upload — then join each one to WiFi via Part 2.6.

**Nothing changes on the Pi.** `buttons_mqtt.py`, `care_events.py`,
`pi/settings.py` — none of it needs edits. Both devices publish to the
same `babymonitor/button` MQTT topic on the same broker, and the Pi logs
whichever event type (`feed`/`change`) each message says it is,
regardless of which device sent it. `device_id` is stored per event
purely as informational metadata (useful later for "which button device
logged this," not required for anything to function).

**Cost of switching later**: roughly one more ESP32 (~$8) + one more
project box (~$10) + (if not reusing the spare) one more button (~$2) —
see [HARDWARE_CATALOG.md](HARDWARE_CATALOG.md) for current pricing —
plus the time to wire, drill, and flash a second small device, similar
effort to Part 2 above but smaller in scope (one button instead of two).

---

## After assembly

Both units are now physically ready but need software:
- The Pi needs its OS, this repo's code, and the trained models — see
  [INSTALL.md](INSTALL.md)'s Part 1.
- The ESP32 needs its firmware flashed — see [INSTALL.md](INSTALL.md)'s
  Part 2 (or [pi/firmware/README.md](../pi/firmware/README.md) for full
  firmware-specific detail).

If anything about the wiring is unclear or doesn't match your specific
board/case, [pi/HARDWARE.md](../pi/HARDWARE.md) has the underlying
reasoning for each choice (why a shutdown button instead of a bare power
switch, why two separate physical units, etc.) which may help troubleshoot
a substitution.
