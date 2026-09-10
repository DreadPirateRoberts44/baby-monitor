# Hardware catalog (shopping list)

Everything needed to build both physical units: the **monitor** (Pi +
mic, does cry detection) and the **button device** (ESP32 + 2 buttons,
logs feed/diaper-change events). Organized so you can order it all in
one pass.

Product names/links below are specific starting points from research at
the time this was written — prices, stock, and exact listings on
Amazon/etc. change, so treat these as "search for this" rather than
guaranteed live links. Re-verify price/availability before ordering.
Nothing here is sponsored or specially vetted beyond fitting the
requirements below (works with a Pi/ESP32, reasonable reviews, fits the
"looks like a product" goal) — if you find an equivalent you like better,
any part meeting the same spec works fine, see [pi/HARDWARE.md](../pi/HARDWARE.md)
for the underlying requirements each part needs to satisfy.

## Monitor unit

| Part | What to get | Why |
|---|---|---|
| **Raspberry Pi 4 Model B, 2GB or 4GB RAM** | Search "Raspberry Pi 4 Model B 2GB" (or 4GB). ~$35-45 for the board alone. **Recommended: buy a kit that bundles the board + power supply + SD card** — e.g. the **CanaKit Raspberry Pi 4 Starter Kit** (canakit.com or Amazon) — simpler than sourcing each piece separately and CanaKit's kits are a long-standing reputable source specifically for Pi bundles. | Runs all the Pi-side code. A Pi 5 also works and gives more headroom, but isn't necessary — inference here is lightweight. Don't get less than 2GB RAM. |
| **microSD card, 32GB+, if not already in a kit** | SanDisk or Samsung branded, "A1" or "A2" rated (application performance class — matters for boot/read speed). Avoid unbranded/no-name cards. | Cheap/counterfeit SD cards are a well-known source of Pi reliability problems (corruption, random failures) — not worth saving a few dollars here. |
| **5V/3A USB-C power supply, if not already in a kit** | The **official Raspberry Pi USB-C Power Supply** (search "Raspberry Pi 15W USB-C Power Supply") if not bundled in the kit above. | Underpowered/generic phone chargers cause intermittent instability (random reboots, USB device dropouts) that's hard to diagnose remotely — worth using the specified/official one. |
| **USB microphone** | A small USB condenser desktop/conference mic with a gooseneck or clip, omnidirectional pickup, plug-and-play (no special drivers) — e.g. search "CMTECK USB computer microphone" or "USB conference microphone omnidirectional plug and play" on Amazon; many options in the $10-20 range meet this. Confirm the listing says Linux/"class-compliant USB audio" compatible, or at minimum doesn't say Windows/Mac-only software required — nearly all basic USB mics use the standard USB Audio Class and need no driver on Linux, but double-check the listing doesn't require a bundled Windows app for basic recording. | Placed on furniture a few feet from the crib, not attached to it (see [Microphone placement](../pi/HARDWARE.md#microphone-placement)) — a small desktop mic with a flexible gooseneck is easy to aim and unobtrusive. Any standard USB Audio Class mic works via `sounddevice`/PortAudio, so this doesn't need to be a specific model — just avoid anything requiring proprietary Windows/Mac-only setup software. |
| **Case** | **Argon ONE V2** (search "Argon ONE V2 case Raspberry Pi 4", available via Amazon, thepihut.com, pimoroni.com, argon40.com — roughly $20-25) — aluminum, built-in fan + heatsink lid, ports rerouted to the back for a clean cable look, and has a **built-in power button** you can wire to double as the shutdown button (see [ASSEMBLY.md](ASSEMBLY.md)). Simpler alternative: any basic ventilated Pi 4 case (e.g. **CanaKit Premium Case**, ~$8-10) if you'd rather wire a separate physical shutdown button yourself. | This is what makes the monitor unit look like a finished product instead of a bare circuit board — see [pi/HARDWARE.md](../pi/HARDWARE.md)'s enclosure notes. The Argon ONE's rear port routing and built-in button are worth the extra cost for a device that'll sit visibly in a nursery. |
| **Ethernet cable (optional)** | Any standard cable, length to reach a nearby jack. | More reliable than WiFi for an always-on device if a jack happens to be close. Skip if none is nearby — WiFi works fine. |
| **Momentary push-button (shutdown switch)** — *skip if using the Argon ONE V2's built-in power button* | A generic 12mm momentary push-button (normally-open), any panel-mount style — search "12mm momentary push button panel mount." A few dollars for a pack. | Wired to GPIO3 + GND, triggers a clean shutdown (see [pi/HARDWARE.md](../pi/HARDWARE.md)'s "Off switch" section) so the Pi can be safely powered down without SD-card corruption risk. Not needed if the case has a built-in power button already wired to the right pins (Argon ONE V2 does). |
| **Physical power switch or smart plug** | An inline power switch on the cable, or a basic mechanical power-strip switch. No smart-home features needed. | What actually cuts power *after* the shutdown button has finished its clean shutdown — see [ASSEMBLY.md](ASSEMBLY.md) for the sequence. |

## Button device

| Part | What to get | Why |
|---|---|---|
| **ESP32 dev board** | Any standard **ESP32-WROOM-32 DevKit** board — search "ESP32 DevKitC" or "ESP32-WROOM-32 development board," ~$6-10. Brand doesn't matter much (HiLetgo, DOIT, Espressif-official all work) as long as it's a standard 30 or 38-pin DevKit form factor. | Runs the button firmware (see [pi/firmware/README.md](../pi/firmware/README.md)). WiFi is built in. |
| **2x momentary push-buttons** | 12mm panel-mount momentary buttons — the same generic part as the shutdown button above, just buy 2 extra. Optionally pick two different colors (e.g. one green for feed, one blue for change) so they're distinguishable by feel/glance in a dark room. | Panel-mount means the button threads through a drilled hole in the enclosure and is held with a nut from inside — flush, finished look, no bare PCB tactile switches showing. |
| **USB cable + wall adapter (power)** | Any USB-A-to-Micro-USB or USB-C cable (check which port your specific ESP32 board has) + a basic 5V USB wall adapter (an old phone charger works fine). | Simplest power option — no battery to charge/monitor. See [pi/HARDWARE.md](../pi/HARDWARE.md) for the battery-powered alternative if cable-free placement ever becomes a priority; not necessary for v1. |
| **Enclosure** | A small **ABS plastic junction/project box**, roughly 3-4 inches per side — search "ABS plastic project box enclosure" or "waterproof junction box small" on Amazon (~$8-12 for a 2-3 pack). Pick one large enough to fit the ESP32 board plus wiring with the lid closed. | Two 12mm holes drilled in the lid for the panel-mount buttons is a straightforward first electronics-enclosure project — see [ASSEMBLY.md](ASSEMBLY.md) for the drilling steps. A plain box (not an arcade-style button box, which reads more "gaming" than "baby product") keeps a cleaner look; paint or a wrap can dress it up further if desired. If you have access to a 3D printer, a custom-printed box is a nicer-fitting alternative — search Printables/Thingiverse for "ESP32 DevKit enclosure," but isn't necessary. |

This list builds **one** button device with both buttons together — the
lower-cost default, since it's not yet known whether separate locations
will actually be wanted. If that turns out to matter later, it's an
add-on, not a redo: one more ESP32 (~$8), one more button (~$2, or reuse
the spare from splitting the original pair), and one more small
enclosure (~$10) — no code changes needed. See
[ASSEMBLY.md's "Splitting into two separate button devices"](ASSEMBLY.md#splitting-into-two-separate-button-devices)
for the full walkthrough if/when that's wanted.

## Tools you'll need (not consumed, likely already own some)

| Tool | Used for |
|---|---|
| Phillips screwdriver (small) | Case assembly |
| Drill with a 12mm (or 1/2") step bit or drill bit | Drilling the two button holes in the button-device enclosure |
| Wire strippers | Prepping wires for the shutdown button and the two ESP32 buttons |
| Soldering iron + solder (basic) | Attaching wires to the push-buttons (most panel-mount buttons have solder tabs, not screw terminals) |
| Multimeter (optional but recommended for a first build) | Confirming continuity/wiring before powering on — cheap ($10-15) and catches mistakes before they matter |
| Computer with a USB port | Flashing the ESP32 firmware (see [INSTALL.md](INSTALL.md)) |

## Estimated total cost

- Monitor unit: ~$70-100 (Pi kit ~$50-60 + mic ~$15 + Argon ONE V2 case ~$20-25)
- Button device: ~$20-30 (ESP32 ~$8 + buttons ~$5 + enclosure ~$10 + cable/adapter, often already on hand)
- Tools (if not already owned): ~$25-40 (drill bit, wire strippers, basic soldering iron, multimeter)

**Total: roughly $115-170**, most of it in the monitor unit's Pi/case/mic.

## Ordering checklist

- [ ] Raspberry Pi 4 kit (board + power supply + SD card)
- [ ] USB microphone
- [ ] Argon ONE V2 case (or basic Pi case + separate shutdown button)
- [ ] Ethernet cable (optional)
- [ ] Inline power switch or smart plug
- [ ] ESP32 DevKit board
- [ ] 2-3x 12mm momentary push-buttons (2 for the device, 1 spare/for the shutdown button if not using Argon ONE V2's built-in one)
- [ ] USB cable + wall adapter for the ESP32
- [ ] Small ABS project box for the button device
- [ ] Wire strippers, soldering iron, 12mm drill bit, multimeter (if not already owned)

Once parts arrive, move on to [ASSEMBLY.md](ASSEMBLY.md).
