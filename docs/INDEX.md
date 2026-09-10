# Documentation index

Where to find things. This project has two kinds of docs: **guides**
(step-by-step, for building/setting up the physical device) and
**reference** (what the code does and why, for when you're changing
something). Start with the guides in order if you're building this for
the first time.

## Guides (do these, in order)

| Doc | What it's for |
|---|---|
| [HARDWARE_CATALOG.md](HARDWARE_CATALOG.md) | Shopping list — every part needed, with specific product recommendations, so you can order everything at once. **Start here.** |
| [ASSEMBLY.md](ASSEMBLY.md) | Physical build — wiring, case assembly, putting the two units together. No programming knowledge assumed. Do this after parts arrive. |
| [INSTALL.md](INSTALL.md) | Software setup — getting the code, models, and firmware onto the devices; making everything start automatically; setting up remote access for updates. Do this after (or interleaved with) assembly. |

## Reference (look things up as needed)

| Doc | What it covers |
|---|---|
| [README.md](../README.md) (repo root) | The ML pipeline — training data, stage 1 (cry detection) and stage 2 (cry-reason) models, why certain modeling choices were made. Read this if you're retraining or changing the models. |
| [pi/README.md](../pi/README.md) | What each Raspberry Pi Python module does and why — session tracking, notifications, cry history, care events, app sync API, database reset, pausing predictions. Read this if you're changing the Pi-side code's *behavior*. |
| [pi/HARDWARE.md](../pi/HARDWARE.md) | Physical design reasoning — why two separate units, microphone placement reasoning, the off-switch design, network architecture diagram. The "why" behind [ASSEMBLY.md](ASSEMBLY.md) and [HARDWARE_CATALOG.md](HARDWARE_CATALOG.md)'s choices. |
| [pi/firmware/README.md](../pi/firmware/README.md) | ESP32 button-device firmware reference — message format, offline-queue behavior, testing the firmware in isolation. |
| [experiments/README.md](../experiments/README.md) | Modeling approaches that were tried and did NOT beat the shipped stage-2 model — kept so the same dead ends aren't re-explored. |

## Quick orientation: what's actually in this repo

```
config.py, data.py, model.py, train.py,          Training the ML models
train_cry_reason.py, export_*.py, etc.            (repo root, Python,
                                                    see root README.md)

pi/                                                Raspberry Pi runtime
  monitor.py, session.py, inference.py, ...        (see pi/README.md)
  care_events.py, cry_history.py, ...
  sync_api.py, notify.py, prediction_state.py
  firmware/button_device/                          ESP32 button firmware
                                                    (Arduino C++, see
                                                    pi/firmware/README.md)

docs/                                               This folder — build/
  HARDWARE_CATALOG.md, ASSEMBLY.md, INSTALL.md      install guides
```

**Not in this repo**: the mobile app. This repo is Pi-side + ML only —
the app is a separate project that talks to the Pi over MQTT (alerts)
and the local HTTP sync API (`pi/sync_api.py`), both described in
`pi/README.md`.

## If you're a novice and just want to build this

Read the three guides above in order — [HARDWARE_CATALOG.md](HARDWARE_CATALOG.md)
first (order the parts), then [ASSEMBLY.md](ASSEMBLY.md) (put it
together physically), then [INSTALL.md](INSTALL.md) (get the software
running). They're written assuming no prior Raspberry Pi/electronics/
Linux experience. You shouldn't need to open any of the reference docs
unless something goes wrong or you want to understand *why* something
works the way it does.
