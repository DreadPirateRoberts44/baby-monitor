# Pi runtime

Always-on monitor loop: capture audio from a microphone, run the two-stage
model, and notify on a detected cry. Deliberately separate from the repo
root's training code — see the module docstrings for why (mainly: a much
lighter dependency footprint, meant to run on a Raspberry Pi).

**Building or setting this up for the first time?** This is a reference
doc (what each module does and why) — for step-by-step build/install
instructions, see [docs/](../docs/INDEX.md): start with
[HARDWARE_CATALOG.md](../docs/HARDWARE_CATALOG.md) (what to buy),
then [ASSEMBLY.md](../docs/ASSEMBLY.md) (physical build), then
[INSTALL.md](../docs/INSTALL.md) (software setup).

## Files

```
settings.py       Pi-side configuration (audio params, model paths,
                   thresholds). Mirrors a subset of the repo root's
                   config.py — kept separate to avoid pulling in the full
                   training dependency stack. If you change
                   config.SAMPLE_RATE, config.DURATION, or
                   config.CONFIDENCE_THRESHOLD during a retrain, update the
                   matching constants here too.
highpass.py        Standalone copy of the 300Hz high-pass filter stage 2
                   was trained with (see audio_preprocessing.py at the
                   repo root). Kept separate rather than imported so this
                   doesn't pull in librosa.
inference.py        Loads the three TFLite models and runs the two-stage
                   pipeline on a captured waveform. Uses tflite_runtime if
                   available, falls back to tensorflow.lite.Interpreter
                   for local testing.
capture.py         Continuous microphone capture via sounddevice, chunked
                   into fixed-length windows.
session.py         Tracks whether a crying session is ongoing across
                   consecutive windows, so notifications fire once at
                   session start/end rather than once per window (see
                   Session tracking below).
notify.py           Publishes cry alerts/live-updates to the app over MQTT
                   (the same local broker buttons_mqtt.py uses) — see
                   Notifications below. No push service (APNs/FCM); the
                   app is expected to stay connected and alert locally.
monitor.py         Main loop: capture -> inference -> session tracking ->
                   notify. Entry point.
care_events.py     Local SQLite-backed store for caregiver-logged events
                   ("fed", "changed") — see Care events below.
cry_history.py     Local SQLite-backed store for completed crying
                   sessions (start time, end time, aggregated cry-reason)
                   — see Cry history below. Deliberately does NOT store
                   laugh/silence/noise detections or raw audio.
device_events.py   Local SQLite-backed store for monitor startup/shutdown
                   events (NOT power outages/crashes as such) — see
                   Device events below. Purely so gaps in cry_history
                   during historical review are explainable.
prediction_state.py App-controlled pause/resume flag for stage1/stage2
                   inference — see Pausing predictions below. The
                   escape hatch for a misbehaving model.
buttons_mqtt.py    Subscribes to the wireless ESP32 button device over
                   MQTT and calls care_events.log_event() on each press.
                   Entry point for the button listener. Requires a local
                   MQTT broker (Mosquitto) running on the Pi — see Setup.
firmware/          ESP32 firmware for the wireless button device (Arduino
                   C++, separate toolchain) — see firmware/README.md.
sync_api.py        Local HTTP API for app<->Pi sync (read/write
                   care_events, read/write prediction_state, read-only
                   cry_history + device_events, plus the reset endpoint)
                   — see App sync below. LAN-only, no auth.
reset_db.py        Wipes care_events, cry_history, and device_events —
                   see Database reset below. Triggered only via
                   sync_api.py's POST /reset; the app owns all
                   confirmation/warning UX for this.
deploy_models.py   Copies exported models from the repo root's output/
                   into pi/models/ (run from the repo root's Python env).
models/            Populated by deploy_models.py; gitignored (same reasons
                   as the repo root's output/ — regenerate, don't commit).
```

See `HARDWARE.md` for the physical parts list (Pi, mic, ESP32, buttons,
enclosures) and network architecture, and `REMOTE_ACCESS.md` for how to
set up Tailscale (remote SSH) and systemd (auto-start/restart) so the
device can be maintained remotely once it's shipped somewhere you can't
physically access.

## Why stage 1 and stage 2 each get their own embedding

Stage 1 (`classifier_head.tflite`) was trained on the *unfiltered*
waveform's YAMNet embedding. Stage 2 (`cry_reason_mlp.tflite`) was trained
on the *300Hz-high-pass-filtered* waveform's embedding — that filter was
found to meaningfully improve stage-2 accuracy (see the repo root
README's "Preprocessing" section) but was never applied to stage 1.
`inference.py` computes two separate embeddings per detected-cry window
(`embed_raw` for stage 1, `embed_filtered` for stage 2) rather than
sharing one, specifically to avoid feeding either model a distribution it
wasn't trained on.

## Session tracking

A single crying episode spans many consecutive capture windows (a real cry
doesn't stop after 4 seconds), so `monitor.py` doesn't notify on every
"cry"/"uncertain" window — that would fire a notification roughly every
`CAPTURE_INTERVAL_SECONDS` for the entire episode (a 5-minute cry at the
default 4s interval would otherwise be ~75 notifications).

`session.py`'s `CrySession` tracks state across windows instead, and
`monitor.py` reacts to the events it returns — deliberately split into
**alerts** (should interrupt/notify the caregiver) and a **live update**
(should refresh an already-visible session without a new alert):

- **`"started"` (alert → `notify_cry_started`)**: fires once
  `settings.SESSION_START_MIN_WINDOWS` *consecutive* confidently-"cry"
  windows have been seen (derived from `SESSION_START_MIN_SECONDS`, default
  6s of confirmation — see `settings.py`). A single confident-but-wrong
  window (a bark, a static burst, anything that briefly clears
  `STAGE1_CONFIDENCE_THRESHOLD`) used to be enough on its own to fire a
  full alert; requiring a short consecutive run first filters that out at
  the cost of a few seconds of alert latency. Before confirmation, the
  candidate is in a `PENDING` sub-state: any confident NON-cry window
  before it's confirmed discards the candidate entirely (deliberately
  stricter than the merge window below — an unconfirmed candidate has no
  track record yet, so one interruption means "that wasn't really a cry").
  The confirmed session's `started_at_utc` is backdated to the *first*
  window of the confirmed run, not the confirming one, so logged/reported
  start times reflect when the crying actually began.
- **`"reason_updated"` (live update → `send_reason_update`, NOT an alert)**:
  fires mid-session whenever the aggregated cry-reason estimate's
  top-ranked label changes (e.g. the running average flips from "fussy"
  leading to "hungry" leading, as more audio accumulates). This is
  intentionally not an alert — a caregiver who's already been notified
  that the baby is crying shouldn't get re-buzzed just because the reason
  guess refined itself; the app should treat this as silently updating a
  field on the session already on screen. Does not fire on every window —
  only when the top reason actually changes, so a long stable stretch of
  "hungry" leading doesn't spam repeat updates.
- **`"ended"` (alert → `notify_cry_ended`)**: fires once
  `settings.SESSION_MERGE_WINDOW_SECONDS` (default 10 minutes) have passed
  since the *last* confidently-"cry" window. Until that gap elapses, any
  further confident "cry" — even minutes later — is folded into the SAME
  session rather than starting a new one: the session's `started_at_utc`
  stays the original start, and stage-2 aggregation keeps accumulating.
  This is deliberately time-based rather than a small fixed count of
  windows, to cover the case where the baby (still crying) is picked up
  and carried away from the monitor's mic, or just settles and cries again
  a few minutes later — a short grace window reliably miscounted that as
  two separate episodes (a false "ended" immediately followed by a false
  "started"). The session's logged `duration_seconds`/end time reflect the
  *last confident cry*, not whenever the merge window happens to lapse —
  a 5-minute cry followed by 8 minutes of silence logs as a 5-minute
  session, not 13. See Notifications below for why "ended" is alerted
  despite firing well after the fact.
- **"uncertain" windows never discard a `PENDING` candidate or end/extend
  a confirmed session's merge-window gap** (ambiguous isn't evidence the
  baby stopped, but isn't confirmation it's still crying either — matches
  the project's stated preference for false positives over false
  negatives), don't confirm/start a session on their own, and (since
  stage 2 doesn't run on "uncertain" windows) don't contribute to the
  aggregated reason either.

**Cry density**: because a confirmed session can now span a multi-minute
quiet gap (absorbed by the merge window above), `duration_seconds` alone
can no longer distinguish "12 minutes, wall-to-wall crying" from "12
minutes, but only 90s of it was confirmed crying, the rest was an absorbed
gap." `session.confirmed_cry_seconds` (confident-"cry" window count ×
`CAPTURE_INTERVAL_SECONDS` — an approximation, not a precise audio-level
measurement) and `session.cry_density` (that, as a 0.0–1.0 ratio to
`duration_seconds`) capture that distinction. Both are included in the
`reason_updated` and `cry_ended` payloads and in `cry_history.py` rows —
see Notifications and Cry history below.

**Stage-2 (cry-reason) predictions are aggregated across the session** as a
running average of probability vectors (`session.aggregated_stage2_probs`),
sent with the `reason_updated` live update and persisted to
`cry_history.py` when the session finally closes, rather than just
whichever window happened to trigger the start alert. The reasoning: each
window's stage-2 output is a noisy sample, and
averaging many samples from the same episode should reduce that noise —
but this is **unverified** against real multi-window sessions or against
systematic (non-random) error in the model. Some of stage 2's known
weaknesses (e.g. `hungry`'s recall — see the repo root README's
"Why fussy?" section) look like bias from a genuinely weak acoustic
signature, not sampling noise, and averaging doesn't fix bias. Treat the
aggregated reason as a refinement to validate once real session data
exists, not an assumed improvement.

## Notifications

`notify.py` publishes the alerts and live-update from Session tracking above
to the app over MQTT, on the same local Mosquitto broker `buttons_mqtt.py`
already uses for button presses (`settings.MQTT_NOTIFY_TOPIC`, a
different topic from the button one — same broker, distinct message
shape/purpose). LAN-only, no internet, same trust model as the rest of
`pi/` — no auth on the topic, no push-notification service.

**Deliberately not APNs/FCM.** Real push notifications that wake a
backgrounded/closed app and buzz the lock screen require a push service
(Apple's APNs, Google's FCM), which in turn needs a small piece of cloud
state this project otherwise doesn't have — a device-token registry, so
the Pi knows which token(s) to send to — plus, for iOS specifically, an
Apple Developer account ($99/year) just to get push entitlements at all,
independent of App Store distribution. Given this project's local-only
design and that it isn't going through app stores, that tradeoff wasn't
worth it for what's needed here: **the app is expected to stay connected
to the broker and play its own local audible alert** when a
`"cry_started"` message arrives, rather than depending on a system push
notification. Practically, that means:
- **Android**: hold the MQTT connection via a foreground service (the
  same mechanism music/navigation apps use to keep running and make
  sound with the screen off) — no Google account or FCM needed.
- **iOS**: declare a background audio session (the mechanism monitor/
  music-player apps use to stay alive with the screen off) to keep the
  MQTT connection alive; true silent backgrounding without this isn't
  possible on iOS without push.
- Whichever platform, the alert is the app playing a sound locally on
  receiving the message — not a system notification banner triggered
  remotely.

**Message shape** — one topic, `"event"` distinguishes the three kinds
(matching Session tracking's alert/live-update split above):
```json
{"event": "cry_started",    "timestamp": "...", "stage1_confidence": 0.97,
 "stage2_probs": {...}, "context": {"seconds_since_feed": 1820.4, "seconds_since_change": null}}

{"event": "reason_updated", "timestamp": "...", "duration_seconds": 42.1,
 "aggregated_stage2_probs": {...}, "confirmed_cry_seconds": 38.0,
 "cry_density": 0.903, "context": {...}}

{"event": "cry_ended",      "timestamp": "...", "duration_seconds": 96.3,
 "aggregated_stage2_probs": {...}, "confirmed_cry_seconds": 52.0,
 "cry_density": 0.54, "context": {...}}
```
`"context"` (time since last feed/diaper change) is deliberately kept
separate from the model's probabilities in every payload — see the note
under Session tracking / Care events for why (no labeled data yet to
learn a real fusion rule; shown as plain context for the caregiver to
reason about, not blended into the prediction).

`"confirmed_cry_seconds"`/`"cry_density"` (see Session tracking's "Cry
density" above) appear on `reason_updated` and `cry_ended` but not
`cry_started` — at a fresh "started" there's only ever been confirmed
crying so far, so `cry_density` would trivially read `1.0` and add nothing.

**Why "cry ended" is still alerted despite firing late.** `"ended"` fires
up to `settings.SESSION_MERGE_WINDOW_SECONDS` (default 10 min) after the
caregiver likely already knows about and has dealt with the cry — worth
calling out because it might look like it should just be a live update
like `reason_updated`. It's kept as a real alert specifically so the app
has a reliable, unambiguous signal to stop showing an active-session
UI/alert state on its own — without it, an app that isn't independently
timing out that UI (or one whose caregiver put the phone down mid-session
and never saw a later state change) would show "crying now" indefinitely.
Treat it as lower-urgency than `cry_started` in UI/sound choices if that
distinction matters to the app (see `PI_CONTRACT.md` in the app repo).

**Published with `retain=True`.** The broker holds the most recent
message on this topic and delivers it immediately to any client that
(re)connects or subscribes — so an app that briefly drops off WiFi
mid-session and reconnects immediately catches up, instead of staying
silent until the next state change. The tradeoff: a freshly-connecting
app also immediately receives whatever the last message ever was, even
if it's from hours or days ago — this is exactly why every payload
carries `"timestamp"`; the app should compare it against "now" and
ignore/no-op on a stale replayed message rather than alerting on it.

`notify.py` keeps one persistent `paho-mqtt` client (connected via
`connect_async()` + `loop_start()`, not the blocking `connect()`
`buttons_mqtt.py` uses for its own listener loop) rather than
reconnecting per call — `monitor.py` may call these functions many times
per session as the aggregated reason keeps refining, and a broker that's
momentarily unavailable (e.g. still starting up) must not block or crash
inference; paho retries the connection in the background on its own.

## Care events (feed / diaper change)

`notes.txt` and the repo root README's "Why fussy?" section document that
acoustic-only cry-reason classification has a real, fairly hard ceiling on
this dataset (best result: ~46% accuracy / high-80s% top-2 on 3 classes).
Non-acoustic context — how long since the baby last ate, how long since
the last diaper change — is a plausible way to help distinguish reasons
the audio alone can't separate well (e.g. hungry vs. fussy), without
needing better training data. This is being built in two parts: first,
*receiving* that context (this section); later, feeding it into the
stage-2 probability model (not yet implemented).

**Design**: the Pi is the source of truth for `time_since("feed")` /
`time_since("change")`, because `monitor.py` needs these synchronously,
mid-prediction — it can't block on network availability. The wireless
button device is the only input path — an app-side way to log these
events was considered and deliberately dropped: the buttons are already
wireless/movable, so there's no realistic scenario where logging from an
app would be more convenient than a nearby button, and it's not worth
the added complexity (sync logic, conflict handling, an API layer) for
that. `care_events.log_event()` / `get_events()` / `time_since()` remain
the interface; SQLite is used mainly for simple historical queries via
plain SQL rather than hand-rolled filtering.

Two dedicated buttons (feed, change) rather than one button with
short/long-press timing — simpler, and avoids a tired caregiver
mis-timing a press at 3am and logging the wrong event. The buttons live
on a **separate wireless device** (an ESP32, see `firmware/`) rather than
wired directly to the Pi, so they can be placed wherever is most
convenient (e.g. a changing table across the room) instead of needing to
be within cable reach of the Pi. See `HARDWARE.md` for the physical parts
list and network architecture, and `firmware/README.md` for how to flash
the ESP32.

This adds real complexity versus wired buttons — a second physical
device with its own firmware, a local MQTT broker to run on the Pi, and
a WiFi dependency for the button device — chosen deliberately for
placement flexibility. It was a considered tradeoff, not an oversight:
see the "wired vs. wireless" discussion in the project history for the
reasoning (mic placement doesn't need to be wireless — infant crying is
loud enough to pick up from normal furniture placement a few feet from
the crib, and crib-safety guidance rules out mounting anything in/on the
crib anyway — but caregivers pressing feed/change buttons genuinely
benefits from placement flexibility, e.g. a changing table isn't
necessarily near wherever the monitor unit sits).

`buttons_mqtt.py` subscribes to the local broker and logs each event:

```
python buttons_mqtt.py     # blocks, listening for MQTT button-press messages
```

Requires Mosquitto (the MQTT broker) running locally on the Pi — see
Setup below. Run `buttons_mqtt.py` as a second process alongside
`monitor.py` (see the systemd section in Setup) — they're independent
loops that both touch `care_events.sqlite`.

## Cry history

`cry_history.py` persists every completed crying session locally
(`cry_history.sqlite`) — start time, end time, duration, the session's
final aggregated cry-reason estimate (`session.aggregated_stage2_probs` at
the moment it ended, both the top label and the full probability
breakdown), and `confirmed_cry_seconds`/`cry_density` (see Session
tracking's "Cry density" above). `monitor.py` calls `log_session()` right
after handling the `"ended"` event, using `session.started_at_utc` and
`session.last_cry_at_utc` (real timestamps — distinct from
`session.started_at`/`duration_seconds`, which use `time.monotonic()` and
are only meaningful for measuring elapsed time, not as an absolute clock
time). **The logged end time is the LAST confident cry in the episode**,
not whenever `"ended"` actually fired — since a session can now sit open
for up to `settings.SESSION_MERGE_WINDOW_SECONDS` of trailing silence
waiting to see if crying resumes (see Session tracking above), using "now"
as the end time would inflate every session's duration by however long
that trailing silence happened to be.

`confirmed_cry_seconds` was added after the table already existed in the
field, so `_migrate_add_confirmed_cry_seconds()` runs an `ALTER TABLE` on
every `_connect()` (a no-op once already applied, checked against the live
schema first) rather than assuming a fresh `CREATE TABLE IF NOT EXISTS` —
existing `cry_history.sqlite` files on deployed Pis need to pick up the
new column without losing their history.

Scoped deliberately narrow, per-request:
- **Only completed cry sessions are stored** — not individual
  laugh/silence/noise window classifications, which aren't useful
  history to review later and would add a lot of low-value volume.
- **No raw audio is stored.** Audio is processed in-memory per window
  and discarded. Storing clips (e.g. to later build a real labeled
  retraining dataset) was considered and explicitly deferred — it would
  put a real labeling burden on whoever's using the device, which isn't
  worth it without a concrete plan for who'd do that labeling.

`get_sessions(since=None, limit=None)` returns completed sessions most
recent first, as plain dicts — this is what `sync_api.py`'s `GET
/cry_history` calls to answer the app. Read-only — the app never
originates cry-session data (unlike care_events, see App sync below).

## Device events

`device_events.py` logs when `monitor.py` starts and stops — NOT power
outages or crashes as such, just the ordinary "the device was
intentionally turned off/on" case (the shutdown button, see
`HARDWARE.md`'s "Off switch"). The purpose is narrow and specific:
without this, a long stretch with no entries in `cry_history.py` is
ambiguous during historical review — was the baby just not crying for
an unusually long time, or was the monitor off? A startup/shutdown pair
bracketing that gap answers the question, without asking a caregiver to
manually log anything (a deliberate choice — the burden was judged not
worth it given cries already have "extended" enough data as-is).

Hooked directly into `monitor.py`'s own lifecycle rather than a separate
service:
- **Startup**: logged at the top of `run()`, so it fires every time the
  systemd-managed service starts (on boot, or after a crash-triggered
  restart via `Restart=always`).
- **Shutdown**: logged on `SIGTERM` (systemd sends this on both a normal
  `systemctl stop` and during a full system shutdown) and on
  `KeyboardInterrupt` (Ctrl+C during manual/local testing), each tagged
  with a `reason` (`"sigterm"` / `"keyboard_interrupt"`) so it's clear
  which triggered it.

**On purpose, a crash-and-restart also produces a shutdown+startup pair**
close together, same as a deliberate power cycle — this isn't filtered
out, because "the monitor briefly went down from a crash" is itself
useful information when reviewing history, not noise to hide.

One known imprecision: `SIGTERM` is only checked once per captured
window (`iter_windows()` blocks for up to
`settings.CAPTURE_INTERVAL_SECONDS` while capturing), so the logged
shutdown timestamp can lag the actual signal by a few seconds. Not a
correctness issue — systemd's grace period before force-killing a
service is far longer than that — just means "exactly when" isn't
precise to the second.

## Pausing predictions

`prediction_state.py` is the escape hatch for when the classifier is
misbehaving in production — e.g. hallucinating cries out of ambient
noise, or fragmenting one real crying episode into many separate
sessions. Rather than mom/dad silencing notifications on the phone (which
would leave `predictor.predict()` running, still writing bad sessions
into `cry_history.sqlite` and still driving `session.py`'s state
machine), the app can pause inference on the Pi itself via
`POST /prediction_state`. `monitor.py` checks this once per captured
window:

- **Capture keeps running while paused** — only the predict/session/
  notify/history steps are skipped. Resuming is instant (no model reload,
  no re-initializing the microphone).
- **The wireless buttons are entirely unaffected.** `buttons_mqtt.py`
  writes to `care_events.sqlite` independently of `monitor.py` and
  doesn't touch `prediction_state.py` at all — feed/change logging keeps
  working normally while paused.
- **A CONFIRMED in-progress crying session is force-ended (and alerted,
  and logged) when pause takes effect**, rather than left open
  indefinitely (nothing would ever call `session.update()` again to close
  it otherwise) — same `notify_cry_ended` + `cry_history.py` logging as a
  normal merge-window "ended", just tagged internally as ended due to
  pause. **An unconfirmed (`PENDING`) candidate is silently discarded
  instead** — it was never alerted or logged in the first place (see
  Session tracking above), so there's nothing to close out.
- Same polling-latency caveat as the `SIGTERM` handling: pause/resume
  takes effect on the next captured window, up to
  `settings.CAPTURE_INTERVAL_SECONDS` after the app's request.

```
GET  /prediction_state    -> {"paused": true|false}
POST /prediction_state    {"paused": true|false} -> {"paused": ..., "changed_at": ...}
```

## App sync

`sync_api.py` runs a small local HTTP API so a caregiver's phone can:
- **Review/correct history** — `GET /care_events`, `GET /cry_history`,
  `DELETE /care_events/<id>` for the "simple error fixing" case (an
  accidental double-press, a wrong entry).
- **Log feed/change events while away from the house**, synced back once
  the app is on the same WiFi as the Pi — `POST /care_events`.

This is deliberately **not** reachable from outside the home network — no
Tailscale, no cloud relay, same LAN-only/no-auth trust model as the MQTT
broker (see Setup below for what that implies and why it was judged
acceptable here). The app is expected to keep its own local store and
sync opportunistically whenever it detects it's on the same network as
the Pi, not push in real time from anywhere:

```
GET  /care_events?since=<ISO8601>   pull events since a given time
POST /care_events                   push one event: {"event_type": "feed"|"change",
                                     "device_id": "...", "note": "...", "timestamp": "..."}
                                     (device_id/note/timestamp optional)
DELETE /care_events/<id>            remove one event
GET  /cry_history?since=<ISO8601>   pull cry sessions since a given time (read-only)
GET  /device_events?since=<ISO8601> pull startup/shutdown events since a given
                                     time (read-only) -- for explaining gaps in
                                     cry_history, see Device events above
GET  /prediction_state              current paused/active state
POST /prediction_state              {"paused": true|false} -- pause/resume
                                     inference, see Pausing predictions above
POST /reset                         {"confirm": "RESET"} -- wipes all stored
                                     history, see Database reset below
```

**Multiple family members** (mom, dad, grandparents) are handled as
multiple independent app instances, each with their own `device_id` (a
per-install identifier the app generates and keeps) sent with anything
they log — purely informational (`care_events.device_id`, "who logged
this"), not an auth mechanism. This only works well if everyone's phone
regularly ends up on the home WiFi (true for parents living there, and
per this project's own scoping, true enough for grandparents who visit
in person regularly) — someone who mostly wants remote-only access
without visiting wouldn't be well served by this design, which was a
deliberate simplicity tradeoff, not an oversight (see the conversation
history around this decision for the reasoning against a Tailscale- or
cloud-based always-reachable alternative).

**Duplicate handling**: `POST /care_events` runs incoming events through
`care_events.is_likely_duplicate()` — if an event of the same type
already exists within `DEDUP_WINDOW_MINUTES` (default 5) of the given
timestamp, it's treated as a likely duplicate (e.g. a feed logged both
via the app while out and the physical button once home) and skipped
rather than double-counted. Returns `200 {"status": "skipped_duplicate"}`,
not an error — the app shouldn't need special-case handling for it. This
trades a small false-negative risk (a genuine second event that close
together gets silently dropped) for a cleaner history; a mistakenly
dropped or wrongly kept event is still fixable via `DELETE`.

**iOS vs. Android**: no impact on any Pi-side code. `sync_api.py` is a
plain HTTP/JSON API — either platform's app talks to it the same way.
The one place OS choice would matter is push notifications (APNs vs.
FCM are different services), which is an app/notification-service
concern entirely outside `pi/` — see `notify.py`, still a stub, for
where that would eventually plug in.

## Database reset

`reset_db.py` wipes all stored baby-specific history — `care_events`
(feed/change log), `cry_history` (completed cry sessions), and
`device_events` (startup/shutdown log) — for reusing the device with a
different baby, regifting it, or clearing out data accumulated during
testing. Triggered via `POST /reset` on `sync_api.py`:

```
POST /reset   {"confirm": "RESET"} -> {"status": "reset", "deleted": {"care_events": N, "cry_history": N, "device_events": N}}
```

**The app owns all confirmation/warning UX for this — deliberately.**
`reset_db.py`/`sync_api.py` do the deletion unconditionally the moment a
valid request arrives; there's no "are you sure?" step, undo, or backup
on the Pi side. The `{"confirm": "RESET"}` body is a guard against a
stray or malformed request (an empty POST hitting the wrong endpoint by
accident), not a real authorization check — this API has none (see
above) — so it shouldn't be read as a substitute for the app making the
caregiver actually confirm before calling this.

**`prediction_state.py`'s pause flag is deliberately NOT reset.** It's
current operational state (is inference on or off right now), not
accumulated history — resetting it as a side effect of "clear the
baby's data" would be a surprising, unrelated change to make at the
same time.

This is irreversible — there's no backup/undo built in (consistent with
the rest of this project's local-only storage; see Current limitations).
If that ever needs to change (e.g. an "export before reset" step), it
belongs in the app, which already has the read endpoints
(`GET /care_events`, `/cry_history`, `/device_events`) needed to pull
everything first.

## Setup on the Pi

Full step-by-step install instructions (writing the SD card, getting
models onto the Pi, installing dependencies, setting up systemd services
and Tailscale) live in [docs/INSTALL.md](../docs/INSTALL.md) — this
section is intentionally not duplicated here. The short version, for
reference: models are trained/exported on a dev machine
(`export_tflite.py`, `export_yamnet_embedding_tflite.py`,
`pi/deploy_models.py`) and copied to the Pi alongside this `pi/`
directory; `pip install -r requirements.txt`; Mosquitto installed for
the wireless buttons; `monitor.py` and `buttons_mqtt.py` run as systemd
services for always-on operation. See `docs/INSTALL.md` for the actual
commands.

## Current limitations

- **Non-overlapping windows only.** `capture.py` records a full
  `DURATION_SECONDS` (4s) window, blocking, before the next one starts.
  A cry that starts partway through a window still gets captured (the
  model sees whatever's in the 4s clip), but detection latency is bounded
  below by the window length. Overlapping/streaming capture would need a
  `sounddevice.InputStream` callback with a rolling buffer instead of the
  current blocking `sd.rec()` — not implemented yet.
- **Notification transport (MQTT) has no built-in guarantee the app is
  actually listening.** `notify.py` publishes with `retain=True` so a
  reconnecting app catches up on the last message, but if the app is
  never running/connected at all (uninstalled, force-quit, phone off),
  there's no fallback path (no push service) — see Notifications above
  for why that tradeoff was made. Worth revisiting if it turns out family
  members don't reliably keep the app running.
- **Individual window-level predictions (laugh/silence/noise, or each
  raw stage-1/stage-2 result within a cry session) still aren't stored**
  — only completed cry sessions are, on purpose (see Cry history above).
  If per-window detail ever turns out to matter, that's a deliberate
  scope decision to revisit, not an oversight.
- **Feed/change timing is surfaced as context, not blended into the
  model.** Every notify payload now includes a `context` block
  (`seconds_since_feed` / `seconds_since_change`, see `notify._context()`)
  alongside the model's prediction — but this is deliberately NOT used to
  adjust `cry_reason_mlp`'s output probabilities. There's no labeled data
  yet showing how feed/change timing should actually shift a cry-reason
  estimate, so rather than guess at a fusion rule (a real risk given this
  project's history of plausible-sounding changes that didn't help — see
  `experiments/README.md`), the app is expected to show this as plain
  context ("3.2h since feed") next to the prediction and let the
  caregiver reason about both together. Revisit once real paired data
  exists to learn an actual fusion.
- **The ESP32 button device's offline queue is RAM-only.** If WiFi/the
  MQTT broker is unreachable when a button is pressed, the press is held
  in a small in-memory queue and retried — both on the next press and
  proactively every `RETRY_INTERVAL_MS` (`config.h`, default 10s) — until
  it's delivered. Each queued event carries how long ago it was actually
  pressed (`seconds_ago`, from `millis()` — no wall-clock/NTP sync
  needed), so `buttons_mqtt.py` logs the real press time even for a
  delayed delivery, not when the message happened to arrive. The one
  remaining gap: the queue doesn't survive the ESP32 itself
  resetting/losing power while events are queued (RAM is cleared on
  reset) — a narrower edge case (an outage AND a device reset, both
  overlapping an unflushed press) than the plain "no queue at all" gap
  this replaces. Not persisted to flash, to avoid flash-wear complexity
  for that narrower case; revisit if it turns out to matter in practice.
