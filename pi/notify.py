"""Publishes cry alerts/live-updates to the app over MQTT, on the same
local broker buttons_mqtt.py already uses (Mosquitto, running on the Pi
-- see settings.MQTT_NOTIFY_TOPIC). LAN-only, same trust model as
everything else in pi/ -- no internet, no push-notification service
(APNs/FCM), no cloud account. The app is expected to stay connected to
the broker (foregrounded, or kept alive via a platform-appropriate
mechanism -- e.g. a foreground service on Android, background audio mode
on iOS) and play a local audible alert itself when a "cry_started"
message arrives, rather than relying on a system push notification to
wake it. This was a deliberate choice: real APNs/FCM push would need a
small piece of cloud state (a device-token registry) that doesn't fit
this project's local-only design, plus an Apple Developer account
($99/year) for iOS push specifically -- not worth it for an audible
alert that a left-open/backgrounded app can already provide via MQTT.

Published with retain=True: the broker holds the most recent message on
this topic and delivers it immediately to any client that (re)connects
or subscribes, even if it missed the live moment (e.g. the app's WiFi
blipped during an active cry session). This means a freshly-connecting
app also immediately receives whatever the last message ever was, even
if it's old (e.g. from hours/days ago) -- every payload carries a
"timestamp" field specifically so the app can tell a fresh event from a
stale replayed one and decide whether to actually alert.

Three call sites, driven by session.CrySession (see monitor.py) -- split
into two kinds on purpose:

  ALERTS (should interrupt/notify the caregiver -- e.g. push notification,
  phone buzz):
    - notify_cry_started: fired once when a crying session is CONFIRMED
      (settings.SESSION_START_MIN_WINDOWS consecutive confident "cry"
      windows -- see session.py) -- not necessarily the very first window
      that looked like a cry.
    - notify_cry_ended: fired once a confirmed session ends (no confident
      crying for settings.SESSION_MERGE_WINDOW_SECONDS -- see session.py).
      This exists specifically so the app has a reliable "the episode is
      over" signal to stop showing an active-session UI/alert state on its
      own timeline -- without it, a caregiver who put the phone down mid-
      session has nothing telling the app to stand down, and an app that
      isn't independently timing out that UI state would show "crying now"
      indefinitely. Fires up to SESSION_MERGE_WINDOW_SECONDS after the
      caregiver likely already handled the cry, so it's intentionally a
      quieter/lower-urgency alert than cry_started -- see PI_CONTRACT.md
      in the app repo for how the app is expected to treat it.

  LIVE UPDATE (should update an already-visible session in the app WITHOUT
  a new alert -- e.g. a websocket push / silent data update to a session
  card already on screen):
    - send_reason_update: fired each time the aggregated cry-reason
      estimate's top-ranked label changes during an ongoing session (see
      session.py's "reason_updated" event). This is deliberately NOT an
      alert -- a session that's already been surfaced to the caregiver
      shouldn't re-notify just because the reason estimate refined itself
      (e.g. "fussy" -> "hungry" as more audio accumulates).

No per-window calls of any kind while a session's reason estimate is
unchanged -- see session.py's docstring for why.

reason_updated and cry_ended both also carry "confirmed_cry_seconds" and
"cry_density" (session.CrySession.confirmed_cry_seconds/.cry_density) --
now that a session can span a long quiet gap absorbed by the merge window,
duration_seconds alone can't tell a dense, wall-to-wall cry from a sparse
one with a long silent stretch in the middle. cry_started omits these --
at a fresh "started" there's only ever been confirmed crying so far, so
cry_density would trivially read 1.0 and add nothing.

Every payload also carries a "context" block (time since last feed/diaper
change, from care_events.py). This is DELIBERATELY separate from
stage2_probs/aggregated_stage2_probs, not blended into them -- there's no
labeled data yet to learn how feed/change timing should actually shift a
cry-reason probability (see the project's README/notes.txt discussion),
so rather than guess at a fusion rule, the app is expected to show this as
plain context next to the model's prediction ("3.2h since feed") and let
the caregiver do that reasoning themselves. If/when enough paired data
exists to learn a real fusion, that would replace this, not extend it.

Each build_*_payload()/message also carries "event" (see below) so a
single MQTT topic can carry all three message kinds and the app can branch
on it -- "cry_started"/"cry_ended" are alerts, "reason_updated" is the
live update (see module docstring above and session.py).
"""
import datetime
import json

import paho.mqtt.client as mqtt

import settings
from care_events import time_since, FEED, CHANGE

_client = None


def _get_client():
    """Lazily creates a single persistent MQTT client, reused across calls
    -- monitor.py may call these functions many times per session (e.g.
    once per window while stage-2 reason keeps refining), so reconnecting
    per-message would be wasteful and slower than necessary.

    Uses connect_async() + loop_start() rather than the blocking connect()
    buttons_mqtt.py uses for its own always-connected listener loop: this
    module is called from inside monitor.py's main capture loop, so a
    broker that's briefly unavailable (e.g. Mosquitto still starting up)
    must not block or crash inference -- paho retries the connection and
    any future publish() calls in the background on their own."""
    global _client
    if _client is None:
        _client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        _client.connect_async(settings.MQTT_BROKER_HOST, settings.MQTT_BROKER_PORT)
        _client.loop_start()
    return _client


def _publish(payload):
    client = _get_client()
    client.publish(settings.MQTT_NOTIFY_TOPIC, json.dumps(payload), qos=1, retain=True)


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _context():
    """Non-model context to show alongside a prediction, not to blend
    into it -- see module docstring. Values are None if that event type
    has never been logged (distinct from "just happened")."""
    feed_secs = time_since(FEED)
    change_secs = time_since(CHANGE)
    return {
        "seconds_since_feed": round(feed_secs, 1) if feed_secs is not None else None,
        "seconds_since_change": round(change_secs, 1) if change_secs is not None else None,
    }


def build_started_payload(result):
    """result: the dict returned by inference.CryPredictor.predict(), from
    the window that triggered session start."""
    return {
        "event": "cry_started",
        "timestamp": _now(),
        "stage1_confidence": result["stage1_confidence"],
        "stage2_probs": result["stage2_probs"],
        "context": _context(),
    }


def build_reason_update_payload(session):
    """session: the session.CrySession instance, read right after
    update() returned "reason_updated"."""
    return {
        "event": "reason_updated",
        "timestamp": _now(),
        "duration_seconds": round(session.duration_seconds, 1),
        "aggregated_stage2_probs": session.aggregated_stage2_probs,
        "confirmed_cry_seconds": round(session.confirmed_cry_seconds, 1),
        "cry_density": round(session.cry_density, 3),
        "context": _context(),
    }


def build_ended_payload(session):
    """session: the session.CrySession instance, read right after
    update() returned "ended" (before session.clear())."""
    return {
        "event": "cry_ended",
        "timestamp": _now(),
        "duration_seconds": round(session.duration_seconds, 1),
        "aggregated_stage2_probs": session.aggregated_stage2_probs,
        "confirmed_cry_seconds": round(session.confirmed_cry_seconds, 1),
        "cry_density": round(session.cry_density, 3),
        "context": _context(),
    }


def notify_cry_started(result):
    """Alert: should interrupt/notify the caregiver."""
    payload = build_started_payload(result)
    _publish(payload)
    print(f"[notify:ALERT] published: {json.dumps(payload)}")


def notify_cry_ended(session):
    """Alert: should tell the app the episode is over -- see module
    docstring for why this exists despite firing well after the fact."""
    payload = build_ended_payload(session)
    _publish(payload)
    print(f"[notify:ALERT] published: {json.dumps(payload)}")


def send_reason_update(session):
    """Live update: should refresh the already-visible session in the app
    WITHOUT a new alert/notification -- see module docstring."""
    payload = build_reason_update_payload(session)
    _publish(payload)
    print(f"[notify:live-update] published: {json.dumps(payload)}")
