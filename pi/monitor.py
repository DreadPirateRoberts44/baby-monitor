"""Main Pi monitor loop: capture audio -> run two-stage inference -> track
crying sessions -> alert once at session start and once at session end,
plus a non-alerting live update whenever the aggregated cry-reason
estimate changes mid-session. A session only starts/alerts after
settings.SESSION_START_MIN_WINDOWS consecutive confident "cry" windows
(see session.py) -- filters out a single misclassified window (a bark, a
static burst) before it ever reaches the app. Every completed session is
persisted to cry_history.py (start time + the LAST confident cry in the
episode as end time + aggregated reason + how much of it was actually
confirmed crying vs. absorbed quiet gaps) once it actually closes
(settings.SESSION_MERGE_WINDOW_SECONDS after the last confident cry -- see
session.py) -- only cry sessions, not every laugh/silence/noise window,
which aren't useful history to review later. Entry point for the
always-on monitor.

Also logs a device_events.py startup event when this starts running, and
a shutdown event on SIGTERM (sent by systemd on both a normal service
stop and a full system shutdown -- see REMOTE_ACCESS.md) -- purely so
gaps in cry_history.py during historical review are explainable ("the
monitor was off then", not "the baby mysteriously didn't cry"). Not
meant to catch unexpected power loss/crashes as such -- see
device_events.py's docstring.

Checks prediction_state.py once per captured window and skips inference
entirely while paused -- the app-controlled escape hatch for when the
model is misbehaving (hallucinated cries / a cry fragmented into many
sessions). Capture keeps running while paused so resuming is instant.
See prediction_state.py's docstring for why this is a separate concept
from silencing notifications in the app.

Usage:
    python monitor.py
"""
import signal
import sys
import time

import settings
from capture import iter_windows
from cry_history import log_session
from device_events import log_event as log_device_event, STARTUP, SHUTDOWN
from inference import CryPredictor
from notify import notify_cry_started, notify_cry_ended, send_reason_update
from prediction_state import is_paused
from session import CrySession, CRYING, PENDING

_shutdown_requested = False


def _handle_sigterm(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True


def _close_session(session, reason):
    """Logs and notifies an "ended" session outside the normal update()
    flow -- used when pausing predictions mid-CRYING-session, so a
    session never hangs open forever just because nothing is calling
    session.update() anymore (predictions being paused means update()
    stops being called at all, which would otherwise leave
    state == CRYING indefinitely)."""
    print(f"  >> cry session force-ended ({reason}, {session.duration_seconds:.0f}s, "
          f"aggregated reason: {session.aggregated_stage2_probs})")
    notify_cry_ended(session)
    log_session(
        started_at=session.started_at_utc,
        ended_at=session.last_cry_at_utc,
        duration_seconds=session.duration_seconds,
        reason_probs=session.aggregated_stage2_probs,
        confirmed_cry_seconds=session.confirmed_cry_seconds,
    )
    session.clear()


def run():
    signal.signal(signal.SIGTERM, _handle_sigterm)
    log_device_event(STARTUP)

    print("Loading models...")
    predictor = CryPredictor()
    session = CrySession()
    print(f"Ready. Capturing {settings.DURATION_SECONDS}s windows every "
          f"{settings.CAPTURE_INTERVAL_SECONDS}s.")

    was_paused = False
    for waveform in iter_windows():
        # Checked once per captured window -- iter_windows() itself blocks
        # for up to settings.CAPTURE_INTERVAL_SECONDS while capturing, so
        # a SIGTERM mid-capture isn't noticed until the current window
        # finishes (up to ~4s delay by default). Not a correctness issue
        # -- systemd gives services a grace period before force-killing --
        # just means the logged shutdown timestamp can lag the actual
        # SIGTERM by a few seconds.
        if _shutdown_requested:
            print("SIGTERM received, shutting down.")
            log_device_event(SHUTDOWN, reason="sigterm")
            return

        if is_paused():
            if session.state == CRYING:
                _close_session(session, reason="predictions paused")
            elif session.state == PENDING:
                # Never confirmed -- nothing was alerted or logged for it,
                # so just discard the candidate rather than force-ending
                # a "session" that was never actually started.
                session.clear()
            if not was_paused:
                print("Predictions paused (app request) -- capture continues, inference skipped.")
                was_paused = True
            continue
        if was_paused:
            print("Predictions resumed.")
            was_paused = False

        t0 = time.monotonic()
        result = predictor.predict(waveform)
        elapsed = time.monotonic() - t0

        summary = f"{result['stage1_label']} (conf {result['stage1_confidence']:.2f}, {elapsed * 1000:.0f}ms)"
        if result["stage2_probs"]:
            top_reason = next(iter(result["stage2_probs"]))
            summary += f" -> reason: {top_reason} ({result['stage2_probs'][top_reason]:.2f})"
        print(summary)

        event = session.update(result)
        if event == "started":
            print(f"  >> cry session started (confirmed after "
                  f"{settings.SESSION_START_MIN_WINDOWS} consecutive cry window(s))")
            notify_cry_started(result)
        elif event == "reason_updated":
            print(f"  >> reason estimate updated: {session.aggregated_stage2_probs}")
            send_reason_update(session)
        elif event == "ended":
            _close_session(session, reason="no more crying detected")


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\nStopped.")
        log_device_event(SHUTDOWN, reason="keyboard_interrupt")
        sys.exit(0)
