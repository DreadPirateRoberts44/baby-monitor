"""Runtime on/off switch for stage1/stage2 predictions, controllable from
the app via sync_api.py -- the escape hatch for when the model is
misbehaving in production (e.g. hallucinating cries, or fragmenting one
cry into many separate sessions) and mom/dad want it to stop without my
intervention and without losing the buttons (care_events keeps working
independently either way -- see buttons_mqtt.py, which never touches this
module).

Deliberately NOT the same thing as silencing notifications on the phone:
that would leave predictor.predict() running, still writing (bad) entries
to cry_history.py and still driving session.py's state machine, polluting
exactly the data you'd want to trust later. Pausing here stops inference
itself; capture keeps running (see monitor.py) so resuming is instant --
no model reload, no re-init of the audio device.

A single-row SQLite table rather than a plain flag file, purely for
consistency with the rest of pi/ (care_events.py, cry_history.py,
device_events.py all use sqlite3) -- there's no concurrency need here
beyond what sqlite already gives for free.
"""
import datetime
import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(__file__), "prediction_state.sqlite")

_ROW_ID = 1  # single-row table -- there is only ever one current state


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_state (
            id INTEGER PRIMARY KEY,
            paused INTEGER NOT NULL,
            changed_at TEXT NOT NULL   -- ISO 8601 UTC, when it was last toggled
        )
        """
    )
    return conn


def is_paused():
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT paused FROM prediction_state WHERE id = ?", (_ROW_ID,)
        ).fetchone()
    finally:
        conn.close()
    return bool(row[0]) if row else False


def set_paused(paused):
    """Sets the paused flag. Returns the UTC datetime it was recorded at.

    monitor.py polls is_paused() once per captured window (see run()) --
    this doesn't push to the monitor process, so there's up to
    CAPTURE_INTERVAL_SECONDS latency before a change takes effect, same
    caveat as the existing SIGTERM handling."""
    changed_at = datetime.datetime.now(datetime.timezone.utc)
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO prediction_state (id, paused, changed_at) VALUES (?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET paused = excluded.paused, changed_at = excluded.changed_at
            """,
            (_ROW_ID, int(bool(paused)), changed_at.isoformat()),
        )
        conn.commit()
    finally:
        conn.close()
    return changed_at
