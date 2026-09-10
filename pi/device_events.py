"""Local store for monitor power-on/shutdown events -- NOT outages
(unexpected power loss / crashes aren't distinguishable this way, see
below), just the ordinary "the device was intentionally turned off/on"
cases (the shutdown button, see HARDWARE.md's "Off switch"). Purpose:
so historical analysis of cry_history.py doesn't mistake "the monitor
was off" for "the baby didn't cry for an unusually long time" -- a gap
with no shutdown/startup record in between reads as a real gap; a gap
that spans a shutdown->startup pair is explained.

Deliberately its own store, not folded into care_events.py (caregiver
actions) or cry_history.py (cry sessions) -- this is monitor/device
state, a different kind of record from either.

Hooked into monitor.py's own lifecycle (see monitor.py): a "startup"
event is logged when monitor.py starts running, and a "shutdown" event
when it receives SIGTERM (systemd sends this both on a normal service
stop and during a full system shutdown -- see REMOTE_ACCESS.md). This
means a monitor.py CRASH-and-restart (systemd's Restart=always) also
produces a shutdown+startup pair close together -- that's intentional,
not a bug: it's genuinely useful to know the monitor briefly went down,
not just when it was deliberately power-cycled. reason distinguishes the
two cases where it's known (see log_event()).
"""
import datetime
import os
import sqlite3
import uuid

DB_PATH = os.path.join(os.path.dirname(__file__), "device_events.sqlite")

STARTUP = "startup"
SHUTDOWN = "shutdown"
EVENT_TYPES = {STARTUP, SHUTDOWN}


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS device_events (
            id TEXT PRIMARY KEY,
            event_type TEXT NOT NULL,
            timestamp TEXT NOT NULL,   -- ISO 8601 UTC
            reason TEXT                -- e.g. "sigterm", "startup" -- free text,
                                       -- see log_event(); NULL if unknown
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_device_events_ts ON device_events(timestamp)")
    return conn


def log_event(event_type, reason=None, timestamp=None):
    """Record a device startup/shutdown event. Returns the event's id.

    event_type: STARTUP or SHUTDOWN.
    reason: free-text context if known -- e.g. "sigterm" (the normal
        shutdown-button/systemd-stop path). Left as free text rather than
        a fixed vocabulary since there's only ever going to be a couple
        of callers (monitor.py); not meant for programmatic branching.
    timestamp: defaults to now (UTC).
    """
    if event_type not in EVENT_TYPES:
        raise ValueError(f"Unknown event_type {event_type!r}, expected one of {EVENT_TYPES}")

    ts = timestamp or datetime.datetime.now(datetime.timezone.utc)
    event_id = str(uuid.uuid4())

    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO device_events (id, event_type, timestamp, reason) VALUES (?, ?, ?, ?)",
            (event_id, event_type, ts.isoformat(), reason),
        )
        conn.commit()
    finally:
        conn.close()
    return event_id


def get_events(since=None, limit=None):
    """Returns device events, most recent first, as dicts with keys: id,
    event_type, timestamp (datetime), reason.

    since: optional datetime, only events at or after this time."""
    query = "SELECT id, event_type, timestamp, reason FROM device_events WHERE 1=1"
    params = []
    if since is not None:
        query += " AND timestamp >= ?"
        params.append(since.isoformat())
    query += " ORDER BY timestamp DESC"
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)

    conn = _connect()
    try:
        rows = conn.execute(query, params).fetchall()
    finally:
        conn.close()

    return [
        {
            "id": id_,
            "event_type": event_type,
            "timestamp": datetime.datetime.fromisoformat(ts),
            "reason": reason,
        }
        for id_, event_type, ts, reason in rows
    ]
