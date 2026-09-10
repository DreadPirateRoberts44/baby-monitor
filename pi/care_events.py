"""Local store for caregiver-logged events ("fed", "changed") used to
compute contextual features like time-since-last-feed for the cry-reason
model (see notes.txt / README for the motivation -- acoustic-only
cry-reason classification has a hard ceiling on this dataset; non-acoustic
context like feeding/diaper timing is a plausible way to help distinguish
reasons the audio alone can't separate well, e.g. hungry vs. fussy).

Design: the Pi is the source of truth. monitor.py needs "time since last
feed" synchronously, mid-prediction -- it can't block on network
availability, so the Pi always has its own copy it can read instantly.
The wireless button device (buttons_mqtt.py) is the primary writer for
normal use (logging from an app when a button is right there and
movable isn't more convenient -- see the button-vs-app discussion this
project settled on). The app IS a legitimate second writer for a
different case: entering feed/change events while away from the house
entirely (e.g. at a store), synced back once the app is on the same
WiFi as the Pi -- see sync_api.py, which is the only thing that should
call log_event(source="app", ...) / delete_event().

Backed by SQLite (stdlib, no new dependency) rather than a flat JSON file
-- simple historical queries (get_events(since=...)) via plain SQL rather
than hand-rolled filtering, and safe concurrent writes now that there are
two real writers (buttons_mqtt.py and sync_api.py).
"""
import datetime
import os
import sqlite3
import uuid

import settings

DB_PATH = os.path.join(os.path.dirname(__file__), "care_events.sqlite")

# Event types this module understands. Kept as a fixed set (not free text)
# so time_since() and any future per-type logic has a known vocabulary.
FEED = "feed"
CHANGE = "change"
EVENT_TYPES = {FEED, CHANGE}


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS care_events (
            id TEXT PRIMARY KEY,
            event_type TEXT NOT NULL,
            timestamp TEXT NOT NULL,   -- ISO 8601 UTC
            source TEXT NOT NULL,      -- "button" or "app"
            device_id TEXT,            -- which app instance logged it, e.g. a
                                       -- per-install UUID the app generates and
                                       -- keeps -- NULL for button-sourced events
                                       -- (there's only one button device).
                                       -- Purely informational (debugging/history,
                                       -- "who logged this"), not used in
                                       -- time_since() or dedup logic.
            note TEXT
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_care_events_type_ts ON care_events(event_type, timestamp)")
    return conn


DEDUP_WINDOW_MINUTES = 5


def is_likely_duplicate(event_type, timestamp, window_minutes=DEDUP_WINDOW_MINUTES):
    """True if an event of this type already exists within window_minutes
    of the given timestamp (before or after). Used by sync_api.py when
    ingesting events pushed from the app, to avoid double-counting e.g. a
    feed logged both via the app (while away) and the physical button
    (once home) for the same real-world event.

    Trades a small false-negative risk (a genuine second event of the
    same type within the window gets silently skipped) for a cleaner
    history -- deliberate, not accidental; see sync_api.py. NOT applied
    to log_event() itself / the button path, which should always record
    exactly what it's told.
    """
    if event_type not in EVENT_TYPES:
        raise ValueError(f"Unknown event_type {event_type!r}, expected one of {EVENT_TYPES}")

    window = datetime.timedelta(minutes=window_minutes)
    lower = (timestamp - window).isoformat()
    upper = (timestamp + window).isoformat()

    conn = _connect()
    try:
        row = conn.execute(
            "SELECT 1 FROM care_events WHERE event_type = ? AND timestamp >= ? AND timestamp <= ? LIMIT 1",
            (event_type, lower, upper),
        ).fetchone()
    finally:
        conn.close()
    return row is not None


def log_event(event_type, source="button", device_id=None, note=None, timestamp=None):
    """Record a caregiver event. Returns the event's id.

    event_type: one of FEED, CHANGE (care_events.FEED / care_events.CHANGE).
    source: "button" (the wireless button device) or "app" (see
        sync_api.py -- entries synced from a caregiver's phone). Purely
        informational (for debugging/history), not used in time_since()
        calculations.
    device_id: which app instance logged it (a per-install identifier the
        app generates), when source="app". None for button-sourced events.
    timestamp: defaults to now (UTC). Pass explicitly when backfilling an
        event whose actual occurrence predates when it's being logged --
        e.g. a button press delivered late by the ESP32's offline queue
        (see buttons_mqtt.py), or an app entry logged while offline and
        synced later (see sync_api.py).
    """
    if event_type not in EVENT_TYPES:
        raise ValueError(f"Unknown event_type {event_type!r}, expected one of {EVENT_TYPES}")

    ts = timestamp or datetime.datetime.now(datetime.timezone.utc)
    event_id = str(uuid.uuid4())

    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO care_events (id, event_type, timestamp, source, device_id, note) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (event_id, event_type, ts.isoformat(), source, device_id, note),
        )
        conn.commit()
    finally:
        conn.close()
    return event_id


def delete_event(event_id):
    """Remove one event by id (for the "simple error fixing" case -- an
    accidental double-press, a wrong entry, etc.). Returns True if a row
    was actually deleted, False if no event had that id."""
    conn = _connect()
    try:
        cursor = conn.execute("DELETE FROM care_events WHERE id = ?", (event_id,))
        conn.commit()
    finally:
        conn.close()
    return cursor.rowcount > 0


def last_event(event_type):
    """Returns (timestamp: datetime, source, note) for the most recent
    event of this type, or None if none logged yet."""
    if event_type not in EVENT_TYPES:
        raise ValueError(f"Unknown event_type {event_type!r}, expected one of {EVENT_TYPES}")

    conn = _connect()
    try:
        row = conn.execute(
            "SELECT timestamp, source, note FROM care_events "
            "WHERE event_type = ? ORDER BY timestamp DESC LIMIT 1",
            (event_type,),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return None
    timestamp_str, source, note = row
    return datetime.datetime.fromisoformat(timestamp_str), source, note


def time_since(event_type, now=None):
    """Seconds since the most recent event of this type, or None if no
    such event has ever been logged (distinct from 0 -- "just happened" --
    so callers/models can treat "unknown" differently from "very recent")."""
    result = last_event(event_type)
    if result is None:
        return None
    last_ts, _, _ = result
    now = now or datetime.datetime.now(datetime.timezone.utc)
    return max(0.0, (now - last_ts).total_seconds())


def get_events(event_type=None, since=None, limit=None):
    """Returns a list of completed events, most recent first, as dicts
    with keys: id, event_type, timestamp (datetime), source, device_id,
    note. event_type/since/limit are optional filters -- since: datetime,
    only events at or after this time. This is what sync_api.py's GET
    /care_events calls to answer "what's changed since the app last
    synced"."""
    query = "SELECT id, event_type, timestamp, source, device_id, note FROM care_events WHERE 1=1"
    params = []
    if event_type is not None:
        if event_type not in EVENT_TYPES:
            raise ValueError(f"Unknown event_type {event_type!r}, expected one of {EVENT_TYPES}")
        query += " AND event_type = ?"
        params.append(event_type)
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
            "event_type": event_type_,
            "timestamp": datetime.datetime.fromisoformat(ts),
            "source": source,
            "device_id": device_id,
            "note": note,
        }
        for id_, event_type_, ts, source, device_id, note in rows
    ]
