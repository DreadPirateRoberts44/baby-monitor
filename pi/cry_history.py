"""Local store for completed crying sessions: start time, end time, and
the session's aggregated cry-reason estimate. Deliberately narrower than
care_events.py's scope -- only cry SESSIONS are stored here (not every
individual laugh/silence/noise window, which aren't useful history to
review later and would add a lot of low-value volume).

Same design rationale as care_events.py: SQLite (stdlib, no new
dependency) rather than a flat file, since it's simple, handles
concurrent access safely if a future local API server reads this
alongside monitor.py writing to it, and gives free query/filtering for a
future app to show history.
"""
import datetime
import os
import sqlite3
import uuid

DB_PATH = os.path.join(os.path.dirname(__file__), "cry_history.sqlite")


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cry_sessions (
            id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,   -- ISO 8601 UTC
            ended_at TEXT NOT NULL,     -- ISO 8601 UTC
            duration_seconds REAL NOT NULL,
            top_reason TEXT,            -- highest-probability aggregated reason, or NULL
                                         -- if stage 2 never ran this session (e.g. every
                                         -- window was "uncertain")
            reason_probs_json TEXT NOT NULL,  -- full aggregated_stage2_probs, as JSON,
                                               -- for anything beyond just the top label
            confirmed_cry_seconds REAL NOT NULL DEFAULT 0  -- session.CrySession.
                -- confirmed_cry_seconds: approximate seconds of ACTUAL confident
                -- crying within the session, vs. duration_seconds which also
                -- includes any quiet gaps absorbed by the merge window (see
                -- session.py). Ratio to duration_seconds == cry_density.
                -- DEFAULT 0 is only ever hit by the ALTER TABLE migration below
                -- backfilling pre-existing rows that predate this column -- every
                -- row logged going forward always passes a real value explicitly.
        )
        """
    )
    _migrate_add_confirmed_cry_seconds(conn)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cry_sessions_started ON cry_sessions(started_at)")
    return conn


def _migrate_add_confirmed_cry_seconds(conn):
    """ALTER TABLE for DBs created before confirmed_cry_seconds existed --
    CREATE TABLE IF NOT EXISTS above is a no-op against an existing table,
    so a pre-existing cry_history.sqlite needs this to pick up the new
    column. Safe to run every connect(): checked against the live schema
    first, so it's a no-op once already applied."""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(cry_sessions)")}
    if "confirmed_cry_seconds" not in columns:
        conn.execute(
            "ALTER TABLE cry_sessions ADD COLUMN confirmed_cry_seconds REAL NOT NULL DEFAULT 0"
        )


def log_session(started_at, ended_at, duration_seconds, reason_probs, confirmed_cry_seconds):
    """Record one completed crying session.

    started_at, ended_at: timezone-aware datetime objects (UTC).
    duration_seconds: float, from session.CrySession.duration_seconds at
        the moment it ended.
    reason_probs: session.CrySession.aggregated_stage2_probs at the moment
        it ended -- a dict like {"hungry": 0.6, "fussy": 0.3, ...},
        already sorted descending. May be empty if stage 2 never ran.
    confirmed_cry_seconds: float, from session.CrySession.
        confirmed_cry_seconds at the moment it ended -- approximate
        seconds of actual confident crying within the session, as opposed
        to duration_seconds (which also counts any quiet gaps the merge
        window absorbed). get_sessions() derives cry_density from this.

    Returns the session's id.
    """
    import json

    session_id = str(uuid.uuid4())
    top_reason = next(iter(reason_probs), None)

    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO cry_sessions "
            "(id, started_at, ended_at, duration_seconds, top_reason, reason_probs_json, "
            "confirmed_cry_seconds) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (session_id, started_at.isoformat(), ended_at.isoformat(),
             duration_seconds, top_reason, json.dumps(reason_probs),
             confirmed_cry_seconds),
        )
        conn.commit()
    finally:
        conn.close()
    return session_id


def get_sessions(since=None, limit=None):
    """Returns a list of completed sessions, most recent first, as dicts
    with keys: id, started_at, ended_at, duration_seconds, top_reason,
    reason_probs (parsed back from JSON into a dict), confirmed_cry_seconds,
    cry_density (confirmed_cry_seconds / duration_seconds, 0.0-1.0; 0.0 if
    duration_seconds is 0 -- see session.CrySession.cry_density).

    since: optional datetime, only sessions that started at or after this."""
    import json

    query = (
        "SELECT id, started_at, ended_at, duration_seconds, top_reason, "
        "reason_probs_json, confirmed_cry_seconds FROM cry_sessions WHERE 1=1"
    )
    params = []
    if since is not None:
        query += " AND started_at >= ?"
        params.append(since.isoformat())
    query += " ORDER BY started_at DESC"
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
            "started_at": datetime.datetime.fromisoformat(started_at),
            "ended_at": datetime.datetime.fromisoformat(ended_at),
            "duration_seconds": duration_seconds,
            "top_reason": top_reason,
            "reason_probs": json.loads(reason_probs_json),
            "confirmed_cry_seconds": confirmed_cry_seconds,
            "cry_density": min(1.0, confirmed_cry_seconds / duration_seconds) if duration_seconds > 0 else 0.0,
        }
        for id_, started_at, ended_at, duration_seconds, top_reason, reason_probs_json, confirmed_cry_seconds
        in rows
    ]
