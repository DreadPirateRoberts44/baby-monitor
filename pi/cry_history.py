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
            reason_probs_json TEXT NOT NULL  -- full aggregated_stage2_probs, as JSON,
                                              -- for anything beyond just the top label
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cry_sessions_started ON cry_sessions(started_at)")
    return conn


def log_session(started_at, ended_at, duration_seconds, reason_probs):
    """Record one completed crying session.

    started_at, ended_at: timezone-aware datetime objects (UTC).
    duration_seconds: float, from session.CrySession.duration_seconds at
        the moment it ended.
    reason_probs: session.CrySession.aggregated_stage2_probs at the moment
        it ended -- a dict like {"hungry": 0.6, "fussy": 0.3, ...},
        already sorted descending. May be empty if stage 2 never ran.

    Returns the session's id.
    """
    import json

    session_id = str(uuid.uuid4())
    top_reason = next(iter(reason_probs), None)

    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO cry_sessions "
            "(id, started_at, ended_at, duration_seconds, top_reason, reason_probs_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, started_at.isoformat(), ended_at.isoformat(),
             duration_seconds, top_reason, json.dumps(reason_probs)),
        )
        conn.commit()
    finally:
        conn.close()
    return session_id


def get_sessions(since=None, limit=None):
    """Returns a list of completed sessions, most recent first, as dicts
    with keys: id, started_at, ended_at, duration_seconds, top_reason,
    reason_probs (parsed back from JSON into a dict).

    since: optional datetime, only sessions that started at or after this."""
    import json

    query = "SELECT id, started_at, ended_at, duration_seconds, top_reason, reason_probs_json FROM cry_sessions WHERE 1=1"
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
        }
        for id_, started_at, ended_at, duration_seconds, top_reason, reason_probs_json in rows
    ]
