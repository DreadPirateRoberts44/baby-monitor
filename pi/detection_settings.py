"""App-tunable detection parameters, stored in SQLite so monitor.py and
sync_api.py (separate processes) share them, same pattern as
prediction_state.py. Only overrides are stored; anything never set falls
back to the defaults in settings.py.

monitor.py reads these once per captured window, so a change takes effect
on the next window (up to CAPTURE_INTERVAL_SECONDS later).
"""
import datetime
import os
import sqlite3

import settings

DB_PATH = os.path.join(os.path.dirname(__file__), "detection_settings.sqlite")

# key -> (type, min, max, default). Bounds are sanity limits, not tuned values.
SCHEMA = {
    "stage1_confidence_threshold": (float, 0.30, 0.99, settings.STAGE1_CONFIDENCE_THRESHOLD),
    "session_start_min_windows": (int, 1, 100, settings.SESSION_START_MIN_WINDOWS),
    "session_end_missed_windows": (int, 1, 10000, settings.SESSION_END_MISSED_WINDOWS),
}


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS detection_settings (
            key TEXT PRIMARY KEY,
            value REAL NOT NULL,
            changed_at TEXT NOT NULL   -- ISO 8601 UTC
        )
        """
    )
    return conn


def get_settings():
    """Returns {key: value} for every key in SCHEMA, with overrides applied."""
    conn = _connect()
    try:
        rows = dict(conn.execute("SELECT key, value FROM detection_settings").fetchall())
    finally:
        conn.close()
    return {
        key: typ(rows[key]) if key in rows else default
        for key, (typ, _lo, _hi, default) in SCHEMA.items()
    }


def validate(updates):
    """Returns an error string, or None if updates is a valid partial
    update. Booleans are rejected (bool is an int subclass in Python)."""
    if not isinstance(updates, dict) or not updates:
        return "body must be a non-empty JSON object"
    for key, value in updates.items():
        if key not in SCHEMA:
            return f"unknown setting '{key}'; valid: {sorted(SCHEMA)}"
        typ, lo, hi, _default = SCHEMA[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"{key} must be a number"
        if typ is int and value != int(value):
            return f"{key} must be an integer"
        if not lo <= value <= hi:
            return f"{key} must be between {lo} and {hi}"
    return None


def set_settings(updates):
    """Applies a validated partial update (call validate() first). Returns
    the full resulting settings dict."""
    changed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    conn = _connect()
    try:
        for key, value in updates.items():
            conn.execute(
                """
                INSERT INTO detection_settings (key, value, changed_at) VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, changed_at = excluded.changed_at
                """,
                (key, float(value), changed_at),
            )
        conn.commit()
    finally:
        conn.close()
    return get_settings()


def reset_settings():
    """Drops all overrides, restoring settings.py defaults."""
    conn = _connect()
    try:
        conn.execute("DELETE FROM detection_settings")
        conn.commit()
    finally:
        conn.close()
    return get_settings()
