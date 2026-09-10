"""Wipes all stored baby-specific history so the device can be reused for
a different baby, regifted, or cleared after testing -- without needing
physical/SSH access to the Pi.

Triggered exclusively via sync_api.py's POST /reset. Deliberately does
NOT implement any confirmation/warning UX here -- the app owns that
entirely (this module does the deletion unconditionally the moment it's
called); see sync_api.py's docstring for why.

Scope: everything that's specific to one baby's history --
care_events.py (feed/change log), cry_history.py (completed cry
sessions). device_events.py (startup/shutdown log) is included too,
since a monitor being regifted starts that history over as well.

Deliberately NOT reset: prediction_state.py (app-controlled pause flag)
-- that's current operational state, not accumulated history, and
resetting it out from under an in-progress pause would be a surprising
side effect of what's meant to be a "clear the baby's data" action.

Each store is cleared via its own connection (not by deleting the
.sqlite files) so this works correctly even while monitor.py /
buttons_mqtt.py are running and hold their own connections open.
"""
import care_events
import cry_history
import device_events


def reset_all():
    """Deletes every row from care_events, cry_history, and device_events.
    Irreversible. Returns a dict of how many rows were removed from each,
    for the app to show a confirmation summary after the fact."""
    counts = {}

    conn = care_events._connect()
    try:
        counts["care_events"] = conn.execute("DELETE FROM care_events").rowcount
        conn.commit()
    finally:
        conn.close()

    conn = cry_history._connect()
    try:
        counts["cry_history"] = conn.execute("DELETE FROM cry_sessions").rowcount
        conn.commit()
    finally:
        conn.close()

    conn = device_events._connect()
    try:
        counts["device_events"] = conn.execute("DELETE FROM device_events").rowcount
        conn.commit()
    finally:
        conn.close()

    return counts
