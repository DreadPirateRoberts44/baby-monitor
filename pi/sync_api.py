"""Local HTTP API for app<->Pi sync, reachable only on the home WiFi
network -- same trust model as the MQTT broker (no auth; the network
boundary IS the security boundary, appropriate for a private home
network with no other authed component either).

Motivation: caregivers may want to (a) review/correct history from the
app (see cry_history.py's read-only history + care_events' delete_event
for corrections), and (b) log a feed/change event while away from the
house entirely (e.g. at a store) and have it merge into the Pi's
long-term storage once the app is back on the same WiFi. This is
deliberately NOT reachable from outside the home network (no Tailscale,
no cloud relay) -- the app is expected to store entries locally and sync
opportunistically when it detects it's on the same network as the Pi,
not push in real time from anywhere.

Endpoints:
  GET    /care_events?since=<ISO8601>        -- pull events (for the app
                                                  to catch up on what the
                                                  button/other app
                                                  instances have logged)
  POST   /care_events                        -- push one offline-collected
                                                  event (runs through dedup
                                                  -- see below)
  DELETE /care_events/<id>                   -- remove one event (error
                                                  correction)
  GET    /cry_history?since=<ISO8601>        -- read-only cry-session
                                                  history (the app never
                                                  originates this data)
  GET    /device_events?since=<ISO8601>      -- read-only startup/shutdown
                                                  history (device_events.py),
                                                  for explaining gaps in
                                                  cry_history when reviewing
  GET    /prediction_state                   -- current paused/active state
                                                  (prediction_state.py)
  POST   /prediction_state                   -- {"paused": true/false} --
                                                  app-controlled escape hatch
                                                  for a misbehaving model;
                                                  see prediction_state.py
  POST   /reset                              -- {"confirm": "RESET"} --
                                                  wipes care_events,
                                                  cry_history, and
                                                  device_events entirely;
                                                  see reset_db.py

POST /reset wipes all stored baby-specific history (feed/change log,
cry-session history, startup/shutdown log) -- for reusing the device for
a different baby, regifting it, or clearing test data. Irreversible, and
deliberately unconditional on this end the moment a valid request
arrives: any "are you sure?" warning/confirmation UX is entirely the
app's responsibility, not this API's -- this endpoint's only guard
against a stray/malformed request is requiring the literal body
{"confirm": "RESET"}, not a real authorization check (this API has none,
see above). prediction_state.py's pause flag is NOT touched by this --
it's current operational state, not history.

POST /prediction_state is the app-controlled pause/resume escape hatch
for when the classifier is misbehaving in production (e.g. hallucinating
cries, or fragmenting one crying episode into many sessions) -- see
prediction_state.py's docstring for why this pauses inference itself
rather than just silencing notifications on the phone, and how an
in-progress session is handled when pause takes effect.

POST /care_events dedup: if an incoming event is the same type as an
existing one within care_events.DEDUP_WINDOW_MINUTES, it's treated as a
likely duplicate (e.g. someone logged a feed in the app while out, then
also pressed the physical button once home for the same feed) and
skipped rather than stored twice. Trades a small false-negative risk (a
genuine second event that close together gets dropped) for a cleaner
history -- see care_events.is_likely_duplicate()'s docstring. The
skipped/duplicate response still returns 200 with a flag, not an error,
so the app doesn't need special-case handling for it.

Uses Python's stdlib http.server rather than a framework (Flask, etc.)
-- deliberately, to avoid adding a dependency for what's a small,
low-traffic API (occasional sync, not a real workload).
"""
import datetime
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import care_events
import cry_history
import device_events
import prediction_state
import reset_db
import settings

_EVENT_ID_PATH = re.compile(r"^/care_events/([^/]+)$")


def _parse_since(query):
    since_str = query.get("since", [None])[0]
    if not since_str:
        return None
    try:
        return datetime.datetime.fromisoformat(since_str)
    except ValueError:
        return None


def _event_to_json(event):
    return {
        "id": event["id"],
        "event_type": event["event_type"],
        "timestamp": event["timestamp"].isoformat(),
        "source": event["source"],
        "device_id": event["device_id"],
        "note": event["note"],
    }


def _session_to_json(session):
    return {
        "id": session["id"],
        "started_at": session["started_at"].isoformat(),
        "ended_at": session["ended_at"].isoformat(),
        "duration_seconds": session["duration_seconds"],
        "top_reason": session["top_reason"],
        "reason_probs": session["reason_probs"],
        "confirmed_cry_seconds": session["confirmed_cry_seconds"],
        "cry_density": session["cry_density"],
    }


def _device_event_to_json(event):
    return {
        "id": event["id"],
        "event_type": event["event_type"],
        "timestamp": event["timestamp"].isoformat(),
        "reason": event["reason"],
    }


class SyncRequestHandler(BaseHTTPRequestHandler):
    # Quiet down the default per-request stderr logging -- this API gets
    # hit occasionally by a phone on the LAN, not worth a log line each time.
    def log_message(self, format, *args):
        pass

    def _send_json(self, status, body):
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)

        if parsed.path == "/care_events":
            since = _parse_since(query)
            events = care_events.get_events(since=since)
            self._send_json(200, {"events": [_event_to_json(e) for e in events]})
            return

        if parsed.path == "/cry_history":
            since = _parse_since(query)
            sessions = cry_history.get_sessions(since=since)
            self._send_json(200, {"sessions": [_session_to_json(s) for s in sessions]})
            return

        if parsed.path == "/device_events":
            since = _parse_since(query)
            events = device_events.get_events(since=since)
            self._send_json(200, {"events": [_device_event_to_json(e) for e in events]})
            return

        if parsed.path == "/prediction_state":
            self._send_json(200, {"paused": prediction_state.is_paused()})
            return

        self._send_json(404, {"error": "not found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path not in ("/care_events", "/prediction_state", "/reset"):
            self._send_json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, ValueError):
            self._send_json(400, {"error": "invalid JSON body"})
            return

        if parsed.path == "/reset":
            # The app owns all confirmation/warning UX -- this is just a
            # guard against a stray/malformed POST wiping data by accident,
            # not a real authorization check (this API has none).
            if body.get("confirm") != "RESET":
                self._send_json(400, {"error": 'must include {"confirm": "RESET"}'})
                return
            counts = reset_db.reset_all()
            self._send_json(200, {"status": "reset", "deleted": counts})
            return

        if parsed.path == "/prediction_state":
            paused = body.get("paused")
            if not isinstance(paused, bool):
                self._send_json(400, {"error": "paused must be a JSON boolean"})
                return
            changed_at = prediction_state.set_paused(paused)
            self._send_json(200, {"paused": paused, "changed_at": changed_at.isoformat()})
            return

        event_type = body.get("event_type")
        if event_type not in care_events.EVENT_TYPES:
            self._send_json(400, {"error": f"event_type must be one of {sorted(care_events.EVENT_TYPES)}"})
            return

        timestamp_str = body.get("timestamp")
        if timestamp_str:
            try:
                timestamp = datetime.datetime.fromisoformat(timestamp_str)
            except ValueError:
                self._send_json(400, {"error": "timestamp must be ISO 8601"})
                return
        else:
            timestamp = datetime.datetime.now(datetime.timezone.utc)

        if care_events.is_likely_duplicate(event_type, timestamp):
            self._send_json(200, {"status": "skipped_duplicate"})
            return

        event_id = care_events.log_event(
            event_type,
            source="app",
            device_id=body.get("device_id"),
            note=body.get("note"),
            timestamp=timestamp,
        )
        self._send_json(201, {"status": "created", "id": event_id})

    def do_DELETE(self):
        match = _EVENT_ID_PATH.match(urlparse(self.path).path)
        if not match:
            self._send_json(404, {"error": "not found"})
            return

        event_id = match.group(1)
        deleted = care_events.delete_event(event_id)
        if deleted:
            self._send_json(200, {"status": "deleted"})
        else:
            self._send_json(404, {"error": "no event with that id"})


def run_sync_api():
    """Blocks forever, serving the sync API on settings.SYNC_API_PORT."""
    server = ThreadingHTTPServer(("0.0.0.0", settings.SYNC_API_PORT), SyncRequestHandler)
    print(f"[sync_api] listening on 0.0.0.0:{settings.SYNC_API_PORT} (LAN only -- no auth, see module docstring)")
    server.serve_forever()


if __name__ == "__main__":
    run_sync_api()
