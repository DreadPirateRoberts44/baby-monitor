"""Crying-session state tracking, so monitor.py notifies once at the start
of a crying episode instead of once per capture window (a multi-minute cry
would otherwise fire a notification every CAPTURE_INTERVAL_SECONDS -- e.g.
~75 notifications for a 5-minute cry at the default 4s interval).

A session only actually STARTS (and alerts) after
settings.SESSION_START_MIN_WINDOWS *consecutive* confident "cry" windows
-- a single confident-but-wrong window (a bark, a static burst, anything
that briefly clears STAGE1_CONFIDENCE_THRESHOLD) used to be enough on its
own to fire a full alert. Any non-cry window before that count is reached
throws away the candidate entirely (see PENDING below) -- this is
deliberately NOT the same tolerance as the merge window below, which only
applies to an already-confirmed session; an unconfirmed candidate has no
track record yet, so a single interruption is treated as "that wasn't
really a cry" rather than "brief pause mid-cry".

Once confirmed, a session stays open across gaps of up to
settings.SESSION_MERGE_WINDOW_SECONDS since the last confident "cry"
window -- e.g. baby cries, is picked up and carried out of mic range (or
just settles down), then cries again 10 minutes later: that's folded into
the SAME session rather than starting a new one and firing a second
"started" alert for what's really one episode. Only a confident "cry"
label extends the gap timer; "uncertain" neither extends it nor closes the
session on its own (see update()). The session is only actually considered
over once that gap elapses with no further confident crying.

Critically, the session's start/end -- and therefore duration_seconds --
reflect the FIRST (of the confirmed run) and LAST confident "cry" windows,
not whenever the merge window happens to lapse. A 5-minute cry followed by
30 minutes of silence before the merge window closes the session should
log as a 5-minute session, not 35 minutes -- see last_cry_at/
last_cry_at_utc below.

Also tracks CRY DENSITY: how much of the session's duration was actually
confident crying vs. quiet gaps absorbed by the merge window above.
confirmed_cry_seconds is (confident-cry window count) *
settings.CAPTURE_INTERVAL_SECONDS -- an approximation (each window's
crying could be anywhere within that window, and windows may overlap if
CAPTURE_INTERVAL_SECONDS < DURATION_SECONDS) but good enough to distinguish
"12 minutes, wall-to-wall crying" from "12 minutes, but only 90s of it was
confirmed crying, rest was a merge-window gap" -- the latter became
possible to construct once sessions started tolerating multi-minute gaps,
and duration_seconds alone can no longer tell them apart. cry_density is
the same thing as a 0.0-1.0 ratio of confirmed_cry_seconds / duration_seconds.

Also aggregates stage-2 (cry-reason) predictions across the session as a
running average of probability vectors, on the hypothesis that averaging
many noisy per-window estimates reduces variance in the reason estimate
compared to trusting a single window. This is a plausible improvement but
UNVERIFIED against real multi-window session data or systematic bias in
the underlying model (see the project's README/experiments/ history: some
of stage 2's error looks like bias -- e.g. hungry's weak recall stems from
a genuinely weak standalone acoustic signature, not sampling noise -- which
averaging would not fix). Treat the aggregated reason as a refinement to
evaluate, not an assumed win.

The aggregated reason is pushed to the app each time the top-ranked reason
changes during an ongoing session (update() returns "reason_updated"), NOT
as a repeated alert -- monitor.py sends these via notify.send_reason_update,
a distinct call from notify_cry_started, so the app can render it as a live
update to the existing session rather than a new notification/alert.

"ended" (fires settings.SESSION_MERGE_WINDOW_SECONDS after the last
confident cry) IS alerted -- see notify.py -- so the app can stop showing
an active session even if nothing else told it to.
"""
import datetime
import time

import settings

IDLE = "idle"
PENDING = "pending"
CRYING = "crying"


class CrySession:
    """Feed it one predict() result per captured window via update(). It
    returns a session event each call:
      "started"        - a crying session was just CONFIRMED this window
                          (settings.SESSION_START_MIN_WINDOWS consecutive
                          confident "cry" windows reached) -- not
                          necessarily the first window that looked like a
                          cry; see started_at_utc, which is backdated to
                          that first window, not this confirming one
      "reason_updated" - session ongoing, and the top-ranked aggregated
                          stage-2 reason changed since the last update
                          (see top_reason_changed() below) -- distinct from
                          "started"/"ended" because the app should treat
                          this as a live update to an existing session, not
                          a fresh alert
      "ended"           - a confirmed session just ended (no confident
                          crying for settings.SESSION_MERGE_WINDOW_SECONDS
                          since the last one)
      None              - nothing session-relevant happened this window
                          (includes an unconfirmed candidate still
                          accumulating, or one that was just discarded)
    Exposes the running-average stage-2 probabilities for the current (or
    just-ended) session via aggregated_stage2_probs, and the session's
    start time via started_at_utc (a real datetime, for logging -- see
    duration_seconds/started_at, which use time.monotonic() and are only
    valid for measuring elapsed time, not as an absolute timestamp).
    """

    def __init__(self):
        self.state = IDLE
        self.started_at = None       # time.monotonic() -- for duration_seconds only,
                                      # NOT a real timestamp (monotonic has no fixed epoch)
        self.started_at_utc = None   # datetime.datetime (UTC) -- for actual start-time
                                      # timestamps, e.g. cry_history.py logging
        self.last_cry_at = None      # time.monotonic() of the most recent confident
                                      # "cry" window -- the merge-window gap is measured
                                      # from this, NOT from started_at
        self.last_cry_at_utc = None  # datetime.datetime (UTC) of that same window --
                                      # this, not "now", is the session's real end time
        self._pending_streak = 0     # consecutive confident-"cry" windows seen so far
                                      # while state == PENDING, toward SESSION_START_MIN_WINDOWS
        self._cry_window_count = 0   # confident-"cry" windows in the CONFIRMED session so
                                      # far -- see confirmed_cry_seconds/cry_density
        self._stage2_prob_sum = {}
        self._stage2_sample_count = 0
        self._last_reported_top_reason = None

    @property
    def duration_seconds(self):
        """Start of the episode to the LAST confident cry, not to "now" --
        so a session that's sitting open through its trailing merge-window
        silence still reports the real episode length, not one inflated by
        however long it's been quiet.

        Floored at CAPTURE_INTERVAL_SECONDS (one window-width): started_at
        and last_cry_at are both time.monotonic() readings taken AT THE
        MOMENT a window was classified, not the span the window's audio
        actually covered, so a session still on its very first confirming
        window (last_cry_at == started_at, elapsed wall-clock ~0) would
        otherwise read as a 0-length session despite representing a real
        CAPTURE_INTERVAL_SECONDS of confirmed crying -- that both misstates
        the episode's real length and would make cry_density divide by
        ~0 (see below)."""
        if self.started_at is None:
            return 0.0
        return max(settings.CAPTURE_INTERVAL_SECONDS, self.last_cry_at - self.started_at)

    @property
    def confirmed_cry_seconds(self):
        """Approximate seconds of ACTUAL confident crying within the
        session, as opposed to duration_seconds (which includes any quiet
        gaps the merge window absorbed). windows * CAPTURE_INTERVAL_SECONDS
        -- an approximation, not a precise audio-level measurement (see
        module docstring), but enough to tell a dense session from a sparse
        one now that sessions can span long quiet gaps."""
        return self._cry_window_count * settings.CAPTURE_INTERVAL_SECONDS

    @property
    def cry_density(self):
        """confirmed_cry_seconds / duration_seconds, as a 0.0-1.0 ratio --
        1.0 means wall-to-wall confirmed crying with no absorbed gaps.
        0.0 (rather than a division error) if duration_seconds is 0 (e.g.
        read before any window has been processed)."""
        duration = self.duration_seconds
        if duration <= 0:
            return 0.0
        return min(1.0, self.confirmed_cry_seconds / duration)

    @property
    def aggregated_stage2_probs(self):
        """Running-average stage-2 probabilities across the session so far,
        sorted descending. Empty dict if stage 2 hasn't run this session
        (e.g. every window so far was "uncertain" with no stage2_probs)."""
        if self._stage2_sample_count == 0:
            return {}
        averaged = {
            label: total / self._stage2_sample_count
            for label, total in self._stage2_prob_sum.items()
        }
        return dict(sorted(averaged.items(), key=lambda kv: kv[1], reverse=True))

    def _accumulate_stage2(self, stage2_probs):
        if not stage2_probs:
            return
        for label, prob in stage2_probs.items():
            self._stage2_prob_sum[label] = self._stage2_prob_sum.get(label, 0.0) + prob
        self._stage2_sample_count += 1

    def _top_reason(self):
        probs = self.aggregated_stage2_probs
        return next(iter(probs), None)

    def _reset(self):
        self.state = IDLE
        self.started_at = None
        self.started_at_utc = None
        self.last_cry_at = None
        self.last_cry_at_utc = None
        self._pending_streak = 0
        self._cry_window_count = 0
        self._stage2_prob_sum = {}
        self._stage2_sample_count = 0
        self._last_reported_top_reason = None

    def update(self, result):
        """result: the dict returned by inference.CryPredictor.predict().
        Returns "started", "reason_updated", "ended", or None. On "ended",
        duration_seconds and aggregated_stage2_probs still reflect the
        just-finished session -- read them before the next update() call,
        which starts clearing state for the next (potential) session."""
        label = result["stage1_label"]
        # "uncertain" doesn't reset an active/pending session (ambiguous,
        # not evidence the baby stopped) but also doesn't start or confirm
        # one on its own, and doesn't extend the merge-window gap either --
        # only a confident "cry" label does any of those.
        is_cry = label == settings.CRY_LABEL

        if self.state == IDLE:
            if is_cry:
                self._begin_candidate(result)
                if settings.SESSION_START_MIN_WINDOWS <= 1:
                    return self._confirm()
                self.state = PENDING
            return None

        if self.state == PENDING:
            if is_cry:
                self._extend_candidate(result)
                if self._pending_streak >= settings.SESSION_START_MIN_WINDOWS:
                    return self._confirm()
                return None
            if label == settings.UNCERTAIN_LABEL:
                # Ambiguous doesn't confirm the candidate, but doesn't
                # discard it either -- treated the same as an active
                # session's "uncertain" window (see module docstring).
                return None
            # A confident NON-cry window while unconfirmed -- this wasn't
            # a real cry, discard the candidate entirely (deliberately
            # stricter than the merge window: an unconfirmed candidate
            # gets no gap tolerance).
            self._reset()
            return None

        # self.state == CRYING (confirmed)
        if is_cry:
            self._extend_candidate(result)
            current_top = self._top_reason()
            if current_top is not None and current_top != self._last_reported_top_reason:
                self._last_reported_top_reason = current_top
                return "reason_updated"
            return None

        # Not a confident "cry" this window (includes "uncertain") -- the
        # session stays open, silently, until the gap since the last
        # confident cry crosses the merge window.
        if time.monotonic() - self.last_cry_at >= settings.SESSION_MERGE_WINDOW_SECONDS:
            self.state = IDLE  # leaves started_at/last_cry_at/aggregated probs
                                # readable until next update() or clear()
            return "ended"
        return None

    def _begin_candidate(self, result):
        """First confident "cry" window of a brand new (unconfirmed or
        immediately-confirmed) candidate -- sets start/last-cry timestamps
        and accumulates stage 2, but does NOT change self.state; caller
        decides PENDING vs. immediate confirmation."""
        now = time.monotonic()
        self.started_at = now
        self.started_at_utc = datetime.datetime.now(datetime.timezone.utc)
        self.last_cry_at = now
        self.last_cry_at_utc = self.started_at_utc
        self._pending_streak = 1
        self._cry_window_count = 1
        self._accumulate_stage2(result["stage2_probs"])
        self._last_reported_top_reason = self._top_reason()

    def _extend_candidate(self, result):
        """A subsequent confident "cry" window, in either PENDING or
        CRYING state -- advances last-cry timestamps, the pending streak
        (harmless once confirmed; only checked while PENDING), and the
        confirmed cry-window count / stage-2 aggregation."""
        self.last_cry_at = time.monotonic()
        self.last_cry_at_utc = datetime.datetime.now(datetime.timezone.utc)
        self._pending_streak += 1
        self._cry_window_count += 1
        self._accumulate_stage2(result["stage2_probs"])

    def _confirm(self):
        """Transitions PENDING (or an immediately-confirmed IDLE start,
        when SESSION_START_MIN_WINDOWS <= 1) into CRYING and returns
        "started". started_at_utc is already the FIRST candidate window's
        timestamp (set in _begin_candidate), not this confirming window's
        -- so a confirmed session's logged start time reflects when the
        crying actually began, not when the app was told about it."""
        self.state = CRYING
        return "started"

    def clear(self):
        """Call after handling an "ended" event to fully reset for the
        next session. Separated from the state transition in update() so
        the caller has a chance to read duration_seconds /
        aggregated_stage2_probs first."""
        self._reset()
