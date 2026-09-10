"""Crying-session state tracking, so monitor.py notifies once at the start
and once at the end of a crying episode instead of once per capture window
(a multi-minute cry would otherwise fire a notification every
CAPTURE_INTERVAL_SECONDS -- e.g. ~75 notifications for a 5-minute cry at
the default 4s interval).

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
a distinct call from notify_cry_started/notify_cry_ended, so the app can
render it as a live update to the existing session rather than a new
notification/alert.
"""
import datetime
import time

import settings

IDLE = "idle"
CRYING = "crying"


class CrySession:
    """Feed it one predict() result per captured window via update(). It
    returns a session event each call:
      "started"        - a new crying session began this window
      "reason_updated" - session ongoing, and the top-ranked aggregated
                          stage-2 reason changed since the last update
                          (see top_reason_changed() below) -- distinct from
                          "started"/"ended" because the app should treat
                          this as a live update to an existing session, not
                          a fresh alert
      "ended"           - the session just ended
      None              - nothing session-relevant happened this window
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
        self.non_cry_streak = 0
        self._stage2_prob_sum = {}
        self._stage2_sample_count = 0
        self._last_reported_top_reason = None

    @property
    def duration_seconds(self):
        if self.started_at is None:
            return 0.0
        return time.monotonic() - self.started_at

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
        self.non_cry_streak = 0
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
        # "uncertain" doesn't reset an active session (ambiguous, not
        # evidence the baby stopped) but also doesn't start a new one on
        # its own -- only a confident "cry" label starts a session.
        is_cry = label == settings.CRY_LABEL
        is_ambiguous = label == settings.UNCERTAIN_LABEL

        if self.state == IDLE:
            if is_cry:
                self.state = CRYING
                self.started_at = time.monotonic()
                self.started_at_utc = datetime.datetime.now(datetime.timezone.utc)
                self.non_cry_streak = 0
                self._accumulate_stage2(result["stage2_probs"])
                self._last_reported_top_reason = self._top_reason()
                return "started"
            return None

        # self.state == CRYING
        if is_cry or is_ambiguous:
            self.non_cry_streak = 0
            self._accumulate_stage2(result["stage2_probs"])
            current_top = self._top_reason()
            if current_top is not None and current_top != self._last_reported_top_reason:
                self._last_reported_top_reason = current_top
                return "reason_updated"
            return None

        self.non_cry_streak += 1
        if self.non_cry_streak >= settings.SESSION_END_GRACE_WINDOWS:
            self.state = IDLE  # leaves started_at/aggregated probs readable until next update()
            return "ended"
        return None

    def clear(self):
        """Call after handling an "ended" event to fully reset for the
        next session. Separated from the state transition in update() so
        the caller has a chance to read duration_seconds /
        aggregated_stage2_probs first."""
        self._reset()
