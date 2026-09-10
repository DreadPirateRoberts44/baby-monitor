"""Cry-bout / pause rhythm features.

The scalar prosody features in prosody_features.py (mean F0, mean RMS, ...)
collapse the whole clip to summary statistics and cannot see temporal
*pattern* — e.g. the widely-cited description of hungry cries as "long,
low-pitched, and repeating with short pauses" is a statement about rhythm
(bout/pause structure over time), not about any single scalar value.

This module segments a clip into voiced "cry bouts" separated by silent
pauses (via an RMS energy gate with hysteresis + minimum-duration
smoothing, standard in cry/speech activity detection), then derives
rhythm features from that segmentation: bout count, bout/pause duration
stats, duty cycle, and pitch register per bout. These are general
infant-cry-acoustics properties (bout/pause structure, register, contour)
established in cry-acoustics literature (e.g. Wasz-Hockert-era and later
dysphonation/pitch-register studies), not features tied to one dataset.
"""
import numpy as np
import librosa

import config

RHYTHM_FEATURE_NAMES = [
    "bout_count",
    "bout_duration_mean", "bout_duration_std",
    "pause_duration_mean", "pause_duration_std",
    "duty_cycle",              # fraction of clip spent in a cry bout
    "bout_rate",                # bouts per second
    "f0_register_mean",         # mean voiced-frame F0 across the whole clip
    "f0_register_low_frac",     # fraction of voiced frames below 300 Hz ("low-pitched")
    "contour_rise_frac",        # fraction of within-bout frames where pitch is rising
    "contour_fall_frac",
]
NUM_RHYTHM_FEATURES = len(RHYTHM_FEATURE_NAMES)

# Frame parameters shared with the RMS/F0 extraction below.
HOP_LENGTH = 512
FRAME_LENGTH = 2048
MIN_BOUT_FRAMES = 3   # ~96ms at 16kHz/hop 512; ignore shorter energy blips
MIN_PAUSE_FRAMES = 2  # ~64ms; merge shorter gaps into the same bout


def _energy_gate(rms, threshold_ratio=0.15):
    """Binary voiced/unvoiced mask from an RMS envelope, gated relative to
    the clip's own peak energy so it adapts to recording loudness."""
    if rms.size == 0:
        return np.zeros(0, dtype=bool)
    threshold = threshold_ratio * rms.max()
    return rms > threshold


def _smooth_mask(mask, min_true_run, min_false_run):
    """Merge short gaps and drop short blips via run-length smoothing."""
    if mask.size == 0:
        return mask
    mask = mask.copy()

    # fill short False runs (pauses) surrounded by True (merge into bout)
    runs = _runs(mask)
    for start, end, val in runs:
        if not val and (end - start) < min_false_run:
            mask[start:end] = True

    # drop short True runs (blips) surrounded by False
    runs = _runs(mask)
    for start, end, val in runs:
        if val and (end - start) < min_true_run:
            mask[start:end] = False

    return mask


def _runs(mask):
    """Yield (start, end, value) for each contiguous run in a boolean array."""
    if mask.size == 0:
        return []
    changes = np.flatnonzero(np.diff(mask.astype(np.int8))) + 1
    boundaries = np.concatenate(([0], changes, [mask.size]))
    return [(boundaries[i], boundaries[i + 1], bool(mask[boundaries[i]]))
            for i in range(len(boundaries) - 1)]


def extract_rhythm_features(waveform, sr=config.SAMPLE_RATE):
    y = waveform.astype(np.float32)
    frame_time = HOP_LENGTH / sr

    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    voiced_mask = _energy_gate(rms)
    voiced_mask = _smooth_mask(voiced_mask, MIN_BOUT_FRAMES, MIN_PAUSE_FRAMES)

    bout_runs = [(s, e) for s, e, v in _runs(voiced_mask) if v]
    pause_runs = [(s, e) for s, e, v in _runs(voiced_mask) if not v]
    # ignore leading/trailing silence as "pauses" — only interior gaps
    # reflect the cry's own rhythm
    if pause_runs and bout_runs:
        first_bout_start = bout_runs[0][0]
        last_bout_end = bout_runs[-1][1]
        pause_runs = [(s, e) for s, e in pause_runs if s > first_bout_start and e < last_bout_end]

    bout_durations = np.array([(e - s) * frame_time for s, e in bout_runs])
    pause_durations = np.array([(e - s) * frame_time for s, e in pause_runs])

    bout_count = len(bout_runs)
    bout_duration_mean = float(bout_durations.mean()) if bout_durations.size else 0.0
    bout_duration_std = float(bout_durations.std()) if bout_durations.size else 0.0
    pause_duration_mean = float(pause_durations.mean()) if pause_durations.size else 0.0
    pause_duration_std = float(pause_durations.std()) if pause_durations.size else 0.0

    clip_duration = len(y) / sr
    duty_cycle = float(voiced_mask.mean()) if voiced_mask.size else 0.0
    bout_rate = bout_count / clip_duration if clip_duration > 0 else 0.0

    # Pitch register + contour, using the same fast yin tracker as
    # prosody_features.py, gated by the same voiced mask.
    f0 = librosa.yin(y, sr=sr, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))
    n = min(len(f0), len(voiced_mask))
    f0, mask = f0[:n], voiced_mask[:n]
    voiced_f0 = f0[mask]
    voiced_f0 = voiced_f0[~np.isnan(voiced_f0)]

    f0_register_mean = float(voiced_f0.mean()) if voiced_f0.size else 0.0
    f0_register_low_frac = float(np.mean(voiced_f0 < 300)) if voiced_f0.size else 0.0

    # contour shape: within each bout, what fraction of frame-to-frame
    # steps are rising vs falling (captures "rise-fall" cry melody shape
    # separately from bout timing)
    rise_count = fall_count = total_steps = 0
    for s, e in bout_runs:
        seg = f0[s:min(e, len(f0))]
        seg = seg[~np.isnan(seg)]
        if seg.size < 2:
            continue
        diffs = np.diff(seg)
        rise_count += int(np.sum(diffs > 0))
        fall_count += int(np.sum(diffs < 0))
        total_steps += diffs.size

    contour_rise_frac = rise_count / total_steps if total_steps else 0.0
    contour_fall_frac = fall_count / total_steps if total_steps else 0.0

    return np.array([
        bout_count,
        bout_duration_mean, bout_duration_std,
        pause_duration_mean, pause_duration_std,
        duty_cycle,
        bout_rate,
        f0_register_mean,
        f0_register_low_frac,
        contour_rise_frac,
        contour_fall_frac,
    ], dtype=np.float32)
