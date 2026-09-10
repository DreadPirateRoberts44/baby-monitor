"""Audio preprocessing steps applied upstream of feature extraction:

1. A high-pass filter at ~300Hz to remove room rumble, handling noise, and
   HVAC/low-frequency hum below the fundamental frequency range of infant
   cries -- a standard step in cry-acoustics literature.

2. Extraction of a representative ~1-second cry window (one bout plus its
   trailing pause) from a longer clip, rather than using the whole
   fixed-duration clip. Reuses the energy-gated bout/pause segmentation
   from cry_rhythm_features.py. An exploratory pass across several real
   clips (see conversation/PR history) found the cry -> pause -> next-bout
   rhythm structure clearly visible via energy alone, but did NOT find a
   reliable secondary "gasp" signature via spectral flatness or any other
   simple spectral measure tried -- so this extracts a cry+pause window
   without attempting to isolate a gasp sub-phase specifically.

Both steps are meant to be applied before any of prosody_features.py,
cry_rhythm_features.py, voice_quality_features.py, spectrogram_features.py,
or yamnet_features.py -- they operate on the *preprocessed* waveform.
"""
import numpy as np
import scipy.signal

import config
from cry_rhythm_features import _energy_gate, _smooth_mask, _runs, HOP_LENGTH, FRAME_LENGTH, MIN_BOUT_FRAMES, MIN_PAUSE_FRAMES

HIGHPASS_CUTOFF_HZ = 300
HIGHPASS_ORDER = 4

CRY_WINDOW_DURATION = 1.0  # seconds
CRY_WINDOW_SAMPLES = int(CRY_WINDOW_DURATION * config.SAMPLE_RATE)


def highpass_filter(waveform, sr=config.SAMPLE_RATE, cutoff=HIGHPASS_CUTOFF_HZ, order=HIGHPASS_ORDER):
    """Zero-phase Butterworth high-pass filter. sosfiltfilt avoids the
    phase distortion a causal filter would introduce, which matters here
    since later pitch/rhythm features are sensitive to waveform shape."""
    sos = scipy.signal.butter(order, cutoff, btype="highpass", fs=sr, output="sos")
    return scipy.signal.sosfiltfilt(sos, waveform).astype(np.float32)


def _select_best_bout(waveform, sr):
    """Returns (start_sample, end_sample) of the highest-energy bout in the
    clip, using the same energy-gated segmentation as
    cry_rhythm_features.py. Falls back to the whole clip if no bout is
    detected (e.g. a near-silent clip)."""
    import librosa

    rms = librosa.feature.rms(y=waveform, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    voiced_mask = _energy_gate(rms)
    voiced_mask = _smooth_mask(voiced_mask, MIN_BOUT_FRAMES, MIN_PAUSE_FRAMES)
    bout_runs = [(s, e) for s, e, v in _runs(voiced_mask) if v]

    if not bout_runs:
        return 0, len(waveform)

    # pick the loudest bout (highest peak RMS within the run), not just the
    # longest -- a short, sharp cry burst can be more representative than a
    # long, quiet stretch
    def bout_peak_energy(run):
        s, e = run
        return rms[s:e].max() if e > s else 0.0

    best_s, best_e = max(bout_runs, key=bout_peak_energy)
    start_sample = best_s * HOP_LENGTH
    end_sample = min(len(waveform), best_e * HOP_LENGTH)
    return start_sample, end_sample


def extract_cry_window(waveform, sr=config.SAMPLE_RATE, window_samples=CRY_WINDOW_SAMPLES):
    """Extracts a fixed-length (window_samples) segment centered on the
    clip's most prominent cry bout, including some of its surrounding
    pause. Pads with the clip's own edge values (not zeros) if the clip is
    shorter than the window, so downstream energy-based features don't see
    an artificial hard silence edge."""
    bout_start, bout_end = _select_best_bout(waveform, sr)
    bout_center = (bout_start + bout_end) // 2

    half = window_samples // 2
    start = bout_center - half
    end = start + window_samples

    if start < 0:
        end -= start
        start = 0
    if end > len(waveform):
        start -= (end - len(waveform))
        end = len(waveform)
    start = max(0, start)

    window = waveform[start:end]
    if len(window) < window_samples:
        pad_amount = window_samples - len(window)
        window = np.pad(window, (0, pad_amount), mode="edge")
    return window[:window_samples].astype(np.float32)


def preprocess_waveform(waveform, sr=config.SAMPLE_RATE, apply_highpass=True, apply_windowing=True):
    """Convenience wrapper: high-pass filter, then extract the ~1s cry
    window. Either step can be disabled for A/B comparison."""
    y = waveform
    if apply_highpass:
        y = highpass_filter(y, sr)
    if apply_windowing:
        y = extract_cry_window(y, sr)
    return y
