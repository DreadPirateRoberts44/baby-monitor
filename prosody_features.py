"""Hand-engineered prosodic/acoustic features for cry-reason classification.

Infant cry-reason literature (e.g. Donate-a-Cry, dunstan baby language
studies) points to pitch contour and energy dynamics — not general timbre —
as the signal that distinguishes *why* a baby is crying. YAMNet embeddings
are tuned for sound-event identity (crying vs. laughing vs. silence) and
were shown (via t-SNE + centroid analysis) not to separate cry sub-types.
This module extracts a small, interpretable feature vector per clip instead.

Feature groups (all scalar, ~20 dims total):
  - F0 (pitch) stats: mean, std, min, max, range, slope (rise/fall trend)
  - Voiced fraction: proportion of frames with detected pitch (voicing/
    phonation ratio — cries vary in how continuously voiced they are)
  - Energy/RMS stats: mean, std, max
  - Spectral shape: centroid mean/std, bandwidth mean, rolloff mean
    (captures harshness/breathiness differences)
  - Zero-crossing rate mean/std (noisiness proxy)
  - Tempo proxy: onset strength mean/std (burst/rhythm pattern of cry)
"""
import numpy as np
import librosa

import config

FEATURE_NAMES = [
    "f0_mean", "f0_std", "f0_min", "f0_max", "f0_range", "f0_slope",
    "voiced_fraction",
    "rms_mean", "rms_std", "rms_max",
    "spectral_centroid_mean", "spectral_centroid_std",
    "spectral_bandwidth_mean",
    "spectral_rolloff_mean",
    "zcr_mean", "zcr_std",
    "onset_strength_mean", "onset_strength_std",
]
NUM_FEATURES = len(FEATURE_NAMES)


def _safe_stats(x):
    """mean/std that don't NaN out on empty/constant arrays."""
    if x.size == 0:
        return 0.0, 0.0
    return float(np.mean(x)), float(np.std(x))


def extract_prosody_features(waveform, sr=config.SAMPLE_RATE):
    """waveform: 1-D float32 array. Returns (NUM_FEATURES,) float32 vector."""
    y = waveform.astype(np.float32)

    # yin is much faster than pyin (no HMM/Viterbi pass) at the cost of a
    # cruder voiced/unvoiced decision, which we approximate via an RMS
    # energy gate below — adequate for summary pitch statistics.
    f0 = librosa.yin(y, sr=sr, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))
    frame_rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    n = min(len(f0), len(frame_rms))
    f0, frame_rms = f0[:n], frame_rms[:n]
    voiced_flag = frame_rms > (0.1 * frame_rms.max() if frame_rms.size else 0.0)
    voiced_f0 = f0[voiced_flag]
    voiced_f0 = voiced_f0[~np.isnan(voiced_f0)]

    if voiced_f0.size > 0:
        f0_mean, f0_std = _safe_stats(voiced_f0)
        f0_min, f0_max = float(voiced_f0.min()), float(voiced_f0.max())
        f0_range = f0_max - f0_min
        # slope: linear fit of f0 over time as a crude rise/fall trend
        if voiced_f0.size > 1:
            f0_slope = float(np.polyfit(np.arange(voiced_f0.size), voiced_f0, 1)[0])
        else:
            f0_slope = 0.0
    else:
        f0_mean = f0_std = f0_min = f0_max = f0_range = f0_slope = 0.0

    voiced_fraction = float(np.mean(voiced_flag)) if voiced_flag is not None and voiced_flag.size else 0.0

    rms = librosa.feature.rms(y=y)[0]
    rms_mean, rms_std = _safe_stats(rms)
    rms_max = float(rms.max()) if rms.size else 0.0

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    centroid_mean, centroid_std = _safe_stats(centroid)

    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    bandwidth_mean, _ = _safe_stats(bandwidth)

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    rolloff_mean, _ = _safe_stats(rolloff)

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_mean, zcr_std = _safe_stats(zcr)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_mean, onset_std = _safe_stats(onset_env)

    return np.array([
        f0_mean, f0_std, f0_min, f0_max, f0_range, f0_slope,
        voiced_fraction,
        rms_mean, rms_std, rms_max,
        centroid_mean, centroid_std,
        bandwidth_mean,
        rolloff_mean,
        zcr_mean, zcr_std,
        onset_mean, onset_std,
    ], dtype=np.float32)
