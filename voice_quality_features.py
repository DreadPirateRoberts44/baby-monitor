"""Voice-quality (dysphonation) features via Praat (parselmouth), plus
MFCCs, as a new signal source distinct from everything tried previously
(YAMNet embeddings, scalar prosody, bout/pause rhythm, frame-level
sequences). Infant-cry-acoustics literature associates pain/distress
cries with more "strained"/"rough"/dysphonated voicing -- jitter, shimmer,
and harmonics-to-noise ratio (HNR) are the standard quantitative measures
of that, not something the earlier prosody_features.py captured at all.

MFCCs are included too: a more detailed spectral-envelope/timbre
representation than the centroid/bandwidth/rolloff summary stats already
in prosody_features.py, and the standard feature for this kind of
classification task in the speech/cry literature.
"""
import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call

import config

N_MFCC = 13
VOICE_QUALITY_FEATURE_NAMES = [
    "jitter_local", "shimmer_local", "hnr_mean",
] + [f"mfcc{i}_mean" for i in range(N_MFCC)] + [f"mfcc{i}_std" for i in range(N_MFCC)]
NUM_VOICE_QUALITY_FEATURES = len(VOICE_QUALITY_FEATURE_NAMES)


def _safe_call(fn, default=0.0):
    try:
        val = fn()
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            return default
        return float(val)
    except Exception:
        return default


def extract_voice_quality_features(waveform, sr=config.SAMPLE_RATE):
    y = waveform.astype(np.float64)  # parselmouth wants float64

    snd = parselmouth.Sound(y, sampling_frequency=sr)

    def get_jitter():
        point_process = call(snd, "To PointProcess (periodic, cc)", 75, 600)
        return call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)

    def get_shimmer():
        point_process = call(snd, "To PointProcess (periodic, cc)", 75, 600)
        return call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    def get_hnr():
        harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        return call(harmonicity, "Get mean", 0, 0)

    jitter_local = _safe_call(get_jitter)
    shimmer_local = _safe_call(get_shimmer)
    hnr_mean = _safe_call(get_hnr)

    mfccs = librosa.feature.mfcc(y=waveform.astype(np.float32), sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = mfccs.mean(axis=1)
    mfcc_std = mfccs.std(axis=1)

    return np.concatenate([
        [jitter_local, shimmer_local, hnr_mean],
        mfcc_mean, mfcc_std,
    ]).astype(np.float32)
