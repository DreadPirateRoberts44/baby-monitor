"""Frame-level (non-pooled) feature sequences for the cry-reason sequence
models in train_cry_reason_sequence.py. Unlike prosody_features.py (which
collapses a clip to scalar summary stats) or yamnet_features.py (which
mean-pools YAMNet's per-frame embeddings), these keep the time axis so a
Conv1D/GRU can learn temporal/rhythm patterns directly instead of relying
on hand-designed bout segmentation.

Clip duration is fixed (config.DURATION), so frame count is fixed too —
no padding/masking needed downstream.
"""
import numpy as np
import librosa
import tensorflow_hub as hub

import config

HOP_LENGTH = 512
FRAME_LENGTH = 2048
PROSODY_SEQ_FEATURE_NAMES = ["f0", "voiced", "rms", "spectral_centroid", "zcr"]
N_PROSODY_SEQ_FEATURES = len(PROSODY_SEQ_FEATURE_NAMES)
N_FRAMES = 1 + config.N_SAMPLES // HOP_LENGTH  # librosa's centered-frame count

_yamnet_model = None


def _get_yamnet():
    global _yamnet_model
    if _yamnet_model is None:
        _yamnet_model = hub.load(config.YAMNET_HANDLE)
    return _yamnet_model


def extract_prosody_sequence(waveform, sr=config.SAMPLE_RATE):
    """Returns (N_FRAMES, N_PROSODY_SEQ_FEATURES) float32 array: per-frame
    F0, voicing flag, RMS energy, spectral centroid, zero-crossing rate."""
    y = waveform.astype(np.float32)

    f0 = librosa.yin(y, sr=sr, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))
    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]

    n = min(len(f0), len(rms), len(centroid), len(zcr), N_FRAMES)
    f0, rms, centroid, zcr = f0[:n], rms[:n], centroid[:n], zcr[:n]

    voiced = (rms > 0.15 * rms.max()).astype(np.float32) if rms.size else np.zeros(n, dtype=np.float32)
    f0_clean = np.nan_to_num(f0, nan=0.0)

    seq = np.stack([f0_clean, voiced, rms, centroid, zcr], axis=1).astype(np.float32)
    if seq.shape[0] < N_FRAMES:
        pad = np.zeros((N_FRAMES - seq.shape[0], N_PROSODY_SEQ_FEATURES), dtype=np.float32)
        seq = np.concatenate([seq, pad], axis=0)
    return seq[:N_FRAMES]


def extract_yamnet_sequence(waveform):
    """Returns (n_yamnet_frames, 1024) float32 array — YAMNet's own
    per-frame (~0.48s) embeddings, NOT mean-pooled over time."""
    model = _get_yamnet()
    _, embeddings, _ = model(waveform.astype(np.float32))
    return embeddings.numpy().astype(np.float32)
