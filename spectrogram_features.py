"""Log-mel spectrogram extraction for the CNN-on-raw-spectrogram cry-reason
model (train_cry_reason_cnn.py). Unlike every other feature source tried in
this project (YAMNet embeddings, scalar prosody, rhythm, voice-quality),
this keeps the full time-frequency representation and lets a CNN learn its
own features directly, rather than relying on hand-engineered summary
statistics or a general-purpose sound-event embedding not trained for cry
discrimination.

Small datasets are the main risk with this approach (a CNN over raw
spectrograms is more data-hungry than a head on top of a pretrained
embedding) -- see train_cry_reason_cnn.py for the augmentation and
regularization used to manage that.
"""
import numpy as np
import librosa

import config

N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256
# (64, 251) for a 4.0s clip at 16kHz with these params
N_TIME_FRAMES = 1 + config.N_SAMPLES // HOP_LENGTH


def extract_log_mel_spectrogram(waveform, sr=config.SAMPLE_RATE):
    """waveform: 1-D float32 array. Returns (N_MELS, N_TIME_FRAMES) float32
    log-mel spectrogram, normalized to roughly [-1, 1] via a fixed dB range
    (not per-clip min/max, so silence/quiet clips don't get rescaled into
    looking like loud ones)."""
    y = waveform.astype(np.float32)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH,
    )
    log_mel = librosa.power_to_db(mel, top_db=80.0)  # roughly [-80, 0] dB
    normalized = (log_mel + 40.0) / 40.0  # map [-80, 0] -> [-1, 1]

    if normalized.shape[1] < N_TIME_FRAMES:
        pad = N_TIME_FRAMES - normalized.shape[1]
        normalized = np.pad(normalized, ((0, 0), (0, pad)), mode="constant", constant_values=-1.0)
    return normalized[:, :N_TIME_FRAMES].astype(np.float32)
