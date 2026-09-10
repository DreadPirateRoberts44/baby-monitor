"""Standalone copy of audio_preprocessing.highpass_filter's core logic, for
the Pi runtime. Not imported from audio_preprocessing.py directly because
that module also imports cry_rhythm_features (for the rejected cry-window
extraction path), which pulls in librosa -- a much heavier dependency than
the Pi needs just for a Butterworth filter. Keep this in sync with
audio_preprocessing.HIGHPASS_CUTOFF_HZ / HIGHPASS_ORDER / highpass_filter()
if those change.
"""
import numpy as np
import scipy.signal

HIGHPASS_CUTOFF_HZ = 300
HIGHPASS_ORDER = 4


def highpass_filter(waveform, sr, cutoff=HIGHPASS_CUTOFF_HZ, order=HIGHPASS_ORDER):
    sos = scipy.signal.butter(order, cutoff, btype="highpass", fs=sr, output="sos")
    return scipy.signal.sosfiltfilt(sos, waveform).astype(np.float32)
