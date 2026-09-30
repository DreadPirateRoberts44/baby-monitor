"""Continuous microphone capture, chunked into fixed-length windows for
inference. Uses sounddevice (PortAudio bindings) -- works with a USB mic or
an I2S mic exposed as a standard ALSA input device on the Pi, and is
cross-platform enough to test on a dev machine with any working microphone.
"""
from math import gcd

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

import settings

# Many USB mics only open at 44.1/48 kHz through ALSA; cache the rate that
# worked so the fallback probe runs once, not every window.
_native_rate_cache = {}


def _record(n_samples, rate, device):
    recording = sd.rec(n_samples, samplerate=rate, channels=1, dtype="float32", device=device)
    sd.wait()
    return recording.reshape(-1)


def capture_window(duration_seconds=settings.DURATION_SECONDS, sample_rate=settings.SAMPLE_RATE,
                    device=settings.AUDIO_DEVICE):
    """Blocks until duration_seconds of audio has been recorded, returns a
    1-D float32 array of exactly round(duration_seconds * sample_rate)
    samples, mono. If the device can't open at sample_rate, records at its
    native rate and resamples."""
    n_samples = int(round(duration_seconds * sample_rate))
    rate = _native_rate_cache.get(device, sample_rate)
    if rate == sample_rate:
        try:
            return _record(n_samples, sample_rate, device)
        except sd.PortAudioError:
            rate = int(sd.query_devices(device, "input")["default_samplerate"])
            if rate == sample_rate:
                raise
            _native_rate_cache[device] = rate
    raw = _record(int(round(duration_seconds * rate)), rate, device)
    g = gcd(int(rate), int(sample_rate))
    out = resample_poly(raw, int(sample_rate) // g, int(rate) // g).astype(np.float32)
    if len(out) < n_samples:
        out = np.pad(out, (0, n_samples - len(out)))
    return out[:n_samples]


def iter_windows(interval_seconds=settings.CAPTURE_INTERVAL_SECONDS,
                  duration_seconds=settings.DURATION_SECONDS,
                  sample_rate=settings.SAMPLE_RATE, device=settings.AUDIO_DEVICE):
    """Generator: yields one capture_window() at a time, forever. When
    interval_seconds == duration_seconds (the default), windows are
    back-to-back and non-overlapping -- simplest behavior and what
    monitor.py uses by default. A shorter interval than duration is not
    handled here (would need a rolling buffer / overlapping capture, which
    sounddevice's blocking sd.rec() API doesn't support directly) --
    revisit with a streaming InputStream callback if lower latency to
    first detection is needed later.
    """
    if interval_seconds < duration_seconds:
        raise NotImplementedError(
            "Overlapping windows (interval < duration) need a streaming "
            "InputStream, not the current blocking sd.rec() capture. "
            "Set settings.CAPTURE_INTERVAL_SECONDS >= DURATION_SECONDS for now."
        )
    while True:
        yield capture_window(duration_seconds, sample_rate, device)
        idle = interval_seconds - duration_seconds
        if idle > 0:
            sd.sleep(int(idle * 1000))


def list_input_devices():
    """Prints available input devices with their sounddevice index, so
    settings.AUDIO_DEVICE can be set to a specific mic instead of the
    system default."""
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            print(f"  [{idx}] {dev['name']} (max_input_channels={dev['max_input_channels']})")


if __name__ == "__main__":
    print("Available input devices:")
    list_input_devices()
