import numpy as np
import tensorflow_hub as hub

import config

_yamnet_model = None


def _get_model():
    global _yamnet_model
    if _yamnet_model is None:
        _yamnet_model = hub.load(config.YAMNET_HANDLE)  # downloads once, then cached locally
    return _yamnet_model


def get_embedding(waveform):
    """waveform: 1-D float32 array at 16kHz. Returns mean-pooled (1024,) embedding."""
    model = _get_model()
    _, embeddings, _ = model(waveform)
    return embeddings.numpy().mean(axis=0).astype(np.float32)
