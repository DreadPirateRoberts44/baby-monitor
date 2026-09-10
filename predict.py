"""Reference inference path: YAMNet embedding -> classifier head ->
confidence-thresholded label. Mirrors what an on-device (TFLite) consumer
of output/classifier_head.tflite should do; this version uses the Keras
model directly for simplicity.
"""
import sys

import numpy as np
import tensorflow as tf

import config
from data import load_and_extract


def load_class_names():
    with open(config.LABELS_PATH) as f:
        return [line.strip() for line in f if line.strip()]


def predict(model, class_names, filepath, threshold=config.CONFIDENCE_THRESHOLD):
    """Returns (label, confidence). label is config.UNCERTAIN_LABEL when the
    top prediction's confidence falls below threshold."""
    embedding = load_and_extract(filepath)
    probs = model.predict(embedding[None, ...], verbose=0)[0]
    top_idx = int(probs.argmax())
    confidence = float(probs[top_idx])

    if confidence < threshold:
        return config.UNCERTAIN_LABEL, confidence
    return class_names[top_idx], confidence


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <wav_or_ogg_file> [...]")
        sys.exit(1)

    class_names = load_class_names()
    model = tf.keras.models.load_model(config.MODEL_PATH)

    for filepath in sys.argv[1:]:
        label, confidence = predict(model, class_names, filepath)
        print(f"{filepath}: {label} (confidence {confidence:.3f})")


if __name__ == "__main__":
    main()
