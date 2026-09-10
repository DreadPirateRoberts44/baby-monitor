import urllib.request

import tensorflow as tf

import config
from data import list_files_and_labels, load_and_extract

YAMNET_TFLITE_URL = (
    "https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1?lite-format=tflite"
)


def download_yamnet_tflite():
    """One-time download of Google's official pretrained YAMNet TFLite model.
    Runs at export time only — the Pi never needs internet access."""
    print("Downloading yamnet.tflite ...")
    req = urllib.request.Request(YAMNET_TFLITE_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(config.YAMNET_TFLITE_PATH, "wb") as f:
        f.write(resp.read())
    print(f"Saved to {config.YAMNET_TFLITE_PATH}")


def representative_dataset():
    filepaths, _, _ = list_files_and_labels(config.DATA_DIR)
    for fp in filepaths[:200]:  # subset is enough to calibrate quantization
        emb = load_and_extract(fp)
        yield [emb[None, ...]]


def export_classifier_head():
    model = tf.keras.models.load_model(config.MODEL_PATH)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    # dynamic-range quantization: weights only, float activations — the head
    # is just dense layers so this is safe and needs no op-support checks
    tflite_model = converter.convert()

    with open(config.TFLITE_PATH, "wb") as f:
        f.write(tflite_model)

    print(f"Saved classifier head to {config.TFLITE_PATH}")
    print(f"Size: {len(tflite_model) / 1024:.1f} KB")


def export_cry_reason_head():
    """Same conversion as export_classifier_head(), for the stage-2
    (cry-reason) model. Reuses the same representative dataset (stage-1
    embeddings) for quantization calibration -- the head only cares about
    the distribution of embedding values, not the label, so this is a fine
    stand-in for cry-specific calibration data."""
    model = tf.keras.models.load_model(config.CRY_REASON_MODEL_PATH)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    tflite_model = converter.convert()

    with open(config.CRY_REASON_TFLITE_PATH, "wb") as f:
        f.write(tflite_model)

    print(f"Saved cry-reason head to {config.CRY_REASON_TFLITE_PATH}")
    print(f"Size: {len(tflite_model) / 1024:.1f} KB")


def main():
    download_yamnet_tflite()
    export_classifier_head()
    export_cry_reason_head()


if __name__ == "__main__":
    main()