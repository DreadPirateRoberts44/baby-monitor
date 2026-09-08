import tensorflow as tf

import config
from data import list_files_and_labels, load_and_extract


def representative_dataset():
    filepaths, _, _ = list_files_and_labels(config.DATA_DIR)
    for fp in filepaths[:200]:  # subset is enough to calibrate quantization
        feat = load_and_extract(fp)
        yield [feat[None, ...]]


def main():
    model = tf.keras.models.load_model(config.MODEL_PATH)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()

    with open(config.TFLITE_PATH, "wb") as f:
        f.write(tflite_model)

    print(f"Saved quantized TFLite model to {config.TFLITE_PATH}")
    print(f"Size: {len(tflite_model) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
