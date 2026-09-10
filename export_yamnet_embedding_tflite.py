"""Export a YAMNet-embedding TFLite model for the Pi runtime.

output/yamnet.tflite (downloaded by export_tflite.py) is Google's official
*classification* build: fixed 15600-sample (0.975s) input, 521-class
AudioSet softmax output, no embedding layer exposed. It cannot produce the
1024-dim mean-pooled embedding the classifier heads (classifier_head.tflite,
cry_reason_mlp.keras) were actually trained on -- that embedding comes from
yamnet_features.get_embedding(), which uses the full TF Hub YAMNet module's
'output_1' (the pre-classification embedding, shape (frames, 1024)) and
mean-pools across frames.

This wraps that same embedding output in a tf.Module, fixes the input
length to config.N_SAMPLES (so the TFLite converter doesn't need to handle
a fully dynamic shape) and does the mean-pooling inside the graph, then
converts to TFLite -- so the Pi can get an embedding via the lightweight
TFLite runtime, matching training exactly, without depending on
tensorflow_hub or a full TensorFlow install.
"""
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

import config
import yamnet_features

EMBEDDING_TFLITE_PATH = config.YAMNET_EMBEDDING_TFLITE_PATH


class YamnetEmbedding(tf.Module):
    def __init__(self, yamnet_model):
        super().__init__()
        self._yamnet = yamnet_model

    @tf.function(input_signature=[tf.TensorSpec(shape=[config.N_SAMPLES], dtype=tf.float32)])
    def __call__(self, waveform):
        _, embeddings, _ = self._yamnet(waveform)
        return tf.reduce_mean(embeddings, axis=0)


def export():
    yamnet_model = hub.load(config.YAMNET_HANDLE)
    wrapped = YamnetEmbedding(yamnet_model)

    concrete_fn = wrapped.__call__.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn], wrapped)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,  # YAMNet uses a few ops outside core TFLite builtins
    ]
    tflite_model = converter.convert()

    with open(EMBEDDING_TFLITE_PATH, "wb") as f:
        f.write(tflite_model)
    print(f"Saved {EMBEDDING_TFLITE_PATH} ({len(tflite_model) / 1024 / 1024:.1f} MB)")
    return tflite_model


def verify(tflite_model, tolerance=1e-4):
    """Check the exported TFLite model's output matches
    yamnet_features.get_embedding() (what training actually used) on a
    handful of real clips, within floating-point tolerance."""
    import os
    import librosa

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    sample_dir = os.path.join(config.DATA_DIR, "hungry")
    sample_files = sorted(os.listdir(sample_dir))[:3]

    max_diff = 0.0
    for fname in sample_files:
        fp = os.path.join(sample_dir, fname)
        y, _ = librosa.load(fp, sr=config.SAMPLE_RATE, mono=True)
        if len(y) < config.N_SAMPLES:
            y = np.pad(y, (0, config.N_SAMPLES - len(y)))
        else:
            y = y[: config.N_SAMPLES]
        y = y.astype(np.float32)

        reference = yamnet_features.get_embedding(y)

        interpreter.set_tensor(input_details["index"], y)
        interpreter.invoke()
        tflite_output = interpreter.get_tensor(output_details["index"])

        diff = np.abs(reference - tflite_output).max()
        max_diff = max(max_diff, diff)
        print(f"  {fname}: max abs diff = {diff:.6f}")

    if max_diff < tolerance:
        print(f"OK: TFLite embedding matches yamnet_features.get_embedding() "
              f"within {tolerance} (max diff {max_diff:.6f})")
    else:
        print(f"WARNING: max diff {max_diff:.6f} exceeds tolerance {tolerance} "
              f"-- TFLite embedding may not match training-time embeddings")


def main():
    tflite_model = export()
    print("\nVerifying against yamnet_features.get_embedding()...")
    verify(tflite_model)


if __name__ == "__main__":
    main()
