"""Two-stage inference on a captured audio window, using only TFLite
models -- no full TensorFlow, tensorflow_hub, or librosa dependency, so
this runs light on the Pi. Mirrors predict.py's logic at the repo root
(which uses the full Keras models for convenience during development) but
against the exported .tflite artifacts instead.

IMPORTANT: stage 1 (train.py/data.py) and stage 2 (train_cry_reason.py)
were trained on DIFFERENT preprocessing. Stage 1 was trained on the raw
waveform's embedding, with no high-pass filter. Stage 2 was trained on the
high-pass-filtered (300Hz) waveform's embedding -- that filter was found to
meaningfully improve stage-2 accuracy (see README.md "Preprocessing") but
was never applied/evaluated for stage 1. Feeding stage 1 a filtered
embedding would be feeding it a distribution it never saw in training, so
this module computes TWO embeddings per window (one from the raw waveform
for stage 1, one from the filtered waveform for stage 2) rather than
sharing one -- slightly more YAMNet inference cost, but avoids a train/
serve skew bug. If stage 1 is ever retrained with the same filter, this
should be revisited to share a single embedding again.

Pipeline per window:
  1. YAMNet embedding of the raw waveform via yamnet_embedding.tflite
     (1024-dim, mean-pooled over the clip -- verified in
     export_yamnet_embedding_tflite.py to numerically match
     yamnet_features.get_embedding(), what stage-1 training actually used).
  2. Stage 1 (classifier_head.tflite) on that raw-waveform embedding: cry /
     laugh / silence / noise. Below settings.STAGE1_CONFIDENCE_THRESHOLD,
     treated as "uncertain".
  3. If stage 1 says "cry": high-pass filter the waveform at 300Hz
     (pi/highpass.py, matching audio_preprocessing.highpass_filter), embed
     THAT with YAMNet again, and run stage 2 (cry_reason_mlp.tflite) on the
     filtered embedding: belly pain / hungry / fussy, returned as a full
     ranked probability list (not just argmax) -- stage 2 is a
     low-confidence suggestion tool, not a verdict, per the project's
     established framing.

Uses tflite_runtime if available (the lightweight package meant for
exactly this kind of deployment); falls back to tensorflow.lite.Interpreter
if only full TensorFlow is installed, so this also runs on a dev machine
for testing without requiring tflite_runtime specifically.
"""
import os

import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf  # dev-machine fallback
    Interpreter = tf.lite.Interpreter

import settings
from highpass import highpass_filter


class TFLiteModel:
    """Thin wrapper around a TFLite interpreter for a single-input,
    single-output float32 model -- covers all three models used here."""

    def __init__(self, model_path):
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self._input = self.interpreter.get_input_details()[0]
        self._output = self.interpreter.get_output_details()[0]

    def run(self, x):
        x = np.asarray(x, dtype=self._input["dtype"])
        self.interpreter.set_tensor(self._input["index"], x)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self._output["index"])


def load_labels(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


class CryPredictor:
    """Loads all three models once; call predict(waveform) per captured
    window. waveform must be a 1-D float32 array of exactly
    settings.N_SAMPLES samples at settings.SAMPLE_RATE."""

    def __init__(self, model_dir=None):
        model_dir = model_dir or settings.MODEL_DIR
        for name, path in [
            ("yamnet_embedding.tflite", settings.YAMNET_EMBEDDING_TFLITE_PATH),
            ("classifier_head.tflite", settings.STAGE1_TFLITE_PATH),
            ("cry_reason_mlp.tflite", settings.STAGE2_TFLITE_PATH),
            ("labels.txt", settings.STAGE1_LABELS_PATH),
            ("cry_reason_labels.txt", settings.STAGE2_LABELS_PATH),
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Missing {name} at {path}. Copy the exported models from "
                    f"output/ into {model_dir} -- see pi/README.md."
                )

        self.yamnet = TFLiteModel(settings.YAMNET_EMBEDDING_TFLITE_PATH)
        self.stage1_model = TFLiteModel(settings.STAGE1_TFLITE_PATH)
        self.stage2_model = TFLiteModel(settings.STAGE2_TFLITE_PATH)
        self.stage1_labels = load_labels(settings.STAGE1_LABELS_PATH)
        self.stage2_labels = load_labels(settings.STAGE2_LABELS_PATH)

    def _check_shape(self, waveform):
        if waveform.shape != (settings.N_SAMPLES,):
            raise ValueError(
                f"Expected waveform shape ({settings.N_SAMPLES},), got {waveform.shape}"
            )

    def embed_raw(self, waveform):
        """Embedding of the unfiltered waveform -- what stage 1 expects."""
        self._check_shape(waveform)
        embedding = self.yamnet.run(waveform.astype(np.float32))
        return np.asarray(embedding, dtype=np.float32).reshape(-1)

    def embed_filtered(self, waveform):
        """Embedding of the 300Hz-high-pass-filtered waveform -- what
        stage 2 expects (see module docstring)."""
        self._check_shape(waveform)
        filtered = highpass_filter(waveform, settings.SAMPLE_RATE)
        embedding = self.yamnet.run(filtered)
        return np.asarray(embedding, dtype=np.float32).reshape(-1)

    def predict_stage1(self, embedding):
        """Returns (label, confidence, probs_dict). label is
        settings.UNCERTAIN_LABEL if confidence < STAGE1_CONFIDENCE_THRESHOLD."""
        probs = self.stage1_model.run(embedding[None, :])[0]
        top_idx = int(np.argmax(probs))
        confidence = float(probs[top_idx])
        label = self.stage1_labels[top_idx]
        if confidence < settings.STAGE1_CONFIDENCE_THRESHOLD:
            label = settings.UNCERTAIN_LABEL
        probs_dict = dict(zip(self.stage1_labels, (float(p) for p in probs)))
        return label, confidence, probs_dict

    def predict_stage2(self, embedding):
        """Returns a dict of {label: probability}, sorted descending. Only
        meaningful when stage 1 said "cry" -- see predict() below. Returned
        as a full ranked distribution, not a single label: stage 2 is a
        low-confidence suggestion, and callers should decide how many of
        the top results to surface (project preference is top-2)."""
        probs = self.stage2_model.run(embedding[None, :])[0]
        probs_dict = dict(zip(self.stage2_labels, (float(p) for p in probs)))
        return dict(sorted(probs_dict.items(), key=lambda kv: kv[1], reverse=True))

    def predict(self, waveform):
        """Full pipeline for one captured window. Returns a result dict:
          {
            "stage1_label": str,        # class name, or settings.UNCERTAIN_LABEL
            "stage1_confidence": float,
            "stage1_probs": {label: prob, ...},
            "stage2_probs": {label: prob, ...} or None,  # None unless stage1_label == "cry"
          }
        """
        raw_embedding = self.embed_raw(waveform)
        stage1_label, stage1_confidence, stage1_probs = self.predict_stage1(raw_embedding)

        stage2_probs = None
        if stage1_label == settings.CRY_LABEL:
            filtered_embedding = self.embed_filtered(waveform)
            stage2_probs = self.predict_stage2(filtered_embedding)

        return {
            "stage1_label": stage1_label,
            "stage1_confidence": stage1_confidence,
            "stage1_probs": stage1_probs,
            "stage2_probs": stage2_probs,
        }
