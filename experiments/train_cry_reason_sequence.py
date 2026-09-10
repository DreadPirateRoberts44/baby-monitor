"""Sequence-aware cry-reason models: small Conv1D networks over frame-level
feature sequences, as a comparison against the scalar-feature logistic
regression in train_cry_reason.py. Scoped strictly to the second-stage
cry-reason classifier (belly pain/burping/cold_hot/discomfort/hungry/
tired) — does NOT touch the first-stage cry/laugh/silence/noise model.

Two variants, trained and evaluated on the same split:
  - prosody_seq: Conv1D over (126, 5) frame-level F0/voicing/RMS/centroid/
    ZCR sequence (sequence_features.extract_prosody_sequence). Directly
    targets rhythm/pitch-contour pattern without hand-designed bout
    segmentation.
  - yamnet_seq: Conv1D over (8, 1024) un-pooled YAMNet per-frame
    embeddings (sequence_features.extract_yamnet_sequence). Only 8
    timesteps for a 4s clip, so temporal resolution is coarse; included
    to check whether YAMNet's learned representation carries any residual
    cry-reason signal once given a chance to see structure, not just the
    mean-pooled summary the first-stage model uses.

Both use small architectures (Conv1D -> GlobalAvgPool -> Dense) sized for
on-device inference (few thousand to ~100K params, sub-ms latency), and
class_weight balancing + label smoothing to match the "prefer false
positives over false negatives" priority established for this project.
"""
import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, top_k_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

import config
from sequence_features import (
    extract_prosody_sequence, extract_yamnet_sequence,
    N_FRAMES, N_PROSODY_SEQ_FEATURES,
)

YAMNET_SEQ_FRAMES = 8  # empirically: config.DURATION=4.0s -> 8 YAMNet frames
EXPERIMENT_OUTPUT_DIR = os.path.join(config.OUTPUT_DIR, "experiments")


def list_cry_reason_files():
    filepaths, labels = [], []
    for idx, cls in enumerate(config.CRY_REASON_CLASSES):
        cls_dir = os.path.join(config.DATA_DIR, cls)
        cls_files = [f for f in os.listdir(cls_dir) if f.lower().endswith((".wav", ".ogg"))]
        print(f"  {cls}: {len(cls_files)} files")
        for fname in cls_files:
            filepaths.append(os.path.join(cls_dir, fname))
            labels.append(idx)
    return filepaths, np.array(labels)


def load_waveform(fp):
    y, _ = librosa.load(fp, sr=config.SAMPLE_RATE, mono=True)
    if len(y) < config.N_SAMPLES:
        y = np.pad(y, (0, config.N_SAMPLES - len(y)))
    else:
        y = y[: config.N_SAMPLES]
    return y.astype(np.float32)


def extract_sequences(filepaths, desc=""):
    prosody_seqs = np.zeros((len(filepaths), N_FRAMES, N_PROSODY_SEQ_FEATURES), dtype=np.float32)
    yamnet_seqs = np.zeros((len(filepaths), YAMNET_SEQ_FRAMES, config.EMBEDDING_DIM), dtype=np.float32)
    for i, fp in enumerate(filepaths):
        y = load_waveform(fp)
        prosody_seqs[i] = extract_prosody_sequence(y)
        yseq = extract_yamnet_sequence(y)
        n = min(len(yseq), YAMNET_SEQ_FRAMES)
        yamnet_seqs[i, :n] = yseq[:n]
        if desc and (i + 1) % 50 == 0:
            print(f"  [{desc}] {i + 1}/{len(filepaths)}")
    return prosody_seqs, yamnet_seqs


def build_conv1d_model(input_shape, num_classes, filters=(16, 32)):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv1D(filters[0], kernel_size=5, activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(filters[1], kernel_size=5, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


def normalize_sequences(train_seq, test_seq):
    """Per-feature z-score using train statistics only."""
    mean = train_seq.reshape(-1, train_seq.shape[-1]).mean(axis=0)
    std = train_seq.reshape(-1, train_seq.shape[-1]).std(axis=0) + 1e-6
    return (train_seq - mean) / std, (test_seq - mean) / std


def train_and_report(X_train, X_test, y_train, y_test, class_names, label, epochs=60):
    num_classes = len(class_names)
    class_weights = compute_class_weight("balanced", classes=np.arange(num_classes), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

    y_train_oh = keras.utils.to_categorical(y_train, num_classes)
    y_test_oh = keras.utils.to_categorical(y_test, num_classes)

    model = build_conv1d_model(X_train.shape[1:], num_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=config.LABEL_SMOOTHING),
        metrics=["accuracy"],
    )

    val_split = 0.15
    n_val = int(len(X_train) * val_split)
    rng = np.random.RandomState(config.SEED)
    idx = rng.permutation(len(X_train))
    val_idx, fit_idx = idx[:n_val], idx[n_val:]

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
    ]
    model.fit(
        X_train[fit_idx], y_train_oh[fit_idx],
        validation_data=(X_train[val_idx], y_train_oh[val_idx]),
        epochs=epochs, batch_size=32, class_weight=class_weight_dict,
        callbacks=callbacks, verbose=0,
    )

    probs = model.predict(X_test, verbose=0)
    y_pred = probs.argmax(axis=1)

    print(f"\n=== {label} ===")
    print(f"Params: {model.count_params()}")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
    top2 = top_k_accuracy_score(y_test, probs, k=2, labels=np.arange(num_classes))
    print(f"Top-2 accuracy: {top2:.4f}")

    cm = confusion_matrix(y_test, y_pred, labels=np.arange(num_classes))
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Cry-reason confusion matrix ({label})")
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    out_path = f"{EXPERIMENT_OUTPUT_DIR}/cry_reason_confusion_{label}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")

    return model


def main():
    os.makedirs(EXPERIMENT_OUTPUT_DIR, exist_ok=True)

    filepaths, labels = list_cry_reason_files()
    class_names = config.CRY_REASON_CLASSES

    train_fp, test_fp, y_train, y_test = train_test_split(
        filepaths, labels, test_size=config.TEST_SPLIT, stratify=labels,
        random_state=config.SEED,
    )

    print("Extracting sequences (train)...")
    prosody_seq_train, yamnet_seq_train = extract_sequences(train_fp, desc="train")
    print("Extracting sequences (test)...")
    prosody_seq_test, yamnet_seq_test = extract_sequences(test_fp, desc="test")

    prosody_seq_train_n, prosody_seq_test_n = normalize_sequences(prosody_seq_train, prosody_seq_test)
    yamnet_seq_train_n, yamnet_seq_test_n = normalize_sequences(yamnet_seq_train, yamnet_seq_test)

    train_and_report(prosody_seq_train_n, prosody_seq_test_n, y_train, y_test, class_names,
                      label="prosody_seq_conv1d")
    train_and_report(yamnet_seq_train_n, yamnet_seq_test_n, y_train, y_test, class_names,
                      label="yamnet_seq_conv1d")


if __name__ == "__main__":
    main()
