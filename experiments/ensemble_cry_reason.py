"""Ensemble comparison for the second-stage cry-reason classifier.

Trains all four previously-explored approaches on an IDENTICAL stratified
train/test split, then averages their predict_proba outputs to check
whether combining models that each pick up different partial signal beats
any single model. This is purely a comparison/decision tool, not itself
the shipped model -- see train_cry_reason.py's docstring for why none of
the individual approaches cleared a usable accuracy bar on their own.

Models combined:
  A. YAMNet (mean-pooled embedding) + MLP softmax head
  B. Scalar prosody + rhythm features + logistic regression (CV-tuned C)
  C. Frame-level prosody sequence + Conv1D
  D. YAMNet per-frame (un-pooled) sequence + Conv1D
"""
import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, top_k_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

import config
import yamnet_features
from prosody_features import extract_prosody_features, FEATURE_NAMES
from cry_rhythm_features import extract_rhythm_features, RHYTHM_FEATURE_NAMES
from sequence_features import (
    extract_prosody_sequence, extract_yamnet_sequence,
    N_FRAMES, N_PROSODY_SEQ_FEATURES,
)
from experiments.train_cry_reason_sequence import build_conv1d_model, normalize_sequences, YAMNET_SEQ_FRAMES

EXPERIMENT_OUTPUT_DIR = os.path.join(config.OUTPUT_DIR, "experiments")


def list_cry_reason_files():
    filepaths, labels = [], []
    for idx, cls in enumerate(config.CRY_REASON_CLASSES):
        cls_dir = os.path.join(config.DATA_DIR, cls)
        cls_files = [f for f in os.listdir(cls_dir) if f.lower().endswith((".wav", ".ogg"))]
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


def extract_everything(filepaths, desc=""):
    n = len(filepaths)
    yamnet_pooled = np.zeros((n, config.EMBEDDING_DIM), dtype=np.float32)
    prosody_scalar = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32)
    rhythm_scalar = np.zeros((n, len(RHYTHM_FEATURE_NAMES)), dtype=np.float32)
    prosody_seq = np.zeros((n, N_FRAMES, N_PROSODY_SEQ_FEATURES), dtype=np.float32)
    yamnet_seq = np.zeros((n, YAMNET_SEQ_FRAMES, config.EMBEDDING_DIM), dtype=np.float32)

    for i, fp in enumerate(filepaths):
        y = load_waveform(fp)
        yamnet_pooled[i] = yamnet_features.get_embedding(y)
        prosody_scalar[i] = extract_prosody_features(y)
        rhythm_scalar[i] = extract_rhythm_features(y)
        prosody_seq[i] = extract_prosody_sequence(y)
        yseq = extract_yamnet_sequence(y)
        m = min(len(yseq), YAMNET_SEQ_FRAMES)
        yamnet_seq[i, :m] = yseq[:m]
        if desc and (i + 1) % 50 == 0:
            print(f"  [{desc}] {i + 1}/{n}")

    return {
        "yamnet_pooled": yamnet_pooled,
        "prosody_scalar": prosody_scalar,
        "rhythm_scalar": rhythm_scalar,
        "prosody_seq": prosody_seq,
        "yamnet_seq": yamnet_seq,
    }


def train_mlp_head(X_train, X_test, y_train, y_test, num_classes):
    class_weights = compute_class_weight("balanced", classes=np.arange(num_classes), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    y_train_oh = keras.utils.to_categorical(y_train, num_classes)

    inputs = keras.Input(shape=(config.EMBEDDING_DIM,))
    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=config.LABEL_SMOOTHING),
        metrics=["accuracy"],
    )

    n_val = int(len(X_train) * 0.15)
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
        epochs=60, batch_size=32, class_weight=class_weight_dict,
        callbacks=callbacks, verbose=0,
    )
    return model.predict(X_test, verbose=0)


def train_logreg(X_train, X_test, y_train, y_test):
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=config.SEED)),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    search = GridSearchCV(pipe, {"clf__C": [0.001, 0.01, 0.1, 1.0, 10.0]}, cv=cv,
                           scoring="balanced_accuracy", n_jobs=-1)
    search.fit(X_train, y_train)
    return search.best_estimator_.predict_proba(X_test)


def train_conv1d(X_train, X_test, y_train, y_test, num_classes):
    class_weights = compute_class_weight("balanced", classes=np.arange(num_classes), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    y_train_oh = keras.utils.to_categorical(y_train, num_classes)

    model = build_conv1d_model(X_train.shape[1:], num_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=config.LABEL_SMOOTHING),
        metrics=["accuracy"],
    )
    n_val = int(len(X_train) * 0.15)
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
        epochs=60, batch_size=32, class_weight=class_weight_dict,
        callbacks=callbacks, verbose=0,
    )
    return model.predict(X_test, verbose=0)


def report(y_test, probs, class_names, label):
    num_classes = len(class_names)
    y_pred = probs.argmax(axis=1)
    print(f"\n=== {label} ===")
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


def main():
    os.makedirs(EXPERIMENT_OUTPUT_DIR, exist_ok=True)
    class_names = config.CRY_REASON_CLASSES
    num_classes = len(class_names)

    filepaths, labels = list_cry_reason_files()
    train_fp, test_fp, y_train, y_test = train_test_split(
        filepaths, labels, test_size=config.TEST_SPLIT, stratify=labels,
        random_state=config.SEED,
    )

    print("Extracting all feature types (train)...")
    train_feats = extract_everything(train_fp, desc="train")
    print("Extracting all feature types (test)...")
    test_feats = extract_everything(test_fp, desc="test")

    scalar_train = np.concatenate([train_feats["prosody_scalar"], train_feats["rhythm_scalar"]], axis=1)
    scalar_test = np.concatenate([test_feats["prosody_scalar"], test_feats["rhythm_scalar"]], axis=1)

    prosody_seq_train_n, prosody_seq_test_n = normalize_sequences(
        train_feats["prosody_seq"], test_feats["prosody_seq"])
    yamnet_seq_train_n, yamnet_seq_test_n = normalize_sequences(
        train_feats["yamnet_seq"], test_feats["yamnet_seq"])

    print("\nTraining model A: YAMNet-pooled + MLP...")
    probs_a = train_mlp_head(train_feats["yamnet_pooled"], test_feats["yamnet_pooled"],
                              y_train, y_test, num_classes)
    report(y_test, probs_a, class_names, "A_yamnet_mlp")

    print("\nTraining model B: scalar prosody+rhythm + logreg...")
    probs_b = train_logreg(scalar_train, scalar_test, y_train, y_test)
    report(y_test, probs_b, class_names, "B_scalar_logreg")

    print("\nTraining model C: prosody sequence + Conv1D...")
    probs_c = train_conv1d(prosody_seq_train_n, prosody_seq_test_n, y_train, y_test, num_classes)
    report(y_test, probs_c, class_names, "C_prosody_seq_conv1d")

    print("\nTraining model D: YAMNet sequence + Conv1D...")
    probs_d = train_conv1d(yamnet_seq_train_n, yamnet_seq_test_n, y_train, y_test, num_classes)
    report(y_test, probs_d, class_names, "D_yamnet_seq_conv1d")

    ensemble_probs = np.mean([probs_a, probs_b, probs_c, probs_d], axis=0)
    report(y_test, ensemble_probs, class_names, "ENSEMBLE_average")


if __name__ == "__main__":
    main()
