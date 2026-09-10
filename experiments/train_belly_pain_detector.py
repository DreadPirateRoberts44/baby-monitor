"""Focused binary detector: belly pain vs. everything else (hungry + fussy
merged into "not belly pain"), given a clip already identified as a cry.

Every experiment run on the 3-way cry-reason split (belly pain/hungry/
fussy) and even a from-scratch spectrogram CNN diagnostic (see
train_cry_reason_cnn.py and experiments/overfit_control_cnn.py) found the
same pattern: belly pain has real, learnable acoustic signal (34-58%
recall, often 70-95% precision, across nearly every feature/model
combination tried), while hungry and fussy are not reliably separable from
each other even by an unregularized model fit directly on its own training
data. Rather than keep forcing a 3-way split where 2 of 3 classes lack
separable signal, this isolates the one signal that IS there: a focused
binary belly-pain detector.

Uses the same YAMNet-embedding + small-MLP architecture that has
consistently outperformed every other feature/model combination tried in
this project (scalar prosody, rhythm, voice-quality, sequence models,
gradient boosting, spectrogram CNNs).

Output is meant to complement, not replace, train_cry_reason.py's 3-way
model: a caregiver-facing "possible belly pain" flag with its own
(hopefully higher) precision/recall, while hungry-vs-fussy stays a lower-
confidence suggestion from the 3-way model.
"""
import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

import config
import yamnet_features

EXPERIMENT_OUTPUT_DIR = os.path.join(config.OUTPUT_DIR, "experiments")
BELLY_PAIN_MODEL_PATH = os.path.join(EXPERIMENT_OUTPUT_DIR, "belly_pain_detector.keras")
BELLY_PAIN_CONFUSION_PATH = os.path.join(EXPERIMENT_OUTPUT_DIR, "belly_pain_confusion_matrix.png")


def list_belly_pain_files():
    """belly pain (1) vs. everything else in CRY_REASON_GROUPS (0) --
    i.e. hungry + fussy's underlying folders, merged."""
    filepaths, labels = [], []
    for cls in config.CRY_REASON_CLASSES:
        is_belly_pain = cls == "belly pain"
        for folder in config.CRY_REASON_GROUPS[cls]:
            folder_dir = os.path.join(config.DATA_DIR, folder)
            folder_files = [
                os.path.join(folder_dir, f) for f in os.listdir(folder_dir)
                if f.lower().endswith((".wav", ".ogg"))
            ]
            print(f"  {folder} ({'belly_pain' if is_belly_pain else 'not_belly_pain'}): {len(folder_files)} files")
            filepaths.extend(folder_files)
            labels.extend([int(is_belly_pain)] * len(folder_files))
    return filepaths, np.array(labels)


def load_waveform(fp):
    y, _ = librosa.load(fp, sr=config.SAMPLE_RATE, mono=True)
    if len(y) < config.N_SAMPLES:
        y = np.pad(y, (0, config.N_SAMPLES - len(y)))
    else:
        y = y[: config.N_SAMPLES]
    return y.astype(np.float32)


def extract_embeddings(filepaths, desc=""):
    feats = np.zeros((len(filepaths), config.EMBEDDING_DIM), dtype=np.float32)
    for i, fp in enumerate(filepaths):
        feats[i] = yamnet_features.get_embedding(load_waveform(fp))
        if desc and (i + 1) % 50 == 0:
            print(f"  [{desc}] {i + 1}/{len(filepaths)}")
    return feats


def build_binary_head(embedding_dim):
    inputs = keras.Input(shape=(embedding_dim,))
    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(2, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="belly_pain_detector")


def main():
    os.makedirs(EXPERIMENT_OUTPUT_DIR, exist_ok=True)

    filepaths, labels = list_belly_pain_files()
    class_names = ["not_belly_pain", "belly_pain"]

    train_fp, test_fp, y_train, y_test = train_test_split(
        filepaths, labels, test_size=config.TEST_SPLIT, stratify=labels,
        random_state=config.SEED,
    )

    print("Extracting YAMNet embeddings (train)...")
    X_train = extract_embeddings(train_fp, desc="train")
    print("Extracting YAMNet embeddings (test)...")
    X_test = extract_embeddings(test_fp, desc="test")

    class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    print(f"\nClass weights (balanced): {dict(zip(class_names, [round(w, 2) for w in class_weights]))}")

    y_train_oh = keras.utils.to_categorical(y_train, 2)

    model = build_binary_head(config.EMBEDDING_DIM)
    model.compile(
        optimizer=keras.optimizers.Adam(config.LEARNING_RATE),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=config.LABEL_SMOOTHING),
        metrics=["accuracy"],
    )
    model.summary()

    n_val = int(len(X_train) * config.VAL_SPLIT)
    rng = np.random.RandomState(config.SEED)
    idx = rng.permutation(len(X_train))
    val_idx, fit_idx = idx[:n_val], idx[n_val:]

    callbacks = [
        keras.callbacks.ModelCheckpoint(BELLY_PAIN_MODEL_PATH, monitor="val_accuracy", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
    ]
    model.fit(
        X_train[fit_idx], y_train_oh[fit_idx],
        validation_data=(X_train[val_idx], y_train_oh[val_idx]),
        epochs=config.EPOCHS, batch_size=config.BATCH_SIZE,
        class_weight=class_weight_dict, callbacks=callbacks,
    )

    probs = model.predict(X_test, verbose=0)
    y_pred = probs.argmax(axis=1)

    print(f"\n=== belly_pain_detector (belly pain vs. not) ===")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
    auc = roc_auc_score(y_test, probs[:, 1])
    print(f"ROC AUC: {auc:.4f}")

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Belly pain detector confusion matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(BELLY_PAIN_CONFUSION_PATH, dpi=150)
    print(f"Saved {BELLY_PAIN_CONFUSION_PATH}")
    print(f"Saved model to {BELLY_PAIN_MODEL_PATH}")


if __name__ == "__main__":
    main()
