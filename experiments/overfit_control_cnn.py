"""Diagnostic-only, throwaway script -- NOT a candidate model.

train_cry_reason_cnn.py's regularized spectrogram CNN collapsed to
predicting "fussy" almost exclusively on two different fixes (class
weighting, then full class-balanced oversampling + explicit sample
weights) -- see experiments/README.md. This isolates whether that's
because the spectrograms genuinely carry no separable signal for
hungry-vs-fussy, or because the regularization (Dropout/L2/SpecAugment)
was too aggressive for the model to learn anything nuanced at all.

Strips every regularizer (no Dropout, no L2, no SpecAugment, no class
weighting) and fits on the raw un-oversampled training set, checking the
TRAINING-set confusion matrix (not held-out test/val) after enough epochs
to memorize if it's going to. If the model still can't separate
hungry/belly-pain/fussy on the data it's directly fitting, that's strong
evidence the spectrograms don't carry the signal -- not a regularization
artifact. If it CAN, the regularized version was too conservative.
"""
import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import config
from spectrogram_features import extract_log_mel_spectrogram, N_MELS, N_TIME_FRAMES


def list_cry_reason_files():
    class_names = config.CRY_REASON_CLASSES
    class_index = {cls: idx for idx, cls in enumerate(class_names)}
    filepaths, labels = [], []
    for cls in class_names:
        cls_files = []
        for folder in config.CRY_REASON_GROUPS[cls]:
            folder_dir = os.path.join(config.DATA_DIR, folder)
            folder_files = [
                os.path.join(folder_dir, f) for f in os.listdir(folder_dir)
                if f.lower().endswith((".wav", ".ogg"))
            ]
            cls_files.extend(folder_files)
        filepaths.extend(cls_files)
        labels.extend([class_index[cls]] * len(cls_files))
    return filepaths, np.array(labels)


def load_waveform(fp):
    y, _ = librosa.load(fp, sr=config.SAMPLE_RATE, mono=True)
    if len(y) < config.N_SAMPLES:
        y = np.pad(y, (0, config.N_SAMPLES - len(y)))
    else:
        y = y[: config.N_SAMPLES]
    return y.astype(np.float32)


def extract_spectrograms(filepaths, desc=""):
    specs = np.zeros((len(filepaths), N_MELS, N_TIME_FRAMES), dtype=np.float32)
    for i, fp in enumerate(filepaths):
        specs[i] = extract_log_mel_spectrogram(load_waveform(fp))
        if desc and (i + 1) % 100 == 0:
            print(f"  [{desc}] {i + 1}/{len(filepaths)}")
    return specs[..., np.newaxis]


def build_unregularized_cnn(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(16, (3, 3), padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


def main():
    filepaths, labels = list_cry_reason_files()
    class_names = config.CRY_REASON_CLASSES
    num_classes = len(class_names)

    # No oversampling, no held-out test set needed -- we're only checking
    # whether the model can fit the raw training data at all.
    train_fp, _, y_train, _ = train_test_split(
        filepaths, labels, test_size=0.15, stratify=labels, random_state=config.SEED,
    )

    print(f"Training on {len(train_fp)} raw (non-oversampled) clips, no regularization...")
    X_train = extract_spectrograms(train_fp, desc="train")
    y_train_oh = keras.utils.to_categorical(y_train, num_classes)

    model = build_unregularized_cnn((N_MELS, N_TIME_FRAMES, 1), num_classes)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train_oh, epochs=60, batch_size=32, verbose=2)

    probs = model.predict(X_train, verbose=0)
    y_pred = probs.argmax(axis=1)

    print("\n=== TRAINING-SET fit quality (overfitting control, no held-out eval) ===")
    print(classification_report(y_train, y_pred, target_names=class_names, zero_division=0))
    print(confusion_matrix(y_train, y_pred))


if __name__ == "__main__":
    main()
