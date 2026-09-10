"""Second-stage cry-reason model, spectrogram-CNN variant.

Every prior stage-2 attempt used either a general-purpose embedding
(YAMNet, mean-pooled or as a short sequence) or hand-engineered scalar
features (prosody, rhythm, voice-quality) feeding a linear/MLP/small-Conv1D
head -- see config.py's CRY_REASON_GROUPS comment and experiments/README.md
for the full history. Best result: YAMNet+MLP at 46% accuracy / 84.6%
top-2 on the 3-class (belly pain/hungry/fussy) scheme.

This tries the highest-ceiling option from that comparison: a small 2D CNN
trained directly on log-mel spectrograms (spectrogram_features.py), so the
model learns its own time-frequency features instead of relying on
hand-engineered summaries or an embedding never trained for cry-reason
discrimination. This is the standard approach in published infant-cry
classification work, including small-dataset results (~1-2K clips) that
report high accuracy -- motivating trying it here even though the dataset
is modest (~840 training clips for 3 classes after the split).

Overfitting is the central risk with this approach on a dataset this size
(a spectrogram CNN has more capacity and less inductive bias toward this
task than a head on a pretrained embedding), so this script leans hard on
regularization before trusting any result:
  - SpecAugment-style time/frequency masking (data augmentation directly
    on the spectrogram, not just the waveform)
  - waveform-level augmentation (gain/noise/shift, reusing data.py's
    augment_waveform) BEFORE spectrogram extraction, for oversampled
    duplicates
  - BatchNorm + Dropout + L2 weight decay throughout
  - a deliberately small architecture (few conv layers, few filters) sized
    for a ~1K-clip dataset, not a research-scale one
  - early stopping on validation accuracy, and a train/val/test split
    with untouched test-set reporting

Scoped to the 3-class scheme only (belly pain/hungry/fussy) per project
decision -- the 6-class scheme's ceiling is well established at this point
(see experiments/) and not worth re-testing with every new architecture.
"""
import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, top_k_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

import config
from data import augment_waveform
from spectrogram_features import extract_log_mel_spectrogram, N_MELS, N_TIME_FRAMES

EXPERIMENT_OUTPUT_DIR = os.path.join(config.OUTPUT_DIR, "experiments")
CNN_MODEL_PATH = os.path.join(EXPERIMENT_OUTPUT_DIR, "cry_reason_cnn.keras")
CNN_CONFUSION_PATH = os.path.join(EXPERIMENT_OUTPUT_DIR, "cry_reason_confusion_cnn.png")

# Oversample minority classes up to this multiple of their original count.
# First attempt at 2.0x still let the model collapse to predicting the
# majority class "fussy" almost exclusively (see experiments/README.md and
# the git history of this file for that result) even though hungry (382
# files) was already at full parity with fussy (483) under that cap --
# belly pain (124 files) was the only class still under-represented, at
# 248 vs 483. Raised to fully equalize all three classes' raw training
# counts, on the hypothesis that loss-reweighting (class_weight="balanced")
# alone wasn't enough to overcome the batch-composition imbalance a
# spectrogram CNN sees during SGD.
MAX_OVERSAMPLE_RATIO = 4.0


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
        print(f"  {cls}: {len(cls_files)} files")
        filepaths.extend(cls_files)
        labels.extend([class_index[cls]] * len(cls_files))
    return filepaths, np.array(labels)


def oversample_capped(filepaths, labels, max_ratio):
    """Like data.py's oversample(), but caps duplication at max_ratio x the
    class's own original count instead of matching the majority class
    exactly -- keeps a CNN (higher capacity than the MLP head) from seeing
    too many near-identical duplicates of the smallest class."""
    filepaths = np.array(filepaths)
    labels = np.array(labels)
    counts = np.bincount(labels)
    target_counts = np.minimum(counts.max(), (counts * max_ratio).astype(int))

    out_fp, out_lbl, out_aug = [], [], []
    for cls, (count, target) in enumerate(zip(counts, target_counts)):
        cls_fp = filepaths[labels == cls]
        out_fp.extend(cls_fp)
        out_lbl.extend([cls] * count)
        out_aug.extend([False] * count)

        if count < target:
            extra_idx = np.random.RandomState(config.SEED).choice(len(cls_fp), target - count, replace=True)
            out_fp.extend(cls_fp[extra_idx])
            out_lbl.extend([cls] * (target - count))
            out_aug.extend([True] * (target - count))

    return list(out_fp), np.array(out_lbl), out_aug


def load_waveform(fp):
    y, _ = librosa.load(fp, sr=config.SAMPLE_RATE, mono=True)
    if len(y) < config.N_SAMPLES:
        y = np.pad(y, (0, config.N_SAMPLES - len(y)))
    else:
        y = y[: config.N_SAMPLES]
    return y.astype(np.float32)


def extract_spectrograms(filepaths, augment_flags=None, desc=""):
    if augment_flags is None:
        augment_flags = [False] * len(filepaths)
    specs = np.zeros((len(filepaths), N_MELS, N_TIME_FRAMES), dtype=np.float32)
    for i, (fp, aug) in enumerate(zip(filepaths, augment_flags)):
        y = load_waveform(fp)
        if aug:
            y = augment_waveform(y)
        specs[i] = extract_log_mel_spectrogram(y)
        if desc and (i + 1) % 100 == 0:
            print(f"  [{desc}] {i + 1}/{len(filepaths)}")
    return specs[..., np.newaxis]  # add channel dim: (N, mels, time, 1)


def spec_augment(spec, freq_mask_pct=0.15, time_mask_pct=0.15, n_masks=2):
    """SpecAugment-style masking applied in-place on a copy of a single
    (mels, time, 1) spectrogram. Standard augmentation for spectrogram CNNs
    on small datasets -- randomly zeroes out frequency and time bands so
    the model can't overfit to exact positions of features."""
    spec = spec.copy()
    n_mels, n_time = spec.shape[0], spec.shape[1]
    rng = np.random
    for _ in range(n_masks):
        f_width = int(n_mels * freq_mask_pct * rng.uniform(0.3, 1.0))
        if f_width > 0 and n_mels - f_width > 0:
            f0 = rng.randint(0, n_mels - f_width)
            spec[f0:f0 + f_width, :, :] = -1.0  # -1.0 == silence floor after normalization

        t_width = int(n_time * time_mask_pct * rng.uniform(0.3, 1.0))
        if t_width > 0 and n_time - t_width > 0:
            t0 = rng.randint(0, n_time - t_width)
            spec[:, t0:t0 + t_width, :] = -1.0
    return spec


def make_train_dataset(specs, labels_oh, sample_weights, batch_size, augment=True):
    """Yields (x, y, sample_weight) triples rather than relying on
    model.fit(class_weight=...), whose interaction with a
    tf.data.Dataset.from_generator pipeline is not reliably verifiable in
    Keras 3 (no error is raised either way, so silent no-ops are a real
    risk) -- explicit per-example sample_weight in the dataset itself is
    unambiguous."""
    def generator():
        n = len(specs)
        while True:
            idx = np.random.permutation(n)
            for i in idx:
                x = spec_augment(specs[i]) if augment else specs[i]
                yield x, labels_oh[i], sample_weights[i]

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(N_MELS, N_TIME_FRAMES, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(labels_oh.shape[1],), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
        ),
    )
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_spectrogram_cnn(input_shape, num_classes, l2_weight=1e-3):
    reg = regularizers.l2(l2_weight)
    inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(16, (3, 3), padding="same", kernel_regularizer=reg)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(32, activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs, name="cry_reason_cnn")


def main():
    os.makedirs(EXPERIMENT_OUTPUT_DIR, exist_ok=True)

    filepaths, labels = list_cry_reason_files()
    class_names = config.CRY_REASON_CLASSES
    num_classes = len(class_names)

    train_fp, test_fp, y_train, y_test = train_test_split(
        filepaths, labels, test_size=config.TEST_SPLIT, stratify=labels,
        random_state=config.SEED,
    )
    train_fp, val_fp, y_train, y_val = train_test_split(
        train_fp, y_train, test_size=config.VAL_SPLIT / (1 - config.TEST_SPLIT),
        stratify=y_train, random_state=config.SEED,
    )

    print(f"Train: {len(train_fp)}, Val: {len(val_fp)}, Test: {len(test_fp)}")

    train_fp, y_train, train_aug_flags = oversample_capped(train_fp, y_train, MAX_OVERSAMPLE_RATIO)
    print(f"Train after capped oversampling (max {MAX_OVERSAMPLE_RATIO}x): {len(train_fp)}")

    print("Extracting spectrograms (train)...")
    X_train = extract_spectrograms(train_fp, train_aug_flags, desc="train")
    print("Extracting spectrograms (val)...")
    X_val = extract_spectrograms(val_fp, desc="val")
    print("Extracting spectrograms (test)...")
    X_test = extract_spectrograms(test_fp, desc="test")

    class_weights = compute_class_weight("balanced", classes=np.arange(num_classes), y=y_train)
    print(f"\nClass weights (balanced): "
          f"{dict(zip(class_names, [round(w, 2) for w in class_weights]))}")
    train_sample_weights = class_weights[y_train].astype(np.float32)

    y_train_oh = keras.utils.to_categorical(y_train, num_classes)
    y_val_oh = keras.utils.to_categorical(y_val, num_classes)

    train_ds = make_train_dataset(X_train, y_train_oh, train_sample_weights, config.BATCH_SIZE, augment=True)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val_oh)).batch(config.BATCH_SIZE)

    model = build_spectrogram_cnn((N_MELS, N_TIME_FRAMES, 1), num_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(config.LEARNING_RATE),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=config.LABEL_SMOOTHING),
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(CNN_MODEL_PATH, monitor="val_accuracy", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=15, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6),
    ]

    steps_per_epoch = max(1, len(X_train) // config.BATCH_SIZE)
    model.fit(
        train_ds,
        validation_data=val_ds,
        steps_per_epoch=steps_per_epoch,
        epochs=config.EPOCHS * 2,  # augmented generator + small dataset -> allow more epochs
        callbacks=callbacks,
    )

    probs = model.predict(X_test, verbose=0)
    y_pred = probs.argmax(axis=1)

    print(f"\n=== cry_reason_cnn (belly pain / hungry / fussy, spectrogram CNN) ===")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
    top2 = top_k_accuracy_score(y_test, probs, k=2, labels=np.arange(num_classes))
    print(f"Top-2 accuracy: {top2:.4f}")

    # Overfitting check: compare train-set (non-augmented) accuracy to test
    # accuracy. A large gap is the signature of overfitting on a dataset
    # this size, which is the main risk this architecture was built to
    # manage but can't rule out without checking.
    train_probs = model.predict(extract_spectrograms(train_fp, desc="train_eval"), verbose=0)
    train_acc = (train_probs.argmax(axis=1) == y_train).mean()
    test_acc = (y_pred == y_test).mean()
    print(f"\nTrain accuracy (no augmentation at eval time): {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Train-test gap: {train_acc - test_acc:.4f} "
          f"({'large -- likely overfitting' if train_acc - test_acc > 0.25 else 'moderate/small'})")

    cm = confusion_matrix(y_test, y_pred, labels=np.arange(num_classes))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Cry-reason confusion matrix (spectrogram CNN)")
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(CNN_CONFUSION_PATH, dpi=150)
    print(f"Saved {CNN_CONFUSION_PATH}")
    print(f"Saved model to {CNN_MODEL_PATH}")


if __name__ == "__main__":
    main()
