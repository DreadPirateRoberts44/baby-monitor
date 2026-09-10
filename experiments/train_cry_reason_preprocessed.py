"""Second-stage cry-reason model (belly pain / hungry / fussy), trained on
preprocessed audio: a 300Hz high-pass filter, then extraction of a ~1s
window centered on the clip's most prominent cry bout (see
audio_preprocessing.py), rather than the raw full 4s clip used by
train_cry_reason.py.

Same architecture as the current best model (train_cry_reason.py: YAMNet
mean-pooled embedding -> small MLP, class-weight balancing, label
smoothing) so this is a clean, isolated A/B comparison of preprocessing
alone -- not a different model. If this beats train_cry_reason.py's 46%
accuracy / 84.6% top-2 result, promote this preprocessing into the shipped
pipeline; if not, leave train_cry_reason.py as-is.
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
import yamnet_features
from audio_preprocessing import preprocess_waveform

EXPERIMENT_OUTPUT_DIR = os.path.join(config.OUTPUT_DIR, "experiments")



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


def load_and_preprocess(fp, apply_highpass=True, apply_windowing=True):
    y, _ = librosa.load(fp, sr=config.SAMPLE_RATE, mono=True)
    if len(y) < config.N_SAMPLES:
        y = np.pad(y, (0, config.N_SAMPLES - len(y)))
    else:
        y = y[: config.N_SAMPLES]
    return preprocess_waveform(y.astype(np.float32), apply_highpass=apply_highpass, apply_windowing=apply_windowing)


def extract_embeddings(filepaths, desc="", apply_highpass=True, apply_windowing=True):
    feats = np.zeros((len(filepaths), config.EMBEDDING_DIM), dtype=np.float32)
    for i, fp in enumerate(filepaths):
        feats[i] = yamnet_features.get_embedding(
            load_and_preprocess(fp, apply_highpass=apply_highpass, apply_windowing=apply_windowing)
        )
        if desc and (i + 1) % 50 == 0:
            print(f"  [{desc}] {i + 1}/{len(filepaths)}")
    return feats


def build_mlp_head(embedding_dim, num_classes):
    inputs = keras.Input(shape=(embedding_dim,))
    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="cry_reason_head_preprocessed")


def main(apply_highpass=True, apply_windowing=True):
    os.makedirs(EXPERIMENT_OUTPUT_DIR, exist_ok=True)
    mode = ("highpass" if apply_highpass else "") + ("+window" if apply_windowing else "")
    print(f"Preprocessing mode: {mode or 'none'}")

    filepaths, labels = list_cry_reason_files()
    class_names = config.CRY_REASON_CLASSES
    num_classes = len(class_names)

    train_fp, test_fp, y_train, y_test = train_test_split(
        filepaths, labels, test_size=config.TEST_SPLIT, stratify=labels,
        random_state=config.SEED,
    )

    print("Extracting YAMNet embeddings from preprocessed audio (train)...")
    X_train = extract_embeddings(train_fp, desc="train", apply_highpass=apply_highpass, apply_windowing=apply_windowing)
    print("Extracting YAMNet embeddings from preprocessed audio (test)...")
    X_test = extract_embeddings(test_fp, desc="test", apply_highpass=apply_highpass, apply_windowing=apply_windowing)

    class_weights = compute_class_weight("balanced", classes=np.arange(num_classes), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    print(f"\nClass weights (balanced): "
          f"{dict(zip(class_names, [round(w, 2) for w in class_weights]))}")

    y_train_oh = keras.utils.to_categorical(y_train, num_classes)

    model = build_mlp_head(config.EMBEDDING_DIM, num_classes)
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

    safe_mode = (mode or "none").replace("+", "_")
    model_path = os.path.join(EXPERIMENT_OUTPUT_DIR, f"cry_reason_mlp_preprocessed_{safe_mode}.keras")
    callbacks = [
        keras.callbacks.ModelCheckpoint(model_path, monitor="val_accuracy", save_best_only=True),
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

    print(f"\n=== cry_reason_mlp_preprocessed (belly pain / hungry / fussy, mode={mode or 'none'}) ===")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
    top2 = top_k_accuracy_score(y_test, probs, k=2, labels=np.arange(num_classes))
    print(f"Top-2 accuracy: {top2:.4f}")
    print(f"\nFor comparison, current shipped model (train_cry_reason.py, no preprocessing): "
          f"46% accuracy / 84.6% top-2")

    cm = confusion_matrix(y_test, y_pred, labels=np.arange(num_classes))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Cry-reason confusion matrix (mode={mode or 'none'})")
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    confusion_path = os.path.join(EXPERIMENT_OUTPUT_DIR, f"cry_reason_confusion_preprocessed_{safe_mode}.png")
    fig.savefig(confusion_path, dpi=150)
    print(f"Saved {confusion_path}")
    print(f"Saved model to {model_path}")  # written by ModelCheckpoint during fit (best val_accuracy epoch)


if __name__ == "__main__":
    import sys
    mode_arg = sys.argv[1] if len(sys.argv) > 1 else "highpass+window"
    main(
        apply_highpass="highpass" in mode_arg,
        apply_windowing="window" in mode_arg,
    )
