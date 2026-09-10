"""Second-stage cry-reason model, cascade variant.

The single 3-way model (belly pain / hungry / fussy, see
train_cry_reason.py) reached 46% accuracy but the confusion matrix showed
"fussy" swallowing "hungry" (79% of true hungry samples predicted as
fussy) even with class_weight="balanced" -- likely because fussy's four
merged sub-reasons (burping/cold_hot/discomfort/tired) already looked
acoustically similar to hungry in every earlier experiment (the original
8-class confusion matrix showed the same four collapsing into "hungry").

This tries a structural fix instead of a data-reweighting one: two
binary decisions instead of one 3-way decision.

  Stage 1: hungry vs. {belly pain, fussy}
  Stage 2 (only for stage-1 "not hungry" predictions): belly pain vs. fussy

Binary classifiers tend to be more robust to class imbalance and weak
features than a single multiclass boundary, and this directly targets the
observed failure mode (one class dominating the decision) rather than
just reweighting the existing 3-way loss.

Both stages use the same YAMNet-embedding + small-MLP architecture as
train_cry_reason.py, trained independently with class_weight="balanced"
and label smoothing. Reports each stage's own metrics plus the combined
end-to-end 3-way accuracy so it's directly comparable to the single-model
result.
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
    return keras.Model(inputs, outputs)


def train_binary(X_train, y_train_bin, label):
    class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train_bin)
    class_weight_dict = dict(enumerate(class_weights))
    y_train_oh = keras.utils.to_categorical(y_train_bin, 2)

    model = build_binary_head(config.EMBEDDING_DIM)
    model.compile(
        optimizer=keras.optimizers.Adam(config.LEARNING_RATE),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=config.LABEL_SMOOTHING),
        metrics=["accuracy"],
    )

    n_val = int(len(X_train) * config.VAL_SPLIT)
    rng = np.random.RandomState(config.SEED)
    idx = rng.permutation(len(X_train))
    val_idx, fit_idx = idx[:n_val], idx[n_val:]

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
    ]
    print(f"\nTraining stage: {label} (class weights: {class_weight_dict})")
    model.fit(
        X_train[fit_idx], y_train_oh[fit_idx],
        validation_data=(X_train[val_idx], y_train_oh[val_idx]),
        epochs=config.EPOCHS, batch_size=config.BATCH_SIZE,
        class_weight=class_weight_dict, callbacks=callbacks, verbose=0,
    )
    return model


def report_binary(model, X_test, y_test_bin, pos_label, neg_label, stage_name):
    probs = model.predict(X_test, verbose=0)
    y_pred = probs.argmax(axis=1)
    print(f"\n=== Stage report: {stage_name} ===")
    print(classification_report(y_test_bin, y_pred, target_names=[neg_label, pos_label], zero_division=0))


def main():
    os.makedirs(EXPERIMENT_OUTPUT_DIR, exist_ok=True)

    filepaths, labels = list_cry_reason_files()
    class_names = config.CRY_REASON_CLASSES  # ["belly pain", "fussy", "hungry"] (sorted)
    hungry_idx = class_names.index("hungry")
    belly_idx = class_names.index("belly pain")
    fussy_idx = class_names.index("fussy")

    train_fp, test_fp, y_train, y_test = train_test_split(
        filepaths, labels, test_size=config.TEST_SPLIT, stratify=labels,
        random_state=config.SEED,
    )

    print("Extracting YAMNet embeddings (train)...")
    X_train = extract_embeddings(train_fp, desc="train")
    print("Extracting YAMNet embeddings (test)...")
    X_test = extract_embeddings(test_fp, desc="test")

    # Stage 1: hungry (1) vs. not-hungry (0)
    y_train_s1 = (y_train == hungry_idx).astype(np.int32)
    y_test_s1 = (y_test == hungry_idx).astype(np.int32)
    stage1 = train_binary(X_train, y_train_s1, label="stage1_hungry_vs_rest")
    report_binary(stage1, X_test, y_test_s1, pos_label="hungry", neg_label="not_hungry",
                  stage_name="Stage 1 (hungry vs rest)")

    # Stage 2: trained only on non-hungry examples. belly pain (1) vs fussy (0)
    non_hungry_train_mask = y_train != hungry_idx
    X_train_s2 = X_train[non_hungry_train_mask]
    y_train_s2_raw = y_train[non_hungry_train_mask]
    y_train_s2 = (y_train_s2_raw == belly_idx).astype(np.int32)
    stage2 = train_binary(X_train_s2, y_train_s2, label="stage2_bellypain_vs_fussy")

    non_hungry_test_mask = y_test != hungry_idx
    X_test_s2 = X_test[non_hungry_test_mask]
    y_test_s2_raw = y_test[non_hungry_test_mask]
    y_test_s2 = (y_test_s2_raw == belly_idx).astype(np.int32)
    report_binary(stage2, X_test_s2, y_test_s2, pos_label="belly_pain", neg_label="fussy",
                  stage_name="Stage 2 (belly pain vs fussy, non-hungry subset only)")

    # End-to-end: run the full cascade on the test set and report the
    # combined 3-way result, directly comparable to train_cry_reason.py.
    stage1_probs = stage1.predict(X_test, verbose=0)
    stage1_pred_hungry = stage1_probs.argmax(axis=1) == 1

    final_pred = np.full(len(X_test), fussy_idx, dtype=np.int32)
    final_pred[stage1_pred_hungry] = hungry_idx

    not_hungry_idx = np.where(~stage1_pred_hungry)[0]
    if len(not_hungry_idx):
        stage2_probs = stage2.predict(X_test[not_hungry_idx], verbose=0)
        stage2_pred_belly = stage2_probs.argmax(axis=1) == 1
        final_pred[not_hungry_idx[stage2_pred_belly]] = belly_idx

    print("\n=== End-to-end cascade (3-way, comparable to single-model result) ===")
    print(classification_report(y_test, final_pred, target_names=class_names, zero_division=0))

    cm = confusion_matrix(y_test, final_pred, labels=np.arange(len(class_names)))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Cry-reason confusion matrix (cascade)")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    out_path = f"{EXPERIMENT_OUTPUT_DIR}/cry_reason_confusion_cascade.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")

    stage1.save(os.path.join(EXPERIMENT_OUTPUT_DIR, "cry_reason_cascade_stage1.keras"))
    stage2.save(os.path.join(EXPERIMENT_OUTPUT_DIR, "cry_reason_cascade_stage2.keras"))
    print(f"Saved cascade models to {EXPERIMENT_OUTPUT_DIR}/cry_reason_cascade_stage{{1,2}}.keras")


if __name__ == "__main__":
    main()
