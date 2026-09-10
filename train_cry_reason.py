"""Second-stage model: given a clip already identified as a cry, predict
the likeliest reason among belly pain / hungry / fussy.

Every attempt at the original 6-way split (belly pain/burping/cold_hot/
discomfort/hungry/tired) landed near chance across 5 independent modeling
approaches and a 4-way ensemble of them -- best case was 28% accuracy /
45.6% top-2 on 6 classes (see config.py's CRY_REASON_GROUPS comment for the
full history). burping/cold_hot/discomfort/tired are collapsed into one
"fussy" catch-all; belly pain and hungry are kept distinct since they
showed the most consistent, above-chance recall across those experiments.

Uses the best-performing architecture from that comparison: YAMNet
mean-pooled embedding -> small MLP softmax head, with class-weight
balancing and label smoothing (matches the project's stated preference for
false positives over false negatives -- don't let the majority class
suppress minority-class recall, and keep softmax confidence meaningful).

Audio is high-pass filtered at ~300Hz (audio_preprocessing.highpass_filter)
before embedding extraction -- removes room rumble/handling noise/HVAC hum
below the fundamental frequency range of infant cries. An isolated A/B
comparison (train_cry_reason_preprocessed.py, kept in experiments/) found
this genuinely improves results and, unlike every prior attempt on this
3-way split, gives balanced, above-chance recall on all three classes
rather than one class swallowing the others (see
output/cry_reason_confusion_matrix.png after training -- belly pain,
fussy, and hungry are all meaningfully populated on the diagonal). A
second preprocessing idea tried in the same comparison -- cropping each
clip to a ~1s window around its single loudest cry bout -- was tested and
REJECTED: it hurt performance noticeably, most likely because it discards
multi-bout rhythm context that the mean-pooled YAMNet embedding was
otherwise picking up across the full clip. Only the high-pass filter is
applied here.
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
from audio_preprocessing import highpass_filter


def list_cry_reason_files():
    """Scan config.CRY_REASON_GROUPS, return (filepaths, labels).
    A class like "fussy" pulls files from multiple data/ subfolders."""
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
    return highpass_filter(y.astype(np.float32))


def extract_embeddings(filepaths, desc=""):
    feats = np.zeros((len(filepaths), config.EMBEDDING_DIM), dtype=np.float32)
    for i, fp in enumerate(filepaths):
        feats[i] = yamnet_features.get_embedding(load_waveform(fp))
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
    return keras.Model(inputs, outputs, name="cry_reason_head")


def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    filepaths, labels = list_cry_reason_files()
    class_names = config.CRY_REASON_CLASSES
    num_classes = len(class_names)

    train_fp, test_fp, y_train, y_test = train_test_split(
        filepaths, labels, test_size=config.TEST_SPLIT, stratify=labels,
        random_state=config.SEED,
    )

    print("Extracting YAMNet embeddings (train)...")
    X_train = extract_embeddings(train_fp, desc="train")
    print("Extracting YAMNet embeddings (test)...")
    X_test = extract_embeddings(test_fp, desc="test")

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

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            config.CRY_REASON_MODEL_PATH, monitor="val_accuracy", save_best_only=True
        ),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
    ]
    model.fit(
        X_train[fit_idx], y_train_oh[fit_idx],
        validation_data=(X_train[val_idx], y_train_oh[val_idx]),
        epochs=config.EPOCHS, batch_size=config.BATCH_SIZE,
        class_weight=class_weight_dict, callbacks=callbacks,
    )

    with open(config.CRY_REASON_LABELS_PATH, "w") as f:
        f.write("\n".join(class_names))

    probs = model.predict(X_test, verbose=0)
    y_pred = probs.argmax(axis=1)

    print(f"\n=== cry_reason_mlp (belly pain / hungry / fussy) ===")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
    top2 = top_k_accuracy_score(y_test, probs, k=2, labels=np.arange(num_classes))
    print(f"Top-2 accuracy: {top2:.4f}")

    cm = confusion_matrix(y_test, y_pred, labels=np.arange(num_classes))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Cry-reason confusion matrix (belly pain / hungry / fussy)")
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    out_path = f"{config.OUTPUT_DIR}/cry_reason_confusion_matrix.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    print(f"Saved model to {config.CRY_REASON_MODEL_PATH}")


if __name__ == "__main__":
    main()
