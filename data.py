import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split

import config
import yamnet_features


AUDIO_EXTENSIONS = (".wav", ".ogg")


def list_files_and_labels(data_dir):
    """Scan label subfolders, group them into config.CLASS_GROUPS, and return
    (filepaths, labels, class_names). Folders not mentioned in CLASS_GROUPS
    are skipped."""
    group_of = {}
    for group, folders in config.CLASS_GROUPS.items():
        for folder in folders:
            group_of[folder] = group
    class_names = sorted(config.CLASS_GROUPS.keys())
    class_index = {cls: idx for idx, cls in enumerate(class_names)}

    available = sorted(
        d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
    )
    unmapped = [d for d in available if d not in group_of]
    if unmapped:
        print(f"  WARNING: folders not in config.CLASS_GROUPS, skipping: {unmapped}")

    filepaths, labels = [], []
    counts = {cls: 0 for cls in class_names}
    for folder in available:
        if folder not in group_of:
            continue
        cls = group_of[folder]
        cls_dir = os.path.join(data_dir, folder)
        cls_files = [f for f in os.listdir(cls_dir) if f.lower().endswith(AUDIO_EXTENSIONS)]
        counts[cls] += len(cls_files)
        for fname in cls_files:
            filepaths.append(os.path.join(cls_dir, fname))
            labels.append(class_index[cls])

    for cls in class_names:
        print(f"  {cls}: {counts[cls]} files")
        if counts[cls] < 5:
            print(f"    WARNING: too few samples to split reliably into train/val/test")

    return filepaths, labels, class_names


def augment_waveform(y):
    # small random gain + noise + time shift; keeps oversampled duplicates
    # from being identical training examples
    y = y * np.random.uniform(0.85, 1.15)
    y = y + np.random.normal(0, 0.005, size=y.shape)
    shift = np.random.randint(-config.SAMPLE_RATE // 4, config.SAMPLE_RATE // 4)
    y = np.roll(y, shift)
    return y


def load_and_extract(filepath, augment=False):
    """Load a WAV file, pad/trim to fixed length, return YAMNet embedding."""
    y, _ = librosa.load(filepath, sr=config.SAMPLE_RATE, mono=True)

    if len(y) < config.N_SAMPLES:
        y = np.pad(y, (0, config.N_SAMPLES - len(y)))
    else:
        y = y[: config.N_SAMPLES]

    if augment:
        y = augment_waveform(y)

    return yamnet_features.get_embedding(y.astype(np.float32))


def oversample(filepaths, labels):
    """Duplicate minority-class samples up to the majority class count.
    Returns filepaths, labels, and a parallel is_duplicate flag."""
    filepaths = np.array(filepaths)
    labels = np.array(labels)
    counts = np.bincount(labels)
    max_count = counts.max()

    out_fp, out_lbl, out_dup = [], [], []
    for cls, count in enumerate(counts):
        cls_fp = filepaths[labels == cls]
        out_fp.extend(cls_fp)
        out_lbl.extend([cls] * count)
        out_dup.extend([False] * count)

        if count == 0:
            print(f"  WARNING: class index {cls} has 0 training samples — "
                  f"it will never be predicted. Add more data for this class.")
            continue

        if count < max_count:
            extra_idx = np.random.choice(len(cls_fp), max_count - count, replace=True)
            out_fp.extend(cls_fp[extra_idx])
            out_lbl.extend([cls] * (max_count - count))
            out_dup.extend([True] * (max_count - count))

    return list(out_fp), out_lbl, out_dup


def extract_features(filepaths, labels, augment_flags=None, desc=""):
    """Compute YAMNet embeddings ONCE up front (not per epoch) and return
    them as plain numpy arrays. This is what makes training fast — the
    classifier head then trains on cached arrays, not live audio."""
    if augment_flags is None:
        augment_flags = [False] * len(filepaths)

    feats = np.zeros((len(filepaths), config.EMBEDDING_DIM), dtype=np.float32)
    for i, (fp, aug) in enumerate(zip(filepaths, augment_flags)):
        feats[i] = load_and_extract(fp, augment=aug)
        if desc and (i + 1) % 50 == 0:
            print(f"  [{desc}] {i + 1}/{len(filepaths)}")
    return feats, np.array(labels, dtype=np.int32)


def build_dataset(features, labels, batch_size, num_classes, shuffle=False):
    one_hot = tf.keras.utils.to_categorical(labels, num_classes=num_classes)
    ds = tf.data.Dataset.from_tensor_slices((features, one_hot))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(features), seed=config.SEED)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def get_datasets():
    """Returns train_ds, val_ds, test_ds, class_names, input_shape."""
    filepaths, labels, class_names = list_files_and_labels(config.DATA_DIR)

    train_fp, temp_fp, train_lbl, temp_lbl = train_test_split(
        filepaths, labels,
        test_size=config.VAL_SPLIT + config.TEST_SPLIT,
        stratify=labels, random_state=config.SEED,
    )
    val_ratio = config.VAL_SPLIT / (config.VAL_SPLIT + config.TEST_SPLIT)
    val_fp, test_fp, val_lbl, test_lbl = train_test_split(
        temp_fp, temp_lbl,
        test_size=1 - val_ratio,
        stratify=temp_lbl, random_state=config.SEED,
    )

    # balance only the training set; val/test stay untouched for honest metrics
    train_fp, train_lbl, train_dup = oversample(train_fp, train_lbl)

    print("Extracting YAMNet embeddings (one-time cost, not per epoch)...")
    train_feat, train_lbl = extract_features(train_fp, train_lbl, train_dup, desc="train")
    val_feat, val_lbl = extract_features(val_fp, val_lbl, desc="val")
    test_feat, test_lbl = extract_features(test_fp, test_lbl, desc="test")

    num_classes = len(class_names)
    train_ds = build_dataset(train_feat, train_lbl, config.BATCH_SIZE, num_classes, shuffle=True)
    val_ds = build_dataset(val_feat, val_lbl, config.BATCH_SIZE, num_classes)
    test_ds = build_dataset(test_feat, test_lbl, config.BATCH_SIZE, num_classes)

    input_shape = (config.EMBEDDING_DIM,)
    return train_ds, val_ds, test_ds, class_names, input_shape