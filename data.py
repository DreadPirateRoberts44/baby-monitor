import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split

import config


def list_files_and_labels(data_dir):
    """Scan label subfolders, return (filepaths, labels, class_names)."""
    class_names = sorted(
        d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
    )
    filepaths, labels = [], []
    for idx, cls in enumerate(class_names):
        cls_dir = os.path.join(data_dir, cls)
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(".wav"):
                filepaths.append(os.path.join(cls_dir, fname))
                labels.append(idx)
    return filepaths, labels, class_names


def load_and_extract(filepath):
    """Load a WAV file, pad/trim to fixed length, return log-mel spectrogram."""
    y, _ = librosa.load(filepath, sr=config.SAMPLE_RATE, mono=True)

    if len(y) < config.N_SAMPLES:
        y = np.pad(y, (0, config.N_SAMPLES - len(y)))
    else:
        y = y[: config.N_SAMPLES]

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=config.SAMPLE_RATE,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    # normalize to roughly [0, 1] for stable training
    log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8)
    return log_mel.astype(np.float32)[..., np.newaxis]  # (n_mels, time, 1)


def build_dataset(filepaths, labels, batch_size, shuffle=False):
    def generator():
        for fp, lbl in zip(filepaths, labels):
            yield load_and_extract(fp), lbl

    sample_feat = load_and_extract(filepaths[0])
    output_signature = (
        tf.TensorSpec(shape=sample_feat.shape, dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )
    ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(filepaths), seed=config.SEED)
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

    train_ds = build_dataset(train_fp, train_lbl, config.BATCH_SIZE, shuffle=True)
    val_ds = build_dataset(val_fp, val_lbl, config.BATCH_SIZE)
    test_ds = build_dataset(test_fp, test_lbl, config.BATCH_SIZE)

    input_shape = load_and_extract(filepaths[0]).shape
    return train_ds, val_ds, test_ds, class_names, input_shape
