"""Calibrate a confidence-rejection threshold for the classifier head.

Two signals:
1. Val set (known classes): sweep thresholds, report accuracy-if-accepted
   and %-rejected at each, so we can pick a threshold that filters out
   low-confidence errors without rejecting too much good data.
2. OOD holdout (data_ood_holdout/noise): a subset of noise/ files excluded
   from train/val/test entirely, used purely to check whether the model's
   confidence stays low on sounds unlike anything it trained on. If this
   folder is missing, that check is skipped.

Threshold selection rule: the smallest threshold t such that accuracy-if-
accepted on val reaches (or gets within EPS of) 1.0 among not-too-few
rejections, i.e. the "knee" of the accuracy-vs-rejection curve.
"""
import os
import numpy as np
import tensorflow as tf

import config
from data import get_datasets, load_and_extract

OOD_DIR = "data_ood_holdout/noise"


def confidences_and_correctness(model, ds):
    confs, correct = [], []
    for x_batch, y_batch in ds:
        probs = model.predict(x_batch, verbose=0)
        preds = probs.argmax(axis=1)
        y_true = np.argmax(y_batch.numpy(), axis=1)
        confs.extend(probs.max(axis=1))
        correct.extend(preds == y_true)
    return np.array(confs), np.array(correct)


def sweep_thresholds(confs, correct, thresholds):
    print(f"{'thresh':>8} {'accepted':>10} {'rejected':>10} {'acc@accept':>12} {'errors_kept':>12}")
    for t in thresholds:
        accept = confs >= t
        n_accept = accept.sum()
        n_reject = (~accept).sum()
        acc_if_accept = correct[accept].mean() if n_accept else float("nan")
        errors_kept = (accept & ~correct).sum()
        print(f"{t:8.2f} {n_accept:10d} {n_reject:10d} {acc_if_accept:12.4f} {errors_kept:12d}")


def pick_threshold(confs, correct, thresholds, min_accept_frac=0.75, target_acc=0.995):
    """Smallest threshold reaching target_acc accuracy-if-accepted while
    still accepting at least min_accept_frac of the val set."""
    n = len(confs)
    best = thresholds[-1]
    for t in thresholds:
        accept = confs >= t
        if accept.sum() / n < min_accept_frac:
            break
        acc_if_accept = correct[accept].mean() if accept.sum() else 0.0
        if acc_if_accept >= target_acc:
            best = t
            break
    return best


def main():
    _, val_ds, _, class_names, _ = get_datasets()
    model = tf.keras.models.load_model(config.MODEL_PATH)

    print("=== Val set: confidence calibration ===")
    confs, correct = confidences_and_correctness(model, val_ds)
    print(f"Correct predictions: mean conf {confs[correct].mean():.3f}, "
          f"median {np.median(confs[correct]):.3f}")
    if (~correct).sum():
        print(f"Incorrect predictions: mean conf {confs[~correct].mean():.3f}, "
              f"median {np.median(confs[~correct]):.3f}")
    else:
        print("Incorrect predictions: none in val set")

    thresholds = np.round(np.arange(0.3, 1.0, 0.05), 2)
    sweep_thresholds(confs, correct, thresholds)

    threshold = pick_threshold(confs, correct, thresholds)
    print(f"\nSelected threshold: {threshold:.2f}")

    print("\n=== OOD holdout: sounds excluded from training ===")
    if not os.path.isdir(OOD_DIR):
        print(f"  {OOD_DIR} not found, skipping OOD check.")
    else:
        files = sorted(f for f in os.listdir(OOD_DIR) if f.lower().endswith(".ogg"))
        feats = np.zeros((len(files), config.EMBEDDING_DIM), dtype=np.float32)
        for i, fname in enumerate(files):
            feats[i] = load_and_extract(os.path.join(OOD_DIR, fname))
        probs = model.predict(feats, verbose=0)
        ood_confs = probs.max(axis=1)
        ood_preds = probs.argmax(axis=1)

        print(f"{len(files)} held-out noise files (never seen in train/val/test)")
        print(f"Mean confidence: {ood_confs.mean():.3f}, median: {np.median(ood_confs):.3f}")
        below = (ood_confs < threshold).sum()
        print(f"Below threshold {threshold:.2f} (correctly flagged uncertain): "
              f"{below}/{len(files)} ({100 * below / len(files):.1f}%)")

        pred_counts = np.bincount(ood_preds, minlength=len(class_names))
        print("Prediction distribution (pre-threshold):")
        for idx, cls in enumerate(class_names):
            print(f"  {cls:10s}: {pred_counts[idx]:3d}")

        correct_class = class_names.index("noise")
        above_and_right = ((ood_confs >= threshold) & (ood_preds == correct_class)).sum()
        above_and_wrong = ((ood_confs >= threshold) & (ood_preds != correct_class)).sum()
        print(f"\nAccepted (conf >= {threshold:.2f}) as correct noise: {above_and_right}")
        print(f"Accepted (conf >= {threshold:.2f}) as WRONG class (false alarm risk): {above_and_wrong}")


if __name__ == "__main__":
    main()
