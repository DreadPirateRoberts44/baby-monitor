"""Compare voice-quality features (jitter/shimmer/HNR + MFCCs, new in this
pass) and gradient-boosted trees (XGBoost, new in this pass) against the
existing scalar prosody+rhythm features and logistic regression baseline
-- across BOTH the shipped 3-class scheme (belly pain/hungry/fussy) and
the original 6-class scheme (belly pain/burping/cold_hot/discomfort/
hungry/tired), so results are comparable to every earlier experiment in
this project regardless of which class scheme they used.

Feature sets compared (all scalar, all on the same train/test split per
class scheme):
  - prosody_rhythm: prosody_features.py + cry_rhythm_features.py (what
    train_cry_reason.py's predecessor logistic-regression experiments used)
  - voice_quality: jitter/shimmer/HNR + MFCCs (new)
  - all_combined: prosody + rhythm + voice_quality

Models compared:
  - Logistic regression (CV-tuned C, class_weight="balanced") -- matches
    every earlier logistic-regression experiment in this project
  - XGBoost (gradient-boosted trees) -- new; handles nonlinear feature
    interactions and mixed feature scales better than a linear model
    without needing MLP-scale training data

This is a comparison/decision script, not itself a shipped model. Six
(feature-set x model) combinations x 2 class schemes = 12 runs total per
invocation of main().
"""
import os
import itertools
import numpy as np
import librosa
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, top_k_accuracy_score, balanced_accuracy_score
from sklearn.dummy import DummyClassifier
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import matplotlib.pyplot as plt

import config
from prosody_features import extract_prosody_features, FEATURE_NAMES as PROSODY_NAMES
from cry_rhythm_features import extract_rhythm_features, RHYTHM_FEATURE_NAMES
from voice_quality_features import extract_voice_quality_features, VOICE_QUALITY_FEATURE_NAMES

EXPERIMENT_OUTPUT_DIR = os.path.join(config.OUTPUT_DIR, "experiments")


def list_cry_reason_files(groups, class_names):
    class_index = {cls: idx for idx, cls in enumerate(class_names)}
    filepaths, labels = [], []
    for cls in class_names:
        cls_files = []
        for folder in groups[cls]:
            folder_dir = os.path.join(config.DATA_DIR, folder)
            folder_files = [
                os.path.join(folder_dir, f) for f in os.listdir(folder_dir)
                if f.lower().endswith((".wav", ".ogg"))
            ]
            cls_files.extend(folder_files)
        print(f"    {cls}: {len(cls_files)} files")
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


def extract_all_scalar_features(filepaths, desc=""):
    prosody = np.zeros((len(filepaths), len(PROSODY_NAMES)), dtype=np.float32)
    rhythm = np.zeros((len(filepaths), len(RHYTHM_FEATURE_NAMES)), dtype=np.float32)
    voice_quality = np.zeros((len(filepaths), len(VOICE_QUALITY_FEATURE_NAMES)), dtype=np.float32)
    for i, fp in enumerate(filepaths):
        y = load_waveform(fp)
        prosody[i] = extract_prosody_features(y)
        rhythm[i] = extract_rhythm_features(y)
        voice_quality[i] = extract_voice_quality_features(y)
        if desc and (i + 1) % 50 == 0:
            print(f"    [{desc}] {i + 1}/{len(filepaths)}")
    return prosody, rhythm, voice_quality


def fit_logreg(X_train, y_train, X_test):
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=config.SEED)),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    search = GridSearchCV(pipe, {"clf__C": [0.001, 0.01, 0.1, 1.0, 10.0]}, cv=cv,
                           scoring="balanced_accuracy", n_jobs=-1)
    search.fit(X_train, y_train)
    return search.best_estimator_.predict_proba(X_test), search.best_params_


def fit_xgboost(X_train, y_train, X_test, num_classes):
    sample_weight = compute_sample_weight("balanced", y_train)
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        random_state=config.SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model.predict_proba(X_test), model


def report(y_test, probs, class_names, label, save_confusion=True):
    num_classes = len(class_names)
    y_pred = probs.argmax(axis=1)
    acc = (y_pred == y_test).mean()
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    top2 = top_k_accuracy_score(y_test, probs, k=min(2, num_classes - 1), labels=np.arange(num_classes)) \
        if num_classes > 2 else float("nan")

    print(f"\n--- {label} ---")
    print(f"Accuracy: {acc:.4f}  Balanced accuracy: {bal_acc:.4f}  Top-2 accuracy: {top2:.4f}")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

    if save_confusion:
        cm = confusion_matrix(y_test, y_pred, labels=np.arange(num_classes))
        fig, ax = plt.subplots(figsize=(max(5, num_classes), max(4, num_classes - 1)))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{label}")
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.colorbar(im)
        fig.tight_layout()
        safe_label = label.lower().replace(" ", "_").replace("/", "_")
        out_path = f"{EXPERIMENT_OUTPUT_DIR}/cry_reason_confusion_{safe_label}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    return {"label": label, "accuracy": acc, "balanced_accuracy": bal_acc, "top2_accuracy": top2}


def run_scheme(scheme_name, groups, class_names):
    print(f"\n{'=' * 70}\nCLASS SCHEME: {scheme_name} ({len(class_names)} classes: {class_names})\n{'=' * 70}")
    filepaths, labels = list_cry_reason_files(groups, class_names)
    num_classes = len(class_names)

    train_fp, test_fp, y_train, y_test = train_test_split(
        filepaths, labels, test_size=config.TEST_SPLIT, stratify=labels,
        random_state=config.SEED,
    )

    print("  Extracting scalar features (train)...")
    prosody_train, rhythm_train, vq_train = extract_all_scalar_features(train_fp, desc="train")
    print("  Extracting scalar features (test)...")
    prosody_test, rhythm_test, vq_test = extract_all_scalar_features(test_fp, desc="test")

    dummy = DummyClassifier(strategy="stratified", random_state=config.SEED)
    dummy.fit(prosody_train, y_train)
    dummy_acc = dummy.score(prosody_test, y_test)
    print(f"\n  Dummy (stratified-random) baseline accuracy: {dummy_acc:.4f}")

    feature_sets = {
        "prosody_rhythm": (
            np.concatenate([prosody_train, rhythm_train], axis=1),
            np.concatenate([prosody_test, rhythm_test], axis=1),
        ),
        "voice_quality": (vq_train, vq_test),
        "all_combined": (
            np.concatenate([prosody_train, rhythm_train, vq_train], axis=1),
            np.concatenate([prosody_test, rhythm_test, vq_test], axis=1),
        ),
    }

    results = []
    for feat_name, (X_train, X_test) in feature_sets.items():
        print(f"\n  >> Feature set: {feat_name} ({X_train.shape[1]} dims)")

        probs_lr, best_params = fit_logreg(X_train, y_train, X_test)
        label_lr = f"{scheme_name}_{feat_name}_logreg"
        print(f"     Logistic regression best C: {best_params['clf__C']}")
        results.append(report(y_test, probs_lr, class_names, label_lr))

        probs_xgb, _ = fit_xgboost(X_train, y_train, X_test, num_classes)
        label_xgb = f"{scheme_name}_{feat_name}_xgboost"
        results.append(report(y_test, probs_xgb, class_names, label_xgb))

    return results


def main():
    os.makedirs(EXPERIMENT_OUTPUT_DIR, exist_ok=True)

    all_results = []
    all_results += run_scheme("3class", config.CRY_REASON_GROUPS, config.CRY_REASON_CLASSES)
    all_results += run_scheme("6class", config.CRY_REASON_GROUPS_6CLASS, config.CRY_REASON_CLASSES_6CLASS)

    print(f"\n{'=' * 70}\nSUMMARY (sorted by balanced accuracy)\n{'=' * 70}")
    all_results.sort(key=lambda r: r["balanced_accuracy"], reverse=True)
    print(f"{'label':45s} {'acc':>8s} {'bal_acc':>8s} {'top2':>8s}")
    for r in all_results:
        print(f"{r['label']:45s} {r['accuracy']:8.4f} {r['balanced_accuracy']:8.4f} {r['top2_accuracy']:8.4f}")


if __name__ == "__main__":
    main()
