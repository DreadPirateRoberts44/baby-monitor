import os

# Paths
DATA_DIR = "data"  # root dir containing one subfolder per label

# Target classes -> which data/ subfolders feed into each.
# The original cry-reason subfolders (belly pain/burping/cold_hot/discomfort/
# hungry/tired) are not separable in YAMNet embedding space (confirmed via
# t-SNE + centroid distance analysis: between-class centroid distances of
# 0.4-1.0 vs. within-class spread of ~4.6-4.9), so they're merged into one
# "cry" class. laugh/silence are acoustically distinct and stay split out.
# noise (ESC-50 environmental sounds) is added so the model has a class for
# "not a baby sound" instead of forcing every non-baby input into one of the
# baby-sound labels.
CLASS_GROUPS = {
    "cry": ["belly pain", "burping", "cold_hot", "discomfort", "hungry", "tired"],
    "laugh": ["laugh"],
    "silence": ["silence"],
    "noise": ["noise"],
}
OUTPUT_DIR = "output"
MODEL_PATH = os.path.join(OUTPUT_DIR, "classifier_head.keras")
TFLITE_PATH = os.path.join(OUTPUT_DIR, "classifier_head.tflite")
YAMNET_TFLITE_PATH = os.path.join(OUTPUT_DIR, "yamnet.tflite")  # downloaded, not trained
# NOTE: yamnet.tflite above is Google's classification-only build (521-class
# AudioSet softmax, fixed 0.975s input) -- it does NOT expose the 1024-dim
# embedding layer the classifier heads were trained on. The Pi (and any
# other TFLite-only consumer) needs yamnet_embedding.tflite instead, built
# by export_yamnet_embedding_tflite.py, which wraps TF Hub YAMNet's
# embedding output + the same mean-pooling yamnet_features.get_embedding()
# does, verified to numerically match it.
YAMNET_EMBEDDING_TFLITE_PATH = os.path.join(OUTPUT_DIR, "yamnet_embedding.tflite")
LABELS_PATH = os.path.join(OUTPUT_DIR, "labels.txt")

# Second-stage cry-reason model: only ever runs after the first-stage model
# has already decided a clip is a "cry". Every prior attempt at the full
# 6-way split (belly pain/burping/cold_hot/discomfort/hungry/tired) landed
# near chance across 5 independent modeling approaches (MLP+YAMNet, scalar
# prosody+logreg, bout/pause rhythm+logreg, prosody-sequence Conv1D,
# YAMNet-sequence Conv1D) and an ensemble of all four -- best case was 28%
# accuracy / 45.6% top-2 on 6 classes, with cold_hot essentially never
# learnable (0-6% recall almost everywhere). burping/cold_hot/discomfort/
# tired are collapsed into one "fussy" catch-all; belly pain and hungry
# stay distinct as the two reasons that showed the most consistent,
# above-chance recall across those experiments.
CRY_REASON_GROUPS = {
    "belly pain": ["belly pain"],
    "hungry": ["hungry"],
    "fussy": ["burping", "cold_hot", "discomfort", "tired"],
}
CRY_REASON_CLASSES = sorted(CRY_REASON_GROUPS.keys())
CRY_REASON_MODEL_PATH = os.path.join(OUTPUT_DIR, "cry_reason_mlp.keras")
CRY_REASON_TFLITE_PATH = os.path.join(OUTPUT_DIR, "cry_reason_mlp.tflite")
CRY_REASON_LABELS_PATH = os.path.join(OUTPUT_DIR, "cry_reason_labels.txt")

# Original, unmerged 6-way split -- kept alongside CRY_REASON_GROUPS so new
# feature/model experiments (voice-quality features, gradient boosting,
# etc. -- see cry_reason_feature_comparison.py) can be evaluated against
# both the 3-class scheme actually shipped and the original 6-class scheme,
# for direct comparison against every earlier experiment run at 6 classes.
CRY_REASON_GROUPS_6CLASS = {
    "belly pain": ["belly pain"],
    "burping": ["burping"],
    "cold_hot": ["cold_hot"],
    "discomfort": ["discomfort"],
    "hungry": ["hungry"],
    "tired": ["tired"],
}
CRY_REASON_CLASSES_6CLASS = sorted(CRY_REASON_GROUPS_6CLASS.keys())

# Audio params
SAMPLE_RATE = 16000
DURATION = 4.0  # seconds; clips are padded/truncated to this length
N_SAMPLES = int(SAMPLE_RATE * DURATION)

# YAMNet (TF Hub) — used only at train/export time to compute embeddings
YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"
EMBEDDING_DIM = 1024

# Training
BATCH_SIZE = 32
EPOCHS = 60
LEARNING_RATE = 1e-3
LABEL_SMOOTHING = 0.1  # keeps softmax confidence meaningful for threshold-based rejection

# Below this max-softmax confidence, treat a prediction as "uncertain" rather
# than trusting the top class. Calibrated via calibrate_threshold.py: on the
# val set this is the knee where accuracy-if-accepted reaches 1.0 without
# rejecting too much of the set (~80% kept), and it holds up on a held-out
# noise subset never seen during training. Re-run calibrate_threshold.py and
# update this value after any retrain.
CONFIDENCE_THRESHOLD = 0.90
UNCERTAIN_LABEL = "uncertain"
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SEED = 42

