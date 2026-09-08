import os

# Paths
DATA_DIR = "data"  # root dir containing one subfolder per label
OUTPUT_DIR = "output"
MODEL_PATH = os.path.join(OUTPUT_DIR, "cry_classifier.keras")
TFLITE_PATH = os.path.join(OUTPUT_DIR, "cry_classifier.tflite")
LABELS_PATH = os.path.join(OUTPUT_DIR, "labels.txt")

# Audio params
SAMPLE_RATE = 16000
DURATION = 4.0  # seconds; clips are padded/truncated to this length
N_SAMPLES = int(SAMPLE_RATE * DURATION)

# Feature extraction (log-mel spectrogram)
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512
# Resulting feature shape: (N_MELS, ceil(N_SAMPLES / HOP_LENGTH), 1)

# Training
BATCH_SIZE = 32
EPOCHS = 60
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SEED = 42
