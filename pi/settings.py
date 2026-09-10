"""Pi runtime configuration.

Deliberately separate from the repo-root config.py: that file drives
training (paths into data/, hyperparameters, class-group definitions used
to build the training set) and importing it would pull in the full
training dependency stack. The Pi only needs the handful of constants
below, plus whatever the exported model/label files already encode
(number of classes, embedding dim inferred from the TFLite model itself).

Audio constants (SAMPLE_RATE, DURATION) MUST match config.py's values --
they're duplicated here rather than imported, since a mismatch would
silently produce wrong-shaped input to the models. If you change
config.SAMPLE_RATE or config.DURATION for a retrain, update these too.
"""
import os

# Must match config.SAMPLE_RATE / config.DURATION in the repo root.
SAMPLE_RATE = 16000
DURATION_SECONDS = 4.0
N_SAMPLES = int(SAMPLE_RATE * DURATION_SECONDS)

# Where the exported models/labels live on the Pi. Defaults to a sibling
# "models/" directory -- copy output/{classifier_head,cry_reason_mlp,
# yamnet_embedding}.tflite and output/{labels,cry_reason_labels}.txt here
# when deploying (see pi/README.md).
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

YAMNET_EMBEDDING_TFLITE_PATH = os.path.join(MODEL_DIR, "yamnet_embedding.tflite")
STAGE1_TFLITE_PATH = os.path.join(MODEL_DIR, "classifier_head.tflite")
STAGE1_LABELS_PATH = os.path.join(MODEL_DIR, "labels.txt")
STAGE2_TFLITE_PATH = os.path.join(MODEL_DIR, "cry_reason_mlp.tflite")
STAGE2_LABELS_PATH = os.path.join(MODEL_DIR, "cry_reason_labels.txt")

# Must match config.CONFIDENCE_THRESHOLD / config.UNCERTAIN_LABEL. See
# calibrate_threshold.py at the repo root for how this was chosen.
STAGE1_CONFIDENCE_THRESHOLD = 0.90
UNCERTAIN_LABEL = "uncertain"

# Stage 1 label that means "run stage 2 next".
CRY_LABEL = "cry"

# How often to capture and classify a window, in seconds. Set equal to
# DURATION_SECONDS for back-to-back non-overlapping windows (simplest);
# set lower than DURATION_SECONDS for overlapping windows (catches a cry
# that starts mid-window sooner, at the cost of more inference calls).
CAPTURE_INTERVAL_SECONDS = 4.0

# Microphone device index/name for sounddevice. None = system default input.
AUDIO_DEVICE = None

# Session tracking (see session.py): how many consecutive non-"cry" windows
# must be seen before considering a crying session over. Set > 1 so a
# single brief pause/gasp mid-cry (correctly classified as e.g. "silence"
# or "uncertain" for one window) doesn't end the session and trigger a
# spurious "cry ended" notification immediately followed by a new "cry
# started" one. At CAPTURE_INTERVAL_SECONDS=4.0, a value of 2 means ~8s of
# continuous non-crying before the session is considered ended.
SESSION_END_GRACE_WINDOWS = 2

# Wireless button device (ESP32, see firmware/ and buttons_mqtt.py) for
# logging feed/diaper-change events used by care_events.py. The Pi runs
# the MQTT broker (Mosquitto) itself -- local network only, no internet/
# cloud broker involved. "localhost" here refers to the broker running on
# THIS machine (the Pi); the ESP32 connects to the Pi's LAN IP/hostname,
# configured separately in firmware/button_device/config.h.
MQTT_BROKER_HOST = "localhost"
MQTT_BROKER_PORT = 1883
MQTT_BUTTON_TOPIC = "babymonitor/button"

# Topic notify.py publishes cry alerts/live-updates to -- the app
# subscribes to this on the same local broker (see notify.py). Same
# broker as the button topic above; a distinct topic since these are a
# different message shape/purpose, not because a second broker is needed.
MQTT_NOTIFY_TOPIC = "babymonitor/notify"

# Local sync API (see sync_api.py) -- lets the app read/write care_events
# and read cry_history when it's on the same WiFi network as the Pi. LAN
# only, no auth (same trust model as the MQTT broker above) -- do not
# port-forward this or otherwise expose it to the internet.
SYNC_API_PORT = 8081
