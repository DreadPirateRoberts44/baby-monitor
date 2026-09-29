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
import math
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

# Session tracking (see session.py): how long, in seconds, after the LAST
# confidently-"cry" window before a session is considered truly over. Any
# confident "cry" detected before this gap elapses is folded into the SAME
# session (start time unchanged, aggregated reason keeps accumulating)
# rather than starting a new one -- e.g. baby cries, is picked up and
# carried away from the monitor (or just pauses/settles), then cries again
# 10 minutes later: that's one episode, not two. Set high enough to cover
# "picked up, walked to another room, calmed down or not" -- a short gap
# (e.g. the old ~8s grace window) reliably mistook that for two separate
# episodes. The session's logged end time is still the LAST confident cry
# in the episode, not whenever this window finally lapses -- see
# session.py's last_cry_at_utc/duration_seconds.
SESSION_MERGE_WINDOW_SECONDS = 10 * 60

# Session tracking (see session.py): minimum confirmed crying, in seconds,
# before a session is actually started/alerted. A single confident "cry"
# window used to be enough on its own -- a 4s misclassification (a bark, a
# static burst, anything that briefly clears STAGE1_CONFIDENCE_THRESHOLD)
# would fire a full "started" alert. Expressed as SECONDS rather than a
# raw window count so it means the same thing (e.g. "~6s of confirmation")
# regardless of CAPTURE_INTERVAL_SECONDS -- SESSION_START_MIN_WINDOWS below
# is derived from this, not set directly, so changing the capture interval
# later doesn't silently change how much confirmation lag this adds.
SESSION_START_MIN_SECONDS = 6.0

# Derived: how many CONSECUTIVE confident-"cry" windows are needed before
# starting/alerting a session -- see session.py. Rounds up (ceil), and
# always at least 1, so SESSION_START_MIN_SECONDS <= CAPTURE_INTERVAL_SECONDS
# still means "the very first window is enough", matching the old behavior.
SESSION_START_MIN_WINDOWS = max(1, math.ceil(SESSION_START_MIN_SECONDS / CAPTURE_INTERVAL_SECONDS))

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
