# Baby Monitor — Cry Detection & Reason Classification

A two-stage audio classification pipeline for a baby monitor:

1. **Stage 1 — sound classification.** Given a short audio clip, decide whether
   it's a `cry`, `laugh`, `silence`, or `noise` (anything that isn't a baby
   sound). Uses [YAMNet](https://tfhub.dev/google/yamnet/1) embeddings feeding
   a small classifier head, exported to TFLite for on-device (Raspberry
   Pi-class) inference.
2. **Stage 2 — cry-reason classification.** Only runs when stage 1 says
   `cry`. Predicts the likeliest reason for the cry: `belly pain`, `hungry`,
   or `fussy` (a catch-all covering burping/cold/hot/discomfort/tiredness —
   see [Why "fussy"?](#why-fussy) below). Intended as a ranked, low-confidence
   *suggestion* to a caregiver, not a diagnosis.

Both stages share the same underlying approach: extract a fixed-length
embedding or feature vector from a 4-second audio clip, then train a small
classifier on top. All training/export scripts are driven by [config.py](config.py).

**Building the physical device?** See [docs/INDEX.md](docs/INDEX.md) for
the full documentation map — hardware shopping list, physical assembly,
and software install guides all live in [docs/](docs/).

## Repository layout

```
config.py               Central configuration: paths, class schemes, audio
                         params, training hyperparameters. Start here.
data.py                 Stage-1 dataset loading, oversampling, tf.data pipeline.
model.py                Stage-1 classifier head architecture (Dense/Dropout).
yamnet_features.py      YAMNet embedding extraction (mean-pooled).

train.py                Train the stage-1 cry/laugh/silence/noise model.
evaluate.py              Evaluate stage-1 on the test split, save a confusion matrix.
export_tflite.py        Export the stage-1 AND stage-2 classifier heads to TFLite.
export_yamnet_embedding_tflite.py  Export a TFLite YAMNet that outputs the
                         1024-dim embedding (see Pi runtime below for why
                         this differs from output/yamnet.tflite).
calibrate_threshold.py  Pick/validate the stage-1 confidence-rejection threshold.
predict.py               Reference stage-1 inference path (file -> label + confidence).

train_cry_reason.py     Train the stage-2 cry-reason model (belly pain/hungry/fussy).
audio_preprocessing.py  High-pass filtering + cry-window extraction, applied
                         before feature/embedding extraction for STAGE 2 ONLY
                         (see Preprocessing) — stage 1 is trained on unfiltered
                         audio; do not apply this before stage-1 inference.
prosody_features.py     Scalar prosodic features (pitch, energy, spectral shape).
cry_rhythm_features.py  Cry bout/pause segmentation + rhythm features.
sequence_features.py    Frame-level (non-pooled) feature sequences.
voice_quality_features.py  Jitter/shimmer/HNR (Praat) + MFCCs.
spectrogram_features.py Log-mel spectrogram extraction (used by experiments/
                         train_cry_reason_cnn.py; not part of the shipped path).

experiments/            Exploratory comparisons that did NOT outperform the
                         shipped stage-2 model, kept for reference. See
                         experiments/README.md.

pi/                      Raspberry Pi runtime: mic capture, on-device
                         two-stage inference (TFLite only, no full
                         TensorFlow/librosa), session tracking, cry
                         history, and MQTT notifications to the (not yet
                         built) mobile app. See pi/README.md.

docs/                    Build/install guides: hardware shopping list,
                         physical assembly, software setup. See
                         docs/INDEX.md.

data/                    Training audio, one subfolder per source label
                         (gitignored — not checked in; see Data below).
output/                  Trained models, TFLite exports, confusion matrices
                         (gitignored — regenerate by re-running the scripts).
```

## Setup

```
pip install -r requirements.txt
```

Requires Python 3.11 (matches the pinned TensorFlow/tensorflow-hub versions).

## Data

`data/` is not checked into git (see `.gitignore`) — it's ~350MB of audio.
Populate it with one subfolder per source label under `data/`:

```
data/
  belly pain/   burping/   cold_hot/   discomfort/
  hungry/       tired/     laugh/      silence/     noise/
```

`belly pain` through `tired` are cry-reason recordings; `laugh`/`silence` are
non-cry baby-adjacent sounds; `noise` is generic environmental/ambient audio
(used so stage 1 has a class for "not a baby sound" instead of forcing every
non-baby input into a baby-sound label). Both `.wav` and `.ogg` files are
supported.

## Training stage 1 (cry / laugh / silence / noise)

```
python train.py              # trains output/classifier_head.keras
python evaluate.py           # test-set metrics + output/confusion_matrix.png
python calibrate_threshold.py  # sanity-checks/re-derives CONFIDENCE_THRESHOLD
python export_tflite.py      # exports output/classifier_head.tflite + cry_reason_mlp.tflite + downloads yamnet.tflite
```

`config.CLASS_GROUPS` controls which `data/` subfolders feed into each stage-1
class. The six cry-reason subfolders are all merged into one `cry` class here
— they were found to be inseparable in YAMNet embedding space (see the
docstring/comments in `config.py`), so stage 1 only needs to detect "this is
a cry," not which kind.

Predictions below `config.CONFIDENCE_THRESHOLD` are treated as `"uncertain"`
rather than trusted outright — see `predict.py` for the reference inference
path, and `calibrate_threshold.py` for how that threshold was chosen
(project preference: prefer false positives over false negatives, so the
threshold favors flagging uncertainty over confidently guessing wrong).

## Training stage 2 (cry reason)

```
python train_cry_reason.py   # trains output/cry_reason_mlp.keras
```

Only meaningful for clips stage 1 has already classified as `cry`. Uses the
same YAMNet-embedding + small-MLP architecture as stage 1, trained on
`config.CRY_REASON_GROUPS`. Audio is high-pass filtered at ~300Hz
(`audio_preprocessing.highpass_filter`) before embedding extraction — see
[Preprocessing](#preprocessing) below.

### Why "fussy"? <a name="why-fussy"></a>

The original data has six cry-reason labels (belly pain, burping, cold_hot,
discomfort, hungry, tired). Extensive experimentation — documented in
`config.py`'s comments and in `experiments/` — found that four of these
(burping, cold_hot, discomfort, tired) are not reliably distinguishable from
each other or from hungry, across every feature representation and model
family tried (YAMNet embeddings, scalar prosody, bout/pause rhythm,
frame-level sequences, voice-quality/MFCC features, raw spectrogram CNNs;
logistic regression, gradient-boosted trees, MLPs, small CNNs, and
ensembles of these). Best result on the full 6-class split: ~28% accuracy,
6-class chance is ~17%.

Collapsing those four into one `fussy` class, keeping `belly pain` and
`hungry` distinct (the two reasons that showed the most consistent,
above-chance signal), gets meaningfully better results — see
`output/cry_reason_confusion_matrix.png` after training for current
per-class numbers (re-run `train_cry_reason.py` to regenerate; exact
numbers vary run to run with training randomness, roughly in the
40-50% accuracy / high-80s% top-2 range on 3 classes, chance ≈ 33%).
Earlier versions of this model had `hungry` recall stuck near 0 (nearly
every `hungry` clip predicted as `fussy`) even after several different
class-balancing and architecture attempts — see
[Preprocessing](#preprocessing) for what actually fixed that.

Given the project's preference for false positives over false negatives,
stage 2 is meant to be surfaced as a ranked/top-2 suggestion to a caregiver
("possibly hungry or fussy"), not a confident single-label verdict.

### Preprocessing <a name="preprocessing"></a>

`audio_preprocessing.py` high-pass filters audio at ~300Hz (Butterworth,
zero-phase) before feature/embedding extraction, removing room rumble,
handling noise, and low-frequency hum below the fundamental frequency range
of infant cries. **This is applied only in `train_cry_reason.py` (stage 2).
Stage 1 (`train.py`/`data.py`) was trained on unfiltered audio and has never
been evaluated with this filter** — applying it before stage-1 inference
would feed that model an input distribution it never saw in training. Any
new stage-1 experiment that wants to try this filter should treat it as an
untested change, not an established win (unlike for stage 2, where it's
been A/B-verified).

This was arrived at after an A/B comparison (`experiments/
train_cry_reason_preprocessed.py`) against a second preprocessing idea —
cropping each clip to a ~1s window around its single loudest cry bout —
which was tested and **rejected**: it hurt accuracy, most likely by
discarding multi-bout rhythm context the mean-pooled YAMNet embedding was
otherwise using across the full clip. High-pass filtering alone, keeping
the full clip length, is what's shipped for stage 2.

## experiments/

Scripts that tried to beat the shipped stage-2 model and didn't. Kept for
reference so the same dead ends aren't re-explored from scratch. Run them
as modules from the repo root, e.g.:

```
python -m experiments.ensemble_cry_reason
```

Their output (confusion matrices, etc.) goes to `output/experiments/`,
separate from the shipped stage-1/stage-2 artifacts in `output/`.

## Pi runtime

`pi/` contains the always-on Raspberry Pi monitor: microphone capture, the
same two-stage inference pipeline run purely via TFLite (no full
TensorFlow/tensorflow_hub/librosa), session tracking, cry history, and
MQTT-based notifications to the (not-yet-built) mobile app. See
`pi/README.md` for what each module does.

Deploying requires an extra export step beyond the two above, because
`output/yamnet.tflite` (downloaded by `export_tflite.py`) is Google's
official *classification*-only YAMNet build — fixed 0.975s input, 521-class
AudioSet softmax output — and does not expose the 1024-dim embedding layer
the classifier heads were actually trained on:

```
python export_yamnet_embedding_tflite.py   # exports output/yamnet_embedding.tflite,
                                            # verified to numerically match
                                            # yamnet_features.get_embedding()
python pi/deploy_models.py                 # copies all exported models into pi/models/
```

For the full deployment walkthrough (writing the Pi's SD card, copying
`pi/` including the now-populated `pi/models/` over, dependencies,
always-on services, remote access), see
[docs/INSTALL.md](docs/INSTALL.md).
