# Cry-reason experiments

Approaches tried for stage-2 (cry-reason) classification that did **not**
outperform the shipped model (`train_cry_reason.py` at the repo root, YAMNet
embedding + MLP + 300Hz high-pass filtering, 3-class belly pain/hungry/fussy
scheme). Kept as reference so these aren't re-explored from scratch. Run as
modules from the repo root, e.g. `python -m experiments.ensemble_cry_reason`.

| Script | Approach | Result |
|---|---|---|
| `train_cry_reason_sequence.py` | Small Conv1D over frame-level prosody sequences, and separately over un-pooled YAMNet per-frame embeddings, instead of mean-pooling to scalars | 19–24% accuracy (6-class); didn't beat mean-pooled YAMNet+MLP |
| `ensemble_cry_reason.py` | Averages predict_proba across 4 models (YAMNet+MLP, scalar prosody+rhythm+logreg, prosody-sequence Conv1D, YAMNet-sequence Conv1D) on an identical split | 23% accuracy (6-class) — worse than the best individual model (YAMNet+MLP alone, 28%); weaker models diluted the ensemble |
| `train_cry_reason_cascade.py` | Two binary decisions instead of one 3-way: hungry-vs-rest, then belly-pain-vs-fussy on the remainder | 38% end-to-end (3-class) — worse than the single 3-way model. Disproved the hypothesis that fussy was structurally "crowding out" hungry: hungry-vs-rest alone is weak (50% acc, 23% recall), so the bottleneck is hungry's weak standalone acoustic signature, not the 3-way boundary |
| `cry_reason_feature_comparison.py` | Adds jitter/shimmer/HNR (Praat) + MFCC features, and XGBoost as an alternative to logistic regression, tested across both the 3-class and 6-class schemes | Voice-quality features roughly matched (didn't beat) existing prosody+rhythm features; XGBoost consistently underperformed logistic regression (likely underfit-prone at ~840 training examples) and neither beat the shipped MLP |
| `train_cry_reason_cnn.py` + `overfit_control_cnn.py` | 2D CNN directly on log-mel spectrograms (SpecAugment, BatchNorm/Dropout/L2, capped oversampling), motivated by small-dataset cry-classification papers reporting high accuracy | Collapsed to predicting "fussy" almost exclusively even after two attempted fixes (stronger class weighting, explicit per-example sample weights via the dataset). `overfit_control_cnn.py` stripped all regularization and confirmed the model can't even memorize the *training* set past ~53% accuracy — strong evidence the spectrograms genuinely don't separate hungry/fussy, not a regularization or class-imbalance artifact |
| `train_belly_pain_detector.py` | Focused binary belly-pain-vs-everything-else detector (YAMNet+MLP), motivated by belly pain's consistent signal across other experiments | Recall dropped to 16% (AUC 0.58, barely above chance) — worse than belly pain's recall within the 3-way model. Contradicts the hypothesis that isolating belly pain as its own binary problem would help; likely explanation is that "not belly pain" as a single merged class (hungry+fussy) is a messier decision boundary than belly pain being one of three competing softmax outputs |
| `train_cry_reason_preprocessed.py` (`highpass_window` mode) | Same YAMNet+MLP architecture, but audio is high-pass filtered at 300Hz AND cropped to a ~1s window around the single loudest cry bout before embedding extraction | 34% accuracy / 79.9% top-2 — worse than the unpreprocessed baseline. The windowing/cropping step specifically hurt, likely by discarding multi-bout rhythm context the mean-pooled YAMNet embedding was otherwise using across the full 4s clip. See the `highpass`-only mode below, which is what got promoted to the shipped model |

**Promoted to shipped model:** `train_cry_reason_preprocessed.py`'s
`highpass`-only mode (300Hz high-pass filter, no windowing) genuinely
improved results with balanced gains across all three classes, and was
merged into `train_cry_reason.py` at the repo root. The isolated A/B script
is kept here for reference on how that conclusion was reached.

## Takeaway

Across ~9 independent (feature representation × model family) combinations —
spanning linear models, tree ensembles, small MLPs, and small CNNs, and
spanning general-purpose embeddings, hand-engineered prosody, rhythm, and
voice-quality features — accuracy on the full 6-class cry-reason split
consistently ceilings around 25–30%, with `cold_hot` essentially never
learnable (0–6% recall almost everywhere). That convergence across so many
different methods is itself evidence the bottleneck is the **data/labels**,
not the feature representation or model choice — most plausibly because
several of these labels don't correspond to a consistent acoustic
difference in how the baby actually cries (e.g. "cold_hot" may be assigned
from room-temperature context rather than anything audible in the cry).

The 3-class collapse (`belly pain` / `hungry` / `fussy`) sidesteps this by
only keeping the two reasons that showed consistent, above-chance signal
across every experiment, and is the scheme the shipped model uses.

On top of the 3-class scheme, most further architecture/feature changes
(spectrogram CNNs, a focused belly-pain binary detector, cropping to a
single cry bout) also failed to improve on the plain YAMNet+MLP baseline —
reinforcing that model/feature sophistication wasn't the limiting factor.
What *did* help was a simple signal-processing step applied before feature
extraction: a 300Hz high-pass filter to remove low-frequency room noise,
which gave the first genuinely balanced result across all three classes
(rather than one class dominating) and is now part of the shipped
pipeline. That's a useful pattern for any future work here: prefer
cleaning up the input signal over adding model complexity, since the
latter has now failed to help in nearly every form tried.
