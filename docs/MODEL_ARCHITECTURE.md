# Model Architecture

## Overview

```
red/IR samples ─► validate fs & arrays ─► 10 s windows ─► per-channel preprocessing ─► SQI gate ─┐
                                                                                                │ reject → "re-measure"
                                          feature vector (≈80 features) ◄── beats / spectrum ◄───┘
                                                    │
                                     sklearn Pipeline: median impute → standardise → regressor
                                                    │
                                    per-window SBP, DBP ─► median over accepted windows ─► result
```

## Feature-based models (`src/rakshpulse/models/model_factory.py`)

Each one is a scikit-learn `Pipeline(SimpleImputer(median) → StandardScaler → model)` fitted
inside each training fold, so no statistics leak from test subjects.

| Model | Configuration | Hyper-parameters tuned (inner subject-grouped 3-fold CV, MAE) |
|---|---|---|
| Mean baseline | `DummyRegressor(mean)` | — |
| Demographics baseline | ridge on age, sex, height, weight | α |
| Linear regression | OLS | — |
| Ridge | L2 | α ∈ {1, 10, 100, 1000} |
| SVR | RBF kernel | C ∈ {1, 10, 30}, ε ∈ {1, 3} |
| Random forest | 300 trees | min_samples_leaf ∈ {5, 20}, max_features ∈ {0.3, 0.6} |
| XGBoost | hist, 400 rounds, subsample 0.7, colsample 0.6, λ = 5 | max_depth ∈ {2, 4}, learning rate ∈ {0.03, 0.1} |

The grids are small on purpose. There are only 148 independent labels (one cuff reading per
subject), and wide searches mostly fit noise in the validation folds.

## Raw-waveform model: 1D CNN (`src/rakshpulse/models/cnn.py`)

| Layer | Output (for 2×1000 input) |
|---|---|
| Conv1d(2→16, k7, s2) + BN + ReLU | 16×500 |
| Conv1d(16→16, k7, s2) + BN + ReLU | 16×250 |
| Conv1d(16→32, k5, s2) + BN + ReLU | 32×125 |
| Conv1d(32→32, k5, s2) + BN + ReLU | 32×63 |
| Conv1d(32→64, k3, s2) + BN + ReLU | 64×32 (receptive field ≈ 1.9 s, i.e. more than one beat) |
| Global average pool → Dropout(0.3) → Linear(64→2) | SBP, DBP (standardised) |

16,450 parameters for red+IR input (16,338 IR-only). Measured desktop CPU latency is about 0.5 ms per 10 s window. AdamW (lr 1e-3, weight
decay 1e-3), smooth-L1 loss, early stopping on a held-out **group of training subjects**,
and train-only augmentation (±10 % amplitude scale, circular time shift). Targets are
standardised with training-fold statistics.

**Recurrent and attention models (RakshPulse 3.0, `src/rakshpulse/models/sequence.py`).** The 2.0 note said an
LSTM should be revisited only with continuous beat-level references. Auricle provides those (CNAP), so 3.0 adds
four compact alternatives. All share `CNNRegressor`'s training loop (`CNNRegressor(in_ch, cfg, arch=...)`): same
subject-held-out early stopping, augmentation, multi-task SBP + DBP output and target standardisation, so
differences come from the architecture alone.

| `arch` | Structure (input 1×1000, 100 Hz) | Parameters |
|---|---|---:|
| `cnn` | BPNet above | 16,338 |
| `gru` | strided conv stem (3 layers, 1000 → 125 steps × 32 ch) → GRU(32) → last state → Linear | 14,434 |
| `cnn_lstm` | same stem → LSTM(32) → last state → Linear | 16,546 |
| `tcn` | same stem → 4 residual dilated conv blocks (dilation 1, 2, 4, 8) → mean pool → Linear | 20,770 |
| `transformer` | same stem → learned positions → 2 encoder layers (d = 32, 4 heads, FF 64) → mean pool → Linear | 29,186 |

The models are deliberately small: each fold trains on about 30 subjects. Results are in RESEARCH_FINDINGS.md V4
(`scripts/run_deep_benchmark.py`).

## Inference bundle (`src/rakshpulse/inference/`)

`ModelBundle` (joblib) contains the fitted SBP and DBP pipelines, the ordered feature list,
the required channels, a demographics flag, a version string, training-data provenance and
**the subject-wise CV error statistics**. `BPPredictor` reports those statistics as
population-level uncertainty. It does not invent per-reading confidence intervals, which the
data cannot support.

## Deployment footprint (measured, not assumed)

See `results/final_model.json` and `results/cnn_experiments.json`:
- Ridge bundle (PPG + demographics): 84 features, 170 parameters in total, 5.7 kB serialised (joblib).
  A full 30 s prediction (5 windows, including feature extraction) took 236 ms on a desktop CPU. Inference cost is
  dominated by feature extraction (filtering, peak detection, beat features), which is
  O(n) per window and well within microcontroller-class budgets in principle. It has
  **not** been ported or profiled on an ESP32.
- CNN: 16.4k parameters, so about 66 kB as float32 or about 16 kB int8. About 0.5 ms per window on a desktop CPU.

**ONNX / TFLite Micro.** Export is *not* done, by design. The measured results show no model
that beats the baseline in a way worth deploying. Porting format work should follow a model
that has shown value on behind-ear data. When that happens: export ridge as plain C
coefficients (no runtime needed), and the CNN via `torch.onnx.export` →
onnxruntime / TFLite (int8) for ESP32-S3.

## Population model + personalisation (Auricle, ear PPG)

```
ear PPG ─► 10 s windows ─► SQI gate ─► HR + pulse-shape features ─► population ridge ─► population SBP/DBP
                                                                                  │
            user's reference readings (cuff) ─► PersonalizationEngine (per user) ─┴─► personalised SBP/DBP
                                              └─► BaselineTracker (per user) ─────────► deviation from own baseline
```

- **Population model** (`models/auricle_population_ear_ridge.joblib`): ridge on 28 ear features (HR + 27
  HR-invariant shape descriptors), trained on all session-1 healthy-cohort windows that pass ear SQI. Ridge was chosen
  before the run (no model beat it earlier). On unseen subjects it is **no better than the population mean**
  (RESEARCH_FINDINGS §P1). It exists so the pipeline has an initial estimate and something to calibrate. It is not
  a meaningful absolute-BP estimator.
- **PersonalizationEngine** (`src/rakshpulse/personalization/engine.py`): personal = a + b·population.
  - `offset` (b = 1) needs only one reading.
  - `linear` uses a ridge-shrunk slope (towards 1), only with ≥ 4 readings spanning ≥ 10 mmHg.
  - Updates are recency-weighted (half-life 6 h by default) and use trusted reference pairs only, in time order.
    Each update first records the prequential residual (error of the calibration in force).
  - Updates that jump more than 25 mmHg or push the slope out of [0.25, 2] are flagged unstable. Every version
    can be rolled back. State is saved as JSON per user.
  - Recalibration is recommended when the calibration is older than 12 h, recent residuals exceed 12 mmHg,
    or the last update was unstable.
- **BaselineTracker** (`baseline.py`): mean, median, SD, rolling 5/10/20 means, trend per day and deviation of the
  user's own reference readings. The ACC/AHA category is a lookup for a single reading and is never applied to the
  history.
- **RealtimeEstimator** (`src/rakshpulse/inference/realtime.py`): hardware-independent. Accepts ear PPG (required),
  finger PPG (quality-reported only), sampling rate, timestamp, and optional engine and baseline.

## RakshPulse 4.0 components

- **Graded quality score** (`src/rakshpulse/preprocessing/quality_score.py`) puts every window into one of four
  classes and gives it a 0–100 score:
  - `good`;
  - `degraded` (passes the original gate but misses a pre-specified margin);
  - `unreliable` (fails the original gate, or is rail-clipped);
  - `missing` (non-finite samples, flat or detached sensor, no beats).

  The score combines eight components: template correlation, cardiac power ratio (SNR), IBI CV (irregular pulse),
  spike fraction (motion), baseline power ratio, broadband noise ratio, amplitude relative to the stream
  (perfusion/contact) and plateau fraction (saturation). Margins were fixed before evaluation.
- **Pulse-contour features** (`src/rakshpulse/features/physiological.py`) are computed on the median unit-beat
  template: dicrotic notch time/height (with an inflection fallback), diastolic peak, systolic–diastolic interval,
  inflection-point area, 10–90 % upstroke, diastolic decay constant, beat amplitude CV and beat-template similarity.
- **Cross-day normalisation** (`src/rakshpulse/evaluation/domain_shift.py`) offers six causal, label-free
  options: raw, session warm-up median, subject day-1 z-score, robust (median/MAD, winsorised), adaptive (time
  EWMA), and domain moment-matching from a day-2 warm-up. It also provides shift diagnostics.
- **HierarchicalCalibrator** (`src/rakshpulse/personalization/hierarchical.py`) models the residual of a base
  prediction as person offset + session offset + within-session drift (capped linear variogram).
  - Hyper-parameters come from training subjects by method of moments.
  - Readings from earlier days and from the current session enter one exact Gaussian posterior. This yields a
    correction and a predictive SD that grows with time since the last reading.
  - Using the training-mean BP as the base gives the matched no-PPG control (`bayes_reference_hier`).
- **Statistics** (`src/rakshpulse/evaluation/statistics.py`) work at subject level: paired bootstrap CI,
  sign-flip permutation p (exact for ≤ 16 subjects), Wilcoxon, Cohen's d_z, Holm and Benjamini–Hochberg,
  repeated-measures Bland–Altman, subject-weighted error distribution and ICC(3,1).
- **Export** (`src/rakshpulse/inference/export.py`) folds the linear bundle (impute → scale → ridge) into one
  weight vector and bias per target. It is written as JSON and a C99 header. No ML runtime is needed, so ONNX was
  not required (it is not installed here and was not tested).
