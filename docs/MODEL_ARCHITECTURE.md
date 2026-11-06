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

**LSTM: not run.** With one cuff label per subject and 148 subjects there is not enough
independent supervision for a recurrent model. The CNN, which has fewer parameters and a
stronger inductive bias, already failed to beat the mean baseline (MODEL_EVALUATION.md).
Adding an LSTM would add capacity without adding information. It should be revisited only
with beat-level continuous references (e.g. CNAP in Auricle) or far more subjects.

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
