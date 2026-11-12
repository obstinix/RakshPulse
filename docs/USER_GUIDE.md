# RakshPulse 5.0 User Guide

RakshPulse is a **research prototype**. It is not a medical device, and no output of this software is a clinically
validated blood-pressure measurement. Simulated data is labelled everywhere and is never evidence about blood
pressure.

## 1. Installation

```bash
git clone https://github.com/obstinix/RakshPulse.git
cd RakshPulse
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e .[dev]                                  # core + pytest/ruff
pip install -e .[serial]                               # optional: USB-serial sensor input (pyserial)
pip install -e .[deep]                                 # optional: CNN / TCN / Transformer models (torch)
rakshpulse doctor                                      # environment check + simulated self-test
```

Python 3.10–3.12 is supported and tested in CI.

## 2. The workflow at a glance

```
record / simulate ─► dataset (CSV) ─► preprocess ─► features (+ cuff references) ─► train ─► calibrate ─► evaluate
                                                                                        └──► export (float32 C header)
live source ─────────────────────────────────────────────────────────────────────────────► infer (streaming)
```

| Command | What it does | Main outputs |
|---|---|---|
| `rakshpulse doctor` | checks Python, required/optional packages, runs a simulated self-test | exit code 0/1 |
| `rakshpulse simulate --out DS` | creates a **simulated** dataset (subjects × sessions, synthetic cuff readings) | `DS/sessions/...` |
| `rakshpulse record --out DS --subject S --session D1 --source sim\|replay\|serial` | records a session | raw, gaps, metadata, manifest |
| `rakshpulse reference --dataset DS --subject S --session D1 --time 120 --sbp 121 --dbp 79` | appends a cuff reading (append-only, re-hashed) | `reference_bp.csv` |
| `rakshpulse inspect --dataset DS [--validate]` | lists sessions; validates checksums, schema, ordering, ranges, references | table / report |
| `rakshpulse preprocess --dataset DS [--tag v1]` | windows + quality + features per session | `derived/windows*.csv`, `derived/segments*.csv` |
| `rakshpulse features --dataset DS --out table.csv` | one table; cuff readings associated with windows (same session, within `--tolerance`) | `table.csv` (`y_sbp`, `y_dbp`) |
| `rakshpulse train --table table.csv` | cross-validated training (subject / session / cross-day), versioned bundle | `models/<version>.joblib`, `_oof.csv`, `registry.csv` |
| `rakshpulse calibrate --oof <version>_oof.csv` | compares calibration strategies leakage-safely; fits hierarchical parameters | metrics CSV, `--params-out` CSV |
| `rakshpulse evaluate --model M --table T` | evaluates a saved model on new data **without refitting** | metrics with subject-bootstrap CI and Bland–Altman |
| `rakshpulse infer --model M --source ...` | streaming inference with quality gating, uncertainty and optional calibration | predictions CSV |
| `rakshpulse export --model M --out DIR` | float32 JSON + C99 header of a linear model, with an equivalence check | `.json`, `.h` |

Every command has `-h`. Errors are printed as `error: ...` with exit code 2.

## 3. Recording

```bash
# simulated sensor (software testing; flagged sensor_type=simulator)
rakshpulse record --out data/mystudy --subject P001 --session D1 --source sim --duration 300

# replay an existing raw_ppg.csv as if it were live
rakshpulse record --out data/mystudy --subject P001 --session D1-replay --source replay --input path/raw_ppg.csv

# MAX30102 behind a USB-serial bridge (see HARDWARE_INTEGRATION.md for the firmware contract)
rakshpulse record --out data/mystudy --subject P001 --session D1 --source serial --port COM5 --baud 115200 \
    --fs 100 --fields sample_index,red,ir --sensor-config '{"sample_rate_hz": 100, "pulse_width_us": 411}' \
    --placement behind_ear --contact "elastic band, moderate"
```

- A session folder is created **exclusively**. Recording into an existing subject/session fails, and nothing is
  overwritten.
- Missing samples (sample-counter gaps) are logged to `gaps.csv`.
- Ctrl-C stops a live recording cleanly: files are closed, hashed and registered.

## 4. Training guide

```bash
rakshpulse preprocess --dataset data/mystudy
rakshpulse features   --dataset data/mystudy --out tables/mystudy.csv --tolerance 30
rakshpulse train --table tables/mystudy.csv --model ridge --features ir_core --split subject --folds 5
rakshpulse train --table tables/mystudy.csv --model ridge --split cross_day --train-sessions D1   # day 1 → day 2
```

- **Splits** (`evaluation/splits.py`): `subject` keeps each subject entirely in train or test. `cross_day` is
  subject-disjoint and trains on `--train-sessions` only. `session` with seen subjects is available in the
  Python API. Every split raises if a subject or session leaks.
- **Features:** `ir_core` (HR + 27 HR-invariant shape features, the 3.0/4.0 definition), `all` (every numeric
  feature column), or an explicit comma-separated list.
- **Models:** `ridge`, `elasticnet`, `svr`, `random_forest`, `extra_trees`, `hist_gb`, `xgboost`, `lightgbm`,
  `pls`, `mean` (always train `mean` as the baseline to beat). Set `--hyperparams` explicitly, or use `--tune` for
  an inner `GroupKFold` search over training subjects only.
- **Versioning:** bundles are named `<name>-<UTC>-<data sha8>`. `models/registry.csv` records the configuration,
  CV metrics, conformal half-widths, the training-data SHA-256 and the git commit. Out-of-fold predictions are
  written next to the bundle for `calibrate`.
- **Raw-waveform models** (CNN, GRU, CNN-LSTM, TCN, Transformer) live in `models/cnn.py` / `models/sequence.py` and
  are benchmarked by `scripts/run_deep_benchmark.py`. 3.0 found no gain over simple baselines, so the CLI trains
  feature models only.
- `train` refuses an all-simulated table unless `--allow-simulated` is given (for software tests).

## 5. Calibration guide

```bash
rakshpulse calibrate --oof models/<version>_oof.csv --k 1 3 --spacing 60 --expiry 3600 --params-out calib_params.csv
```

Strategies (`personalization/strategies.py`):

| Strategy | Uses PPG | Uses cuff | Notes |
|---|:-:|:-:|---|
| `population` | – | – | training-mean BP: the floor every method must beat |
| `ppg_only` | ✓ | – | the PPG model, uncalibrated |
| `cuff_last` / `cuff_mean` | – | ✓ | carry the last / mean reading |
| `cuff_hierarchical` | – | ✓ | hierarchical Bayesian shrinkage; **the matched no-PPG control** |
| `combined` | ✓ | ✓ | the same hierarchical model on the PPG prediction |
| `subject_linear` | ✓ | ✓ | `a + b·pred` with the slope shrunk towards 1 (≥ 2 readings) |

Rules that are enforced in code:
- calibration readings, and windows within `exclude_s` of a reading, are **never** scored;
- evaluation windows come after the last reading of their session;
- calibration fit (error at the readings, in-sample) and independent prediction error are reported in separate
  columns;
- `CalibrationPolicy` expires a calibration by age (`--expiry`), by predictive half-width, or when a new reading's
  residual is large. A reading from another session does not validate the current one unless
  `allow_cross_session` is set.

**Claim PPG value only when `combined` beats `cuff_hierarchical`.** Beating `cuff_last` alone can come purely
from statistical shrinkage (RESEARCH_FINDINGS 3.0/4.0).

## 6. Evaluation

```bash
rakshpulse evaluate --model models/<version>.joblib --table tables/new_site.csv --out eval.csv
```

The report gives MAE, RMSE, mean error, SD of error, Pearson r, subject-level MAE with a subject-bootstrap 95 % CI,
repeated-measures Bland–Altman limits, and a paired test against an *oracle* constant (the evaluation set's own
mean; it is optimistic for the baseline). For multiple comparisons, use `evaluation/statistics.py` (Holm / BH,
sign-flip permutation, Cohen's d_z). The subject is always the unit of analysis.

## 7. Streaming inference

```bash
rakshpulse infer --model models/<version>.joblib --source serial --port COM5 --fs 100 --hop 5 \
    --calibration-params calib_params.csv --reference 120,124,81 --out live_predictions.csv
```

- Each prediction row has a timestamp, status (`usable`, `degraded`, `unusable`, `missing`, `warming_up` or
  `error`), reasons, quality score, the window and aggregated SBP/DBP, a 90 % interval, the estimate type
  (`population` / `calibrated`), calibration validity and latency.
- `--prefix-map ir_:ear_` runs the 4.0 Auricle ear bundle on IR input. That crosses sensor and site domains and
  is **not** validated.

## 8. Export

```bash
rakshpulse export --model models/<version>.joblib --out export/ --table tables/mystudy.csv
```

This writes a float32 JSON file and a C99 header (`<name>_predict(const float *x, float *sbp, float *dbp)`) and
checks equivalence with scikit-learn. float32 is the default and the recommendation. 4.0 showed that naive int8
quantisation introduces errors of several mmHg.

## 9. Python API

```python
from rakshpulse.acquisition.simulator import SimConfig
from rakshpulse.acquisition.sources import SimulatedSensor
from rakshpulse.acquisition.ingest import Recorder
from rakshpulse.datastore.store import DatasetStore, SessionMetadata
from rakshpulse.preprocessing.pipeline import process_recording, PipelineConfig

store = DatasetStore("data/demo")
with store.create_session(SessionMetadata("SIM1", "D1", 100.0, sensor_type="simulator")) as w:
    rec = Recorder(SimulatedSensor(SimConfig(seed=1), 120), [w.write_raw])
    rec.run()
    w.add_gaps(rec.sync.gap_table())
windows, segments = process_recording(store.raw("SIM1", "D1", chunksize=50_000), PipelineConfig(), "SIM1", "D1")
```

See docs/DATA_COLLECTION_PROTOCOL.md for the dataset format and recording protocol, and
docs/HARDWARE_INTEGRATION.md for connecting a MAX30102.
