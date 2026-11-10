# RakshPulse ML

Research pipeline for **cuffless blood-pressure estimation from red/IR PPG**. It supports
RakshPulse, a proposed behind-the-ear BP monitor for military personnel in field and
high-altitude settings, built around the MAX30102 optical sensor.

> **Not a medical device.** Nothing here is clinically validated, and nothing has been tested
> on behind-ear MAX30102 recordings. Every result below comes from public **fingertip-type**
> datasets. Read [Current limitations](#current-limitations) before using any number.

## Status at a glance

| Component | Status |
|---|---|
| Requirements extracted from project PDFs | ✅ Implemented ([docs/PROJECT_REQUIREMENTS.md](docs/PROJECT_REQUIREMENTS.md)) |
| Dataset survey (16 datasets, weighted scoring) | ✅ Implemented ([docs/DATASET_RESEARCH.md](docs/DATASET_RESEARCH.md)) |
| Dataset download, checksums, provenance, validation | ✅ Implemented (3 datasets, ≈650 MB) |
| Preprocessing + signal-quality gate + features | ✅ Implemented and unit-tested |
| Baseline and advanced models, subject-wise nested CV | ✅ Implemented. Results below |
| 1D CNN (raw waveform) | ✅ Experimental |
| ECG pulse-arrival-time study | 🧪 Auxiliary research only (no ECG on MAX30102) |
| Inference API with quality-based rejection | ✅ Implemented (proposed interface) |
| Tests + CI (ruff, pytest, Python 3.10–3.12) | ✅ Implemented |
| Finger vs earlobe study (Auricle, 41 subjects) | ✅ Implemented ([MODEL_EVALUATION §F](docs/MODEL_EVALUATION.md)) |
| Cross-day calibration study (Auricle, 36 subjects × 2 days) | ✅ Completed and independently audited ([RESEARCH_FINDINGS](docs/RESEARCH_FINDINGS.md)) |
| Standardised Auricle dataset (CSV tables + NPZ waveforms, 30,841 windows) | ✅ Built, validated (19 checks), published ([data/README.md](data/README.md)) |
| Population model, personalisation engine, baseline tracker, adaptive calibration | ✅ Implemented, tested, evaluated ([RESEARCH_FINDINGS](docs/RESEARCH_FINDINGS.md#population-model-and-personalisation-standardised-dataset-dataauricle)) |
| Real-time estimation interface (ear PPG + optional calibration state) | ✅ Implemented (hardware-independent; research use) |
| Behind-ear validation | ❌ **Not done.** No behind-ear data exists yet ([docs/DOMAIN_ADAPTATION.md](docs/DOMAIN_ADAPTATION.md)) |
| ESP32 / ONNX / TFLite deployment | ❌ Planned. Footprint measured only |
| Clinical validation (ISO 81060-2 / IEEE 1708) | ❌ Not started |

## Key findings (honest summary)

1. **No public dataset has behind-ear PPG with BP labels.** This is a research gap. RakshPulse
   must collect its own reference-labelled MAX30102 recordings.
2. On 148 healthy young adults (Guilin dual-wavelength dataset, subject-independent 5-fold
   nested CV), **PPG-only models did not beat the predict-the-mean baseline for SBP**
   (best MAE 11.1 vs 10.8 mmHg). For DBP they did slightly better (6.7 vs 7.6 mmHg, r ≈ 0.4).
3. Most of the predictable SBP variance came from **demographics** (sex, height, weight). In a
   paired subject-level bootstrap, adding PPG to demographics improved **DBP** reliably
   (6.18 vs 7.28 mmHg MAE, 95 % CI of the difference [−1.82, −0.31]). For **SBP** there was no
   detectable gain (8.59 vs 9.21, CI [−1.53, +0.30]). Bland–Altman limits of agreement are about
   ±22 mmHg for SBP.
4. **Red + IR gave no benefit over IR or red alone** on this data.
5. All models **regress to the mean**: the 5 hypertensive subjects (SBP ≥ 140) were
   underestimated by about 32–36 mmHg. A screening device built on these models would miss hypertension.
6. **Models trained on young adults failed on older external cohorts** (Foggia, PPG-BP), with
   −7 to −15 mmHg bias. Population-level, calibration-free BP from PPG did not generalise.
7. An auxiliary ECG experiment found that **pulse arrival time** (timing) carries SBP signal
   (r = −0.35) that single-site PPG shape lacks. A two-site optical *pulse transit time* design is
   worth evaluating. ECG is not available on the MAX30102.
8. **Earlobe evidence (Auricle, 41 healthy subjects, simultaneous finger + earlobe PPG, continuous BP):**
   PPG tracks within-session SBP changes slightly better than heart rate (ear r 0.28 vs 0.19, finger r 0.34 vs 0.28,
   after removing rail-clipped windows). With one-point
   calibration, earlobe SBP error beat carry-forward by about 1.2 mmHg (95 % CI [−2.08, −0.34]), but absolute
   MAE was still about 14 mmHg and DBP showed no benefit. **Finger-trained models lost most tracking at the
   ear (r 0.34 → ≤ 0.19)**, so a behind-ear model needs behind-ear training data.
9. **Cross-day (36 subjects, two recording days, audited):** a day-1 calibration does **not** hold on day 2. The model
   does not follow day-to-day resting-BP shifts (slope 0.02, 95 % CI [−0.04, 0.09]), calibration offsets drift by
   9–12 mmHg, and day-2 SBP error with day-1 calibration (ear 16.2 mmHg MAE) is no better than reporting the
   calibration value. Within a session, ear pulse shape does carry heart-rate-independent SBP trend information.
   See [docs/RESEARCH_FINDINGS.md](docs/RESEARCH_FINDINGS.md).
10. **Population vs personalisation (standardised dataset):** on unseen subjects no population model beat predicting
    the training-mean BP (0 of 64 paired tests better). Personal calibration improves day-2 errors only through
    new cuff readings: a PPG-free tracker of the same readings did statistically as well, and with a reading
    every 2 min the PPG engine never beat "repeat the last reading". PPG added no measurable value beyond the
    references in these data.
11. The realistic path forward is **per-user cuff calibration plus behind-ear data collection**.
   It is not a bigger model on fingertip data.

## Installation

```bash
git clone https://github.com/obstinix/RakshPulse.git && cd RakshPulse
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"                                 # core + pytest + ruff
pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional, for the CNN
```

## Reproduce everything

```bash
# 1. data (≈650 MB download, ≈1.6 GB extracted); see data/README.md
python -m rakshpulse.data.download guilin_dualwave foggia_multiwave ppg_bp
python scripts/extract_datasets.py
python -m rakshpulse.data.dataset_validator
python -m rakshpulse.data.build guilin_dualwave foggia_multiwave ppg_bp guilin_unlabelled
python -m rakshpulse.data.metadata

# 2. experiments (≈10 min on an 8-core CPU)
python scripts/run_experiments.py     # feature models, channel ablation, external tests
python scripts/run_cnn.py             # 1D CNN
python scripts/run_ecg_aux.py         # auxiliary ECG/PAT study

# Auricle finger vs earlobe (≈3 GB+ download, throttled; healthy cohort first)
python -m rakshpulse.data.physionet auricle --via-zip --jobs 3
python -m rakshpulse.data.build_auricle
python scripts/run_auricle.py
python scripts/visualize_auricle.py
python scripts/run_cross_day.py                                   # all models, 2,000 subject-bootstrap draws
python scripts/run_cross_day.py --models ridge --no-rail-gate --tag _no_rail_gate
python scripts/compare_rail_gate.py
python -m rakshpulse.data.physionet auricle --verify               # acquisition manifest
python scripts/audit_cross_day.py                                 # independent audit (must pass)
python scripts/make_cross_day_figures.py && python scripts/write_findings.py

# or everything in order:
python scripts/reproduce.py

# standardised dataset + personalisation
python -m rakshpulse.data build && python -m rakshpulse.data validate && python -m rakshpulse.data summary
python scripts/run_personalization.py
python -m rakshpulse.data.release fetch waveforms   # or fetch the published NPZ instead of building
python scripts/compare_models.py      # paired bootstrap CIs
python scripts/visualize_ppg.py       # figures in docs/figures/

# 3. reference inference bundle
python scripts/train_final.py --feature-set ppg_ir_red+demo --model ridge

# 4. tests
pytest
```

All random seeds are fixed (CV seeds 0, 1, 2; model seeds 0). Results are written to `results/`.

## Pipeline

1. **Preprocessing** ([docs/DATA_PREPROCESSING.md](docs/DATA_PREPROCESSING.md)): DC removal →
   anti-aliased resampling to 100 Hz → Hampel despike → 0.5–10 Hz zero-phase Butterworth →
   orientation → per-window z-score → 10 s windows → beat segmentation (adaptive peaks,
   intersecting-tangent feet).
2. **Signal-quality gate:** beat count, HR plausibility, RR regularity, template correlation,
   cardiac-band power, ADC flatline. Failing windows are rejected, never repaired.
3. **Features:** about 80 per window (morphology, APG, HRV, spectral, red/IR cross-channel).
4. **Models** ([docs/MODEL_ARCHITECTURE.md](docs/MODEL_ARCHITECTURE.md)): mean and demographics
   baselines, linear, ridge, SVR, random forest, XGBoost, and a 1D CNN of 16.4k parameters.
5. **Evaluation** ([docs/MODEL_EVALUATION.md](docs/MODEL_EVALUATION.md)): subject-independent
   nested CV, subject-level metrics (MAE, RMSE, R², bias, SD, Bland–Altman, per-BP-range),
   external datasets, paired bootstrap.

## Inference

```python
from rakshpulse.inference.predictor import BPPredictor

predictor = BPPredictor.from_file("models/rakshpulse_bp_v0.joblib")
result = predictor.predict(red_signal=red, ir_signal=ir, sampling_rate=100,
                           demographics={"age": 25, "sex": 1, "height_cm": 175, "weight_kg": 70})
if result.accepted:
    print(result.systolic_bp, result.diastolic_bp, result.uncertainty)
else:
    print("re-measure:", result.rejection_reasons)
```

It rejects the measurement (no BP returned) when signal quality is insufficient, a required
channel is missing, or the recording is shorter than 10 s. See
[docs/HARDWARE_INTEGRATION.md](docs/HARDWARE_INTEGRATION.md).

## Personalised, real-time estimation

```python
from rakshpulse.inference.realtime import RealtimeEstimator
from rakshpulse.personalization import BaselineTracker, PersonalizationEngine, RefPoint

est = RealtimeEstimator.from_file("models/auricle_population_ear_ridge.joblib")   # ear PPG, ridge, HR + shape
first = est.estimate(ear_ppg=ear, sampling_rate=1000, timestamp=t0)               # population estimate
engine = PersonalizationEngine("user-1")                                           # one engine per user
engine.calibrate([RefPoint(t0, first.population_sbp, first.population_dbp, cuff_sbp, cuff_dbp)])
baseline = BaselineTracker("user-1"); baseline.add(t0, cuff_sbp, cuff_dbp)
later = est.estimate(ear_ppg=ear2, sampling_rate=1000, timestamp=t1, engine=engine, baseline=baseline)
later.sbp, later.dbp, later.estimate_type          # "personalised"
later.recalibration_recommended, later.recalibration_reasons
engine.update(RefPoint(t2, p_sbp, p_dbp, cuff2_sbp, cuff2_dbp))   # adaptive update; engine.rollback() undoes it
engine.save("state/user-1.json")
```

Poor or invalid signals return `accepted=False` and no BP. The uncertainty fields are research validation
errors, not clinical confidence intervals.

## Current limitations

- **Site mismatch:** the training data is fingertip-type PPG. Behind-ear performance is unknown.
- **Labels:** one oscillometric cuff reading per subject. Windows are pseudo-replicates, so the
  effective sample size is 148.
- **Population:** healthy adults aged 18–36 with a narrow BP range (SBP IQR 105–124) and few
  hypertensive subjects. That is not enough to show BP *tracking*.
- **Generalisation:** external tests show large bias in older cohorts.
- **Motion:** no accelerometer in the data or on the MAX30102. Motion robustness is untested.
- **Standards:** ISO 81060-2 / BHS statistics are reported for orientation only. Compliance is
  **not** claimed. See [docs/MODEL_EVALUATION.md](docs/MODEL_EVALUATION.md#validation-requirements-before-any-real-world-use).

## Repository layout

```
src/rakshpulse/{data,preprocessing,features,models,evaluation,inference,personalization}
scripts/      reproducible experiment entry points
tests/        pytest suite (synthetic fixtures; no dataset needed)
notebooks/    01_ppg_exploration.ipynb
docs/         requirements, dataset research, preprocessing, models, evaluation, domain adaptation, hardware
data/auricle/ populated Auricle CSV dataset (waveform NPZ + raw WFDB on GitHub Releases)
data/         other raw data git-ignored; provenance/manifests committed
results/      metrics CSV/JSON (committed); large per-window prediction caches git-ignored
```

## Data licences and attribution

Guilin dual-wave (CC BY 4.0), Foggia multiwavelength (CC BY 4.0), PPG-BP (CC0) and Auricle (ODC-By 1.0;
redistributed with attribution on this repository's Releases). Required
citations are in [docs/DATASET_ACQUISITION.md](docs/DATASET_ACQUISITION.md). Code is released
under CC0 1.0 (see LICENSE).
