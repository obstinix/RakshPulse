# Research Findings: Ear vs Finger PPG for Cuffless BP Tracking (Auricle, healthy cohort)

> Research results only. **RakshPulse is not a clinically validated BP monitor.** The reference is CNAP
> (a calibrated continuous finger-cuff estimate), the PPG sensors are Nonin transmissive clips (earlobe and
> finger), and none of this is behind-the-ear MAX30102 data.

## Question

Can ear PPG provide reliable, generalisable information for cuffless BP **tracking**, and exactly where
does it succeed or fail? Specifically: (a) absolute BP without calibration; (b) relative tracking within a
day; (c) whether a calibration made on one day still holds on another day.

## Methods (summary; full detail in MODEL_EVALUATION.md §F and `src/rakshpulse/evaluation/auricle_study.py`)

- **Data:** Auricle v1.0.0 (PhysioNet), healthy cohort. Sessions 1 and 2 were recorded on different days.
  Every file was verified against PhysioNet SHA-256 sums (`data/metadata/auricle_manifest.json`).
- **Windows:** 10 s, 5 s step. Labelled only when the CNAP reliability mask is valid for the whole window.
  A window passes only if its site's SQI passes and fewer than 1 % of its samples are on the ±10 V rail.
- **Feature sets:** `hr_only` (heart rate); `shape_only` (dimensionless, beat-duration-normalised morphology
  that does not change merely because heart rate changes); `hr_shape`; `ppg_all`.
- **Population models:** trained on **session-1 windows of training subjects only** (5 subject-grouped folds),
  tuned with inner GroupKFold on those subjects, then applied to held-out subjects' sessions 1 and 2.
- **Calibration** (from the first 3 rest windows ≈ 20 s of session 1):
  - *carry-forward*: report the calibration BP (no PPG; the baseline)
  - *offset*: 1-point offset on the population prediction
  - *linear*: per-subject linear map fitted on all of session 1
  - *personal model*: per-subject ridge fitted on session 1 only
- **Scopes:**
  - *within_day*: later windows of session 1
  - *within_day_s2*: session 2 with its own rest calibration (day-2 control)
  - *cross_day*: session 2 using the session-1 calibration
- **Statistics:** subject-level bootstrap (2,000 draws) for every CI, and paired per-subject differences for
  every comparison. The model-selection criterion was fixed before the run: cross-day MAE rank.
- **Audit:** `scripts/audit_cross_day.py` independently re-derives the results (11 checks). The tables below are
  produced by `scripts/write_findings.py` only when the audit has no hard failures.

## Results (auto-generated from audited outputs)

<!-- BEGIN AUTO -->

### R1. Data and audit

- Healthy session-2 records verified: **37/37**; session-1: 42/42. Records with untrusted labels (no CNAP mask): HC_1_1.
- Subjects analysed: 42 (both sessions: 36); windows by session: {'1': 8458, '2': 7961}. Bootstrap: 2000 subject-level draws, seed 0.
- Independent audit (`results/cross_day/audit.json`): 1 PASS, 2 PASS, 3 PASS, 4 PASS, 5 PASS, 10 PASS, 6 PASS, 7 PASS, 8 PASS, 9 PASS, 11 PASS

### R2. Within-day vs cross-day, by calibration strategy (ridge, HR + shape features)

| Target | Site | Scope | Calibration | MAE | MAE 95% CI | RMSE | Bias | SD err | Within-subj Pearson | Within-subj Spearman | n subj |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SBP | ear | within_day | carry_forward | 14.98 | [13.04, 17.04] | 19.16 | 5.47 | 18.36 | — | — | 38 |
| SBP | ear | within_day | uncalibrated | 14.96 | [12.87, 17.31] | 18.87 | -0.17 | 18.87 | 0.41 | 0.39 | 38 |
| SBP | ear | within_day | offset | 13.84 | [11.88, 16.03] | 18.06 | 3.76 | 17.67 | 0.41 | 0.39 | 38 |
| SBP | ear | within_day_s2 | carry_forward | 13.29 | [11.12, 15.64] | 17.64 | 1.80 | 17.55 | — | — | 29 |
| SBP | ear | within_day_s2 | uncalibrated | 15.40 | [12.54, 18.47] | 19.55 | -3.65 | 19.21 | 0.40 | 0.35 | 29 |
| SBP | ear | within_day_s2 | offset | 12.25 | [10.29, 14.52] | 16.20 | 0.09 | 16.20 | 0.40 | 0.35 | 29 |
| SBP | ear | cross_day | carry_forward | 16.52 | [13.84, 19.11] | 20.91 | 1.65 | 20.85 | — | — | 31 |
| SBP | ear | cross_day | uncalibrated | 15.10 | [12.52, 17.87] | 19.21 | -3.10 | 18.96 | 0.38 | 0.33 | 31 |
| SBP | ear | cross_day | offset | 16.14 | [13.51, 18.97] | 20.24 | -0.20 | 20.24 | 0.38 | 0.33 | 31 |
| SBP | ear | cross_day | linear | 14.22 | [11.75, 16.94] | 18.46 | -4.77 | 17.83 | 0.38 | 0.33 | 31 |
| SBP | ear | cross_day | personal model (day-1 fit) | 15.01 | [12.32, 18.11] | 20.02 | -4.08 | 19.61 | 0.33 | 0.32 | 32 |
| SBP | finger | within_day | carry_forward | 15.84 | [13.76, 18.18] | 20.10 | 5.09 | 19.45 | — | — | 40 |
| SBP | finger | within_day | uncalibrated | 15.57 | [13.34, 17.89] | 19.89 | -0.10 | 19.90 | 0.30 | 0.30 | 40 |
| SBP | finger | within_day | offset | 15.45 | [13.33, 17.66] | 19.65 | 4.74 | 19.07 | 0.30 | 0.30 | 40 |
| SBP | finger | within_day_s2 | carry_forward | 12.70 | [10.54, 15.07] | 16.94 | 1.74 | 16.85 | — | — | 33 |
| SBP | finger | within_day_s2 | uncalibrated | 16.30 | [13.10, 20.11] | 20.79 | -1.80 | 20.72 | 0.24 | 0.17 | 33 |
| SBP | finger | within_day_s2 | offset | 12.19 | [10.28, 14.27] | 16.04 | 0.32 | 16.04 | 0.24 | 0.17 | 33 |
| SBP | finger | cross_day | carry_forward | 15.59 | [13.11, 18.15] | 19.99 | 1.70 | 19.92 | — | — | 35 |
| SBP | finger | cross_day | uncalibrated | 15.94 | [12.83, 19.30] | 20.35 | -1.16 | 20.32 | 0.25 | 0.18 | 35 |
| SBP | finger | cross_day | offset | 15.54 | [12.98, 18.22] | 19.85 | 1.66 | 19.78 | 0.25 | 0.18 | 35 |
| SBP | finger | cross_day | linear | 13.97 | [11.82, 16.21] | 18.39 | -3.23 | 18.11 | 0.19 | 0.13 | 35 |
| SBP | finger | cross_day | personal model (day-1 fit) | 13.61 | [11.23, 16.14] | 18.74 | -2.03 | 18.64 | 0.36 | 0.31 | 35 |
| DBP | ear | within_day | carry_forward | 8.94 | [7.36, 10.75] | 12.66 | -0.97 | 12.62 | — | — | 38 |
| DBP | ear | within_day | uncalibrated | 10.94 | [8.91, 12.98] | 15.10 | -0.53 | 15.09 | 0.19 | 0.19 | 38 |
| DBP | ear | within_day | offset | 9.50 | [8.02, 11.17] | 13.56 | -2.20 | 13.38 | 0.19 | 0.19 | 38 |
| DBP | ear | within_day_s2 | carry_forward | 8.87 | [7.16, 10.83] | 11.69 | -2.57 | 11.40 | — | — | 29 |
| DBP | ear | within_day_s2 | uncalibrated | 9.97 | [7.82, 12.83] | 13.19 | -1.92 | 13.05 | 0.11 | 0.12 | 29 |
| DBP | ear | within_day_s2 | offset | 9.04 | [7.35, 11.00] | 11.81 | -2.96 | 11.44 | 0.11 | 0.12 | 29 |
| DBP | ear | cross_day | carry_forward | 11.59 | [8.99, 14.49] | 14.86 | -1.41 | 14.80 | — | — | 31 |
| DBP | ear | cross_day | uncalibrated | 9.96 | [8.02, 12.40] | 13.10 | -1.30 | 13.04 | 0.13 | 0.12 | 31 |
| DBP | ear | cross_day | offset | 12.82 | [10.17, 15.80] | 16.06 | -2.85 | 15.80 | 0.13 | 0.12 | 31 |
| DBP | ear | cross_day | linear | 10.50 | [8.25, 13.24] | 13.98 | -1.68 | 13.88 | 0.14 | 0.12 | 31 |
| DBP | ear | cross_day | personal model (day-1 fit) | 11.90 | [9.12, 15.30] | 16.18 | -2.18 | 16.04 | 0.27 | 0.26 | 32 |
| DBP | finger | within_day | carry_forward | 9.24 | [7.70, 10.92] | 13.05 | -0.74 | 13.03 | — | — | 40 |
| DBP | finger | within_day | uncalibrated | 11.49 | [9.61, 13.56] | 15.32 | -0.35 | 15.31 | 0.05 | 0.04 | 40 |
| DBP | finger | within_day | offset | 9.84 | [8.25, 11.52] | 13.71 | -0.58 | 13.70 | 0.05 | 0.04 | 40 |
| DBP | finger | within_day_s2 | carry_forward | 8.75 | [7.15, 10.51] | 11.75 | -2.50 | 11.48 | — | — | 33 |
| DBP | finger | within_day_s2 | uncalibrated | 10.32 | [8.18, 12.89] | 13.54 | -0.66 | 13.52 | 0.04 | 0.01 | 33 |
| DBP | finger | within_day_s2 | offset | 9.35 | [7.80, 10.97] | 12.08 | -2.60 | 11.80 | 0.04 | 0.01 | 33 |
| DBP | finger | cross_day | carry_forward | 11.41 | [9.10, 14.12] | 14.66 | -0.69 | 14.65 | — | — | 35 |
| DBP | finger | cross_day | uncalibrated | 10.24 | [8.25, 12.51] | 13.37 | -0.12 | 13.37 | 0.06 | 0.04 | 35 |
| DBP | finger | cross_day | offset | 11.72 | [9.59, 14.15] | 14.80 | -1.15 | 14.75 | 0.06 | 0.04 | 35 |
| DBP | finger | cross_day | linear | 11.10 | [8.80, 13.79] | 15.01 | -0.13 | 15.01 | 0.06 | 0.04 | 35 |
| DBP | finger | cross_day | personal model (day-1 fit) | 12.22 | [9.35, 15.39] | 17.56 | 1.47 | 17.50 | 0.28 | 0.26 | 35 |

### R3. Heart rate vs pulse shape (ridge, 1-point offset calibration)

| Target | Site | Features | Scope | MAE | MAE 95% CI | Within-subj r | r 95% CI |
|---|---|---|---|---|---|---|---|
| SBP | ear | hr_only | within_day | 14.71 | [12.72, 16.89] | 0.28 | [0.16, 0.40] |
| SBP | ear | hr_only | cross_day | 16.17 | [13.72, 18.62] | 0.09 | [-0.13, 0.24] |
| SBP | ear | shape_only | within_day | 14.13 | [12.01, 16.48] | 0.43 | [0.35, 0.48] |
| SBP | ear | shape_only | cross_day | 17.46 | [14.71, 20.49] | 0.38 | [0.28, 0.53] |
| SBP | ear | hr_shape | within_day | 13.84 | [11.88, 16.03] | 0.41 | [0.27, 0.47] |
| SBP | ear | hr_shape | cross_day | 16.14 | [13.51, 18.97] | 0.38 | [0.26, 0.46] |
| SBP | ear | ppg_all | within_day | 13.66 | [11.76, 15.83] | 0.41 | [0.24, 0.48] |
| SBP | ear | ppg_all | cross_day | 15.76 | [13.26, 18.50] | 0.38 | [0.19, 0.49] |
| SBP | finger | hr_only | within_day | 15.55 | [13.34, 17.88] | 0.33 | [0.15, 0.41] |
| SBP | finger | hr_only | cross_day | 15.44 | [13.21, 17.83] | 0.17 | [-0.02, 0.28] |
| SBP | finger | shape_only | within_day | 15.38 | [13.32, 17.58] | 0.27 | [0.17, 0.37] |
| SBP | finger | shape_only | cross_day | 15.55 | [13.03, 18.33] | 0.21 | [0.15, 0.33] |
| SBP | finger | hr_shape | within_day | 15.45 | [13.33, 17.66] | 0.30 | [0.18, 0.40] |
| SBP | finger | hr_shape | cross_day | 15.54 | [12.98, 18.22] | 0.25 | [0.13, 0.36] |
| SBP | finger | ppg_all | within_day | 15.17 | [13.12, 17.45] | 0.33 | [0.20, 0.44] |
| SBP | finger | ppg_all | cross_day | 14.98 | [12.49, 17.68] | 0.35 | [0.25, 0.39] |
| DBP | ear | hr_only | within_day | 9.09 | [7.60, 10.81] | 0.18 | [0.09, 0.29] |
| DBP | ear | hr_only | cross_day | 11.69 | [9.29, 14.25] | 0.02 | [-0.11, 0.13] |
| DBP | ear | shape_only | within_day | 9.51 | [7.98, 11.26] | 0.17 | [0.09, 0.36] |
| DBP | ear | shape_only | cross_day | 12.84 | [10.12, 15.95] | 0.13 | [0.03, 0.24] |
| DBP | ear | hr_shape | within_day | 9.50 | [8.02, 11.17] | 0.19 | [0.11, 0.41] |
| DBP | ear | hr_shape | cross_day | 12.82 | [10.17, 15.80] | 0.13 | [-0.04, 0.23] |
| DBP | ear | ppg_all | within_day | 9.36 | [7.91, 11.00] | 0.19 | [0.13, 0.36] |
| DBP | ear | ppg_all | cross_day | 12.71 | [10.18, 15.52] | 0.16 | [0.04, 0.29] |
| DBP | finger | hr_only | within_day | 9.44 | [7.96, 11.08] | 0.14 | [0.04, 0.34] |
| DBP | finger | hr_only | cross_day | 11.26 | [9.10, 13.80] | -0.01 | [-0.12, 0.18] |
| DBP | finger | shape_only | within_day | 9.87 | [8.28, 11.54] | 0.03 | [-0.04, 0.26] |
| DBP | finger | shape_only | cross_day | 11.82 | [9.63, 14.32] | 0.05 | [-0.07, 0.17] |
| DBP | finger | hr_shape | within_day | 9.84 | [8.25, 11.52] | 0.05 | [-0.03, 0.29] |
| DBP | finger | hr_shape | cross_day | 11.72 | [9.59, 14.15] | 0.06 | [-0.07, 0.16] |
| DBP | finger | ppg_all | within_day | 9.79 | [8.22, 11.45] | 0.04 | [-0.05, 0.19] |
| DBP | finger | ppg_all | cross_day | 11.71 | [9.53, 14.22] | 0.06 | [-0.07, 0.13] |

### R4. Population models (HR + shape, cross-day, offset calibration)

Ranking criterion (fixed in advance): mean rank of cross-day MAE across the four site × target conditions, then worst rank, then within→cross-day gap. Correlation is not used for selection.

| model | cross_day_mae_mean | mean_rank | worst_rank | within_to_cross_day_gap |
|---|---|---|---|---|
| svr | 13.79 | 2.25 | 5.0 | 1.84 |
| elasticnet | 13.83 | 2.5 | 4.0 | 1.74 |
| extra_trees | 13.9 | 3.25 | 4.0 | 1.75 |
| ridge | 14.05 | 4.5 | 7.0 | 1.89 |
| xgboost | 14.46 | 4.75 | 6.0 | 1.87 |
| random_forest | 14.56 | 5.25 | 6.0 | 1.99 |
| hist_gb | 15.05 | 5.5 | 7.0 | 2.11 |

| comparison | mae_diff_a_minus_b | ci_low | ci_high | significant |
|---|---|---|---|---|
| ear sbp cross_day: elasticnet vs ridge (offset hr_shape) | 0.27 | -0.56 | 1.12 | False |
| ear sbp cross_day: svr vs ridge (offset hr_shape) | 0.14 | -0.68 | 0.95 | False |
| ear sbp cross_day: random_forest vs ridge (offset hr_shape) | 0.09 | -1.3 | 1.48 | False |
| ear sbp cross_day: extra_trees vs ridge (offset hr_shape) | -0.3 | -1.27 | 0.64 | False |
| ear sbp cross_day: hist_gb vs ridge (offset hr_shape) | -0.83 | -2.52 | 0.96 | False |
| ear sbp cross_day: xgboost vs ridge (offset hr_shape) | -0.61 | -1.86 | 0.61 | False |
| ear dbp cross_day: elasticnet vs ridge (offset hr_shape) | -0.22 | -0.62 | 0.11 | False |
| ear dbp cross_day: svr vs ridge (offset hr_shape) | -0.03 | -0.97 | 0.96 | False |
| ear dbp cross_day: random_forest vs ridge (offset hr_shape) | 0.67 | -0.45 | 1.84 | False |
| ear dbp cross_day: extra_trees vs ridge (offset hr_shape) | -0.06 | -0.72 | 0.65 | False |
| ear dbp cross_day: hist_gb vs ridge (offset hr_shape) | 1.59 | 0.1 | 3.25 | True |
| ear dbp cross_day: xgboost vs ridge (offset hr_shape) | 0.72 | -0.6 | 2.16 | False |
| finger sbp cross_day: elasticnet vs ridge (offset hr_shape) | -0.3 | -0.97 | 0.31 | False |
| finger sbp cross_day: svr vs ridge (offset hr_shape) | -0.62 | -1.55 | 0.34 | False |
| finger sbp cross_day: random_forest vs ridge (offset hr_shape) | -0.24 | -1.22 | 0.75 | False |
| finger sbp cross_day: extra_trees vs ridge (offset hr_shape) | -0.56 | -1.23 | 0.1 | False |
| finger sbp cross_day: hist_gb vs ridge (offset hr_shape) | 0.23 | -1.0 | 1.47 | False |
| finger sbp cross_day: xgboost vs ridge (offset hr_shape) | 0.04 | -0.97 | 1.05 | False |
| finger dbp cross_day: elasticnet vs ridge (offset hr_shape) | -0.22 | -0.63 | 0.2 | False |
| finger dbp cross_day: svr vs ridge (offset hr_shape) | -0.43 | -1.24 | 0.44 | False |
| finger dbp cross_day: random_forest vs ridge (offset hr_shape) | 0.53 | -0.89 | 2.39 | False |
| finger dbp cross_day: extra_trees vs ridge (offset hr_shape) | -0.1 | -1.11 | 1.08 | False |
| finger dbp cross_day: hist_gb vs ridge (offset hr_shape) | 1.45 | -0.59 | 4.32 | False |
| finger dbp cross_day: xgboost vs ridge (offset hr_shape) | 0.73 | -0.6 | 2.48 | False |

### R5. Key paired subject-level comparisons (Δ = MAE(a) − MAE(b); negative favours a)

| comparison | mae_diff_a_minus_b | ci_low | ci_high | significant |
|---|---|---|---|---|
| ear sbp within_day: offset hr_shape vs carry-forward | -1.03 | -1.82 | -0.25 | True |
| ear sbp within_day: offset hr_only vs carry-forward | -0.39 | -1.06 | 0.3 | False |
| ear sbp within_day: offset hr_shape vs offset hr_only | -0.63 | -1.32 | -0.01 | True |
| ear sbp within_day: offset shape_only vs offset hr_only | -0.29 | -1.21 | 0.54 | False |
| ear sbp cross_day: offset hr_shape vs carry-forward | -0.7 | -1.98 | 0.59 | False |
| ear sbp cross_day: offset hr_only vs carry-forward | -0.84 | -2.16 | 0.33 | False |
| ear sbp cross_day: offset hr_shape vs offset hr_only | 0.14 | -1.07 | 1.35 | False |
| ear sbp cross_day: offset shape_only vs offset hr_only | 1.76 | 0.07 | 3.54 | True |
| ear sbp cross_day: linear hr_shape vs offset hr_shape | -2.44 | -5.63 | 0.81 | False |
| ear sbp cross_day: personal hr_shape vs personal hr_only | 2.24 | 0.45 | 4.04 | True |
| ear sbp cross_day: personal hr_shape vs carry-forward | -1.37 | -4.68 | 2.24 | False |
| ear dbp within_day: offset hr_shape vs carry-forward | 0.44 | 0.01 | 0.91 | True |
| ear dbp within_day: offset hr_only vs carry-forward | 0.13 | -0.14 | 0.4 | False |
| ear dbp within_day: offset hr_shape vs offset hr_only | 0.31 | -0.08 | 0.71 | False |
| ear dbp within_day: offset shape_only vs offset hr_only | 0.3 | -0.12 | 0.75 | False |
| ear dbp cross_day: offset hr_shape vs carry-forward | 1.19 | 0.32 | 2.14 | True |
| ear dbp cross_day: offset hr_only vs carry-forward | -0.04 | -0.53 | 0.47 | False |
| ear dbp cross_day: offset hr_shape vs offset hr_only | 1.22 | 0.24 | 2.33 | True |
| ear dbp cross_day: offset shape_only vs offset hr_only | 1.2 | 0.17 | 2.38 | True |
| ear dbp cross_day: linear hr_shape vs offset hr_shape | -2.06 | -4.64 | 0.3 | False |
| ear dbp cross_day: personal hr_shape vs personal hr_only | 1.69 | 0.13 | 3.18 | True |
| ear dbp cross_day: personal hr_shape vs carry-forward | 0.82 | -2.07 | 3.93 | False |
| finger sbp within_day: offset hr_shape vs carry-forward | -0.57 | -1.59 | 0.41 | False |
| finger sbp within_day: offset hr_only vs carry-forward | -0.55 | -1.36 | 0.25 | False |
| finger sbp within_day: offset hr_shape vs offset hr_only | -0.01 | -0.53 | 0.5 | False |
| finger sbp within_day: offset shape_only vs offset hr_only | 0.02 | -0.47 | 0.51 | False |
| finger sbp cross_day: offset hr_shape vs carry-forward | -0.22 | -1.49 | 1.02 | False |
| finger sbp cross_day: offset hr_only vs carry-forward | -0.46 | -1.37 | 0.44 | False |
| finger sbp cross_day: offset hr_shape vs offset hr_only | 0.24 | -0.69 | 1.26 | False |
| finger sbp cross_day: offset shape_only vs offset hr_only | 0.28 | -0.6 | 1.2 | False |
| finger sbp cross_day: linear hr_shape vs offset hr_shape | -2.06 | -4.79 | 0.5 | False |
| finger sbp cross_day: personal hr_shape vs personal hr_only | 1.26 | -0.25 | 3.26 | False |
| finger sbp cross_day: personal hr_shape vs carry-forward | -1.69 | -4.83 | 1.49 | False |
| finger dbp within_day: offset hr_shape vs carry-forward | 0.47 | 0.08 | 0.89 | True |
| finger dbp within_day: offset hr_only vs carry-forward | 0.15 | -0.18 | 0.51 | False |
| finger dbp within_day: offset hr_shape vs offset hr_only | 0.31 | 0.04 | 0.62 | True |
| finger dbp within_day: offset shape_only vs offset hr_only | 0.34 | 0.05 | 0.69 | True |
| finger dbp cross_day: offset hr_shape vs carry-forward | 0.37 | -0.4 | 1.18 | False |
| finger dbp cross_day: offset hr_only vs carry-forward | -0.07 | -0.52 | 0.42 | False |
| finger dbp cross_day: offset hr_shape vs offset hr_only | 0.44 | -0.2 | 1.08 | False |
| finger dbp cross_day: offset shape_only vs offset hr_only | 0.53 | -0.14 | 1.23 | False |
| finger dbp cross_day: linear hr_shape vs offset hr_shape | -0.65 | -3.14 | 1.56 | False |
| finger dbp cross_day: personal hr_shape vs personal hr_only | 1.7 | 0.45 | 3.21 | True |
| finger dbp cross_day: personal hr_shape vs carry-forward | 0.81 | -1.8 | 3.65 | False |
| ear sbp cross_day: finger-trained vs ear-trained (offset) | 0.6 | -1.73 | 2.55 | False |
| ear sbp cross_day: finger-trained vs ear-trained (linear) | -0.17 | -0.66 | 0.41 | False |
| sbp cross_day: ear vs finger (offset hr_shape, each site's own windows) | 0.65 | -1.22 | 2.61 | False |
| ear dbp cross_day: finger-trained vs ear-trained (offset) | -0.91 | -1.92 | -0.0 | True |
| ear dbp cross_day: finger-trained vs ear-trained (linear) | -0.28 | -0.75 | 0.16 | False |
| dbp cross_day: ear vs finger (offset hr_shape, each site's own windows) | 0.55 | -0.48 | 1.66 | False |

### R6. Day-to-day resting BP shift and calibration stability

| Site | Target | n | True rest shift mean | SD | Slope pred-shift~true | Slope 95% CI | r | r 95% CI | Min detectable |r| | Offset day1~day2 r | r 95% CI | Mean |offset change| |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ear | SBP | 29 | 4.69 | 15.40 | 0.02 | [-0.04, 0.09] | 0.10 | [-0.28, 0.38] | 0.50 | 0.42 | [0.07, 0.70] | 11.95 |
| ear | DBP | 29 | 2.94 | 11.65 | -0.03 | [-0.15, 0.05] | -0.13 | [-0.45, 0.22] | 0.50 | 0.33 | [0.07, 0.64] | 9.62 |
| finger | SBP | 33 | 3.10 | 16.40 | -0.01 | [-0.11, 0.10] | -0.04 | [-0.43, 0.40] | 0.47 | 0.37 | [-0.01, 0.69] | 12.46 |
| finger | DBP | 33 | 1.25 | 11.74 | 0.02 | [-0.07, 0.09] | 0.12 | [-0.27, 0.44] | 0.47 | 0.44 | [0.22, 0.69] | 8.81 |

### R7. Amplifier rail-clipping gate ablation (ridge, HR + shape, offset)

Finger windows clipped: 8.1 %; ear: 0.2 %. Subjects with any clipped finger window: 29; with > 25 % clipped: 5.

| Site | Target | Scope | MAE no gate | MAE gated | Δ | r no gate | r gated | Windows removed |
|---|---|---|---|---|---|---|---|---|
| ear | SBP | cross_day | 16.14 | 16.14 | 0.00 | 0.38 | 0.38 | 0 |
| ear | SBP | within_day | 13.84 | 13.84 | 0.00 | 0.41 | 0.41 | 0 |
| ear | SBP | within_day_s2 | 12.25 | 12.25 | 0.00 | 0.41 | 0.41 | 0 |
| ear | DBP | cross_day | 12.82 | 12.82 | 0.00 | 0.13 | 0.13 | 0 |
| ear | DBP | within_day | 9.50 | 9.50 | 0.00 | 0.19 | 0.19 | 0 |
| ear | DBP | within_day_s2 | 9.04 | 9.04 | 0.00 | 0.11 | 0.11 | 0 |
| finger | SBP | cross_day | 15.40 | 15.54 | 0.15 | 0.26 | 0.25 | 283 |
| finger | SBP | within_day | 14.97 | 15.45 | 0.48 | 0.33 | 0.30 | 490 |
| finger | SBP | within_day_s2 | 12.30 | 12.19 | -0.12 | 0.28 | 0.24 | 277 |
| finger | DBP | cross_day | 11.73 | 11.72 | -0.02 | 0.04 | 0.06 | 283 |
| finger | DBP | within_day | 9.62 | 9.84 | 0.22 | 0.04 | 0.05 | 490 |
| finger | DBP | within_day_s2 | 9.61 | 9.36 | -0.25 | 0.06 | 0.04 | 277 |

### R8. Confounds and subject structure (SBP)

| Site | Within-session r(SBP, HR) | Partial r(SBP, shape pred | HR) | 95% CI | Between-subject share of pred. variance | r(subject-mean pred, truth) |
|---|---|---|---|---|---|
| ear | 0.19 | 0.22 | [0.16, 0.31] | 0.39 | 0.14 |
| finger | 0.23 | 0.11 | [0.02, 0.17] | 0.26 | -0.03 |

<!-- END AUTO -->

## Interpretation (from the audited results above)

> **Correction (2026-11-11).** The standardised-dataset work found 3,458 ear windows from flat or disconnected
> sensors (raw SD < 0.003 mV). 31 of them had passed the SQI gate and entered the labelled analysis set. A
> signal-presence gate now rejects them, and every analysis on this page was rerun. MAEs moved by ≤ 0.62 mmHg
> and tracking r by ≤ 0.09. One earlier statement no longer holds: ElasticNet is not significantly better than
> ridge for ear DBP. The audit passes 11/11.

**1. Relative tracking within a session: modest but real, and better at the ear.**
Within session 1, ear PPG (HR + shape, ridge) tracks SBP changes with a median within-subject r of 0.41 (finger 0.30).
With a 20 s rest calibration it beats carry-forward by 1.03 mmHg (95 % CI [−1.82, −0.25]). Pulse shape carries
SBP information **beyond heart rate at the ear**: within-session partial r(SBP, shape prediction | HR) = 0.23
[0.16, 0.31], against 0.12 [0.02, 0.17] at the finger. At the ear, HR + shape beats HR-only by 0.63 mmHg [−1.32, −0.01].
The stressors therefore do not explain the tracking through heart rate alone. The effect, however, is small:
absolute errors stay at 12–16 mmHg MAE.

**2. Cross-day: a day-1 calibration does not hold on day 2.**
- True resting SBP changed between days by 4.7 ± 15.4 mmHg (ear subjects), and DBP by 2.9 ± 11.7.
- The model's predicted shift does **not** follow it. The slope of predicted on true shift is 0.02, 95 % CI
  [−0.04, 0.09] for SBP at the ear, and similar for every site and target. The CI **excludes 1**, so the
  failure to track day-level baseline changes is statistically supported.
- Each subject's calibration offset is only weakly reproducible across days (day-1 vs day-2 offset r 0.33–0.44,
  mean |change| 9–12 mmHg).
- So a 1-point day-1 calibration did not beat carry-forward on day 2 for SBP (ear −0.70 [−1.98, +0.59]; finger
  −0.22 [−1.49, +1.02]). For ear DBP it was **worse** than carry-forward (+1.19 [+0.32, +2.14]).
- Applying day-1 calibration *hurt* relative to the uncalibrated population model (ear DBP +2.83 [+0.23, +5.68]),
  because a stale offset adds noise.

**3. What survives across days is within-session *shape* tracking, not absolute level.**
Within-subject SBP correlation on day 2 using day-1-trained models stays at r 0.38 for ear shape features (HR-only
drops to 0.09). PPG shape relates to *changes* within a recording, but carries almost no information about a
person's *absolute* BP level or its day-to-day drift. Subject-mean predictions correlate with subject-mean truth at
r 0.14 (ear) and −0.03 (finger).

**4. Independence is not proven.** The correlation between true and predicted day shifts is near zero (ear SBP r
0.10 [−0.28, 0.38]), but with n = 29–33 the minimum correlation detectable at 80 % power is |r| ≈ 0.5. The data
therefore **do not establish independence**: moderate associations up to about 0.4 cannot be excluded. What *is*
established is that the slope is far below 1, so the model cannot be relied on to follow day-to-day changes.

**5. DBP behaves differently from SBP.** No PPG feature set improved DBP over carry-forward at any scope.
Finger DBP tracking r was ≈ 0.05, and ear DBP reached 0.19 within day but only 0.13 across days. Within a day,
calibrated ear DBP was slightly *worse* than carry-forward (+0.44 [+0.01, +0.91]).

**6. Ear vs finger.** The ear showed stronger within-session SBP tracking and the only clear evidence of shape
information beyond heart rate. Cross-day absolute errors did not differ between sites (ear − finger SBP
+0.65 [−1.22, +2.61]). A finger-trained model with ear calibration was not significantly worse than an
ear-trained one here (+0.60 [−1.73, +2.55]). For ear DBP the finger-trained model was marginally *better* (−0.91 [−1.92, −0.00]),
which is consistent with the absence of a usable DBP signal at either site. In session 1, however, finger→ear transfer lost tracking
(MODEL_EVALUATION.md §F3).

**7. Model choice does not matter at this sample size.** By the pre-specified criterion (mean cross-day MAE rank),
SVR ranks first and ElasticNet second (ElasticNet has the lowest mean MAE and best worst-case rank). **No
model is significantly better than ridge**. The only significant difference is HistGradientBoosting being worse
for ear DBP (+1.59 [+0.10, +3.25]). We recommend the regularised linear models
(ridge / ElasticNet): they are as good as any, simpler, and portable to a microcontroller.

**8. Amplifier clipping.** Removing rail-clipped windows (8.1 % of finger windows, 29 subjects affected, 5 with
> 25 % clipped) changed finger MAEs by ≤ 0.5 mmHg and never changed a conclusion. Ear data were unaffected.

**9. Population models vs simple baselines.** Calibration-free population predictions were not better than
carry-forward within a day. Across days they were numerically better than carry-forward (ear SBP −2.60
[−6.14, +0.90]) only because carry-forward suffers from day-to-day drift, and the CIs include 0.

### Bottom line for RakshPulse

Ear PPG contains a small amount of genuine, heart-rate-independent information about **within-session SBP
changes**. It does **not** provide reliable absolute BP, does not track day-to-day baseline shifts, gives no DBP
benefit, and its per-user calibration degrades within a day or two. Any RakshPulse BP feature would need frequent
cuff re-calibration, could at best report *trends within a session*, and would still need validation with the
actual behind-ear MAX30102 hardware against an ISO 81060-2/IEEE 1708-compliant reference.

### Limitations

- CNAP (calibrated finger-cuff) reference, not auscultatory, oscillometric-arm or arterial-line BP.
- Earlobe transmissive Nonin clip, not behind-ear reflective MAX30102. Healthy, young cohort; 36 subjects with
  both sessions. Stressors are laboratory manoeuvres, not field conditions.
- Session-2 analyses exclude subjects without ≥ 3 SQI-passing rest windows in session 1 (per-subject reasons in
  `results/cross_day/audit.json` check 9).
- The heart-failure cohort was not analysed (download stopped at 0/28 records, with 3 resumable partial files).

---

## Population model and personalisation (standardised dataset `data/auricle/`)

**Question:** can a population model give a usable first estimate for a new person, and does personal calibration
(and adaptive updating as new reference readings arrive) give better BP tracking than simply using those reference
readings?

**Setup** (`scripts/run_personalization.py`; engine in `src/rakshpulse/personalization/`):
- Population models are trained on session-1 windows of training subjects only (5 subject-grouped folds), with
  HR + pulse-shape features. They are evaluated uncalibrated on the held-out subjects' session 1 and session 2.
- Strategy comparison: every strategy is evaluated on the **same** day-2 windows (after the first three day-2 rest
  references). "baseline" strategies use only the reference readings (no PPG) via `BaselineTracker`. "pop+…"
  strategies apply `PersonalizationEngine` to the ridge population prediction. Personal models are ridge or
  ElasticNet fitted to the subject's own day-1 features and labels.
- Adaptive experiment: within each session a reference reading arrives every 2 minutes (nearest labelled
  window). Each method predicts the following windows using only past references.
- All CIs come from 2,000 subject-level bootstrap draws. Paired tests use per-subject MAE differences.

<!-- BEGIN PERSONALIZATION -->

### P1. Population models on unseen subjects (uncalibrated; HR + shape features)

| Site | Target | Model | MAE S1 | 95% CI | MAE S2 | 95% CI | Within-subj r (S1) | n subj (S1) |
|---|---|---|---|---|---|---|---|---|
| ear | SBP | mean | 15.29 | [13.27, 17.37] | 15.72 | [12.92, 18.97] | — | 39 |
| ear | DBP | mean | 10.72 | [8.86, 12.55] | 9.90 | [7.79, 12.65] | — | 39 |
| ear | SBP | ridge | 15.01 | [12.95, 17.24] | 15.16 | [12.63, 17.97] | 0.39 | 39 |
| ear | DBP | ridge | 10.98 | [9.00, 12.94] | 9.88 | [8.01, 12.37] | 0.18 | 39 |
| ear | SBP | elasticnet | 15.00 | [12.97, 17.22] | 15.11 | [12.48, 18.10] | 0.38 | 39 |
| ear | DBP | elasticnet | 10.80 | [8.82, 12.68] | 9.82 | [7.84, 12.54] | 0.24 | 39 |
| ear | SBP | svr | 15.44 | [13.23, 17.91] | 15.65 | [12.88, 18.76] | 0.36 | 39 |
| ear | DBP | svr | 11.24 | [9.28, 13.22] | 10.01 | [8.02, 12.76] | 0.18 | 39 |
| ear | SBP | random_forest | 16.11 | [13.88, 18.55] | 15.77 | [13.06, 18.96] | 0.33 | 39 |
| ear | DBP | random_forest | 11.35 | [9.45, 13.30] | 10.65 | [8.77, 13.17] | 0.17 | 39 |
| ear | SBP | extra_trees | 15.54 | [13.35, 17.88] | 15.34 | [12.66, 18.49] | 0.40 | 39 |
| ear | DBP | extra_trees | 10.99 | [9.08, 12.94] | 10.00 | [8.07, 12.62] | 0.17 | 39 |
| ear | SBP | hist_gb | 16.40 | [14.13, 18.93] | 16.12 | [13.34, 19.36] | 0.30 | 39 |
| ear | DBP | hist_gb | 11.74 | [9.84, 13.71] | 11.19 | [9.36, 13.76] | 0.15 | 39 |
| ear | SBP | xgboost | 15.87 | [13.61, 18.29] | 15.62 | [12.91, 18.84] | 0.35 | 39 |
| ear | DBP | xgboost | 11.55 | [9.69, 13.49] | 10.62 | [8.84, 13.16] | 0.15 | 39 |
| ear | SBP | lightgbm | 16.32 | [14.01, 18.85] | 15.98 | [13.23, 19.27] | 0.36 | 39 |
| ear | DBP | lightgbm | 11.83 | [9.96, 13.76] | 11.34 | [9.47, 13.85] | 0.16 | 39 |
| finger | SBP | mean | 15.88 | [13.82, 18.27] | 15.67 | [13.12, 18.47] | — | 41 |
| finger | DBP | mean | 10.88 | [9.39, 12.55] | 9.99 | [8.04, 12.35] | — | 41 |
| finger | SBP | ridge | 15.50 | [13.35, 17.95] | 16.07 | [13.28, 19.18] | 0.25 | 41 |
| finger | DBP | ridge | 11.45 | [9.77, 13.39] | 10.28 | [8.31, 12.50] | 0.06 | 41 |
| finger | SBP | elasticnet | 15.40 | [13.28, 17.91] | 15.60 | [12.82, 18.80] | 0.28 | 41 |
| finger | DBP | elasticnet | 11.03 | [9.47, 12.86] | 10.00 | [8.03, 12.29] | 0.05 | 41 |
| finger | SBP | svr | 15.45 | [13.22, 17.96] | 15.30 | [12.67, 18.20] | 0.42 | 41 |
| finger | DBP | svr | 12.09 | [10.13, 14.17] | 10.34 | [8.42, 12.39] | 0.15 | 41 |
| finger | SBP | random_forest | 15.81 | [13.69, 18.26] | 15.62 | [13.51, 17.91] | 0.38 | 41 |
| finger | DBP | random_forest | 12.06 | [10.19, 14.16] | 11.33 | [9.61, 13.20] | 0.16 | 41 |
| finger | SBP | extra_trees | 15.80 | [13.69, 18.28] | 15.48 | [12.95, 18.25] | 0.34 | 41 |
| finger | DBP | extra_trees | 11.71 | [9.90, 13.74] | 10.51 | [8.78, 12.43] | 0.14 | 41 |
| finger | SBP | hist_gb | 16.03 | [13.99, 18.39] | 16.22 | [13.81, 18.64] | 0.38 | 41 |
| finger | DBP | hist_gb | 12.46 | [10.62, 14.43] | 12.52 | [10.28, 15.02] | 0.17 | 41 |
| finger | SBP | xgboost | 15.82 | [13.69, 18.33] | 15.93 | [13.35, 18.68] | 0.35 | 41 |
| finger | DBP | xgboost | 12.29 | [10.41, 14.31] | 11.45 | [9.72, 13.36] | 0.12 | 41 |
| finger | SBP | lightgbm | 15.90 | [13.83, 18.25] | 16.02 | [13.79, 18.31] | 0.38 | 41 |
| finger | DBP | lightgbm | 12.43 | [10.55, 14.47] | 12.01 | [10.08, 14.05] | 0.15 | 41 |

Paired tests of each model against the training-mean predictor: 64 tests, 0 significantly better, 13 significantly worse (hist_gb, lightgbm, random_forest, xgboost).

### P2. Cross-day strategies (common day-2 evaluation set)

| Site | Target | Strategy | Information used | MAE | 95% CI | RMSE | Bias | SD err | Pearson | Spearman | Within-subj r | n subj |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ear | SBP | population | none | 15.40 | [12.54, 18.47] | 19.55 | -3.65 | 19.21 | 0.38 | 0.33 | 0.40 | 29 |
| ear | SBP | pop+S1cal | S1 refs | 16.42 | [13.48, 19.58] | 20.57 | -0.75 | 20.56 | 0.48 | 0.43 | 0.40 | 29 |
| ear | SBP | pop+S1cal+S2update | S1+S2 refs | 12.05 | [10.08, 14.37] | 16.02 | 0.03 | 16.02 | 0.66 | 0.66 | 0.40 | 29 |
| ear | SBP | pop+S2cal | S1+S2 refs | 12.25 | [10.29, 14.52] | 16.20 | 0.09 | 16.20 | 0.65 | 0.65 | 0.40 | 29 |
| ear | SBP | pop+S1linear | S1 refs | 14.62 | [11.88, 17.53] | 18.88 | -4.77 | 18.27 | 0.50 | 0.49 | 0.40 | 29 |
| ear | SBP | personal_ridge | S1 features+labels | 15.46 | [12.47, 18.84] | 20.63 | -4.37 | 20.16 | 0.45 | 0.43 | 0.30 | 29 |
| ear | SBP | personal_elasticnet | S1 features+labels | 14.27 | [11.50, 17.35] | 19.01 | -4.42 | 18.49 | 0.52 | 0.49 | 0.35 | 29 |
| ear | SBP | baseline_S1 | S1 refs | 17.49 | [14.25, 21.07] | 22.36 | 1.53 | 22.31 | 0.35 | 0.30 | — | 29 |
| ear | SBP | baseline_S1+S2 | S1+S2 refs | 13.13 | [11.12, 15.50] | 17.40 | 1.42 | 17.35 | 0.57 | 0.58 | — | 29 |
| ear | DBP | population | none | 9.97 | [7.82, 12.83] | 13.19 | -1.92 | 13.05 | 0.30 | 0.23 | 0.11 | 29 |
| ear | DBP | pop+S1cal | S1 refs | 12.78 | [10.02, 15.78] | 16.12 | -3.70 | 15.70 | 0.38 | 0.38 | 0.11 | 29 |
| ear | DBP | pop+S1cal+S2update | S1+S2 refs | 8.92 | [7.27, 10.88] | 11.67 | -3.00 | 11.28 | 0.58 | 0.51 | 0.11 | 29 |
| ear | DBP | pop+S2cal | S1+S2 refs | 9.04 | [7.35, 11.00] | 11.81 | -2.96 | 11.44 | 0.57 | 0.50 | 0.11 | 29 |
| ear | DBP | pop+S1linear | S1 refs | 10.84 | [8.40, 14.06] | 14.27 | -2.17 | 14.11 | 0.35 | 0.44 | 0.11 | 29 |
| ear | DBP | personal_ridge | S1 features+labels | 12.07 | [9.14, 15.67] | 16.42 | -2.97 | 16.15 | 0.31 | 0.43 | 0.25 | 29 |
| ear | DBP | personal_elasticnet | S1 features+labels | 10.70 | [8.17, 13.75] | 14.48 | -2.82 | 14.21 | 0.38 | 0.49 | 0.31 | 29 |
| ear | DBP | baseline_S1 | S1 refs | 11.44 | [8.79, 14.51] | 14.84 | -2.19 | 14.68 | 0.37 | 0.39 | — | 29 |
| ear | DBP | baseline_S1+S2 | S1+S2 refs | 8.69 | [6.87, 10.93] | 11.55 | -2.63 | 11.25 | 0.57 | 0.49 | — | 29 |
| finger | SBP | population | none | 16.30 | [13.10, 20.11] | 20.79 | -1.80 | 20.72 | 0.16 | 0.16 | 0.24 | 33 |
| finger | SBP | pop+S1cal | S1 refs | 15.68 | [12.98, 18.66] | 20.08 | 1.01 | 20.06 | 0.47 | 0.46 | 0.24 | 33 |
| finger | SBP | pop+S1cal+S2update | S1+S2 refs | 12.05 | [10.17, 14.09] | 15.86 | 0.36 | 15.86 | 0.66 | 0.67 | 0.24 | 33 |
| finger | SBP | pop+S2cal | S1+S2 refs | 12.19 | [10.28, 14.27] | 16.04 | 0.32 | 16.04 | 0.66 | 0.66 | 0.24 | 33 |
| finger | SBP | pop+S1linear | S1 refs | 14.08 | [11.93, 16.38] | 18.39 | -2.39 | 18.23 | 0.49 | 0.48 | 0.22 | 33 |
| finger | SBP | personal_ridge | S1 features+labels | 13.96 | [11.39, 16.57] | 18.95 | -2.00 | 18.85 | 0.51 | 0.54 | 0.32 | 33 |
| finger | SBP | personal_elasticnet | S1 features+labels | 13.70 | [11.17, 16.31] | 18.44 | -2.21 | 18.31 | 0.51 | 0.53 | 0.37 | 33 |
| finger | SBP | baseline_S1 | S1 refs | 15.80 | [12.92, 18.85] | 20.50 | 0.74 | 20.49 | 0.44 | 0.41 | — | 33 |
| finger | SBP | baseline_S1+S2 | S1+S2 refs | 12.09 | [10.39, 13.95] | 16.23 | 1.07 | 16.19 | 0.63 | 0.63 | — | 33 |
| finger | DBP | population | none | 10.32 | [8.18, 12.89] | 13.54 | -0.66 | 13.52 | 0.16 | 0.09 | 0.04 | 33 |
| finger | DBP | pop+S1cal | S1 refs | 11.51 | [9.09, 13.98] | 14.64 | -2.14 | 14.48 | 0.39 | 0.37 | 0.04 | 33 |
| finger | DBP | pop+S1cal+S2update | S1+S2 refs | 9.12 | [7.57, 10.71] | 11.86 | -2.57 | 11.58 | 0.54 | 0.49 | 0.04 | 33 |
| finger | DBP | pop+S2cal | S1+S2 refs | 9.35 | [7.80, 10.97] | 12.08 | -2.60 | 11.80 | 0.53 | 0.47 | 0.04 | 33 |
| finger | DBP | pop+S1linear | S1 refs | 10.41 | [8.13, 13.10] | 14.01 | -0.77 | 13.99 | 0.32 | 0.40 | 0.02 | 33 |
| finger | DBP | personal_ridge | S1 features+labels | 12.55 | [9.29, 16.03] | 17.87 | 0.96 | 17.84 | 0.20 | 0.34 | 0.23 | 33 |
| finger | DBP | personal_elasticnet | S1 features+labels | 11.65 | [8.76, 14.64] | 16.02 | 0.77 | 16.01 | 0.24 | 0.36 | 0.21 | 33 |
| finger | DBP | baseline_S1 | S1 refs | 11.19 | [8.59, 13.86] | 14.59 | -1.94 | 14.46 | 0.36 | 0.36 | — | 33 |
| finger | DBP | baseline_S1+S2 | S1+S2 refs | 8.51 | [6.84, 10.40] | 11.50 | -2.39 | 11.25 | 0.56 | 0.50 | — | 33 |

### P3. Paired subject-level comparisons (Δ = MAE(a) − MAE(b); negative favours a)

| site | target | comparison | mae_diff | ci_low | ci_high | significant |
|---|---|---|---|---|---|---|
| ear | sbp | pop+S1cal vs baseline_S1 | -1.11 | -2.55 | 0.24 | False |
| ear | sbp | pop+S1cal+S2update vs baseline_S1+S2 | -0.97 | -2.05 | 0.1 | False |
| ear | sbp | pop+S2cal vs baseline_S1+S2 | -0.81 | -1.98 | 0.34 | False |
| ear | sbp | pop+S1cal+S2update vs pop+S1cal | -4.77 | -7.81 | -2.09 | True |
| ear | sbp | pop+S1linear vs pop+S1cal | -1.86 | -5.0 | 1.37 | False |
| ear | sbp | personal_ridge vs baseline_S1 | -1.98 | -6.13 | 2.36 | False |
| ear | sbp | personal_elasticnet vs baseline_S1 | -3.32 | -7.44 | 0.94 | False |
| ear | sbp | population vs baseline_S1 | -2.6 | -6.55 | 1.12 | False |
| ear | sbp | personal_elasticnet vs personal_ridge | -1.34 | -2.03 | -0.75 | True |
| ear | dbp | pop+S1cal vs baseline_S1 | 1.16 | 0.24 | 2.11 | True |
| ear | dbp | pop+S1cal+S2update vs baseline_S1+S2 | 0.2 | -0.72 | 1.11 | False |
| ear | dbp | pop+S2cal vs baseline_S1+S2 | 0.32 | -0.75 | 1.41 | False |
| ear | dbp | pop+S1cal+S2update vs pop+S1cal | -3.73 | -6.47 | -1.34 | True |
| ear | dbp | pop+S1linear vs pop+S1cal | -1.76 | -4.14 | 0.53 | False |
| ear | dbp | personal_ridge vs baseline_S1 | 0.87 | -2.13 | 4.15 | False |
| ear | dbp | personal_elasticnet vs baseline_S1 | -0.58 | -3.35 | 2.2 | False |
| ear | dbp | population vs baseline_S1 | -1.55 | -4.38 | 1.09 | False |
| ear | dbp | personal_elasticnet vs personal_ridge | -1.44 | -2.45 | -0.67 | True |
| ear | sbp | adaptive S1: rolling3_reference vs last_reference | 0.2 | -0.44 | 0.84 | False |
| ear | sbp | adaptive S1: engine_offset vs last_reference | -0.34 | -1.11 | 0.49 | False |
| ear | sbp | adaptive S1: engine_linear vs last_reference | -0.07 | -0.93 | 0.84 | False |
| ear | sbp | adaptive S1: population vs last_reference | 5.16 | 2.83 | 7.5 | True |
| ear | sbp | adaptive S2: rolling3_reference vs last_reference | 0.18 | -0.48 | 0.89 | False |
| ear | sbp | adaptive S2: engine_offset vs last_reference | 0.1 | -1.1 | 1.29 | False |
| ear | sbp | adaptive S2: engine_linear vs last_reference | 0.34 | -0.97 | 1.64 | False |
| ear | sbp | adaptive S2: population vs last_reference | 6.38 | 3.86 | 9.14 | True |
| finger | sbp | pop+S1cal vs baseline_S1 | -0.26 | -1.75 | 1.22 | False |
| finger | sbp | pop+S1cal+S2update vs baseline_S1+S2 | 0.18 | -0.95 | 1.43 | False |
| finger | sbp | pop+S2cal vs baseline_S1+S2 | 0.29 | -0.94 | 1.61 | False |
| finger | sbp | pop+S1cal+S2update vs pop+S1cal | -4.02 | -7.02 | -1.36 | True |
| finger | sbp | pop+S1linear vs pop+S1cal | -2.28 | -5.13 | 0.27 | False |
| finger | sbp | personal_ridge vs baseline_S1 | -2.56 | -5.91 | 0.64 | False |
| finger | sbp | personal_elasticnet vs baseline_S1 | -2.87 | -6.1 | 0.34 | False |
| finger | sbp | population vs baseline_S1 | -0.69 | -4.66 | 3.46 | False |
| finger | sbp | personal_elasticnet vs personal_ridge | -0.31 | -0.74 | 0.14 | False |
| finger | dbp | pop+S1cal vs baseline_S1 | 0.43 | -0.34 | 1.24 | False |
| finger | dbp | pop+S1cal+S2update vs baseline_S1+S2 | 0.78 | -0.26 | 1.81 | False |
| finger | dbp | pop+S2cal vs baseline_S1+S2 | 1.01 | -0.11 | 2.14 | False |
| finger | dbp | pop+S1cal+S2update vs pop+S1cal | -2.24 | -4.59 | -0.12 | True |
| finger | dbp | pop+S1linear vs pop+S1cal | -0.95 | -3.32 | 1.15 | False |
| finger | dbp | personal_ridge vs baseline_S1 | 0.78 | -2.13 | 3.67 | False |
| finger | dbp | personal_elasticnet vs baseline_S1 | 0.14 | -2.54 | 2.86 | False |
| finger | dbp | population vs baseline_S1 | -0.72 | -3.53 | 1.76 | False |
| finger | dbp | personal_elasticnet vs personal_ridge | -0.64 | -1.28 | -0.14 | True |
| finger | sbp | adaptive S1: rolling3_reference vs last_reference | 0.62 | -0.05 | 1.35 | False |
| finger | sbp | adaptive S1: engine_offset vs last_reference | 0.83 | 0.09 | 1.58 | True |
| finger | sbp | adaptive S1: engine_linear vs last_reference | 1.13 | 0.33 | 1.99 | True |
| finger | sbp | adaptive S1: population vs last_reference | 5.65 | 3.5 | 8.15 | True |
| finger | sbp | adaptive S2: rolling3_reference vs last_reference | 0.26 | -0.4 | 0.95 | False |
| finger | sbp | adaptive S2: engine_offset vs last_reference | 1.17 | 0.22 | 2.06 | True |
| finger | sbp | adaptive S2: engine_linear vs last_reference | 1.3 | 0.27 | 2.25 | True |
| finger | sbp | adaptive S2: population vs last_reference | 7.51 | 4.88 | 10.34 | True |

### P4. Adaptive within-day tracking (a reference every 2 min; strictly causal; SBP)

| Site | Session | Method | MAE | 95% CI | n subj |
|---|---|---|---|---|---|
| ear | 1 | last_reference | 9.84 | [8.89, 10.78] | 39 |
| ear | 1 | rolling3_reference | 10.02 | [9.01, 11.00] | 39 |
| ear | 1 | engine_offset | 9.48 | [8.54, 10.46] | 39 |
| ear | 1 | engine_linear | 9.75 | [8.75, 10.81] | 39 |
| ear | 1 | population | 14.89 | [12.86, 17.08] | 39 |
| ear | 2 | last_reference | 9.04 | [8.04, 10.19] | 33 |
| ear | 2 | rolling3_reference | 9.17 | [8.14, 10.30] | 33 |
| ear | 2 | engine_offset | 9.07 | [8.06, 10.12] | 33 |
| ear | 2 | engine_linear | 9.38 | [8.20, 10.61] | 33 |
| ear | 2 | population | 15.19 | [12.62, 18.01] | 33 |
| finger | 1 | last_reference | 9.65 | [8.67, 10.66] | 41 |
| finger | 1 | rolling3_reference | 9.86 | [8.83, 10.99] | 41 |
| finger | 1 | engine_offset | 10.26 | [9.29, 11.40] | 41 |
| finger | 1 | engine_linear | 10.51 | [9.44, 11.78] | 41 |
| finger | 1 | population | 15.53 | [13.29, 18.08] | 41 |
| finger | 2 | last_reference | 8.68 | [7.83, 9.73] | 37 |
| finger | 2 | rolling3_reference | 8.93 | [7.99, 9.90] | 37 |
| finger | 2 | engine_offset | 9.67 | [8.77, 10.61] | 37 |
| finger | 2 | engine_linear | 9.89 | [8.84, 11.00] | 37 |
| finger | 2 | population | 16.04 | [13.19, 19.17] | 37 |

### P5. Drift monitor (first day-2 reference vs the day-1 calibration)

| Site | Subjects | Recalibration recommended | Day-2 MAE if recommended | Day-2 MAE otherwise | r(abs first residual, day-2 MAE) |
|---|---|---|---|---|---|
| ear | 29 | 3 | 22.15 | 16.20 | 0.55 |
| finger | 33 | 5 | 20.26 | 15.97 | 0.50 |

### P6. Baseline tracking (day-2 resting SBP vs the day-1 personal reference baseline)

| Site | Subjects | Median day-1 references | Mean shift (mmHg) | SD of shift | % subjects with abs(z) > 2 |
|---|---|---|---|---|---|
| ear | 29 | 113.00 | 4.69 | 15.40 | 13.79 |
| finger | 33 | 112.00 | 3.10 | 16.40 | 21.21 |

<!-- END PERSONALIZATION -->

### Interpretation

1. **The population model gives no usable first estimate for a new person.** On unseen subjects no model was
   better than predicting the training-mean BP. All significant differences were tree ensembles doing *worse*.
   The PPG features learned here do not identify a person's absolute BP.
2. **Personal calibration helps only through the reference readings it contains.** Adding day-2 references to a
   day-1 calibration reduced SBP error by about 4–5 mmHg, but a PPG-free tracker of the same references did
   statistically as well. With references every 2 min within a day, the engine never beat "repeat the last
   reference" (it was worse at the finger). Under these conditions the PPG signal adds no measurable value on top
   of the cuff readings themselves.
3. **Personal feature models** (fitted on one day, applied to the next) were not better than the day-1 baseline.
   ElasticNet beat personal Ridge in 3 of 4 site × target conditions (stronger regularisation suits about 100–250 correlated windows per subject).
4. **Drift detection needs references.** The first day-2 residual against the day-1 calibration correlates with the
   eventual day-2 error (r ≈ 0.5). The engine therefore uses residuals plus calibration age to recommend
   recalibration. PPG alone cannot indicate a stale calibration.
5. **Baseline tracking** shows how much a person's resting BP moves between days relative to their own day-1 spread.
   This is the variability any calibration must absorb.

**What can and cannot be claimed.** Two recording days per person, in a laboratory, with CNAP references, support
statements about within-day and next-day behaviour only. They do not support claims about long-term
personalisation, weeks of drift, or behind-ear MAX30102 hardware. The engine is implemented and unit-tested for
long-horizon use (decay, rollback, persistence), but long-term performance is **untested**.

