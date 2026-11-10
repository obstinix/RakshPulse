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

---

## RakshPulse 3.0: information beyond the reference reading

**Primary question:** can PPG waveforms provide reliable additional information for blood-pressure estimation
*beyond historical reference measurements*, across different subjects, recording sessions and physiological
conditions?

The 2.0 results above are unchanged; 3.0 does not overwrite them. 3.0 adds more sensitive tests of the same question,
a second dataset with *real* intermittent cuff readings, and the requested model families.

**Setup** (all runs are in the provenance registry `results/runs/runs.csv`: dataset SHA-256, git commit, software
versions, splits, metrics). All five runs were executed from commit `c13491c`. Their `git_dirty` flag reflects only
uncommitted docs, the report renderer and the new result files; experiment code was identical to the commit.
- **V1, beyond-reference** (`scripts/run_beyond_reference.py`, `src/rakshpulse/evaluation/beyond_reference.py`).
  Every window is paired with the most recent earlier reference reading. Methods predict *reference + change*:
  - `last_reference` carries the reading forward;
  - `historical_mean` averages all readings so far;
  - `population_mean` is the training-subject mean;
  - `mean_change` is the training-subject average change since a reference (no signal; the control);
  - `delta_<features>` is a ridge model, trained on other subjects only, mapping **feature changes since the
    reference** to BP changes.

  The signal adds information only if `delta_*` beats `mean_change`. Auricle references are simulated every 2, 5
  and 10 min (CNAP truth) and, for cross-day, as the day-1 rest mean. VitalDB references are the patients' **real
  NIBP cuff readings** and the truth is the arterial line (gap ≤ 30 min). The 2.0 ridge model is included as
  `rakshpulse2_offset`.
- **V2, calibration count** (`scripts/run_calibration_study.py`): k = 0–10 cuff readings ≥ 60 s apart at the start
  of day 1, applied to the 2.0 model's cached subject-held-out predictions, within day and on day 2. Methods: plain
  offset, empirical-Bayes shrunk offset, free and slope-shrunk linear calibration, with two no-PPG baselines: the
  reading mean, and **`bayes_reference`**, which shrinks the reading mean towards the training mean with the same
  rule. `bayes_offset` vs `bayes_reference` isolates the waveform's contribution from plain statistical shrinkage.
  Uncertainty: cross-subject conformal 90 % intervals.
- **V3, features** (`scripts/run_feature_analysis.py`): within-subject Spearman association with SBP per day,
  sign agreement between days, resting ICC between days, VIF, and Lasso stability selection on within-subject-centred
  data.
- **V4, deep models** (`scripts/run_deep_benchmark.py`): CNN, GRU, CNN-LSTM, TCN and Transformer on raw ear
  waveforms vs training mean, ridge (HR + shape) and reference-only, all on the same five subject folds.
- **V6, streaming cost** (`scripts/benchmark_inference.py`).

All CIs come from 2,000 subject-level bootstrap draws; paired tests use per-subject MAE differences.

<!-- BEGIN V3 -->

### V1. Does the waveform add information beyond the last reference? (Δ MAE vs the `mean_change` control, mmHg; negative = the signal helps; * = 95 % subject-bootstrap CI excludes 0)

| Data | Site | Target | Reference scenario | MAE last ref | MAE pop. mean | HR only | HR + shape | All PPG | HR + shape + PAT | 2.0 model offset vs last ref | n subj |
|---|---|---|---|---|---|---|---|---|---|---|---|
| auricle | ear | sbp | within_day_ref_every_120s | 9.98 | 15.77 | -0.01 [-0.12, +0.10] | -0.57 [-0.86, -0.27]* | -0.58 [-0.89, -0.27]* | -0.66 [-0.99, -0.34]* | -0.15 [-0.47, +0.19] | 40 |
| auricle | ear | sbp | within_day_ref_every_300s | 11.88 | 15.77 | +0.05 [-0.09, +0.22] | -0.74 [-1.29, -0.23]* | -0.72 [-1.29, -0.21]* | -0.70 [-1.27, -0.12]* | -0.43 [-0.91, +0.09] | 40 |
| auricle | ear | sbp | within_day_ref_every_600s | 13.64 | 15.77 | -0.02 [-0.57, +0.49] | -1.20 [-2.22, -0.29]* | -1.19 [-2.17, -0.33]* | -1.02 [-2.13, +0.18] | -0.87 [-1.59, -0.15]* | 40 |
| auricle | ear | sbp | cross_day_day1_rest_reference | 16.52 | 15.82 | -0.35 [-1.88, +1.07] | -1.48 [-4.15, +1.51] | -1.68 [-4.47, +1.55] | -1.39 [-4.60, +2.32] | -0.70 [-1.99, +0.59] | 31 |
| auricle | ear | dbp | within_day_ref_every_120s | 6.41 | 10.30 | -0.02 [-0.10, +0.05] | -0.12 [-0.29, +0.07] | -0.07 [-0.25, +0.14] | -0.11 [-0.29, +0.08] | +0.20 [+0.01, +0.39]* | 40 |
| auricle | ear | dbp | within_day_ref_every_300s | 7.56 | 10.30 | +0.02 [-0.07, +0.10] | +0.01 [-0.32, +0.36] | +0.06 [-0.30, +0.47] | +0.08 [-0.25, +0.44] | +0.07 [-0.15, +0.29] | 40 |
| auricle | ear | dbp | within_day_ref_every_600s | 8.55 | 10.30 | -0.06 [-0.27, +0.14] | -0.33 [-0.76, +0.07] | -0.30 [-0.80, +0.13] | -0.20 [-0.68, +0.26] | +0.01 [-0.34, +0.37] | 40 |
| auricle | ear | dbp | cross_day_day1_rest_reference | 11.59 | 9.88 | -0.25 [-1.14, +0.49] | +2.56 [+0.91, +4.55]* | +3.07 [+1.10, +5.37]* | +2.27 [+0.16, +4.61]* | +1.19 [+0.32, +2.14]* | 31 |
| auricle | finger | sbp | within_day_ref_every_120s | 9.66 | 15.82 | -0.04 [-0.15, +0.09] | -0.39 [-0.70, -0.06]* | -0.68 [-1.05, -0.30]* | -0.81 [-1.23, -0.32]* | +0.72 [+0.21, +1.30]* | 42 |
| auricle | finger | sbp | within_day_ref_every_300s | 11.38 | 15.82 | -0.11 [-0.34, +0.13] | -0.30 [-0.85, +0.26] | -0.65 [-1.26, -0.06]* | -0.75 [-1.41, -0.03]* | +0.45 [-0.23, +1.16] | 42 |
| auricle | finger | sbp | within_day_ref_every_600s | 13.15 | 15.82 | -0.54 [-1.15, +0.08] | -0.23 [-1.23, +0.70] | -0.80 [-1.95, +0.27] | -0.99 [-2.12, +0.10] | +0.29 [-0.61, +1.17] | 42 |
| auricle | finger | sbp | cross_day_day1_rest_reference | 15.59 | 15.84 | -0.33 [-1.14, +0.41] | +0.81 [-0.59, +2.20] | +0.38 [-1.30, +2.04] | +1.77 [-0.16, +4.05] | -0.22 [-1.49, +1.02] | 35 |
| auricle | finger | dbp | within_day_ref_every_120s | 6.31 | 10.28 | +0.02 [-0.04, +0.07] | -0.20 [-0.36, -0.04]* | -0.22 [-0.40, -0.02]* | -0.24 [-0.43, -0.01]* | +0.42 [+0.24, +0.62]* | 42 |
| auricle | finger | dbp | within_day_ref_every_300s | 7.45 | 10.28 | +0.01 [-0.09, +0.10] | -0.21 [-0.60, +0.18] | -0.19 [-0.63, +0.25] | -0.21 [-0.65, +0.22] | +0.31 [-0.03, +0.64] | 42 |
| auricle | finger | dbp | within_day_ref_every_600s | 8.38 | 10.28 | -0.05 [-0.28, +0.17] | +0.04 [-0.57, +0.61] | -0.06 [-0.75, +0.57] | -0.09 [-0.76, +0.53] | +0.34 [-0.16, +0.84] | 42 |
| auricle | finger | dbp | cross_day_day1_rest_reference | 11.41 | 9.71 | +0.02 [-0.60, +0.68] | +0.45 [-0.57, +1.40] | +1.16 [-0.04, +2.28] | +0.58 [-0.44, +1.49] | +0.37 [-0.40, +1.18] | 35 |
| vitaldb | finger | sbp | real_nibp_references | 18.26 | 15.89 | -0.11 [-0.67, +0.45] | -0.31 [-2.39, +2.06] | +0.17 [-2.13, +2.81] | -0.57 [-2.77, +1.89] | — | 34 |
| vitaldb | finger | dbp | real_nibp_references | 11.14 | 10.48 | -0.06 [-0.49, +0.48] | -0.75 [-1.64, +0.10] | -0.71 [-1.66, +0.21] | -0.98 [-1.95, -0.06]* | — | 34 |

### V2. Calibration count (SBP MAE, mmHg; 2.0 ridge population model, cached OOF predictions)

**ear**

| Scenario | Method | k=0 | k=1 | k=2 | k=3 | k=5 | k=10 |
|---|---|---|---|---|---|---|---|
| cross_day | bayes_offset | — | 13.96 | 13.82 | 13.94 | 14.10 | 14.07 |
| cross_day | bayes_reference | — | 14.77 | 14.21 | 14.68 | 14.54 | 14.89 |
| cross_day | linear_free | — | — | 32.49 | 20.35 | 17.93 | 16.69 |
| cross_day | linear_shrunk | — | — | 15.97 | 15.25 | 15.10 | 14.81 |
| cross_day | offset | — | 17.65 | 15.63 | 15.18 | 14.89 | 14.39 |
| cross_day | population | 15.07 | — | — | — | — | — |
| cross_day | reference_mean | — | 17.95 | 15.44 | 15.72 | 15.03 | 15.20 |
| cross_day | training_mean | 15.64 | — | — | — | — | — |
| within_day | bayes_offset | — | 12.82 | 11.77 | 11.47 | 12.14 | 11.94 |
| within_day | bayes_reference | — | 13.00 | 12.09 | 12.24 | 12.88 | 12.88 |
| within_day | linear_free | — | — | 25.54 | 16.43 | 15.04 | 13.81 |
| within_day | linear_shrunk | — | — | 12.55 | 11.72 | 12.62 | 12.03 |
| within_day | offset | — | 16.10 | 12.90 | 12.11 | 12.68 | 12.13 |
| within_day | population | 15.01 | — | — | — | — | — |
| within_day | reference_mean | — | 16.52 | 13.28 | 13.25 | 13.55 | 13.12 |
| within_day | training_mean | 15.29 | — | — | — | — | — |

**finger**

| Scenario | Method | k=0 | k=1 | k=2 | k=3 | k=5 | k=10 |
|---|---|---|---|---|---|---|---|
| cross_day | bayes_offset | — | 14.70 | 14.62 | 14.62 | 14.63 | 14.32 |
| cross_day | bayes_reference | — | 14.06 | 14.20 | 14.37 | 14.14 | 14.35 |
| cross_day | linear_free | — | — | 38.89 | 16.12 | 14.11 | 14.68 |
| cross_day | linear_shrunk | — | — | 15.21 | 14.58 | 13.74 | 13.82 |
| cross_day | offset | — | 16.07 | 15.37 | 15.14 | 14.93 | 14.45 |
| cross_day | population | 15.99 | — | — | — | — | — |
| cross_day | reference_mean | — | 15.57 | 15.37 | 15.18 | 14.50 | 14.54 |
| cross_day | training_mean | 15.61 | — | — | — | — | — |
| within_day | bayes_offset | — | 13.51 | 13.00 | 12.62 | 12.80 | 12.71 |
| within_day | bayes_reference | — | 13.26 | 12.85 | 12.86 | 13.13 | 13.33 |
| within_day | linear_free | — | — | 30.90 | 16.82 | 13.38 | 14.27 |
| within_day | linear_shrunk | — | — | 13.78 | 12.96 | 12.48 | 12.75 |
| within_day | offset | — | 15.72 | 13.98 | 13.05 | 13.11 | 12.74 |
| within_day | population | 15.50 | — | — | — | — | — |
| within_day | reference_mean | — | 15.45 | 14.06 | 13.69 | 13.65 | 13.48 |
| within_day | training_mean | 15.88 | — | — | — | — | — |

Paired (Δ MAE; negative favours the first method. `bayes_offset vs bayes_reference` is the PPG's own contribution; the other rows also credit shrinkage):

| Site | Scenario | k | Comparison | Δ MAE [95% CI] |
|---|---|---|---|---|
| ear | cross_day | 1 | offset vs reference_mean | -0.69 [-2.74, +1.43] |
| ear | cross_day | 1 | bayes_offset vs reference_mean | -4.38 [-6.59, -2.17]* |
| ear | cross_day | 1 | bayes_offset vs bayes_reference | -0.64 [-1.75, +0.42] |
| ear | cross_day | 3 | offset vs reference_mean | -0.51 [-1.71, +0.50] |
| ear | cross_day | 3 | bayes_offset vs reference_mean | -1.65 [-2.86, -0.50]* |
| ear | cross_day | 3 | bayes_offset vs bayes_reference | -0.52 [-1.50, +0.33] |
| ear | cross_day | 10 | offset vs reference_mean | -0.68 [-1.50, +0.08] |
| ear | cross_day | 10 | bayes_offset vs reference_mean | -0.98 [-1.74, -0.24]* |
| ear | cross_day | 10 | bayes_offset vs bayes_reference | -0.66 [-1.47, +0.10] |
| ear | within_day | 1 | offset vs reference_mean | -0.75 [-2.33, +0.95] |
| ear | within_day | 1 | bayes_offset vs reference_mean | -3.87 [-5.68, -2.15]* |
| ear | within_day | 1 | bayes_offset vs bayes_reference | -0.17 [-0.90, +0.61] |
| ear | within_day | 3 | offset vs reference_mean | -1.26 [-2.23, -0.40]* |
| ear | within_day | 3 | bayes_offset vs reference_mean | -1.82 [-2.83, -0.90]* |
| ear | within_day | 3 | bayes_offset vs bayes_reference | -0.80 [-1.48, -0.17]* |
| ear | within_day | 10 | offset vs reference_mean | -0.85 [-1.66, -0.08]* |
| ear | within_day | 10 | bayes_offset vs reference_mean | -1.03 [-1.86, -0.22]* |
| ear | within_day | 10 | bayes_offset vs bayes_reference | -0.78 [-1.53, -0.05]* |
| finger | cross_day | 1 | offset vs reference_mean | +0.30 [-1.17, +1.82] |
| finger | cross_day | 1 | bayes_offset vs reference_mean | -1.24 [-3.35, +0.94] |
| finger | cross_day | 1 | bayes_offset vs bayes_reference | +0.68 [-0.27, +1.72] |
| finger | cross_day | 3 | offset vs reference_mean | -0.19 [-1.60, +1.22] |
| finger | cross_day | 3 | bayes_offset vs reference_mean | -0.82 [-2.47, +0.85] |
| finger | cross_day | 3 | bayes_offset vs bayes_reference | +0.19 [-1.01, +1.46] |
| finger | cross_day | 10 | offset vs reference_mean | -0.29 [-1.40, +0.92] |
| finger | cross_day | 10 | bayes_offset vs reference_mean | -0.42 [-1.60, +0.87] |
| finger | cross_day | 10 | bayes_offset vs bayes_reference | -0.21 [-1.28, +0.94] |
| finger | within_day | 1 | offset vs reference_mean | +0.02 [-1.61, +1.63] |
| finger | within_day | 1 | bayes_offset vs reference_mean | -1.93 [-3.62, -0.18]* |
| finger | within_day | 1 | bayes_offset vs bayes_reference | +0.12 [-0.76, +1.05] |
| finger | within_day | 3 | offset vs reference_mean | -0.90 [-2.07, +0.17] |
| finger | within_day | 3 | bayes_offset vs reference_mean | -1.31 [-2.68, -0.06]* |
| finger | within_day | 3 | bayes_offset vs bayes_reference | -0.44 [-1.40, +0.49] |
| finger | within_day | 10 | offset vs reference_mean | -1.00 [-1.69, -0.32]* |
| finger | within_day | 10 | bayes_offset vs reference_mean | -1.08 [-1.83, -0.32]* |
| finger | within_day | 10 | bayes_offset vs bayes_reference | -0.86 [-1.52, -0.22]* |

Cross-subject conformal 90 % intervals (k = 3):

| Site | Scenario | Method | Coverage | Mean half-width (mmHg) |
|---|---|---|---|---|
| ear | cross_day | bayes_offset | 0.90 | 29.17 |
| ear | cross_day | bayes_reference | 0.89 | 32.66 |
| ear | cross_day | linear_free | 0.90 | 40.17 |
| ear | cross_day | linear_shrunk | 0.90 | 31.09 |
| ear | cross_day | offset | 0.90 | 30.37 |
| ear | cross_day | reference_mean | 0.89 | 34.13 |
| ear | within_day | bayes_offset | 0.90 | 25.28 |
| ear | within_day | bayes_reference | 0.90 | 27.93 |
| ear | within_day | linear_free | 0.90 | 34.68 |
| ear | within_day | linear_shrunk | 0.90 | 25.62 |
| ear | within_day | offset | 0.90 | 27.01 |
| ear | within_day | reference_mean | 0.90 | 29.55 |
| finger | cross_day | bayes_offset | 0.90 | 29.61 |
| finger | cross_day | bayes_reference | 0.90 | 31.35 |
| finger | cross_day | linear_free | 0.90 | 35.06 |
| finger | cross_day | linear_shrunk | 0.90 | 29.76 |
| finger | cross_day | offset | 0.90 | 30.40 |
| finger | cross_day | reference_mean | 0.90 | 32.85 |
| finger | within_day | bayes_offset | 0.90 | 27.56 |
| finger | within_day | bayes_reference | 0.90 | 29.47 |
| finger | within_day | linear_free | 0.90 | 35.84 |
| finger | within_day | linear_shrunk | 0.90 | 28.12 |
| finger | within_day | offset | 0.90 | 28.12 |
| finger | within_day | reference_mean | 0.90 | 30.78 |

Error vs time since calibration (within day, k = 3, SBP):

| Site | Method | Minutes since calibration | MAE | Bias | n subj |
|---|---|---|---|---|---|
| ear | offset | 0-5 | 9.27 | 0.82 | 39 |
| ear | offset | 5-10 | 11.15 | 4.59 | 39 |
| ear | offset | 10-20 | 13.51 | 1.03 | 39 |
| ear | offset | 20-40 | 13.85 | 2.50 | 34 |
| ear | reference_mean | 0-5 | 9.92 | 1.17 | 39 |
| ear | reference_mean | 5-10 | 13.06 | 4.66 | 39 |
| ear | reference_mean | 10-20 | 14.79 | -0.48 | 39 |
| ear | reference_mean | 20-40 | 14.29 | 1.07 | 34 |
| finger | offset | 0-5 | 10.25 | -0.13 | 41 |
| finger | offset | 5-10 | 12.61 | 4.48 | 40 |
| finger | offset | 10-20 | 14.43 | -0.00 | 40 |
| finger | offset | 20-40 | 13.75 | 0.24 | 38 |
| finger | reference_mean | 0-5 | 10.13 | 0.78 | 41 |
| finger | reference_mean | 5-10 | 13.51 | 4.32 | 40 |
| finger | reference_mean | 10-20 | 15.41 | -0.35 | 40 |
| finger | reference_mean | 20-40 | 14.21 | 0.00 | 38 |

### V3. Features: within-subject association with SBP, cross-day stability, redundancy (top 12 by |within-subject r|)

| Site | Feature | Within-subj Spearman day 1 [95% CI] | Day 2 | Sign agreement day1/day2 | Rest ICC(3,1) day1/day2 | VIF | Stability-selection freq. |
|---|---|---|---|---|---|---|---|
| ear | ear_w25_sd_ratio | -0.40 [-0.45, -0.32] | -0.39 | 0.87 | 0.75 | 37.3 | 0.40 |
| ear | ear_sys_ratio | +0.39 [+0.34, +0.49] | +0.33 | 0.81 | 0.70 | 25.4 | 0.98 |
| ear | ear_w10_sd_ratio | -0.38 [-0.50, -0.37] | -0.37 | 0.84 | 0.76 | 31.0 | 0.14 |
| ear | ear_pat_foot_ms | -0.37 [-0.48, -0.26] | -0.41 | 0.87 | 0.75 | 45.2 | 0.41 |
| ear | ear_area_ratio | -0.37 [-0.44, -0.27] | -0.31 | 0.84 | 0.78 | 48.4 | 0.07 |
| ear | ear_h2_h1 | -0.36 [-0.41, -0.25] | -0.26 | 0.90 | 0.39 | 4.9 | 1.00 |
| ear | ear_pat_slope_ms | -0.34 [-0.48, -0.19] | -0.36 | 0.84 | 0.76 | 61.5 | 0.79 |
| ear | ear_w90_n | +0.33 [+0.21, +0.46] | +0.29 | 0.71 | 0.52 | 5.3 | 0.49 |
| ear | ear_max_slope_n | -0.32 [-0.48, -0.23] | -0.29 | 0.77 | 0.80 | 30.4 | 0.28 |
| ear | ear_w75_n | +0.31 [+0.22, +0.42] | +0.29 | 0.84 | 0.47 | 8.6 | 0.79 |
| ear | ear_h3_h1 | -0.30 [-0.44, -0.26] | -0.36 | 0.87 | 0.69 | 5.9 | 0.43 |
| ear | ear_aging_index | +0.28 [+0.18, +0.34] | +0.25 | 0.68 | 0.71 | 6.4 | 0.33 |
| finger | finger_pat_slope_ms | -0.50 [-0.56, -0.40] | -0.51 | 0.89 | 0.61 | 90.6 | 0.85 |
| finger | finger_pat_foot_ms | -0.50 [-0.57, -0.40] | -0.50 | 0.86 | 0.65 | 91.5 | 0.39 |
| finger | finger_aging_index | -0.23 [-0.32, -0.13] | -0.24 | 0.77 | 0.81 | 7.3 | 0.94 |
| finger | finger_apg_b_a | -0.22 [-0.29, -0.15] | -0.16 | 0.74 | 0.80 | 6.1 | 0.81 |
| finger | finger_h3_h1 | -0.20 [-0.28, -0.16] | -0.24 | 0.80 | 0.65 | 3.1 | 0.80 |
| finger | finger_hr | +0.18 [+0.08, +0.41] | +0.10 | 0.60 | 0.57 | 22.8 | 1.00 |
| finger | finger_wav_e1 | +0.17 [+0.05, +0.24] | +0.05 | 0.66 | 0.62 | 19157.3 | 0.01 |
| finger | finger_apg_c_a | +0.16 [+0.02, +0.24] | +0.13 | 0.54 | 0.63 | 2.9 | 0.74 |
| finger | finger_wav_e2 | +0.16 [+0.05, +0.24] | +0.04 | 0.66 | 0.62 | 19272.8 | 0.72 |
| finger | finger_wav_e3 | +0.16 [+0.04, +0.24] | -0.00 | 0.63 | 0.61 | 34.7 | 0.85 |
| finger | finger_w75_n | +0.15 [+0.05, +0.25] | +0.21 | 0.60 | 0.72 | 11.3 | 0.83 |
| finger | finger_max_slope_n | -0.14 [-0.32, -0.05] | -0.05 | 0.63 | 0.67 | 16.0 | 0.99 |

### V4. Raw-waveform deep models vs baselines (ear; held-out subjects; MAE mmHg; in brackets Δ vs `training_mean` (absolute) or vs `reference_only` (calibrated))

| Target | Method | Same day, absolute | Other day, absolute | Same day, offset-calibrated | Other day, offset-calibrated |
|---|---|---|---|---|---|
| SBP | training_mean | 15.14 | 15.74 | 14.72 | 13.03 |
| SBP | reference_only | — | — | 14.72 | 13.03 |
| SBP | ridge_hr_shape | 15.12 (+0.11 [-0.90, +1.20]) | 14.71 (-0.96 [-1.89, -0.05]*) | 14.67 (-0.41 [-1.49, +0.68]) | 12.41 (-0.42 [-1.39, +0.49]) |
| SBP | cnn | 15.75 (+0.66 [-0.31, +1.73]) | 15.04 (-0.70 [-1.82, +0.35]) | 14.78 (-0.18 [-1.41, +0.99]) | 13.29 (+0.29 [-0.70, +1.38]) |
| SBP | gru | 15.20 (+0.09 [-0.93, +1.11]) | 14.35 (-1.13 [-2.62, +0.23]) | 15.05 (+0.17 [-0.91, +1.26]) | 12.95 (+0.04 [-1.02, +0.90]) |
| SBP | cnn_lstm | 15.11 (-0.03 [-0.65, +0.63]) | 14.89 (-0.75 [-1.55, -0.03]*) | 14.11 (-0.54 [-1.38, +0.20]) | 13.00 (+0.05 [-0.49, +0.56]) |
| SBP | tcn | 15.62 (+0.58 [-0.55, +1.81]) | 14.67 (-1.02 [-2.03, +0.05]) | 14.92 (+0.01 [-1.22, +1.27]) | 13.07 (+0.02 [-1.01, +1.23]) |
| SBP | transformer | 15.13 (+0.10 [-1.09, +1.36]) | 15.10 (-0.65 [-2.17, +0.72]) | 13.91 (-0.92 [-2.04, +0.19]) | 12.61 (-0.26 [-1.37, +0.97]) |
| DBP | training_mean | 10.68 | 9.84 | 9.96 | 10.28 |
| DBP | reference_only | — | — | 9.96 | 10.28 |
| DBP | ridge_hr_shape | 10.69 (-0.01 [-0.57, +0.58]) | 9.52 (-0.14 [-0.92, +0.59]) | 11.22 (+1.05 [+0.32, +1.81]*) | 10.84 (+0.42 [-0.32, +1.15]) |
| DBP | cnn | 11.22 (+0.27 [-0.37, +0.98]) | 10.10 (+0.27 [-0.78, +1.39]) | 10.28 (+0.34 [-0.22, +0.89]) | 10.57 (+0.14 [-0.93, +1.12]) |
| DBP | gru | 11.42 (+0.36 [-0.43, +1.16]) | 10.24 (+0.38 [-0.56, +1.31]) | 10.66 (+0.69 [+0.24, +1.19]*) | 10.72 (+0.35 [-0.15, +0.85]) |
| DBP | cnn_lstm | 10.72 (-0.11 [-0.58, +0.34]) | 9.70 (-0.13 [-0.68, +0.41]) | 10.02 (+0.08 [-0.42, +0.53]) | 10.33 (+0.19 [-0.17, +0.58]) |
| DBP | tcn | 11.78 (+0.85 [+0.04, +1.72]*) | 10.55 (+0.54 [-0.36, +1.50]) | 10.71 (+0.75 [+0.05, +1.53]*) | 10.80 (+0.42 [-0.48, +1.26]) |
| DBP | transformer | 11.63 (+0.60 [-0.30, +1.61]) | 10.09 (+0.26 [-1.04, +1.63]) | 10.16 (+0.25 [-0.40, +0.91]) | 10.18 (+0.02 [-0.99, +0.98]) |

### V5. VitalDB subset (surgical patients, CC BY 4.0)

| statistic | value | unit |
|---|---|---|
| cases | 40.0 | nan |
| cases_eligible_in_vitaldb | 2881.0 | nan |
| total_hours | 160.9 | h |
| windows_total | 28958.0 | nan |
| windows_label_valid | 24847.0 | nan |
| windows_finger_sqi_ok | 23992.0 | nan |
| windows_label_valid_and_sqi_ok | 21662.0 | nan |
| windows_with_pat | 28345.0 | nan |
| nibp_readings | 526.0 | nan |
| art_sbp_mean | 117.5 | mmHg |
| art_sbp_sd | 20.9 | mmHg |
| art_sbp_within_case_sd_median | 15.6 | mmHg |
| age_median | 57.5 | years |
| features_per_window | 49.0 | nan |

### V6. Streaming inference cost (desktop CPU; not a microcontroller measurement)

| Metric | 100 Hz | 1000 Hz |
|---|---|---|
| signal_s | 300.0 | 300.0 |
| windows_evaluated | 59 | 59 |
| startup_bundle_load_ms | 1.05 | 1.05 |
| feature_ms_p50 | 12.61 | 24.66 |
| feature_ms_p95 | 23.16 | 29.34 |
| model_ms_p50 | 3.18 | 6.096 |
| model_ms_p95 | 7.23 | 7.471 |
| throughput_windows_per_s | 52.7 | 32.3 |
| realtime_factor | 268.2 | 164.3 |
| cpu_utilisation | 0.95 | 0.97 |
| peak_python_mem_mb | 0.22 | 0.96 |
| bundle_bytes | 2830 | 2830 |
| machine | Intel64 Family 6 Model 198 Stepping 2, GenuineIntel | Intel64 Family 6 Model 198 Stepping 2, GenuineIntel |
| python | 3.11.9 | 3.11.9 |

<!-- END V3 -->

### Interpretation (3.0)

1. **Within a recording session, yes, but the effect is small.** At both sites, changes in pulse *shape* since the
   last reference predict part of the BP change that follows. HR-only change models never beat the `mean_change`
   control, so this information is not heart rate. At the ear the gain is about 0.6–1.2 mmHg SBP (roughly 5–9 % of
   the carry-forward error), and the CIs exclude zero for references every 2, 5 and 10 min. The calibration study
   agrees: against the matched no-PPG shrinkage control, the PPG-based calibration gains about 0.8 mmHg at the ear
   for k ≥ 3 readings. This refines 2.0 finding 2. The 2.0 offset engine (which applies the population model's
   *absolute* prediction) did not show this, because its errors in absolute level swamp the small change signal.
   Modelling *changes* directly recovers it.
2. **Across days, no.** No change-based model beat the control from a day-1 reference to day-2 windows (all CIs
   include zero), and neither did any calibration method against its no-PPG control. This confirms the 2.0
   cross-day result with methods built specifically to find a change signal.
3. **With real cuff readings in surgical patients (VitalDB), not demonstrated.** Feature changes correlate with
   arterial-line changes (r ≈ 0.5 for HR + shape + PAT), but for SBP no MAE gain over the control is significant
   (34 patients). For DBP only HR + shape + PAT reached significance (about −1.0 mmHg, CI just excluding 0), one cell
   among many tested. There, the cuff itself disagrees with the arterial line by about 13 mmHg MAE (mean −6.7 mmHg) at
   matched times. Carrying a real cuff reading forward was therefore *worse* than the population mean. Reference
   quality limits any reference-anchored method, which matters for RakshPulse's own calibration protocol.
4. **Features.** The most consistent within-person SBP correlates are systolic-area and width ratios, harmonic
   ratios and (with ECG) PAT. Their direction agrees between days for 81–90 % of subjects, and resting values are
   reasonably reproducible between days (ICC about 0.7–0.8). Heart rate is associated on day 1 only (sign agreement
   0.61). Haar relative energies are collinear by construction (VIF up to about 3 × 10⁴) and add little. Ear→finger
   PTT carried no information about BP changes in any analysis, so a two-site *optical* transit-time design gets no
   support from these data.
5. **Calibration practice.** Free per-person linear calibration (slope and offset) is harmful with few readings
   (+3–4 mmHg at k = 3). Shrinking the slope or the offset towards the population is consistently safer. Error grows
   with time since calibration (ear SBP offset MAE about 9 mmHg in the first 5 min, about 14 mmHg after 20 min).
   Conformal 90 % intervals reach nominal coverage, but their half-widths (about 25–30 mmHg) are too wide to be
   clinically useful.
6. **Deep models add nothing here.** No architecture was reliably better than the training-mean or reference-only
   baselines. Of the few significant cells, one favoured a deep model (CNN-LSTM, other-day absolute SBP,
   −0.75 mmHg); the rest were *worse* (TCN and GRU for DBP). The ridge HR + shape model behaved the same way. More
   capacity does not create information the windows lack.

**Answer to the primary question.** PPG waveforms carry a *small, heart-rate-independent* amount of information
about BP changes *within a session*, beyond the last reference reading. That information is not reliable across
days, was not demonstrated with real cuff references in a surgical population, and is far below what a monitoring
device would need. RakshPulse remains a research prototype and is **not** a clinically validated BP monitor. The
data support frequent cuff referencing plus a small, change-based PPG correction within a session. They do not
support PPG-only estimation or calibration carried across days.
