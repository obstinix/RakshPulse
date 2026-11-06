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
| SBP | ear | within_day | carry_forward | 14.98 | [13.03, 17.03] | 19.14 | 5.54 | 18.33 | — | — | 38 |
| SBP | ear | within_day | uncalibrated | 14.99 | [12.91, 17.35] | 18.87 | -0.24 | 18.87 | 0.41 | 0.39 | 38 |
| SBP | ear | within_day | offset | 13.85 | [11.89, 15.99] | 18.00 | 3.90 | 17.57 | 0.41 | 0.39 | 38 |
| SBP | ear | within_day_s2 | carry_forward | 13.28 | [11.11, 15.63] | 17.64 | 1.80 | 17.55 | — | — | 29 |
| SBP | ear | within_day_s2 | uncalibrated | 15.44 | [12.58, 18.50] | 19.59 | -3.60 | 19.26 | 0.41 | 0.35 | 29 |
| SBP | ear | within_day_s2 | offset | 12.23 | [10.27, 14.51] | 16.18 | 0.13 | 16.18 | 0.41 | 0.35 | 29 |
| SBP | ear | cross_day | carry_forward | 16.53 | [13.84, 19.12] | 20.91 | 1.67 | 20.85 | — | — | 31 |
| SBP | ear | cross_day | uncalibrated | 15.14 | [12.53, 17.92] | 19.25 | -3.05 | 19.01 | 0.38 | 0.32 | 31 |
| SBP | ear | cross_day | offset | 16.18 | [13.55, 19.00] | 20.28 | -0.11 | 20.28 | 0.38 | 0.32 | 31 |
| SBP | ear | cross_day | linear | 14.19 | [11.76, 16.90] | 18.42 | -4.78 | 17.80 | 0.38 | 0.33 | 31 |
| SBP | ear | cross_day | personal model (day-1 fit) | 14.71 | [12.08, 17.75] | 19.73 | -3.73 | 19.38 | 0.33 | 0.31 | 32 |
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
| DBP | ear | within_day | carry_forward | 8.95 | [7.38, 10.74] | 12.64 | -0.89 | 12.61 | — | — | 38 |
| DBP | ear | within_day | uncalibrated | 11.01 | [8.95, 13.06] | 15.21 | -0.63 | 15.20 | 0.21 | 0.21 | 38 |
| DBP | ear | within_day | offset | 9.45 | [7.98, 11.13] | 13.54 | -2.09 | 13.38 | 0.21 | 0.21 | 38 |
| DBP | ear | within_day_s2 | carry_forward | 8.87 | [7.15, 10.83] | 11.68 | -2.57 | 11.40 | — | — | 29 |
| DBP | ear | within_day_s2 | uncalibrated | 9.98 | [7.82, 12.81] | 13.19 | -1.87 | 13.06 | 0.11 | 0.12 | 29 |
| DBP | ear | within_day_s2 | offset | 9.00 | [7.33, 10.94] | 11.77 | -2.93 | 11.40 | 0.11 | 0.12 | 29 |
| DBP | ear | cross_day | carry_forward | 11.60 | [8.99, 14.48] | 14.87 | -1.39 | 14.80 | — | — | 31 |
| DBP | ear | cross_day | uncalibrated | 9.97 | [8.04, 12.41] | 13.11 | -1.24 | 13.05 | 0.13 | 0.11 | 31 |
| DBP | ear | cross_day | offset | 12.79 | [10.13, 15.78] | 16.03 | -2.76 | 15.79 | 0.13 | 0.11 | 31 |
| DBP | ear | cross_day | linear | 10.53 | [8.28, 13.30] | 14.03 | -1.68 | 13.93 | 0.12 | 0.09 | 31 |
| DBP | ear | cross_day | personal model (day-1 fit) | 12.12 | [9.44, 15.45] | 16.32 | -1.89 | 16.22 | 0.27 | 0.26 | 32 |
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
| SBP | ear | hr_only | within_day | 14.71 | [12.73, 16.89] | 0.28 | [0.16, 0.40] |
| SBP | ear | hr_only | cross_day | 16.16 | [13.71, 18.63] | 0.09 | [-0.13, 0.24] |
| SBP | ear | shape_only | within_day | 14.14 | [12.04, 16.48] | 0.42 | [0.37, 0.48] |
| SBP | ear | shape_only | cross_day | 17.52 | [14.78, 20.51] | 0.38 | [0.27, 0.52] |
| SBP | ear | hr_shape | within_day | 13.85 | [11.89, 15.99] | 0.41 | [0.26, 0.46] |
| SBP | ear | hr_shape | cross_day | 16.18 | [13.55, 19.00] | 0.38 | [0.26, 0.45] |
| SBP | ear | ppg_all | within_day | 13.68 | [11.78, 15.84] | 0.39 | [0.24, 0.48] |
| SBP | ear | ppg_all | cross_day | 15.77 | [13.25, 18.51] | 0.38 | [0.19, 0.47] |
| SBP | finger | hr_only | within_day | 15.55 | [13.34, 17.88] | 0.33 | [0.15, 0.41] |
| SBP | finger | hr_only | cross_day | 15.44 | [13.21, 17.83] | 0.17 | [-0.02, 0.28] |
| SBP | finger | shape_only | within_day | 15.38 | [13.32, 17.58] | 0.27 | [0.17, 0.37] |
| SBP | finger | shape_only | cross_day | 15.55 | [13.03, 18.33] | 0.21 | [0.15, 0.33] |
| SBP | finger | hr_shape | within_day | 15.45 | [13.33, 17.66] | 0.30 | [0.18, 0.40] |
| SBP | finger | hr_shape | cross_day | 15.54 | [12.98, 18.22] | 0.25 | [0.13, 0.36] |
| SBP | finger | ppg_all | within_day | 15.17 | [13.12, 17.45] | 0.33 | [0.20, 0.44] |
| SBP | finger | ppg_all | cross_day | 14.98 | [12.49, 17.68] | 0.35 | [0.25, 0.39] |
| DBP | ear | hr_only | within_day | 9.09 | [7.61, 10.79] | 0.18 | [0.09, 0.28] |
| DBP | ear | hr_only | cross_day | 11.69 | [9.29, 14.25] | 0.02 | [-0.11, 0.13] |
| DBP | ear | shape_only | within_day | 9.45 | [7.92, 11.19] | 0.18 | [0.11, 0.35] |
| DBP | ear | shape_only | cross_day | 12.81 | [10.06, 15.93] | 0.13 | [0.03, 0.24] |
| DBP | ear | hr_shape | within_day | 9.45 | [7.98, 11.13] | 0.21 | [0.12, 0.42] |
| DBP | ear | hr_shape | cross_day | 12.79 | [10.13, 15.78] | 0.13 | [-0.03, 0.22] |
| DBP | ear | ppg_all | within_day | 9.43 | [7.96, 11.07] | 0.22 | [0.11, 0.34] |
| DBP | ear | ppg_all | cross_day | 12.64 | [10.10, 15.47] | 0.14 | [0.03, 0.24] |
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
| svr | 13.8 | 2.25 | 5.0 | 1.81 |
| elasticnet | 13.73 | 2.5 | 4.0 | 1.72 |
| extra_trees | 13.92 | 3.25 | 4.0 | 1.74 |
| ridge | 14.06 | 4.5 | 7.0 | 1.91 |
| xgboost | 14.5 | 4.75 | 6.0 | 1.93 |
| random_forest | 14.54 | 5.0 | 6.0 | 2.0 |
| hist_gb | 15.15 | 5.75 | 7.0 | 2.03 |

| comparison | mae_diff_a_minus_b | ci_low | ci_high | significant |
|---|---|---|---|---|
| ear sbp cross_day: elasticnet vs ridge (offset hr_shape) | 0.12 | -0.56 | 0.77 | False |
| ear sbp cross_day: svr vs ridge (offset hr_shape) | 0.02 | -0.8 | 0.87 | False |
| ear sbp cross_day: random_forest vs ridge (offset hr_shape) | 0.03 | -1.44 | 1.52 | False |
| ear sbp cross_day: extra_trees vs ridge (offset hr_shape) | -0.31 | -1.33 | 0.66 | False |
| ear sbp cross_day: hist_gb vs ridge (offset hr_shape) | -0.66 | -2.6 | 1.23 | False |
| ear sbp cross_day: xgboost vs ridge (offset hr_shape) | -0.63 | -1.89 | 0.57 | False |
| ear dbp cross_day: elasticnet vs ridge (offset hr_shape) | -0.71 | -1.37 | -0.08 | True |
| ear dbp cross_day: svr vs ridge (offset hr_shape) | 0.0 | -0.97 | 1.02 | False |
| ear dbp cross_day: random_forest vs ridge (offset hr_shape) | 0.57 | -0.55 | 1.75 | False |
| ear dbp cross_day: extra_trees vs ridge (offset hr_shape) | -0.01 | -0.8 | 0.79 | False |
| ear dbp cross_day: hist_gb vs ridge (offset hr_shape) | 1.52 | 0.03 | 3.2 | True |
| ear dbp cross_day: xgboost vs ridge (offset hr_shape) | 0.75 | -0.51 | 2.15 | False |
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
| ear sbp within_day: offset hr_shape vs carry-forward | -1.03 | -1.82 | -0.27 | True |
| ear sbp within_day: offset hr_only vs carry-forward | -0.4 | -1.06 | 0.3 | False |
| ear sbp within_day: offset hr_shape vs offset hr_only | -0.64 | -1.31 | -0.03 | True |
| ear sbp within_day: offset shape_only vs offset hr_only | -0.28 | -1.2 | 0.54 | False |
| ear sbp cross_day: offset hr_shape vs carry-forward | -0.63 | -1.87 | 0.62 | False |
| ear sbp cross_day: offset hr_only vs carry-forward | -0.84 | -2.15 | 0.32 | False |
| ear sbp cross_day: offset hr_shape vs offset hr_only | 0.21 | -0.99 | 1.44 | False |
| ear sbp cross_day: offset shape_only vs offset hr_only | 1.85 | 0.15 | 3.67 | True |
| ear sbp cross_day: linear hr_shape vs offset hr_shape | -2.43 | -5.57 | 0.81 | False |
| ear sbp cross_day: personal hr_shape vs personal hr_only | 1.7 | 0.21 | 3.16 | True |
| ear sbp cross_day: personal hr_shape vs carry-forward | -1.86 | -5.15 | 1.81 | False |
| ear dbp within_day: offset hr_shape vs carry-forward | 0.4 | -0.03 | 0.87 | False |
| ear dbp within_day: offset hr_only vs carry-forward | 0.13 | -0.14 | 0.4 | False |
| ear dbp within_day: offset hr_shape vs offset hr_only | 0.27 | -0.12 | 0.68 | False |
| ear dbp within_day: offset shape_only vs offset hr_only | 0.25 | -0.18 | 0.7 | False |
| ear dbp cross_day: offset hr_shape vs carry-forward | 1.17 | 0.33 | 2.1 | True |
| ear dbp cross_day: offset hr_only vs carry-forward | -0.04 | -0.54 | 0.47 | False |
| ear dbp cross_day: offset hr_shape vs offset hr_only | 1.21 | 0.24 | 2.31 | True |
| ear dbp cross_day: offset shape_only vs offset hr_only | 1.19 | 0.14 | 2.38 | True |
| ear dbp cross_day: linear hr_shape vs offset hr_shape | -2.07 | -4.59 | 0.24 | False |
| ear dbp cross_day: personal hr_shape vs personal hr_only | 1.78 | 0.45 | 3.1 | True |
| ear dbp cross_day: personal hr_shape vs carry-forward | 0.89 | -1.86 | 3.89 | False |
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
| ear sbp cross_day: finger-trained vs ear-trained (offset) | 0.53 | -1.86 | 2.54 | False |
| ear sbp cross_day: finger-trained vs ear-trained (linear) | -0.15 | -0.63 | 0.42 | False |
| sbp cross_day: ear vs finger (offset hr_shape, each site's own windows) | 0.72 | -1.13 | 2.73 | False |
| ear dbp cross_day: finger-trained vs ear-trained (offset) | -0.9 | -1.91 | 0.01 | False |
| ear dbp cross_day: finger-trained vs ear-trained (linear) | -0.25 | -0.63 | 0.15 | False |
| dbp cross_day: ear vs finger (offset hr_shape, each site's own windows) | 0.54 | -0.49 | 1.63 | False |

### R6. Day-to-day resting BP shift and calibration stability

| Site | Target | n | True rest shift mean | SD | Slope pred-shift~true | Slope 95% CI | r | r 95% CI | Min detectable |r| | Offset day1~day2 r | r 95% CI | Mean |offset change| |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ear | SBP | 29 | 4.66 | 15.38 | 0.02 | [-0.04, 0.09] | 0.09 | [-0.30, 0.37] | 0.50 | 0.43 | [0.08, 0.71] | 11.91 |
| ear | DBP | 29 | 2.94 | 11.65 | -0.03 | [-0.16, 0.05] | -0.14 | [-0.46, 0.23] | 0.50 | 0.34 | [0.08, 0.64] | 9.56 |
| finger | SBP | 33 | 3.10 | 16.40 | -0.01 | [-0.11, 0.10] | -0.04 | [-0.43, 0.40] | 0.47 | 0.37 | [-0.01, 0.69] | 12.46 |
| finger | DBP | 33 | 1.25 | 11.74 | 0.02 | [-0.07, 0.09] | 0.12 | [-0.27, 0.44] | 0.47 | 0.44 | [0.22, 0.69] | 8.81 |

### R7. Amplifier rail-clipping gate ablation (ridge, HR + shape, offset)

Finger windows clipped: 8.1 %; ear: 0.2 %. Subjects with any clipped finger window: 29; with > 25 % clipped: 5.

| Site | Target | Scope | MAE no gate | MAE gated | Δ | r no gate | r gated | Windows removed |
|---|---|---|---|---|---|---|---|---|
| ear | SBP | cross_day | 16.18 | 16.18 | 0.00 | 0.38 | 0.38 | 0 |
| ear | SBP | within_day | 13.85 | 13.85 | 0.00 | 0.41 | 0.41 | 0 |
| ear | SBP | within_day_s2 | 12.23 | 12.23 | 0.00 | 0.41 | 0.41 | 0 |
| ear | DBP | cross_day | 12.79 | 12.79 | 0.00 | 0.12 | 0.12 | 0 |
| ear | DBP | within_day | 9.45 | 9.45 | 0.00 | 0.21 | 0.21 | 0 |
| ear | DBP | within_day_s2 | 9.00 | 9.00 | 0.00 | 0.11 | 0.11 | 0 |
| finger | SBP | cross_day | 15.40 | 15.54 | 0.15 | 0.26 | 0.25 | 283 |
| finger | SBP | within_day | 14.97 | 15.45 | 0.48 | 0.33 | 0.30 | 490 |
| finger | SBP | within_day_s2 | 12.30 | 12.19 | -0.12 | 0.28 | 0.24 | 277 |
| finger | DBP | cross_day | 11.73 | 11.72 | -0.02 | 0.04 | 0.06 | 283 |
| finger | DBP | within_day | 9.62 | 9.84 | 0.22 | 0.04 | 0.05 | 490 |
| finger | DBP | within_day_s2 | 9.61 | 9.36 | -0.25 | 0.06 | 0.04 | 277 |

### R8. Confounds and subject structure (SBP)

| Site | Within-session r(SBP, HR) | Partial r(SBP, shape pred | HR) | 95% CI | Between-subject share of pred. variance | r(subject-mean pred, truth) |
|---|---|---|---|---|---|
| ear | 0.19 | 0.23 | [0.17, 0.29] | 0.39 | 0.13 |
| finger | 0.23 | 0.11 | [0.02, 0.17] | 0.26 | -0.03 |

<!-- END AUTO -->

## Interpretation (from the audited results above)

**1. Relative tracking within a session: modest but real, and better at the ear.**
Within session 1, ear PPG (HR + shape, ridge) tracks SBP changes with a median within-subject r of 0.41 (finger 0.30).
With a 20 s rest calibration it beats carry-forward by 1.03 mmHg (95 % CI [−1.82, −0.27]). Pulse shape carries
SBP information **beyond heart rate at the ear**: within-session partial r(SBP, shape prediction | HR) = 0.23
[0.17, 0.29], against 0.12 [0.02, 0.17] at the finger. At the ear, HR + shape beats HR-only by 0.64 mmHg [−1.31, −0.03].
The stressors therefore do not explain the tracking through heart rate alone. The effect, however, is small:
absolute errors stay at 12–16 mmHg MAE.

**2. Cross-day: a day-1 calibration does not hold on day 2.**
- True resting SBP changed between days by 4.7 ± 15.4 mmHg (ear subjects), and DBP by 2.9 ± 11.7.
- The model's predicted shift does **not** follow it. The slope of predicted on true shift is 0.02, 95 % CI
  [−0.04, 0.09] for SBP at the ear, and similar for every site and target. The CI **excludes 1**, so the
  failure to track day-level baseline changes is statistically supported.
- Each subject's calibration offset is only weakly reproducible across days (day-1 vs day-2 offset r 0.34–0.44,
  mean |change| 9–12 mmHg).
- So a 1-point day-1 calibration did not beat carry-forward on day 2 for SBP (ear −0.63 [−1.87, +0.62]; finger
  −0.22 [−1.49, +1.02]). For ear DBP it was **worse** than carry-forward (+1.17 [+0.33, +2.10]).
- Applying day-1 calibration *hurt* relative to the uncalibrated population model (ear DBP +2.80 [+0.22, +5.66]),
  because a stale offset adds noise.

**3. What survives across days is within-session *shape* tracking, not absolute level.**
Within-subject SBP correlation on day 2 using day-1-trained models stays at r 0.38 for ear shape features (HR-only
drops to 0.09). PPG shape relates to *changes* within a recording, but carries almost no information about a
person's *absolute* BP level or its day-to-day drift. Subject-mean predictions correlate with subject-mean truth at
r 0.13 (ear) and −0.03 (finger).

**4. Independence is not proven.** The correlation between true and predicted day shifts is near zero (ear SBP r
0.09 [−0.30, 0.37]), but with n = 29–33 the minimum correlation detectable at 80 % power is |r| ≈ 0.5. The data
therefore **do not establish independence**: moderate associations up to about 0.4 cannot be excluded. What *is*
established is that the slope is far below 1, so the model cannot be relied on to follow day-to-day changes.

**5. DBP behaves differently from SBP.** No PPG feature set improved DBP over carry-forward at any scope.
Finger DBP tracking r was ≈ 0.05, and ear DBP reached 0.21 within day but only 0.13 across days.

**6. Ear vs finger.** The ear showed stronger within-session SBP tracking and the only clear evidence of shape
information beyond heart rate. Cross-day absolute errors did not differ between sites (ear − finger SBP
+0.72 [−1.13, +2.73]). A finger-trained model with ear calibration was not significantly worse than an
ear-trained one here (+0.53 [−1.86, +2.54]). In session 1, however, finger→ear transfer lost tracking
(MODEL_EVALUATION.md §F3).

**7. Model choice does not matter at this sample size.** By the pre-specified criterion (mean cross-day MAE rank),
SVR ranks first and ElasticNet second (ElasticNet has the lowest mean MAE and best worst-case rank). **No
model differs significantly from ridge** except ElasticNet for ear DBP (−0.71 [−1.37, −0.08]) and
HistGradientBoosting being worse for ear DBP (+1.52 [+0.03, +3.20]). We recommend the regularised linear models
(ridge / ElasticNet): they are as good as any, simpler, and portable to a microcontroller.

**8. Amplifier clipping.** Removing rail-clipped windows (8.1 % of finger windows, 29 subjects affected, 5 with
> 25 % clipped) changed finger MAEs by ≤ 0.5 mmHg and never changed a conclusion. Ear data were unaffected.

**9. Population models vs simple baselines.** Calibration-free population predictions were not better than
carry-forward within a day. Across days they were numerically better than carry-forward (ear SBP −2.54
[−6.06, +0.93]) only because carry-forward suffers from day-to-day drift, and the CIs include 0.

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
