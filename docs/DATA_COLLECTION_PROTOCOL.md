# Data Collection Protocol and Dataset Format (RakshPulse 5.0)

This protocol is for RakshPulse's **own** recordings (MAX30102, behind-ear or other sites). It reflects what
2.0–4.0 found: within a session, PPG adds a small amount of information beyond cuff readings; across days it does
not, because of BP-unrelated measurement shift. The protocol is therefore built to measure and control the
day-to-day shift. It is a research protocol, not a clinical validation protocol (see MODEL_EVALUATION.md,
"Validation requirements").

## 1. Ethics and data handling (before any recording)

- Approval from the responsible ethics committee and informed consent from every participant are required.
- **Pseudonymous IDs only** (`P001`, ...). Names, contact details or other identifiers never enter the dataset.
  The key linking IDs to people is kept offline, separately, by the study lead.
- Participant recordings are **not** pushed to the public repository. Only code, documentation and aggregate
  results are. `data/` is git-ignored except the public-dataset tables that were already licence-checked.
- Never edit raw files. Corrections are new derived files. `rakshpulse inspect --validate` detects any change to
  a recorded file through the manifest checksums.

## 2. Equipment and settings to record for every session

| Item | Where it goes |
|---|---|
| Sensor type and firmware version | `session_metadata.csv` → `sensor_type`, `notes` |
| MAX30102 settings (sample rate, averaging, pulse width, ADC range, LED currents) | `sensor_config` (JSON, validated by `MAX30102Config`) |
| Placement (e.g. `behind_ear_left`, `finger_index_right`) | `placement` |
| Contact conditions: mounting method, band tension / force if measured, hair/skin notes | `contact_conditions` |
| Room temperature; MAX30102 die temperature if the firmware reports it (`temperature_c`; this is **not** skin temperature) | `notes`, `raw_ppg.csv` |
| Reference cuff device (make/model, validation status, cuff size, arm) | `reference_bp.csv` → `measurement_device` |

## 3. Session procedure

1. Seated rest for 5 min, cuff on the upper arm at heart level, sensor fitted, no talking.
2. Start recording (`rakshpulse record ...`). Note the session time of each event.
3. Cuff readings at 1, 3 and 5 min of rest, then **every 5 min** (at least 3 per session; at least 60 s apart).
   Enter each one with its session time (`rakshpulse reference ...`, or during recording through the API). The
   reading time is when the cuff reports, usually at the end of deflation.
4. Optional BP-changing manoeuvres, each with a cuff reading during or immediately after: slow breathing, Valsalva
   (only if approved), cold pressor, light exercise, posture change. Log the start and end times in `notes`.
5. End with 3 min of rest and a final cuff reading.

**Repeat-day protocol (the 5.0 research question):** at least 2 sessions on different days per participant, with
the **same** placement, mount and band tension, documented with a photo reference kept with the offline records,
and one deliberately re-seated sensor within a session (remove and refit, logged with its time). This separates
within-session repositioning effects from day-to-day effects. It is the measurement that 4.0 could not make with
Auricle.

## 4. Dataset format

```
<dataset>/
  dataset_manifest.csv      path, kind, rows, bytes, sha256, created_utc
  subjects.csv              subject_id, cohort, notes
  sessions/<subject>/<session>/
    session_metadata.csv
    raw_ppg.csv
    reference_bp.csv
    gaps.csv
    derived/windows[_tag].csv, segments[_tag].csv, ...
```

### `raw_ppg.csv` (one row per sample; written in chunks)

| Column | Type | Meaning |
|---|---|---|
| `timestamp` | float s | session time from `TimestampSync` (device time, or counter / fs) |
| `subject_id`, `session_id` | str | pseudonymous IDs |
| `sample_index` | int | device sample counter (unwrapped). Jumps = missing samples (also listed in `gaps.csv`) |
| `red`, `infrared` | float (ADC counts) | raw MAX30102 counts, 18-bit at 411 µs pulse width; never filtered |
| `temperature_c` | float, optional | MAX30102 die temperature, if the firmware sends it |

### `reference_bp.csv`

| Column | Meaning |
|---|---|
| `subject_id`, `session_id` | pseudonymous IDs |
| `timestamp` | session time of the reading (s) |
| `sbp`, `dbp` | mmHg; plausibility enforced (SBP 50–260, DBP 25–160, SBP > DBP) |
| `heart_rate` | cuff-reported HR (optional) |
| `measurement_device` | device make/model (`simulator` for synthetic readings) |

### `session_metadata.csv`

`subject_id, session_id, recording_date, sensor_type, placement, sampling_rate, contact_conditions, notes,
sensor_config, software_version, simulated` (`simulated` is true exactly when `sensor_type == "simulator"`).

### `gaps.csv`

`timestamp, first_missing_index, n_missing, detected_by` (`counter`, `device_time` or `host_time`).

### `derived/windows.csv` (from `rakshpulse preprocess`)

One row per window: `window_id, t_start_s, t_end_s, first_sample_index, n_samples, status` (`usable`, `degraded`,
`unusable`, `missing`), `reasons` (every rejection, fill or resampling reason), `quality_class`, `quality_score`,
then features: `ir_*` / `red_*` (definitions unchanged since 1.0), `red_ir_corr`, `ratio_of_ratios`, normalised
shape `ir_*_n`, and 4.0 contour features `ir_notch_t` and the rest.

## 5. How cuff readings are matched to PPG

`datastore.association.associate_references` labels a window with a reading only if:
- it is from the same subject and session;
- the reading lies within `tolerance_s` (default 30 s) of the window;
- the window is usable or degraded.

The nearest reading wins. Readings that match no window are reported with a reason. A reading is never copied to
unrelated windows: carrying a value forward is a calibration *method*, evaluated separately.

## 6. Quality control after each session

```bash
rakshpulse inspect --dataset <dataset> --validate     # checksums, schema, ordering, ADC range, references, gaps
rakshpulse preprocess --dataset <dataset> --subject P001 --session D1
```

Re-record if more than 5 % of samples are missing, if fewer than half of the rest windows are usable, or if the
references are implausible. Keep the failed session (never delete raw data) and note why it was excluded.
