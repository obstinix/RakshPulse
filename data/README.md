# data/

```
data/
├── auricle/    standardised Auricle dataset: populated CSV tables (committed) + waveform NPZ (GitHub Release)
├── vitaldb/    3.0: 40-case VitalDB subset, derived CSV tables (committed; raw .vital files git-ignored)
├── metadata/   provenance.json, auricle_manifest.json, validation/summary reports (committed)
├── raw/        original downloads, read-only, checksum-verified (git-ignored; Auricle WFDB also on a Release)
├── extracted/  archives unpacked as distributed (git-ignored)
├── interim/    caches and release staging (git-ignored)
└── processed/  older per-study feature tables for Guilin / Foggia / PPG-BP / Auricle (git-ignored, regenerable)
```

## Auricle standardised dataset (`data/auricle/`)

Source: *Auricle* v1.0.0, PhysioNet, doi:10.13026/swme-vp69, **ODC-By 1.0** (redistribution allowed with
attribution; cite the dataset and PhysioNet). Healthy cohort only: 42 subjects, 79 recordings (42 from session 1,
37 from session 2, recorded on different days), 43.0 h at 1 kHz.

### Where each artifact lives

| Artifact | Format | Size | Location |
|---|---|---:|---|
| `windows.csv`: 30,841 windows × 41 columns | CSV | 8.9 MB | this repo |
| `features.csv`: 30,841 windows × 100 features | CSV | 25.5 MB | this repo |
| `feature_schema.csv`, `subjects.csv`, `recordings.csv`, `summary.csv`, `validation_report.csv` | CSV | < 0.1 MB | this repo |
| `release_manifest.csv`: every release asset with size and SHA-256 | CSV | small | this repo |
| `waveforms/auricle_waveforms_session1.npz` / `session2.npz` | NPZ (float32) | 176 / 162 MB | [Release `auricle-waveforms-v1`](https://github.com/obstinix/RakshPulse/releases/tag/auricle-waveforms-v1) |
| Original WFDB `.hea`/`.dat` (79 records) + PhysioNet metadata | WFDB | 11.7 GB | [Release `auricle-raw-wfdb-v1.0.0`](https://github.com/obstinix/RakshPulse/releases/tag/auricle-raw-wfdb-v1.0.0) |

Why the split: git rejects files over 100 MB (the largest `.dat` is 255 MB), and Git LFS has a metered quota.
GitHub Release assets may be up to 2 GiB each, with no total-size or bandwidth limit, so the binaries are
release assets, verified by SHA-256. No file needed chunking; the release tooling splits automatically if one
ever exceeds 1.9 GB.

```bash
python -m rakshpulse.data.release fetch waveforms   # download + verify NPZ into data/auricle/waveforms/
python -m rakshpulse.data.release fetch raw         # download + verify original WFDB into data/raw/auricle/
# or rebuild everything from PhysioNet:
python -m rakshpulse.data.physionet auricle --via-zip && python -m rakshpulse.data.physionet auricle --verify
python -m rakshpulse.data build && python -m rakshpulse.data validate
```

### Tables

**`windows.csv`**, one row per 10 s window (5 s step):

| Column | Meaning |
|---|---|
| `window_id` | `<record>_w<index>`, primary key shared with `features.csv` and the NPZ files |
| `subject_id`, `record_id`, `cohort`, `session` | identity; session 1 and 2 are different days |
| `t_start_s`, `t_end_s`, `sample_start`, `sample_end`, `source_fs_hz` | position in the original WFDB record (1 kHz samples) |
| `waveform_file`, `waveform_index` | row of this window in the session NPZ |
| `activity_code`, `activity_name`, `activity_pure` | protocol phase (rest, breath_hold, valsalva, exercise, cold_pressor, not_applicable, unlabelled) |
| `cnap_mask_available`, `cnap_valid` | CNAP reliability mask present / valid for the whole window |
| `label_valid`, `label_status` | BP label usable (`ok`), else `cnap_unreliable`, `no_cnap_mask`, `implausible_bp` |
| `sbp_mmHg`, `dbp_mmHg`, `map_mmHg` | median beat-level CNAP pressures in the window; **blank unless `label_valid`** |
| `sbp_range_in_window_mmHg` | SBP range inside the window (label stability) |
| `<site>_sqi_ok`, `<site>_sqi_reasons`, `<site>_template_corr`, `<site>_cardiac_power_ratio`, `<site>_n_beats` | signal quality (site = `ear`, `finger`) |
| `<site>_rail_pct` | % of samples at the ±10 V amplifier rail (> 1 % ⇒ `sqi_ok` false) |
| (`<site>_raw_std` in `features.csv`) | raw SD < 0.003 mV ⇒ flat/disconnected sensor, `sqi_ok` false (`no_signal_flatline`) |
| `<site>_hr_bpm` | PPG-derived heart rate |
| `<site>_calibration_eligible` | rest window with a valid label and good SQI (usable as a calibration reference) |
| `<site>_feature_status` | `complete` / `partial` / `failed` |

**`features.csv`**: 50 features per site, documented with units and definitions in `feature_schema.csv`.
Groups: time domain (raw mean, SD, variance, RMS, peak, peak-to-peak, skewness, kurtosis), morphology (rise time,
widths at 10–90 %, areas, upstroke slope, second-derivative a–e ratios, pulse amplitude), frequency (dominant
frequency, harmonic ratios, spectral entropy, band powers), physiology (HR, IBI mean/SD, SDNN, RMSSD). No BP- or
CNAP-derived quantity is an input feature (checked by the validator). Features are computed for every window.
Filter on `<site>_sqi_ok` before use.

**`subjects.csv`**: demographics (age, sex, height, weight from PhysioNet), sessions, window counts,
cross-day eligibility and exclusion reasons. Recording dates are unavailable (de-identified).
**`recordings.csv`**: channels and units, duration, samples, CNAP-valid fraction, rail fractions and SHA-256 per record.

Unit note: PhysioNet's WFDB headers label every channel `mV`. The BP channels are numerically mmHg and are
reported here as mmHg.

### NPZ waveform arrays

Keys `raw_ear`, `raw_finger` (mV, anti-aliased resampling to 100 Hz), `proc_ear`, `proc_finger` (0.5–10 Hz
band-passed, upward-oriented, z-scored; 100 Hz), `window_id`, `metadata` (JSON with units and the full
preprocessing configuration). Shape `(n_windows, 1000)`, float32.

### Key statistics (from `summary.csv`)

30,841 windows, of which 16,419 are labelled (8,458 in session 1, 7,961 in session 2). Labelled SBP 115.8 ± 19.8 mmHg
(61–195), DBP 70.1 ± 13.4 mmHg. Ear SQI passes in 22,789 windows and finger SQI in 24,813. 2,700 finger windows
are rail-clipped. 3,458 ear and 339 finger windows come from a flat or disconnected sensor (raw SD < 0.003 mV) and
are rejected (`no_signal_flatline`). Whole recordings without ear signal include HC_23_1, HC_23_2, HC_5_2, HC_6_2
and HC_35_2. Age 19–57 (median 23), 50 % female.


### Advanced features (RakshPulse 3.0): `features_advanced.csv`, `feature_schema_advanced.csv`

One row per window (same `window_id` as `windows.csv`), built by
`python -m rakshpulse.data export-advanced --jobs 3` (`src/rakshpulse/data/auricle_advanced.py`):

- `{ear,finger}_pat_foot_ms`, `_pat_slope_ms`, `_pat_foot_sd_ms`: ECG R-peak → PPG foot / max-slope delay
  (median over beats, 80–500 ms). ECG is not available on a MAX30102; this is an upper-bound experiment.
- `{ear,finger}_wav_e1..e6`, `_wav_a6`: relative energies of an orthonormal 6-level Haar decomposition of the
  processed 100 Hz window (they sum to 1, so they are collinear by construction; see VIF in RESEARCH_FINDINGS V3).
- `ear_finger_ptt_ms`, `_sd_ms`: ear foot → next finger foot delay (0–300 ms); `ear_finger_corr`.

PAT is available in 22,542 ear and 24,579 finger windows; median ear PAT 220 ms, finger 239 ms, ear→finger
delay 24 ms.

## VitalDB subset (`data/vitaldb/`, RakshPulse 3.0)

Source: *VitalDB, a high-fidelity multi-parameter vital signs database in surgical patients* v1.0.0, PhysioNet,
doi:10.13026/czw8-9p62, **CC BY 4.0** (redistribution of derived data allowed with attribution; cite Lee et al.,
Sci Data 9:279 (2022) and PhysioNet). Surgical patients under anaesthesia, so it is **not** a wearable population.
It is used because it is the only open dataset found with fingertip PPG, ECG, a continuous arterial line *and*
real intermittent cuff (NIBP) readings in the same recordings. That makes it a test of "beyond the last real cuff
reading".

Cases: 40 cases drawn at random (seed 0) from the 2,881 that have all required tracks
(`data/metadata/vitaldb_selection.csv`; track list from the public VitalDB API, metadata only). The `.vital` files
are downloaded from PhysioNet into `data/raw/vitaldb/` and SHA-256 verified (git-ignored). Reading needs the optional
`vitaldb` package: `pip install -e .[vitaldb]`.

```bash
python -m rakshpulse.data.physionet vitaldb          # or download only the selected files
python -m rakshpulse.data.vitaldb_dataset select     # re-creates the fixed selection
python -m rakshpulse.data.vitaldb_dataset build --jobs 3
python -m rakshpulse.data.vitaldb_dataset summarize
```

| File | Rows | Content |
|---|---|---|
| `cases.csv` | 40 | duration, window counts, NIBP count, demographics and surgery type from `clinical_data.csv` |
| `windows.csv` | 28,958 | 10 s windows every 20 s; arterial-line SBP/DBP/MBP labels (median of the monitor numerics; valid only if plausible and the in-window SBP range ≤ 30 mmHg); finger SQI; PAT availability |
| `features.csv` | 28,958 | finger HR + pulse-shape features (same pipeline as Auricle), Haar energies, ECG→PLETH PAT |
| `nibp_readings.csv` | 526 | the real intermittent cuff readings (time, SBP, DBP) |
| `summary.csv` | – | key statistics |

Key statistics: 160.9 h; 24,847 label-valid windows, 21,662 also SQI-accepted; arterial SBP 117.5 ± 20.9 mmHg
(median within-case SD 15.6); median age 57.5. The monitor's PLETH lags its ECG by about 0.4–0.5 s, so PAT here
uses a 300–1000 ms search range. Absolute PAT is therefore not physiological; only within-case *changes* are used.


## RakshPulse 4.0 additions

### `data/auricle/features_physio.csv`

Pulse-contour landmarks per window and site (`ear_*`, `finger_*`): `notch_present`, `notch_t`, `notch_h`,
`dia_peak_h`, `dia_peak_t`, `sys_dia_dt`, `ipa`, `upstroke_10_90`, `decay_tau`, `beat_amp_cv`,
`beat_template_corr`. These are computed on the median unit-beat template of the processed 100 Hz window. Times
are beat fractions and heights are relative to the systolic peak. Build with
`python -m rakshpulse.data export-physio --jobs 3`. The per-window templates are cached in
`data/interim/auricle_templates.npz` (git-ignored, regenerable).

### VitalDB extension (`data/vitaldb_ext/`)

60 further VitalDB v1.0.0 cases (CC BY 4.0) were drawn at random (seed 1) from the 2,881 eligible cases,
**excluding** all 40 cases of the 3.0 subset. They are used only as independent validation patients: models
fitted on the 3.0 cases are evaluated on them without refitting (RESEARCH_FINDINGS W5). The tables and their
format are the same as for `data/vitaldb/`.

```bash
python -m rakshpulse.data.vitaldb_dataset select --extension --n 60 --seed 1
python -m rakshpulse.data.physionet vitaldb --records <files listed in data/metadata/vitaldb_selection_ext.csv>
python -m rakshpulse.data.vitaldb_dataset build --set ext --jobs 3
```

Key statistics: 222.0 h; 29,898 label-valid and SQI-accepted windows; 920 NIBP readings; arterial SBP
122.1 ± 21.1 mmHg (median within-case SD 16.1); median age 60.5. Provenance (`data/metadata/provenance.json`)
lists all 106 downloaded VitalDB files (40 + 60 cases + metadata) with a download history.

### How the datasets differ (they are not interchangeable)

| | Auricle (healthy cohort) | VitalDB (3.0 subset and 4.0 extension) |
|---|---|---|
| Population | 42 healthy adults, 19–57 y (median 23) | surgical patients under anaesthesia, median age 57.5 / 60.5 |
| PPG sensor / site | transmissive PPG at the finger and the earlobe, simultaneous | monitor finger plethysmogram (SNUADC/PLETH) |
| Sampling | 1 kHz (resampled to 100 Hz for features) | 500 Hz (resampled to 100 Hz) |
| BP reference | CNAP continuous non-invasive (with reliability mask) and arm cuff | invasive arterial line (truth) and intermittent NIBP cuff (reference readings) |
| Reference timing | continuous; references simulated on a grid | real NIBP every ~14–18 min on average |
| Conditions | lab protocol: rest, breath hold, Valsalva, exercise, cold pressor; 2 days | induction, maintenance, vasoactive drugs; one session per patient |
| Format | WFDB (+ standardised CSV/NPZ) | `.vital` (+ standardised CSV) |
| Notes | AC-coupled PPG: no perfusion index | PLETH lags ECG ~0.4–0.5 s, so absolute PAT is not physiological |

## Other datasets (earlier Guilin study)

| key | Source | Licence | Raw size | Role |
|---|---|---|---:|---|
| `guilin_dualwave` | Mendeley doi:10.17632/pz2zzr8vhm.1 | CC BY 4.0 | 645.7 MB | Training + subject-wise CV |
| `foggia_multiwave` | Mendeley doi:10.17632/gs7jvwtcgt.1 | CC BY 4.0 | 2.0 MB | External test (red+IR) |
| `ppg_bp` | figshare doi:10.6084/m9.figshare.5459299.v5 | CC0 1.0 | 1.5 MB | External test (IR only) |

```bash
python -m rakshpulse.data.download guilin_dualwave foggia_multiwave ppg_bp
python scripts/extract_datasets.py && python -m rakshpulse.data.dataset_validator
python -m rakshpulse.data.build guilin_dualwave foggia_multiwave ppg_bp guilin_unlabelled
```
