# data/

```
data/
├── auricle/    standardised Auricle dataset: populated CSV tables (committed) + waveform NPZ (GitHub Release)
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
(61–195), DBP 70.1 ± 13.4 mmHg. Ear SQI passes in 22,872 windows and finger SQI in 24,813. 2,700 finger windows
are rail-clipped. Age 19–57 (median 23), 50 % female.

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
