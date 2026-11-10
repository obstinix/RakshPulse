# Dataset Acquisition

## What was downloaded (2026-11-06)

| Dataset | File | Size | Checksum (as published by host) | Verified |
|---|---|---:|---|---|
| Guilin dual-wave | `A dataset of simultaneous collected ECG and PPG signals.zip` | 645,718,569 B | sha256 `6a98ba2e…4dfd` | ✔ |
| Foggia multiwave | `PPG_DATASET.rar` | 1,958,773 B | sha256 `f11da47b…04a3` | ✔ |
| PPG-BP | `PPG-BP Database.zip` | 1,510,347 B | md5 `3b8eae44…9828` | ✔ |

Full URLs, versions and timestamps are in `data/metadata/provenance.json`.
`python -m rakshpulse.data.dataset_validator` re-checks checksums, file counts (444 Guilin
recordings, 127 Foggia, 657 PPG-BP segments), the 250 Hz header and label plausibility.
The last run passed with no errors (`data/metadata/validation_report.json`).

Storage: about 650 MB of archives plus about 1.6 GB extracted. The downloader checks for
free disk space of at least 3× the archive size before starting.

## Auricle (PhysioNet, ODC-By 1.0), added in a follow-up session

- **Why:** this is the only open dataset found with **ear-site PPG** (earlobe, Nonin, transmissive)
  recorded *simultaneously* with finger PPG and continuous BP (CNAP finger-cuff, beat-level SBP/DBP)
  under BP-changing stressors. It allows the finger→ear transfer and per-user calibration studies.
- **Bandwidth constraint:** PhysioNet throttled this client to about 0.65 MB/s in aggregate. Downloading
  the full 15.4 GB would have taken about 6.5 h.
- **Method:** `python -m rakshpulse.data.physionet auricle --via-zip`. Individual records are pulled
  as members of the official project ZIP via HTTP Range requests (deflate halves the transfer). Each
  extracted file is verified against PhysioNet's `SHA256SUMS.txt` (and the ZIP CRC-32), then set read-only.
- **Subset (scientifically motivated, in priority order):** (1) healthy cohort session 1, the closest
  to the target users (healthy adults); (2) healthy session 2 (different day); (3) heart-failure
  cohort. The analysis uses whatever was fully verified when it ran (count in
  `results/auricle_experiments.json` → `meta`). Nothing unverified is used.
- **Exclusions:** `HC_1_1` has no CNAP reliability-mask channel, so its BP labels cannot be screened for
  calibration artefacts and the record is excluded. A few sessions (HC_38_2, HFC_108_1, HFC_124_1) have
  no activity channel. They are kept, but they cannot provide rest-calibration windows.
- **Final acquisition state (2026-11-07):** 79 records verified against SHA256SUMS (42 healthy session 1 and
  37 healthy session 2), 0 corrupt, 0 malformed. 78 have trusted labels (HC_1_1 lacks the CNAP mask). The
  heart-failure cohort was deliberately stopped after the healthy cohort completed: 0/28 records, 3 resumable
  `.part` files. Manifest: `data/metadata/auricle_manifest.json`.
- Raw files stay in `data/raw/auricle/` exactly as published and are read directly (no extraction
  copy). Interim per-record windows go to `data/interim/auricle/`.
- Citation: *Auricle: A Multimodal Wearable Sensing Dataset for Cardiovascular and Hemodynamic
  Assessment* (v1.0.0), PhysioNet (2026), doi:10.13026/swme-vp69, together with the standard PhysioNet
  citation (Goldberger et al., Circulation 2000).

## How the downloader works (`src/rakshpulse/data/download.py`)

- One declarative `DatasetSpec` per dataset, holding the landing page, DOI, version, licence,
  citation, file URLs, sizes and host checksums.
- Downloads go to `*.part` and resume with HTTP `Range` (up to 5 retries).
- After verification the file is renamed and set read-only. A file that is already present
  and verified is skipped, so the same file is never downloaded twice.
- Provenance is merged into `provenance.json`, keeping the original download date.

## Datasets identified but not downloaded

| Dataset | Reason | How to obtain |
|---|---|---|
| Pulse Transit Time PPG (PhysioNet, ODbL, 2.9 GB) | Only 22 subjects, BP only at activity start/end | `wget -r -N -c -np https://physionet.org/files/pulse-transit-time-ppg/1.1.0/` |
| CardioWave-Portable (PhysioNet) | **Credentialed access**: requires a PhysioNet credentialed account, CITI training and a signed DUA | Apply at physionet.org. Not attempted |
| MIMIC-III-Ext-PPG | Credentialed | As above |
| Aurora-BP | DUA application to Microsoft Research | Not attempted |
| PulseDB / MIMIC | Open, but out of scope (ICU invasive ABP, single unspecified-wavelength PPG) | See DATASET_RESEARCH.md |
| VitalDB (40-case subset, 3.0) | Downloaded from PhysioNet, SHA-256 verified, CC BY 4.0. Used only as an external "beyond the cuff" test (surgical patients) | data/README.md |

No dataset was substituted silently. The datasets in use are exactly those listed above.

## Required citations

- Li S. (2026). *A dataset of simultaneous collected ECG and PPG signals*. Mendeley Data V1.
  doi:10.17632/pz2zzr8vhm.1 (CC BY 4.0)
- *A multiwavelength photoplethysmography dataset with blood pressure and heart rate reference
  measurements*. Data in Brief (2025), PMC12682042. Mendeley Data doi:10.17632/gs7jvwtcgt.1 (CC BY 4.0)
- Liang Y, Chen Z, Liu G, Elgendi M. *A new, short-recorded photoplethysmogram dataset for
  blood pressure monitoring in China.* Sci Data 5:180020 (2018). doi:10.1038/sdata.2018.20 (CC0)

## GitHub policy

Raw archives, extracted recordings and per-window tables are git-ignored. Only code,
provenance, validation reports and aggregate statistics are committed.
