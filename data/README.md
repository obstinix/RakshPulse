# data/

Datasets are **not** stored in Git. Everything here except `metadata/` and this README is
git-ignored, and every file can be regenerated with the commands below.

```
data/
├── raw/        original archives, read-only after download (checksum-verified)
├── extracted/  archives unpacked exactly as distributed
├── interim/    scratch space (unused at present)
├── processed/  windowed features (*.csv) and waveforms (*.npz) built by rakshpulse.data.build
└── metadata/   provenance.json, validation_report.json, dataset_summary.json  (committed)
```

## Reproduce

```bash
pip install -e ".[dev]"
python -m rakshpulse.data.download guilin_dualwave foggia_multiwave ppg_bp   # ~650 MB, resumable
python scripts/extract_datasets.py                                          # unzip / unrar into data/extracted
python -m rakshpulse.data.dataset_validator                                 # checksums + structure
python -m rakshpulse.data.build guilin_dualwave foggia_multiwave ppg_bp guilin_unlabelled
python -m rakshpulse.data.metadata
```

## Datasets in use

| key | Source | Licence | Raw size | Role |
|---|---|---|---:|---|
| `guilin_dualwave` | Mendeley doi:10.17632/pz2zzr8vhm.1 | CC BY 4.0 | 645.7 MB (1.6 GB extracted) | Training + subject-wise CV |
| `foggia_multiwave` | Mendeley doi:10.17632/gs7jvwtcgt.1 | CC BY 4.0 | 2.0 MB | External test (red+IR) |
| `ppg_bp` | figshare doi:10.6084/m9.figshare.5459299.v5 | CC0 1.0 | 1.5 MB | External test (IR only) |

Download dates, URLs and checksums are recorded in `metadata/provenance.json`.
Attribution: see `docs/DATASET_ACQUISITION.md` for the required citations.
