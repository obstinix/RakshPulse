"""Dataset command line.

python -m rakshpulse.data build            # Auricle -> data/auricle/ (CSV tables + NPZ waveforms)
python -m rakshpulse.data validate         # integrity, schema, label and waveform-reference checks
python -m rakshpulse.data export-features  # (re)write features.csv + feature_schema.csv from a built dataset
python -m rakshpulse.data summary          # print and write summary.csv
python -m rakshpulse.data package          # split waveform NPZ into checksummed release chunks
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rakshpulse.data")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--jobs", type=int, default=4)
    v = sub.add_parser("validate")
    v.add_argument("--no-raw", action="store_true", help="skip checks that need the raw WFDB files")
    sub.add_parser("export-features")
    xa = sub.add_parser("export-advanced", help="RakshPulse 3.0 PAT / PTT / wavelet table (features_advanced.csv)")
    xa.add_argument("--jobs", type=int, default=4)
    xp = sub.add_parser("export-physio", help="RakshPulse 4.0 pulse-contour table (features_physio.csv)")
    xp.add_argument("--jobs", type=int, default=4)
    sub.add_parser("summary")
    p = sub.add_parser("package")
    p.add_argument("--chunk-mb", type=int, default=1900)
    a = ap.parse_args(argv)

    from . import auricle_dataset as ds

    if a.cmd == "build":
        print(ds.build(a.jobs).to_string(index=False))
    elif a.cmd == "validate":
        rep = ds.validate(check_raw=not a.no_raw)
        with pd.option_context("display.max_colwidth", 120, "display.width", 200):
            print(rep.to_string(index=False))
        return 1 if (rep.status == "FAIL").any() else 0
    elif a.cmd == "export-features":
        ds.feature_schema().to_csv(ds.OUT_DIR / "feature_schema.csv", index=False)
        print(f"feature_schema.csv: {len(ds.feature_schema())} documented features")
    elif a.cmd == "export-physio":
        from .auricle_advanced import build_physio

        build_physio(a.jobs)
    elif a.cmd == "export-advanced":
        from .auricle_advanced import build_advanced

        build_advanced(a.jobs)
    elif a.cmd == "summary":
        print(ds.summary().to_string(index=False))
    elif a.cmd == "package":
        from .release import package_waveforms

        print(package_waveforms(a.chunk_mb).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
