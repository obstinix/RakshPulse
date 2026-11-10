"""Write data/auricle/release_manifest.csv and upload the large Auricle artifacts to GitHub Releases.

python scripts/publish_dataset.py --manifest-only   # checksums + manifest, no upload
python scripts/publish_dataset.py                   # manifest + upload missing assets (needs gh auth)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rakshpulse.data.release import MANIFEST, package_raw, package_waveforms, upload  # noqa: E402


def main() -> int:
    m = pd.concat([package_waveforms(), package_raw()], ignore_index=True)
    m.drop(columns=["local_source"]).to_csv(MANIFEST, index=False)
    print(f"manifest: {len(m)} assets, {m.size_bytes.sum() / 1e9:.2f} GB -> {MANIFEST.relative_to(ROOT)}")
    if "--manifest-only" not in sys.argv:
        upload(m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
