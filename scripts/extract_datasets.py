"""Extract downloaded archives into data/extracted/<key>/ (idempotent, raw files untouched).

Uses Python's zipfile for .zip. .rar needs libarchive's `bsdtar`, which ships with
Windows 10+ as C:\\Windows\\System32\\tar.exe and is available as `bsdtar` on Linux/macOS.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rakshpulse.data.download import DATA_ROOT, DATASETS  # noqa: E402


def bsdtar() -> str:
    for cand in (r"C:\Windows\System32\tar.exe", shutil.which("bsdtar")):
        if cand and Path(cand).exists():
            return cand
    raise RuntimeError("bsdtar (libarchive) is required to extract .rar archives")


def main() -> int:
    for key, spec in DATASETS.items():
        out = DATA_ROOT / "extracted" / key
        for rf in spec.files:
            src = DATA_ROOT / "raw" / key / rf.filename
            if not src.exists():
                print(f"[skip] {key}: {rf.filename} not downloaded")
                continue
            marker = out / f".extracted_{rf.checksum[:12]}"
            if marker.exists():
                print(f"[skip] {key}: already extracted")
                continue
            out.mkdir(parents=True, exist_ok=True)
            if src.suffix.lower() == ".zip":
                with zipfile.ZipFile(src) as zf:
                    for m in zf.namelist():  # zip-slip guard
                        if Path(m).is_absolute() or ".." in Path(m).parts:
                            raise RuntimeError(f"unsafe path in archive: {m}")
                    zf.extractall(out)
            else:
                subprocess.run([bsdtar(), "-xf", str(src), "-C", str(out)], check=True)
            marker.touch()
            print(f"[ok] {key}: extracted to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
