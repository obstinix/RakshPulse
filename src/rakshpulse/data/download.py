"""Reproducible, resumable dataset acquisition with provenance records.

Every dataset is described in ``DATASETS`` with its official URL, license and the
checksum published by the hosting repository. Downloads are resumable (HTTP Range),
verified against the published checksum, and recorded in
``data/metadata/provenance.json``. Raw files are written once to ``data/raw/<key>/``
and never modified afterwards.

Usage::

    python -m rakshpulse.data.download --list
    python -m rakshpulse.data.download guilin_dualwave ppg_bp
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import sys
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = REPO_ROOT / "data"
PROVENANCE_FILE = DATA_ROOT / "metadata" / "provenance.json"
USER_AGENT = "RakshPulse-ML-dataset-fetcher/0.1"


@dataclass(frozen=True)
class RemoteFile:
    filename: str
    url: str
    size: int
    checksum: str
    algo: str  # "sha256" or "md5"


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    title: str
    landing_page: str
    doi: str
    version: str
    license: str
    citation: str
    files: tuple[RemoteFile, ...]


DATASETS: dict[str, DatasetSpec] = {
    "guilin_dualwave": DatasetSpec(
        key="guilin_dualwave",
        title="A dataset of simultaneous collected ECG and PPG signals (dual-wavelength red/IR PPG + lead-I ECG + cuff BP)",
        landing_page="https://data.mendeley.com/datasets/pz2zzr8vhm/1",
        doi="10.17632/pz2zzr8vhm.1",
        version="1 (published 2026-03-03)",
        license="CC BY 4.0",
        citation="Li, Shiyong (2026). A dataset of simultaneous collected ECG and PPG signals. Mendeley Data, V1. doi:10.17632/pz2zzr8vhm.1",
        files=(
            RemoteFile(
                filename="A dataset of simultaneous collected ECG and PPG signals.zip",
                url="https://data.mendeley.com/public-files/datasets/pz2zzr8vhm/files/01fe8a7c-175b-4a54-b5f0-4633ba63bb3d/file_downloaded",
                size=645718569,
                checksum="6a98ba2e702a6ab2eb98944f856abff644491442c4c5bb006df09a54fc1e4dfd",
                algo="sha256",
            ),
        ),
    ),
    "ppg_bp": DatasetSpec(
        key="ppg_bp",
        title="PPG-BP Database (Guilin People's Hospital, Liang et al. 2018)",
        landing_page="https://doi.org/10.6084/m9.figshare.5459299.v5",
        doi="10.6084/m9.figshare.5459299.v5",
        version="figshare v5 (2022-09-22)",
        license="CC0 1.0",
        citation="Liang Y, Chen Z, Liu G, Elgendi M. A new, short-recorded photoplethysmogram dataset for blood pressure monitoring in China. Sci Data 5, 180020 (2018). doi:10.1038/sdata.2018.20",
        files=(
            RemoteFile(
                filename="PPG-BP Database.zip",
                url="https://ndownloader.figshare.com/files/9441097",
                size=1510347,
                checksum="3b8eae44f45799aeb3c5f55af8589828",
                algo="md5",
            ),
        ),
    ),
    "foggia_multiwave": DatasetSpec(
        key="foggia_multiwave",
        title="A Multiwavelength Photoplethysmography Dataset with Blood Pressure and Heart Rate Reference Measurements (MAX86150, fingertip)",
        landing_page="https://data.mendeley.com/datasets/gs7jvwtcgt/1",
        doi="10.17632/gs7jvwtcgt.1",
        version="1 (2025-06-19)",
        license="CC BY 4.0",
        citation="Data in Brief (2025), PMC12682042. Mendeley Data doi:10.17632/gs7jvwtcgt.1",
        files=(
            RemoteFile(
                filename="PPG_DATASET.rar",
                url="https://data.mendeley.com/public-files/datasets/gs7jvwtcgt/files/b9dc7bce-1001-4e3e-8aa7-3ca114db5909/file_downloaded",
                size=1958773,
                checksum="f11da47b6e5031cee5adee7fa6b12244aab40e218bd7ae6b6d8ac69f86e404a3",
                algo="sha256",
            ),
        ),
    ),
}


def file_checksum(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_resumable(rf: RemoteFile, dest: Path) -> None:
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    if have >= rf.size:
        return
    req = urllib.request.Request(rf.url, headers={"User-Agent": USER_AGENT})
    if have:
        req.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(req, timeout=60) as resp:
        # Server may ignore Range; restart in that case.
        mode = "ab" if (have and resp.status == 206) else "wb"
        if mode == "wb":
            have = 0
        with part.open(mode) as out:
            done = have
            last = -1
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                pct = int(100 * done / rf.size)
                if pct // 5 != last:
                    last = pct // 5
                    print(f"  {rf.filename}: {done / 1e6:.1f}/{rf.size / 1e6:.1f} MB ({pct}%)", flush=True)


def fetch(spec: DatasetSpec, retries: int = 5) -> list[dict]:
    raw_dir = DATA_ROOT / "raw" / spec.key
    raw_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for rf in spec.files:
        dest = raw_dir / rf.filename
        if dest.exists() and file_checksum(dest, rf.algo) == rf.checksum:
            print(f"[skip] {dest.name} already present and verified")
        else:
            free = shutil.disk_usage(raw_dir).free
            if free < rf.size * 3:  # archive + extraction headroom
                raise RuntimeError(f"Insufficient disk space for {rf.filename}: {free / 1e9:.1f} GB free")
            for attempt in range(1, retries + 1):
                try:
                    _download_resumable(rf, dest)
                    break
                except OSError as exc:  # network errors -> resume
                    print(f"  attempt {attempt} failed: {exc}; resuming", file=sys.stderr)
            else:
                raise RuntimeError(f"Failed to download {rf.url}")
            part = dest.with_name(dest.name + ".part")
            got = file_checksum(part, rf.algo)
            if got != rf.checksum:
                raise RuntimeError(f"Checksum mismatch for {rf.filename}: {got} != {rf.checksum}")
            part.replace(dest)
            dest.chmod(0o444)  # raw files are read-only
            print(f"[ok] {dest.name} verified ({rf.algo})")
        records.append(
            {
                "dataset": spec.key,
                "file": str(dest.relative_to(REPO_ROOT)).replace("\\", "/"),
                "url": rf.url,
                "size_bytes": rf.size,
                rf.algo: rf.checksum,
                "verified": True,
                "downloaded_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            }
        )
    return records


def update_provenance(spec: DatasetSpec, records: list[dict]) -> None:
    PROVENANCE_FILE.parent.mkdir(parents=True, exist_ok=True)
    prov = json.loads(PROVENANCE_FILE.read_text()) if PROVENANCE_FILE.exists() else {}
    prev = prov.get(spec.key, {}).get("files", [])
    # Keep the original download date if the file was already recorded.
    first_seen = {r["file"]: r["downloaded_utc"] for r in prev}
    for r in records:
        r["downloaded_utc"] = first_seen.get(r["file"], r["downloaded_utc"])
    meta = asdict(spec)
    meta.pop("files")
    prov[spec.key] = {**meta, "files": records}
    PROVENANCE_FILE.write_text(json.dumps(prov, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="*", help="dataset keys to fetch")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)
    if args.list or not args.datasets:
        for k, s in DATASETS.items():
            total = sum(f.size for f in s.files) / 1e6
            print(f"{k:18s} {total:8.1f} MB  {s.license:10s} {s.title}")
        return 0
    for key in args.datasets:
        spec = DATASETS[key]
        print(f"== {key}: {spec.title}")
        update_provenance(spec, fetch(spec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
