"""Publish / retrieve large dataset artifacts through GitHub Releases, verified by SHA-256.

Ordinary git rejects files > 100 MB, and Git LFS has a metered quota. GitHub Release assets may be up to
2 GiB each, with no total-size or bandwidth limit, so the large binaries live there:

* ``auricle-waveforms-v1``: processed NPZ waveform arrays (split into <= ``chunk_mb`` parts if needed);
* ``auricle-raw-wfdb-v1.0.0``: the original PhysioNet WFDB files (ODC-By 1.0, redistributed with attribution).

``data/auricle/release_manifest.csv`` lists every asset with its size, SHA-256 and the original file it
belongs to (``part`` / ``n_parts`` for chunked files), so anyone can download, verify and reassemble:

    python -m rakshpulse.data.release fetch waveforms   # -> data/auricle/waveforms/*.npz
    python -m rakshpulse.data.release fetch raw         # -> data/raw/auricle/data/<record>/*
    python -m rakshpulse.data.release verify            # server-side SHA-256 check of every asset
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import urllib.request
from pathlib import Path

import pandas as pd

from .download import DATA_ROOT, USER_AGENT

REPO = "obstinix/RakshPulse"
BASE_URL = "https://github.com"  # overridable for tests
MANIFEST = DATA_ROOT / "auricle" / "release_manifest.csv"
TAGS = {"waveforms": "auricle-waveforms-v1", "raw": "auricle-raw-wfdb-v1.0.0"}
STAGING = DATA_ROOT / "interim" / "release_staging"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _split(src: Path, chunk_bytes: int) -> list[Path]:
    """Copy ``src`` into staging as one file (if small) or as ``.partNN`` chunks."""
    STAGING.mkdir(parents=True, exist_ok=True)
    size = src.stat().st_size
    if size <= chunk_bytes:
        return [src]
    parts = []
    with src.open("rb") as fh:
        i = 0
        while chunk := fh.read(chunk_bytes):
            p = STAGING / f"{src.name}.part{i:02d}"
            p.write_bytes(chunk)
            parts.append(p)
            i += 1
    return parts


def package_waveforms(chunk_mb: int = 1900) -> pd.DataFrame:
    rows = []
    for src in sorted((DATA_ROOT / "auricle" / "waveforms").glob("*.npz")):
        parts = _split(src, chunk_mb * 1024 * 1024)
        full = sha256(src)
        for i, p in enumerate(parts):
            rows.append(
                {
                    "collection": "waveforms",
                    "release_tag": TAGS["waveforms"],
                    "asset_name": p.name,
                    "local_source": str(p),
                    "original_file": f"data/auricle/waveforms/{src.name}",
                    "part": i,
                    "n_parts": len(parts),
                    "size_bytes": p.stat().st_size,
                    "sha256": sha256(p),
                    "original_sha256": full,
                }
            )
    return pd.DataFrame(rows)


def package_raw() -> pd.DataFrame:
    root = DATA_ROOT / "raw" / "auricle"
    rec = pd.read_csv(DATA_ROOT / "auricle" / "recordings.csv")
    rows = []
    for rid in rec.record_id:
        for ext in (".hea", ".dat"):
            p = root / "data" / rid / f"{rid}{ext}"
            rows.append(
                {
                    "collection": "raw",
                    "release_tag": TAGS["raw"],
                    "asset_name": p.name,
                    "local_source": str(p),
                    "original_file": f"data/raw/auricle/data/{rid}/{p.name}",
                    "part": 0,
                    "n_parts": 1,
                    "size_bytes": p.stat().st_size,
                    "sha256": sha256(p),
                    "original_sha256": sha256(p),
                }
            )
    for name in (
        "LICENSE.txt",
        "README.txt",
        "RECORDS",
        "SHA256SUMS.txt",
        "demographics.csv",
        "activity_labels.csv",
        "sensor_coordinates.csv",
    ):
        p = root / name
        rows.append(
            {
                "collection": "raw",
                "release_tag": TAGS["raw"],
                "asset_name": f"auricle_{name}",
                "local_source": str(p),
                "original_file": f"data/raw/auricle/{name}",
                "part": 0,
                "n_parts": 1,
                "size_bytes": p.stat().st_size,
                "sha256": sha256(p),
                "original_sha256": sha256(p),
            }
        )
    return pd.DataFrame(rows)


def remote_assets(tag: str) -> dict[str, dict]:
    """name -> {size, sha256} for a release, using GitHub's server-side asset digests."""
    import json as _json

    r = subprocess.run(
        ["gh", "api", f"repos/{REPO}/releases/tags/{tag}", "--paginate", "--jq", ".assets[]"],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        return {}
    out = {}
    for line in r.stdout.splitlines():
        a = _json.loads(line)
        out[a["name"]] = {"size": a["size"], "sha256": (a.get("digest") or "").removeprefix("sha256:")}
    return out


def upload(df: pd.DataFrame) -> None:
    """Upload assets with the GitHub CLI. Assets already present with the same SHA-256 are skipped,
    and assets whose remote digest differs are replaced."""
    for tag, g in df.groupby("release_tag"):
        existing = remote_assets(tag)
        for r in g.itertuples():
            if existing.get(r.asset_name, {}).get("sha256") == r.sha256:
                continue
            src = Path(r.local_source)
            target = src if src.name == r.asset_name else STAGING / r.asset_name  # never write into data/raw
            if target != src:
                STAGING.mkdir(parents=True, exist_ok=True)
                target.write_bytes(src.read_bytes())
            subprocess.run(["gh", "release", "upload", tag, str(target), "-R", REPO, "--clobber"], check=True)
            print(f"uploaded {r.asset_name} ({r.size_bytes / 1e6:.1f} MB)", flush=True)


def verify_remote(manifest: Path | None = None) -> pd.DataFrame:
    """Compare every manifest entry with the release's server-side size and SHA-256 digest."""
    m = pd.read_csv(manifest or MANIFEST)
    rows = []
    for tag, g in m.groupby("release_tag"):
        remote = remote_assets(tag)
        for r in g.itertuples():
            ra = remote.get(r.asset_name)
            ok = ra is not None and ra["size"] == r.size_bytes and ra["sha256"] == r.sha256
            rows.append(
                {
                    "release_tag": tag,
                    "asset_name": r.asset_name,
                    "present": ra is not None,
                    "size_match": bool(ra and ra["size"] == r.size_bytes),
                    "sha256_match": bool(ra and ra["sha256"] == r.sha256),
                    "ok": ok,
                }
            )
    return pd.DataFrame(rows)


def fetch(collection: str, dest_root: Path = DATA_ROOT.parent, manifest: Path | None = None) -> list[Path]:
    """Download a collection from the release, verify every asset and reassemble chunked files."""
    m = pd.read_csv(manifest or MANIFEST)
    m = m[m.collection == collection]
    out = []
    for original, g in m.groupby("original_file"):
        target = dest_root / original
        if target.exists() and sha256(target) == g.original_sha256.iloc[0]:
            out.append(target)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".download")
        bad = None
        with tmp.open("wb") as fh:
            for r in g.sort_values("part").itertuples():
                url = f"{BASE_URL}/{REPO}/releases/download/{r.release_tag}/{r.asset_name}"
                req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = resp.read()
                if hashlib.sha256(data).hexdigest() != r.sha256:
                    bad = r.asset_name
                    break
                fh.write(data)
        if bad is not None:  # file is closed here, so it can be removed on Windows too
            tmp.unlink()
            raise RuntimeError(f"checksum mismatch for asset {bad}")
        if sha256(tmp) != g.original_sha256.iloc[0]:
            tmp.unlink()
            raise RuntimeError(f"reassembled {original} does not match its SHA-256")
        tmp.replace(target)
        out.append(target)
        print(f"verified {original}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rakshpulse.data.release")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("collection", choices=["waveforms", "raw"])
    sub.add_parser("verify", help="check every manifest asset against GitHub's server-side SHA-256 digest")
    a = ap.parse_args(argv)
    if a.cmd == "fetch":
        fetch(a.collection)
    elif a.cmd == "verify":
        v = verify_remote()
        print(v.groupby("release_tag").ok.agg(["sum", "count"]).to_string())
        bad = v[~v.ok]
        if len(bad):
            print(bad.to_string(index=False))
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
