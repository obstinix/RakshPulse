"""Manifest-driven download of open-access PhysioNet projects.

PhysioNet publishes ``SHA256SUMS.txt`` for every project version. This module downloads each
listed file (optionally a filtered subset) with HTTP Range resume, verifies its SHA-256 against
the manifest, sets it read-only, and records provenance. Only **open-access** projects can be
fetched this way. Credentialed projects return HTTP 403, and no attempt is made to bypass that.

Usage::

    python -m rakshpulse.data.physionet auricle --jobs 4
    python -m rakshpulse.data.physionet auricle --records HC_1_1 HC_2_1
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from .download import DATA_ROOT, PROVENANCE_FILE, USER_AGENT


@dataclass(frozen=True)
class PhysioNetProject:
    key: str
    slug: str
    version: str
    title: str
    doi: str
    license: str
    citation: str
    approx_bytes: int

    @property
    def base(self) -> str:
        return f"https://physionet.org/files/{self.slug}/{self.version}/"

    @property
    def zip(self) -> str:
        return f"https://physionet.org/content/{self.slug}/get-zip/{self.version}/"


PROJECTS = {
    "auricle": PhysioNetProject(
        key="auricle",
        slug="auricle",
        version="1.0.0",
        title="Auricle: A Multimodal Wearable Sensing Dataset for Cardiovascular and Hemodynamic Assessment",
        doi="10.13026/swme-vp69",
        license="Open Data Commons Attribution License v1.0 (ODC-By)",
        citation="Auricle (v1.0.0), PhysioNet (2026), doi:10.13026/swme-vp69. Cite also Goldberger et al., PhysioNet, Circulation 101(23) (2000).",
        approx_bytes=15_400_000_000,
    ),
    "vitaldb": PhysioNetProject(
        key="vitaldb",
        slug="vitaldb",
        version="1.0.0",
        title="VitalDB, a high-fidelity multi-parameter vital signs database in surgical patients",
        doi="10.13026/czw8-9p62",
        license="Creative Commons Attribution 4.0 International (CC BY 4.0)",
        citation="Lee HC et al. VitalDB (v1.0.0), PhysioNet (2022), doi:10.13026/czw8-9p62; Sci Data 9:279 (2022).",
        approx_bytes=2_000_000_000,  # disk check for the selected subset only
    ),
}


def _get(url: str, start: int = 0):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    if start:
        req.add_header("Range", f"bytes={start}-")
    return urllib.request.urlopen(req, timeout=60)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(project: PhysioNetProject, dest: Path) -> dict[str, str]:
    man = dest / "SHA256SUMS.txt"
    if not man.exists():
        with _get(project.base + "SHA256SUMS.txt") as r:
            man.write_bytes(r.read())
    out = {}
    for line in man.read_text().splitlines():
        if line.strip():
            digest, rel = line.split(maxsplit=1)
            out[rel.strip()] = digest
    return out


def fetch_file(project: PhysioNetProject, rel: str, digest: str, dest: Path, retries: int = 6) -> dict:
    target = dest / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and sha256(target) == digest:
        return {"file": rel, "sha256": digest, "status": "present"}
    part = target.with_name(target.name + ".part")
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        try:
            with _get(project.base + rel, have) as r:
                mode = "ab" if (have and r.status == 206) else "wb"
                with part.open(mode) as out:
                    shutil.copyfileobj(r, out, 1 << 20)
            break
        except urllib.error.HTTPError as e:
            if e.code == 416:  # already complete
                break
            if e.code in (401, 403):
                raise PermissionError(f"{rel}: access denied (HTTP {e.code}); credentialed data is not fetched") from e
            print(f"  {rel}: HTTP {e.code}, retry {attempt}", file=sys.stderr)
        except OSError as e:
            print(f"  {rel}: {e}, retry {attempt}", file=sys.stderr)
    got = sha256(part)
    if got != digest:
        part.unlink()
        raise RuntimeError(f"{rel}: sha256 mismatch ({got[:12]} != {digest[:12]}); partial file removed")
    part.replace(target)
    target.chmod(0o444)
    return {"file": rel, "sha256": digest, "status": "downloaded"}


def fetch_project(key: str, records: list[str] | None = None, jobs: int = 4) -> list[dict]:
    project = PROJECTS[key]
    dest = DATA_ROOT / "raw" / key
    dest.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(dest).free
    if free < 1.5 * project.approx_bytes:
        raise RuntimeError(f"Need ~{1.5 * project.approx_bytes / 1e9:.0f} GB free, have {free / 1e9:.0f} GB")
    manifest = read_manifest(project, dest)
    items = sorted(manifest.items())
    if records:
        keep = set(records)
        items = [(r, d) for r, d in items if "/" not in r or r.split("/")[1] in keep]
    # metadata first, then large .dat files
    items.sort(key=lambda kv: (kv[0].endswith(".dat"), kv[0]))
    done = []
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(fetch_file, project, r, d, dest): r for r, d in items}
        for i, f in enumerate(as_completed(futs), 1):
            res = f.result()
            done.append(res)
            print(f"[{i}/{len(items)}] {res['status']:10s} {res['file']}", flush=True)
    prov = json.loads(PROVENANCE_FILE.read_text()) if PROVENANCE_FILE.exists() else {}
    prov[key] = {
        "key": key,
        "title": project.title,
        "landing_page": f"https://physionet.org/content/{project.slug}/{project.version}/",
        "doi": project.doi,
        "version": project.version,
        "license": project.license,
        "citation": project.citation,
        "manifest": "SHA256SUMS.txt (all files verified against it)",
        "n_files": len(done),
        "total_bytes": sum((dest / d["file"]).stat().st_size for d in done),
        "subset": records or "all",
        "downloaded_utc": prov.get(key, {}).get("downloaded_utc")
        or dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    PROVENANCE_FILE.write_text(json.dumps(prov, indent=2) + "\n")
    return done


class HttpRangeFile(io.RawIOBase):
    """Read-only, seekable view of a remote file via HTTP Range requests (with read-ahead).

    Lets ``zipfile`` pull single members out of PhysioNet's project ZIP without downloading
    the whole archive. Members are deflate-compressed (~2x), which halves the transfer when the
    server's bandwidth per client is the bottleneck.
    """

    def __init__(self, url: str, block: int = 4 << 20):
        super().__init__()
        self.url, self.block, self.pos = url, block, 0
        for attempt in range(8):
            try:
                req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=60) as r:
                    self.size = int(r.headers["Content-Length"])
                break
            except OSError as e:  # includes HTTP 429/5xx; back off politely
                print(f"  HEAD {url}: {e}; retry {attempt + 1}", file=sys.stderr)
                time.sleep(10 * (attempt + 1))
        else:
            raise OSError(f"HEAD failed for {url}")
        self._buf_start, self._buf = 0, b""

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: off, io.SEEK_CUR: self.pos + off, io.SEEK_END: self.size + off}[whence]
        return self.pos

    def _fetch(self, start: int, end: int) -> bytes:
        for attempt in range(6):
            try:
                req = urllib.request.Request(
                    self.url, headers={"User-Agent": USER_AGENT, "Range": f"bytes={start}-{end}"}
                )
                with urllib.request.urlopen(req, timeout=45) as r:  # per-socket-op timeout
                    data = r.read()
                if len(data) == end - start + 1:
                    return data
            except OSError as e:
                print(f"  range {start}-{end}: {e}; retry {attempt + 1}", file=sys.stderr)
                time.sleep(5 * (attempt + 1))
        raise OSError(f"failed to fetch bytes {start}-{end}")

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        rel = self.pos - self._buf_start
        if not (0 <= rel and rel + n <= len(self._buf)):
            end = min(self.size - 1, self.pos + max(n, self.block) - 1)
            self._buf, self._buf_start, rel = self._fetch(self.pos, end), self.pos, 0
        out = self._buf[rel : rel + n]
        self.pos += len(out)
        return out

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def zip_url(project: PhysioNetProject) -> str:
    return project.zip


def fetch_member_from_zip(project: PhysioNetProject, rel: str, digest: str, dest: Path) -> dict:
    target = dest / rel
    if target.exists() and sha256(target) == digest:
        return {"file": rel, "sha256": digest, "status": "present"}
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    with zipfile.ZipFile(HttpRangeFile(zip_url(project))) as zf:
        name = next(n for n in zf.namelist() if n.endswith("/" + rel) or n == rel)
        with zf.open(name) as src, part.open("wb") as out:  # zipfile also checks the member CRC-32
            shutil.copyfileobj(src, out, 1 << 20)
    got = sha256(part)
    if got != digest:
        part.unlink()
        raise RuntimeError(f"{rel}: sha256 mismatch after zip extraction")
    part.replace(target)
    target.chmod(0o444)
    return {"file": rel, "sha256": digest, "status": "downloaded(zip-member)"}


def fetch_records_via_zip(key: str, records: list[str], jobs: int = 3) -> list[dict]:
    """Fetch the .hea/.dat of ``records`` (in the given priority order) as ZIP members."""
    project = PROJECTS[key]
    dest = DATA_ROOT / "raw" / key
    dest.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(project, dest)
    meta = [(r, d) for r, d in manifest.items() if "/" not in r]
    for r, d in meta:  # small top-level files directly
        fetch_file(project, r, d, dest)
    items = []
    for rec in records:
        for ext in (".hea", ".dat"):
            rel = f"data/{rec}/{rec}{ext}"
            if rel in manifest:
                items.append((rel, manifest[rel]))
    done = []
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(fetch_member_from_zip, project, r, d, dest): r for r, d in items}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                res = f.result()
            except Exception as e:  # report and continue; a rerun retries only missing files
                print(f"[{i}/{len(items)}] FAILED {futs[f]}: {e}", flush=True)
                continue
            done.append(res)
            print(f"[{i}/{len(items)}] {res['status']:22s} {res['file']}", flush=True)
            _record_provenance(project, dest, done, records)
    return done


def _record_provenance(project: PhysioNetProject, dest: Path, done: list[dict], subset) -> None:
    prov = json.loads(PROVENANCE_FILE.read_text()) if PROVENANCE_FILE.exists() else {}
    files = sorted({d["file"] for d in done})
    prov[project.key] = {
        "key": project.key,
        "title": project.title,
        "landing_page": f"https://physionet.org/content/{project.slug}/{project.version}/",
        "doi": project.doi,
        "version": project.version,
        "license": project.license,
        "citation": project.citation,
        "method": (
            "members extracted from the official project ZIP via HTTP Range; each verified against SHA256SUMS.txt"
        ),
        "verified_data_files": len(files),
        "verified_bytes": sum((dest / f).stat().st_size for f in files if (dest / f).exists()),
        "requested_subset": list(subset),
        "downloaded_utc": prov.get(project.key, {}).get("downloaded_utc")
        or dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    PROVENANCE_FILE.write_text(json.dumps(prov, indent=2) + "\n")


def priority_records(key: str) -> list[str]:
    """Healthy cohort session 1, then healthy session 2, then heart-failure cohort."""
    recs = (DATA_ROOT / "raw" / key / "RECORDS").read_text().split()
    recs = [r.split("/")[0] for r in recs]

    def rank(r):
        cohort, num, sess = r.split("_")
        return (cohort != "HC", int(sess) != 1, int(num), int(sess))

    return sorted(recs, key=rank)


REQUIRED_CHANNELS = {"PPG_Ear", "PPG_Finger", "Systolic_BP", "Diastolic_BP"}
ANNOTATION_CHANNELS = {"CNAP_Annotations", "Activity_Annotations"}


def verify_project(key: str, repair: bool = False) -> dict:
    """Re-hash every manifest-listed record file and validate the channel mapping.

    Writes ``data/metadata/<key>_manifest.json`` (committed: hashes and channel facts only, no data).
    A record is ``trusted`` only if both files match SHA256SUMS, all required signal channels exist and
    the CNAP reliability mask exists. ``repair=True`` deletes files whose hash mismatches so that the
    next download run re-fetches them. Corrupt files are never kept silently.
    """
    import wfdb

    project = PROJECTS[key]
    dest = DATA_ROOT / "raw" / key
    manifest = read_manifest(project, dest)
    records = [r.split("/")[0] for r in (dest / "RECORDS").read_text().split()]
    rows = []
    for rid in records:
        row: dict = {"record": rid, "cohort": rid.split("_")[0], "session": int(rid.split("_")[2])}
        status = "ok"
        for ext in (".hea", ".dat"):
            rel = f"data/{rid}/{rid}{ext}"
            p = dest / rel
            if not p.exists():
                status = "partial" if p.with_name(p.name + ".part").exists() else "missing"
                row[ext[1:]] = status
                continue
            if sha256(p) != manifest[rel]:
                row[ext[1:]] = "sha256_mismatch"
                status = "corrupt"
                if repair:
                    p.chmod(0o644)
                    p.unlink()
            else:
                row[ext[1:]] = "verified"
        if status == "ok":
            h = wfdb.rdheader(str(dest / "data" / rid / rid))
            names = set(h.sig_name)
            row.update(
                fs=h.fs,
                duration_s=round(h.sig_len / h.fs, 1),
                n_channels=len(names),
                missing_required=sorted(REQUIRED_CHANNELS - names),
                missing_annotations=sorted(ANNOTATION_CHANNELS - names),
            )
            if row["missing_required"]:
                status = "malformed"
        row["status"] = status
        row["trusted_labels"] = status == "ok" and "CNAP_Annotations" not in row.get("missing_annotations", [])
        rows.append(row)
    summary = {
        "project": key,
        "version": project.version,
        "checked_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "n_records": len(rows),
        "status_counts": {s: sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows})},
        "trusted_records": sum(r["trusted_labels"] for r in rows),
        "records": rows,
    }
    out = DATA_ROOT / "metadata" / f"{key}_manifest.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("project", choices=sorted(PROJECTS))
    ap.add_argument("--records", nargs="*")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--via-zip", action="store_true", help="fetch records as members of the project ZIP")
    ap.add_argument("--verify", action="store_true", help="re-hash all files and write the acquisition manifest")
    ap.add_argument("--repair", action="store_true", help="with --verify: delete corrupt files for re-download")
    a = ap.parse_args(argv)
    if a.verify:
        s = verify_project(a.project, a.repair)
        print(json.dumps({k: v for k, v in s.items() if k != "records"}, indent=2))
        return 0 if not s["status_counts"].get("corrupt") and not s["status_counts"].get("malformed") else 1
    if a.via_zip:
        fetch_records_via_zip(a.project, a.records or priority_records(a.project), a.jobs)
    else:
        fetch_project(a.project, a.records, a.jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
