"""Session-based CSV dataset store for recordings made with RakshPulse (5.0).

Layout (all CSV, human-readable; no Parquet)::

    <root>/
      dataset_manifest.csv            one row per file: path, kind, rows, bytes, sha256, created_utc
      subjects.csv                    subject_id, cohort, notes   (pseudonymous IDs only; no names or contact data)
      sessions/<subject>/<session>/
        session_metadata.csv          one row (schema SESSION_METADATA_COLUMNS)
        raw_ppg.csv                   timestamp, subject_id, session_id, sample_index, red, infrared[, temperature_c]
        reference_bp.csv              subject_id, session_id, timestamp, sbp, dbp, heart_rate, measurement_device
        gaps.csv                      missing-sample events found during acquisition
        derived/<name>.csv            processed windows, features, predictions (never overwrite raw files)

Rules:
* A session directory is created **exclusively**: an existing recording is never overwritten (``FileExistsError``).
* Raw samples are appended in chunks (``ChunkedCSVWriter``), so long recordings never sit in memory.
* When a session is closed, every file is hashed into ``dataset_manifest.csv``. ``validate()`` re-hashes files and
  reports any change, as well as schema, ordering, range and reference-reading problems.
* Derived files are written once per name; writing the same name again raises unless ``overwrite_derived=True``.
  Raw files cannot be overwritten through the store at all.
* ``sensor_type == "simulator"`` marks simulated sessions. They are reported separately and must never be
  presented as measurements.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

RAW_COLUMNS = ["timestamp", "subject_id", "session_id", "sample_index", "red", "infrared"]
RAW_OPTIONAL = ["temperature_c"]
REFERENCE_COLUMNS = ["subject_id", "session_id", "timestamp", "sbp", "dbp", "heart_rate", "measurement_device"]
SESSION_METADATA_COLUMNS = [
    "subject_id",
    "session_id",
    "recording_date",
    "sensor_type",
    "placement",
    "sampling_rate",
    "contact_conditions",
    "notes",
    "sensor_config",
    "software_version",
    "simulated",
]
GAP_COLUMNS = ["timestamp", "first_missing_index", "n_missing", "detected_by"]
SBP_RANGE, DBP_RANGE = (50.0, 260.0), (25.0, 160.0)
MANIFEST = "dataset_manifest.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _safe_id(x: str, what: str) -> str:
    x = str(x)
    if not x or any(c in x for c in r'\/:*?"<>|') or x in (".", ".."):
        raise ValueError(f"invalid {what}: {x!r}")
    return x


@dataclass
class SessionMetadata:
    subject_id: str
    session_id: str
    sampling_rate: float
    sensor_type: str = "max30102"
    placement: str = "behind_ear"
    recording_date: str = field(default_factory=lambda: dt.date.today().isoformat())
    contact_conditions: str = ""
    notes: str = ""
    sensor_config: str = ""  # e.g. MAX30102Config as "key=value;..." text
    software_version: str = ""

    @property
    def simulated(self) -> bool:
        return self.sensor_type == "simulator"

    def row(self) -> dict:
        d = asdict(self)
        d["simulated"] = self.simulated
        return {k: d.get(k, "") for k in SESSION_METADATA_COLUMNS}


class ChunkedCSVWriter:
    """Appends DataFrame chunks to one CSV, writing the header once. Column order is fixed at the first write."""

    def __init__(self, path: Path, columns: list[str]):
        self.path, self.columns, self.rows = Path(path), list(columns), 0
        if self.path.exists():
            raise FileExistsError(f"refusing to overwrite {self.path}")
        self._fh = self.path.open("w", newline="", encoding="utf-8")
        self._w = csv.writer(self._fh)
        self._w.writerow(self.columns)

    def write(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.columns if c not in df]
        if missing:
            raise ValueError(f"{self.path.name}: missing columns {missing}")
        self._w.writerows(df[self.columns].itertuples(index=False, name=None))
        self.rows += len(df)

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()


class SessionWriter:
    """Writes one recording session. Use as a context manager, or call :meth:`close` to hash and register files."""

    def __init__(self, store: DatasetStore, meta: SessionMetadata, with_temperature: bool = False):
        self.store, self.meta = store, meta
        self.dir = store.session_dir(meta.subject_id, meta.session_id)
        self.dir.mkdir(parents=True, exist_ok=False)  # never overwrite an existing session
        (self.dir / "derived").mkdir()
        pd.DataFrame([meta.row()]).to_csv(self.dir / "session_metadata.csv", index=False)
        cols = RAW_COLUMNS + (["temperature_c"] if with_temperature else [])
        self.raw = ChunkedCSVWriter(self.dir / "raw_ppg.csv", cols)
        self.refs = ChunkedCSVWriter(self.dir / "reference_bp.csv", REFERENCE_COLUMNS)
        self.gaps = ChunkedCSVWriter(self.dir / "gaps.csv", GAP_COLUMNS)
        self.closed = False

    def write_raw(self, rows: pd.DataFrame) -> None:
        """Sink for :class:`~rakshpulse.acquisition.ingest.Recorder` (rows from ``TimestampSync``)."""
        r = rows.assign(subject_id=self.meta.subject_id, session_id=self.meta.session_id)
        if "temperature_c" in self.raw.columns and "temperature_c" not in r:
            r["temperature_c"] = np.nan
        self.raw.write(r)

    __call__ = write_raw

    def add_reference(
        self, timestamp: float, sbp: float, dbp: float, heart_rate: float | None = None, measurement_device: str = ""
    ) -> None:
        """Record a cuff reading taken at session time ``timestamp`` (s). Implausible values are rejected."""
        if not (SBP_RANGE[0] <= sbp <= SBP_RANGE[1] and DBP_RANGE[0] <= dbp <= DBP_RANGE[1] and sbp > dbp):
            raise ValueError(f"implausible reference reading SBP={sbp} DBP={dbp}")
        self.refs.write(
            pd.DataFrame(
                [
                    {
                        "subject_id": self.meta.subject_id,
                        "session_id": self.meta.session_id,
                        "timestamp": float(timestamp),
                        "sbp": float(sbp),
                        "dbp": float(dbp),
                        "heart_rate": np.nan if heart_rate is None else float(heart_rate),
                        "measurement_device": measurement_device,
                    }
                ]
            )
        )

    def add_gaps(self, gaps: pd.DataFrame) -> None:
        if len(gaps):
            self.gaps.write(gaps)

    def close(self) -> dict:
        if self.closed:
            return {}
        for w in (self.raw, self.refs, self.gaps):
            w.close()
        self.closed = True
        self.store._register_session(self.meta)
        return {"raw_rows": self.raw.rows, "references": self.refs.rows, "gap_events": self.gaps.rows}

    def __enter__(self) -> SessionWriter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class DatasetStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------------------------------ layout
    def session_dir(self, subject_id: str, session_id: str) -> Path:
        return self.root / "sessions" / _safe_id(subject_id, "subject_id") / _safe_id(session_id, "session_id")

    def add_subject(self, subject_id: str, cohort: str = "", notes: str = "") -> None:
        p = self.root / "subjects.csv"
        df = (
            pd.read_csv(p, dtype=str).fillna("")
            if p.exists()
            else pd.DataFrame(columns=["subject_id", "cohort", "notes"])
        )
        if _safe_id(subject_id, "subject_id") in set(df.subject_id):
            return
        pd.concat([df, pd.DataFrame([{"subject_id": subject_id, "cohort": cohort, "notes": notes}])]).to_csv(
            p, index=False
        )

    def create_session(self, meta: SessionMetadata, with_temperature: bool = False) -> SessionWriter:
        self.add_subject(meta.subject_id)
        return SessionWriter(self, meta, with_temperature)

    def sessions(self) -> pd.DataFrame:
        rows = [
            pd.read_csv(p, dtype={"subject_id": str, "session_id": str})
            for p in sorted((self.root / "sessions").glob("*/*/session_metadata.csv"))
        ]
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=SESSION_METADATA_COLUMNS)

    # ------------------------------------------------------------------------------------------ reading
    def raw(self, subject_id: str, session_id: str, chunksize: int | None = None):
        p = self.session_dir(subject_id, session_id) / "raw_ppg.csv"
        return pd.read_csv(p, chunksize=chunksize) if chunksize else pd.read_csv(p)

    def references(self, subject_id: str | None = None, session_id: str | None = None) -> pd.DataFrame:
        files = (
            [self.session_dir(subject_id, session_id) / "reference_bp.csv"]
            if subject_id and session_id
            else sorted((self.root / "sessions").glob("*/*/reference_bp.csv"))
        )
        dfs = [pd.read_csv(p, dtype={"subject_id": str, "session_id": str}) for p in files if p.exists()]
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=REFERENCE_COLUMNS)

    def append_reference(
        self,
        subject_id: str,
        session_id: str,
        timestamp: float,
        sbp: float,
        dbp: float,
        heart_rate: float | None = None,
        measurement_device: str = "",
    ) -> None:
        """Append a cuff reading to a *closed* session (readings are often entered after recording). The file is
        appended, never rewritten, and its manifest checksum is updated with kind ``session:reference_appended``."""
        d = self.session_dir(subject_id, session_id)
        if not (d / "reference_bp.csv").exists():
            raise FileNotFoundError(f"no session {subject_id}/{session_id}")
        if not (SBP_RANGE[0] <= sbp <= SBP_RANGE[1] and DBP_RANGE[0] <= dbp <= DBP_RANGE[1] and sbp > dbp):
            raise ValueError(f"implausible reference reading SBP={sbp} DBP={dbp}")
        row = [
            subject_id,
            session_id,
            float(timestamp),
            float(sbp),
            float(dbp),
            "" if heart_rate is None else float(heart_rate),
            measurement_device,
        ]
        with (d / "reference_bp.csv").open("a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(row)
        self._manifest_add([d / "reference_bp.csv"], kind="session:reference_appended")

    def metadata(self, subject_id: str, session_id: str) -> dict:
        return pd.read_csv(self.session_dir(subject_id, session_id) / "session_metadata.csv").iloc[0].to_dict()

    # ------------------------------------------------------------------------------------------ derived
    def write_derived(
        self, subject_id: str, session_id: str, name: str, df: pd.DataFrame, overwrite_derived: bool = False
    ) -> Path:
        name = _safe_id(name, "derived name")
        if name in ("raw_ppg", "reference_bp", "session_metadata", "gaps"):
            raise ValueError("derived outputs may not use raw file names")
        if not (self.session_dir(subject_id, session_id) / "session_metadata.csv").exists():
            raise FileNotFoundError(f"no session {subject_id}/{session_id} in {self.root}")
        p = self.session_dir(subject_id, session_id) / "derived" / f"{name}.csv"
        p.parent.mkdir(exist_ok=True)  # empty folders are lost by git/zip transfers; recreate it
        if p.exists() and not overwrite_derived:
            raise FileExistsError(f"{p} exists (pass overwrite_derived=True to replace a derived file)")
        df.to_csv(p, index=False)
        self._manifest_add([p], kind=f"derived:{name}")
        return p

    # ------------------------------------------------------------------------------------------ manifest
    def _manifest_add(self, paths: list[Path], kind: str) -> None:
        mp = self.root / MANIFEST
        old = (
            pd.read_csv(mp)
            if mp.exists()
            else pd.DataFrame(columns=["path", "kind", "rows", "bytes", "sha256", "created_utc"])
        )
        rows = []
        for p in paths:
            rel = p.relative_to(self.root).as_posix()
            n = sum(1 for _ in p.open(encoding="utf-8")) - 1 if p.suffix == ".csv" else np.nan
            rows.append(
                {
                    "path": rel,
                    "kind": kind,
                    "rows": n,
                    "bytes": p.stat().st_size,
                    "sha256": sha256(p),
                    "created_utc": _now(),
                }
            )
        new = pd.DataFrame(rows)
        old = old[~old.path.isin(new.path)]
        pd.concat([old, new], ignore_index=True).sort_values("path").to_csv(mp, index=False)

    def _register_session(self, meta: SessionMetadata) -> None:
        d = self.session_dir(meta.subject_id, meta.session_id)
        files = [d / n for n in ("session_metadata.csv", "raw_ppg.csv", "reference_bp.csv", "gaps.csv")]
        self._manifest_add([f for f in files if f.exists()], kind="session")

    def manifest(self) -> pd.DataFrame:
        mp = self.root / MANIFEST
        return pd.read_csv(mp) if mp.exists() else pd.DataFrame()

    # ------------------------------------------------------------------------------------------ validation
    def validate(self, adc_max: float = 2**18 - 1) -> pd.DataFrame:
        """One row per (session, check): status PASS / WARN / FAIL and detail. Raw data is read in chunks."""
        out = []
        man = self.manifest().set_index("path") if len(self.manifest()) else pd.DataFrame()

        def add(sub, ses, check, ok, detail="", warn=False):
            out.append(
                {
                    "subject_id": sub,
                    "session_id": ses,
                    "check": check,
                    "status": "PASS" if ok else ("WARN" if warn else "FAIL"),
                    "detail": detail,
                }
            )

        for _, m in self.sessions().iterrows():
            sub, ses = str(m.subject_id), str(m.session_id)
            d = self.session_dir(sub, ses)
            add(sub, ses, "metadata_complete", all(pd.notna(m.get(c)) for c in ("sampling_rate", "sensor_type")))
            add(
                sub,
                ses,
                "simulated_flag_consistent",
                bool(m.simulated) == (m.sensor_type == "simulator"),
                "simulated session" if m.simulated else "",
            )
            for f in ("raw_ppg.csv", "reference_bp.csv", "session_metadata.csv", "gaps.csv"):
                rel = (d / f).relative_to(self.root).as_posix()
                if rel in man.index:
                    ok = sha256(d / f) == man.loc[rel, "sha256"]
                    add(sub, ses, f"checksum:{f}", ok, "" if ok else "file changed since it was recorded")
                else:
                    add(sub, ses, f"checksum:{f}", False, "not in manifest (session not closed?)")
            n, last_t, last_i, bad_order, out_range, nonfinite, t_min = 0, None, None, 0, 0, 0, None
            cols_ok = True
            for ch in pd.read_csv(d / "raw_ppg.csv", chunksize=200_000):
                if not set(RAW_COLUMNS) <= set(ch.columns):
                    cols_ok = False
                    break
                t, i = ch.timestamp.to_numpy(float), ch.sample_index.to_numpy(np.int64)
                if t_min is None and len(t):
                    t_min = t[0]
                bad_order += int(np.sum(np.diff(np.r_[last_t if last_t is not None else -np.inf, t]) <= 0))
                bad_order += int(np.sum(np.diff(np.r_[last_i if last_i is not None else -1, i]) <= 0))
                v = ch[["red", "infrared"]].to_numpy(float)
                nonfinite += int((~np.isfinite(v)).sum())
                out_range += int(((v < 0) | (v > adc_max)).sum())
                n += len(ch)
                if len(ch):
                    last_t, last_i = t[-1], i[-1]
            add(sub, ses, "raw_schema", cols_ok, f"{n} samples")
            add(sub, ses, "raw_time_and_counter_increase", bad_order == 0, f"{bad_order} non-increasing steps")
            add(
                sub,
                ses,
                "raw_values_finite_and_in_adc_range",
                nonfinite == 0 and out_range == 0,
                f"non-finite {nonfinite}, out of range {out_range}",
            )
            refs = pd.read_csv(d / "reference_bp.csv")
            plaus = refs.sbp.between(*SBP_RANGE) & refs.dbp.between(*DBP_RANGE) & (refs.sbp > refs.dbp)
            add(sub, ses, "references_plausible", bool(plaus.all()), f"{len(refs)} readings")
            if len(refs) and t_min is not None:
                inside = refs.timestamp.between(t_min - 120, last_t + 120)
                add(
                    sub,
                    ses,
                    "references_within_recording",
                    bool(inside.all()),
                    f"{int((~inside).sum())} readings > 2 min outside the recording",
                    warn=True,
                )
            gaps = pd.read_csv(d / "gaps.csv")
            frac = gaps.n_missing.sum() / max(n + gaps.n_missing.sum(), 1)
            add(sub, ses, "missing_samples_below_5pct", frac < 0.05, f"{frac:.2%} missing", warn=True)
        return pd.DataFrame(out, columns=["subject_id", "session_id", "check", "status", "detail"])

    # ------------------------------------------------------------------------------------------ import / export
    def import_session(
        self,
        meta: SessionMetadata,
        raw_csv: str | Path,
        references_csv: str | Path | None = None,
        chunksize: int = 200_000,
    ) -> dict:
        """Copy an external recording (raw columns: timestamp, sample_index, red, infrared[, temperature_c]) into
        the store. Optional reference CSV columns: timestamp, sbp, dbp[, heart_rate, measurement_device]."""
        head = pd.read_csv(raw_csv, nrows=1)
        with self.create_session(meta, with_temperature="temperature_c" in head) as w:
            for ch in pd.read_csv(raw_csv, chunksize=chunksize):
                w.write_raw(ch)
            if references_csv is not None:
                for r in pd.read_csv(references_csv).itertuples():
                    w.add_reference(
                        r.timestamp,
                        r.sbp,
                        r.dbp,
                        getattr(r, "heart_rate", None),
                        str(getattr(r, "measurement_device", "") or ""),
                    )
        return {"raw_rows": w.raw.rows, "references": w.refs.rows}

    def export_session(self, subject_id: str, session_id: str, dest: str | Path) -> Path:
        """Copy a session folder (raw, references, metadata, gaps, derived) plus its manifest rows to ``dest``."""
        src = self.session_dir(subject_id, session_id)
        dst = Path(dest) / subject_id / session_id
        shutil.copytree(src, dst)  # raises if dst exists
        man = self.manifest()
        rel = src.relative_to(self.root).as_posix()
        man[man.path.str.startswith(rel)].to_csv(dst / "manifest_rows.csv", index=False)
        return dst
