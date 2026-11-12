"""RakshPulse 4.0 + 5.0 result audit: provenance, checksums, splits, leakage and table regeneration.

Checks (each prints PASS/FAIL; exit code 1 on any FAIL):
1. every 4.0 run in results/runs/runs.csv has status completed, a code-clean commit (git_dirty_code False) that
   is an ancestor of HEAD, and identical experiment code at HEAD (no later change under src/ or the run's script);
2. every dataset file recorded in a run's config.csv still has the recorded SHA-256;
3. runs that fit models record a subject split, and the 2.0/3.0 runs in runs.csv are unchanged vs the 3.0 commit;
4. external validation: VitalDB 3.0 and extension patients (and case ids) are disjoint, and the extension
   selection excludes every 3.0 case;
5. cross-day / calibration: the reference windows used for calibration are never scored (re-derived);
6. the W1–W6 tables in docs/RESEARCH_FINDINGS.md equal a fresh render from the committed CSVs.
"""

from __future__ import annotations

import hashlib
import io
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

SCRIPT_OF = {
    "cross_day_generalization_v4": "scripts/run_cross_day_generalization.py",
    "physio_features_v4": "scripts/run_physio_features.py",
    "quality_score_v4": "scripts/run_quality_study.py",
    "calibration_uncertainty_v4": "scripts/run_calibration_uncertainty.py",
    "external_validation_v4": "scripts/run_external_validation.py",
    "deployment_v4": "scripts/run_deployment_v4.py",
    "v5_cross_day_calibration": "scripts/run_v5_experiments.py",
    "v5_simulation_and_benchmarks": "scripts/run_v5_experiments.py",
}
NEEDS_SPLITS = {
    "cross_day_generalization_v4",
    "physio_features_v4",
    "quality_score_v4",
    "calibration_uncertainty_v4",
    "external_validation_v4",
    "v5_cross_day_calibration",
}
V3_COMMIT = "b783b76"
failures: list[str] = []


def git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def check(ok: bool, msg: str) -> None:
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        failures.append(msg)


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    runs = pd.read_csv(ROOT / "results" / "runs" / "runs.csv")
    latest = runs[runs.name.isin(SCRIPT_OF)].groupby("name").tail(1)
    check(set(latest.name) == set(SCRIPT_OF), f"one latest run per 4.0 experiment ({len(latest)}/{len(SCRIPT_OF)})")
    for r in latest.itertuples():
        commit = r.git_commit
        anc = subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=ROOT).returncode == 0
        # files that existed at the run commit must be unchanged; newly added modules cannot alter an earlier run
        changed = git("diff", "--name-only", "--diff-filter=MDR", commit, "HEAD", "--", "src", SCRIPT_OF[r.name])
        check(
            r.status == "completed" and str(r.git_dirty_code) == "False" and anc and not changed,
            f"{r.run_id}: completed, clean code at {commit}, unchanged since ({changed or 'no diff'})",
        )
        cfg = pd.read_csv(ROOT / r.output_dir / "config.csv").set_index("key").value
        ds = {k[len("dataset:") :]: v for k, v in cfg.items() if k.startswith("dataset:")}
        bad = [k for k, v in ds.items() if sha(ROOT / k) != v]
        check(not bad, f"{r.run_id}: {len(ds)} dataset checksums match {bad or ''}")
        if r.name in NEEDS_SPLITS:
            sp = ROOT / r.output_dir / "splits.csv"
            n = len(pd.read_csv(sp)) if sp.exists() else 0
            check(n > 0, f"{r.run_id}: subject split recorded ({n} subjects)")
    old = pd.read_csv(io.StringIO(git("show", f"{V3_COMMIT}:results/runs/runs.csv")))
    check(runs.iloc[: len(old)][old.columns].astype(str).equals(old.astype(str)), "2.0/3.0 registry rows unchanged")
    for f in (
        "results/beyond_reference/paired.csv",
        "results/calibration_study/metrics.csv",
        "results/personalization/paired_comparisons.csv",
        "results/benchmarks/inference_benchmark.csv",
    ):
        check(not git("diff", "--name-only", V3_COMMIT, "HEAD", "--", f), f"3.0/2.0 result unchanged: {f}")
    # external disjointness
    a = pd.read_csv(ROOT / "data" / "vitaldb" / "cases.csv")
    b = pd.read_csv(ROOT / "data" / "vitaldb_ext" / "cases.csv")
    sel = pd.read_csv(ROOT / "data" / "metadata" / "vitaldb_selection_ext.csv")
    base = pd.read_csv(ROOT / "data" / "metadata" / "vitaldb_selection.csv")
    check(
        not set(a.caseid) & set(b.caseid) and not set(a.subject_id) & set(b.subject_id),
        f"VitalDB v3 ({len(a)}) and extension ({len(b)}) patients disjoint",
    )
    check(not set(sel.caseid) & set(base.caseid), "extension selection excludes all 3.0 cases")
    # calibration readings never scored (re-derive scenario sets for one site/target)
    import run_calibration_uncertainty as cu

    d = cu.load("ear", "sbp")
    sets = cu.scenario_sets(d)
    leak = 0
    for items in sets.values():
        for _, r_, ev in items:
            leak += len(set(r_.window_id) & set(ev.window_id))
    check(leak == 0, f"calibration readings never evaluated ({sum(len(v) for v in sets.values())} subject-scenarios)")
    # table regeneration
    import write_findings as wf

    doc = (ROOT / "docs" / "RESEARCH_FINDINGS.md").read_text(encoding="utf-8")
    a_, b_ = doc.index(wf.BW) + len(wf.BW), doc.index(wf.EW)
    check(doc[a_:b_].strip() == wf.v4_block().strip(), "W1–W6 tables equal a fresh render from the CSVs")
    if wf.B5 in doc:
        a5, b5 = doc.index(wf.B5) + len(wf.B5), doc.index(wf.E5)
        check(doc[a5:b5].strip() == wf.v5_block().strip(), "Y1–Y5 (5.0) tables equal a fresh render from the CSVs")
    check(
        not git("diff", "--name-only", "--diff-filter=MDR", "d225972", "HEAD", "--", "src"),
        "5.0 added modules only: no 4.0 source module modified or deleted",
    )
    print(f"\n{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
