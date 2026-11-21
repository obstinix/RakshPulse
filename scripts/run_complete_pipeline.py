"""Run the complete RakshPulse ML workflow with one command.

    python scripts/run_complete_pipeline.py                  # everything (resumable)
    python scripts/run_complete_pipeline.py --evaluate-only  # re-evaluate the exported model, no training
    python scripts/run_complete_pipeline.py --smoke-test     # SYNTHETIC data, checks that the code runs end to end

Stages: 1 dependencies · 2 dataset availability · 3 downloads · 4 integrity · 5 preprocessing check ·
6 normalisation and cleaning · 7 model-ready dataset CSV · 8 training and tuning · 9 evaluation ·
10 model selection · 11 model export · 12 figures · 13 report.

Outputs (full run):
    data/ml/auricle_ear_bp_windows.csv, data/ml/auricle_ear_bp_cleaning_log.csv   model-ready dataset
    models/rakshpulse_ear_bp.joblib (+ .json)                                     exported model bundle
    reports/final_model_evaluation.md, reports/*.csv, reports/figures/*.png       evaluation report
Smoke test: everything under reports/smoke_test/ (git-ignored), clearly labelled as synthetic.

Resumable: the prepared dataset, the trained models and the exported bundle are reused when their inputs and
settings are unchanged (fingerprints in reports/pipeline_state.json). ``--force`` recomputes everything.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

REQUIRED = ("numpy", "scipy", "pandas", "sklearn", "xgboost", "lightgbm", "joblib", "matplotlib")
OPTIONAL = {"nbformat": "notebook execution", "nbclient": "notebook execution", "serial": "live MAX30102 input"}
MODEL_PATH = ROOT / "models" / "rakshpulse_ear_bp.joblib"
REPORTS = ROOT / "reports"
SMOKE_DIR = REPORTS / "smoke_test"
CACHE = ROOT / "data" / "ml" / "cache"
STATE = REPORTS / "pipeline_state.json"


class Stages:
    def __init__(self, total: int = 13):
        self.i, self.total, self.t0 = 0, total, time.perf_counter()

    def __call__(self, title: str) -> None:
        self.i += 1
        print(f"\n[{self.i:>2}/{self.total}] {title}  (+{time.perf_counter() - self.t0:.0f} s)", flush=True)


def note(msg: str) -> None:
    print(f"        {msg}", flush=True)


def _sha(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p if isinstance(p, bytes) else str(p).encode())
    return h.hexdigest()


def _code_hash() -> str:
    """Fingerprint of the code that determines the trained models: the parsed syntax tree of each file, so that
    formatting, comments and docstring-free whitespace changes do not force a retrain."""
    import ast

    files = sorted((ROOT / "src" / "rakshpulse" / "ml").glob("*.py")) + [
        ROOT / "src" / "rakshpulse" / "models" / "model_factory.py",
        ROOT / "src" / "rakshpulse" / "features" / "feature_pipeline.py",
    ]
    return _sha(*(ast.dump(ast.parse(f.read_text(encoding="utf-8"))) for f in files))


def _commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _state() -> dict:
    """Saved fingerprints; an unreadable file (e.g. a run killed while writing) means: recompute everything."""
    if not STATE.exists():
        return {}
    try:
        s = json.loads(STATE.read_text(encoding="utf-8"))
        return s if isinstance(s, dict) else {}
    except (OSError, ValueError):
        note(f"{STATE.name} is unreadable; recomputing every stage")
        return {}


def _save_state(state: dict) -> None:
    """Atomic write: the old file stays intact if the process is interrupted."""
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(STATE)


def check_dependencies() -> dict:
    found, missing = {}, []
    for m in REQUIRED:
        try:
            found[m] = getattr(importlib.import_module(m), "__version__", "installed")
        except ImportError:
            missing.append(m)
    for m, why in OPTIONAL.items():
        try:
            importlib.import_module(m)
            found[m] = "installed"
        except ImportError:
            note(f"optional '{m}' not installed ({why})")
    for k, v in found.items():
        note(f"ok  {k} {v}")
    if missing:
        sys.exit(f'missing required packages {missing}: run  pip install -e ".[train]"')
    return found


# ------------------------------------------------------------------------------------------------ smoke test
def run_smoke(args) -> int:
    import numpy as np

    from rakshpulse.acquisition.simulator import SimConfig, simulate
    from rakshpulse.inference.api import BPEstimator
    from rakshpulse.ml import PipelineConfig, export_bundle, make_smoke_dataset
    from rakshpulse.ml import report as R
    from rakshpulse.ml.workflow import make_figures, train_and_evaluate

    st = Stages(8)
    print("SMOKE TEST: synthetic data only. Results are NOT model validation.")
    st("dependencies")
    check_dependencies()
    st("synthetic dataset (simulator → production feature extractor; invented labels)")
    d = make_smoke_dataset(n_subjects=10, windows_per_session=10, seed=args.seed)
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    d.to_csv(SMOKE_DIR / "synthetic_windows.csv", index=False)
    note(
        f"{len(d)} SYNTHETIC windows, {d.subject_id.nunique()} synthetic subjects → "
        f"{SMOKE_DIR / 'synthetic_windows.csv'}"
    )
    st("training and evaluation (all candidate models, reduced grids)")
    cfg = PipelineConfig(seed=args.seed, cv_folds=3, n_bootstrap=200, n_jobs=args.n_jobs)
    result = train_and_evaluate(d, cfg=cfg, outer_folds=3, log=note)
    st("model export")
    bundle = export_bundle(result, version="smoke-test", synthetic=True, calibration=result["calibration"])
    path = bundle.save(SMOKE_DIR / "smoke_model.joblib")
    note(f"wrote {path}")
    st("inference through the hardware API")
    rec = simulate(SimConfig(fs=100.0, seed=args.seed + 7), 30.0)
    est = BPEstimator(path, sensor="simulator")
    r = est.estimate(rec.ir, 100.0, red=rec.red)
    note(f"status={r.status} sbp={r.sbp} dbp={r.dbp} mode={r.inference_mode} latency={r.latency_ms} ms")
    bad = est.estimate(np.full(300, np.nan), 100.0)
    note(f"invalid input → status={bad.status}: {bad.message}")
    cal = est.calibrate(120.0, 80.0, rec.ir, 100.0, red=rec.red)
    rc = est.estimate(rec.ir, 100.0, red=rec.red)
    note(f"calibration: {cal['message']} → {rc.calibration_status}, sbp={rc.sbp} dbp={rc.dbp}")
    if r.status != "ok" or bad.status != "invalid_input" or not rc.calibration_status.startswith("calibrated"):
        print("SMOKE TEST FAILED: unexpected inference status")
        return 1
    st("figures")
    f = make_figures(result, SMOKE_DIR / "figures")
    st("tables")
    R.write_tables(result, SMOKE_DIR)
    st("report")
    ctx = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "commit": _commit(),
        "version": _pkg_version(),
        "dataset_name": "SYNTHETIC smoke-test windows (simulator)",
        "model_path": path.relative_to(ROOT).as_posix(),
        "model_version": bundle.version,
    }
    rep = R.write_markdown(result, ctx, SMOKE_DIR / "smoke_test_report.md", f, smoke=True)
    note(f"wrote {rep}")
    print("\nSMOKE TEST PASSED (the code runs end to end). These results are synthetic and are not model validation.")
    return 0


def _pkg_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("rakshpulse")
    except PackageNotFoundError:
        return "source"


# ------------------------------------------------------------------------------------------------ evaluate only
def run_evaluate_only(args) -> int:
    import numpy as np
    import pandas as pd

    from rakshpulse.inference.contract import build_feature_vector, validate_bundle
    from rakshpulse.inference.model_loader import load_bundle
    from rakshpulse.ml import DatasetPaths, build_ear_dataset, load_ml_dataset
    from rakshpulse.ml import report as R
    from rakshpulse.ml.training import bootstrap_mae_difference, metrics, split_subjects

    st = Stages(6)
    st("dependencies")
    check_dependencies()
    st("exported model")
    if not MODEL_PATH.exists():
        print(f"no exported model at {MODEL_PATH}: run the full pipeline first")
        return 2
    b = load_bundle(MODEL_PATH)
    c = validate_bundle(b)
    note(
        f"{MODEL_PATH.name}: {b.version}, {len(b.features)} features, selected "
        f"{c['selected_models']['sbp']['model']} (SBP) / {c['selected_models']['dbp']['model']} (DBP)"
    )
    st("dataset")
    paths = DatasetPaths()
    d = load_ml_dataset(paths) if paths.ml_dataset.exists() else build_ear_dataset(paths)[0]
    train_s, test_s = split_subjects(d, 0.2, args.seed)
    test, train = d[d.subject_id.isin(test_s)], d[d.subject_id.isin(train_s) & d.fit_window]
    note(f"{len(test)} test windows from {len(test_s)} held-out subjects")
    st("prediction through the production contract path")
    preds = []
    for _, row in test.iterrows():
        feats = {spec["source"]: row[spec["name"]] for spec in c["features"]}
        X, _ = build_feature_vector(c, feats, {})
        preds.append((float(b.sbp.predict(X)[0]), float(b.dbp.predict(X)[0])))
    p = np.asarray(preds)
    rows, out = [], REPORTS / "evaluate_only"
    out.mkdir(parents=True, exist_ok=True)
    long = []
    for j, t in enumerate(("sbp", "dbp")):
        base = np.full(len(test), train[t].mean())
        m, mb = metrics(test[t], p[:, j], test.subject_id), metrics(test[t], base, test.subject_id)
        cmp_ = bootstrap_mae_difference(test[t].to_numpy(), p[:, j], base, test.subject_id.to_numpy(), seed=args.seed)
        rows += [
            {"target": t.upper(), "model": c["selected_models"][t]["model"], **m},
            {"target": t.upper(), "model": "mean", **mb},
        ]
        note(
            f"{t.upper()}: MAE {m['mae']:.2f}  RMSE {m['rmse']:.2f}  R² {m['r2']:.3f}  (baseline MAE {mb['mae']:.2f}; "
            f"difference {cmp_['mae_difference']:+.2f} [{cmp_['ci_low']:+.2f}, {cmp_['ci_high']:+.2f}])"
        )
        stored = c["performance"].get(f"{t}_mae")
        if stored is not None and abs(stored - m["mae"]) > 1e-6:
            note(f"WARNING: differs from the MAE stored in the bundle ({stored:.4f})")
        for model, pr in ((c["selected_models"][t]["model"], p[:, j]), ("mean", base)):
            long.append(
                pd.DataFrame(
                    {
                        "window_id": test.window_id,
                        "subject_id": test.subject_id,
                        "target": t,
                        "model": model,
                        "y": test[t],
                        "pred": pr,
                    }
                )
            )
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False, float_format="%.4f")
    preds_df = pd.concat(long)
    sel = pd.DataFrame([{"target": t, "model": c["selected_models"][t]["model"]} for t in ("sbp", "dbp")])
    st("figures")
    R.plot_actual_vs_predicted(preds_df, sel, out / "actual_vs_predicted.png")
    R.plot_residuals(preds_df, sel, out / "residuals.png")
    st("summary")
    note(f"wrote {out / 'metrics.csv'} and figures in {out}")
    return 0


# ------------------------------------------------------------------------------------------------ full run
def run_full(args) -> int:
    import joblib
    import pandas as pd

    from rakshpulse.ml import (
        DatasetPaths,
        PipelineConfig,
        audit_datasets,
        build_ear_dataset,
        export_bundle,
        load_ml_dataset,
    )
    from rakshpulse.ml import report as R
    from rakshpulse.ml.dataset import feature_reproducibility, input_fingerprint, validate_auricle
    from rakshpulse.ml.workflow import make_figures, train_and_evaluate

    st, paths, state = Stages(), DatasetPaths(), {} if args.force else _state()
    st("dependencies")
    deps = check_dependencies()

    st("dataset availability")
    audit = audit_datasets(paths)
    REPORTS.mkdir(parents=True, exist_ok=True)
    audit.to_csv(REPORTS / "dataset_audit.csv", index=False)
    for _, r in audit.iterrows():
        note(f"{r.status:<15} {r.file}")
    required = audit[(audit.dataset == "auricle") & (audit.status != "ok")]
    if len(required):
        print("required Auricle tables are missing; restore them with `git checkout -- data/auricle`")
        return 2

    st("downloads")
    wave = audit[audit.dataset.str.startswith("auricle waveforms")]
    if (wave.status != "ok").any():
        if args.with_waveforms:
            from rakshpulse.data.release import fetch

            note("downloading the published Auricle waveform arrays (≈350 MB) from the GitHub release")
            fetch("waveforms")
        else:
            note(
                "optional raw waveform arrays not downloaded (needed only for the feature reproducibility check "
                "and the notebook's PPG plots): add --with-waveforms"
            )
    else:
        note("all required and optional datasets present; nothing to download")

    st("integrity validation")
    val = validate_auricle(paths)
    val.to_csv(REPORTS / "dataset_validation.csv", index=False)
    for _, r in val.iterrows():
        note(f"{r.status}  {r.check} {r.detail}")
    if (val.status != "PASS").any():
        print("dataset validation failed; see reports/dataset_validation.csv")
        return 1

    st("preprocessing check: production extractor on raw PPG vs stored features")
    repro = feature_reproducibility(paths, n=args.repro_windows)
    if repro is None:
        note("skipped (raw waveforms not downloaded)")
    else:
        repro.to_csv(REPORTS / "feature_reproducibility.csv", index=False)
        note(f"{int(repro.n_windows.max())} windows re-processed; median Spearman ρ = {repro.spearman.median():.4f}")

    st("cleaning, label alignment and outlier handling")
    fp = input_fingerprint(paths)
    if paths.ml_dataset.exists() and state.get("dataset_fingerprint") == fp:
        d = load_ml_dataset(paths)
        log = pd.read_csv(paths.cleaning_log)
        note(f"up to date, reused {paths.ml_dataset.relative_to(ROOT)}")
    else:
        d, log = build_ear_dataset(paths)
        state["dataset_fingerprint"] = fp
    for _, r in log.iterrows():
        note(f"{r['filter']:<28} {r.windows:>6} windows  {r.subjects:>3} subjects")

    st("model-ready dataset")
    note(
        f"{paths.ml_dataset.relative_to(ROOT)}: {len(d)} windows, {d.subject_id.nunique()} subjects, "
        f"{d.fit_window.sum()} non-overlapping windows used for fitting"
    )

    st("training and hyper-parameter optimisation (subject-wise GroupKFold)")
    cfg = PipelineConfig(seed=args.seed, n_jobs=args.n_jobs)
    train_fp = _sha(
        fp,
        json.dumps(cfg.__dict__, default=str, sort_keys=True),
        _code_hash(),
        json.dumps({k: deps[k] for k in REQUIRED}, sort_keys=True),
    )
    cache = CACHE / "training_result.joblib"
    if cache.exists() and state.get("training_fingerprint") == train_fp:
        result = joblib.load(cache)
        note("up to date, reused the trained models")
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = train_and_evaluate(d, cfg=cfg, log=note)
        CACHE.mkdir(parents=True, exist_ok=True)
        joblib.dump(result, cache, compress=3)
        state["training_fingerprint"] = train_fp

    st("evaluation (cross-validation, independent test, cross-session, baseline)")
    for _, r in result["baseline_comparison"].iterrows():
        note(
            f"{r.target.upper()}: {r.model} vs baseline, test MAE difference {r.mae_difference:+.2f} "
            f"[{r.ci_low:+.2f}, {r.ci_high:+.2f}] mmHg"
        )
    for _, r in result["cross_session"].iterrows():
        note(f"cross-session {r.target.upper()} {r.model:<14} MAE {r.mae:.2f}  R² {r.r2:.3f}")
    for _, r in result["nested"]["summary"].iterrows():
        note(f"nested CV ({r.n_subjects} subjects) {r.target.upper()} {r.model:<16} MAE {r.mae:.2f}  R² {r.r2:.3f}")
    for _, r in result["calibration"]["summary"].iterrows():
        note(f"calibration {r.scenario:<24} {r.target.upper()} {r.method:<14} MAE {r.mae:.2f}")

    st("model selection")
    for _, r in result["selection"].iterrows():
        note(
            f"{r.target.upper()}: {r.label} {r.params}; CV MAE {r.cv_mae:.2f}, test MAE {r.test_mae:.2f}, "
            f"RMSE {r.test_rmse:.2f}, R² {r.test_r2:.3f}"
        )

    st("model export")
    version = f"rakshpulse-ear-bp-1.1+{train_fp[:8]}"
    if MODEL_PATH.exists() and state.get("model_fingerprint") == train_fp:
        note(f"up to date: {MODEL_PATH.relative_to(ROOT)}")
    else:
        bundle = export_bundle(result, version, calibration=result["calibration"])
        bundle.save(MODEL_PATH)
        meta = {
            "version": version,
            "features": bundle.features,
            "contract": bundle.contract,
            "validation": bundle.validation,
            "created_utc": bundle.created_utc,
        }
        MODEL_PATH.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))
        state["model_fingerprint"] = train_fp
        note(f"wrote {MODEL_PATH.relative_to(ROOT)} and {MODEL_PATH.with_suffix('.json').name}")

    st("figures")
    f = make_figures(result, REPORTS / "figures")
    note(", ".join(p.name for p in f.values()))

    st("report")
    R.write_tables(result, REPORTS)
    log.to_csv(REPORTS / "dataset_cleaning_log.csv", index=False)
    ctx = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "commit": _commit(),
        "version": _pkg_version(),
        "dataset_name": "Auricle (PhysioNet v1.0.0), healthy cohort: earlobe PPG with CNAP continuous BP, "
        "two sessions on different days",
        "cleaning_log": log[["filter", "rule", "windows", "subjects"]],
        "validation": val,
        "reproducibility": repro,
        "model_path": MODEL_PATH.relative_to(ROOT).as_posix(),
        "model_version": version,
    }
    rep = R.write_markdown(result, ctx, REPORTS / "final_model_evaluation.md", f)
    _save_state(state)
    note(f"wrote {rep.relative_to(ROOT)}")
    print("\nDone. Offline validation on the Auricle earlobe dataset only; no MAX30102 hardware validation exists.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--evaluate-only", action="store_true", help="re-evaluate the exported model; no training")
    mode.add_argument("--smoke-test", action="store_true", help="SYNTHETIC data: check that the code runs")
    ap.add_argument("--force", action="store_true", help="ignore cached results and recompute everything")
    ap.add_argument("--with-waveforms", action="store_true", help="download the raw Auricle waveform arrays")
    ap.add_argument("--repro-windows", type=int, default=200, help="windows for the feature reproducibility check")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args(argv)
    if args.smoke_test:
        return run_smoke(args)
    if args.evaluate_only:
        return run_evaluate_only(args)
    return run_full(args)


if __name__ == "__main__":
    sys.exit(main())
