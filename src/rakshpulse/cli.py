"""``rakshpulse`` command-line interface (RakshPulse 5.0).

    rakshpulse doctor                       check the environment, optional packages and a simulated self-test
    rakshpulse simulate   --out DIR ...      create a SIMULATED dataset (clearly flagged) for software testing
    rakshpulse record     --out DIR ...      record from a source (sim | replay | serial) into a dataset
    rakshpulse reference  --dataset DIR ...  append a cuff reading to a recorded session
    rakshpulse inspect    --dataset DIR      list sessions, references, gaps; --validate runs all checks
    rakshpulse preprocess --dataset DIR      windows + quality + features per session (derived/windows*.csv)
    rakshpulse features   --dataset DIR      one training table with cuff references associated to windows
    rakshpulse train      --table CSV        train + cross-validate a model; versioned bundle in --models-dir
    rakshpulse calibrate  --oof CSV          compare calibration strategies leakage-safely; fit hierarchical params
    rakshpulse evaluate   --model P --table  evaluate a saved model (no refitting); subject-level bootstrap CIs
    rakshpulse infer      --model P ...      streaming inference from a source; predictions CSV
    rakshpulse export     --model P --out D  float32 JSON + C header (runtime-free); equivalence check

Use ``rakshpulse <command> -h`` for options. Nothing here is a medical device. Simulated data is never evidence.
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SIM_WARNING = "WARNING: simulated data; results describe the software, not blood pressure."


def _fail(msg: str, code: int = 2) -> int:
    print(f"error: {msg}", file=sys.stderr)
    return code


# ------------------------------------------------------------------------------------------------ doctor
def cmd_doctor(a) -> int:
    ok = True
    print(f"python {platform.python_version()} on {platform.platform()}")
    for mod, req in (
        ("numpy", True),
        ("scipy", True),
        ("pandas", True),
        ("sklearn", True),
        ("joblib", True),
        ("xgboost", False),
        ("lightgbm", False),
        ("torch", False),
        ("serial", False),
        ("vitaldb", False),
    ):
        try:
            m = importlib.import_module(mod)
            print(f"  ok   {mod} {getattr(m, '__version__', '')}")
        except ImportError:
            print(f"  {'FAIL' if req else 'n/a '} {mod}{'' if req else ' (optional)'}")
            ok = ok and not req
    try:
        from .acquisition.simulator import SimConfig, simulate
        from .preprocessing.pipeline import PipelineConfig, process_recording

        r = simulate(SimConfig(seed=1), 30)
        w, _ = process_recording(
            pd.DataFrame({"timestamp": r.t, "sample_index": r.sample_index, "red": r.red, "infrared": r.ir}),
            PipelineConfig(),
        )
        hr_err = float(np.nanmedian(np.abs(w.ir_hr - 60 / np.median(np.diff(r.beat_times)))))
        good = (w.status == "usable").all() and hr_err < 5
        print(
            f"  {'ok  ' if good else 'FAIL'} self-test: simulated 30 s -> {len(w)} windows, HR error {hr_err:.1f} bpm"
        )
        ok = ok and good
    except Exception as exc:  # pragma: no cover - reported to the user
        ok = False
        print(f"  FAIL self-test: {exc}")
    print("doctor:", "all required checks passed" if ok else "problems found")
    return 0 if ok else 1


# ------------------------------------------------------------------------------------------------ simulate / record
def _sim_config(a, seed: int, hr: float):
    from .acquisition.simulator import SimConfig

    motion = tuple(np.random.default_rng(seed).uniform(10, a.duration - 10, a.motion_events)) if a.motion_events else ()
    return SimConfig(fs=a.fs, hr_bpm=hr, seed=seed, dropout_prob=a.dropout, motion_times_s=motion, noise_frac=a.noise)


def cmd_simulate(a) -> int:
    from .acquisition.ingest import Recorder
    from .acquisition.sources import SimulatedSensor
    from .datastore.store import DatasetStore, SessionMetadata

    store = DatasetStore(a.out)
    rng = np.random.default_rng(a.seed)
    print(SIM_WARNING)
    for s in range(a.subjects):
        sid = f"SIM{s + 1:03d}"
        base_sbp, base_dbp, base_hr = rng.normal(120, 10), rng.normal(75, 7), rng.uniform(60, 85)
        for d in range(a.sessions):
            seed = int(a.seed * 1000 + s * 10 + d)
            cfg = _sim_config(a, seed, base_hr + rng.normal(0, 4))
            src = SimulatedSensor(cfg, a.duration)
            meta = SessionMetadata(
                sid,
                f"D{d + 1}",
                a.fs,
                sensor_type="simulator",
                placement="simulated",
                notes="synthetic signal and synthetic cuff readings; not a measurement",
                sensor_config=json.dumps(cfg.to_dict()),
            )
            with store.create_session(meta) as w:
                rec = Recorder(src, [w.write_raw])
                rec.run(duration_s=a.duration)
                w.add_gaps(rec.sync.gap_table())
                # synthetic cuff readings: a stated toy relation to the simulated HR (for exercising the software only)
                day_shift = rng.normal(0, 5)
                for t in np.arange(a.reference_every, a.duration, a.reference_every):
                    hr = float(np.interp(t, src.rec.t, src.rec.hr_true))
                    sbp = base_sbp + day_shift + 0.5 * (hr - base_hr) + rng.normal(0, 3)
                    dbp = base_dbp + 0.6 * day_shift + 0.3 * (hr - base_hr) + rng.normal(0, 2)
                    w.add_reference(float(t), round(sbp, 1), round(dbp, 1), round(hr, 1), "simulator")
            print(f"  {sid}/D{d + 1}: {rec.sync.n_samples} samples, {rec.sync.n_missing} missing")
    return 0


def _source(a):
    from .acquisition.sources import LineFormat, ReplaySource, SerialSource, SimulatedSensor

    if a.source == "sim":
        from .acquisition.simulator import SimConfig

        return SimulatedSensor(SimConfig(fs=a.fs, hr_bpm=a.sim_hr, seed=a.seed), a.duration or 60, realtime=a.realtime)
    if a.source == "replay":
        if not a.input:
            raise ValueError("--input raw_ppg.csv is required for --source replay")
        return ReplaySource(a.input, a.fs)
    if a.source == "serial":
        if not a.port:
            raise ValueError("--port is required for --source serial")
        return SerialSource(a.port, a.baud, a.fs, LineFormat(tuple(a.fields.split(",")), a.delimiter))
    raise ValueError(a.source)


def cmd_record(a) -> int:
    from .acquisition.ingest import Recorder, TimestampSync
    from .acquisition.sources import MAX30102Config
    from .datastore.store import DatasetStore, SessionMetadata

    sensor_cfg = ""
    if a.source == "serial":
        mc = MAX30102Config(**json.loads(a.sensor_config)) if a.sensor_config else MAX30102Config()
        for w in mc.validate():
            print("warning:", w)
        if abs(mc.output_rate_hz - a.fs) > 1e-9:
            return _fail(f"--fs {a.fs} does not match the sensor output rate {mc.output_rate_hz}")
        sensor_cfg = json.dumps(mc.to_dict())
    src = _source(a)
    sim = a.source == "sim"
    if sim:
        print(SIM_WARNING)
    meta = SessionMetadata(
        a.subject,
        a.session,
        a.fs,
        sensor_type="simulator" if sim else src.sensor_type,
        placement=a.placement,
        contact_conditions=a.contact,
        notes=a.notes,
        sensor_config=sensor_cfg,
    )
    store = DatasetStore(a.out)
    with store.create_session(meta) as w:
        rec = Recorder(src, [w.write_raw], TimestampSync(a.fs, counter_modulus=a.counter_modulus))
        try:
            s = rec.run(duration_s=a.duration)
        except KeyboardInterrupt:
            rec.stop()
            s = {"samples": rec.sync.n_samples, "missing_samples": rec.sync.n_missing, "stopped_by": "ctrl-c"}
        w.add_gaps(rec.sync.gap_table())
    print(json.dumps(s, default=float))
    return 0


def cmd_reference(a) -> int:
    from .datastore.store import DatasetStore

    DatasetStore(a.dataset).append_reference(a.subject, a.session, a.time, a.sbp, a.dbp, a.hr, a.device)
    print("reading appended")
    return 0


# ------------------------------------------------------------------------------------------------ inspect / process
def cmd_inspect(a) -> int:
    from .datastore.store import DatasetStore

    st = DatasetStore(a.dataset)
    ses = st.sessions()
    if ses.empty:
        return _fail(f"no sessions in {a.dataset}")
    rows = []
    for _, m in ses.iterrows():
        d = st.session_dir(str(m.subject_id), str(m.session_id))
        n = sum(len(c) for c in pd.read_csv(d / "raw_ppg.csv", chunksize=500_000, usecols=["timestamp"]))
        g = pd.read_csv(d / "gaps.csv")
        rows.append(
            {
                "subject_id": m.subject_id,
                "session_id": m.session_id,
                "sensor": m.sensor_type,
                "simulated": m.simulated,
                "fs": m.sampling_rate,
                "samples": n,
                "minutes": round(n / float(m.sampling_rate) / 60, 1),
                "references": len(pd.read_csv(d / "reference_bp.csv")),
                "missing": int(g.n_missing.sum()),
                "derived": ";".join(p.stem for p in sorted((d / "derived").glob("*.csv"))),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))
    if ses.simulated.any():
        print(SIM_WARNING)
    if a.validate:
        v = st.validate()
        print(v[v.status != "PASS"].to_string(index=False) if (v.status != "PASS").any() else "validation: all PASS")
        return 1 if (v.status == "FAIL").any() else 0
    return 0


def cmd_preprocess(a) -> int:
    from .datastore.store import DatasetStore
    from .preprocessing.pipeline import PipelineConfig, process_recording

    st = DatasetStore(a.dataset)
    ses = st.sessions()
    if a.subject:
        ses = ses[ses.subject_id.astype(str) == a.subject]
    if a.session:
        ses = ses[ses.session_id.astype(str) == a.session]
    if ses.empty:
        return _fail("no matching sessions")
    for _, m in ses.iterrows():
        cfg = PipelineConfig(fs=float(m.sampling_rate), window_s=a.window, step_s=a.step, band=tuple(a.band))
        w, seg = process_recording(
            st.raw(str(m.subject_id), str(m.session_id), chunksize=a.chunksize),
            cfg,
            str(m.subject_id),
            str(m.session_id),
        )
        name = f"windows{('_' + a.tag) if a.tag else ''}"
        st.write_derived(str(m.subject_id), str(m.session_id), name, w, overwrite_derived=a.overwrite)
        st.write_derived(
            str(m.subject_id),
            str(m.session_id),
            f"segments{('_' + a.tag) if a.tag else ''}",
            seg,
            overwrite_derived=a.overwrite,
        )
        counts = w.status.value_counts().to_dict() if len(w) else {}
        print(f"  {m.subject_id}/{m.session_id}: {len(w)} windows {counts}")
    return 0


def cmd_features(a) -> int:
    from .datastore.association import associate_references
    from .datastore.store import DatasetStore

    st = DatasetStore(a.dataset)
    name = f"windows{('_' + a.tag) if a.tag else ''}.csv"
    parts, assigned = [], []
    for _, m in st.sessions().iterrows():
        p = st.session_dir(str(m.subject_id), str(m.session_id)) / "derived" / name
        if not p.exists():
            return _fail(f"{p} missing: run `rakshpulse preprocess` first")
        w = pd.read_csv(p, dtype={"subject_id": str, "session_id": str})
        w["simulated"] = bool(m.simulated)
        lw, lr = associate_references(
            w, st.references(str(m.subject_id), str(m.session_id)), a.tolerance, require_usable=True
        )
        parts.append(lw)
        assigned.append(lr)
    t = pd.concat(parts, ignore_index=True).rename(columns={"ref_sbp": "y_sbp", "ref_dbp": "y_dbp"})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    if Path(a.out).exists() and not a.overwrite:
        return _fail(f"{a.out} exists (use --overwrite)")
    t.to_csv(a.out, index=False)
    r = pd.concat(assigned, ignore_index=True)
    print(
        f"{len(t)} windows, {int(t.y_sbp.notna().sum())} labelled by {int((r.status == 'assigned').sum())} readings; "
        f"unassigned readings: {r[r.status != 'assigned'].status.value_counts().to_dict()}"
    )
    if t.simulated.any():
        print(SIM_WARNING)
    return 0


# ------------------------------------------------------------------------------------------------ train / evaluate
def cmd_train(a) -> int:
    from .evaluation.splits import SplitSpec
    from .models.registry import TrainConfig, save_model, train_model

    t = pd.read_csv(a.table, dtype={"subject_id": str, "session_id": str})
    if "simulated" in t and t.simulated.all() and not a.allow_simulated:
        return _fail("the table is entirely simulated; pass --allow-simulated to train for software testing only")
    feats = a.features.split(",") if "," in a.features else a.features
    cfg = TrainConfig(
        name=a.name,
        model=a.model,
        features=feats,
        tune=a.tune,
        seed=a.seed,
        split=SplitSpec(a.split, a.folds, a.seed, tuple(a.train_sessions.split(","))),
        hyperparams=json.loads(a.hyperparams) if a.hyperparams else {},
    )
    bundle, cv, oof = train_model(t, cfg)
    path = save_model(bundle, a.models_dir)
    oof.assign(model_version=bundle.version).to_csv(Path(a.models_dir) / f"{bundle.version}_oof.csv", index=False)
    print(cv.groupby("target").mean(numeric_only=True).drop(columns="fold").round(2).to_string())
    print(f"saved {path} (+ out-of-fold predictions)")
    if "simulated" in t and t.simulated.any():
        print(SIM_WARNING)
    return 0


def cmd_evaluate(a) -> int:
    from .evaluation.statistics import bland_altman_repeated, paired_test, subject_errors
    from .inference.model_loader import load_bundle

    b = load_bundle(a.model)
    t = pd.read_csv(a.table, dtype={"subject_id": str, "session_id": str})
    t = t[t.status.isin(a.statuses.split(","))].dropna(subset=["y_sbp", "y_dbp"])
    missing = [c for c in b.features if c not in t]
    if missing:
        return _fail(f"table lacks model features {missing[:5]}")
    rows = []
    rng = np.random.default_rng(a.seed)
    for tgt in ("sbp", "dbp"):
        p = np.ravel(getattr(b, tgt).predict(t[b.features].to_numpy(float)))
        f = pd.DataFrame({"subject_id": t.subject_id, "session_id": t.session_id, "y": t[f"y_{tgt}"], "yhat": p})
        se = subject_errors(f)
        boot = [se.to_numpy()[rng.integers(0, se.size, se.size)].mean() for _ in range(a.n_boot)]
        e = f.yhat - f.y
        rows.append(
            {
                "target": tgt,
                "mae": float(e.abs().mean()),
                "rmse": float(np.sqrt((e**2).mean())),
                "mean_error": float(e.mean()),
                "sd_error": float(e.std()),
                "pearson_r": float(np.corrcoef(f.y, f.yhat)[0, 1]) if f.yhat.std() > 0 else np.nan,
                "subject_mae": float(se.mean()),
                "subject_mae_ci_low": float(np.percentile(boot, 2.5)),
                "subject_mae_ci_high": float(np.percentile(boot, 97.5)),
                "n_subjects": int(se.size),
                **{f"ba_{k}": v for k, v in bland_altman_repeated(f, 200).items() if k.startswith(("bias", "loa"))},
            }
        )
        # optimistic baseline: a constant equal to this table's own mean (an oracle the model never sees)
        base = f.assign(yhat=float(f.y.mean()))
        r = paired_test(se, subject_errors(base), a.n_boot, seed=a.seed)
        rows[-1].update(
            {f"vs_oracle_constant_{k}": v for k, v in r.items() if k in ("mean_diff", "ci_low", "ci_high", "p_perm")}
        )
    out = pd.DataFrame(rows)
    print(out.round(2).T.to_string(header=False))
    if a.out:
        out.to_csv(a.out, index=False)
    return 0


def cmd_calibrate(a) -> int:
    from .personalization.hierarchical import fit_params
    from .personalization.strategies import STRATEGIES, CalibrationPolicy, evaluate

    oof = pd.read_csv(a.oof, dtype={"subject_id": str, "session_id": str})
    pol = CalibrationPolicy(min_spacing_s=a.spacing, expiry_s=a.expiry)
    rows = []
    for tgt in ("sbp", "dbp"):
        f = oof.rename(columns={"t_start_s": "t", f"y_{tgt}": "y", f"p_{tgt}": "pred"}).dropna(subset=["y", "pred"])
        sessions = sorted(f.session_id.unique())  # only sessions that have out-of-fold predictions
        f = f.assign(pmean=f.y.mean())  # replaced per fold by the training mean below
        subjects = sorted(f.subject_id.unique())
        folds = {s: i % a.folds for i, s in enumerate(subjects)}
        for k in range(a.folds):  # training-mean baseline from training subjects only
            tr = [s for s in subjects if folds[s] != k]
            f.loc[f.subject_id.map(folds) == k, "pmean"] = f[f.subject_id.isin(tr)].y.mean()
        scen = {
            f"within_session_k{k}": {"k": k, "reading_session": sessions[0], "eval_sessions": (sessions[0],)}
            for k in a.k
        }
        if len(sessions) > 1:
            scen["cross_session_transfer"] = {
                "k": max(a.k),
                "reading_session": sessions[0],
                "eval_sessions": tuple(sessions[1:]),
            }
        strategies = a.strategies.split(",") if a.strategies else list(STRATEGIES)
        P, fit = evaluate(f, folds, strategies, pol, scen)
        P["ae"] = (P.yhat - P.y).abs()
        m = (
            P.groupby(["scenario", "strategy"])
            .apply(lambda g: g.groupby("subject_id").ae.mean().mean(), include_groups=False)
            .rename("independent_subject_mae")
        )
        fm = fit.groupby(["scenario", "strategy"]).fit_mae_at_readings.mean().rename("calibration_fit_mae")
        res = pd.concat([m, fm], axis=1).reset_index().assign(target=tgt)
        rows.append(res)
        if a.params_out:
            p = fit_params(f.assign(r=f.y - f.pred, t_start_s=f.t, session=f.session_id))
            pd.DataFrame([{"target": tgt, **p.as_dict()}]).to_csv(
                a.params_out, mode="a", header=not Path(a.params_out).exists(), index=False
            )
    out = pd.concat(rows, ignore_index=True)
    print(out.round(2).to_string(index=False))
    print(
        "independent_subject_mae: held-out windows only (never the calibration readings);"
        " calibration_fit_mae: error at the readings themselves (in-sample)."
    )
    if a.out:
        out.to_csv(a.out, index=False)
    return 0


# ------------------------------------------------------------------------------------------------ infer / export
def cmd_infer(a) -> int:
    from .acquisition.ingest import Recorder, TimestampSync
    from .inference.engine import EngineConfig, InferenceEngine
    from .inference.model_loader import load_bundle
    from .personalization.hierarchical import HierParams

    b = load_bundle(a.model)
    pmap = dict(x.split(":") for x in a.prefix_map.split(",")) if a.prefix_map else {}
    hier = None
    if a.calibration_params:
        cp = pd.read_csv(a.calibration_params)
        hier = {r.target: HierParams(r.sigma_b2, r.sigma_c2, r.sigma_r2, r.sigma_w2, r.kappa) for r in cp.itertuples()}
    eng = InferenceEngine(b, EngineConfig(fs=a.fs, hop_s=a.hop, feature_prefix_map=pmap), hier_params=hier)
    refs = sorted((tuple(map(float, r.split(","))) for r in (a.reference or [])), key=lambda x: x[0])

    def sink(rows):
        eng.push(rows)
        while refs and eng.history and eng.history[-1].timestamp >= refs[0][0]:
            t, s, d = refs.pop(0)
            print("reference:", eng.add_reference(t, s, d))

    src = _source(a)
    if a.source == "sim":
        print(SIM_WARNING)
    Recorder(src, [sink], TimestampSync(a.fs)).run(duration_s=a.duration)
    p = eng.predictions()
    if a.out:
        p.to_csv(a.out, index=False)
    cols = ["timestamp", "status", "sbp", "sbp_lo", "sbp_hi", "dbp", "estimate_type", "latency_ms"]
    print(p[cols].round(1).tail(a.show).to_string(index=False))
    return 0


def cmd_export(a) -> int:
    from .inference.export import export_linear, predict_exported, to_c_header
    from .inference.model_loader import load_bundle

    b = load_bundle(a.model)
    params = export_linear(b)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(a.model).stem
    (out / f"{stem}.json").write_text(json.dumps(params, indent=1) + "\n")
    (out / f"{stem}.h").write_text(to_c_header(params))
    if a.table:
        t = pd.read_csv(a.table).reindex(columns=b.features).to_numpy(float)
        for tgt in ("sbp", "dbp"):
            d = np.abs(predict_exported(params, t, tgt, "float32") - np.ravel(getattr(b, tgt).predict(t)))
            print(f"{tgt}: float32 export max |diff| = {d.max():.2e} mmHg over {len(t)} windows")
    print(f"wrote {out / (stem + '.json')} and {out / (stem + '.h')} (float32; int8 is not recommended)")
    return 0


# ------------------------------------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="rakshpulse", description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", help="environment and self-test")

    s = sub.add_parser("simulate", help="create a SIMULATED dataset")
    s.add_argument("--out", required=True)
    s.add_argument("--subjects", type=int, default=3)
    s.add_argument("--sessions", type=int, default=2)
    s.add_argument("--duration", type=float, default=300.0)
    s.add_argument("--fs", type=float, default=100.0)
    s.add_argument("--reference-every", type=float, default=60.0)
    s.add_argument("--motion-events", type=int, default=1)
    s.add_argument("--dropout", type=float, default=0.0005)
    s.add_argument("--noise", type=float, default=0.0004)
    s.add_argument("--seed", type=int, default=0)

    def source_args(p, need_out=True):
        p.add_argument("--source", choices=("sim", "replay", "serial"), default="sim")
        p.add_argument("--fs", type=float, default=100.0)
        p.add_argument("--duration", type=float, default=None, help="seconds (default: until the source ends)")
        p.add_argument("--input", help="raw_ppg.csv for --source replay")
        p.add_argument("--port", help="serial port, e.g. COM5 or /dev/ttyUSB0")
        p.add_argument("--baud", type=int, default=115200)
        p.add_argument("--fields", default="sample_index,red,ir", help="firmware line fields, comma-separated")
        p.add_argument("--delimiter", default=",")
        p.add_argument("--sim-hr", type=float, default=72.0)
        p.add_argument("--realtime", action="store_true", help="pace the simulator with the wall clock")
        p.add_argument("--seed", type=int, default=0)

    r = sub.add_parser("record", help="record from a source into a dataset")
    source_args(r)
    r.add_argument("--out", required=True)
    r.add_argument("--subject", required=True)
    r.add_argument("--session", required=True)
    r.add_argument("--placement", default="behind_ear")
    r.add_argument("--contact", default="", help="contact conditions (free text)")
    r.add_argument("--notes", default="")
    r.add_argument("--sensor-config", default="", help="MAX30102 settings as JSON, e.g. '{\"sample_rate_hz\": 100}'")
    r.add_argument("--counter-modulus", type=int, default=None)

    f = sub.add_parser("reference", help="append a cuff reading to a session")
    f.add_argument("--dataset", required=True)
    f.add_argument("--subject", required=True)
    f.add_argument("--session", required=True)
    f.add_argument("--time", type=float, required=True, help="session time of the reading (s)")
    f.add_argument("--sbp", type=float, required=True)
    f.add_argument("--dbp", type=float, required=True)
    f.add_argument("--hr", type=float, default=None)
    f.add_argument("--device", default="")

    i = sub.add_parser("inspect", help="summarise / validate a dataset")
    i.add_argument("--dataset", required=True)
    i.add_argument("--validate", action="store_true")

    p = sub.add_parser("preprocess", help="windows, quality and features per session")
    p.add_argument("--dataset", required=True)
    p.add_argument("--subject")
    p.add_argument("--session")
    p.add_argument("--window", type=float, default=10.0)
    p.add_argument("--step", type=float, default=5.0)
    p.add_argument("--band", type=float, nargs=2, default=(0.5, 10.0))
    p.add_argument("--chunksize", type=int, default=200_000)
    p.add_argument("--tag", default="", help="suffix for the derived file names (keeps earlier outputs)")
    p.add_argument("--overwrite", action="store_true", help="replace existing derived files with the same tag")

    fe = sub.add_parser("features", help="training table with associated references")
    fe.add_argument("--dataset", required=True)
    fe.add_argument("--out", required=True)
    fe.add_argument("--tag", default="")
    fe.add_argument("--tolerance", type=float, default=30.0, help="max distance of a reading from a window (s)")
    fe.add_argument("--overwrite", action="store_true")

    t = sub.add_parser("train", help="train and cross-validate a model")
    t.add_argument("--table", required=True)
    t.add_argument("--models-dir", default="models")
    t.add_argument("--name", default="rakshpulse")
    t.add_argument(
        "--model",
        default="ridge",
        help="ridge, elasticnet, svr, random_forest, extra_trees, hist_gb, xgboost, lightgbm, pls, mean",
    )
    t.add_argument("--features", default="ir_core", help="ir_core | all | comma-separated list")
    t.add_argument("--split", choices=("subject", "session", "cross_day"), default="subject")
    t.add_argument("--train-sessions", default="D1")
    t.add_argument("--folds", type=int, default=5)
    t.add_argument("--hyperparams", default="", help="JSON, e.g. '{\"alpha\": 100}'")
    t.add_argument("--tune", action="store_true", help="inner GroupKFold grid search on training subjects")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--allow-simulated", action="store_true")

    c = sub.add_parser("calibrate", help="compare calibration strategies on out-of-fold predictions")
    c.add_argument("--oof", required=True, help="<version>_oof.csv written by `rakshpulse train`")
    c.add_argument("--strategies", default="")
    c.add_argument("--k", type=int, nargs="+", default=[1, 3])
    c.add_argument("--spacing", type=float, default=60.0)
    c.add_argument("--expiry", type=float, default=3600.0)
    c.add_argument("--folds", type=int, default=5)
    c.add_argument("--out", default="")
    c.add_argument("--params-out", default="", help="append fitted hierarchical parameters (for `infer`)")

    e = sub.add_parser("evaluate", help="evaluate a saved model without refitting")
    e.add_argument("--model", required=True)
    e.add_argument("--table", required=True)
    e.add_argument("--statuses", default="usable,degraded")
    e.add_argument("--n-boot", type=int, default=2000)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--out", default="")

    n = sub.add_parser("infer", help="streaming inference from a source")
    source_args(n)
    n.add_argument("--model", required=True)
    n.add_argument("--hop", type=float, default=5.0)
    n.add_argument("--prefix-map", default="", help="e.g. ir_:ear_ to run the 4.0 ear bundle on IR input")
    n.add_argument("--calibration-params", default="")
    n.add_argument("--reference", action="append", help="cuff reading 'time,sbp,dbp' (repeatable)")
    n.add_argument("--out", default="")
    n.add_argument("--show", type=int, default=10)

    x = sub.add_parser("export", help="runtime-free float32 export (JSON + C header)")
    x.add_argument("--model", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--table", default="", help="optional feature table for an equivalence check")
    return ap


COMMANDS = {
    "doctor": cmd_doctor,
    "simulate": cmd_simulate,
    "record": cmd_record,
    "reference": cmd_reference,
    "inspect": cmd_inspect,
    "preprocess": cmd_preprocess,
    "features": cmd_features,
    "train": cmd_train,
    "calibrate": cmd_calibrate,
    "evaluate": cmd_evaluate,
    "infer": cmd_infer,
    "export": cmd_export,
}


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        return COMMANDS[a.cmd](a)
    except (FileExistsError, FileNotFoundError, ValueError, KeyError, RuntimeError) as exc:
        return _fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
