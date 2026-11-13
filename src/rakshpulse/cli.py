"""``rakshpulse`` command-line interface (RakshPulse 5.1).

    rakshpulse doctor                       check the environment, optional packages, the default model, self-test
    rakshpulse models                        list installed models, their contracts and sensor compatibility
    rakshpulse ports                         list serial ports (needs pyserial)
    rakshpulse check-serial --port P --fs F  read a few seconds from the device and check the data contract
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
    rakshpulse analyze    --dataset DIR --subject S --session D   validate → process → predict → compare → export

Exit codes: 0 success · 1 a check failed / no trustworthy result · 2 invalid usage or input · 3 device or
acquisition problem · 4 incompatible or unsupported model.

Use ``rakshpulse <command> -h`` for options. Nothing here is a medical device. Simulated data is never evidence, and
predictions labelled ``demonstration`` are not blood-pressure measurements.
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
DEMO_WARNING = (
    "DEMONSTRATION MODE: this model has not been validated for this sensor; the numbers show that the "
    "pipeline runs and are NOT blood-pressure measurements."
)
EXIT_OK, EXIT_CHECK, EXIT_USAGE, EXIT_DEVICE, EXIT_MODEL = 0, 1, 2, 3, 4
DEFAULT_MODEL_NAME = "auricle_ear_ridge_v2.joblib"


def _fail(msg: str, code: int = EXIT_USAGE) -> int:
    print(f"error: {msg}", file=sys.stderr)
    return code


def _models_dir(arg: str | None = None) -> Path:
    """``--models-dir``, else ./models, else the repository's models/ folder."""
    if arg:
        return Path(arg)
    here = Path("models")
    return here if here.is_dir() else Path(__file__).resolve().parents[2] / "models"


def _model_path(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    p = _models_dir() / DEFAULT_MODEL_NAME
    if not p.exists():
        raise FileNotFoundError(f"default model {p} not found; pass --model PATH (see `rakshpulse models`)")
    return p


def _demographics(text: str | None) -> dict | None:
    """'age=35,sex=1,height_cm=172,weight_kg=70' → dict (values validated later by the model contract)."""
    if not text:
        return None
    out = {}
    for part in text.split(","):
        if "=" not in part:
            raise ValueError(f"--demographics expects key=value pairs, got {part!r}")
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _default_sensor(a) -> str:
    if getattr(a, "sensor", None):
        return a.sensor
    if a.source == "sim":
        return "simulator"
    if a.source == "serial":
        return "max30102"
    raise ValueError("--sensor is required for --source replay (e.g. max30102 or simulator)")


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
    try:
        from .inference.contract import check_compatibility, validate_bundle
        from .inference.model_loader import load_bundle

        mp = _model_path(None)
        c = validate_bundle(load_bundle(mp))
        compat = check_compatibility(c, "max30102", 100.0)
        print(f"  ok   default model {mp.name}: {c.get('model_id')} runs on max30102 data in '{compat['mode']}' mode")
    except Exception as exc:
        ok = False
        print(f"  FAIL default model: {exc}")
    try:
        import serial  # noqa: F401  # type: ignore[import-not-found]

        print("  ok   serial recording available (pyserial installed)")
    except ImportError:
        print("  n/a  serial recording needs: pip install -e .[serial]")
    print("doctor:", "all required checks passed" if ok else "problems found")
    return 0 if ok else 1


# ------------------------------------------------------------------------------------------------ models / ports
def cmd_models(a) -> int:
    from .inference.contract import IncompatibleModelError, check_compatibility, get_contract, summary, validate_bundle
    from .inference.model_loader import load_bundle

    paths = [Path(a.model)] if a.model else sorted(_models_dir(a.models_dir).glob("*.joblib"))
    if not paths:
        return _fail(f"no model bundles found in {_models_dir(a.models_dir)}", EXIT_CHECK)
    rows, bad = [], 0
    for path in paths:
        try:
            b = load_bundle(path)
            c = validate_bundle(b)
        except Exception as exc:
            bad += 1
            rows.append({"file": path.name, "status": f"INVALID: {exc}"})
            continue
        try:
            compat = check_compatibility(c, a.sensor, a.fs, demographics=None if c.get("demographics") else {})
            verdict = compat["mode"]
            unc = "available" if compat["uncertainty"] else "unavailable"
        except IncompatibleModelError as exc:
            msg = str(exc)
            verdict = "needs demographics" if "demographic" in msg else f"unsupported ({msg})"
            unc = "-"
        row = {
            "file": path.name,
            "version": b.version,
            **summary(get_contract(b)),
            f"mode_for_{a.sensor}": verdict,
            "uncertainty_for_sensor": unc,
        }
        rows.append(row)
    if a.verbose:
        for r in rows:
            print("\n".join(f"{k:>24}: {v}" for k, v in r.items()), end="\n\n")
    else:
        cols = [
            "file",
            "model_id",
            "version",
            f"mode_for_{a.sensor}",
            "uncertainty_for_sensor",
            "demographics",
            "channels",
            "features",
        ]
        print(pd.DataFrame(rows).reindex(columns=cols).fillna("").to_string(index=False))
        print(
            "\nModes: validated (none exist yet) > experimental (trained on this sensor type, cross-validated only) > "
            "demonstration (pipeline demo, not BP). Use --verbose for limitations."
        )
    return EXIT_CHECK if bad else EXIT_OK


def cmd_ports(a) -> int:
    from .acquisition.sources import list_ports

    ports = list_ports()
    if not ports:
        print("no serial ports found: plug in the board, install its USB driver (CP210x / CH340), then retry")
        return EXIT_DEVICE
    print(pd.DataFrame(ports).to_string(index=False))
    return EXIT_OK


def cmd_check_serial(a) -> int:
    from .acquisition.sources import LineFormat, SerialSource, check_serial

    src = SerialSource(
        a.port, a.baud, a.fs, LineFormat(tuple(a.fields.split(",")), a.delimiter), no_data_timeout_s=a.timeout
    )
    r = check_serial(src, seconds=a.seconds)
    for k, v in r.items():
        if k != "problems":
            print(f"{k:>18}: {v}")
    if r["ok"]:
        print("serial check: PASS (data matches the RakshPulse contract)")
        return EXIT_OK
    for pr in r["problems"]:
        print("  problem:", pr)
    print("serial check: FAIL (see docs/HARDWARE_HANDOFF.md, Troubleshooting)")
    return EXIT_DEVICE if r["samples"] == 0 else EXIT_CHECK


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

        if a.fs is None:
            a.fs = 100.0  # simulator default, stated in --help
        return SimulatedSensor(SimConfig(fs=a.fs, hr_bpm=a.sim_hr, seed=a.seed), a.duration or 60, realtime=a.realtime)
    if a.fs is None:
        raise ValueError(
            f"--fs (effective sampling rate in Hz) is required for --source {a.source}; it is never "
            "assumed. For a MAX30102 it is sample_rate / sample_averaging."
        )
    if a.source == "replay":
        if not a.input:
            raise ValueError("--input raw_ppg.csv is required for --source replay")
        return ReplaySource(a.input, a.fs)
    if a.source == "serial":
        if not a.port:
            raise ValueError("--port is required for --source serial (list them with `rakshpulse ports`)")
        return SerialSource(
            a.port, a.baud, a.fs, LineFormat(tuple(a.fields.split(",")), a.delimiter), no_data_timeout_s=a.timeout
        )
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
    from .acquisition.sources import AcquisitionError

    src = _source(a)
    sim = a.source == "sim"
    if sim:
        print(SIM_WARNING)
    sensor = _default_sensor(a)
    store = DatasetStore(a.out)
    if store.session_dir(a.subject, a.session).exists():
        return _fail(f"session {a.subject}/{a.session} already exists in {a.out}; choose a new --session")
    state: dict = {"writer": None}

    def sink(rows):  # the session is created only when valid data arrives and the firmware rate matches
        if state["writer"] is None:
            declared = getattr(src, "device_info", {}).get("fs_hz")
            if declared is not None and abs(float(declared) - a.fs) > 1e-6:
                raise AcquisitionError(f"firmware declares fs_hz={declared} but --fs is {a.fs:g}; fix one of them")
            info = getattr(src, "device_info", {})
            notes = "; ".join(
                x
                for x in (
                    a.notes,
                    f"transport={getattr(src, 'transport', a.source)}",
                    f"firmware: {json.dumps(info)}" if info else "",
                )
                if x
            )
            meta = SessionMetadata(
                a.subject,
                a.session,
                a.fs,
                sensor_type=sensor,
                placement=a.placement,
                contact_conditions=a.contact,
                notes=notes,
                sensor_config=sensor_cfg,
            )
            state["writer"] = store.create_session(meta)
            print(f"recording {a.subject}/{a.session} at {a.fs:g} Hz from {a.source} (Ctrl-C to stop)")
        state["writer"].write_raw(rows)

    rec = Recorder(src, [sink], TimestampSync(a.fs, counter_modulus=a.counter_modulus))
    try:
        try:
            s = rec.run(duration_s=a.duration)
        except KeyboardInterrupt:
            rec.stop()
            s = {"samples": rec.sync.n_samples, "missing_samples": rec.sync.n_missing, "stopped_by": "ctrl-c"}
    finally:
        if state["writer"] is not None:  # always close: files are flushed, hashed and registered
            state["writer"].add_gaps(rec.sync.gap_table())
            state["writer"].close()
    if state["writer"] is None:
        print(json.dumps(s, default=float))
        return _fail("no valid samples were received; nothing was recorded", EXIT_DEVICE)
    s.update(
        duplicates=rec.sync.n_duplicates, counter_resets=rec.sync.n_resets, malformed_lines=getattr(src, "bad_lines", 0)
    )
    print(json.dumps(s, default=float))
    drift = s.get("drift_ppm")
    if drift is not None and np.isfinite(drift) and abs(drift) > 20_000:
        print(
            f"warning: the device delivered samples {drift / 1e4:+.1f} % faster/slower than --fs {a.fs:g} Hz; "
            "check the firmware sampling rate and averaging"
        )
    if s["samples"] and s["missing_samples"] / (s["samples"] + s["missing_samples"]) > 0.05:
        print("warning: more than 5 % of samples are missing (see gaps.csv)")
    return EXIT_OK


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
    code = EXIT_OK
    if a.quality:
        q = []
        for _, m in ses.iterrows():
            wp = st.session_dir(str(m.subject_id), str(m.session_id)) / "derived" / "windows.csv"
            if not wp.exists():
                q.append(
                    {
                        "subject_id": m.subject_id,
                        "session_id": m.session_id,
                        "windows": "not processed (run `rakshpulse preprocess`)",
                    }
                )
                continue
            w = pd.read_csv(wp)
            reasons = w.reasons.fillna("").str.split("|").explode()
            reasons = reasons[reasons != ""].value_counts().head(3)
            q.append(
                {
                    "subject_id": m.subject_id,
                    "session_id": m.session_id,
                    "windows": len(w),
                    **{k: f"{v:.0%}" for k, v in w.status.value_counts(normalize=True).items()},
                    "top_reasons": "; ".join(f"{k} ({v})" for k, v in reasons.items()),
                }
            )
        print(pd.DataFrame(q).fillna("").to_string(index=False))
    if a.validate:
        v = st.validate()
        print(v[v.status != "PASS"].to_string(index=False) if (v.status != "PASS").any() else "validation: all PASS")
        code = EXIT_CHECK if (v.status == "FAIL").any() else EXIT_OK
    return code


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
        w["sensor_type"] = str(m.sensor_type)
        w["sampling_rate"] = float(m.sampling_rate)
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

    b = load_bundle(_model_path(a.model))
    pmap = dict(x.split(":") for x in a.prefix_map.split(",")) if a.prefix_map else {}
    hier = None
    if a.calibration_params:
        cp = pd.read_csv(a.calibration_params)
        hier = {r.target: HierParams(r.sigma_b2, r.sigma_c2, r.sigma_r2, r.sigma_w2, r.kappa) for r in cp.itertuples()}
    src = _source(a)  # resolves --fs (explicit for replay/serial)
    eng = InferenceEngine(
        b,
        EngineConfig(
            fs=a.fs,
            sensor=_default_sensor(a),
            hop_s=a.hop,
            feature_prefix_map=pmap,
            demographics=_demographics(a.demographics),
        ),
        hier_params=hier,
    )
    info = eng.describe()
    print(f"model {info['model_id']} ({info['version']}) on {info['sensor']} at {info['fs']:g} Hz: mode={info['mode']}")
    if info["mode"] == "demonstration":
        print(DEMO_WARNING)
    if not eng.halfwidth:
        print(f"note: {eng.uncertainty_note}")
    refs = sorted((tuple(map(float, r.split(","))) for r in (a.reference or [])), key=lambda x: x[0])

    def sink(rows):
        eng.push(rows)
        while refs and eng.history and eng.history[-1].timestamp >= refs[0][0]:
            t, s, d = refs.pop(0)
            print("reference:", eng.add_reference(t, s, d))

    if a.source == "sim":
        print(SIM_WARNING)
    try:
        Recorder(src, [sink], TimestampSync(a.fs)).run(duration_s=a.duration)
    except KeyboardInterrupt:
        print("stopped")
    p = eng.predictions()
    if a.out:
        p.to_csv(a.out, index=False)
    cols = ["timestamp", "status", "inference_mode", "sbp", "sbp_lo", "sbp_hi", "dbp", "estimate_type", "latency_ms"]
    print(p[cols].round(1).tail(a.show).to_string(index=False))
    ok = p.sbp.notna().any()
    if not ok:
        print("no trustworthy prediction: no window passed the quality and input checks (see the reasons column)")
    return EXIT_OK if ok else EXIT_CHECK


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


# ------------------------------------------------------------------------------------------------ analyze
def cmd_analyze(a) -> int:
    from .datastore.association import associate_references
    from .datastore.store import DatasetStore
    from .inference.contract import get_contract
    from .inference.engine import predict_windows
    from .inference.model_loader import load_bundle
    from .preprocessing.pipeline import PipelineConfig, process_recording

    st = DatasetStore(a.dataset)
    d = st.session_dir(a.subject, a.session)
    if not (d / "session_metadata.csv").exists():
        return _fail(f"no session {a.subject}/{a.session} in {a.dataset}")
    m = st.metadata(a.subject, a.session)
    fs, sensor = float(m["sampling_rate"]), a.sensor or str(m["sensor_type"])
    b = load_bundle(_model_path(a.model))
    print(f"[1/5] validating {a.subject}/{a.session} ({sensor}, {fs:g} Hz)")
    v = st.validate()
    v = v[(v.subject_id.astype(str) == a.subject) & (v.session_id.astype(str) == a.session)]
    fails = v[v.status == "FAIL"]
    if len(fails):
        print(fails.to_string(index=False))
        if not a.force:
            return _fail("session failed validation (use --force to analyse anyway)", EXIT_CHECK)
    print("[2/5] processing windows")
    wpath = d / "derived" / "windows.csv"
    if wpath.exists():
        windows = pd.read_csv(wpath, dtype={"subject_id": str, "session_id": str})
    else:
        windows, seg = process_recording(
            st.raw(a.subject, a.session, chunksize=200_000), PipelineConfig(fs=fs), a.subject, a.session
        )
        st.write_derived(a.subject, a.session, "windows", windows)
        st.write_derived(a.subject, a.session, "segments", seg)
    if windows.empty:
        return _fail("recording too short: no complete 10 s window without a gap (record at least 30 s)", EXIT_CHECK)
    print("[3/5] model checks and prediction")
    pred, info = predict_windows(b, windows, sensor, fs, _demographics(a.demographics))
    c = get_contract(b)
    print(f"      model {c.get('model_id')} ({b.version}): mode={info['mode']}")
    if info["mode"] == "demonstration":
        print("      " + DEMO_WARNING)
    acc = pred[pred.sbp.notna()]
    counts = pred.status.value_counts().to_dict()
    print(f"      windows: {len(pred)} total, {len(acc)} predicted; status counts {counts}")
    if acc.empty:
        top = pred.reasons.fillna("").str.split("|").explode().value_counts().head(5)
        print("      top reasons:", "; ".join(f"{k} ({n})" for k, n in top.items() if k))
        return _fail("no trustworthy prediction: no window passed the quality and input checks", EXIT_CHECK)
    print("[4/5] cuff references")
    refs = st.references(a.subject, a.session)
    refs = refs[refs.sbp.between(50, 260) & refs.dbp.between(25, 160) & (refs.sbp > refs.dbp)].sort_values("timestamp")
    lw, lr = associate_references(
        windows.assign(quality_class=windows.quality_class), refs, a.tolerance, require_usable=True
    )
    pred = pred.merge(lw[["window_id", "ref_sbp", "ref_dbp", "ref_timestamp"]], on="window_id", how="left")
    pred["sbp_calibrated"] = np.nan
    pred["dbp_calibrated"] = np.nan
    pred["role"] = "prediction"
    cal_t = None
    if a.calibrate:
        cal = pred[pred.ref_sbp.notna() & pred.sbp.notna()].drop_duplicates("ref_timestamp").head(a.calibrate)
        if len(cal) < a.calibrate:
            print(f"      only {len(cal)} usable readings for --calibrate {a.calibrate}; no calibration applied")
        else:
            cal_t = float(cal.ref_timestamp.max())
            off_s, off_d = float((cal.ref_sbp - cal.sbp).mean()), float((cal.ref_dbp - cal.dbp).mean())
            pred.loc[pred.ref_timestamp.isin(cal.ref_timestamp), "role"] = "calibration"
            after = (pred.t_start_s > cal_t) & (pred.role != "calibration") & pred.sbp.notna()
            pred.loc[after, "sbp_calibrated"] = pred.loc[after, "sbp"] + off_s
            pred.loc[after, "dbp_calibrated"] = pred.loc[after, "dbp"] + off_d
            print(
                f"      calibrated with {len(cal)} readings (offset SBP {off_s:+.1f}, DBP {off_d:+.1f} mmHg); "
                "calibration windows are excluded from the comparison"
            )
    cmp_ = pred[pred.ref_sbp.notna() & pred.sbp.notna() & (pred.role == "prediction")].drop_duplicates("ref_timestamp")
    if len(cmp_):
        use = "calibrated" if cal_t is not None else "population"
        ps = cmp_.sbp_calibrated if cal_t is not None else cmp_.sbp
        pd_ = cmp_.dbp_calibrated if cal_t is not None else cmp_.dbp
        ok = ps.notna()
        if ok.any():
            es, ed = (ps - cmp_.ref_sbp)[ok], (pd_ - cmp_.ref_dbp)[ok]
            print(
                f"      {int(ok.sum())} independent readings vs {use} estimate: SBP MAE {es.abs().mean():.1f} "
                f"(mean error {es.mean():+.1f}), DBP MAE {ed.abs().mean():.1f} (mean error {ed.mean():+.1f}) mmHg"
                + ("  [demonstration mode: not an accuracy claim]" if info["mode"] == "demonstration" else "")
            )
    else:
        print("      no independent cuff reading matched an accepted window (nothing to compare)")
    print("[5/5] writing results")
    name = "predictions_" + "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(c.get("model_id")))
    out = st.write_derived(a.subject, a.session, name, pred, overwrite_derived=a.overwrite)
    med_s, med_d = float(acc.sbp.median()), float(acc.dbp.median())
    hw = info["uncertainty"]
    unc = f" ± {hw['sbp']:.0f}/{hw['dbp']:.0f} (90 %)" if hw else f"  ({info['uncertainty_note']})"
    print(f"      median estimate over accepted windows: {med_s:.0f}/{med_d:.0f} mmHg{unc}")
    print(f"      wrote {out}")
    return EXIT_OK


# ------------------------------------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="rakshpulse", description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", help="environment, default model and self-test")

    mo = sub.add_parser("models", help="list models, contracts and compatibility with a sensor")
    mo.add_argument("--models-dir", default=None)
    mo.add_argument("--model", default=None, help="inspect one bundle file")
    mo.add_argument("--sensor", default="max30102", help="sensor to check compatibility for (default max30102)")
    mo.add_argument("--fs", type=float, default=100.0, help="raw sampling rate to check (Hz)")
    mo.add_argument("--verbose", action="store_true", help="show the full contract summary and limitations")

    sub.add_parser("ports", help="list serial ports (needs pyserial)")

    cs = sub.add_parser(
        "check-serial",
        help="read a few seconds from the device and check the data contract (requires the connected board)",
    )
    cs.add_argument("--port", required=True, help="e.g. COM5 (Windows) or /dev/ttyUSB0 (Linux)")
    cs.add_argument("--baud", type=int, default=115200)
    cs.add_argument("--fs", type=float, required=True, help="effective sampling rate = sample_rate / averaging")
    cs.add_argument("--seconds", type=float, default=10.0)
    cs.add_argument("--timeout", type=float, default=5.0, help="fail if no valid sample arrives for this long (s)")
    cs.add_argument("--fields", default="sample_index,red,ir")
    cs.add_argument("--delimiter", default=",")

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
        p.add_argument(
            "--source",
            choices=("sim", "replay", "serial"),
            default="sim",
            help="sim (simulated), replay (raw_ppg.csv) or serial (connected board)",
        )
        p.add_argument(
            "--fs",
            type=float,
            default=None,
            help="effective sampling rate in Hz (required for replay/serial; simulator default 100)",
        )
        p.add_argument(
            "--sensor",
            default=None,
            help="input sensor key (default: simulator for sim, max30102 for serial; required for replay)",
        )
        p.add_argument("--timeout", type=float, default=5.0, help="serial: fail if no valid sample for this long (s)")
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
    i.add_argument("--validate", action="store_true", help="checksums, schema, ordering, ranges, references")
    i.add_argument("--quality", action="store_true", help="signal-quality summary of processed sessions")

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
    n.add_argument("--model", default=None, help=f"model bundle (default models/{DEFAULT_MODEL_NAME})")
    n.add_argument("--hop", type=float, default=5.0, help="seconds between predictions")
    n.add_argument(
        "--demographics", default="", help="for models that need them: age=..,sex=0|1,height_cm=..,weight_kg=.."
    )
    n.add_argument("--prefix-map", default="", help="e.g. ir_:ear_ to run the 4.0 ear bundle on IR input")
    n.add_argument("--calibration-params", default="")
    n.add_argument("--reference", action="append", help="cuff reading 'time,sbp,dbp' (repeatable)")
    n.add_argument("--out", default="")
    n.add_argument("--show", type=int, default=10)

    x = sub.add_parser("export", help="runtime-free float32 export (JSON + C header)")
    x.add_argument("--model", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--table", default="", help="optional feature table for an equivalence check")

    an = sub.add_parser("analyze", help="validate, process, predict and export one recorded session")
    an.add_argument("--dataset", required=True)
    an.add_argument("--subject", required=True)
    an.add_argument("--session", required=True)
    an.add_argument("--model", default=None, help=f"model bundle (default models/{DEFAULT_MODEL_NAME})")
    an.add_argument("--sensor", default=None, help="override the session's sensor type")
    an.add_argument(
        "--demographics", default="", help="age=..,sex=0|1,height_cm=..,weight_kg=.. (if the model needs them)"
    )
    an.add_argument("--calibrate", type=int, default=0, help="use the first N cuff readings as calibration")
    an.add_argument("--tolerance", type=float, default=30.0, help="max distance of a cuff reading from a window (s)")
    an.add_argument("--force", action="store_true", help="analyse even if validation fails")
    an.add_argument("--overwrite", action="store_true", help="replace an earlier predictions file for this model")
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
    "models": cmd_models,
    "ports": cmd_ports,
    "check-serial": cmd_check_serial,
    "analyze": cmd_analyze,
}


def main(argv: list[str] | None = None) -> int:
    from .acquisition.sources import AcquisitionError
    from .inference.contract import IncompatibleModelError

    a = build_parser().parse_args(argv)
    try:
        return COMMANDS[a.cmd](a)
    except AcquisitionError as exc:
        return _fail(str(exc), EXIT_DEVICE)
    except IncompatibleModelError as exc:
        return _fail(f"model not usable for this input: {exc}", EXIT_MODEL)
    except (FileExistsError, FileNotFoundError, ValueError, KeyError, RuntimeError) as exc:
        return _fail(str(exc), EXIT_USAGE)


if __name__ == "__main__":
    raise SystemExit(main())
