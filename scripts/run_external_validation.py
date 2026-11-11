"""RakshPulse 4.0: external validation on VitalDB, with zero-shot and adaptation results kept apart.

No model is ever fitted on the patients it is evaluated on.

* **E1, zero-shot absolute BP.** Auricle finger population ridge (HR + shape; all Auricle subjects, session 1,
  SQI-accepted) applied to VitalDB finger windows (3.0 subset ``v3`` and the 4.0 extension ``ext``). Truth is the
  arterial line. Control: the Auricle training-mean BP. Reference in-domain row: a VitalDB-v3 ridge applied to
  ``ext``.
* **E2, zero-shot change model.** Auricle finger within-day change model (feature changes since a reference →
  BP change; references every 300 s) applied to VitalDB with **real NIBP** references. Controls:
  ``last_reference``, and ``mean_change`` taken from Auricle (zero-shot, no VitalDB information).
* **E3, independent patients.** Change models fitted on all 40 ``v3`` patients (real NIBP references) and
  evaluated on the 60 ``ext`` patients, without refitting. Controls: ``last_reference``, ``mean_change`` (v3),
  ``historical_mean``, ``population_mean`` (v3 mean arterial BP).
* **E4, domain adaptation (labelled as such).** The same change model fitted on Auricle + VitalDB v3 pooled,
  evaluated on ``ext``. This is compared with E2 (Auricle only) and E3 (VitalDB only) on identical windows.

Subject (= patient) level paired sign-flip tests with bootstrap CIs; Holm correction across the whole family per
target. Bland–Altman (repeated measures) for E1. A feature-compatibility table documents Auricle vs VitalDB
feature distributions.

Outputs: results/external_v4/{metrics,paired,bland_altman,feature_shift}.csv and a provenance run.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_beyond_reference import grid_refs  # noqa: E402

from rakshpulse.evaluation import auricle_study as st  # noqa: E402
from rakshpulse.evaluation.beyond_reference import make_pairs  # noqa: E402
from rakshpulse.evaluation.statistics import (  # noqa: E402
    bland_altman_repeated,
    error_distribution,
    holm,
    paired_test,
    subject_errors,
)  # noqa: E402
from rakshpulse.experiments.registry import ExperimentRun  # noqa: E402
from rakshpulse.features.advanced import PAT_COLUMNS, WAVELET_COLUMNS  # noqa: E402

AUR, VDB, VEXT = (ROOT / "data" / n for n in ("auricle", "vitaldb", "vitaldb_ext"))
OUT = ROOT / "results" / "external_v4"
SEED, N_BOOT = 0, 2000
HR_SHAPE = ["finger_hr"] + [f"finger_{c}" for c in st.SHAPE_FEATURES]
HR_SHAPE_PAT = HR_SHAPE + [f"finger_{c}" for c in PAT_COLUMNS[:2]]
log = logging.getLogger("external")


def ridge():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(np.logspace(-1, 4, 12)))


def load_auricle() -> pd.DataFrame:
    w = pd.read_csv(AUR / "windows.csv", low_memory=False)
    f = pd.read_csv(AUR / "features.csv").drop(columns=["subject_id", "record_id", "session"])
    adv = pd.read_csv(AUR / "features_advanced.csv").drop(columns=["subject_id", "record_id", "session"])
    d = w.merge(f, on="window_id").merge(adv, on="window_id")
    d = d[(d.cohort == "HC") & d.label_valid & d.finger_sqi_ok].rename(columns={"activity_code": "activity"})
    st.add_normalised_shape(d, "finger")
    return d.reset_index(drop=True)


def load_vitaldb(folder: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    w = pd.read_csv(folder / "windows.csv")
    f = pd.read_csv(folder / "features.csv").drop(columns=["subject_id"])
    d = w.merge(f, on="window_id")
    d = d[d.label_valid & d.finger_sqi_ok].reset_index(drop=True)
    d["session"], d["activity"] = 1, -1
    st.add_normalised_shape(d, "finger")
    d = d.rename(columns={c: f"finger_{c}" for c in list(PAT_COLUMNS) + list(WAVELET_COLUMNS)})
    return d, pd.read_csv(folder / "nibp_readings.csv")


def vitaldb_pairs(d: pd.DataFrame, nib: pd.DataFrame, cols: list[str], target: str) -> pd.DataFrame:
    nib_col, y = ("nibp_sbp", "art_sbp_mmHg") if target == "sbp" else ("nibp_dbp", "art_dbp_mmHg")
    refs = []
    for r in nib.itertuples():
        g = d[(d.caseid == r.caseid) & ((d.t_start_s - r.t_s).abs() <= 60)]
        if g.empty:
            continue
        x = g.iloc[int((g.t_start_s - r.t_s).abs().argmin())]
        refs.append(
            {
                "subject_id": x.subject_id,
                "session": 1,
                "t_ref_s": r.t_s,
                "ref_reported": getattr(r, nib_col),
                "ref_truth": x[y],
                **{f"ref_{c}": x[c] for c in cols},
            }
        )
    return make_pairs(d, pd.DataFrame(refs), cols, y, min_gap_s=10.0, max_gap_s=1800.0)


def fit_change(pairs: pd.DataFrame, cols: list[str]):
    m = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=0.0), StandardScaler(), RidgeCV(np.logspace(-1, 4, 12))
    )
    return m.fit(pairs[[f"d_{c}" for c in cols]], pairs.y - pairs.ref_truth)


def predict_change(m, pairs: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return pairs.ref_reported.to_numpy() + m.predict(pairs[[f"d_{c}" for c in cols]])


def frame(pairs: pd.DataFrame, yhat: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"subject_id": pairs.subject_id.to_numpy(), "y": pairs.y.to_numpy(), "yhat": yhat})


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    run = ExperimentRun(
        "external_validation_v4",
        [
            AUR / "windows.csv",
            AUR / "features.csv",
            VDB / "windows.csv",
            VDB / "features.csv",
            VDB / "nibp_readings.csv",
            VEXT / "windows.csv",
            VEXT / "features.csv",
            VEXT / "nibp_readings.csv",
        ],
        {"n_boot": N_BOOT, "auricle_refs": "every 300 s (CNAP)", "vitaldb_refs": "real NIBP, gap <= 1800 s"},
        seed=SEED,
        t0=t0,
        dataset_versions={
            "auricle": "PhysioNet 1.0.0",
            "vitaldb_v3": "PhysioNet 1.0.0, 40 cases (seed 0)",
            "vitaldb_ext": "PhysioNet 1.0.0, 60 further cases (seed 1, disjoint)",
        },
        model_config={"population": "ridge HR + shape", "change": "RidgeCV on feature changes"},
    )
    a = load_auricle()
    v3, nib3 = load_vitaldb(VDB)
    vx, nibx = load_vitaldb(VEXT)
    metrics, paired, ba, shift = [], [], [], []
    # feature compatibility (standardised mean difference vs Auricle)
    for c in HR_SHAPE:
        for name, d in (("vitaldb_v3", v3), ("vitaldb_ext", vx)):
            pooled = np.sqrt((a[c].var() + d[c].var()) / 2)
            shift.append(
                {
                    "feature": c,
                    "dataset": name,
                    "auricle_median": a[c].median(),
                    "median": d[c].median(),
                    "smd_vs_auricle": float((d[c].mean() - a[c].mean()) / pooled) if pooled > 0 else np.nan,
                }
            )
    with run.phase("train"):
        pop_a = {
            t: ridge().fit(a.loc[a.session == 1, HR_SHAPE], a.loc[a.session == 1, f"{t}_mmHg"]) for t in ("sbp", "dbp")
        }
        pop_v3 = {t: ridge().fit(v3[HR_SHAPE], v3[f"art_{t}_mmHg"]) for t in ("sbp", "dbp")}
    for target in ("sbp", "dbp"):
        fam, errs = [], {}
        with run.phase("evaluate"):
            # ---------------- E1 zero-shot absolute
            y_col = f"art_{target}_mmHg"
            mean_a = a.loc[a.session == 1, f"{target}_mmHg"].mean()
            for name, d in (("vitaldb_v3", v3), ("vitaldb_ext", vx)):
                rows = {
                    "E1_auricle_ridge_zero_shot": pop_a[target].predict(d[HR_SHAPE]),
                    "E1_auricle_training_mean": np.full(len(d), mean_a),
                }
                if name == "vitaldb_ext":
                    rows["E1_vitaldb_v3_ridge"] = pop_v3[target].predict(d[HR_SHAPE])
                    rows["E1_vitaldb_v3_mean"] = np.full(len(d), v3[y_col].mean())
                for meth, yhat in rows.items():
                    t = pd.DataFrame({"subject_id": d.subject_id, "y": d[y_col], "yhat": yhat})
                    metrics.append(
                        {
                            "experiment": "E1",
                            "test_set": name,
                            "target": target,
                            "method": meth,
                            **st.point_metrics(t),
                            **error_distribution(t),
                        }
                    )
                    ba.append({"test_set": name, "target": target, "method": meth, **bland_altman_repeated(t, 500)})
                    errs[(name, meth)] = subject_errors(t)
                fam.append(
                    {
                        "experiment": "E1",
                        "test_set": name,
                        "comparison": "auricle ridge vs auricle training mean",
                        **paired_test(
                            errs[(name, "E1_auricle_ridge_zero_shot")],
                            errs[(name, "E1_auricle_training_mean")],
                            N_BOOT,
                            seed=SEED,
                        ),
                    }
                )
                if name == "vitaldb_ext":  # in-domain absolute model on new patients (not zero-shot)
                    fam.append(
                        {
                            "experiment": "E1",
                            "test_set": name,
                            "comparison": "vitaldb_v3 ridge vs vitaldb_v3 mean (in-domain, new patients)",
                            **paired_test(
                                errs[(name, "E1_vitaldb_v3_ridge")],
                                errs[(name, "E1_vitaldb_v3_mean")],
                                N_BOOT,
                                seed=SEED,
                            ),
                        }
                    )
            # ---------------- change models
            pa = make_pairs(a, grid_refs(a, 300.0, HR_SHAPE_PAT, f"{target}_mmHg"), HR_SHAPE_PAT, f"{target}_mmHg")
            p3 = vitaldb_pairs(v3, nib3, HR_SHAPE_PAT, target)
            px = vitaldb_pairs(vx, nibx, HR_SHAPE_PAT, target)
            mc_a = float((pa.y - pa.ref_truth).mean())
            mc_v3 = float((p3.y - p3.ref_truth).mean())
            ch_a = fit_change(pa, HR_SHAPE)
            ch_v3 = {"hr_shape": fit_change(p3, HR_SHAPE), "hr_shape_pat": fit_change(p3, HR_SHAPE_PAT)}
            pooled = pd.concat([pa, p3], ignore_index=True)
            ch_pool = fit_change(pooled, HR_SHAPE)
            for name, pp in (("vitaldb_v3", p3), ("vitaldb_ext", px)):
                rows = {
                    "last_reference": pp.ref_reported.to_numpy(),
                    "historical_mean": pp.historical_mean.to_numpy(),
                    "E2_mean_change_auricle": pp.ref_reported.to_numpy() + mc_a,
                    "E2_delta_hr_shape_auricle_zero_shot": predict_change(ch_a, pp, HR_SHAPE),
                }
                if name == "vitaldb_ext":
                    rows |= {
                        "E3_population_mean_v3": np.full(len(pp), v3[y_col].mean()),
                        "E3_mean_change_v3": pp.ref_reported.to_numpy() + mc_v3,
                        "E3_delta_hr_shape_v3": predict_change(ch_v3["hr_shape"], pp, HR_SHAPE),
                        "E3_delta_hr_shape_pat_v3": predict_change(ch_v3["hr_shape_pat"], pp, HR_SHAPE_PAT),
                        "E4_delta_hr_shape_pooled_adapted": predict_change(ch_pool, pp, HR_SHAPE),
                    }
                for meth, yhat in rows.items():
                    t = frame(pp, yhat)
                    metrics.append(
                        {
                            "experiment": meth.split("_")[0] if meth[:2] in ("E2", "E3", "E4") else "control",
                            "test_set": name,
                            "target": target,
                            "method": meth,
                            **st.point_metrics(t),
                        }
                    )
                    errs[(name, meth)] = subject_errors(t)
                comps = [
                    ("E2_delta_hr_shape_auricle_zero_shot", "E2_mean_change_auricle"),
                    ("E2_delta_hr_shape_auricle_zero_shot", "last_reference"),
                ]
                if name == "vitaldb_ext":
                    comps += [
                        ("E3_delta_hr_shape_v3", "E3_mean_change_v3"),
                        ("E3_delta_hr_shape_pat_v3", "E3_mean_change_v3"),
                        ("E3_delta_hr_shape_v3", "last_reference"),
                        ("E3_population_mean_v3", "last_reference"),
                        ("historical_mean", "last_reference"),
                        ("E4_delta_hr_shape_pooled_adapted", "E3_delta_hr_shape_v3"),
                        ("E4_delta_hr_shape_pooled_adapted", "E3_mean_change_v3"),
                    ]
                for x, b in comps:
                    fam.append(
                        {
                            "experiment": x.split("_")[0] if x[:2] in ("E2", "E3", "E4") else "control",
                            "test_set": name,
                            "comparison": f"{x} vs {b}",
                            **paired_test(errs[(name, x)], errs[(name, b)], N_BOOT, seed=SEED),
                        }
                    )
            for r, p in zip(fam, holm([r["p_perm"] for r in fam]), strict=True):
                r["p_holm"] = p
                r["target"] = target
            paired += fam
        log.info("%s done", target)
    for name, rows in (("metrics", metrics), ("paired", paired), ("bland_altman", ba), ("feature_shift", shift)):
        pd.DataFrame(rows).to_csv(OUT / f"{name}.csv", index=False, float_format="%.4f")
    run.splits.update({s: 0 for s in v3.subject_id.unique()} | {s: 1 for s in vx.subject_id.unique()})
    run.add_metrics(pd.DataFrame(metrics))
    run.finish()
    m = pd.DataFrame(metrics)
    print(
        m[["experiment", "test_set", "target", "method", "mae", "bias", "n_subjects"]].round(2).to_string(index=False)
    )
    pr = pd.DataFrame(paired)
    print(
        pr[["target", "test_set", "comparison", "mean_diff", "ci_low", "ci_high", "p_holm"]]
        .round(3)
        .to_string(index=False)
    )
    log.info("done in %.0fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    raise SystemExit(main())
