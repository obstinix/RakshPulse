"""Auxiliary, ECG-assisted experiment (NOT a device configuration; MAX30102 has no ECG).

Computes PAT on the same Guilin windows used by the PPG experiments, then compares
PPG-only vs PPG+PAT vs PAT-only under identical subject folds (ridge, XGBoost).
Writes results/ecg_aux_experiments.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_experiments import FEATURE_SETS, evaluate, fold_map, gate, splits_from_map  # noqa: E402

from rakshpulse.data.build import PROCESSED  # noqa: E402
from rakshpulse.data.dataset_loader import iter_guilin  # noqa: E402
from rakshpulse.evaluation.validation import nested_cv  # noqa: E402
from rakshpulse.features.ecg_aux import PAT_FEATURES, pat_features  # noqa: E402


def _rec_pat(rec, starts):
    rows = []
    for t0 in starts:
        a, b = int(t0 * rec.fs), int((t0 + 10) * rec.fs)
        f = pat_features(rec.channels["ecg"][a:b], rec.channels["ir"][a:b], rec.fs)
        rows.append({"record_id": rec.record_id, "t_start_s": t0, **f})
    return rows


def main() -> int:
    g = pd.read_csv(PROCESSED / "guilin_dualwave_features.csv", dtype={"record_id": str})
    starts = g.groupby("record_id")["t_start_s"].apply(list).to_dict()
    res = Parallel(n_jobs=-1)(delayed(_rec_pat)(r, starts[r.record_id]) for r in iter_guilin())
    pat = pd.DataFrame([x for rs in res for x in rs])
    g = g.merge(pat, on=["record_id", "t_start_s"], how="left")
    print("PAT availability:", g["pat_foot_ms"].notna().mean().round(3), flush=True)
    base = gate(g, "ppg_ir_red")
    sets = {
        "ppg_ir_red (PPG only)": FEATURE_SETS["ppg_ir_red"],
        "ecg_pat_only (aux)": list(PAT_FEATURES),
        "ppg_ir_red+ecg_pat (aux)": FEATURE_SETS["ppg_ir_red"] + list(PAT_FEATURES),
    }
    fmap = fold_map(g, 0)
    out = []
    for name, feats in sets.items():
        for model in ("ridge", "xgboost"):
            for target in ("sbp", "dbp"):
                pdf, _ = nested_cv(base, feats, target, model, seed=0, splits=splits_from_map(base, fmap))
                ev = evaluate(pdf, target)["subject"]
                out.append({"feature_set": name, "model": model, "target": target, **ev})
                print(f"[ECG-aux] {name:28s} {model:8s} {target}: MAE {ev['mae']:.2f} r {ev['pearson_r']:.2f}", flush=True)
    corr = base.groupby("subject_id")[["pat_foot_ms", "sbp", "dbp"]].median().corr().loc["pat_foot_ms"].to_dict()
    (ROOT / "results" / "ecg_aux_experiments.json").write_text(
        json.dumps({"runs": out, "subject_level_corr_pat_foot": corr,
                    "pat_availability": float(g["pat_foot_ms"].notna().mean())}, indent=2, default=float)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
