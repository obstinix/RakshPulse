"""1D-CNN raw-waveform experiments on Guilin (same subject folds as run_experiments.py, seed 0).

Compares IR-only vs red+IR input, reports subject-level metrics, parameter count and CPU
latency. Writes results/cnn_experiments.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_experiments import evaluate, fold_map, gate, splits_from_map  # noqa: E402

from rakshpulse.data.build import PROCESSED  # noqa: E402
from rakshpulse.models.cnn import CNNConfig, CNNRegressor, n_params  # noqa: E402

RES = ROOT / "results"


def main() -> int:
    torch.set_num_threads(8)
    feats = pd.read_csv(PROCESSED / "guilin_dualwave_features.csv")
    waves = np.load(PROCESSED / "guilin_dualwave_windows.npz")["waves"]
    feats["_row"] = np.arange(len(feats))
    out = {"runs": []}
    for name, chans, fs_gate in (("cnn_ir", [0], "ppg_ir"), ("cnn_ir_red", [0, 1], "ppg_ir_red")):
        d = gate(feats, fs_gate)
        X = waves[d["_row"].to_numpy()][:, chans, :]
        y = d[["sbp", "dbp"]].to_numpy(float)
        g = d["subject_id"].to_numpy()
        splits = splits_from_map(d, fold_map(feats, 0))
        pred = np.full_like(y, np.nan)
        t0 = time.time()
        for k, (tr, te) in enumerate(splits):
            m = CNNRegressor(len(chans), CNNConfig(seed=k)).fit(X[tr], y[tr], g[tr])
            pred[te] = m.predict(X[te])
            print(f"{name} fold {k} done ({time.time() - t0:.0f}s)", flush=True)
        row = {"model": name, "params": n_params(m.model), "train_seconds": round(time.time() - t0, 1)}
        xb = torch.zeros(1, len(chans), 1000)
        m.model.eval()
        with torch.no_grad():
            for _ in range(5):
                m.model(xb)
            t1 = time.perf_counter()
            for _ in range(100):
                m.model(xb)
        row["cpu_latency_ms_per_window"] = round((time.perf_counter() - t1) * 10, 3)
        for j, target in enumerate(("sbp", "dbp")):
            pdf = d[["subject_id", "record_id", "t_start_s", target]].copy()
            pdf["pred"] = pred[:, j]
            pdf["fold"] = -1
            pdf.to_csv(RES / "predictions" / f"A_{name}_{target}.csv", index=False)
            ev = evaluate(pdf, target)
            row[target] = {
                "subject": ev["subject"],
                "window_mae": ev["window"]["mae"],
                "bhs_subject": ev["bhs_subject"],
            }
            s = ev["subject"]
            print(
                f"[CNN] {name} {target}: subj MAE {s['mae']:.2f} ME {s['mean_error']:+.2f} SD {s['sd_error']:.2f} "
                f"r {s['pearson_r']:.2f}",
                flush=True,
            )
        out["runs"].append(row)
    (RES / "cnn_experiments.json").write_text(json.dumps(out, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
