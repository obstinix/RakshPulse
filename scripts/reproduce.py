"""Run the full RakshPulse ML pipeline end to end, in order, stopping at the first failure.

    python scripts/reproduce.py                 # everything
    python scripts/reproduce.py --from cross_day  # resume from a stage
    python scripts/reproduce.py --list

Stages are idempotent: verified downloads are skipped and cached per-record windows are reused.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
STAGES: list[tuple[str, list[str]]] = [
    ("download_small", [PY, "-m", "rakshpulse.data.download", "guilin_dualwave", "foggia_multiwave", "ppg_bp"]),
    ("extract", [PY, "scripts/extract_datasets.py"]),
    ("validate", [PY, "-m", "rakshpulse.data.dataset_validator"]),
    (
        "build",
        [PY, "-m", "rakshpulse.data.build", "guilin_dualwave", "foggia_multiwave", "ppg_bp", "guilin_unlabelled"],
    ),
    ("metadata", [PY, "-m", "rakshpulse.data.metadata"]),
    ("guilin_experiments", [PY, "scripts/run_experiments.py"]),
    ("cnn", [PY, "scripts/run_cnn.py"]),
    ("ecg_aux", [PY, "scripts/run_ecg_aux.py"]),
    ("paired", [PY, "scripts/compare_models.py"]),
    ("figures_guilin", [PY, "scripts/visualize_ppg.py"]),
    ("final_model", [PY, "scripts/train_final.py", "--feature-set", "ppg_ir_red+demo", "--model", "ridge"]),
    ("auricle_download", [PY, "-m", "rakshpulse.data.physionet", "auricle", "--via-zip", "--jobs", "3"]),
    ("auricle_verify", [PY, "-m", "rakshpulse.data.physionet", "auricle", "--verify"]),
    ("auricle_build", [PY, "-m", "rakshpulse.data.build_auricle"]),
    ("auricle_session1", [PY, "scripts/run_auricle.py"]),
    ("cross_day", [PY, "scripts/run_cross_day.py"]),
    (
        "cross_day_no_rail_gate",
        [PY, "scripts/run_cross_day.py", "--models", "ridge", "--no-rail-gate", "--tag", "_no_rail_gate"],
    ),
    ("rail_gate_comparison", [PY, "scripts/compare_rail_gate.py"]),
    ("audit", [PY, "scripts/audit_cross_day.py"]),
    ("figures_auricle", [PY, "scripts/visualize_auricle.py"]),
    ("figures_cross_day", [PY, "scripts/make_cross_day_figures.py"]),
    ("dataset_build", [PY, "-m", "rakshpulse.data", "build"]),
    ("dataset_validate", [PY, "-m", "rakshpulse.data", "validate"]),
    ("personalization", [PY, "scripts/run_personalization.py"]),
    ("findings_tables", [PY, "scripts/write_findings.py"]),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default=STAGES[0][0])
    ap.add_argument("--only")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    names = [n for n, _ in STAGES]
    if a.list:
        print("\n".join(names))
        return 0
    todo = [s for s in STAGES if s[0] == a.only] if a.only else STAGES[names.index(a.start) :]
    for name, cmd in todo:
        t0 = time.time()
        logging.info("stage %s: %s", name, " ".join(cmd[1:]))
        rc = subprocess.call(cmd, cwd=ROOT)
        if rc != 0:
            logging.error("stage %s failed with exit code %d", name, rc)
            return rc
        logging.info("stage %s done in %.0fs", name, time.time() - t0)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
