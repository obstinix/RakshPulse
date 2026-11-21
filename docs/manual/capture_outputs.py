"""Capture real RakshPulse command output for the operator manual (docs/manual/outputs/*.txt).

    python docs/manual/capture_outputs.py

Every text block printed in the manual comes from this script, run against the current code. No hardware is
attached on the build machine, so the inputs are labelled:

* the serial checks read the repository's SIMULATED example stream (hardware/examples/serial_stream_valid.txt),
  fed line by line at the real 100 Hz pace through the same `check_serial` code the CLI uses;
* `record`, `reference`, `inspect`, `analyze` and `estimate` run on SIMULATED example data;
* `doctor`, `models` and `--help` describe this installation.

Temporary files go to a scratch folder that is deleted afterwards; paths in the output are relative.
"""

from __future__ import annotations

import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "outputs"
EX = ROOT / "hardware" / "examples"
ENV = {**os.environ, "PYTHONUTF8": "1", "PYTHONWARNINGS": "ignore", "COLUMNS": "110"}


def run(args: list[str], cwd: Path) -> str:
    p = subprocess.run([sys.executable, "-m", "rakshpulse.cli", *args], cwd=cwd, capture_output=True, text=True,
                       encoding="utf-8", env=ENV)
    return (p.stdout + p.stderr).rstrip() + f"\n[exit code {p.returncode}]"


def save(name: str, cmd: str, text: str, note: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.txt").write_text(f"# {note}\n> {cmd}\n{text}\n", encoding="utf-8")
    print("wrote", f"outputs/{name}.txt")


class PacedStream:
    """Yields the lines of a text file at a fixed rate, like a board printing at `fs` samples per second."""

    def __init__(self, path: Path, fs: float, extra: list[str] | None = None):
        self.lines = (extra or []) + path.read_text(encoding="utf-8").splitlines(keepends=True)
        self.dt = 1.0 / fs

    def __iter__(self):
        t0 = time.monotonic()
        for i, ln in enumerate(self.lines):
            wait = t0 + i * self.dt - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            yield ln


def check_serial_text(stream, fs: float, seconds: float) -> str:
    """Same report and verdict lines as `rakshpulse check-serial` (cli.cmd_check_serial)."""
    sys.path.insert(0, str(ROOT / "src"))
    from rakshpulse.acquisition.sources import TextStreamSource, check_serial

    r = check_serial(TextStreamSource(stream, fs), seconds=seconds)
    lines = [f"{k:>18}: {v}" for k, v in r.items() if k != "problems"]
    if r["ok"]:
        lines.append("serial check: PASS (data matches the RakshPulse contract)")
    else:
        lines += [f"  problem: {p}" for p in r["problems"]]
        lines.append("serial check: FAIL (see docs/HARDWARE_HANDOFF.md, Troubleshooting)")
    return "\n".join(lines)


def head(path: Path, n: int, cols: int | None = None) -> str:
    rows = path.read_text(encoding="utf-8").splitlines()[:n]
    if cols:
        rows = [",".join(r.split(",")[:cols]) + (",…" if r.count(",") >= cols else "") for r in rows]
    return "\n".join(rows)


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    tmp = Path(tempfile.mkdtemp(prefix="rakshpulse_manual_"))
    try:
        save("help", "rakshpulse -h", run(["-h"], tmp), "real output: list of commands")
        save("doctor", "rakshpulse doctor", run(["doctor"], tmp), "real output on the build PC (versions will differ)")
        ports = run(["ports"], tmp)
        ports = re.sub(r"(BTHENUM|USB)\\\S+|(BTHENUM|USB) VID\S*", "<hardware id>", ports)
        save("ports", "rakshpulse ports", ports,
             "real output on the build PC: only Bluetooth ports exist there, which are NOT a RakshPulse board")
        save("models", "rakshpulse models --sensor max30102 --fs 100",
             run(["models", "--sensor", "max30102", "--fs", "100"], tmp), "real output")
        save("stream", "(Arduino Serial Monitor at 115200 baud)", head(EX / "serial_stream_valid.txt", 8),
             "SIMULATED example of the firmware output (hardware/examples/serial_stream_valid.txt)")
        save("check_serial_pass", "rakshpulse check-serial --port COM5 --fs 100 --seconds 4",
             check_serial_text(PacedStream(EX / "serial_stream_valid.txt", 100.0), 100.0, 4.0),
             "real check_serial code; input is the SIMULATED example stream played at 100 lines per second")
        save("check_serial_fail_fs", "rakshpulse check-serial --port COM5 --fs 50 --seconds 4",
             check_serial_text(PacedStream(EX / "serial_stream_valid.txt", 100.0), 50.0, 4.0),
             "real check_serial code; wrong --fs (the firmware banner says 100)")
        save("check_serial_fail_malformed", "rakshpulse check-serial --port COM5 --fs 100 --seconds 4",
             check_serial_text(io.StringIO((EX / "serial_stream_malformed.txt").read_text(encoding="utf-8")),
                               100.0, 4.0),
             "real check_serial code; input is the example stream with garbled lines")

        raw = EX / "example_dataset" / "sessions" / "SIM001" / "D1" / "raw_ppg.csv"
        shutil.copy2(raw, tmp / "raw_ppg.csv")
        ds = "data/rakshpulse_hw"
        rec = ["record", "--source", "replay", "--input", "raw_ppg.csv", "--fs", "100", "--sensor", "simulator",
               "--out", ds, "--subject", "P001", "--session", "D1", "--placement", "behind_ear",
               "--contact", "elastic band, moderate"]
        save("record", "rakshpulse " + " ".join(rec).replace("elastic band, moderate", '"elastic band, moderate"'),
             run(rec, tmp), "real output; replays SIMULATED example samples instead of a live board")
        refs = [["30", "121", "79", "72"], ["60", "118", "77", "70"], ["85", "119", "78", "71"]]
        texts = []
        for t, s, d, h in refs:
            args = ["reference", "--dataset", ds, "--subject", "P001", "--session", "D1", "--time", t, "--sbp", s,
                    "--dbp", d, "--hr", h, "--device", "Omron M3"]
            texts.append("> rakshpulse " + " ".join(args).replace("Omron M3", '"Omron M3"') + "\n" + run(args, tmp))
        (OUT / "reference.txt").write_text("# real output; ILLUSTRATIVE cuff values typed by hand for this example\n"
                                           + "\n".join(texts) + "\n", encoding="utf-8")
        print("wrote outputs/reference.txt")
        save("inspect_validate", f"rakshpulse inspect --dataset {ds} --validate",
             run(["inspect", "--dataset", ds, "--validate"], tmp), "real output on the SIMULATED example session")
        save("analyze", f"rakshpulse analyze --dataset {ds} --subject P001 --session D1",
             run(["analyze", "--dataset", ds, "--subject", "P001", "--session", "D1"], tmp),
             "real output on the SIMULATED example session (demonstration mode)")
        save("inspect_quality", f"rakshpulse inspect --dataset {ds} --quality",
             run(["inspect", "--dataset", ds, "--quality"], tmp), "real output on the SIMULATED example session")
        save("estimate", "rakshpulse estimate --source sim --seconds 30",
             run(["estimate", "--source", "sim", "--seconds", "30"], tmp),
             "real output; SIMULATED sensor input, demonstration mode")
        save("estimate_poor", "rakshpulse estimate --source replay --input flat.csv --fs 100 --sensor max30102",
             _poor_signal(tmp), "real output; a SIMULATED flat signal (sensor off the skin)")

        sess = tmp / ds / "sessions" / "P001" / "D1"
        for name, f, n, cols in (("csv_raw", "raw_ppg.csv", 6, None), ("csv_gaps", "gaps.csv", 4, None),
                                 ("csv_reference", "reference_bp.csv", 5, None),
                                 ("csv_metadata", "session_metadata.csv", 2, None)):
            save(name, f"(first lines of {f})", head(sess / f, n, cols), "real file written by the commands above")
        import pandas as pd

        w = pd.read_csv(sess / "derived" / "windows.csv")
        cols = [c for c in ("window_id", "t_start_s", "status", "quality_score", "reasons", "ir_hr", "ir_sys_ratio")
                if c in w]
        save("csv_windows", "(derived/windows.csv, selected columns)", w[cols].head(6).to_string(index=False),
             "real file written by analyze")
        p = pd.read_csv(next((sess / "derived").glob("predictions_*.csv")))
        pc = [c for c in ("window_id", "status", "inference_mode", "sbp", "dbp", "ref_sbp", "ref_dbp", "role") if c in p]
        save("csv_predictions", "(derived/predictions_rakshpulse-ear-bp.csv, selected columns)",
             p[pc].round(1).head(6).to_string(index=False), "real file written by analyze (demonstration values)")
        tree = []
        for path in sorted((tmp / ds).rglob("*")):
            rel = path.relative_to(tmp)
            tree.append("    " * (len(rel.parts) - 2) + rel.parts[-1] + ("/" if path.is_dir() else ""))
        save("tree", f"(contents of {ds})", ds + "/\n" + "\n".join(tree), "real folder created by the commands above")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


def _poor_signal(tmp: Path) -> str:
    import numpy as np
    import pandas as pd

    n = 3000
    rng = np.random.default_rng(0)
    pd.DataFrame({"timestamp": np.arange(n) / 100, "subject_id": "X", "session_id": "X", "sample_index": np.arange(n),
                  "red": 90000 + rng.normal(0, 1, n), "infrared": 120000 + rng.normal(0, 1, n)}).to_csv(
        tmp / "flat.csv", index=False)
    return run(["estimate", "--source", "replay", "--input", "flat.csv", "--fs", "100", "--sensor", "max30102"], tmp)


if __name__ == "__main__":
    sys.exit(main())
