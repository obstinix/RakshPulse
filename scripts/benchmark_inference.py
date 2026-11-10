"""Measure inference cost on THIS machine (desktop CPU). This is not a microcontroller measurement.

Records startup (bundle load), per-window feature-extraction and model latency at 100 Hz (MAX30102-like rate) and
1000 Hz (Auricle native), streaming throughput, peak Python memory (tracemalloc) and CPU utilisation
(process time / wall time). Timing and memory are measured in separate passes, because tracemalloc slows Python
roughly tenfold. Writes results/benchmarks/inference_benchmark.csv and logs a provenance run.
"""

from __future__ import annotations

import platform
import sys
import time
import tracemalloc
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from conftest import synthetic_ppg  # noqa: E402

from rakshpulse.experiments.registry import ExperimentRun  # noqa: E402
from rakshpulse.inference.model_loader import load_bundle  # noqa: E402
from rakshpulse.inference.streaming import StreamingEstimator  # noqa: E402

BUNDLE = ROOT / "models" / "auricle_population_ear_ridge.joblib"
OUT = ROOT / "results" / "benchmarks"


def main() -> int:
    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.perf_counter()
    bundle = load_bundle(BUNDLE)
    startup_ms = (time.perf_counter() - t0) * 1000
    for fs in (100, 1000):
        x = synthetic_ppg(fs, 300, 72, seed=1, raw_counts=False)  # 5 min of signal
        chunk = int(fs)  # 1 s chunks, as a sensor FIFO would deliver
        # pass 1: timing (tracemalloc off; it slows allocation-heavy Python code roughly 10x)
        s = StreamingEstimator(bundle, fs=fs, hop_s=5.0)
        w0, c0 = time.perf_counter(), time.process_time()
        for i in range(0, x.size, chunk):
            s.push(x[i : i + chunk], timestamp=(i + chunk) / fs)
        wall, cpu = time.perf_counter() - w0, time.process_time() - c0
        lat = s.latency_summary()
        # pass 2: peak Python memory only
        s2 = StreamingEstimator(bundle, fs=fs, hop_s=5.0)
        tracemalloc.start()
        for i in range(0, x.size, chunk):
            s2.push(x[i : i + chunk], timestamp=(i + chunk) / fs)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rows.append(
            {
                "fs_hz": fs,
                "signal_s": x.size / fs,
                "windows_evaluated": lat["windows"],
                "startup_bundle_load_ms": round(startup_ms, 2),
                "feature_ms_p50": round(lat["feature_ms_p50"], 2),
                "feature_ms_p95": round(lat["feature_ms_p95"], 2),
                "model_ms_p50": round(lat["model_ms_p50"], 3),
                "model_ms_p95": round(lat["model_ms_p95"], 3),
                "throughput_windows_per_s": round(lat["windows"] / wall, 1),
                "realtime_factor": round((x.size / fs) / wall, 1),
                "cpu_utilisation": round(cpu / wall, 2),
                "peak_python_mem_mb": round(peak / 1e6, 2),
                "bundle_bytes": BUNDLE.stat().st_size,
                "machine": platform.processor() or platform.machine(),
                "python": platform.python_version(),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "inference_benchmark.csv", index=False)
    run = ExperimentRun(
        "inference_benchmark",
        [BUNDLE],
        {"signal": "synthetic two-Gaussian PPG, 72 bpm, 300 s", "chunk": "1 s", "hop_s": 5},
        seed=1,
        t0=t_start,
    )
    run.add_metrics(df)
    run.finish()
    print(df.T.to_string(header=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
