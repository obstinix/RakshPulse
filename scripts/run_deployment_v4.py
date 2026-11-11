"""RakshPulse 4.0: streaming cost after optimisation, and export / quantisation feasibility (desktop only).

* Streaming benchmark, using the **same protocol** as the 3.0 ``benchmark_inference.py``: synthetic PPG, 300 s,
  1 s chunks, 5 s hop, timing pass without tracemalloc and a separate memory pass. It is run with and without
  4.0 quality grading. The 3.0 numbers stay in ``results/benchmarks/inference_benchmark.csv`` and are compared,
  not overwritten.
* Buffer micro-benchmark: 3.0 ``np.roll`` shifting vs the 4.0 ring buffer, per push, at 100 and 1000 Hz.
* Export: the ear population bundle folded into plain coefficients (``inference/export.py``). Predictions on
  all SQI-accepted Auricle ear windows are compared with scikit-learn in float64 / float32 / float16 /
  int8-quantised weights. The C header is written next to the JSON.

These are desktop-CPU measurements. **No embedded hardware was available; no microcontroller numbers are
reported or implied.**

Outputs: results/deployment_v4/{streaming_benchmark,buffer_benchmark,export_equivalence}.csv,
results/deployment_v4/auricle_population_ear_ridge.{json,h} and a provenance run.
"""

from __future__ import annotations

import json
import platform
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from conftest import synthetic_ppg  # noqa: E402

from rakshpulse.evaluation.auricle_study import add_normalised_shape  # noqa: E402
from rakshpulse.experiments.registry import ExperimentRun  # noqa: E402
from rakshpulse.inference.export import export_linear, predict_exported, to_c_header  # noqa: E402
from rakshpulse.inference.model_loader import load_bundle  # noqa: E402
from rakshpulse.inference.streaming import RingBuffer, StreamingEstimator  # noqa: E402

BUNDLE = ROOT / "models" / "auricle_population_ear_ridge.joblib"
OUT = ROOT / "results" / "deployment_v4"
AUR = ROOT / "data" / "auricle"


def stream_bench(bundle, fs: int, grade: bool) -> dict:
    x = synthetic_ppg(fs, 300, 72, seed=1, raw_counts=False)
    chunk = int(fs)
    s = StreamingEstimator(bundle, fs=fs, hop_s=5.0, grade_quality=grade)
    w0, c0 = time.perf_counter(), time.process_time()
    for i in range(0, x.size, chunk):
        s.push(x[i : i + chunk], timestamp=(i + chunk) / fs)
    wall, cpu = time.perf_counter() - w0, time.process_time() - c0
    lat = s.latency_summary()
    s2 = StreamingEstimator(bundle, fs=fs, hop_s=5.0, grade_quality=grade)
    tracemalloc.start()
    for i in range(0, x.size, chunk):
        s2.push(x[i : i + chunk], timestamp=(i + chunk) / fs)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "fs_hz": fs,
        "quality_grading": grade,
        "windows_evaluated": lat["windows"],
        "feature_ms_p50": round(lat["feature_ms_p50"], 2),
        "feature_ms_p95": round(lat["feature_ms_p95"], 2),
        "model_ms_p50": round(lat["model_ms_p50"], 3),
        "model_ms_p95": round(lat["model_ms_p95"], 3),
        "throughput_windows_per_s": round(lat["windows"] / wall, 1),
        "realtime_factor": round(300 / wall, 1),
        "cpu_utilisation": round(cpu / wall, 2),
        "peak_python_mem_mb": round(peak / 1e6, 2),
    }


def buffer_bench(fs: int, n_push: int = 3000) -> list[dict]:
    size, chunk = int(30 * fs), int(fs * 0.32)  # 30 s buffer, 32-sample-FIFO-like bursts at 100 Hz
    x = np.random.default_rng(0).standard_normal(chunk)
    buf = np.zeros(size)
    t = time.perf_counter()
    for _ in range(n_push):
        buf = np.roll(buf, -chunk)
        buf[-chunk:] = x
    roll_us = (time.perf_counter() - t) / n_push * 1e6
    rb = RingBuffer(size)
    t = time.perf_counter()
    for _ in range(n_push):
        rb.write(x)
    ring_us = (time.perf_counter() - t) / n_push * 1e6
    same = np.allclose(buf[-int(10 * fs) :], rb.last(int(10 * fs)))
    return [
        {
            "fs_hz": fs,
            "buffer_samples": size,
            "chunk_samples": chunk,
            "np_roll_us_per_push": round(roll_us, 2),
            "ring_buffer_us_per_push": round(ring_us, 2),
            "identical_last_10s": bool(same),
        }
    ]


def export_check(bundle) -> tuple[list[dict], dict]:
    w = pd.read_csv(AUR / "windows.csv", low_memory=False)
    f = pd.read_csv(AUR / "features.csv").drop(columns=["subject_id", "record_id", "session"])
    d = w[w.ear_sqi_ok].merge(f, on="window_id")
    add_normalised_shape(d, "ear")
    X = d.reindex(columns=bundle.features).to_numpy(float)
    params = export_linear(bundle)
    rows = []
    for target, pipe in (("sbp", bundle.sbp), ("dbp", bundle.dbp)):
        ref = pipe.predict(X)
        for prec in ("float64", "float32", "float16", "int8"):
            t = time.perf_counter()
            p = predict_exported(params, X, target, prec)
            us = (time.perf_counter() - t) / len(X) * 1e6
            e = np.abs(p - ref)
            rows.append(
                {
                    "target": target,
                    "precision": prec,
                    "n_windows": len(X),
                    "max_abs_diff_mmHg": float(e.max()),
                    "p99_abs_diff_mmHg": float(np.percentile(e, 99)),
                    "us_per_window_batched": round(us, 4),
                }
            )
    return rows, params


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    run = ExperimentRun(
        "deployment_v4",
        [BUNDLE, AUR / "windows.csv", AUR / "features.csv"],
        {
            "stream": "synthetic two-Gaussian PPG, 72 bpm, 300 s, 1 s chunks, hop 5 s (3.0 protocol)",
            "machine": platform.processor() or platform.machine(),
            "embedded_hardware": "none available",
        },
        seed=1,
        t0=t0,
        model_config={"bundle": BUNDLE.name, "export": "folded affine, JSON + C99 header"},
    )
    bundle = load_bundle(BUNDLE)
    with run.phase("evaluate"):
        sb = [stream_bench(bundle, fs, g) for fs in (100, 1000) for g in (False, True)]
        bb = buffer_bench(100) + buffer_bench(1000)
        ex, params = export_check(bundle)
    old = pd.read_csv(ROOT / "results" / "benchmarks" / "inference_benchmark.csv")
    sbd = pd.DataFrame(sb)
    sbd["feature_ms_p50_3.0"] = sbd.fs_hz.map(old.set_index("fs_hz").feature_ms_p50)
    sbd["model_ms_p50_3.0"] = sbd.fs_hz.map(old.set_index("fs_hz").model_ms_p50)
    sbd.to_csv(OUT / "streaming_benchmark.csv", index=False)
    pd.DataFrame(bb).to_csv(OUT / "buffer_benchmark.csv", index=False)
    pd.DataFrame(ex).to_csv(OUT / "export_equivalence.csv", index=False, float_format="%.6g")
    (OUT / "auricle_population_ear_ridge.json").write_text(json.dumps(params, indent=1) + "\n")
    (OUT / "auricle_population_ear_ridge.h").write_text(to_c_header(params))
    run.add_metrics(sbd)
    run.finish()
    print(sbd.to_string(index=False))
    print(pd.DataFrame(bb).to_string(index=False))
    print(pd.DataFrame(ex).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
