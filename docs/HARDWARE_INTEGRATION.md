# Hardware Integration: Python Interfaces for Device Integration

To connect a MAX30102 and record data, follow [HARDWARE_HANDOFF.md](HARDWARE_HANDOFF.md) or the illustrated
[Hardware Integration Manual (PDF)](RAKSHPULSE_HARDWARE_INTEGRATION_MANUAL.pdf). This page documents the Python
interfaces a device integration builds on.

## Recommended entry point: `rakshpulse.inference.api`

```python
from rakshpulse.inference.api import BPEstimator, StreamingBPEstimator

est = BPEstimator(sensor="max30102")              # default bundle: models/rakshpulse_ear_bp.joblib
r = est.estimate(ir=ir_counts, fs=100.0, red=red_counts, metadata={"session": "D1"})
print(r.to_dict())
```

| Field | Meaning |
|---|---|
| `status` | `ok`, `poor_signal` (too few windows passed the quality gate), `invalid_input` (bad samples or rate) or `incompatible_model` |
| `sbp`, `dbp` | Median over accepted 10 s windows (calibrated if a cuff calibration is active); `None` unless `status == "ok"` |
| `population_sbp`, `population_dbp` | The uncalibrated model output |
| `heart_rate_bpm` | Median heart rate of the accepted windows |
| `input_validation`, `calibration_status` | `passed` or the reason it failed; `uncalibrated`, `calibrated (...)` or `expired` |
| `sensor_status` | Source, connection and sample counts supplied by the caller (the CLI fills it from the serial port) |
| `sbp_interval_90`, `dbp_interval_90` | 90 % interval, only if the model's uncertainty is valid for this sensor (not `max30102`) |
| `signal_quality` | Window counts per quality class, median quality score, most frequent rejection reasons |
| `inference_mode` | `validated` / `experimental` / `demonstration` from the model contract (`demonstration` for MAX30102) |
| `model_id`, `model_version`, `warnings`, `message`, `latency_ms`, `metadata` | Provenance and diagnostics |

- **Input:** raw MAX30102 ADC counts (IR required, red optional) and the effective sampling rate (sample rate ÷
  averaging), at least 10 s. Other signal units: pass `pipeline_overrides=ANALOG_INPUT` to disable ADC rail checks.
- **Validation:** non-numeric, multi-dimensional, infinite or mostly-missing samples, a non-positive rate, unequal
  channel lengths and recordings shorter than one window return `invalid_input` with a message; nothing raises.
- **Processing:** the production pipeline used for recorded sessions (`preprocessing.pipeline`), then
  `inference.engine.predict_windows` (model contract, sensor and sampling-rate checks). Nothing is reimplemented.
- **Calibration:** `est.calibrate(sbp, dbp, ir, fs)` stores one cuff reading with PPG recorded at the same time;
  later estimates add the bundle's Δ-model prediction (BP change from the feature change) to the cuff values. Readings
  older than `max_calibration_age_s` (30 min) are ignored (`expired`). On earlobe test data this beat the cuff value
  alone only for DBP within a session ([report](../reports/final_model_evaluation.md)).
- **CLI and dashboard:** `rakshpulse estimate --source serial --port COM5 --fs 100 --plot status.png` prints the same
  fields and renders a status dashboard; its banner always states the input source and the inference mode.
- **Streaming:** `StreamingBPEstimator(est, fs, buffer_s=30)` keeps a ring buffer of the last `buffer_s` seconds;
  `push()` any chunk size (e.g. one FIFO read), `estimate()` when needed.
- **Cost:** about 0.1 s of desktop CPU per 30 s of data.
- **Offline vs hardware validation:** the default model was tested on held-out subjects of the Auricle earlobe
  dataset ([reports/final_model_evaluation.md](../reports/final_model_evaluation.md)) and is no better than the
  population mean there. It has never seen MAX30102 data, so on MAX30102 input it runs in `demonstration` mode.

## Status

| Item | Status |
|---|---|
| Hardware inference API (`rakshpulse.inference.api`) | **Implemented and tested** with simulated MAX30102 signals and recorded earlobe PPG |
| Python inference API (`BPPredictor`, 1.0) | **Implemented and tested** with synthetic and real dataset signals |
| Accepts MAX30102 raw red/IR FIFO counts | **Implemented.** `raw_counts=True` handles DC offset and inverted pulses |
| Tested on real MAX30102 behind-ear data | **Not done.** No such data exists yet |
| Serial acquisition, `check-serial`, `record`, `analyze` (5.1) | **Implemented and tested** with simulated and loop-back serial ports |
| Reference streaming firmware (`hardware/firmware/rakshpulse_max30102`) | **Written, not yet compiled or run on a board** |
| On-device (ESP32) inference | **Not started.** float32 C-header export exists for linear models |
| Bluetooth / mobile app protocol | **Not specified** by the project documents |
| Clinical validation | **Not started** |

## Interface

```python
from rakshpulse.inference.predictor import BPPredictor

predictor = BPPredictor.from_file("models/rakshpulse_bp_v0.joblib")   # created locally by scripts/train_final.py
result = predictor.predict(
    red_signal=red_samples,      # 1-D array, raw MAX30102 counts (≥ 10 s; 30 s recommended)
    ir_signal=ir_samples,        # same length as red
    sampling_rate=100,           # samples per second actually configured on the sensor
)
if result.accepted:
    print(result.systolic_bp, result.diastolic_bp)   # mmHg, research estimate only
    print(result.uncertainty)                        # validation MAE/SD + window spread
else:
    print("Please re-measure:", result.rejection_reasons)
print(result.signal_quality)    # windows accepted/total, per-channel rejection counts
```

Errors and rejections:
- `InputError` (exception): missing or invalid `sampling_rate`, non-numeric or non-1-D
  arrays, red/IR length mismatch, or demographics missing for a model that needs them.
- `accepted=False` (no exception): a required channel is absent, the recording is shorter
  than 10 s, or fewer than 50 % of 10 s windows pass the signal-quality gate. **No BP number is
  returned in that case**, so the device should prompt the user to re-measure.

## Recommended sensor configuration (to be confirmed by the hardware team)

| MAX30102 register setting | Recommendation | Reason |
|---|---|---|
| Mode | SpO₂ mode (red + IR) | The model uses both channels |
| Sample rate | 100 or 200 sps | The pipeline resamples to 100 Hz. Higher rates help nothing below 10 Hz |
| Pulse width / ADC range | 411 µs / 18-bit if the rate allows; avoid saturation | The SQI rejects flat-lined (clipped) data |
| Sample averaging | 1–4 | Averaging lowers the effective rate; log the true output rate |
| LED current | Tune per placement so DC stays at roughly 30–70 % of full scale | Behind-ear AC amplitude is expected to be 2–3× smaller than at the finger |
| Recording length per reading | 30 s | Gives 5 overlapping windows, so the median is robust to a single bad window |
| Logging | Raw FIFO counts + LED settings + die temperature | Needed for later re-processing and domain adaptation |

## Path to on-device inference (when justified by data)

1. Port preprocessing to C: biquad cascades for the band-pass (forward-backward needs a
   buffered window, which is fine for a 10 s spot check), integer peak detection, and beat
   features.
2. Ridge: export coefficients plus scaler mean/scale to a C header. No ML runtime is needed.
3. CNN (only if a later model justifies it): `torch.onnx.export` → int8 quantisation →
   TFLite Micro or ESP-DL on ESP32-S3. About 16k parameters fit in internal SRAM.
4. Verify bit-exactness against the Python pipeline on recorded MAX30102 files before trusting
   any on-device number.

## Personalised interface (Auricle-trained, ear PPG)

`RealtimeEstimator.estimate(ear_ppg, sampling_rate, finger_ppg=None, timestamp=None, engine=None, baseline=None)`
returns the population estimate, the personalised estimate (if a calibrated `PersonalizationEngine` is given),
signal-quality counts per site, calibration status, deviation from the user's own baseline, and a recalibration
recommendation with reasons. It returns no BP when ear SQI fails in more than half of the 10 s windows. The device
side is responsible for collecting trusted reference readings (validated cuff) and passing them to
`engine.update(...)`. The engine never treats PPG alone as evidence that a calibration is valid. The Auricle-trained
bundle comes from earlobe Nonin PPG, not behind-ear MAX30102 data. Retraining on device data is required.



## Streaming interface (RakshPulse 3.0, `src/rakshpulse/inference/streaming.py`)

`StreamingEstimator(bundle, fs, engine=None, baseline=None, hop_s=5, buffer_s=30, aggregate_s=30, lost_after=3,
log_path=None)` consumes samples in chunks of any size (e.g. MAX30102 FIFO reads) with the timestamp of the last
sample: `events = s.push(chunk, timestamp=t)`.

- A ring buffer holds the last `buffer_s` seconds. Every `hop_s` seconds of new data, the 10 s window ending at
  the newest sample passes the same SQI gate (plus a flat-signal check) and feature pipeline as training.
- Each evaluated window yields one `StreamEvent` with: per-window estimate; the median over the last `aggregate_s`
  (personalised when a calibrated engine is attached); `status` (`warming_up`, `ok`, `poor_signal`, or
  `signal_lost` after `lost_after` consecutive rejections); rejection reasons; recent acceptance fraction;
  recalibration flag; baseline deviation; and feature/model latency.
- Non-finite input, wrong sampling rates and decreasing timestamps raise `InputError`. Events can be appended to a
  JSON-lines log for auditing.
- After signal loss the last good aggregate is still reported, but the status marks it stale. Consumers must not
  display it as current.

Measured cost on a desktop CPU is in RESEARCH_FINDINGS.md V6 (`scripts/benchmark_inference.py`: startup, p50/p95
feature and model latency at 100 and 1000 Hz, throughput, real-time factor, CPU utilisation, peak Python memory).
These are **not** microcontroller numbers.

## RakshPulse 4.0: hardware-readiness changes (desktop-verified only)

- **Sensor interface** (`src/rakshpulse/inference/sensor.py`): a driver-free `PPGSource` protocol (`fs` plus
  `chunks()` yielding `(samples, timestamp_of_last_sample)`), an `ArraySource` for replaying recordings in
  FIFO-sized bursts, and `run_stream(source, estimator)`. A MAX30102 integration implements `PPGSource` outside
  the inference core. The module docstring has notes on IR channel, sample rate, FIFO draining, DC and inverted
  pulse.
- **Ring buffer**: `StreamingEstimator` now writes into an O(chunk) circular buffer instead of shifting the whole
  buffer on every push (`np.roll`). The window is copied only when it is evaluated. Results are identical
  (test `test_ring_buffer_matches_a_shifting_buffer`; the 3.0 streaming tests are unchanged and pass).
- **Faster features**: skewness and kurtosis now use NumPy (identical to scipy to within 3.6e-15), because
  scipy's wrapper dominated the per-beat cost.
- **In-stream quality**: `StreamingEstimator(..., grade_quality=True)` adds `quality_class` / `quality_score`
  to every event. It costs a few ms per window on a desktop CPU (see RESEARCH_FINDINGS W6).
- **Export path**: `results/deployment_v4/auricle_population_ear_ridge.{json,h}`. float32 reproduces scikit-learn
  to within 1e-4 mmHg and float16 to within about 0.1 mmHg. Naive per-tensor int8 weight quantisation causes
  errors of several mmHg and is **not** recommended. Use float32 on an FPU-equipped MCU (ESP32-S3, Cortex-M4F).
- **Not done:** no embedded hardware was available, so there are no MCU latency, memory or power numbers. All
  timings in RESEARCH_FINDINGS W6 are desktop CPU. The feature extractor itself (filtering, beat segmentation)
  has not been ported to C. That port, and bit-exactness checks on recorded MAX30102 files, are the next
  integration steps.

## Connecting a MAX30102

The complete, current procedure is in **[HARDWARE_HANDOFF.md](HARDWARE_HANDOFF.md)**: the serial contract, the
reference firmware (`hardware/firmware/rakshpulse_max30102`), setup on Windows and Linux, `check-serial` /
`record` / `analyze`, data formats, troubleshooting and the handoff checklist.
