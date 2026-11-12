# Hardware Integration (status: proposed interface, not integrated)

## Status

| Item | Status |
|---|---|
| Python inference API (`BPPredictor`) | **Implemented and tested** with synthetic and real dataset signals |
| Accepts MAX30102 raw red/IR FIFO counts | **Implemented.** `raw_counts=True` handles DC offset and inverted pulses |
| Tested on real MAX30102 behind-ear data | **Not done.** No such data exists yet |
| Firmware / ESP32 port | **Not started** |
| Bluetooth / mobile app protocol | **Not specified** by the project documents |
| Clinical validation | **Not started** |

## Interface

```python
from rakshpulse.inference.predictor import BPPredictor

predictor = BPPredictor.from_file("models/rakshpulse_bp_v0.joblib")   # created by scripts/train_final.py
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

## Connecting a MAX30102 (RakshPulse 5.0): instructions for the hardware team

The software side is complete and hardware-independent:
`acquisition/sources.py` → `acquisition/ingest.py` → `datastore/` → `preprocessing/pipeline.py` →
`inference/engine.py`. Nothing in it talks I²C. The microcontroller (ESP32, STM32, Arduino, ...) reads the
MAX30102 and streams text lines over USB-serial. The only contract between firmware and software is described
below. **None of this has been tested on a physical MAX30102**, because no device was available to the software
team.

### 1. Sensor configuration (firmware)

- SpO2 mode (red + IR). Record the settings in `MAX30102Config` (`rakshpulse record --sensor-config '{...}'`
  validates them against the datasheet options: sample rate 50–3200 sps, pulse width 69/118/215/411 µs = 15–18-bit,
  ADC range 2048–16384 nA, averaging 1–32, LED current 0–51 mA).
- Suggested starting point: 100 sps output (`sample_rate_hz=100, sample_averaging=1`, or 400 with averaging 4),
  411 µs (18-bit), ADC range 16384 nA. Set the LED currents so the raw DC sits well inside the ADC range: the
  pipeline flags samples at the rails (`rail_clipping`) as unusable. The software's `--fs` must equal the
  **output** rate (`sample_rate / averaging`). `record` refuses a mismatch.
- Check the datasheet table of allowed sample-rate / pulse-width combinations. `MAX30102Config.validate()` warns
  for combinations it cannot verify.

### 2. Firmware output contract

1. Drain the 32-sample FIFO often enough that it never overflows (at 100 sps, at least every ~300 ms). Each
   sample holds a red and an IR value (3 bytes each in SpO2 mode). Assemble them as the datasheet's FIFO data
   format specifies.
2. Keep a **monotonic sample counter** (32-bit) that increments once per FIFO sample. If the FIFO overflow counter
   reports lost samples, **add them to the counter** before the next sample. The software then logs them as gaps
   instead of silently stretching time. This is the single most important requirement.
3. Emit one ASCII line per sample, newline-terminated, for example `sample_index,red,ir`:
   ```
   10234,95321,118204
   10235,95318,118199
   ```
   Optional extra fields (declare them with `--fields`): `device_time_us` (a device microsecond clock),
   `temperature_c` (die temperature, repeated until the next temperature read), `ignore`. Lines starting with
   `#` are treated as comments (use them for boot banners). Malformed lines are counted and skipped.
4. Use a baud rate with ample headroom (115200 baud is about 10× what 100 sps needs).

### 3. Connect and record

```bash
pip install -e .[serial]
rakshpulse doctor
rakshpulse record --out data/bench --subject BENCH --session T1 --source serial --port COM5 --baud 115200 \
    --fs 100 --fields sample_index,red,ir --duration 120 \
    --sensor-config '{"sample_rate_hz": 100, "pulse_width_us": 411, "adc_range_na": 16384}' \
    --placement behind_ear --contact "bench test"
rakshpulse inspect --dataset data/bench --validate
rakshpulse preprocess --dataset data/bench
```

`record` prints `samples`, `missing_samples`, `gaps` and `drift_ppm` (device-clock error against the host clock).
If the counter wraps at 16 bits, add `--counter-modulus 65536`.

### 4. Acceptance checklist (bench, then on-body)

| Check | Pass criterion |
|---|---|
| Validation | `inspect --validate`: all PASS |
| Missing samples | < 0.1 % on the bench (gaps logged, never interpolated beyond 2 samples) |
| Clock | `drift_ppm` within the crystal tolerance; otherwise fix `--fs` or the firmware timer |
| ADC headroom | no `rail_clipping` reasons at rest |
| Signal quality at rest | ≥ 80 % of windows `usable` (`derived/windows.csv`) |
| Heart rate | `ir_hr` within ± 3 bpm of a reference ECG / pulse oximeter |
| Detachment | removing the sensor produces `missing` (flat-line) windows within one window length |
| Motion | deliberate motion produces `degraded` / `unusable` windows with reasons |
| Streaming | `rakshpulse infer --source serial ...` produces timestamped predictions with latency recorded |

### 5. What is still hardware-dependent

- Measured behind-ear MAX30102 signal quality, ambient-light rejection (the sensor's ambient-light cancellation is
  configured in firmware), contact-pressure effects and die-temperature behaviour. These can only be measured on
  the device.
- Any model trained on behind-ear MAX30102 data. All existing models come from other sensors and sites
  (DOMAIN_ADAPTATION.md).
- Embedded latency, memory and power. The desktop numbers in RESEARCH_FINDINGS are **not** embedded measurements.
  The runtime-free float32 C header (`rakshpulse export`) is the starting point for on-device inference. The
  feature extractor would still need porting.
