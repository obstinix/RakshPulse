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
