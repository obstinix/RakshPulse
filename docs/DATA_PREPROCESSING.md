# PPG Preprocessing Pipeline

Code: `src/rakshpulse/preprocessing/`, `src/rakshpulse/features/feature_pipeline.py`.
The **same function** (`extract_features`) runs at training time and at inference time.

## Per-channel processing chain

| Step | Implementation | Parameter | Justification |
|---|---|---|---|
| 1. Input validation | `filtering.validate_sampling_rate`, `artifact_removal.interpolate_nonfinite` | fs ≥ 25 Hz (2.5× the 10 Hz band edge); ≤ 5 % non-finite samples, otherwise reject | Nyquist with margin. More missing data than that is not "repaired" |
| 2. DC removal + linear detrend | `median` subtraction, `scipy.signal.detrend` | — | Raw MAX3010x counts sit around 4×10⁵. Resampling with that offset produced an edge step transient that **dominated the window and flipped the orientation check** (found on Guilin subject 001, fixed) |
| 3. Resample | `resample_poly` (anti-aliased) | **100 Hz** | Native MAX30102 rate option. Foggia is 100 Hz. 5× the band edge. Fiducial timing resolution 10 ms |
| 4. Despike | Hampel, half-window 5, k = 3 | — | Removes isolated ADC/I²C glitches without smoothing the pulse (also used by the Foggia authors) |
| 5. Band-pass | 4th-order Butterworth, zero-phase `sosfiltfilt`, odd padding ≤ 2 s | **0.5–10 Hz** | 0.5 Hz removes respiration/baseline below 30 bpm. 10 Hz keeps the first ≈4 harmonics at resting HR (pulse morphology). Matches the Foggia distribution filter and lies within the PPG-BP hardware filter (0.5–12 Hz). Zero phase keeps timing features unbiased |
| 6. Orientation | sign of the 1st-derivative skewness | — | Optical absorption inverts the raw pulse. PPG upstrokes are steeper than downstrokes |
| 7. Normalise | per-window z-score | — | Uses no dataset statistics, so it cannot leak between subjects. Absolute gain is not comparable across sensors or sites |
| 8. Windows | 10 s, 5 s step; first 10 s of long recordings discarded | — | 8–30 beats per window. PulseDB/Foggia convention. The first window of every Guilin recording showed an LED/AGC settling transient |
| 9. Beats | autocorrelation-adaptive peak detection + intersecting-tangent foot | refractory = 0.6 × beat period | See "detector fixes" below |
| 10. Beat outliers | correlation with median beat < 0.8 → dropped | — | Removes ectopic or artefactual beats before medians are taken |
| 11. SQI gate | `signal_quality.assess` | see below | Reject rather than estimate |

### Signal-quality gate (window must pass all)

| Index | Threshold | Basis |
|---|---|---|
| Beats detected | ≥ 4 | Need a stable template |
| Heart rate | 30–220 bpm | Physiological range |
| RR coefficient of variation | ≤ 0.30 | Missed/extra beats or arrhythmia |
| Mean beat–template correlation | ≥ 0.86 | Orphanidou et al. 2015 |
| Cardiac-band power ratio (f₀ ± 30 % and 2f₀ ± 30 %, of 0.5–10 Hz) | ≥ 0.35 | Spectral concentration at HR (cf. RSQI in the Foggia paper) |
| ADC flatline/saturation (runs ≥ 5 identical raw samples) | ≤ 5 % | Clipping or sensor off skin |
| Skewness (Elgendi 2016) | **reported only** | A hard threshold rejected nearly all visually clean Guilin windows (plateau-shaped pulses in young adults have negative skew) |

Measured pass rates (rest, 10 s windows): Guilin IR 96.6 %, red 93.4 %; Foggia 100 %
(distributed pre-cleaned); post-exercise Guilin IR 95.5 %, red 93.0 %. The post-exercise
recordings were captured at rest after exercise, so they are **not** a motion test.

PPG-BP segments are 2.1 s long and contain 1–2 complete beats, so they fail the 4-beat rule
by construction. For the PPG-BP external test only, the gate is relaxed to ≥ 1 complete beat
and cardiac-power ratio ≥ 0.35. The inference API never relaxes it.

### Detector fixes found by unit tests

Synthetic-signal tests (`tests/test_preprocessing.py`, `test_features.py`) exposed two real
bugs, which were fixed **before** the reported experiments were rerun:
1. Foot = argmin over the whole inter-beat interval wandered inside the flat diastolic floor
   (beat lengths of 76–96 samples at a constant 70 bpm). Replaced by the intersecting-tangent
   foot. Beat lengths are now constant within ±1 sample, and features agree between 100 Hz and
   250 Hz input.
2. A fixed 220-bpm refractory distance let a prominent diastolic wave count as a beat at
   50 bpm. Replaced by an autocorrelation-estimated period (distance = 0.6 × period).

### Auricle-specific quality handling

- **CNAP reliability mask:** a window gets a BP label only if the CNAP mask is 1 for all of its 10 s.
  This excludes the uncalibrated start of every session and the ~5 s auto-calibration gaps that occur
  about every 35 s. Records without the mask (HC_1_1) are excluded.
- **Amplifier rail clipping (added after a defect was found):** the BIOPAC front end saturates at ±10 V.
  In the healthy cohort, 11 % of finger windows and 0.2 % of ear windows contained rail-clipped samples. The
  flatline check above runs only on DC-carrying optical data, so it missed them, and 705 clipped finger
  windows had been accepted. Windows with more than 1 % of samples on the rail are now rejected
  (`build_auricle.apply_rail_gate`, reason `clipped_at_rail`). Clipping flattens the systolic peak and
  diastolic trough, which are exactly the morphology the features describe.
- **Site-specific rejection profile:** at both sites most rejections are low template correlation
  (ear 61 %, finger 56 % of rejected windows), followed by irregular intervals. Ear rejections
  concentrate in exercise (46 % of exercise windows fail template correlation, against 17 % at rest),
  which is consistent with motion coupling at the earlobe clip.

## Features (per channel, prefixed `ir_` / `red_`)

- **Morphology** (median over accepted beats, amplitude-normalised beat): beat, systolic and
  diastolic durations, systolic ratio, pulse widths at 10/25/50/75/90 % with diastolic/systolic
  width ratios, systolic and diastolic areas and their ratio, maximum upslope and its timing,
  beat skewness and kurtosis, and second-derivative (APG) b/a, c/a, d/a, e/a, ageing index
  (b−c−d−e)/a and b-wave timing.
- **Time domain:** HR, SDNN, RMSSD, window skewness and kurtosis.
- **Frequency domain:** f₀, 2nd and 3rd harmonic ratios, spectral entropy, band powers
  0.5–2/2–5/5–10 Hz.
- **Cross-channel (red+IR only):** red–IR waveform correlation, and the ratio of ratios
  (AC/DC red ÷ AC/DC IR), which **requires raw DC** and is NaN for pre-filtered datasets.

A missing channel produces NaN features. It is **never** synthesised from the other channel.

## Not implemented (and why)

- **Adaptive motion-artefact cancellation:** no dataset in use has an accelerometer, so it
  could not be validated. The MAX30102 has no IMU either. If the hardware adds one, NLMS/RLS
  cancellation becomes testable.
- **Dicrotic-notch-based indices** (reflection index, stiffness index): notch detection was
  unreliable on young-adult pulses with merged systolic/diastolic waves. The width, area and APG
  features capture similar information more robustly.
