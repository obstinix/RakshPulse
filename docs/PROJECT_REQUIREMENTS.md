# RakshPulse — ML Project Requirements

This file lists only what the supplied project documents support. Anything they do not state is
marked **UNKNOWN**. Nobody should treat an UNKNOWN as decided until the hardware or clinical team
confirms it.

## Source documents reviewed

| # | Document | Pages | Kind |
|---|---|---|---|
| D1 | `SevaInnovation_2026_RAKSHPULSE (1).pdf` | 6 | Seva Innovation Challenge 2026 pitch deck |
| D2 | `CIA3_UD_VaishnaviSurve_Fall2025.pdf` | 53 | Secondary and primary design research on conventional (cuff) BP monitors |
| H1 | Hardware team confirmation (relayed in the ML brief) | — | Sensor choice: MAX30102 |
| DS | Analog Devices MAX30102 datasheet (19-7065 Rev 1) | — | Chip specification |

The PDFs are team documents and are **not** committed to this public repository (see `.gitignore`).
Neither one contains a sensor specification, signal recordings, firmware, an existing ML implementation
or a dataset. When this work started, the repository held only a CC0 `LICENSE`.

## 1. Requirements stated in the documents

| ID | Requirement | Source |
|---|---|---|
| R1 | Portable, **cuffless** blood pressure monitor | D1 slide 1 |
| R2 | Primary users are military personnel, especially soldiers in remote border areas, high-altitude regions and active field operations | D1 slide 2 |
| R3 | Uses **PPG (photoplethysmography)** to detect blood-flow patterns | D1 slide 3 |
| R4 | Worn **behind the ear**, positioned near the **posterior auricular artery** | D1 slide 3 ("How it works", Input) |
| R5 | Estimates blood pressure with an algorithm ("BP estimation software") | D1 slides 3–4 |
| R6 | Hands-free, quick, "on the go" readings during field operations and high-stress missions | D1 slides 1, 3 |
| R7 | Readings shown on the device, or sent over IoT/Bluetooth to a connected phone that keeps records | D1 slide 3; slide 4 lists a Bluetooth module and mobile app |
| R8 | Hardware: miniature PPG sensor, microcontroller, Bluetooth module, rechargeable battery, skin-safe silicone casing | D1 slide 4 |
| R9 | Planned clinical testing **against a standard BP monitor** | D1 slide 4 ("clinical testing facilities for validation against a standard BP monitor") |
| R10 | The concept is **not clinically validated**. Behind-ear BP estimation "would require research and clinical validation" | D1 slide 3 (explicit disclaimer) |
| R11 | Current stage: **Idea** (ticked on the stage selector) | D1 slide 4 |
| R12 | 6-month target: build a wearable prototype, develop the BP estimation algorithm, test sensor performance | D1 slide 4 |
| R13 | Pilot of 10 users; prototype cost ₹3,000–5,000 per unit | D1 slide 4 |
| R14 | Design must suit different users, **including women** | D1 slide 5 |
| R15 | Indian regulatory context: CDSCO registration under the Medical Devices Rules 2017, BIS, ISO standards | D2 ("Health standards and guidelines for BP machines in India") |
| R16 | Users need help interpreting readings, plus a "this measurement is reliable / redo measurement" signal | D2 (gap analysis: "Number → meaning gap", trust) |

## 2. Hardware-team confirmation (H1) checked against the datasheet (DS)

The hardware team named the **MAX30102** as the sensor. The figures below are **chip-level** values
from the official Analog Devices datasheet. A breakout board (for example the common "MAX30102 module")
adds its own regulators, level shifters, pull-ups and an optical window. Those are board properties,
and they are **UNKNOWN** until the team names the exact board.

| Property | MAX30102 chip (datasheet) | Breakout board |
|---|---|---|
| LEDs | Red **660 nm** and IR **880 nm** (nominal peaks; the red peak shifts with temperature, see datasheet Table 13) | UNKNOWN |
| Channels | 2 optical channels (Red, IR). **No ECG, no green LED** | — |
| ADC | 18-bit sigma-delta. Effective resolution 15–18 bits depending on LED pulse width (69–411 µs) | — |
| Sample rate | Selectable 50–3200 sps. The maximum usable rate depends on pulse width | Firmware choice: UNKNOWN |
| Interface | I²C | Pull-ups and level shifting: UNKNOWN |
| Ambient light cancellation | Yes (up to 200 µA) | Optical window and housing: UNKNOWN |
| Measurement mode | Reflective (LEDs and photodetector on the same face) | — |

Consequences for the ML design:
- Inputs are **red and IR PPG only**. ECG-based features such as pulse arrival time are **not**
  available on the target device and cannot be mandatory model inputs.
- No suitable dataset has green-wavelength PPG with BP labels at the target site, and the
  sensor has no green LED, so green-light methods are out of scope.
- The pipeline must accept a configurable sampling rate and resample internally.

## 3. Parameters NOT specified anywhere (UNKNOWN)

| Parameter | Status |
|---|---|
| Target BP accuracy (MAE / SD in mmHg) | UNKNOWN. The documents give no target. ISO 81060-2 (mean error ≤ 5 mmHg, SD ≤ 8 mmHg) is the usual reference point, but the team has **not** adopted it |
| Firmware sampling rate | UNKNOWN |
| LED current / pulse width settings | UNKNOWN |
| Measurement duration per reading | UNKNOWN |
| Spot-check or continuous monitoring | UNKNOWN. "Quick, on the go" (R6) suggests spot checks |
| Calibration procedure (per-user cuff calibration or calibration-free) | UNKNOWN |
| Microcontroller model | UNKNOWN. The ML brief mentions ESP32 as a possible target, but no supplied document names one |
| On-device or phone inference | UNKNOWN |
| Operating altitude, temperature and motion conditions to support | UNKNOWN beyond "remote, high-altitude, field" |
| Breakout board model and optical window | UNKNOWN |
| Reference BP device for clinical validation | UNKNOWN ("standard BP monitor") |
| Population for validation (age range, BP range) | UNKNOWN |

## 4. Derived engineering requirements for the ML component

These follow directly from sections 1–3 and are what this repository implements:

1. **Input contract:** red and IR PPG arrays plus a sampling rate. Optional demographics.
   ECG is never required.
2. **Outputs:** SBP and DBP estimates in mmHg, a signal-quality assessment, and an explicit
   **rejection** when quality is too low (supports R16 "redo measurement").
3. **Subject-independent evaluation**, because a field device meets unseen users.
4. **Report results honestly against ISO 81060-2 style statistics** (mean error, SD of error)
   without claiming compliance.
5. **A small model footprint** that a microcontroller-class device could plausibly run. This
   is only assessed once model size has been measured.
6. **Treat behind-ear placement as unverified.** Public-dataset performance is not evidence
   of behind-ear performance (R10).
