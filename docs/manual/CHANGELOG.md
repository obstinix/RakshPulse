# Manual changelog

## Edition 2.0: Complete Hardware and Operator Manual (software 5.2.0)

This edition replaces `docs/RAKSHPULSE_HARDWARE_INTEGRATION_MANUAL.pdf` (edition 1.0, 43 pages, source in
`docs/manual_source/`). It was rebuilt from scratch after a new audit of the firmware, CLI, model and dataset.

**New**
- **Chapter 2, "Why Am I Not Seeing SBP and DBP?"**, answers the hardware team's main question. It covers what
  the MAX30102, the ESP32 and the computer each do, and includes the table "Why Your ESP32 Is Not Showing SYS and
  DIA — What Is Currently Missing?". It also describes what an ESP32 OLED display would need.
- **Chapter 3** explains how to read a PPG graph: the blue curve, the red peak dots and the axes. The graphs use
  real Auricle data processed by RakshPulse, plus a PPG-versus-blood-pressure figure. It also analyses one frame of
  the team's test video (screen crop only).
- **Chapter 4** names the exact training dataset: Auricle, PhysioNet doi:10.13026/swme-vp69, with a Nonin
  transmissive earlobe sensor (exact model not confirmed) and a CNAP reference. It compares that dataset with the
  MAX30102 and answers "Will the model work with MAX30102?" as five separate questions.
- **Chapter 7, "What Will I Actually See?"**, shows 13 outputs, each labelled real dataset, simulated, real output
  or illustrative.
- **Chapter 11** explains accuracy using the metrics read from `reports/*.csv`, including baselines, confidence
  intervals and calibration. It gives a picture of a 10 mmHg error and distinguishes five kinds of "it works".
- **Chapter 14** adds a presentation sequence, a speaking script and honest answers to likely questions.
- Ten graphs from real data and model reports, and 14 curved-arrow vector diagrams. 25 figures in total.
- All terminal blocks are real output captured from the current code (`capture_outputs.py`), each with its source.
- `verify_manual.py` now also checks repository paths, figure and output files, the synthetic-example labels and
  the required chapter titles.

**Changed**
- Installation covers Windows (PowerShell), Linux (bash) and macOS. macOS is marked as not tested.
- Default I²C pins are given only where a source exists: the Arduino-ESP32 "ESP32C3 Dev Module" variant, cited.
  Supply voltage is still never assumed.
- The troubleshooting table classifies each problem as hardware, firmware, software, model or expected behaviour,
  and is followed by a flowchart.
- The bibliography has 14 verified entries, each with a relevance note.

**Fixed in the software while writing the manual**
- `rakshpulse check-serial` overestimated the sampling rate: a correct 100 Hz stream measured about 108 Hz and
  failed. It also did not fail streams with many malformed lines. Both are fixed and covered by a test.
