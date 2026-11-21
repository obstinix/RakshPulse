# Manual source: RakshPulse Complete Hardware and Operator Manual

This folder builds [`docs/RAKSHPULSE_COMPLETE_HARDWARE_AND_OPERATOR_MANUAL.pdf`](../RAKSHPULSE_COMPLETE_HARDWARE_AND_OPERATOR_MANUAL.pdf).
All text, diagrams and graphs are generated from this folder. Do not edit the PDF by hand.

## Regenerate

Run these from the repository root, with RakshPulse installed (`pip install -e ".[serial]"`):

```bash
pip install reportlab pillow matplotlib pypdf      # documentation tools only, not RakshPulse dependencies
python docs/manual/capture_outputs.py              # 1. real command output      -> docs/manual/outputs/*.txt
python docs/manual/make_figures.py                 # 2. graphs from real data    -> docs/manual/figures/*.png
python docs/manual/build_manual.py                 # 3. the PDF                  -> docs/RAKSHPULSE_COMPLETE_HARDWARE_AND_OPERATOR_MANUAL.pdf
python docs/manual/verify_manual.py                # 4. checks; exit code 0 = PASS
```

On Windows, set `PYTHONUTF8=1` first (PowerShell: `$env:PYTHONUTF8 = "1"`), because the outputs contain `µ`, `²` and `—`.

* **Step 1** runs the real `rakshpulse` commands in a temporary folder. No board is attached on the build machine,
  so the serial checks replay `hardware/examples/*.txt` at the real 100 Hz pace through the same `check_serial`
  code the CLI uses, and `record` / `analyze` / `estimate` use SIMULATED data. The first line of each output file
  states where it came from, and the manual prints that line above the block. Hardware IDs of the build PC's own
  ports are redacted.
* **Step 2** reads the Auricle dataset (`data/auricle/`) and the model reports (`reports/*.csv`). If the raw Auricle
  waveforms are missing, it downloads them from the GitHub release (~350 MB). Every image carries a tag naming its
  source: REAL DATASET, SIMULATED, MODEL RESULTS or HARDWARE TEAM VIDEO.
* **Step 3** draws the vector diagrams (`diagrams.py`) and lays out the chapters (`content.py`). Metric tables are
  read from `reports/*.csv` and the model version from `models/rakshpulse_ear_bp.json` at build time, so they cannot
  drift from the repository.
* **Step 4** checks four things. Every `rakshpulse …` command in the manual must parse with the real CLI parser.
  Every repository path, figure and output it names must exist. The PDF must have 15 chapter bookmarks, at least 20
  figures, the required honesty statements and the synthetic-example labels. And it must not contain forbidden
  claims: invented pin numbers, a validated MAX30102 model, tested firmware, or an ESP32 that shows SBP/DBP.

Fonts: Calibri and Consolas when installed (Windows), otherwise Helvetica and Courier. The PDF then looks slightly
different but contains the same content.

## Files

| File | Purpose |
|---|---|
| `content.py` | The 15 chapters, one function each. Inline markup: `` `code` ``, `**bold**`, `*italic*` |
| `build_manual.py` | Page templates, cover, contents, bookmarks, boxes, tables, terminal blocks, figures |
| `diagrams.py` | Vector diagrams (curved arrows; dashed red = not implemented) |
| `make_figures.py` | Graphs from real data and model reports |
| `capture_outputs.py` | Captures real command output |
| `verify_manual.py` | Consistency checks |
| `theme.py` | Colours and fonts |
| `figures/`, `outputs/` | Generated; committed so the PDF can be rebuilt without the 350 MB waveforms |
| `media/video_plotter_crop.png` | A crop of the laptop screen from the hardware team's test video (no people visible) |

## Rules for editing

* Never present a synthetic number as a measurement. Simulated or invented blood-pressure values need the box
  `ILLUSTRATIVE SYNTHETIC EXAMPLE — NOT A REAL BLOOD-PRESSURE MEASUREMENT`.
* Do not add GPIO numbers or supply voltages unless they come from a cited source.
* The firmware has not been run on a board. Keep it described as untested until it has been.
* See [CHANGELOG.md](CHANGELOG.md) for what changed against the previous manual.
