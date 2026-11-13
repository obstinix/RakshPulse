# RakshPulse Hardware Handoff (MAX30102 + ESP32/Arduino)

This guide is for the hardware team. Follow it top to bottom. No change to the RakshPulse Python code is needed:
the software side is complete and was tested with simulated sensors, replayed recordings and simulated serial
ports.

> **Status, honestly:** the software has never received data from a physical MAX30102. The firmware in
> `hardware/firmware/` has not been compiled or run by the software team (no board was available). No RakshPulse
> model has been validated on MAX30102 recordings, so every blood-pressure number the software shows for your
> device is labelled **demonstration**: it proves the pipeline runs and is **not a BP measurement**. RakshPulse is not
> a medical device.

## 1. What you build and what the software expects

```
MAX30102 ──I²C──► ESP32 / Arduino ──USB serial──► PC running `rakshpulse` ──► CSV session ──► analysis / inference
```

| Item | Requirement |
|---|---|
| Sensor mode | SpO2 mode: red **and** infrared |
| Serial line per sample | `sample_index,red,ir` (decimal integers, newline-terminated). `infrared` is accepted as a column name too |
| `sample_index` | 32-bit counter, +1 per sample. Advance it by the number of samples lost to FIFO overflow (never reuse or skip silently) |
| Other output | Only lines starting with `#` (comments). Print one banner at start: `# rakshpulse fw=1 fs_hz=100 ...` |
| Effective sampling rate | `fs_hz = SAMPLE_RATE / SAMPLE_AVERAGE` (e.g. 100 sps, averaging 1 → **100 Hz**). Give the same value to the software as `--fs`. It is never assumed |
| Supported rates | 50–1000 Hz effective (the shipped model accepts 50–1000 Hz; 100 Hz recommended) |
| Baud rate | 115200 (default in firmware and software; change both together) |
| Raw values | Unfiltered 18-bit counts (0–262143 at 411 µs pulse width). Do not filter or scale on the board |

**You must confirm these hardware-specific values** (they are deliberately not assumed):
- the I²C SDA/SCL pins of your board (`I2C_SDA_PIN`, `I2C_SCL_PIN` in the sketch; −1 means the board default);
- the supply voltage and pull-ups of your MAX30102 breakout;
- the LED current (`LED_POWER`): tune it so that IR at rest sits roughly in the middle of the range, never at
  262143;
- the mechanical mounting and contact pressure at the measurement site (record them in `--contact`).

## 2. Firmware (`hardware/firmware/rakshpulse_max30102/rakshpulse_max30102.ino`)

1. Install the Arduino IDE (or arduino-cli) and, for ESP32, the "esp32 by Espressif" board package.
2. Library Manager → install **"SparkFun MAX3010x Pulse and Proximity Sensor Library"** (header `MAX30105.h`; it
   supports the MAX30102).
3. Open the sketch. Set `SAMPLE_RATE`, `SAMPLE_AVERAGE`, `LED_POWER` and, if needed, `I2C_SDA_PIN`/`I2C_SCL_PIN`.
   Upload.
4. Open the Serial Monitor at 115200 baud **only to look**, then **close it** (only one program can use the port).
   You should see the banner, then fast-scrolling lines like `1234,95321,118204`. If you see
   `# ERROR: MAX30102 not found`, fix wiring, power or pins.

The sketch reads the FIFO overflow register (0x05) before each FIFO read and advances the counter by any lost
samples. This follows the datasheet; verify it on your board, for example by adding a deliberate delay and
confirming that `rakshpulse check-serial` reports missing samples.

## 3. Software setup

### Windows (PowerShell)
```powershell
git clone https://github.com/obstinix/RakshPulse.git
cd RakshPulse
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -e ".[serial]"
rakshpulse doctor
```
Find the port in Device Manager → Ports (COM & LPT), e.g. `COM5`. Install the USB-serial driver (CP210x or CH340)
if the board does not appear.

### Linux (bash)
```bash
git clone https://github.com/obstinix/RakshPulse.git && cd RakshPulse
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[serial]"
sudo usermod -aG dialout "$USER"      # then log out and in once, for /dev/ttyUSB0 / /dev/ttyACM0 access
rakshpulse doctor
```

Python 3.10–3.12 is supported. `rakshpulse doctor` must end with `all required checks passed`.

## 4. Step by step

```bash
rakshpulse ports                                             # 1. find the board's port
rakshpulse check-serial --port COM5 --fs 100 --seconds 10    # 2. verify the data stream (PASS/FAIL + problems)
rakshpulse record --source serial --port COM5 --fs 100 \
    --out data/rakshpulse_hw --subject P001 --session D1 --duration 300 \
    --placement behind_ear --contact "elastic band, moderate"   # 3. record (Ctrl-C stops safely)
rakshpulse reference --dataset data/rakshpulse_hw --subject P001 --session D1 \
    --time 60 --sbp 121 --dbp 79 --hr 72 --device "Omron M3"    # 4. add each cuff reading (session time in s)
rakshpulse inspect --dataset data/rakshpulse_hw --validate   # 5. integrity check (must say all PASS)
rakshpulse analyze --dataset data/rakshpulse_hw --subject P001 --session D1   # 6. quality + inference + export
rakshpulse inspect --dataset data/rakshpulse_hw --quality    # 7. signal-quality summary
```

- `record` creates the session only after valid samples arrive. It refuses to start if the firmware banner's
  `fs_hz` differs from `--fs`, and it never overwrites an existing session (use a new `--session`).
- Stopping: `--duration`, or Ctrl-C. Both close, hash and register the files. Unplugging the board ends the
  recording after `--timeout` seconds without data; the samples already received are kept.
- Cuff reading time: seconds from the start of the recording, at the moment the cuff shows the result. Take at
  least 3 readings per session, ≥ 60 s apart.
- `analyze` writes `derived/windows.csv`, `derived/segments.csv` and `derived/predictions_<model>.csv`. Add
  `--calibrate N` to use the first N cuff readings as a calibration. The remaining readings are then compared
  independently, and the calibration windows are never scored.
- Live view: `rakshpulse infer --source serial --port COM5 --fs 100 --out live.csv` (Ctrl-C to stop).

**Sending recordings to the software team:** zip the dataset folder (`data/rakshpulse_hw`) and share it privately.
Never commit participant recordings to the public repository. Subject IDs must be pseudonyms (`P001`), never names.

## 5. Data formats

Session folder: `<dataset>/sessions/<subject>/<session>/`

| File | Columns |
|---|---|
| `raw_ppg.csv` | `timestamp` (s from start; `sample_index / fs`, or device time), `subject_id`, `session_id`, `sample_index`, `red`, `infrared` (raw counts) |
| `reference_bp.csv` | `subject_id`, `session_id`, `timestamp` (s), `sbp`, `dbp` (mmHg; SBP 50–260, DBP 25–160, SBP > DBP), `heart_rate`, `measurement_device` |
| `session_metadata.csv` | `subject_id`, `session_id`, `recording_date`, `sensor_type` (`max30102`), `placement`, `sampling_rate`, `contact_conditions`, `notes` (includes the firmware banner), `sensor_config`, `software_version`, `simulated` |
| `gaps.csv` | `timestamp`, `first_missing_index`, `n_missing`, `detected_by` (`counter`, `counter_reset`, …) |
| `derived/windows.csv` | one row per 10 s window: `status` (`usable`/`degraded`/`unusable`/`missing`), `reasons`, `quality_score`, features |
| `derived/predictions_<model>.csv` | per window: `status`, `inference_mode`, `sbp`, `dbp`, `imputed`, matched cuff values, `role`, calibrated values |

Examples (simulated signals, for format reference only):
- `hardware/examples/serial_stream_valid.txt`: what the firmware should print (banner + 500 samples);
- `hardware/examples/serial_stream_malformed.txt`: lines the parser skips and counts (short lines, wrong delimiter,
  text, NaN, extra fields);
- `hardware/examples/example_dataset/`: a complete 90 s session with three cuff readings. Try
  `rakshpulse analyze --dataset hardware/examples/example_dataset --subject SIM001 --session D1` on a copy.

## 6. Exit codes

`0` success · `1` a check failed or no trustworthy result · `2` invalid usage/input · `3` device/serial problem ·
`4` incompatible model.

## 7. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `error: ... needs the optional 'pyserial' package` | `pip install -e ".[serial]"` |
| `cannot open serial port 'COM5'` | Wrong port (`rakshpulse ports`), the port is open in the Arduino Serial Monitor (close it), cable or driver missing, or Linux permissions (`dialout` group) |
| `no serial ports found` | Board not connected/powered, charge-only USB cable, missing CP210x/CH340 driver |
| `no valid samples ... for 5 s` | Wrong baud rate (must match `SERIAL_BAUD`), firmware not running, or sensor not detected (the sketch prints `# ERROR: MAX30102 not found`) |
| Many `malformed_lines` | Baud mismatch (garbled text), debug prints in the firmware, or a different line format (adjust `--fields`/`--delimiter`) |
| `firmware declares fs_hz=X but --fs is Y` | Pass `--fs X`, or change the sketch's `SAMPLE_RATE`/`SAMPLE_AVERAGE` |
| `measured rate ... differs from --fs` | The board cannot keep up (lower the rate, or raise the baud rate) or the averaging was not accounted for |
| `missing_samples` / gaps | FIFO overflow (loop too slow, serial too slow): raise the baud rate, avoid `delay()` in `loop()`. The software logs gaps; windows never span them |
| `a channel is constant` / all windows `missing (no_signal_flatline)` | Sensor not touching skin, LED current 0, wiring |
| Windows `unusable` with `rail_clipping` | LED current too high or ambient light saturating: lower `LED_POWER`, shield the sensor |
| Windows `unusable` with `low_template_correlation` / `irregular_beat_intervals` | Motion or poor contact: keep still, improve mounting |
| `recording too short` | Record at least 30 s of contiguous data (each window needs 10 s without a gap) |
| `timestamps imply about X Hz but ... declared as Y Hz` | The session was recorded with the wrong `--fs`: re-record with the correct value |
| `model not usable for this input` (exit 4) | Unsupported sensor or sampling rate, or missing demographics: check `rakshpulse models --verbose` |
| `requires demographic inputs` | Only for models that need them: add `--demographics age=35,sex=1,height_cm=172,weight_kg=70` |
| `no trustworthy prediction` | No window passed the quality and input checks: fix the signal first (see `inspect --quality`) |
| Model fails to load | Use only bundles from this repository (`rakshpulse models`); joblib files from unknown sources are unsafe |

## 8. Handoff checklist

- [ ] `rakshpulse doctor` → all required checks passed
- [ ] Sketch uploaded; Serial Monitor shows the `# rakshpulse ... fs_hz=...` banner and `index,red,ir` lines (then closed)
- [ ] `rakshpulse check-serial --port <PORT> --fs <fs_hz>` → `PASS` (0 malformed lines, measured rate ≈ fs, no constant channel)
- [ ] 5-minute bench recording with `record`; `inspect --validate` → all PASS; `missing_samples` < 0.1 %
- [ ] Removing the finger/ear mid-recording produces `missing` windows (flat line) in `derived/windows.csv`
- [ ] On-body recording at the target site: ≥ 80 % of resting windows `usable` (`inspect --quality`)
- [ ] ≥ 3 cuff readings entered with `reference`; `analyze` reports them matched to windows
- [ ] `analyze` completes and writes `derived/predictions_<model>.csv` (expect `mode=demonstration`)
- [ ] Zipped dataset folder sent privately to the software team

When this checklist passes, the remaining work belongs to the software/ML side: collecting repeat-day recordings with
cuff references (docs/DATA_COLLECTION_PROTOCOL.md), then training and evaluating a MAX30102-specific model
(`rakshpulse features` → `train --split cross_day`). That is the only route to anything beyond demonstration
mode.
