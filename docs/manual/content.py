"""Chapter content of the RakshPulse Complete Hardware and Operator Manual.

Facts come from the repository (firmware, CLI, model bundle, reports/, docs/DATA_COLLECTION_PROTOCOL.md), from
real command output captured by capture_outputs.py, and from the sources listed in chapter 15. Metrics are read from
reports/*.csv at build time, so the PDF always matches the shipped model. Every command is checked against the CLI
parser by verify_manual.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import theme as T

ROOT = Path(__file__).resolve().parents[2]
REP = ROOT / "reports"
DS = "data/rakshpulse_hw"


def _metrics() -> dict:
    sel = pd.read_csv(REP / "selected_models.csv").set_index("target")
    base = pd.read_csv(REP / "model_comparison.csv").query("model == 'mean'").set_index("target")
    bc = pd.read_csv(REP / "baseline_comparison.csv").set_index("target")
    nest = pd.read_csv(REP / "nested_cv_summary.csv").set_index(["target", "model"])
    cal = pd.read_csv(REP / "calibration_summary.csv")
    split = pd.read_csv(REP / "subject_split.csv")
    data = pd.read_csv(ROOT / "data" / "ml" / "auricle_ear_bp_windows.csv", usecols=["subject_id", "record_id"])
    model = json.loads((ROOT / "models" / "rakshpulse_ear_bp.json").read_text(encoding="utf-8"))
    return {"sel": sel, "base": base, "bc": bc, "nest": nest, "cal": cal, "split": split, "data": data,
            "model": model, "n_train": int((split.split == "train").sum()), "n_test": int((split.split == "test").sum())}


M = _metrics()


def f1(x) -> str:
    return f"{float(x):.1f}"


def f2(x) -> str:
    return f"{float(x):.2f}"


def ci(r) -> str:
    return f"{float(r.mae_difference):+.2f} [{float(r.ci_low):+.2f}, {float(r.ci_high):+.2f}]"


def build(b) -> None:
    for ch in (ch01, ch02, ch03, ch04, ch05, ch06, ch07, ch08, ch09, ch10, ch11, ch12, ch13, ch14, ch15):
        ch(b)


# ================================================================================================ 1
def ch01(b):
    b.chapter(1, "Read This First", "The one-minute version, and how to read this manual")
    b.lead("RakshPulse is research software. It reads light measurements from a MAX30102 sensor through an ESP32, "
           "finds heartbeats in them, and uses a machine-learning model to ESTIMATE systolic and diastolic blood "
           "pressure on a computer. This manual explains every part of that chain from the very beginning, including "
           "what is not finished yet.")
    b.box("warning", "the five facts that answer most questions", [(
        "The **MAX30102 measures light**, not blood pressure. It has no cuff and no pressure sensor.",
        "The **ESP32 firmware only reads the sensor and prints numbers** (`sample_index,red,ir`). It never calculates "
        "systolic (SYS/SBP) or diastolic (DIA/DBP) pressure, and there is no display code.",
        "**SBP and DBP estimates are produced on the computer** by the RakshPulse software, after filtering, "
        "heartbeat detection, quality checks and feature extraction.",
        "The shipped model was **trained on earlobe PPG from a different sensor** (the public Auricle dataset). On "
        "MAX30102 input it runs in **demonstration mode**: its numbers show the software works, they are **not blood-"
        "pressure measurements**.",
        "Even on its own earlobe data, the model is **not significantly better than guessing the population "
        "average**. Real accuracy needs MAX30102 recordings with cuff readings, which you are about to collect.",
    )])
    b.figure("big_picture")
    b.h2("What is verified and what is not")
    b.table(["Item", "Status", "How it was checked"], [
        ["Python software (reading, processing, model, files)", "**Verified**", "190 automated tests, CI on Linux and Windows"],
        ["Serial protocol `index,red,ir` at 115200 baud", "**Verified in software**", "simulated and loop-back streams"],
        ["Reference firmware on a real ESP32", "**Not verified**", "the software team had no board; the hardware video uses Arduino's own plotter"],
        ["Model accuracy on the earlobe dataset", "**Measured**", "held-out people and nested cross-validation (Ch. 11)"],
        ["Model accuracy on MAX30102", "**Unknown**", "no MAX30102 recordings with reference BP exist"],
        ["SYS/DIA on an ESP32 display", "**Not implemented**", "no display code anywhere in the repository"],
        ["Clinical validity", "**None**", "research prototype; not a medical device"],
    ], [3.2, 1.6, 3.6])
    b.h2("How to read this manual")
    b.p("Chapters 2–4 explain the ideas (read them once). Chapters 5–10 are the hands-on procedure. Chapter 11 explains "
        "the accuracy numbers, Chapter 12 fixes problems, Chapter 13 is a one-page command card and Chapter 14 helps "
        "you present the project.")
    b.legend([
        ("success", "REAL DATASET", "Real human data from the public Auricle study (earlobe, Nonin sensor). Never MAX30102."),
        ("important", "SIMULATED", "Made by the RakshPulse simulator or example files. No blood-pressure meaning at all."),
        ("info", "REAL OUTPUT", "Text copied from running the real commands on this software version."),
        ("warning", "ILLUSTRATIVE", "Made-up numbers to explain a format, e.g. example cuff readings."),
    ])
    b.p("Images from the hardware team's own video are labelled HARDWARE TEAM VIDEO. No image in this manual shows a "
        "MAX30102 recording processed by RakshPulse, because none has been shared yet.")
    b.box("info", "COM5 is only an example",
          "`COM5` stands for *your* board's serial port. Windows may use any COM number; on Linux it is usually "
          "`/dev/ttyUSB0` or `/dev/ttyACM0`; on macOS something like `/dev/cu.usbserial-…` or `/dev/cu.usbmodem…`.")


# ================================================================================================ 2
def ch02(b):
    b.chapter(2, "Why Am I Not Seeing SBP and DBP?", "What the sensor, the ESP32 and the software each do", T.RED)
    b.lead("This chapter answers the question the hardware team asked most: the sensor works, numbers appear on the "
           "ESP32's serial monitor, so where are systolic and diastolic blood pressure?")
    b.h2("A. What does the MAX30102 actually measure?")
    b.figure("sensor")
    b.bullets([
        "The module has a **red LED** (about 660 nm) and an **infrared (IR) LED** (about 880 nm). They flash many times "
        "per second (100 times per second with the RakshPulse firmware).",
        "The light enters the skin. Some is absorbed by tissue and blood; some **scatters back** to a **photodetector** "
        "on the same face of the module (a *reflective* sensor).",
        "An 18-bit converter turns the returned light into a whole number between 0 and 262,143. The firmware reports "
        "two numbers per sample: **red** and **IR**.",
        "With every heartbeat a little more blood enters the small arteries, so a little more light is absorbed: the "
        "numbers **rise and fall with the pulse**. Plotted over time, this is a **PPG waveform** "
        "(photoplethysmogram, \"light-volume recording\").",
        "Each wave has a **peak**. The software finds the peaks; the time between two peaks is one heartbeat interval, "
        "and **heart rate = 60 ÷ interval in seconds** (0.8 s → 75 beats per minute).",
    ])
    b.box("warning", "the most important sentence in this manual",
          "**The MAX30102 does not directly measure systolic or diastolic blood pressure.** It contains no cuff and no "
          "pressure sensor. It measures light. Any SBP/DBP value must be *estimated* by software from the shape and "
          "timing of the light wave, and that estimate is only as good as the model and the data it was trained on.")
    b.h2("B. What does the ESP32 actually do?")
    b.p("The reference firmware is `hardware/firmware/rakshpulse_max30102/rakshpulse_max30102.ino`. Reading its code, "
        "it does exactly four things:")
    b.bullets([
        "starts the I²C bus and finds the MAX30102 at address `0x57` (if not found it prints "
        "`# ERROR: MAX30102 not found on I2C …` every 2 seconds and stops);",
        "configures the sensor (100 samples per second, red + IR, 411 µs pulses, LED current `0x24`);",
        "prints one start-up **banner** line beginning with `# rakshpulse fw=1 fs_hz=100 …`;",
        "in a loop, reads every new sample and prints one line **`sample_index,red,ir`**, for example "
        "`1234,95321,118204`. If samples were lost, it advances the counter so the software can see the gap.",
    ], numbered=True)
    b.code("MAX30102  ──I²C──►  ESP32  ──USB serial, 115200 baud──►  Computer  ──►  RakshPulse software",
           "The data path")
    b.p("That is all. There is **no blood-pressure code, no heart-rate code and no display code** in the firmware. "
        "Seeing numbers in the Arduino Serial Monitor or curves in the Serial Plotter proves that the sensor, wiring "
        "and firmware deliver light measurements. It does **not** mean anything has calculated blood pressure.")
    b.h2("C. Where do SBP and DBP come from?")
    b.figure("pipeline")
    b.bullets([
        "**Sensor**: the MAX30102 measures red and IR light 100 times per second and stores samples in its buffer.",
        "**ESP32**: reads the buffer and sends each sample as a text line over USB.",
        "**Serial reader** (`rakshpulse.acquisition`): reads the lines, checks the banner and the sampling rate, "
        "counts garbled lines and lost samples.",
        "**Preprocessing**: removes slow baseline drift and noise (0.5–10 Hz band-pass filter), resamples to 100 Hz.",
        "**Windows**: cuts the signal into 10-second pieces, a new one every 5 seconds.",
        "**Quality check**: labels each window `usable`, `degraded`, `unusable` or `missing`, with reasons "
        "(e.g. motion, flat signal, saturation). Unusable and missing windows are dropped.",
        "**Features**: from each accepted window, 40 numbers describing the pulse: heart rate and pulse shape "
        "(rise time, widths, ratios).",
        "**Model contract check**: is this model allowed for this sensor, sampling rate and set of inputs? It returns "
        "the **mode**: `validated`, `experimental` or `demonstration`.",
        "**ML model**: one model turns the 40 features into an SBP estimate, another into a DBP estimate.",
        "**Results**: the median over accepted windows is reported with heart rate, quality, mode, calibration status "
        "and latency, and saved as CSV, JSON and a PNG dashboard.",
    ], numbered=True)
    b.h2("D. Can the current system display SBP and DBP on the ESP32?")
    b.p("**No.** We checked the firmware, the examples and the whole repository: there is no display library, no "
        "OLED code and no channel that sends results back to the board. Three different things are often confused:")
    b.figure("where")
    b.h2("Why Your ESP32 Is Not Showing SYS and DIA — What Is Currently Missing?")
    b.table(["Component", "Current responsibility (verified in the repository)", "Produces SBP/DBP?"], [
        ["MAX30102", "Optical PPG acquisition: red and IR light, 18-bit counts", "**No**"],
        ["ESP32 firmware `rakshpulse_max30102.ino`", "Configure the sensor, read its buffer, print the banner and `index,red,ir` lines at 115200 baud",
         "**No** (not implemented)"],
        ["Computer software `rakshpulse`", "Serial reading, preprocessing, quality check, features, model inference, CSV/JSON/PNG output",
         "**Yes, estimates**; labelled `demonstration` for MAX30102"],
        ["ML model `rakshpulse_ear_bp`", "Maps 40 PPG features to SBP and DBP (ridge + SVR models)",
         "**Yes**, but trained on earlobe data; on unseen people not better than the average"],
        ["OLED / display", "None", "**No** (not implemented)"],
    ], [2.2, 4.2, 2.2])
    b.figure("oled")
    b.box("important", "the current workflow",
          "To see SBP/DBP estimates today, connect the ESP32 to a computer and run `rakshpulse estimate --source serial "
          "--port COM5 --fs 100` (Chapter 8). The computer calculates the estimates; the ESP32 only supplies the light "
          "samples.")


# ================================================================================================ 3
def ch03(b):
    b.chapter(3, "Understanding the PPG Graph", "What the curve, the dots and the axes mean", T.NAVY_2)
    b.p("The hardware team's screenshot shows a filtered PPG/IR waveform with heartbeat peaks marked in red. That "
        "screenshot file was not available when this manual was built, so the graph below is the **same kind of "
        "graph produced by RakshPulse's own preprocessing and peak detector** on a real recording. If your graph "
        "came from another script, the blue curve and red dots mean the same things.")
    b.image("ppg_peak_detection.png", "Reading a PPG peak-detection graph",
            "Real earlobe recording (Auricle dataset), processed by RakshPulse. Heart rate 74 bpm.")
    b.table(["What you see", "What it means", "What it is NOT"], [
        ["Blue curve", "The optical signal after filtering: how much light came back, minus the slow drift, scaled so "
         "the average is 0", "Not pressure. Not mmHg."],
        ["Red dots", "Moments where the software detected a heartbeat peak", "Not systolic pressure. A dot's height is "
         "light, not pressure"],
        ["Horizontal axis", "Time in seconds", ""],
        ["Vertical axis", "Filtered amplitude, no units (after z-scoring)", "Not blood pressure; the scale changes with "
         "skin, contact and LED current"],
        ["Distance between dots", "One heartbeat interval; heart rate = 60 ÷ interval", ""],
    ], [1.5, 4.2, 2.8])
    b.box("warning", "why you cannot read SBP or DBP from this graph",
          "The height of the wave depends on how much light reaches the detector: skin colour, sensor pressure, LED "
          "current, hair, temperature and ambient light all change it. Two people with the same blood pressure can "
          "have completely different heights. Blood pressure is a pressure in millimetres of mercury (mmHg); the PPG "
          "is a light signal with no units. A model can only *estimate* pressure statistically from the wave's "
          "**shape and timing**.")
    b.image("ppg_vs_blood_pressure.png", "PPG waveform versus blood-pressure output",
            "Left: the optical signal (what RakshPulse processes). Right: blood pressure from a separate reference device "
            "(CNAP), in mmHg, over the same recording. They are different quantities.")
    b.table(["PPG waveform", "Blood-pressure output"], [
        ["Optical signal over time, many values per second", "Two numbers per reading: SBP and DBP, in mmHg"],
        ["Shows heartbeats, their timing and pulse shape", "Systolic = highest pressure in the arteries during a beat; "
         "diastolic = lowest, between beats"],
        ["Comes from the MAX30102 (raw) or RakshPulse (filtered)", "Comes from a cuff (measured) or from the RakshPulse "
         "model (estimated)"],
        ["Unit: none (ADC counts or z-score)", "Unit: mmHg"],
    ], [4, 4])
    b.h2("Raw versus filtered signal")
    b.image("raw_red_ir_simulated.png", "Raw red and IR values, as the ESP32 prints them",
            "SIMULATED with the RakshPulse MAX30102 simulator. Real MAX30102 values have the same structure: a large "
            "steady level with a small pulse on top.")
    b.image("raw_vs_filtered.png", "Before and after preprocessing",
            "Real earlobe recording. Preprocessing removes the drift so the beats become clear.")
    b.h2("What the hardware team's video shows")
    b.image("hardware_video_plotter.png", "Frame from the hardware team's test video (laptop screen only)",
            "21 Nov 2026, 4-second clip. Arduino IDE Serial Plotter, board 'ESP32C3 Dev Module', 115200 baud.")
    b.bullets([
        "**Visible:** a small sensor module held between the fingers, wired to a breadboard; on the laptop, the "
        "Arduino IDE **Serial Plotter** at 115200 baud plotting three series, 'value 1', 'value 2' and 'value 3'.",
        "**Consistent with** the RakshPulse firmware, which prints three numbers per line (`sample_index,red,ir`): "
        "'value 1' is an almost straight line, which is what a steadily increasing sample counter looks like. We "
        "cannot confirm from the video which firmware was running.",
        "**What it demonstrates:** the sensor is delivering changing optical values to the ESP32-C3, and the ESP32-C3 "
        "is sending them to the laptop over USB.",
        "**What it does not show:** blood pressure (no SYS/DIA appears anywhere, as expected), a measured heart rate, "
        "or that the data meets the RakshPulse contract. In this short clip the two lower traces drift slowly and "
        "individual heartbeat ripples are not clearly visible at this zoom, which fits a finger that is not resting "
        "steadily on the sensor. Run `rakshpulse check-serial` (Chapter 8) to test the stream properly.",
        "**Note on the board:** it is an **ESP32-C3**. Its default I²C pins and USB settings differ from the classic "
        "ESP32 (Chapter 5).",
    ])


# ================================================================================================ 4
def ch04(b):
    m = M["model"]
    sel = M["sel"]
    b.chapter(4, "The Training Dataset and MAX30102", "Which data and sensor the model learned from, and what that means",
              T.TEAL)
    b.lead("A model can only be trusted on data that resembles what it learned from. This chapter identifies exactly "
           "what the shipped model learned from, and compares it with your MAX30102.")
    b.h2("The exact training dataset")
    n_sub = M["data"].subject_id.nunique()
    n_rec = M["data"].record_id.nunique()
    b.table(["Item", "Value", "Source"], [
        ["Full name", "Auricle: A Multimodal Wearable Sensing Dataset for Cardiovascular and Hemodynamic Assessment, "
         "version 1.0.0", "PhysioNet [7]"],
        ["Authors and publisher", "Christofferson, Bhalla, Kleinknecht, Varickanickal, Kim, De Luca, Simard, Ross, "
         "Cafazzo, Mariakakis; PhysioNet, 2026", "[7]"],
        ["URL / DOI", "https://physionet.org/content/auricle/1.0.0/ · doi:10.13026/swme-vp69", "[7]"],
        ["Participants", "42 healthy adults (19–55 years) + 28 people with heart failure; RakshPulse uses the "
         f"healthy cohort only: {n_sub} people and {n_rec} recordings after cleaning", "[7], data/ml"],
        ["PPG sensor", "'A Nonin PPG sensor' clipped onto the left earlobe in **transmissive** mode. **Exact sensor "
         "model not confirmed from the available source documentation** (the dataset cites only nonin.com)", "[7]"],
        ["Optical channel", "One PPG channel per site. **Wavelength not documented** in the dataset description", "[7]"],
        ["Acquisition", "BIOPAC MP160 with 100-series amplifiers, recorded at 10 kHz, published at 1 kHz; RakshPulse "
         "processes it at 100 Hz", "[7]"],
        ["Blood-pressure reference", "CNAP continuous non-invasive arterial pressure (finger cuffs on the right hand, "
         "BIOPAC-supported); RakshPulse uses the CNAP systolic/diastolic value of each 10 s window", "[7], [10]"],
        ["Features", f"{len(m['features'])} single-channel features: heart rate + pulse shape (production extractor)",
         "models/rakshpulse_ear_bp.json"],
        ["Models", f"SBP: ridge regression (alpha {json.loads(sel.loc['sbp', 'params'])['alpha']:g}); DBP: RBF support "
         "vector regression (C 3, epsilon 2, gamma 0.01); each after outlier clipping, imputation and scaling",
         "same"],
        ["Split", f"by person: {M['n_train']} training, {M['n_test']} test people; tuning with 5-fold GroupKFold; "
         "plus nested cross-validation over all 40", "reports/"],
    ], [1.7, 5.3, 1.6], size=8.6)
    b.h2("Training sensor versus MAX30102")
    b.figure("sensors")
    b.table(["Difference", "Auricle (training)", "MAX30102 (yours)", "Why it matters"], [
        ["Measurement site", "Earlobe", "Behind the ear (target)", "Different tissue and arteries change the pulse shape"],
        ["Optics", "Transmissive clip", "Reflective, LEDs next to detector", "Different light paths give different "
         "wave shapes and noise"],
        ["Wavelength", "Not documented", "Red ~660 nm, IR ~880 nm [1]", "Light penetration and pulse amplitude depend "
         "on wavelength"],
        ["Sampling", "10 kHz → 1 kHz", "100 Hz (firmware default)", "Timing features are resampled to 100 Hz; small "
         "differences remain"],
        ["Amplitude and scale", "mV, centred near 0", "ADC counts with a large DC level", "Removed by z-scoring, but "
         "noise behaves differently"],
        ["Placement and pressure", "Research clip", "Band or adhesive, varies", "Contact pressure reshapes the pulse "
         "and shifts features between days"],
        ["Noise and motion", "Seated lab protocol", "Field use", "More motion artefacts; more rejected windows"],
        ["Reference BP", "CNAP finger cuff", "Your upper-arm cuff", "Different reference devices disagree by several mmHg"],
    ], [1.6, 1.9, 2.0, 3.0], size=8.6)
    b.h2("Will the existing model work accurately with MAX30102?")
    b.figure("ladder")
    b.box("warning", "the direct answer",
          "**There is no evidence that it will, and good reasons to expect that it will not.** The software can read, "
          "process and run the model on MAX30102 data, so you will see numbers. But the model was trained on a "
          "different sensor at a different site, has never seen a MAX30102 recording, and even on its own earlobe "
          "data it is not significantly better than predicting the average blood pressure. Its contract therefore "
          "marks MAX30102 as `demonstration`: every MAX30102 SBP/DBP output is a demonstration, never a measurement.")
    b.h2("What is required before accurate MAX30102 estimates can be claimed")
    b.bullets([
        "MAX30102 recordings at the target site from **many people** (dozens, including people with high blood "
        "pressure), on **at least two days each**, with **at least three cuff readings per session** at known times "
        "(Chapter 9).",
        "Training a MAX30102 model with the same subject-wise evaluation used here, and showing it **beats the "
        "baselines** (population average; and, with calibration, the cuff value alone) on people it has never seen.",
        "Only then may its contract mark `max30102` as `experimental`. A `validated` claim would additionally need an "
        "independent clinical validation following recognised recommendations [4], [5], [6].",
    ])


# ================================================================================================ 5
def ch05(b):
    b.chapter(5, "Hardware: Parts, Wiring and Firmware", "Connecting the MAX30102 and uploading the reference sketch",
              T.BLUE)
    b.h2("Parts")
    b.table(["Part", "What to check"], [
        ["MAX30102 breakout board", "Which pins it exposes (VIN/VCC, GND, SDA, SCL, INT) and which supply voltage it "
         "needs. The bare chip needs 1.8 V plus a separate 3.3 V LED supply [1]; most breakout boards add their own "
         "regulators and pull-up resistors, but designs differ"],
        ["ESP32 board (the team's video shows an ESP32-C3)", "Its pinout: which pins are SDA and SCL, which pin "
         "supplies 3.3 V; whether USB goes through a USB-serial chip (CP210x/CH340) or the chip's native USB"],
        ["USB cable", "A **data** cable, not charge-only"],
        ["Breadboard and jumper wires", "Short, firm connections; loose wires cause noise and I²C errors"],
        ["Band or clip", "Must hold the sensor still with the same pressure every session"],
        ["Upper-arm cuff", "An automatic cuff for reference readings; note its make and model"],
    ], [2.4, 6])
    b.h2("Wiring")
    b.figure("wiring")
    b.bullets([
        "**VIN/VCC → the supply pin your breakout's documentation specifies.** Do not guess 3.3 V or 5 V.",
        "**GND → GND.**",
        "**SDA → the board's I²C data pin; SCL → the board's I²C clock pin.** The firmware uses the board's default I²C "
        "pins unless you set `I2C_SDA_PIN` and `I2C_SCL_PIN` at the top of the sketch.",
        "**INT** is optional; the firmware does not use it.",
        "The MAX30102's I²C address is fixed at `0x57`.",
    ])
    b.box("important", "ESP32-C3 boards (as in the video)",
          ["For the Arduino board setting **'ESP32C3 Dev Module'**, the Arduino-ESP32 core defines the default I²C pins "
           "as **SDA = GPIO 8, SCL = GPIO 9** [12]. These are also boot-strapping pins, and many small ESP32-C3 boards "
           "(e.g. 'Super Mini' clones) label their pins differently. **Check your exact board's pinout.** If you wire "
           "SDA/SCL to other pins, set `I2C_SDA_PIN` and `I2C_SCL_PIN` in the sketch to those GPIO numbers.",
           "ESP32-C3 boards without a USB-serial chip send `Serial` output over the chip's native USB only when "
           "**Tools → USB CDC On Boot = Enabled** [13]. If the Serial Monitor stays empty, check this first."])
    b.h2("Uploading the firmware")
    b.figure("firmware")
    b.box("warning", "untested procedure",
          "The software team could not compile or run the sketch on a real board (none was available). The steps "
          "below follow the Arduino-ESP32 documentation and the sketch's own comments; please report anything that "
          "does not work.")
    steps = [
        ("Install the Arduino IDE 2", "Download from arduino.cc and install.", "It compiles and uploads the sketch.",
         "The IDE opens.", "Using an outdated or unofficial copy.", "Re-download from arduino.cc."),
        ("Add the ESP32 boards", "Boards Manager → search 'esp32' → install **esp32 by Espressif Systems**.",
         "Adds the compiler and upload tools for every ESP32 variant, including the C3.",
         "ESP32 boards appear under Tools → Board.", "Skipping this and picking a different board.",
         "Restart the IDE; check the internet connection."),
        ("Install the sensor library", "Library Manager → **SparkFun MAX3010x Pulse and Proximity Sensor Library** "
         "(header `MAX30105.h`; it supports the MAX30102).", "The sketch uses it to configure and read the sensor.",
         "Listed as installed.", "Installing another MAX30102 library with a different interface.",
         "Remove other MAX3010x libraries if the compiler picks the wrong one."),
        ("Open the sketch", "Open `hardware/firmware/rakshpulse_max30102/rakshpulse_max30102.ino` from the cloned "
         "repository.", "It prints exactly the format RakshPulse expects.", "The sketch opens in its own folder.",
         "Copying the code into a new sketch and changing the print format.", "Open the .ino file directly."),
        ("Check the settings", "Keep `SAMPLE_RATE 100`, `SAMPLE_AVERAGE 1` (= 100 Hz). Set `I2C_SDA_PIN`/`I2C_SCL_PIN` "
         "only if you do not use the default pins.", "The rate must match `--fs` in every RakshPulse command.",
         "Values noted in your lab notebook.", "Changing `SAMPLE_AVERAGE` and forgetting the rate changes.",
         "Read the banner after upload: it states `fs_hz`."),
        ("Select board and port", "Plug in the USB cable. Tools → Board (e.g. **ESP32C3 Dev Module**). For native-USB C3 "
         "boards: Tools → **USB CDC On Boot → Enabled**. Tools → Port: the new port.",
         "The IDE must know the chip and where it is connected.", "A port appears when the board is plugged in.",
         "Charge-only cable; wrong board variant.", "Try another cable; install the CP210x/CH340 driver if your board "
         "has that chip."),
        ("Upload", "Click Upload (→).", "Compiles and flashes the board.", "'Done uploading' (or similar), no errors.",
         "Upload never starts on some ESP32 boards.", "Hold BOOT, tap RESET (or plug in while holding BOOT), release, "
         "upload again."),
        ("Look, then close the Serial Monitor", "Tools → Serial Monitor, set **115200 baud**, read a few lines, then "
         "**close it**.", "Confirms the sensor was found. Only one program can use the port.",
         "A banner `# rakshpulse fw=1 fs_hz=100 …`, then lines like `1234,95321,118204` (see below).",
         "Leaving it open: every RakshPulse command then fails to open the port.",
         "Garbled text: wrong baud. `# ERROR: MAX30102 not found`: wiring, supply or I²C pins."),
    ]
    for i, s in enumerate(steps, 1):
        b.step(i, *s)
    b.output("stream", "Serial Monitor")
    b.p("Each data line is `sample_index,red,ir`: a counter that goes up by one per sample, the red reading and the IR "
        "reading. Lines starting with `#` are comments. With a finger on the sensor the red and IR numbers should "
        "change slightly with each heartbeat and stay well below 262,143 (if they sit at 262,143 the light is too "
        "strong: lower `LED_POWER`).")


# ================================================================================================ 6
def ch06(b):
    b.chapter(6, "Installing the Software", "Windows, Linux and macOS, from zero", T.TEAL)
    b.p("You will type commands into a **terminal**: a window where you type a line and press Enter. Type each command "
        "exactly, on one line, then wait until it finishes before typing the next. Lines that start with `PS>` or `$` "
        "in the grey boxes are what you type (without the `PS>` or `$`); other lines are what the computer answers.")
    b.h2("Windows (PowerShell)")
    b.bullets([
        "**Install Git:** download 'Git for Windows' from git-scm.com, run the installer with the default options.",
        "**Install Python 3.11:** download from python.org; on the first installer screen tick **'Add python.exe to "
        "PATH'**. RakshPulse supports Python 3.10, 3.11 and 3.12.",
        "**Open PowerShell:** press the Windows key, type *PowerShell*, press Enter.",
        "**Type the commands below**, one per line.",
    ], numbered=True)
    b.terminal("""
> git clone https://github.com/obstinix/RakshPulse.git
> cd RakshPulse
> py -3.11 -m venv .venv
> .\\.venv\\Scripts\\Activate.ps1
> pip install -e ".[serial]"
> rakshpulse doctor
""", "PowerShell", "download, create a private Python environment, install, check")
    b.table(["Command", "What it does"], [
        ["`git clone …`", "Downloads the RakshPulse repository into a folder called RakshPulse"],
        ["`cd RakshPulse`", "Opens that folder"],
        ["`py -3.11 -m venv .venv`", "Creates a private Python environment in `.venv` (affects nothing else)"],
        ["`.\\.venv\\Scripts\\Activate.ps1`", "Switches the terminal to that environment; the prompt then starts with `(.venv)`"],
        ["`pip install -e \".[serial]\"`", "Installs RakshPulse and the serial-port library (a few minutes)"],
        ["`rakshpulse doctor`", "Checks the installation (output below)"],
    ], [2.6, 5.4])
    b.box("info", "if Activate.ps1 is blocked",
          "Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, answer Y, then run the activation line again. "
          "In every **new** PowerShell window, run `cd RakshPulse` and the activation line again before using "
          "`rakshpulse`.")
    b.h2("Linux (bash)")
    b.terminal("""
> sudo apt install git python3 python3-venv
> git clone https://github.com/obstinix/RakshPulse.git
> cd RakshPulse
> python3 -m venv .venv
> source .venv/bin/activate
> pip install -e ".[serial]"
> sudo usermod -aG dialout "$USER"
> rakshpulse doctor
""", "bash", "Debian/Ubuntu package names; other distributions use their own installer")
    b.p("The `usermod` line gives your account permission to use serial ports (`/dev/ttyUSB0`, `/dev/ttyACM0`). It takes "
        "effect after you **log out and in again**. Without it, opening the port fails with 'permission denied'.")
    b.h2("macOS (Terminal)")
    b.box("important", "not tested on macOS",
          "RakshPulse is tested automatically on Linux and Windows only. The commands below are the standard macOS "
          "equivalents; please report problems.")
    b.terminal("""
> xcode-select --install
> git clone https://github.com/obstinix/RakshPulse.git
> cd RakshPulse
> python3 -m venv .venv
> source .venv/bin/activate
> pip install -e ".[serial]"
> rakshpulse doctor
""", "zsh (macOS Terminal)", "xcode-select installs git; install Python 3.11 from python.org if python3 is missing")
    b.p("On macOS the board appears as `/dev/cu.usbserial-…`, `/dev/cu.SLAB_USBtoUART` (CP210x chip), "
        "`/dev/cu.wchusbserial…` (CH340) or `/dev/cu.usbmodem…` (native USB, e.g. ESP32-C3). `rakshpulse ports` lists "
        "them. If a CP210x or CH340 board does not appear, install the chip maker's macOS driver and allow it under "
        "System Settings → Privacy & Security.")
    b.h2("Checking the installation")
    b.output("doctor", "Terminal")
    b.box("success", "",
          "The last line must say `doctor: all required checks passed`, and the serial line must say `serial recording "
          "available`. Optional packages may show `n/a`; that is fine.")
    b.p("To see every RakshPulse command: `rakshpulse -h`. For help on one command: `rakshpulse estimate -h`.")
    b.output("help", "Terminal", max_lines=27)


# ================================================================================================ 7
def ch07(b):
    b.chapter(7, "What Will I Actually See?", "Every stage of the output, with real examples", T.NAVY_2)
    b.p("Each item says what the output means, which program produces it, where the example comes from and what to do "
        "next. All terminal text is real output of the current software; the data source is stated above each block.")
    items = [
        ("1. Sensor connection / serial output", "Arduino Serial Monitor, or `rakshpulse ports` and `check-serial`",
         "Shows whether the computer can see the board and read lines from it.", "Run `check-serial` (Ch. 8)."),
        ("2. Firmware start-up banner", "ESP32 firmware, once after reset",
         "`# rakshpulse fw=1 fs_hz=100 …` confirms the RakshPulse sketch is running and states its sampling rate.",
         "Use the same number as `--fs` in every command."),
        ("3. Raw red and IR sample stream", "ESP32 firmware, 100 lines per second",
         "`index,red,ir`: light readings. Not heart rate, not blood pressure.", "Close the Serial Monitor."),
    ]
    for t, prog, mean, nxt in items:
        b.h3(t)
        b.p(f"**Program:** {prog}. **Meaning:** {mean} **Next:** {nxt}")
    b.output("stream", "Serial Monitor")
    b.h3("4–6. PPG waveform, filtered waveform, heartbeat peaks")
    b.p("**Program:** RakshPulse preprocessing (inside `estimate`, `analyze`, the notebooks). The software does not "
        "pop up these graphs during `estimate`; it draws them into a PNG with `--plot` (item 10), and the notebooks "
        "show them. See Figures in Chapter 3. **Next:** if no regular beats are visible, fix contact before recording.")
    b.h3("7–8. Heart rate and signal quality")
    b.image("quality_and_heart_rate.png", "Heart rate, quality score and accepted windows over 5 minutes",
            "Real earlobe recording processed by the RakshPulse pipeline. Heart rate is only trustworthy in accepted windows.")
    b.output("inspect_quality", "Terminal")
    b.p("**Program:** `rakshpulse inspect --dataset data/rakshpulse_hw --quality` (after `analyze`). **Meaning:** share of usable and degraded "
        "windows and the most common reasons. **Next:** aim for at least 80 % usable windows at rest.")
    b.h3("9. Feature extraction")
    b.output("csv_windows", "derived/windows.csv")
    b.p("**Program:** `analyze` writes one row per window with its status, quality score, reasons and about 100 feature "
        "columns (`ir_hr` is heart rate, `ir_sys_ratio` a pulse-shape feature). **Next:** nothing to do; the model "
        "uses these.")
    b.image("feature_distributions.png", "Three of the 40 features the model uses",
            "Real earlobe data. People with higher and lower blood pressure overlap almost completely, which is why "
            "estimating BP from these features is hard.")
    b.h3("10–11. Model inference and the SBP/DBP result format")
    b.box("warning", "ILLUSTRATIVE SYNTHETIC EXAMPLE — NOT A REAL BLOOD-PRESSURE MEASUREMENT",
          "The input below is the RakshPulse simulator, not a person. The SBP/DBP numbers only show the output "
          "format; they are not anyone's blood pressure.")
    b.output("estimate", "Terminal")
    b.p("**Program:** `rakshpulse estimate`. **Meaning:** every line is a status field: input validation, quality, "
        "heart rate, estimated SBP/DBP, the operating mode, calibration status, model version and latency. The "
        "warning says the numbers are a demonstration. **Next:** with `--json r.json` the same fields are saved as JSON; "
        "with `--plot d.png` a dashboard image is drawn (below).")
    b.image("../../assets/inference_dashboard_simulated.png", "The status dashboard written by `estimate --plot`",
            "SIMULATED sensor input. The red banner states the input source and that this is not a blood-pressure "
            "measurement.")
    b.output("estimate_poor", "Terminal")
    b.p("When the sensor is off the skin, no window passes and **no SBP/DBP is returned**: `status poor_signal`, exit "
        "code 1. This is deliberate: the software never shows a number it cannot support.")
    b.h3("12. Dataset CSV output")
    b.output("csv_raw", "raw_ppg.csv")
    b.p("**Program:** `rakshpulse record`. One row per sample, raw values never filtered. Details of every file in "
        "Chapter 10.")
    b.h3("13. Final analysis report")
    b.box("warning", "ILLUSTRATIVE SYNTHETIC EXAMPLE — NOT A REAL BLOOD-PRESSURE MEASUREMENT",
          "Simulated PPG and made-up cuff readings. The comparison numbers show the report format only.")
    b.output("analyze", "Terminal")
    b.p("**Program:** `rakshpulse analyze`. **Meaning:** five steps: validation, windows, model check and prediction, "
        "comparison with the cuff readings, writing files. The comparison line is labelled *not an accuracy claim* in "
        "demonstration mode. **Next:** share the dataset folder (Ch. 10). The model-evaluation report of the whole "
        "project is `reports/final_model_evaluation.md` (Ch. 11).")


# ================================================================================================ 8
def ch08(b):
    b.chapter(8, "First Live Test and Recording", "Finding the port, testing the stream, recording a session", T.TEAL)
    b.h2("1. Find the port")
    b.p("Close the Arduino Serial Monitor. In the activated terminal run `rakshpulse ports` once with the board "
        "unplugged and once plugged in: the new line is your board.")
    b.output("ports", "PowerShell")
    b.p("The build PC has no board attached, so only Bluetooth ports appear: those are **not** your ESP32. A CP210x "
        "board shows a description like *Silicon Labs CP210x USB to UART Bridge (COM5)*, a CH340 board *USB-SERIAL "
        "CH340*, a native-USB ESP32-C3 *USB JTAG/serial debug unit* or *USB Serial Device*.")
    b.h2("2. Test the data stream")
    b.terminal("> rakshpulse check-serial --port COM5 --fs 100", "PowerShell", "replace COM5 with your port")
    b.output("check_serial_pass", "Terminal")
    b.table(["Field", "Good value", "If not"], [
        ["`samples`", "about 100 × seconds", "few or none: baud, firmware or wiring"],
        ["`malformed_lines`", "0", "baud mismatch or extra prints in the firmware"],
        ["`missing_samples`, `gaps`", "0 (a few are tolerated and logged)", "loop or serial too slow"],
        ["`measured_rate_hz`", "close to `--fs`", "wrong `--fs` or averaging"],
        ["`declared_fs_hz`", "equal to `--fs`", "pass the banner's value as `--fs`"],
        ["`red_range`, `ir_range`", "inside 0–262143, changing", "constant: no contact or LED off; 262143: too bright"],
    ], [2.2, 2.6, 3.6])
    b.p("Two real failures and what they say:")
    b.output("check_serial_fail_fs", "Terminal", max_lines=16)
    b.output("check_serial_fail_malformed", "Terminal", max_lines=16)
    b.box("success", "PASS", "The last line reads `serial check: PASS`. Your hardware and firmware deliver the right "
          "data. Do not record until you get this.")
    b.h2("3. A live estimate (optional)")
    b.terminal("> rakshpulse estimate --source serial --port COM5 --fs 100 --seconds 30 --plot live.png",
               "PowerShell", "30 s from the board; prints the status panel and saves a dashboard")
    b.p("This shows heart rate, quality and an SBP/DBP **demonstration** estimate. It is useful to check placement, "
        "not to measure anyone's blood pressure.")
    b.h2("4. Record a session")
    b.terminal(f"> rakshpulse record --source serial --port COM5 --fs 100 --out {DS} --subject P001 --session D1 "
               "--duration 300 --placement behind_ear --contact \"elastic band, moderate\"", "PowerShell",
               "5-minute recording")
    b.p("Real output of the same command reading recorded example samples instead of a board:")
    b.output("record", "Terminal")
    b.bullets([
        "`--subject`: a code such as `P001`, **never a name**. `--session`: `D1`, `D2` … one label per recording.",
        "The session folder is created only when valid samples arrive, and never overwritten: use a new `--session` "
        "label to record again.",
        "Stop with `--duration` or **Ctrl-C**; both save and checksum the files. If the cable is pulled, the recording "
        "ends after `--timeout` seconds and keeps what it received.",
        "The final line summarises samples, missing samples, gaps, duplicates, counter resets and garbled lines.",
    ])


# ================================================================================================ 9
def ch09(b):
    b.chapter(9, "Cuff References and Data Collection", "Why reference readings matter and how to record them", T.ORANGE)
    b.lead("A model can only learn blood pressure from PPG if, for every recording, we know the true blood pressure at "
           "known times. That is what the cuff readings provide. Without them a recording shows signal quality but "
           "can never train or test a model.")
    b.table(["Term", "Meaning", "Source"], [
        ["SBP (systolic)", "Highest arterial pressure during a heartbeat, in mmHg", "cuff (measured) or model (estimated)"],
        ["DBP (diastolic)", "Lowest arterial pressure between beats, in mmHg", "cuff or model"],
        ["Heart rate", "Beats per minute", "from PPG peak intervals, or the cuff's display"],
        ["PPG", "The optical waveform from the MAX30102", "sensor"],
        ["Cuff reference", "SBP/DBP measured by the upper-arm cuff, typed in with its time", "`rakshpulse reference`"],
        ["Model estimate", "SBP/DBP calculated by RakshPulse from PPG features", "`estimate`, `analyze`"],
    ], [1.6, 4.2, 2.6])
    b.h2("The procedure")
    b.bullets([
        "**Prepare:** consent signed, pseudonymous ID assigned (e.g. `P001`); sensor fitted at the measurement site with "
        "the same band and pressure as always; cuff on the upper arm at heart level; participant seated, not talking.",
        "**Connect** the board; close the Serial Monitor.",
        "**Check** the signal: `rakshpulse check-serial --port COM5 --fs 100` must PASS.",
        "**Start recording** (`rakshpulse record …`, Ch. 8) and start a stopwatch at the same moment.",
        "**Cuff readings:** after 5 minutes of seated rest, at 1, 3 and 5 minutes of rest, then every 5 minutes; at least "
        "3 per session, at least 60 s apart. Write down the **stopwatch time when the cuff shows its result**, with "
        "SBP, DBP and heart rate.",
        "**Stop** the recording (Ctrl-C or `--duration`).",
        "**Enter** each reading with `rakshpulse reference` (below).",
        "**Validate:** `rakshpulse inspect --dataset data/rakshpulse_hw --validate` must say `validation: all PASS`.",
        "**Share** the dataset folder with the software team (Ch. 10).",
    ], numbered=True)
    b.figure("timeline")
    b.p("This timing comes from the repository's protocol (`docs/DATA_COLLECTION_PROTOCOL.md`), which also asks for "
        "at least two sessions on different days per participant with identical placement, and one deliberately "
        "re-seated sensor within a session. Follow any protocol approved for your study; this manual does not replace "
        "one.")
    b.h2("Entering the readings")
    b.box("warning", "ILLUSTRATIVE SYNTHETIC EXAMPLE — NOT A REAL BLOOD-PRESSURE MEASUREMENT",
          "The cuff values in this section (121/79, 118/77, 119/78 mmHg) are invented to show the format.")
    b.table(["Stopwatch time (s)", "SBP (mmHg)", "DBP (mmHg)", "HR (bpm)", "Cuff"], [
        ["30", "121", "79", "72", "Omron M3"], ["60", "118", "77", "70", "Omron M3"], ["85", "119", "78", "71", "Omron M3"],
    ], [2, 1.4, 1.4, 1.2, 1.6])
    b.p("**ILLUSTRATIVE values** for the 90-second example session used in this manual (a real session is longer and "
        "follows the timing above). Each row becomes one command:")
    b.output("reference", "PowerShell")
    b.h2("How readings are matched with PPG")
    b.figure("matching")
    b.p("A reading labels only accepted windows of the same session whose time span lies within 30 seconds of the "
        "reading; the nearest window wins. Readings that match no window are reported. This is why the time matters: "
        "a reading entered 2 minutes late is compared with the wrong heartbeats.")
    b.box("warning", "consent, privacy and safe handling", [(
        "Recordings need approval from the responsible ethics committee and informed consent from every participant.",
        "Use pseudonymous IDs only (`P001`). Never put names, contact details or photos in the dataset. Keep the list "
        "linking IDs to people offline, separately, with the study lead.",
        "Never upload participant recordings to the public GitHub repository or public chats. Share them privately.",
        "Never edit recorded files by hand: checksums detect it. Record a new session instead and note why.",
        "RakshPulse is not a medical device: never use its output to advise anyone about their health.",
    )])


# ================================================================================================ 10
def ch10(b):
    b.chapter(10, "Dataset Files and Where Results Are", "The folder RakshPulse writes, file by file", T.NAVY_2)
    b.output("tree", "Folder")
    b.table(["File", "Written by", "Contents"], [
        ["`dataset_manifest.csv`", "record, reference, analyze", "SHA-256 checksum of every file, used by `inspect --validate`"],
        ["`subjects.csv`", "record", "One row per subject code"],
        ["`session_metadata.csv`", "record", "Date, sensor type, placement, sampling rate, contact, notes (with the "
         "firmware banner), sensor settings, software version, `simulated` flag"],
        ["`raw_ppg.csv`", "record", "One row per sample: time, IDs, sample index, red, infrared (raw counts)"],
        ["`gaps.csv`", "record", "Every lost-sample gap: time, first missing index, number missing, how detected"],
        ["`reference_bp.csv`", "reference", "Cuff readings: time, SBP, DBP, heart rate, device"],
        ["`derived/windows.csv`", "analyze", "One row per 10 s window: status, reasons, quality score, features"],
        ["`derived/segments.csv`", "analyze", "Continuous stretches between gaps"],
        ["`derived/predictions_rakshpulse-ear-bp.csv`", "analyze", "Per window: status, mode, SBP/DBP estimate, matched cuff "
         "values, role"],
    ], [3.0, 1.6, 4.0], size=8.6)
    b.h3("Examples of the files (real files from the example session)")
    b.output("csv_metadata", "session_metadata.csv")
    b.output("csv_gaps", "gaps.csv")
    b.output("csv_reference", "reference_bp.csv")
    b.output("csv_predictions", "predictions file")
    b.h2("Validating and sharing")
    b.output("inspect_validate", "Terminal")
    b.bullets([
        "Results are in `data/rakshpulse_hw/sessions/<subject>/<session>/derived/` inside the RakshPulse folder.",
        "To share: zip the **whole** `data/rakshpulse_hw` folder (the checksums must travel with the data) and send it "
        "privately to the software team. Windows: `Compress-Archive -Path data/rakshpulse_hw -DestinationPath "
        "rakshpulse_hw.zip`. Linux/macOS: `zip -r rakshpulse_hw.zip data/rakshpulse_hw`.",
        "Include a short note: board and breakout models, sketch settings, placement and band, cuff model, anything "
        "unusual.",
    ])


# ================================================================================================ 11
def ch11(b):
    sel, base, bc, nest, cal = M["sel"], M["base"], M["bc"], M["nest"], M["cal"]
    b.chapter(11, "The Model's Accuracy, Explained Honestly", "What the numbers mean and what they do not", T.TEAL)
    b.h2("The words")
    b.table(["Term", "Plain meaning"], [
        ["MAE (mean absolute error)", "On average, how many mmHg the estimate is away from the reference. Lower is better."],
        ["RMSE", "Like MAE but punishes large errors more."],
        ["R²", "How much of the real differences in BP the model explains: 1 = all, 0 = none (no better than the "
         "average), below 0 = worse than the average."],
        ["Baseline", "The simplest possible 'model': always answer the average BP of the training people. A useful "
         "model must beat it."],
        ["Subject-independent test", "Tested only on people the model never saw during training or tuning."],
        ["Nested cross-validation", "Repeats training and testing so that every one of the 40 people is predicted once "
         "by a model that never saw them: a more stable estimate than 8 test people."],
        ["Calibration", "Using one cuff reading from the same person to correct later estimates."],
        ["95 % confidence interval", "The range in which the true difference probably lies. If it includes 0, the "
         "difference is not significant."],
    ], [2.6, 6])
    b.figure("error")
    b.h2("The actual results")
    s_rows = []
    for t in ("sbp", "dbp"):
        s_rows.append([t.upper(), sel.loc[t, "label"], f1(sel.loc[t, "test_mae"]), f1(sel.loc[t, "test_rmse"]),
                       f2(sel.loc[t, "test_r2"]), f1(base.loc[t, "test_mae"]), ci(bc.loc[t])])
    b.table(["Target", "Selected model", "MAE", "RMSE", "R²", "Baseline MAE", "Difference [95 % CI]"], s_rows,
            [0.8, 2.3, 0.8, 0.8, 0.7, 1.2, 2.0])
    b.p(f"**Independent test:** {M['n_test']} people never used for training or model choice (mmHg). Source: "
        "`reports/final_model_evaluation.md`.")
    n_rows = []
    for t in ("sbp", "dbp"):
        r, rb = nest.loc[(t, "nested-selected")], nest.loc[(t, "mean")]
        n_rows.append([t.upper(), f1(r.mae), f1(r.rmse), f2(r.r2), f1(rb.mae), ci(r)])
    b.table(["Target", "MAE", "RMSE", "R²", "Baseline MAE", "Difference [95 % CI]"], n_rows, [0.8, 1, 1, 0.8, 1.4, 2.2])
    b.p("**Nested cross-validation over all 40 people.**")
    b.box("warning", "what this means",
          "Neither the SBP nor the DBP model is significantly better than always answering the average: every "
          "confidence interval includes 0, and R² is close to 0. In plain words, **from a 10-second earlobe PPG "
          "window, without a cuff reading, the model cannot tell an individual's blood pressure better than a guess "
          "of the population average.** This is the model that produces the demonstration numbers you see.")
    b.image("actual_vs_predicted.png", "Estimated versus reference blood pressure for the 8 unseen people",
            "Model results on earlobe data. A perfect model would put every point on the red line; these points are "
            "a horizontal band, i.e. the model mostly answers values near the average.")
    b.image("error_distribution.png", "How large the errors are", "Model results on earlobe data.")
    b.image("model_vs_baseline.png", "All seven candidate models against the baseline",
            "Model results on earlobe data. No model is clearly below the red baseline line.")
    b.h2("With one cuff reading (calibration)")
    rows = []
    for _, r in cal.iterrows():
        diff = "" if r.method == "carry_forward" else ci(r)
        rows.append([r.scenario, r.target.upper(), r.method, f1(r.mae), diff])
    b.table(["Scenario", "Target", "Method", "MAE", "vs cuff value alone [95 % CI]"], rows, [2.2, 0.8, 1.6, 0.8, 2.4],
            size=8.4)
    b.p("`carry_forward` = simply repeating the cuff reading. The only method that beats it significantly is the "
        "`delta_ridge` model for DBP within the same session (a small gain). Across days nothing beats the cuff value. "
        "`rakshpulse estimate` uses `delta_ridge` when you calibrate through the Python interface.")
    b.h2("Five different kinds of 'it works'")
    b.table(["Kind of test", "What it shows", "Does it show BP accuracy?"], [
        ["Software tests (190, CI)", "The code does what it is designed to do: parsing, files, errors, contracts", "**No**"],
        ["Signal-processing tests", "Filtering, beat detection and quality checks behave correctly (e.g. heart rate "
         "error 0.3 bpm on simulated signals)", "**No**"],
        ["Synthetic smoke test", "The whole pipeline runs end to end on invented data", "**No**"],
        ["Real-data model evaluation", "Accuracy on real earlobe recordings (this chapter)", "Only for earlobe data like "
         "Auricle; result: not better than the average"],
        ["MAX30102 hardware validation", "Accuracy of MAX30102-based estimates against cuffs on new people",
         "**Would**, but **has not been done**"],
    ], [2.4, 4.2, 2.2])


# ================================================================================================ 12
def ch12(b):
    b.chapter(12, "Troubleshooting", "Symptom → likely cause → solution, and whose problem it is", T.RED)
    rows = [
        ["ESP32 not detected by the computer", "HW", "Charge-only cable, bad USB port, board not powered", "Use a data cable, another port; check the board's power LED"],
        ["COM port missing", "HW / driver", "USB-serial driver missing (CP210x/CH340); native-USB C3 not in CDC mode", "Install the driver; for C3 set USB CDC On Boot = Enabled and re-upload"],
        ["Driver problem (Windows 'Other devices', macOS blocked)", "Driver", "Driver not installed or not allowed", "Install the chip maker's driver; allow it in macOS Privacy & Security"],
        ["`# ERROR: MAX30102 not found on I2C`", "HW / FW", "Wiring, supply, wrong SDA/SCL pins", "Check each wire; use the breakout's supply pin; set I2C_SDA_PIN/I2C_SCL_PIN"],
        ["I²C communication failure (sensor found sometimes)", "HW", "Loose wires, long wires, missing pull-ups", "Short firm wires; check the breakout has pull-ups"],
        ["No red/IR values at all", "FW / SW", "Wrong baud, Serial Monitor still open, firmware not running", "115200 on both sides; close the monitor; re-upload"],
        ["Values remain constant", "HW", "No skin contact, LED current 0, wrong channel", "Place finger/skin on the sensor; check LED_POWER"],
        ["PPG graph is flat", "HW", "Sensor off the skin or saturated at 262143", "Improve contact; lower LED_POWER if at the maximum"],
        ["PPG graph is noisy", "HW / use", "Motion, ambient light, loose mounting", "Keep still; shield light; tighten the band"],
        ["Peaks are missing", "Signal", "Weak pulse (low LED current, cold skin, light pressure)", "Raise LED_POWER a little; adjust pressure; warm hands"],
        ["Too many peaks detected", "Signal", "Noise or motion creates false peaks", "Keep still; these windows are usually rejected anyway"],
        ["`check-serial` FAIL", "Depends", "Read the `problem:` lines", "Each problem line names the fix (Ch. 8)"],
        ["`firmware declares fs_hz=X but --fs is Y`", "SW use", "--fs differs from the firmware", "Use --fs X (the banner's value)"],
        ["`measured rate … differs from --fs`", "FW / SW", "Board too slow, averaging not accounted for", "Check SAMPLE_AVERAGE; avoid delay() in loop()"],
        ["Recording stops early", "HW / SW", "Cable pulled, board rebooted, --timeout reached", "Secure the cable; see counter_resets in the summary"],
        ["Dataset validation fails", "SW use", "A file was edited or copied with changed line endings", "Never edit files; copy the folder as a zip"],
        ["No SBP/DBP displayed", "Expected", "See the flowchart below: the ESP32 never shows them; poor signal returns none", "Run `estimate` on the computer; fix signal quality"],
        ["'DEMONSTRATION MODE' appears", "Expected", "The model is not validated for MAX30102", "Nothing to fix: it means the numbers are not measurements"],
        ["No OLED display output", "Not implemented", "There is no display code", "See Chapter 2 for what would be needed"],
        ["Model refuses inference (`incompatible_model`, exit 4)", "Model", "Unsupported sampling rate (50–1000 Hz) or sensor", "Check --fs; run `rakshpulse models --sensor max30102 --fs 100`"],
        ["CSV file not generated", "SW use", "Command failed earlier (exit code ≠ 0), session already existed, analyze not run", "Read the error line; use a new --session; run analyze"],
        ["`rakshpulse` is not recognised", "SW", "Environment not activated", "Run the activation line (Ch. 6)"],
    ]
    b.table(["Symptom", "Belongs to", "Possible cause", "Solution"], rows, [2.6, 1.1, 2.4, 2.6], head_color=T.RED, size=8.2)
    b.p("HW = hardware/wiring, FW = firmware, SW = software installation or use. 'Expected' means the software is doing "
        "what it should.")
    b.figure("flowchart")
    b.box("tip", "still stuck?",
          ["Send the software team: the exact command, its full output, the exit code (PowerShell: `$LASTEXITCODE`; "
           "bash: `echo $?`), the first 20 lines of the Serial Monitor, your board and breakout models and the sketch "
           "settings."])


# ================================================================================================ 13
def ch13(b):
    b.chapter(13, "Quick Reference", "The commands and files on one page", T.NAVY)
    b.table(["Task", "Command (one line)"], [
        ["Check the installation", "`rakshpulse doctor`"],
        ["List all commands", "`rakshpulse -h`"],
        ["Find the board", "`rakshpulse ports`"],
        ["Test the stream", "`rakshpulse check-serial --port COM5 --fs 100`"],
        ["Live estimate + dashboard", "`rakshpulse estimate --source serial --port COM5 --fs 100 --seconds 30 --plot live.png`"],
        ["Record 5 minutes", f"`rakshpulse record --source serial --port COM5 --fs 100 --out {DS} --subject P001 --session D1 --duration 300`"],
        ["Add a cuff reading", f"`rakshpulse reference --dataset {DS} --subject P001 --session D1 --time 60 --sbp <SBP> --dbp <DBP>`"],
        ["Validate the dataset", f"`rakshpulse inspect --dataset {DS} --validate`"],
        ["Analyse a session", f"`rakshpulse analyze --dataset {DS} --subject P001 --session D1`"],
        ["Quality summary", f"`rakshpulse inspect --dataset {DS} --quality`"],
        ["Model and its limits", "`rakshpulse models --sensor max30102 --fs 100 --verbose`"],
        ["Try without hardware", "`rakshpulse estimate --source sim --seconds 30`"],
    ], [2.2, 6.2], size=8.6)
    b.table(["Exit code", "Meaning"], [["0", "ok"], ["1", "check failed / poor signal / no trustworthy result"],
                                       ["2", "invalid input or usage"], ["3", "device or serial problem"],
                                       ["4", "incompatible model"]], [1, 6], size=8.6)
    b.table(["Status", "Meaning", "SBP/DBP given?"], [
        ["`ok`", "enough windows passed", "yes (demonstration on MAX30102)"],
        ["`poor_signal`", "too few windows passed the quality check", "no"],
        ["`invalid_input`", "samples or sampling rate unusable", "no"],
        ["`incompatible_model`", "model not allowed for this sensor/rate", "no"],
    ], [1.8, 4, 2.4], size=8.6)
    b.p("Firmware defaults: 115200 baud, 100 samples/s, averaging 1, 411 µs pulse width, LED current 0x24, I²C address "
        "0x57. Results: `data/rakshpulse_hw/sessions/<subject>/<session>/derived/`.")


# ================================================================================================ 14
def ch14(b):
    b.chapter(14, "How to Explain and Demonstrate RakshPulse", "A presentation sequence, a script and honest answers",
              T.ORANGE)
    b.figure("present")
    b.h2("A short speaking script")
    script = [
        "**Problem.** Blood-pressure cuffs are bulky and slow. We want to know whether a small optical sensor behind "
        "the ear can estimate blood pressure.",
        "**PPG.** Each heartbeat pushes blood into the small arteries, which absorbs a little more light. Shining light "
        "into the skin and measuring what returns gives a wave: the PPG.",
        "**MAX30102.** This module has a red and an infrared LED and a light detector. It measures light, not pressure.",
        "**ESP32.** The ESP32 reads the sensor 100 times a second and sends the raw numbers to the laptop over USB.",
        "**Live graph.** Here is the wave. Each peak is one heartbeat.",
        "**Heart rate.** The time between peaks gives the heart rate, here about 75 beats per minute.",
        "**ML pipeline.** On the laptop, RakshPulse filters the signal, checks its quality, and measures the shape of "
        "every pulse: 40 numbers per 10-second window.",
        "**SBP/DBP.** A model trained on a public research dataset turns those numbers into systolic and diastolic "
        "estimates.",
        "**Output.** Here is the result with heart rate, quality and the estimate. Notice the label: demonstration.",
        "**Limitations.** The model learned from a different sensor at the earlobe, and even there it is not better "
        "than guessing the average. So these numbers are not measurements.",
        "**Next step.** We are now collecting MAX30102 recordings with cuff readings from many people on several days. "
        "Only a model trained and tested on that data could estimate blood pressure from this device.",
    ]
    b.bullets(script, numbered=True)
    b.h2("Likely questions and honest answers")
    qa = [
        ("Why can't the MAX30102 directly display blood pressure?",
         "Because it only measures how much light comes back from the skin. Blood pressure is a pressure inside the "
         "arteries; there is no pressure sensor or cuff in the module. Software has to estimate pressure from the "
         "shape and timing of the light wave, and that estimate has to be learned from data with real cuff readings."),
        ("Is this a blood-pressure monitor?", "No. It is a research prototype. It is not a medical device and its "
         "numbers must not be used for health decisions."),
        ("How accurate is it?", "On public earlobe data, about 14–15 mmHg average error for systolic pressure on unseen "
         "people, which is not better than guessing the average. On MAX30102 it is unknown."),
        ("Why does the ESP32 not show the numbers?", "The firmware only sends raw light samples. The calculation runs "
         "on the computer. A display would need new firmware and a way to send results back."),
        ("What would make it accurate?", "MAX30102 recordings with cuff readings from many people on several days, a "
         "model trained and tested on them, and eventually a proper clinical validation."),
        ("Does the heart rate work?", "Heart rate comes from peak intervals and is reliable in good-quality windows "
         "(0.3 bpm error on simulated signals); it has not been compared with a reference on MAX30102 recordings."),
    ]
    for q, a in qa:
        b.h3(q)
        b.p(a)


# ================================================================================================ 15
def ch15(b):
    b.chapter(15, "References", "Published sources, and how they relate to RakshPulse's own results", T.NAVY)
    b.p("Numbers in brackets in this manual refer to this list. The accuracy results in Chapter 11 are **RakshPulse's "
        "own experimental results**, not taken from these publications.")
    refs = [
        ("Analog Devices (Maxim Integrated). MAX30102: High-Sensitivity Pulse Oximeter and Heart-Rate Sensor for "
         "Wearable Health. Data sheet, Rev. 1, 2018. https://www.analog.com/en/products/max30102.html",
         "Sensor specifications: reflective module, red and IR LEDs, 1.8 V and 3.3 V supplies, I²C interface."),
        ("Allen J. Photoplethysmography and its application in clinical physiological measurement. Physiological "
         "Measurement 28(3):R1–R39, 2007. doi:10.1088/0967-3334/28/3/R01",
         "Principles of PPG: pulsatile (AC) and steady (DC) components and their physiological origin."),
        ("Elgendi M. On the analysis of fingertip photoplethysmogram signals. Current Cardiology Reviews 8(1):14–25, "
         "2012. doi:10.2174/157340312801215782",
         "PPG waveform features and derivatives of the kind RakshPulse extracts."),
        ("Stergiou GS, Mukkamala R, Avolio A, et al. Cuffless blood pressure measuring devices: review and statement by "
         "the European Society of Hypertension Working Group on Blood Pressure Monitoring and Cardiovascular "
         "Variability. Journal of Hypertension 40(8):1449–1460, 2022. doi:10.1097/HJH.0000000000003224",
         "Why cuffless devices are not yet recommended for clinical use and the validation issues they raise."),
        ("Stergiou GS, Avolio AP, Palatini P, et al. European Society of Hypertension recommendations for the "
         "validation of cuffless blood pressure measuring devices. Journal of Hypertension 41(12):2074–2087, 2023. "
         "doi:10.1097/HJH.0000000000003483",
         "The validation tests a cuffless device such as RakshPulse would have to pass, including recalibration tests."),
        ("ISO 81060-2:2018. Non-invasive sphygmomanometers, Part 2: Clinical investigation of intermittent automated "
         "measurement type. https://www.iso.org/standard/73339.html",
         "The standard used to validate cuff devices, the reference instruments for data collection."),
        ("Christofferson K, Bhalla S, Kleinknecht T, et al. Auricle: A Multimodal Wearable Sensing Dataset for "
         "Cardiovascular and Hemodynamic Assessment (version 1.0.0). PhysioNet, 2026. doi:10.13026/swme-vp69",
         "The dataset the shipped RakshPulse model was trained and tested on (Chapter 4)."),
        ("Mukkamala R, Hahn J-O, Inan OT, et al. Toward ubiquitous blood pressure monitoring via pulse transit time: "
         "theory and practice. IEEE Transactions on Biomedical Engineering 62(8):1879–1901, 2015. "
         "doi:10.1109/TBME.2015.2441951",
         "Physiological basis and practical limits of cuffless BP estimation from pulse timing."),
        ("Charlton PH, Paliakaitė B, Pilt K, et al. Assessing hemodynamics from the photoplethysmogram to gain insights "
         "into vascular age: a review from VascAgeNet. American Journal of Physiology-Heart and Circulatory Physiology "
         "322(4):H493–H522, 2022. doi:10.1152/ajpheart.00392.2021",
         "How pulse morphology relates to arterial stiffness and blood pressure."),
        ("Jeleazcov C, Krajinovic L, Münster T, et al. Precision and accuracy of a new device (CNAP™) for continuous "
         "non-invasive arterial pressure monitoring: assessment during general anaesthesia. British Journal of "
         "Anaesthesia 105(3):264–272, 2010. doi:10.1093/bja/aeq143",
         "Accuracy of the CNAP reference used in the training dataset, compared with an arterial line."),
        ("González S, Hsieh W-T, Chen TP-C. A benchmark for machine-learning based non-invasive blood pressure "
         "estimation using photoplethysmogram. Scientific Data 10:149, 2023. doi:10.1038/s41597-023-02020-6",
         "Shows why subject-independent evaluation is essential for PPG blood-pressure models."),
        ("Espressif Systems. arduino-esp32, variants/esp32c3/pins_arduino.h. "
         "https://github.com/espressif/arduino-esp32/blob/master/variants/esp32c3/pins_arduino.h",
         "Default I²C pins (SDA 8, SCL 9) for the 'ESP32C3 Dev Module' board setting."),
        ("Espressif Systems. Arduino-ESP32 documentation: USB CDC and DFU flashing. "
         "https://docs.espressif.com/projects/arduino-esp32/en/latest/tutorials/cdc_dfu_flash.html",
         "'USB CDC On Boot' setting for serial output over the ESP32-C3's native USB."),
        ("IEEE Std 1708-2014. IEEE Standard for Wearable Cuffless Blood Pressure Measuring Devices (superseded by "
         "IEEE 1708-2025). https://standards.ieee.org/standard/1708-2014.html",
         "Performance-evaluation standard specifically for cuffless devices."),
    ]
    for i, (ref, why) in enumerate(refs, 1):
        b.reference(f"**[{i}]** {ref}", f"*Relevance:* {why}")
