"""Vector diagrams for the RakshPulse operator manual (reportlab.graphics).

Every diagram explains one concept of the real implementation. Things that do not exist in the repository (an
ESP32 display, on-device blood-pressure inference) are drawn dashed and red and labelled NOT IMPLEMENTED. No GPIO
numbers or supply voltages are drawn: they depend on the board and breakout and are explained in the text.
"""

from __future__ import annotations

import math

import theme as T
from reportlab.graphics.shapes import Circle, Drawing, Line, Path, Polygon, PolyLine, Rect, String
from reportlab.pdfbase.pdfmetrics import stringWidth

W = 480


# ------------------------------------------------------------------------------------------------ primitives
def text(d, x, y, s, size=8.5, font=None, color=T.CHARCOAL, anchor="middle"):
    d.add(String(x, y, s, fontName=font or T.SANS, fontSize=size, fillColor=color, textAnchor=anchor))


def lines(d, cx, cy, s, size=8.5, color=T.CHARCOAL, lead=1.22, anchor="middle"):
    parts = s.split("\n")
    top = cy + (len(parts) - 1) * size * lead / 2 - size * 0.33
    for i, p in enumerate(parts):
        f = T.SANS
        if p.startswith("**"):
            p, f = p[2:], T.SANS_B
        elif p.startswith("``"):
            p, f = p[2:], T.MONO
        text(d, cx, top - i * size * lead, p, size, f, color, anchor)


def box(d, x, y, w, h, label="", fill=T.WHITE, stroke=T.NAVY, tc=T.CHARCOAL, size=8.5, r=6, sw=1.2, dash=None):
    d.add(Rect(x, y, w, h, rx=r, ry=r, fillColor=fill, strokeColor=stroke, strokeWidth=sw, strokeDashArray=dash))
    if label:
        lines(d, x + w / 2, y + h / 2, label, size, color=tc)


def head(d, x1, y1, x2, y2, color, size=6.5):
    a = math.atan2(y2 - y1, x2 - x1)
    p1 = (x2 - size * math.cos(a - 0.42), y2 - size * math.sin(a - 0.42))
    p2 = (x2 - size * math.cos(a + 0.42), y2 - size * math.sin(a + 0.42))
    d.add(Polygon([x2, y2, *p1, *p2], fillColor=color, strokeColor=color, strokeWidth=0.5))


def arrow(d, x1, y1, x2, y2, color=T.NAVY, sw=1.4, dash=None, both=False):
    d.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=sw, strokeDashArray=dash))
    head(d, x1, y1, x2, y2, color)
    if both:
        head(d, x2, y2, x1, y1, color)


def curve(d, x1, y1, x2, y2, bend=0.35, color=T.NAVY, sw=1.5, dash=None, label=None, label_color=None, size=7.5,
          label_at=None):
    """Curved arrow from (x1, y1) to (x2, y2); ``bend`` > 0 bulges to the left of the direction of travel."""
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    dx, dy = x2 - x1, y2 - y1
    cx, cy = mx - dy * bend, my + dx * bend
    p = Path(strokeColor=color, strokeWidth=sw, fillColor=None, strokeDashArray=dash)
    p.moveTo(x1, y1)
    p.curveTo(x1 + (cx - x1) * 0.66, y1 + (cy - y1) * 0.66, x2 + (cx - x2) * 0.66, y2 + (cy - y2) * 0.66, x2, y2)
    d.add(p)
    head(d, x2 + (cx - x2) * 0.2, y2 + (cy - y2) * 0.2, x2, y2, color)
    if label:
        tx, ty = label_at or (0.25 * x1 + 0.5 * cx + 0.25 * x2, 0.25 * y1 + 0.5 * cy + 0.25 * y2)
        w = stringWidth(label, T.SANS_B, size) + 8
        d.add(Rect(tx - w / 2, ty - 3.5, w, size + 5, rx=3, ry=3, fillColor=T.WHITE, strokeColor=None))
        text(d, tx, ty, label, size, T.SANS_B, label_color or color)


def badge(d, x, y, n, color=T.ORANGE, r=9):
    d.add(Circle(x, y, r, fillColor=color, strokeColor=T.WHITE, strokeWidth=1.2))
    text(d, x, y - 3.2, str(n), 8.5, T.SANS_B, T.WHITE)


def zone(d, x, y, w, h, title, color, fill, title_bottom=False):
    d.add(Rect(x, y, w, h, rx=10, ry=10, fillColor=fill, strokeColor=color, strokeWidth=1.2, strokeDashArray=[4, 3]))
    text(d, x + 10, y + 7 if title_bottom else y + h - 14, title, 9, T.SANS_B, color, "start")


def pulse(d, x0, y0, w, h, beats=3, color=T.RED, sw=1.6):
    pts = []
    for i in range(121):
        u = i / 120
        ph = (u * beats) % 1
        v = math.exp(-(((ph - 0.25) / 0.07) ** 2)) + 0.38 * math.exp(-(((ph - 0.5) / 0.09) ** 2))
        pts += [x0 + w * u, y0 + h * v / 1.1]
    d.add(PolyLine(pts, strokeColor=color, strokeWidth=sw))


NOT_IMPL = dict(fill=T.RED_BG, stroke=T.RED, dash=[4, 3])


# ------------------------------------------------------------------------------------------------ diagrams
def big_picture():
    d = Drawing(W, 255)
    zone(d, 0, 20, 200, 225, "HARDWARE (your bench)", T.BLUE, T.BLUE_BG)
    zone(d, 252, 20, 228, 225, "COMPUTER (RakshPulse software)", T.TEAL, T.TEAL_BG)
    box(d, 15, 150, 80, 62, "**MAX30102\nshines red + IR\nlight, measures\nwhat returns")
    box(d, 112, 150, 80, 62, "**ESP32\nreads sensor,\nprints numbers\nover USB")
    box(d, 260, 165, 95, 50, "**Signal processing\nfilter · beats\nquality check", stroke=T.TEAL)
    box(d, 370, 165, 95, 50, "**Features + ML\nmodel → SBP/DBP\nestimate", stroke=T.TEAL)
    box(d, 315, 60, 110, 56, "**Results\nHR, quality, SBP/DBP\n(+ mode label)\nCSV · JSON · PNG", stroke=T.ORANGE,
        fill=T.ORANGE_BG)
    curve(d, 95, 190, 112, 190, 0.0, label="I²C")
    curve(d, 192, 178, 260, 186, -0.3)
    text(d, 226, 205, "USB serial", 7.2, T.SANS_B, T.NAVY)
    text(d, 226, 196, "index,red,ir", 7.2, T.MONO, T.NAVY)
    curve(d, 355, 190, 370, 190, 0.0)
    curve(d, 418, 165, 395, 116, -0.3, color=T.ORANGE)
    box(d, 15, 40, 177, 80, "**What you see on the ESP32 side:\nrising numbers in the Serial\nMonitor / Plotter. That is RAW\n"
        "LIGHT DATA, not blood pressure.", fill=T.WHITE, stroke=T.RED, size=8)
    text(d, W / 2, 6, "Blood-pressure estimates only exist on the computer, after the software has processed the samples.",
         8, T.SANS_I, T.RED)
    return d


def sensor_principle():
    d = Drawing(W, 230)
    d.add(Rect(0, 20, 210, 120, rx=8, ry=8, fillColor=T.ORANGE_BG, strokeColor=T.ORANGE, strokeWidth=1.2))
    text(d, 10, 26, "skin and tissue", 9, T.SANS_B, T.ORANGE, "start")
    d.add(Rect(25, 50, 160, 18, rx=9, ry=9, fillColor=T.RED_BG, strokeColor=T.RED, strokeWidth=0.9))
    text(d, 105, 56, "artery: fills a little more with every heartbeat", 7.5, T.SANS_I, T.RED)
    box(d, 40, 160, 60, 34, "**Red LED\n~660 nm", fill=T.RED_BG, stroke=T.RED, size=8)
    box(d, 110, 160, 60, 34, "**IR LED\n~880 nm", fill=T.BLUE_BG, stroke=T.BLUE, size=8)
    box(d, 40, 202, 130, 22, "**MAX30102 (reflective)", fill=T.GREY_BG, size=8)
    curve(d, 70, 160, 92, 72, 0.2, color=T.RED)
    curve(d, 118, 72, 140, 160, 0.2, color=T.BLUE)
    text(d, 62, 108, "light in", 7.8, T.SANS_B, T.RED, "end")
    text(d, 146, 108, "some light", 7.8, T.SANS_B, T.BLUE, "start")
    text(d, 146, 98, "returns", 7.8, T.SANS_B, T.BLUE, "start")
    box(d, 245, 150, 105, 50, "**Photodetector + ADC\nturns returned light\ninto a number", stroke=T.NAVY, size=8)
    curve(d, 170, 175, 245, 175, -0.25)
    d.add(Rect(245, 30, 235, 100, rx=6, ry=6, fillColor=T.WHITE, strokeColor=T.GREY_LINE))
    pulse(d, 255, 50, 215, 55, beats=3, color=T.NAVY)
    text(d, 362, 115, "the number rises and falls with each heartbeat = PPG", 8, T.SANS_B, T.NAVY)
    text(d, 362, 37, "time  →", 7.5, T.SANS_I, T.MUTED)
    curve(d, 300, 150, 330, 130, 0.3, color=T.NAVY)
    text(d, W / 2, 6, "No cuff, no pressure sensor: the MAX30102 measures LIGHT. Blood pressure has to be estimated "
         "from the light pattern.", 8, T.SANS_I, T.RED)
    return d


def pipeline10():
    d = Drawing(W, 330)
    zone(d, 0, 238, 480, 86, "HARDWARE: what the ESP32 + MAX30102 do", T.BLUE, T.BLUE_BG)
    zone(d, 0, 20, 480, 210, "SOFTWARE: what RakshPulse does on the computer", T.TEAL, T.TEAL_BG, title_bottom=True)
    steps = [
        (25, 252, "**Sensor\nmeasures red/IR\nlight (FIFO)"), (265, 252, "**ESP32 sends\nraw samples over\nUSB serial"),
        (25, 160, "**Serial reader\nchecks lines, banner,\nlost samples"), (140, 160, "**Preprocessing\nremove drift,\nband-pass filter"),
        (255, 160, "**10 s windows\none every 5 s"), (370, 160, "**Quality check\nusable · degraded ·\nunusable · missing"),
        (370, 70, "**Features\nheart rate + pulse\nshape (40 numbers)"), (255, 70, "**Model contract\nsensor, rate, inputs\n→ mode label"),
        (140, 70, "**ML model\nSBP model + DBP\nmodel per window"), (25, 70, "**Results\nmedian SBP/DBP, HR,\nquality → files"),
    ]
    for i, (x, y, s) in enumerate(steps, 1):
        box(d, x, y, 95, 52, s, stroke=T.BLUE if i <= 2 else T.TEAL, size=7.8)
        badge(d, x + 6, y + 50, i)
    curve(d, 120, 278, 265, 278, 0.15)
    curve(d, 312, 252, 72, 212, -0.25, label="USB")
    for a, b in ((2, 3), (3, 4), (4, 5)):
        curve(d, steps[a][0] + 95, 186, steps[b][0], 186, 0.2)
    curve(d, 417, 160, 417, 122, -0.4)
    for a, b in ((6, 7), (7, 8), (8, 9)):
        curve(d, steps[a][0], 96, steps[b][0] + 95, 96, 0.2)
    text(d, 417, 58, "unusable windows get no estimate", 7, T.SANS_I, T.RED)
    text(d, W / 2, 6, "Steps 1–2 happen on your bench. Steps 3–10 only happen when the computer runs RakshPulse.", 8,
         T.SANS_I, T.NAVY)
    return d


def where_results_appear():
    d = Drawing(W, 250)
    box(d, 0, 175, 140, 60, "**1. Arduino Serial Monitor\n/ Serial Plotter", fill=T.BLUE_BG, stroke=T.BLUE)
    box(d, 170, 175, 140, 60, "**2. RakshPulse on the\ncomputer", fill=T.TEAL_BG, stroke=T.TEAL)
    box(d, 340, 175, 140, 60, "**3. OLED / display on\nthe ESP32", **NOT_IMPL)
    box(d, 0, 30, 140, 125, "Shows: raw numbers\n``1234,95321,118204\nand their curves.\n\n**SBP/DBP: NO\n(it never computes them)",
        stroke=T.BLUE, size=8)
    box(d, 170, 30, 140, 125, "Shows: heart rate, signal\nquality, estimated SBP/DBP,\nmode label, latency.\n\n"
        "**SBP/DBP: YES (estimates)\nDEMONSTRATION on MAX30102", stroke=T.TEAL, size=8)
    box(d, 340, 30, 140, 125, "**NOT IMPLEMENTED\n\nNeeds: display code in the\nfirmware, a channel that\nsends results back to the\n"
        "ESP32 (or a model running\non the ESP32 itself).", **NOT_IMPL, size=8)
    for x in (70, 240):
        curve(d, x, 175, x, 155, 0.0)
    curve(d, 410, 175, 410, 155, 0.0, color=T.RED, dash=[3, 2])
    text(d, W / 2, 6, "Seeing values on the ESP32 serial monitor only proves the sensor and firmware work.", 8,
         T.SANS_I, T.NAVY)
    return d


def compatibility_ladder():
    d = Drawing(W, 250)
    rows = [
        ("1. Can the software READ MAX30102 data?", "YES", "serial reader + firmware contract (tested with simulated streams)", T.TEAL),
        ("2. Can the pipeline PROCESS the signal?", "YES", "filtering, beats, quality, features work on ADC counts", T.TEAL),
        ("3. Can the model produce an output?", "YES, DEMO", "contract mode 'demonstration' for max30102", T.ORANGE),
        ("4. Was the model TRAINED on MAX30102 data?", "NO", "trained on earlobe PPG from a different sensor", T.RED),
        ("5. Was it VALIDATED on MAX30102 + cuff?", "NO", "no MAX30102 recordings with reference BP exist yet", T.RED),
    ]
    for i, (q, a, why, col) in enumerate(rows):
        y = 205 - i * 44
        box(d, 0 + i * 12, y, 250, 36, "", fill=T.WHITE, stroke=col)
        text(d, 10 + i * 12, y + 21, q, 8.6, T.SANS_B, T.NAVY, "start")
        text(d, 10 + i * 12, y + 8, why, 7.4, T.SANS_I, T.MUTED, "start")
        d.add(Rect(268 + i * 12, y + 4, 80, 28, rx=14, ry=14, fillColor=col, strokeColor=col))
        text(d, 308 + i * 12, y + 14, a, 9, T.SANS_B, T.WHITE)
        if i < 4:
            curve(d, 255 + i * 12, y + 4, 262 + (i + 1) * 12, y - 6, -0.4, color=T.GREY_LINE, sw=1.1)
    box(d, 380, 30, 100, 190, "**Therefore:\n\nany SBP/DBP\nnumber from a\nMAX30102 is a\n**DEMONSTRATION,\n"
        "not a blood-\npressure\nmeasurement.", fill=T.RED_BG, stroke=T.RED, size=8.4)
    text(d, W / 2, 6, "Answers 1–3 are about software. Answers 4–5 are about evidence, which needs your recordings.",
         8, T.SANS_I, T.NAVY)
    return d


def sensor_comparison():
    d = Drawing(W, 240)
    zone(d, 0, 20, 230, 210, "TRAINING DATA: Auricle (PhysioNet)", T.NAVY_2, T.GREY_BG)
    zone(d, 250, 20, 230, 210, "YOUR DEVICE: MAX30102", T.BLUE, T.BLUE_BG)
    left = ["Nonin PPG clip (exact model not stated)", "TRANSMISSIVE: light passes through", "site: left EARLOBE",
            "wavelengths not documented", "BIOPAC MP160, 10 kHz → stored 1 kHz", "analogue output in mV",
            "reference BP: CNAP finger cuff", "42 healthy adults (40 after cleaning)"]
    right = ["MAX30102 module (Analog Devices)", "REFLECTIVE: LEDs + detector side by side", "site: BEHIND THE EAR (target)",
             "red ~660 nm, IR ~880 nm", "50–3200 sps; 100 Hz in the firmware", "18-bit ADC counts, large DC level",
             "reference BP: your upper-arm cuff", "participants: not yet recorded"]
    for i, (a, b) in enumerate(zip(left, right, strict=True)):
        y = 190 - i * 21
        text(d, 12, y, "•  " + a, 8, T.SANS, T.CHARCOAL, "start")
        text(d, 262, y, "•  " + b, 8, T.SANS, T.CHARCOAL, "start")
    d.add(Circle(240, 125, 13, fillColor=T.WHITE, strokeColor=T.RED, strokeWidth=1.4))
    text(d, 240, 120, "≠", 15, T.SANS_B, T.RED)
    text(d, W / 2, 6, "Same idea (optical pulse), different physics, site, scale and reference: the model must be retrained.",
         8, T.SANS_I, T.RED)
    return d


def wiring():
    d = Drawing(W, 270)
    d.add(Rect(0, 40, 150, 200, rx=8, ry=8, fillColor=T.BLUE_BG, strokeColor=T.BLUE, strokeWidth=1.4))
    text(d, 75, 222, "MAX30102 breakout", 9.5, T.SANS_B, T.BLUE)
    text(d, 75, 210, "(pin labels vary by vendor)", 7, T.SANS_I, T.MUTED)
    d.add(Rect(330, 40, 150, 200, rx=8, ry=8, fillColor=T.GREY_BG, strokeColor=T.NAVY, strokeWidth=1.4))
    text(d, 405, 222, "ESP32 / ESP32-C3 board", 9.5, T.SANS_B, T.NAVY)
    text(d, 405, 210, "(read YOUR board's pinout)", 7, T.SANS_I, T.MUTED)
    wires = [("VIN", "supply pin your breakout's\ndocumentation specifies", T.RED, "power"),
             ("GND", "any GND pin", T.CHARCOAL, "ground"),
             ("SDA", "I²C data: board default,\nor I2C_SDA_PIN in the sketch", T.BLUE, "data"),
             ("SCL", "I²C clock: board default,\nor I2C_SCL_PIN in the sketch", T.TEAL, "clock"),
             ("INT", "not used by the firmware", T.MUTED, "optional")]
    for i, (a, b, col, tagname) in enumerate(wires):
        y = 180 - i * 33
        d.add(Rect(116, y - 7, 34, 14, fillColor=T.WHITE, strokeColor=col, strokeWidth=1))
        text(d, 133, y - 3, a, 7.5, T.SANS_B, col)
        d.add(Rect(330, y - 10, 128, 20, fillColor=T.WHITE, strokeColor=col, strokeWidth=1))
        lines(d, 394, y, b, 6.6, lead=1.1)
        dash = [3, 2] if tagname == "optional" else None
        p = Path(strokeColor=col, strokeWidth=2 if not dash else 1, fillColor=None, strokeDashArray=dash)
        p.moveTo(150, y)
        p.curveTo(220, y + 10 - i * 4, 260, y - 10 + i * 4, 330, y)
        d.add(p)
        text(d, 240, y + 6, tagname, 7, T.SANS_I, col)
    text(d, 240, 22, "Supply voltage, pull-ups and logic level: follow YOUR breakout's documentation.", 7.6, T.SANS_I, T.RED)
    text(d, 240, 10, "I²C address of the MAX30102: 0x57. The firmware stops with an error if it cannot find it.", 7.6,
         T.SANS_I, T.NAVY)
    return d


def firmware_steps():
    d = Drawing(W, 200)
    steps = ["Install\nArduino IDE 2", "Boards Manager:\n'esp32 by\nEspressif Systems'", "Library Manager:\n'SparkFun MAX3010x'",
             "Open\nrakshpulse_\nmax30102.ino", "Tools → Board\n(e.g. ESP32C3\nDev Module)",
             "ESP32-C3 native USB:\nUSB CDC On Boot\n= Enabled", "Tools → Port,\nUpload", "Serial Monitor\n115200 baud,\nthen CLOSE it"]
    w, h, gx = 104, 56, 21
    for i, s in enumerate(steps):
        row, col = divmod(i, 4)
        col = col if row == 0 else 3 - col
        x, y = col * (w + gx), 120 - row * 95
        fill = T.ORANGE_BG if i == 5 else (T.TEAL_BG if i == 7 else T.WHITE)
        box(d, x, y, w, h, s, fill=fill, stroke=T.ORANGE if i == 5 else T.NAVY, size=7.6)
        badge(d, x + 6, y + h - 2, i + 1)
    for i in range(3):
        curve(d, i * (w + gx) + w, 148, (i + 1) * (w + gx), 148, 0.25)
        curve(d, (3 - i) * (w + gx), 53, (2 - i) * (w + gx) + w, 53, 0.25)
    curve(d, 3 * (w + gx) + w / 2, 120, 3 * (w + gx) + w / 2, 81, -0.6)
    text(d, W / 2, 4, "The sketch is a reference: the software team could not compile or run it on a board (none available).",
         7.8, T.SANS_I, T.RED)
    return d


def session_timeline():
    d = Drawing(W, 210)
    x0, x1, y = 25, 465, 120
    total = 20
    d.add(Rect(x0, y, x1 - x0, 22, fillColor=T.BLUE_BG, strokeColor=T.BLUE))
    text(d, (x0 + x1) / 2, y + 7, "PPG recording running: rakshpulse record (example: 20 min session)", 8, T.SANS_B, T.BLUE)
    for m in range(0, total + 1, 5):
        xx = x0 + (x1 - x0) * m / total
        d.add(Line(xx, y - 4, xx, y, strokeColor=T.MUTED))
        text(d, xx, y - 14, f"{m} min", 7, T.SANS, T.MUTED)
    x5 = x0 + (x1 - x0) * 5 / total
    d.add(Line(x0, y - 24, x5, y - 24, strokeColor=T.TEAL, strokeWidth=1.4))
    d.add(Line(x0, y - 28, x0, y - 20, strokeColor=T.TEAL, strokeWidth=1.4))
    d.add(Line(x5, y - 28, x5, y - 20, strokeColor=T.TEAL, strokeWidth=1.4))
    text(d, (x0 + x5) / 2, y - 36, "seated rest, no talking", 7.2, T.SANS_I, T.TEAL)
    for k, m in enumerate((1, 3, 5, 10, 15, 20)):
        xx = x0 + (x1 - x0) * m / total
        d.add(Circle(xx, y + 52, 10, fillColor=T.ORANGE, strokeColor=T.WHITE))
        text(d, xx, y + 49, f"C{k + 1}", 7.5, T.SANS_B, T.WHITE)
        curve(d, xx, y + 42, xx, y + 24, 0.0, color=T.ORANGE)
    text(d, x0, 198, "Cuff readings (C) at 1, 3 and 5 min of rest, then every 5 min: at least 3, at least 60 s apart.",
         8, T.SANS_B, T.ORANGE, "start")
    text(d, x0, 185, "Write down the TIME (seconds since the recording started) when the cuff shows its result.",
         8, T.SANS, T.CHARCOAL, "start")
    box(d, 0, 8, 480, 52, "", fill=T.TERM_BG, stroke=T.TERM_BG)
    text(d, 10, 42, "rakshpulse reference --dataset data/rakshpulse_hw --subject P001 --session D1 --time 60 --sbp <S> --dbp <D>",
         7.4, T.MONO, T.TERM_FG, "start")
    text(d, 10, 26, "rakshpulse reference ... --time 180 ...      rakshpulse reference ... --time 300 ...", 7.4, T.MONO,
         T.TERM_FG, "start")
    text(d, 470, 12, "one command per cuff reading, after the recording", 7, T.SANS_I, T.TERM_DIM, "end")
    return d


def cuff_matching():
    d = Drawing(W, 170)
    x0, y = 20, 80
    for k in range(9):
        xs = x0 + k * 45
        d.add(Rect(xs, y + (k % 2) * 14, 90, 11, fillColor=T.TEAL_BG if k not in (3,) else T.RED_BG,
                   strokeColor=T.TEAL if k != 3 else T.RED, strokeWidth=0.8))
    text(d, x0, y + 34, "10 s windows (one every 5 s); red = rejected by the quality check", 7.5, T.SANS_I, T.MUTED,
         "start")
    cuff = x0 + 200
    d.add(Line(cuff, y - 30, cuff, y + 50, strokeColor=T.ORANGE, strokeWidth=2))
    text(d, cuff, y + 55, "cuff reading at t = 200 s", 8, T.SANS_B, T.ORANGE)
    d.add(Rect(cuff - 60, y - 22, 120, 12, fillColor=T.ORANGE_BG, strokeColor=T.ORANGE, strokeDashArray=[2, 2]))
    text(d, cuff, y - 19, "± 30 s tolerance", 7.5, T.SANS_B, T.ORANGE)
    curve(d, cuff - 60, y - 30, cuff - 90, y - 50, 0.3, color=T.ORANGE)
    text(d, x0, 12, "A reading labels only accepted windows of the SAME session within the tolerance; the nearest wins. "
         "A wrong time pairs it with the wrong windows.", 7.6, T.SANS_I, T.NAVY, "start")
    return d


def no_bp_flowchart():
    d = Drawing(W, 330)
    q = [(170, 285, "Did you run a RakshPulse\ncommand on the computer?"),
         (170, 205, "Did 'check-serial' say PASS?"),
         (170, 125, "Did enough windows pass\nthe quality check?"),
         (170, 45, "Was the model allowed\n(contract / mode)?")]
    for x, y, s in q:
        box(d, x, y, 150, 40, s, fill=T.BLUE_BG, stroke=T.BLUE, size=8)
    ans = [(345, 285, "No → the ESP32 and Serial Monitor\nnever show SBP/DBP. Run\n`estimate` or `analyze` (Ch. 7, 8)"),
           (345, 205, "No → fix port, baud, --fs, wiring\nfirst (Troubleshooting, Ch. 12)"),
           (345, 125, "No → status 'poor_signal':\nimprove contact, stay still,\ntune LED_POWER"),
           (345, 45, "No → status 'incompatible_model':\ncheck --fs (50–1000 Hz) and\n`rakshpulse models`")]
    for x, y, s in ans:
        box(d, x, y, 135, 40, s.replace("`", ""), fill=T.RED_BG, stroke=T.RED, size=7.2)
    for i, (x, y, _) in enumerate(q):
        curve(d, x + 150, y + 20, 345, y + 20, 0.0, color=T.RED, label="no", label_color=T.RED)
        if i < 3:
            curve(d, x + 75, y, x + 75, q[i + 1][1] + 40, 0.0, color=T.TEAL, label="yes", label_color=T.TEAL)
    box(d, 0, 125, 150, 120, "**You see SBP/DBP.\n\nOn MAX30102 data they\nare labelled\n'demonstration': the\n"
        "pipeline works, but the\nnumbers are NOT blood-\npressure measurements.", fill=T.TEAL_BG, stroke=T.TEAL, size=7.8)
    curve(d, 170, 65, 75, 125, -0.35, color=T.TEAL, label="yes", label_color=T.TEAL)
    text(d, 75, 300, "START: 'I don't see", 8.5, T.SANS_B, T.NAVY)
    text(d, 75, 288, "SYS / DIA anywhere'", 8.5, T.SANS_B, T.NAVY)
    curve(d, 120, 295, 170, 305, -0.2)
    return d


def oled_requirements():
    d = Drawing(W, 200)
    box(d, 0, 120, 110, 60, "**MAX30102\n+ ESP32\n(implemented)", fill=T.BLUE_BG, stroke=T.BLUE)
    box(d, 185, 120, 110, 60, "**Computer\nRakshPulse estimate\n(implemented)", fill=T.TEAL_BG, stroke=T.TEAL)
    box(d, 370, 120, 110, 60, "**OLED on the ESP32\nSYS / DIA / HR", **NOT_IMPL)
    curve(d, 110, 160, 185, 160, -0.25, label="samples (exists)")
    curve(d, 295, 140, 370, 140, 0.0, color=T.RED, dash=[3, 2])
    text(d, 332, 157, "results back", 7.5, T.SANS_B, T.RED)
    text(d, 332, 147, "(missing)", 7.5, T.SANS_B, T.RED)
    box(d, 0, 12, 480, 90, "", fill=T.WHITE, stroke=T.RED, dash=[4, 3])
    text(d, 12, 86, "What would have to be built (none of this exists in the repository):", 8.8, T.SANS_B, T.RED, "start")
    for i, s in enumerate(["A. A results channel: RakshPulse sends e.g. 'SYS=118,DIA=76,HR=72,MODE=demo' back over serial,",
                           "    and the firmware parses it (or Wi-Fi/BLE instead of USB).",
                           "B. Display code in the firmware (OLED library, layout, the DEMO label kept visible).",
                           "C. OR an on-device model: port preprocessing, features and the model to the ESP32 (large job).",
                           "Either way the numbers stay DEMONSTRATION values until a MAX30102 model is trained and validated."]):
        text(d, 12, 70 - i * 13, s, 7.8, T.SANS, T.CHARCOAL if i < 4 else T.RED, "start")
    return d


def error_meaning():
    d = Drawing(W, 175)
    x0, x1, y = 30, 450, 90

    def xv(v):
        return x0 + (x1 - x0) * (v - 90) / 70

    d.add(Line(x0, y, x1, y, strokeColor=T.CHARCOAL, strokeWidth=1))
    for v in range(90, 161, 10):
        d.add(Line(xv(v), y - 3, xv(v), y + 3, strokeColor=T.CHARCOAL))
        text(d, xv(v), y - 14, f"{v}", 7.5)
    text(d, x1 + 12, y - 3, "mmHg", 7.5, T.SANS_I, T.MUTED, "start")
    d.add(Rect(xv(115), y + 8, xv(135) - xv(115), 22, fillColor=T.ORANGE_BG, strokeColor=T.ORANGE))
    text(d, xv(125), y + 15, "125 ± 10", 8, T.SANS_B, T.ORANGE)
    text(d, xv(135) + 8, y + 15, "an estimate of 125 with a 10 mmHg error either way", 7.6, T.SANS_B, T.ORANGE, "start")
    d.add(Rect(xv(110.5), y + 38, xv(139.5) - xv(110.5), 22, fillColor=T.RED_BG, strokeColor=T.RED))
    text(d, xv(125), y + 45, "125 ± 14.5", 8, T.SANS_B, T.RED)
    text(d, xv(139.5) + 8, y + 51, "this model on unseen people:", 7.6, T.SANS_B, T.RED, "start")
    text(d, xv(139.5) + 8, y + 41, "average SBP error 14.5 mmHg", 7.6, T.SANS_B, T.RED, "start")
    d.add(Circle(xv(125), y, 4, fillColor=T.NAVY, strokeColor=None))
    text(d, W / 2, 30, "A ±10 mmHg error is the difference between 115 and 135 for the same person: too large to tell a "
         "normal reading", 7.8, T.SANS_I, T.NAVY)
    text(d, W / 2, 18, "from a raised one. For comparison, cuff monitors must show a mean error within ±5 mmHg (SD at most "
         "8 mmHg) in validation [6].", 7.8, T.SANS_I, T.NAVY)
    return d


def presentation_flow():
    d = Drawing(W, 150)
    steps = ["Problem", "What is\nPPG", "MAX30102", "ESP32's\nrole", "Live PPG\ngraph", "Heartbeat\ndetection",
             "ML\npipeline", "How SBP/DBP\nare estimated", "Real\noutput", "Limitations", "Next step"]
    w = 40
    for i, s in enumerate(steps):
        row, col = divmod(i, 6)
        x = col * 80 if row == 0 else (5 - col) * 80
        y = 85 if row == 0 else 15
        box(d, x, y, w + 26, 44, s, fill=T.RED_BG if s == "Limitations" else T.WHITE,
            stroke=T.RED if s == "Limitations" else T.NAVY, size=7.4)
        badge(d, x + 6, y + 42, i + 1, r=8)
    for i in range(5):
        curve(d, i * 80 + 66, 107, (i + 1) * 80, 107, 0.2)
    curve(d, 5 * 80 + 33, 85, 5 * 80 + 33, 59, -0.5)
    for k in range(4):
        curve(d, (5 - k) * 80, 37, (4 - k) * 80 + 66, 37, 0.2)
    return d


ALL = {
    "big_picture": (big_picture, "The whole system on one page",
                    "Hardware produces light measurements; only the computer software turns them into heart rate and "
                    "blood-pressure estimates."),
    "sensor": (sensor_principle, "What the MAX30102 actually measures",
               "Two LEDs shine into the skin; the photodetector measures the light that comes back. Each heartbeat "
               "changes the amount of blood, so the number rises and falls: that curve is the PPG."),
    "pipeline": (pipeline10, "Where systolic and diastolic estimates come from: 10 stages",
                 "Only stages 1–2 happen on the hardware. Every blood-pressure estimate is produced in stages 3–10 on "
                 "the computer."),
    "where": (where_results_appear, "Three different places a number could appear",
              "Raw values (1) are not blood pressure. Estimates appear only in RakshPulse (2). An ESP32 display (3) "
              "does not exist yet."),
    "ladder": (compatibility_ladder, "Five different questions about MAX30102 compatibility",
               "The software side is ready (1–3). The evidence side (4–5) does not exist until MAX30102 recordings "
               "with cuff readings are collected and a model is trained and tested on them."),
    "sensors": (sensor_comparison, "Training sensor versus your sensor",
                "Sources: the Auricle dataset description on PhysioNet and the MAX30102 datasheet."),
    "wiring": (wiring, "Which signals connect (not which pin numbers)",
               "Connect supply, ground, SDA and SCL. Pin numbers and supply voltage come from your board's and breakout's "
               "documentation."),
    "firmware": (firmware_steps, "Uploading the reference firmware",
                 "Eight steps in order. Step 6 matters for ESP32-C3 boards that use native USB."),
    "timeline": (session_timeline, "One recording session with cuff readings",
                 "Timing from docs/DATA_COLLECTION_PROTOCOL.md in the repository."),
    "matching": (cuff_matching, "How a cuff reading is paired with PPG windows",
                 "Rule implemented in datastore/association.py (default tolerance 30 s)."),
    "flowchart": (no_bp_flowchart, "Troubleshooting flowchart: 'I don't see SYS / DIA'",
                  "Follow the questions from the top. Each red box names the real status or command involved."),
    "oled": (oled_requirements, "Why your ESP32 is not showing SYS and DIA: what is missing",
             "The dashed red parts are NOT implemented anywhere in the repository."),
    "error": (error_meaning, "What an error of 10 mmHg means in practice",
              "Illustrative scale. The 14.5 mmHg is this project's real nested cross-validation result (Chapter 11)."),
    "present": (presentation_flow, "An 11-step way to present RakshPulse", "Chapter 14 has the matching script."),
}
