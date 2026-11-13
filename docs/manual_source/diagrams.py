"""Original vector diagrams for the RakshPulse hardware integration manual (reportlab.graphics).

Every function returns a ``Drawing`` that is embedded in the PDF as vector graphics. No diagram shows a specific GPIO
number, supply voltage or pull-up value: those depend on the board and breakout the hardware team uses, and the
manual tells them where to look them up.
"""

from __future__ import annotations

import math

import theme as T
from reportlab.graphics.shapes import Circle, Drawing, Group, Line, Polygon, PolyLine, Rect, String
from reportlab.pdfbase.pdfmetrics import stringWidth

W = 480  # drawing width in points (fits the 170 mm text column)


# ------------------------------------------------------------------------------------------------ primitives
def text(d, x, y, s, size=8.5, font=None, color=T.CHARCOAL, anchor="middle"):
    d.add(String(x, y, s, fontName=font or T.SANS, fontSize=size, fillColor=color, textAnchor=anchor))


def lines(d, cx, cy, s, size=8.5, font=None, color=T.CHARCOAL, lead=1.22, anchor="middle"):
    """Multi-line text centred vertically on cy."""
    parts = s.split("\n")
    top = cy + (len(parts) - 1) * size * lead / 2 - size * 0.33
    for i, p in enumerate(parts):
        f = font
        if p.startswith("**"):
            p, f = p[2:], T.SANS_B
        elif p.startswith("``"):
            p, f = p[2:], T.MONO
        text(d, cx, top - i * size * lead, p, size, f, color, anchor)


def box(d, x, y, w, h, label="", fill=T.WHITE, stroke=T.NAVY, tc=T.CHARCOAL, size=8.5, r=5, sw=1.1, dash=None):
    d.add(Rect(x, y, w, h, rx=r, ry=r, fillColor=fill, strokeColor=stroke, strokeWidth=sw, strokeDashArray=dash))
    if label:
        lines(d, x + w / 2, y + h / 2, label, size, color=tc)


def header_box(d, x, y, w, h, title, body="", color=T.NAVY, fill=T.WHITE, size=8, hh=16):
    """Box with a coloured title strip."""
    d.add(Rect(x, y, w, h, rx=4, ry=4, fillColor=fill, strokeColor=color, strokeWidth=1.1))
    d.add(Rect(x, y + h - hh, w, hh, rx=4, ry=4, fillColor=color, strokeColor=color, strokeWidth=1.1))
    d.add(Rect(x, y + h - hh, w, hh / 2, fillColor=color, strokeColor=color, strokeWidth=0))
    text(d, x + w / 2, y + h - hh + 4.6, title, size + 0.5, T.SANS_B, T.WHITE)
    if body:
        lines(d, x + w / 2, y + (h - hh) / 2, body, size)


def arrow(d, x1, y1, x2, y2, color=T.NAVY, sw=1.3, head=6, dash=None, both=False):
    d.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=sw, strokeDashArray=dash))
    _head(d, x1, y1, x2, y2, color, head)
    if both:
        _head(d, x2, y2, x1, y1, color, head)


def _head(d, x1, y1, x2, y2, color, head):
    a = math.atan2(y2 - y1, x2 - x1)
    p1 = (x2 - head * math.cos(a - 0.42), y2 - head * math.sin(a - 0.42))
    p2 = (x2 - head * math.cos(a + 0.42), y2 - head * math.sin(a + 0.42))
    d.add(Polygon([x2, y2, *p1, *p2], fillColor=color, strokeColor=color, strokeWidth=0.5))


def elbow(d, pts, color=T.NAVY, sw=1.3, dash=None):
    """Poly-line arrow through points; head on the last segment."""
    flat = [c for p in pts for c in p]
    d.add(PolyLine(flat, strokeColor=color, strokeWidth=sw, strokeDashArray=dash))
    _head(d, *pts[-2], *pts[-1], color, 6)


def label_on(d, x, y, s, size=7.5, color=T.MUTED, bg=T.WHITE, font=None):
    w = stringWidth(s, font or T.SANS_I, size) + 6
    d.add(Rect(x - w / 2, y - 3, w, size + 4, fillColor=bg, strokeColor=None, strokeWidth=0))
    text(d, x, y, s, size, font or T.SANS_I, color)


def badge(d, x, y, n, color=T.ORANGE, r=8):
    d.add(Circle(x, y, r, fillColor=color, strokeColor=T.WHITE, strokeWidth=1))
    text(d, x, y - 3, str(n), 8, T.SANS_B, T.WHITE)


def flow(d, labels, y, h=40, x0=0, x1=W, gap=14, fills=None, size=8, color=T.NAVY):
    """A row of boxes joined by arrows; returns the box centres."""
    n = len(labels)
    w = (x1 - x0 - gap * (n - 1)) / n
    centres = []
    for i, s in enumerate(labels):
        x = x0 + i * (w + gap)
        f = fills[i] if fills else T.WHITE
        box(d, x, y, w, h, s, fill=f, stroke=color, size=size)
        centres.append((x + w / 2, y + h / 2))
        if i:
            arrow(d, x - gap + 1, y + h / 2, x - 1, y + h / 2, color)
    return centres, w


# ------------------------------------------------------------------------------------------------ 1 overview
def system_overview():
    d = Drawing(W, 200)
    xs = [0, 104, 214, 334, 414]
    ws = [76, 82, 92, 68, 66]
    labs = [
        "**MAX30102\nred + IR\nPPG sensor",
        "**ESP32 /\n**Arduino\nreference firmware",
        "**PC: rakshpulse\ncheck-serial\nrecord / analyze",
        "**Dataset\nCSV session\nfolder",
        "**Results\nquality +\ndemo output",
    ]
    fills = [T.BLUE_BG, T.BLUE_BG, T.TEAL_BG, T.GREY_BG, T.ORANGE_BG]
    strokes = [T.BLUE, T.BLUE, T.TEAL, T.MUTED, T.ORANGE]
    y, h = 105, 62
    for x, w, s, f, st in zip(xs, ws, labs, fills, strokes):
        box(d, x, y, w, h, s, fill=f, stroke=st, size=8.5)
    tags = ["I²C", "USB", "CSV", ""]
    for i in range(4):
        a, b = xs[i] + ws[i], xs[i + 1]
        arrow(d, a + 2, y + h / 2, b - 2, y + h / 2)
        text(d, (a + b) / 2, y + h / 2 + 6, tags[i], 7, T.SANS_I, T.MUTED)
    # cuff path
    box(d, 214, 18, 92, 44, "**Upper-arm cuff\nreference readings", fill=T.ORANGE_BG, stroke=T.ORANGE)
    elbow(d, [(306, 40), (368, 40), (368, 103)], color=T.ORANGE)
    text(d, 372, 70, "rakshpulse", 7, T.SANS_I, T.ORANGE, "start")
    text(d, 372, 61, "reference", 7, T.SANS_I, T.ORANGE, "start")
    box(
        d,
        0,
        18,
        180,
        44,
        "Participant: sensor at the measurement\nsite, cuff on the upper arm",
        fill=T.WHITE,
        stroke=T.MUTED,
        dash=[3, 2],
    )
    arrow(d, 40, 62, 40, 103, color=T.MUTED, dash=[3, 2])
    arrow(d, 180, 40, 212, 40, color=T.MUTED, dash=[3, 2])
    text(d, W / 2, 185, "Hardware team  ←  handoff point  →  Software / ML team", 8, T.SANS_I, T.MUTED)
    d.add(Line(193, 175, 193, 100, strokeColor=T.MUTED, strokeWidth=0.8, strokeDashArray=[2, 2]))
    return d


# ------------------------------------------------------------------------------------------------ 2 MAX30102
def max30102_block():
    d = Drawing(W, 235)
    # skin / tissue
    d.add(Rect(0, 10, 150, 215, fillColor=T.ORANGE_BG, strokeColor=T.ORANGE, strokeWidth=1, rx=6, ry=6))
    text(d, 75, 208, "Skin and tissue", 9, T.SANS_B, T.ORANGE)
    for i, yy in enumerate((60, 110, 160)):
        d.add(Rect(20, yy, 110, 14, rx=7, ry=7, fillColor=T.RED_BG, strokeColor=T.RED, strokeWidth=0.8))
        text(d, 75, yy + 4, "blood vessel (volume pulses)" if i == 1 else "", 7, T.SANS_I, T.RED)
    text(d, 75, 30, "light is partly absorbed;", 7.5, T.SANS_I, T.CHARCOAL)
    text(d, 75, 20, "the rest scatters back", 7.5, T.SANS_I, T.CHARCOAL)
    # chip
    d.add(Rect(185, 10, 175, 215, rx=8, ry=8, fillColor=T.GREY_BG, strokeColor=T.NAVY, strokeWidth=1.4))
    text(d, 272, 208, "MAX30102 module", 9.5, T.SANS_B, T.NAVY)
    box(d, 200, 160, 70, 34, "**Red LED\n~660 nm", fill=T.RED_BG, stroke=T.RED, size=8)
    box(d, 278, 160, 70, 34, "**IR LED\n~880 nm", fill=T.BLUE_BG, stroke=T.BLUE, size=8)
    box(d, 200, 112, 148, 32, "**Photodetector\nmeasures returned light", fill=T.WHITE, size=8)
    box(d, 200, 66, 148, 32, "**18-bit ADC + ambient\nlight cancellation", fill=T.WHITE, size=8)
    box(d, 200, 22, 148, 30, "**FIFO (32 samples)\nholds samples until read", fill=T.TEAL_BG, stroke=T.TEAL, size=8)
    arrow(d, 274, 112, 274, 100)
    arrow(d, 274, 66, 274, 54)
    # light path
    arrow(d, 200, 177, 132, 150, color=T.RED, sw=1.4)
    arrow(d, 132, 130, 200, 128, color=T.BLUE, sw=1.4)
    label_on(d, 166, 172, "emit", bg=T.WHITE)
    label_on(d, 166, 119, "return", bg=T.WHITE)
    # I2C
    box(
        d,
        395,
        70,
        85,
        110,
        "**I²C interface\n\naddress 0x57\nSDA (data)\nSCL (clock)\n+ supply, GND\n(INT optional)",
        fill=T.BLUE_BG,
        stroke=T.BLUE,
        size=8,
    )
    arrow(d, 348, 37, 380, 37)
    elbow(d, [(380, 37), (437, 37), (437, 68)])
    arrow(d, 437, 180, 437, 200, color=T.BLUE)
    text(d, 437, 205, "to the ESP32 / Arduino", 7.5, T.SANS_I, T.MUTED)
    return d


# ------------------------------------------------------------------------------------------------ 3 ESP32
def esp32_block():
    d = Drawing(W, 225)
    d.translate(0, 10)
    box(d, 0, 80, 95, 60, "**MAX30102\nbreakout", fill=T.BLUE_BG, stroke=T.BLUE)
    d.add(Rect(130, 15, 220, 190, rx=8, ry=8, fillColor=T.GREY_BG, strokeColor=T.NAVY, strokeWidth=1.4))
    text(d, 240, 190, "ESP32 / Arduino development board", 9.5, T.SANS_B, T.NAVY)
    box(d, 145, 110, 90, 60, "**I²C controller\nWire library\nSDA / SCL pins", fill=T.WHITE, size=8)
    box(d, 245, 110, 92, 60, "**CPU runs the\n**sketch\nreads FIFO,\nformats lines", fill=T.WHITE, size=8)
    box(
        d,
        145,
        30,
        192,
        60,
        "**USB-to-serial (CP210x / CH340 bridge\n**or native USB, board dependent)\n115200 baud, text lines",
        fill=T.TEAL_BG,
        stroke=T.TEAL,
        size=8,
    )
    arrow(d, 97, 110, 143, 140, both=True)
    label_on(d, 112, 135, "I²C", bg=T.WHITE)
    arrow(d, 235, 140, 243, 140)
    arrow(d, 291, 110, 291, 92)
    box(d, 390, 30, 90, 60, "**PC\nrakshpulse\n(COMx / ttyUSBx)", fill=T.TEAL_BG, stroke=T.TEAL)
    arrow(d, 337, 60, 388, 60, color=T.TEAL)
    label_on(d, 363, 67, "USB cable")
    box(
        d,
        390,
        120,
        90,
        50,
        "Power comes\nfrom USB while\ndeveloping",
        fill=T.WHITE,
        stroke=T.MUTED,
        dash=[3, 2],
        size=7.5,
    )
    arrow(d, 435, 92, 435, 118, color=T.MUTED, dash=[3, 2])
    text(
        d,
        W / 2,
        -6,
        "Pin names and numbers: read your board's pinout (no pin numbers are assumed)",
        7.6,
        T.SANS_I,
        T.RED,
    )
    return d


# ------------------------------------------------------------------------------------------------ 4 PPG waveform
def _ppg(t, hr=72):
    period = 60.0 / hr
    ph = (t % period) / period
    sys_ = math.exp(-(((ph - 0.18) / 0.075) ** 2))
    dic = 0.42 * math.exp(-(((ph - 0.48) / 0.10) ** 2))
    return sys_ + dic


def ppg_waveform():
    d = Drawing(W, 230)
    x0, y0, w, h = 40, 50, 420, 130
    d.add(Rect(x0, y0, w, h, fillColor=T.WHITE, strokeColor=T.GREY_LINE, strokeWidth=0.8))
    for k in range(1, 3):
        d.add(Line(x0 + k * w / 3, y0, x0 + k * w / 3, y0 + h, strokeColor=T.GREY_LINE, strokeWidth=0.4))
    dur = 3.0
    pts = []
    for i in range(400):
        t = dur * i / 399
        v = _ppg(t) / 1.15
        pts += [x0 + w * t / dur, y0 + 22 + v * (h - 40)]
    d.add(PolyLine(pts, strokeColor=T.RED, strokeWidth=1.6))
    # annotations (first beat)
    period = 60 / 72
    tp = 0.18 * period
    xp = x0 + w * tp / dur
    yp = y0 + 22 + _ppg(tp) / 1.15 * (h - 40)
    d.add(Circle(xp, yp, 2.5, fillColor=T.NAVY, strokeColor=None))
    text(d, xp + 4, yp + 6, "systolic peak", 7.5, T.SANS_B, T.NAVY, "start")
    tn = 0.40 * period
    xn = x0 + w * tn / dur
    yn = y0 + 22 + _ppg(tn) / 1.15 * (h - 40)
    arrow(d, xn + 4, y0 + 10, xn + 1, yn - 3, sw=0.8, head=4)
    text(d, xn + 8, y0 + 6, "notch / reflected wave", 7.5, T.SANS_I, T.NAVY, "start")
    # beat interval
    xa, xb = xp, x0 + w * (tp + period) / dur
    yb = y0 + h - 6
    arrow(d, xa, yb, xb, yb, sw=0.8, head=4, both=True)
    label_on(d, (xa + xb) / 2, yb - 3, "beat interval → heart rate", color=T.NAVY)
    # AC / DC
    d.add(Line(x0 - 8, y0 + 22, x0 - 8, y0 + h - 18, strokeColor=T.TEAL, strokeWidth=1.2))
    text(d, x0 - 12, y0 + 70, "AC", 8, T.SANS_B, T.TEAL, "end")
    text(d, x0 - 12, y0 + 60, "pulsatile", 6.5, T.SANS_I, T.TEAL, "end")
    d.add(Rect(x0, y0 - 28, w, 22, fillColor=T.BLUE_BG, strokeColor=T.BLUE, strokeWidth=0.8))
    text(
        d,
        x0 + w / 2,
        y0 - 20,
        "DC level: steady absorption by tissue, bone and non-pulsing blood (much larger than AC)",
        7.5,
        T.SANS_I,
        T.BLUE,
    )
    for k in range(4):
        text(d, x0 + k * w / 3, y0 - 38, f"{k} s", 7, T.SANS, T.MUTED)
    text(
        d,
        x0 + w / 2,
        215,
        "Illustrative PPG shape (not recorded data); raw values arrive inverted or offset depending on the sensor",
        7.5,
        T.SANS_I,
        T.MUTED,
    )
    return d


# ------------------------------------------------------------------------------------------------ 5 ML pipeline
def ml_pipeline():
    d = Drawing(W, 270)
    row1 = [
        "**Raw samples\nred + IR counts\nat --fs Hz",
        "**Clean up\nfill ≤2-sample gaps\nresample 100 Hz",
        "**Filter\nband-pass\n0.5–10 Hz",
        "**Windows\n10 s long,\nevery 5 s",
    ]
    c1, w1 = flow(d, row1, 200, h=52, fills=[T.BLUE_BG] * 4, color=T.BLUE)
    row2 = [
        "**Quality gate\nusable / degraded /\nunusable / missing",
        "**Features\nheart rate + pulse\nshape (28 values)",
        "**Contract check\nsensor, rate, inputs\n→ mode",
        "**Model\nridge regression\n(demonstration)",
    ]
    c2, w2 = flow(d, row2, 115, h=52, fills=[T.TEAL_BG] * 4, color=T.TEAL)
    elbow(d, [(c1[-1][0], 200), (c1[-1][0], 186), (c2[0][0], 186), (c2[0][0], 169)], color=T.NAVY)
    box(
        d,
        0,
        18,
        225,
        60,
        "**Output per window\nstatus, reasons, quality score,\nSBP/DBP numbers + inference_mode",
        fill=T.ORANGE_BG,
        stroke=T.ORANGE,
    )
    box(
        d,
        255,
        18,
        225,
        60,
        "**Cuff comparison (analyze)\nreadings matched to windows,\noptional calibration with first N readings",
        fill=T.ORANGE_BG,
        stroke=T.ORANGE,
    )
    elbow(d, [(c2[-1][0], 115), (c2[-1][0], 96), (112, 96), (112, 80)], color=T.NAVY)
    arrow(d, 227, 48, 253, 48, color=T.ORANGE)
    text(d, W / 2, 258, "Signal processing (no learning)", 8, T.SANS_B, T.BLUE)
    text(d, W / 2, 176, "Quality, features and the trained model", 8, T.SANS_B, T.TEAL)
    label_on(d, 112 + 60, 99, "unusable windows get no number", size=7, color=T.RED)
    return d


# ------------------------------------------------------------------------------------------------ 6 architecture
def architecture():
    d = Drawing(W, 330)
    layers = [
        (
            262,
            "HARDWARE",
            T.BLUE,
            T.BLUE_BG,
            ["MAX30102\n(red + IR LEDs,\nphotodetector)", "ESP32 / Arduino\nreference sketch", "Power and USB\ncable"],
        ),
        (
            190,
            "COMMUNICATION",
            T.NAVY_2,
            T.GREY_BG,
            [
                "I²C, address 0x57\n(sensor → board)",
                "USB serial 115200\n`index,red,ir` lines",
                "Banner\n# rakshpulse fw=1 fs_hz=…",
            ],
        ),
        (
            118,
            "SOFTWARE",
            T.TEAL,
            T.TEAL_BG,
            [
                "acquisition\nSerialSource,\nTimestampSync",
                "datastore\nsession folders,\nchecksums",
                "preprocessing\nwindows, quality,\nfeatures",
            ],
        ),
        (
            46,
            "MACHINE LEARNING",
            T.ORANGE,
            T.ORANGE_BG,
            [
                "model contract\nsensors, rates,\nrequired inputs",
                "inference engine\nmode label on\nevery result",
                "calibration +\ncomparison\n(analyze)",
            ],
        ),
    ]
    for y, name, col, bg, items in layers:
        d.add(Rect(0, y, 372, 62, rx=6, ry=6, fillColor=bg, strokeColor=col, strokeWidth=1.1))
        d.add(Rect(0, y, 22, 62, rx=6, ry=6, fillColor=col, strokeColor=col))
        d.add(Rect(12, y, 10, 62, fillColor=col, strokeColor=col))
        g = Group(String(0, 0, name, fontName=T.SANS_B, fontSize=7, fillColor=T.WHITE, textAnchor="middle"))
        g.transform = (0, 1, -1, 0, 14, y + 31)
        d.add(g)
        for i, s in enumerate(items):
            box(d, 30 + i * 114, y + 6, 106, 50, s.replace("`", ""), fill=T.WHITE, stroke=col, size=7.6)
    for y in (252, 180, 108):
        arrow(d, 186, y + 8, 186, y - 2, color=T.CHARCOAL)
    # reference path
    d.add(
        Rect(
            392,
            46,
            88,
            278,
            rx=6,
            ry=6,
            fillColor=T.WHITE,
            strokeColor=T.ORANGE,
            strokeWidth=1.1,
            strokeDashArray=[4, 2],
        )
    )
    text(d, 436, 310, "REFERENCE PATH", 7.5, T.SANS_B, T.ORANGE)
    box(d, 400, 240, 72, 50, "Upper-arm\ncuff reading", fill=T.ORANGE_BG, stroke=T.ORANGE, size=7.6)
    box(d, 400, 160, 72, 54, "rakshpulse\nreference\n(time, SBP, DBP)", fill=T.WHITE, stroke=T.ORANGE, size=7.6)
    box(d, 400, 70, 72, 60, "reference_bp\n.csv in the\nsession folder", fill=T.WHITE, stroke=T.ORANGE, size=7.6)
    arrow(d, 436, 240, 436, 216, color=T.ORANGE)
    arrow(d, 436, 160, 436, 132, color=T.ORANGE)
    elbow(d, [(400, 100), (380, 100), (380, 80), (374, 80)], color=T.ORANGE)
    text(
        d,
        186,
        22,
        "Output: per-window quality, features and a mode-labelled estimate (demonstration on MAX30102 today)",
        7.6,
        T.SANS_I,
        T.CHARCOAL,
    )
    arrow(d, 186, 46, 186, 32, color=T.CHARCOAL)
    return d


# ------------------------------------------------------------------------------------------------ 7 wiring
def wiring():
    d = Drawing(W, 270)
    # breakout
    d.add(Rect(0, 40, 150, 200, rx=8, ry=8, fillColor=T.BLUE_BG, strokeColor=T.BLUE, strokeWidth=1.4))
    text(d, 75, 222, "MAX30102 breakout", 9.5, T.SANS_B, T.BLUE)
    text(d, 75, 210, "(pin labels vary by vendor)", 7, T.SANS_I, T.MUTED)
    d.add(Rect(330, 40, 150, 200, rx=8, ry=8, fillColor=T.GREY_BG, strokeColor=T.NAVY, strokeWidth=1.4))
    text(d, 405, 222, "ESP32 / Arduino board", 9.5, T.SANS_B, T.NAVY)
    text(d, 405, 210, "(read the board's pinout)", 7, T.SANS_I, T.MUTED)
    wires = [
        ("VIN / VCC", "supply pin the breakout\ndocumentation specifies", T.RED, "power"),
        ("GND", "GND", T.CHARCOAL, "ground"),
        ("SDA", "I²C SDA: board default,\nor I2C_SDA_PIN in sketch", T.BLUE, "data"),
        ("SCL", "I²C SCL: board default,\nor I2C_SCL_PIN in sketch", T.TEAL, "clock"),
        ("INT", "not used by the sketch", T.MUTED, "optional"),
    ]
    for i, (a, b, col, tag) in enumerate(wires):
        y = 180 - i * 33
        d.add(Rect(118, y - 7, 32, 14, fillColor=T.WHITE, strokeColor=col, strokeWidth=1))
        text(d, 134, y - 3, a.split(" ")[0], 7.5, T.SANS_B, col)
        if " " in a:
            text(d, 112, y - 3, a, 7, T.SANS, T.CHARCOAL, "end")
        d.add(Rect(330, y - 9, 120, 18, fillColor=T.WHITE, strokeColor=col, strokeWidth=1))
        lines(d, 390, y, b, 6.6, color=T.CHARCOAL, lead=1.1)
        dash = [3, 2] if tag == "optional" else None
        d.add(Line(150, y, 330, y, strokeColor=col, strokeWidth=2 if not dash else 1, strokeDashArray=dash))
        label_on(d, 240, y + 4, tag, size=7, color=col)
    text(
        d,
        240,
        22,
        "Supply voltage, pull-up resistors and level shifting: follow your breakout's documentation.",
        7.6,
        T.SANS_I,
        T.RED,
    )
    text(d, 240, 10, "This diagram shows which signals connect, not which pin numbers to use.", 7.6, T.SANS_I, T.RED)
    return d


# ------------------------------------------------------------------------------------------------ 8 firmware upload
def firmware_upload():
    d = Drawing(W, 215)
    steps = [
        "Install\nArduino IDE",
        "Add board\npackage (esp32\nby Espressif)",
        "Install SparkFun\nMAX3010x\nlibrary",
        "Open sketch\nrakshpulse_\nmax30102.ino",
    ]
    c, w = flow(d, steps, 140, h=52, size=7.8)
    for i, (x, y) in enumerate(c):
        badge(d, x - w / 2 + 4, y + 26, i + 1)
    steps2 = [
        "Set SAMPLE_RATE,\nSAMPLE_AVERAGE,\nLED_POWER, pins",
        "Select board\nand port,\nUpload",
        "Serial Monitor\n115200: look for\nthe banner",
        "CLOSE the\nSerial Monitor\n(frees the port)",
    ]
    c2, w2 = flow(d, steps2[::-1], 50, h=52, size=7.8, fills=[T.ORANGE_BG, T.TEAL_BG, T.WHITE, T.WHITE])
    # renumber reversed row
    for i, (x, y) in enumerate(c2[::-1]):
        badge(d, x - w2 / 2 + 4, y + 26, i + 5)
    # fix arrow directions: draw over with right-to-left arrows
    for i in range(3):
        xa = c2[i + 1][0] - w2 / 2
        xb = c2[i][0] + w2 / 2
        d.add(Rect(xb + 1, 70, xa - xb - 2, 12, fillColor=T.WHITE, strokeColor=None))
        arrow(d, xa - 1, 76, xb + 1, 76)
    elbow(d, [(c[-1][0], 140), (c[-1][0], 104)])
    text(
        d,
        W / 2,
        200,
        "Firmware upload (Arduino IDE). The sketch has not been compiled on hardware by the software team.",
        7.8,
        T.SANS_I,
        T.MUTED,
    )
    text(d, W / 2, 30, "Then: rakshpulse check-serial (Chapter 9)", 8, T.SANS_B, T.TEAL)
    return d


# ------------------------------------------------------------------------------------------------ 9 serial
def serial_comm():
    d = Drawing(W, 255)
    d.add(Rect(0, 95, 250, 150, rx=6, ry=6, fillColor=T.TERM_BG, strokeColor=T.TERM_BG))
    rows = [
        ("# rakshpulse fw=1 fs_hz=100 sample_rate=100 ...", T.TERM_DIM),
        ("1203,95321,118204", T.TERM_FG),
        ("1204,95318,118199", T.TERM_FG),
        ("1205,95330,118215", T.TERM_FG),
        ("1209,95341,118230", colors_hl()),
        ("1210,95337,118226", T.TERM_FG),
        ("# any comment line", T.TERM_DIM),
        ("1211,95329,118219", T.TERM_FG),
    ]
    for i, (s, col) in enumerate(rows):
        text(d, 10, 225 - i * 16, s, 8.2, T.MONO, col, "start")
    text(d, 125, 105, "what the board prints, one line per sample", 7, T.SANS_I, T.TERM_DIM)
    # field anatomy
    box(d, 285, 190, 195, 50, "", fill=T.GREY_BG, stroke=T.NAVY)
    parts = [
        ("1203", "sample_index", T.NAVY),
        (",", "", T.MUTED),
        ("95321", "red", T.RED),
        (",", "", T.MUTED),
        ("118204", "ir", T.BLUE),
    ]
    x = 305
    for s, lab, col in parts:
        wdt = stringWidth(s, T.MONO_B, 12)
        text(d, x, 218, s, 12, T.MONO_B, col, "start")
        if lab:
            text(d, x + wdt / 2, 200, lab, 7.5, T.SANS_B, col)
        x += wdt + 2
    text(d, 382, 246, "one data line", 8, T.SANS_B, T.NAVY)
    box(
        d,
        285,
        95,
        195,
        82,
        "**Rules\n32-bit counter, +1 per sample\nadvanced by samples lost in FIFO overflow\n'#' lines are comments (skipped)\n"
        "115200 baud, '\\n' at the end of each line\nraw counts, no filtering on the board",
        fill=T.WHITE,
        stroke=T.NAVY,
        size=7.6,
    )
    # gap explanation
    box(d, 0, 10, 480, 70, "", fill=T.ORANGE_BG, stroke=T.ORANGE)
    text(d, 12, 64, "Counter jump 1205 → 1209: three samples (1206–1208) were lost.", 8.5, T.SANS_B, T.ORANGE, "start")
    xs = [14 + i * 41 for i in range(8)]
    for i, v in enumerate(range(1203, 1211)):
        lost = 1206 <= v <= 1208
        d.add(
            Rect(
                xs[i],
                22,
                36,
                24,
                rx=3,
                ry=3,
                fillColor=T.WHITE if not lost else T.RED_BG,
                strokeColor=T.RED if lost else T.NAVY,
                strokeWidth=1,
                strokeDashArray=[2, 2] if lost else None,
            )
        )
        text(d, xs[i] + 18, 31, str(v), 8, T.MONO, T.RED if lost else T.NAVY)
    text(d, 412, 40, "the software writes the", 7.5, T.SANS_I, T.CHARCOAL)
    text(d, 412, 30, "gap to gaps.csv; no 10 s", 7.5, T.SANS_I, T.CHARCOAL)
    text(d, 412, 20, "window spans a gap", 7.5, T.SANS_I, T.CHARCOAL)
    return d


def colors_hl():
    return T.ORANGE


# ------------------------------------------------------------------------------------------------ 10 recording
def recording_workflow():
    d = Drawing(W, 230)
    top = [
        "**rakshpulse ports\nfind COMx or\n``/dev/ttyUSBx",
        "**check-serial\n10 s test\nmust say PASS",
        "**record\n``--source serial\n``--duration 300",
        "**inspect\n``--validate\nall PASS",
    ]
    c, w = flow(d, top, 150, h=56, fills=[T.WHITE, T.TEAL_BG, T.BLUE_BG, T.TEAL_BG], size=8)
    box(d, 0, 20, 300, 100, "", fill=T.GREY_BG, stroke=T.MUTED)
    text(d, 10, 104, "What record does", 8.5, T.SANS_B, T.NAVY, "start")
    for i, s in enumerate(
        [
            "1. opens the port, reads the banner, checks fs_hz against --fs",
            "2. waits for valid samples, then creates the session folder",
            "3. writes raw_ppg.csv continuously; logs gaps and duplicates",
            "4. stops at --duration or Ctrl-C (or --timeout without data)",
            "5. closes, hashes (SHA-256) and registers every file",
        ]
    ):
        text(d, 12, 88 - i * 14, s, 7.8, T.SANS, T.CHARCOAL, "start")
    box(
        d,
        320,
        20,
        160,
        100,
        "**Summary printed at the end\n``samples, missing_samples,\n``gaps, drift_ppm, stopped_by,\n``duplicates, counter_resets,\n``malformed_lines",
        fill=T.TERM_BG,
        stroke=T.TERM_BG,
        tc=T.TERM_FG,
        size=7.6,
    )
    arrow(d, c[2][0], 150, c[2][0], 123, color=T.MUTED, dash=[3, 2])
    text(d, W / 2, 218, "Never record without a PASS from check-serial first", 8, T.SANS_B, T.ORANGE)
    return d


# ------------------------------------------------------------------------------------------------ 11 cuff
def cuff_workflow():
    d = Drawing(W, 220)
    x0, x1, y = 30, 460, 120
    d.add(Rect(x0, y, x1 - x0, 22, fillColor=T.BLUE_BG, strokeColor=T.BLUE, strokeWidth=1))
    text(d, (x0 + x1) / 2, y + 7, "PPG recording running (rakshpulse record, 300 s)", 8, T.SANS_B, T.BLUE)
    for s in range(0, 301, 60):
        xx = x0 + (x1 - x0) * s / 300
        d.add(Line(xx, y - 4, xx, y, strokeColor=T.MUTED, strokeWidth=0.8))
        text(d, xx, y - 14, f"{s} s", 7, T.SANS, T.MUTED)
    for k, s in enumerate((60, 150, 240)):
        xx = x0 + (x1 - x0) * s / 300
        d.add(Circle(xx, y + 48, 11, fillColor=T.ORANGE, strokeColor=T.WHITE))
        text(d, xx, y + 45, f"R{k + 1}", 8, T.SANS_B, T.WHITE)
        arrow(d, xx, y + 37, xx, y + 24, color=T.ORANGE)
        text(d, xx, y + 66, f"cuff result shown at {s} s", 7.2, T.SANS_I, T.ORANGE)
    text(
        d,
        x0,
        200,
        "Example timing only: at least 3 readings per session, at least 60 s apart, times in seconds from the recording start",
        7.5,
        T.SANS_I,
        T.CHARCOAL,
        "start",
    )
    box(d, 0, 10, 480, 72, "", fill=T.TERM_BG, stroke=T.TERM_BG)
    cmds = [
        "rakshpulse reference --dataset data/hw --subject P001 --session D1 --time 60  --sbp .. --dbp ..",
        "rakshpulse reference --dataset data/hw --subject P001 --session D1 --time 150 --sbp .. --dbp ..",
        "rakshpulse reference --dataset data/hw --subject P001 --session D1 --time 240 --sbp .. --dbp ..",
    ]
    for i, s in enumerate(cmds):
        text(d, 10, 62 - i * 16, s, 7.6, T.MONO, T.TERM_FG, "start")
    text(d, 472, 16, "values typed from the cuff display", 6.8, T.SANS_I, T.TERM_DIM, "end")
    return d


# ------------------------------------------------------------------------------------------------ 12 end-to-end
def end_to_end():
    d = Drawing(W, 300)
    stages = [
        ("Wire", "MAX30102 to board\n(Ch. 5)", T.BLUE),
        ("Flash", "reference sketch\n(Ch. 6)", T.BLUE),
        ("Install", "rakshpulse + doctor\n(Ch. 7)", T.TEAL),
        ("Check", "check-serial PASS\n(Ch. 9)", T.TEAL),
        ("Record", "5-min sessions\n(Ch. 10)", T.NAVY_2),
        ("Reference", "≥3 cuff readings\n(Ch. 11)", T.ORANGE),
        ("Analyze", "inspect + analyze\n(Ch. 12)", T.NAVY_2),
        ("Hand off", "zip, share privately\n(Ch. 14)", T.ORANGE),
    ]
    w, h, gx = 105, 64, 20
    for i, (t, s, col) in enumerate(stages):
        row = i // 4
        j = i % 4 if row == 0 else 3 - i % 4
        x = j * (w + gx)
        y = 200 - row * 120
        header_box(d, x, y, w, h, f"{i + 1}. {t}", s, color=col, size=7.8)
        if i % 4:
            if row == 0:
                arrow(d, x - gx + 1, y + 24, x - 1, y + 24)
            else:
                arrow(d, x + w + gx - 1, y + 24, x + w + 1, y + 24)
    elbow(d, [(3 * (w + gx) + w / 2, 200), (3 * (w + gx) + w / 2, 146)])
    # loops
    box(d, 0, 10, 480, 50, "", fill=T.RED_BG, stroke=T.RED, dash=[3, 2])
    text(
        d,
        240,
        42,
        "If any check fails: go to Troubleshooting (Ch. 13), fix, and repeat that stage.",
        8.5,
        T.SANS_B,
        T.RED,
    )
    text(d, 240, 26, "A demonstration-mode number is never a blood-pressure measurement.", 8, T.SANS_I, T.RED)
    return d


# ------------------------------------------------------------------------------------------------ 13 dataset tree
def dataset_tree():
    d = Drawing(W, 250)
    d.add(Rect(0, 0, 300, 250, rx=6, ry=6, fillColor=T.GREY_BG, strokeColor=T.GREY_LINE))
    tree = [
        (0, "data/rakshpulse_hw/", T.NAVY, True),
        (1, "dataset_manifest.csv", T.CHARCOAL, False),
        (1, "subjects.csv", T.CHARCOAL, False),
        (1, "sessions/", T.NAVY, True),
        (2, "P001/", T.NAVY, True),
        (3, "D1/", T.NAVY, True),
        (4, "session_metadata.csv", T.CHARCOAL, False),
        (4, "raw_ppg.csv", T.BLUE, False),
        (4, "gaps.csv", T.CHARCOAL, False),
        (4, "reference_bp.csv", T.ORANGE, False),
        (4, "derived/", T.TEAL, True),
        (5, "windows.csv", T.TEAL, False),
        (5, "segments.csv", T.TEAL, False),
        (5, "predictions_auricle-ear-ridge.csv", T.TEAL, False),
        (3, "D2/  (next day: new session)", T.MUTED, True),
    ]
    for i, (lvl, s, col, folder) in enumerate(tree):
        y = 232 - i * 15.4
        x = 12 + lvl * 16
        if lvl:
            d.add(Line(x - 10, y + 3, x - 3, y + 3, strokeColor=T.MUTED, strokeWidth=0.6))
        text(d, x, y, s, 8, T.MONO_B if folder else T.MONO, col, "start")
    notes = [
        (232 - 7 * 15.4, "raw samples (never overwritten)", T.BLUE),
        (232 - 9 * 15.4, "cuff readings (append-only)", T.ORANGE),
        (232 - 10 * 15.4, "written by analyze / preprocess", T.TEAL),
        (232 - 1 * 15.4, "SHA-256 of every file", T.CHARCOAL),
        (232 - 8 * 15.4, "lost-sample gaps", T.CHARCOAL),
    ]
    for y, s, col in notes:
        arrow(d, 368, y + 3, 306, y + 3, color=col, sw=0.8, head=4)
        text(d, 372, y, s, 7.6, T.SANS_I, col, "start")
    return d


ALL = {
    "overview": (
        system_overview,
        "RakshPulse at a glance",
        "Signal path from the sensor to the analysis, and the cuff reference path. The dashed line marks the handoff point.",
    ),
    "max30102": (
        max30102_block,
        "Inside the MAX30102",
        "Light from two LEDs enters the skin; the photodetector measures what returns. Samples queue in a 32-entry FIFO until the board reads them over I²C.",
    ),
    "esp32": (
        esp32_block,
        "The ESP32 / Arduino board's job",
        "The board only moves data: it reads the sensor over I²C and prints text lines over USB serial.",
    ),
    "ppg": (
        ppg_waveform,
        "What a PPG signal looks like",
        "A small pulsatile (AC) wave rides on a large steady (DC) level. Heart rate comes from beat intervals; the model uses pulse shape.",
    ),
    "ml": (
        ml_pipeline,
        "The processing and ML pipeline",
        "From raw counts to a mode-labelled output. Windows that fail the quality gate never receive a number.",
    ),
    "architecture": (
        architecture,
        "System architecture",
        "Four layers and the separate reference path. Each layer only talks to the next one.",
    ),
    "wiring": (
        wiring,
        "Wiring: which signals connect",
        "Connect power, ground, SDA and SCL. Pin numbers, supply voltage and pull-ups come from your board's and breakout's documentation.",
    ),
    "firmware": (
        firmware_upload,
        "Uploading the firmware",
        "Eight steps, in order. Close the Serial Monitor before running any rakshpulse command.",
    ),
    "serial": (
        serial_comm,
        "Serial communication contract",
        "One text line per sample. The counter lets the software detect lost samples instead of silently joining time.",
    ),
    "recording": (recording_workflow, "Recording workflow", "Check, record, validate: the order matters."),
    "cuff": (
        cuff_workflow,
        "Cuff reference workflow",
        "Each reading is entered with its time in seconds from the start of the recording.",
    ),
    "e2e": (
        end_to_end,
        "End-to-end integration workflow",
        "Eight stages from a bare board to a dataset the software team can use.",
    ),
    "tree": (
        dataset_tree,
        "Dataset folder after recording and analysis",
        "One folder per subject and session. Share the whole dataset folder, zipped, privately.",
    ),
}
