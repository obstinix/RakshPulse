"""Palette and fonts shared by the manual builder and the diagrams.

Fonts: Calibri and Consolas when they are installed (Windows); otherwise the built-in Helvetica / Courier, so the
manual still builds on Linux and macOS (with slightly different metrics).
"""

from __future__ import annotations

import os
from pathlib import Path

from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

NAVY = colors.HexColor("#1B2A4A")
NAVY_2 = colors.HexColor("#2C4372")
CHARCOAL = colors.HexColor("#33373D")
MUTED = colors.HexColor("#6B7280")
TEAL = colors.HexColor("#0F8B8D")
TEAL_BG = colors.HexColor("#E6F4F4")
ORANGE = colors.HexColor("#D9731A")
ORANGE_BG = colors.HexColor("#FDF1E6")
RED = colors.HexColor("#C0392B")
RED_BG = colors.HexColor("#FBEAEA")
BLUE = colors.HexColor("#3E7CB1")
BLUE_BG = colors.HexColor("#E8F1FA")
GREY_BG = colors.HexColor("#F2F3F5")
GREY_LINE = colors.HexColor("#D3D7DD")
TERM_BG = colors.HexColor("#1E2329")
TERM_FG = colors.HexColor("#E6E6E6")
TERM_DIM = colors.HexColor("#8FA1B3")
WHITE = colors.white

SANS, SANS_B, SANS_I, SANS_BI, MONO, MONO_B = (
    "Helvetica",
    "Helvetica-Bold",
    "Helvetica-Oblique",
    "Helvetica-BoldOblique",
    "Courier",
    "Courier-Bold",
)
SYM = None  # symbol font for check boxes and icons; ASCII fallbacks are used when it is missing


def _font_dir() -> Path | None:
    for d in (Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts", Path("/usr/share/fonts/truetype/msttcorefonts")):
        if (d / "calibri.ttf").exists() and (d / "consola.ttf").exists():
            return d
    return None


def register_fonts() -> None:
    """Register Calibri/Consolas once, if available."""
    global SANS, SANS_B, SANS_I, SANS_BI, MONO, MONO_B, SYM
    if SANS == "Body":
        return
    d = _font_dir()
    if d is None:
        return
    for name, f in (
        ("Body", "calibri.ttf"),
        ("Body-Bold", "calibrib.ttf"),
        ("Body-Italic", "calibrii.ttf"),
        ("Body-BoldItalic", "calibriz.ttf"),
        ("Mono", "consola.ttf"),
        ("Mono-Bold", "consolab.ttf"),
    ):
        pdfmetrics.registerFont(TTFont(name, str(d / f)))
    pdfmetrics.registerFontFamily(
        "Body", normal="Body", bold="Body-Bold", italic="Body-Italic", boldItalic="Body-BoldItalic"
    )
    pdfmetrics.registerFontFamily("Mono", normal="Mono", bold="Mono-Bold", italic="Mono", boldItalic="Mono-Bold")
    if (d / "seguisym.ttf").exists():
        pdfmetrics.registerFont(TTFont("Sym", str(d / "seguisym.ttf")))
        SYM = "Sym"
    SANS, SANS_B, SANS_I, SANS_BI, MONO, MONO_B = (
        "Body",
        "Body-Bold",
        "Body-Italic",
        "Body-BoldItalic",
        "Mono",
        "Mono-Bold",
    )


register_fonts()
