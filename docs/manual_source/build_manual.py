"""Build docs/RAKSHPULSE_HARDWARE_INTEGRATION_MANUAL.pdf.

    pip install reportlab          # documentation build tool only; not a RakshPulse dependency
    python docs/manual_source/build_manual.py

Content lives in ``content.py`` (one function per chapter), diagrams in ``diagrams.py``, colours and fonts in
``theme.py``. Inline markup in content strings: `code`, **bold**, *italic*.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import content  # noqa: E402
import diagrams  # noqa: E402
import theme as T  # noqa: E402
from reportlab.lib.enums import TA_CENTER  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    BaseDocTemplate,
    CondPageBreak,
    Flowable,
    Frame,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    XPreformatted,
)
from reportlab.platypus.tableofcontents import TableOfContents  # noqa: E402

OUT = HERE.parent / "RAKSHPULSE_HARDWARE_INTEGRATION_MANUAL.pdf"
TITLE = "RakshPulse Hardware Integration Manual"
VERSION = "Edition 1.0 for software release 5.1"
PAGE_W, PAGE_H = A4
MARGIN_X, MARGIN_TOP, MARGIN_BOTTOM = 20 * mm, 24 * mm, 20 * mm
TEXT_W = PAGE_W - 2 * MARGIN_X
WRAP = 90  # characters per terminal line at 8.6 pt Consolas


# ------------------------------------------------------------------------------------------------ styles
def _s(name, **kw):
    base = dict(fontName=T.SANS, fontSize=10.5, leading=14.6, textColor=T.CHARCOAL, spaceAfter=6)
    base.update(kw)
    return ParagraphStyle(name, **base)


S = {
    "body": _s("body"),
    "lead": _s("lead", fontSize=12, leading=17, textColor=T.NAVY_2, spaceAfter=10),
    "h2": _s(
        "h2", fontName=T.SANS_B, fontSize=15, leading=19, textColor=T.NAVY, spaceBefore=12, spaceAfter=6, keepWithNext=1
    ),
    "h3": _s(
        "h3",
        fontName=T.SANS_B,
        fontSize=12,
        leading=16,
        textColor=T.NAVY_2,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=1,
    ),
    "bullet": _s("bullet", leftIndent=14, bulletIndent=3, spaceAfter=3),
    "cell": _s("cell", fontSize=9.2, leading=12, spaceAfter=0),
    "cellb": _s("cellb", fontName=T.SANS_B, fontSize=9.2, leading=12, spaceAfter=0, textColor=T.WHITE),
    "boxtitle": _s("boxtitle", fontName=T.SANS_B, fontSize=10.5, leading=13, spaceAfter=2),
    "boxbody": _s("boxbody", fontSize=10, leading=13.6, spaceAfter=2),
    "figtitle": _s("figtitle", fontName=T.SANS_B, fontSize=10, leading=13, textColor=T.NAVY, spaceAfter=4),
    "caption": _s(
        "caption",
        fontName=T.SANS_I,
        fontSize=9,
        leading=12,
        textColor=T.MUTED,
        alignment=TA_CENTER,
        spaceBefore=4,
        spaceAfter=12,
    ),
    "code": _s("code", fontName=T.MONO, fontSize=8.6, leading=11.2, textColor=T.CHARCOAL, spaceAfter=0),
    "term": _s("term", fontName=T.MONO, fontSize=8.6, leading=11.4, textColor=T.TERM_FG, spaceAfter=0),
    "termlabel": _s("termlabel", fontName=T.SANS_B, fontSize=7.8, leading=9, textColor=T.TERM_DIM, spaceAfter=0),
    "toc0": _s("toc0", fontName=T.SANS_B, fontSize=11, leading=16, textColor=T.NAVY, leftIndent=0, spaceBefore=3),
    "toc1": _s("toc1", fontSize=9.6, leading=12.6, leftIndent=18, textColor=T.CHARCOAL),
}


def md(s: str) -> str:
    """Escape and convert `code`, **bold**, *italic* to reportlab markup."""
    s = escape(s)
    s = re.sub(r"`([^`]+)`", lambda m: f'<font face="{T.MONO}" size="9.2" color="#7A2E0E">{m.group(1)}</font>', s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", s)
    return s


# ------------------------------------------------------------------------------------------------ flowables
class ChapterBand(Flowable):
    """Full-width coloured chapter header; registers the TOC entry and a PDF bookmark."""

    def __init__(self, num, title, subtitle, color):
        super().__init__()
        self.num, self.title, self.subtitle, self.color = num, title, subtitle, color
        self.key = f"ch{num}"

    def wrap(self, aw, ah):
        self.width = aw
        return aw, 92

    def draw(self):
        c = self.canv
        c.saveState()
        c.setFillColor(self.color)
        c.roundRect(0, 0, self.width, 88, 8, fill=1, stroke=0)
        c.setFillColor(T.WHITE)
        c.setFont(T.SANS_B, 40)
        c.drawString(18, 26, f"{self.num:02d}")
        c.setStrokeColor(T.WHITE)
        c.setLineWidth(1)
        c.line(84, 18, 84, 70)
        c.setFont(T.SANS_B, 21)
        c.drawString(98, 46, self.title)
        c.setFont(T.SANS_I, 10.5)
        c.drawString(98, 26, self.subtitle)
        c.restoreState()
        c.bookmarkPage(self.key)
        c.addOutlineEntry(f"{self.num}. {self.title}", self.key, level=0)


class H2(Paragraph):
    pass


class Doc(BaseDocTemplate):
    def __init__(self, path):
        super().__init__(
            str(path),
            pagesize=A4,
            leftMargin=MARGIN_X,
            rightMargin=MARGIN_X,
            topMargin=MARGIN_TOP,
            bottomMargin=MARGIN_BOTTOM,
            title=TITLE,
            author="RakshPulse software team",
            subject="MAX30102 / ESP32 hardware integration handbook",
            creator="docs/manual_source/build_manual.py",
        )
        body = Frame(
            MARGIN_X,
            MARGIN_BOTTOM,
            TEXT_W,
            PAGE_H - MARGIN_TOP - MARGIN_BOTTOM,
            id="body",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        )
        cover = Frame(0, 0, PAGE_W, PAGE_H, id="cover", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates(
            [PageTemplate("cover", [cover], onPage=draw_cover), PageTemplate("body", [body], onPageEnd=self.decorate)]
        )
        self.chapter = ""
        self._h2n = 0

    def beforeDocument(self):
        self._h2n = 0  # keys must be identical on every multiBuild pass, or the TOC never settles
        self.chapter = ""

    def afterFlowable(self, f):
        if isinstance(f, ChapterBand):
            self.chapter = f"{f.num}. {f.title}"
            self.notify("TOCEntry", (0, f"{f.num}.  {f.title}", self.page, f.key))
        elif isinstance(f, H2):
            self._h2n += 1
            key = f"h2_{self._h2n}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(f.getPlainText(), key, level=1)
            self.notify("TOCEntry", (1, f.getPlainText(), self.page, key))

    def decorate(self, c, doc):
        c.saveState()
        top = PAGE_H - 13 * mm
        c.setFillColor(T.NAVY)
        c.rect(0, PAGE_H - 6, PAGE_W, 6, fill=1, stroke=0)
        c.setFont(T.SANS_B, 8.5)
        c.drawString(MARGIN_X, top, "RakshPulse")
        c.setFont(T.SANS, 8.5)
        c.setFillColor(T.MUTED)
        c.drawString(MARGIN_X + 46, top, "Hardware Integration Manual")
        c.drawRightString(PAGE_W - MARGIN_X, top, self.chapter or "Contents")
        c.setStrokeColor(T.GREY_LINE)
        c.setLineWidth(0.6)
        c.line(MARGIN_X, top - 4, PAGE_W - MARGIN_X, top - 4)
        c.line(MARGIN_X, 13 * mm, PAGE_W - MARGIN_X, 13 * mm)
        c.setFont(T.SANS_I, 8)
        c.drawString(
            MARGIN_X,
            9 * mm,
            "Research prototype. Not a medical device. Demonstration outputs are not blood-pressure measurements.",
        )
        c.setFont(T.SANS_B, 9)
        c.setFillColor(T.NAVY)
        c.drawRightString(PAGE_W - MARGIN_X, 9 * mm, f"Page {doc.page}")
        c.restoreState()


def draw_cover(c, doc):
    c.saveState()
    c.setFillColor(T.NAVY)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    c.setFillColor(T.NAVY_2)
    c.rect(0, PAGE_H * 0.36, PAGE_W, PAGE_H * 0.30, fill=1, stroke=0)
    # stylised pulse line
    import math

    c.setStrokeColor(T.TEAL)
    c.setLineWidth(2.4)
    p = c.beginPath()
    y0 = PAGE_H * 0.51
    for i in range(0, 601):
        x = PAGE_W * i / 600
        ph = (x / 140.0) % 1.0
        v = math.exp(-(((ph - 0.25) / 0.05) ** 2)) + 0.4 * math.exp(-(((ph - 0.45) / 0.07) ** 2))
        y = y0 + 52 * v
        p.moveTo(x, y) if i == 0 else p.lineTo(x, y)
    c.drawPath(p, stroke=1, fill=0)
    c.setFillColor(T.WHITE)
    c.setFont(T.SANS_B, 13)
    c.drawString(MARGIN_X, PAGE_H - 40 * mm, "RAKSHPULSE")
    c.setFillColor(T.TEAL)
    c.rect(MARGIN_X, PAGE_H - 44 * mm, 30 * mm, 2, fill=1, stroke=0)
    c.setFillColor(T.WHITE)
    c.setFont(T.SANS_B, 34)
    c.drawString(MARGIN_X, PAGE_H - 68 * mm, "Hardware Integration")
    c.drawString(MARGIN_X, PAGE_H - 82 * mm, "Manual")
    c.setFont(T.SANS, 14)
    c.setFillColor(colors_light())
    c.drawString(MARGIN_X, PAGE_H - 95 * mm, "MAX30102 + ESP32 / Arduino  →  rakshpulse software")
    c.setFont(T.SANS, 11)
    c.drawString(MARGIN_X, PAGE_H * 0.30, "A step-by-step handbook for the hardware integration team:")
    c.drawString(
        MARGIN_X, PAGE_H * 0.30 - 16, "wiring, firmware, installation, first test, recording, cuff references,"
    )
    c.drawString(MARGIN_X, PAGE_H * 0.30 - 32, "analysis, troubleshooting and handoff.")
    # status box
    bx, by, bw, bh = MARGIN_X, 28 * mm, PAGE_W - 2 * MARGIN_X, 38 * mm
    c.setFillColor(T.ORANGE)
    c.roundRect(bx, by, bw, bh, 6, fill=1, stroke=0)
    c.setFillColor(T.WHITE)
    c.setFont(T.SANS_B, 11)
    c.drawString(bx + 12, by + bh - 17, "Read this first")
    c.setFont(T.SANS, 9.6)
    for i, s in enumerate(
        [
            "RakshPulse is a research prototype, not a medical device.",
            "The reference firmware has not yet been compiled or run on a physical board.",
            "No model is validated for MAX30102 data.",
            "Every blood-pressure number on MAX30102 data is a demonstration, not a measurement.",
        ]
    ):
        c.drawString(bx + 12, by + bh - 34 - i * 14, "•  " + s)
    c.setFont(T.SANS, 9)
    c.setFillColor(colors_light())
    c.drawString(MARGIN_X, 16 * mm, VERSION)
    c.drawRightString(PAGE_W - MARGIN_X, 16 * mm, "RakshPulse software team")
    c.restoreState()


def colors_light():
    from reportlab.lib import colors

    return colors.HexColor("#C9D6EA")


# ------------------------------------------------------------------------------------------------ builder API used by content.py
class Builder:
    def __init__(self):
        self.story = []
        self.figures = 0

    # text
    def p(self, s, style="body"):
        self.story.append(Paragraph(md(s), S[style]))

    def lead(self, s):
        self.p(s, "lead")

    def h2(self, s):
        self.story.append(CondPageBreak(60 * mm))
        self.story.append(H2(md(s), S["h2"]))

    def h3(self, s):
        self.story.append(CondPageBreak(30 * mm))
        self.story.append(Paragraph(md(s), S["h3"]))

    def bullets(self, items, numbered=False):
        for i, s in enumerate(items, 1):
            b = f"{i}." if numbered else "•"
            self.story.append(Paragraph(md(s), S["bullet"], bulletText=b))
        self.story.append(Spacer(1, 4))

    def space(self, h=6):
        self.story.append(Spacer(1, h))

    def chapter(self, num, title, subtitle, color=None):
        if num > 1:
            self.story.append(PageBreak())
        self.story.append(ChapterBand(num, title, subtitle, color or T.NAVY))
        self.story.append(Spacer(1, 14))

    # boxes
    def box(self, kind, title, body):
        colour, bg, icon = {
            "info": (T.BLUE, T.BLUE_BG, "i"),
            "warning": (T.RED, T.RED_BG, "!"),
            "important": (T.ORANGE, T.ORANGE_BG, "★"),
            "success": (T.TEAL, T.TEAL_BG, "✓"),
            "tip": (T.TEAL, T.TEAL_BG, "✓"),
        }[kind]
        label = {
            "info": "TECHNICAL NOTE",
            "warning": "WARNING",
            "important": "IMPORTANT",
            "success": "EXPECTED RESULT",
            "tip": "TIP",
        }[kind]
        tstyle = ParagraphStyle("bt", parent=S["boxtitle"], textColor=colour)
        parts = [Paragraph(f"{label}: {md(title)}" if title else label, tstyle)]
        for para in body if isinstance(body, list) else [body]:
            if isinstance(para, tuple):  # bullet list
                for s in para:
                    parts.append(
                        Paragraph(
                            md(s),
                            ParagraphStyle("bb", parent=S["boxbody"], leftIndent=12, bulletIndent=2),
                            bulletText="•",
                        )
                    )
            else:
                parts.append(Paragraph(md(para), S["boxbody"]))
        if T.SYM is None:
            icon = {"★": "!", "✓": "+"}.get(icon, icon)
        t = Table([[icon, parts]], colWidths=[22, TEXT_W - 22])
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), bg),
                    ("BACKGROUND", (0, 0), (0, 0), colour),
                    ("TEXTCOLOR", (0, 0), (0, 0), T.WHITE),
                    ("FONT", (0, 0), (0, 0), T.SYM if icon in "★✓" and T.SYM else T.SANS_B, 12),
                    ("ALIGN", (0, 0), (0, 0), "CENTER"),
                    ("VALIGN", (0, 0), (0, 0), "TOP"),
                    ("VALIGN", (1, 0), (1, 0), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 7),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("LEFTPADDING", (1, 0), (1, 0), 10),
                    ("RIGHTPADDING", (1, 0), (1, 0), 10),
                ]
            )
        )
        self.story.append(Spacer(1, 4))
        self.story.append(t)
        self.story.append(Spacer(1, 10))

    def legend(self, items):
        """Colour key: (kind, label, text)."""
        cols = {"info": T.BLUE, "important": T.ORANGE, "warning": T.RED, "success": T.TEAL}
        bgs = {"info": T.BLUE_BG, "important": T.ORANGE_BG, "warning": T.RED_BG, "success": T.TEAL_BG}
        lab = ParagraphStyle("lg", parent=S["cell"], fontName=T.SANS_B, textColor=T.WHITE, fontSize=9)
        rows = [[Paragraph(lb, lab), Paragraph(md(tx), S["cell"])] for _, lb, tx in items]
        t = Table(rows, colWidths=[42 * mm, TEXT_W - 42 * mm])
        st = [
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("LINEBELOW", (0, 0), (-1, -1), 2, T.WHITE),
        ]
        for i, (k, _, _) in enumerate(items):
            st += [("BACKGROUND", (0, i), (0, i), cols[k]), ("BACKGROUND", (1, i), (1, i), bgs[k])]
        t.setStyle(TableStyle(st))
        self.story.append(t)
        self.story.append(Spacer(1, 10))

    # code
    def terminal(self, code, shell="Terminal", note=None):
        """Dark terminal block. Lines starting with '>' are typed commands; others are output."""
        rows = []
        src, wrapped = [], False
        for ln in code.strip("\n").split("\n"):
            limit = WRAP + (2 if ln.startswith("> ") else 0)
            while len(ln) > limit:
                cut = ln.rfind(" ", 0, limit)
                cut = cut if cut > 20 else limit
                src.append(ln[:cut])
                ln = "      " + ln[cut:].lstrip()
                limit, wrapped = WRAP, True
            src.append(ln)
        if wrapped:
            note = (note + "; " if note else "") + "long lines wrapped for print: type each command on one line"
        for ln in src:
            if ln.startswith("> "):
                rows.append(
                    f'<font color="#7FD1B9">{"PS>" if shell == "PowerShell" else "$"}</font> '
                    f'<font color="#FFFFFF">{escape(ln[2:])}</font>'
                )
            elif ln.startswith("#"):
                rows.append(f'<font color="#8FA1B3">{escape(ln)}</font>')
            else:
                rows.append(f'<font color="#C9D3DD">{escape(ln)}</font>')
        head = Paragraph(shell.upper() + (f"  ·  {escape(note)}" if note else ""), S["termlabel"])
        body = XPreformatted("\n".join(rows), S["term"])
        t = Table([[head], [body]], colWidths=[TEXT_W])
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), T.TERM_BG),
                    ("BACKGROUND", (0, 0), (0, 0), colors_term_head()),
                    ("LEFTPADDING", (0, 0), (-1, -1), 10),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                    ("TOPPADDING", (0, 0), (0, 0), 4),
                    ("BOTTOMPADDING", (0, 0), (0, 0), 4),
                    ("TOPPADDING", (0, 1), (0, 1), 7),
                    ("BOTTOMPADDING", (0, 1), (0, 1), 8),
                ]
            )
        )
        self.story.append(Spacer(1, 2))
        self.story.append(t)
        self.story.append(Spacer(1, 9))

    def code(self, code, label=None):
        rows = [escape(x) for x in code.strip("\n").split("\n")]
        cells = []
        if label:
            cells.append([Paragraph(escape(label), ParagraphStyle("cl", parent=S["termlabel"], textColor=T.MUTED))])
        cells.append([XPreformatted("\n".join(rows), S["code"])])
        t = Table(cells, colWidths=[TEXT_W])
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), T.GREY_BG),
                    ("BOX", (0, 0), (-1, -1), 0.6, T.GREY_LINE),
                    ("LINEBEFORE", (0, 0), (0, -1), 3, T.MUTED),
                    ("LEFTPADDING", (0, 0), (-1, -1), 10),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        self.story.append(t)
        self.story.append(Spacer(1, 9))

    # tables
    def table(self, header, rows, widths, head_color=None, zebra=True, size=None):
        cs = ParagraphStyle("c", parent=S["cell"], fontSize=size or 9.2, leading=(size or 9.2) * 1.3)
        data = [[Paragraph(md(h), S["cellb"]) for h in header]]
        data += [[Paragraph(md(str(c)), cs) for c in r] for r in rows]
        tot = sum(widths)
        t = Table(data, colWidths=[w / tot * TEXT_W for w in widths], repeatRows=1)
        st = [
            ("BACKGROUND", (0, 0), (-1, 0), head_color or T.NAVY),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LINEBELOW", (0, 0), (-1, -1), 0.4, T.GREY_LINE),
            ("BOX", (0, 0), (-1, -1), 0.6, T.GREY_LINE),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]
        if zebra:
            st += [("BACKGROUND", (0, i), (-1, i), T.GREY_BG) for i in range(2, len(data), 2)]
        t.setStyle(TableStyle(st))
        self.story.append(Spacer(1, 2))
        self.story.append(t)
        self.story.append(Spacer(1, 10))

    def step(self, n, title, what, why, expected, mistakes, trouble):
        """A setup step card: what / why / expected / common mistakes / troubleshooting."""
        lab = ParagraphStyle("sl", parent=S["cell"], fontName=T.SANS_B, textColor=T.NAVY)
        cell = ParagraphStyle("sc", parent=S["cell"], fontSize=9.6, leading=12.8)
        head = Paragraph(
            f'<font color="#FFFFFF"><b>STEP {n}</b>  ·  {md(title)}</font>',
            ParagraphStyle("sh", parent=S["cell"], fontName=T.SANS_B, fontSize=11, leading=14),
        )
        rows = [[head, ""]]
        for k, v in (
            ("What to do", what),
            ("Why", why),
            ("Expected result", expected),
            ("Common mistakes", mistakes),
            ("If it fails", trouble),
        ):
            rows.append([Paragraph(k, lab), Paragraph(md(v), cell)])
        t = Table(rows, colWidths=[30 * mm, TEXT_W - 30 * mm])
        t.setStyle(
            TableStyle(
                [
                    ("SPAN", (0, 0), (1, 0)),
                    ("BACKGROUND", (0, 0), (1, 0), T.NAVY_2),
                    ("BACKGROUND", (0, 3), (1, 3), T.TEAL_BG),
                    ("BACKGROUND", (0, 4), (1, 4), T.ORANGE_BG),
                    ("BOX", (0, 0), (-1, -1), 0.8, T.NAVY_2),
                    ("LINEBELOW", (0, 1), (-1, -1), 0.4, T.GREY_LINE),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ]
            )
        )
        self.story.append(KeepTogether([t]))
        self.story.append(Spacer(1, 10))

    def checklist(self, items):
        cs = ParagraphStyle("ck", parent=S["cell"], fontSize=9.8, leading=13)
        rows = [
            [
                "☐" if T.SYM else "[ ]",
                Paragraph(md(a), cs),
                Paragraph(md(b), ParagraphStyle("ck2", parent=cs, textColor=T.MUTED)),
            ]
            for a, b in items
        ]
        t = Table(rows, colWidths=[16, TEXT_W * 0.58 - 16, TEXT_W * 0.42])
        t.setStyle(
            TableStyle(
                [
                    ("FONT", (0, 0), (0, -1), T.SYM or T.SANS, 13),
                    ("TEXTCOLOR", (0, 0), (0, -1), T.TEAL),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LINEBELOW", (0, 0), (-1, -1), 0.4, T.GREY_LINE),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        self.story.append(t)
        self.story.append(Spacer(1, 10))

    def figure(self, key):
        fn, title, caption = diagrams.ALL[key]
        self.figures += 1
        n = self.figures
        d = fn()
        scale = min(1.0, TEXT_W / d.width)
        d.scale(scale, scale)
        d.width, d.height = d.width * scale, d.height * scale
        d.hAlign = "CENTER"
        self.story.append(
            KeepTogether(
                [
                    Paragraph(f"Figure {n}. {escape(title)}", S["figtitle"]),
                    Spacer(1, 4),
                    d,
                    Paragraph(md(caption), S["caption"]),
                ]
            )
        )
        return n


def colors_term_head():
    from reportlab.lib import colors

    return colors.HexColor("#2D343C")


# ------------------------------------------------------------------------------------------------ build
def build(out: Path = OUT) -> Path:
    b = Builder()
    story = [NextPageTemplate("body"), PageBreak()]
    # contents page
    story.append(Paragraph("Contents", ParagraphStyle("ct", parent=S["h2"], fontSize=24, leading=30, spaceAfter=4)))
    story.append(
        Paragraph(
            "Click any entry to jump to it. PDF bookmarks give the same structure in your viewer's sidebar.",
            ParagraphStyle("ctn", parent=S["body"], textColor=T.MUTED, fontSize=9.5),
        )
    )
    toc = TableOfContents(dotsMinLevel=0)
    toc.levelStyles = [S["toc0"], S["toc1"]]
    story.append(toc)
    story.append(PageBreak())
    content.build(b)
    story += b.story
    doc = Doc(out)
    doc.multiBuild([Spacer(1, 1)] + story)
    return out


if __name__ == "__main__":
    p = build()
    print(f"wrote {p}")
