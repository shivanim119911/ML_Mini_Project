"""ReportLab building blocks that reproduce the AISSMS IOIT report format."""
import os
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Flowable, Frame, Image, KeepTogether,
                                NextPageTemplate, PageBreak, PageTemplate, Paragraph,
                                Preformatted, Spacer, Table, TableStyle)

W, H = 595.3, 841.9
LEFT, RIGHT = 63, 536
CW = RIGHT - LEFT
LOGO = Path(__file__).parent / "assets" / "aissms_logo.jpg"

# ---------------------------------------------------------------- fonts
_CANDIDATES = [
    ("/usr/share/fonts/truetype/liberation/LiberationSerif-{}.ttf", ("Regular", "Bold", "Italic", "BoldItalic")),
    ("C:/Windows/Fonts/{}", ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf")),
    ("/Library/Fonts/{}", ("Times New Roman.ttf", "Times New Roman Bold.ttf",
                           "Times New Roman Italic.ttf", "Times New Roman Bold Italic.ttf")),
    ("/usr/share/fonts/truetype/msttcorefonts/{}", ("Times_New_Roman.ttf", "Times_New_Roman_Bold.ttf",
                                                     "Times_New_Roman_Italic.ttf", "Times_New_Roman_Bold_Italic.ttf")),
]
FONT, BOLD, ITAL, BI = "Times-Roman", "Times-Bold", "Times-Italic", "Times-BoldItalic"
for pat, parts in _CANDIDATES:
    files = [pat.format(x) for x in parts]
    if all(os.path.exists(f) for f in files):
        for nm, f in zip(("Serif", "Serif-Bold", "Serif-Italic", "Serif-BoldItalic"), files):
            pdfmetrics.registerFont(TTFont(nm, f))
        pdfmetrics.registerFontFamily("Serif", normal="Serif", bold="Serif-Bold",
                                      italic="Serif-Italic", boldItalic="Serif-BoldItalic")
        FONT, BOLD, ITAL, BI = "Serif", "Serif-Bold", "Serif-Italic", "Serif-BoldItalic"
        break


def _st(name, **kw):
    base = dict(fontName=FONT, fontSize=11, leading=13.4, alignment=TA_JUSTIFY, spaceAfter=5)
    base.update(kw)
    return ParagraphStyle(name, **base)


S = {
    "body": _st("body"),
    "left": _st("left", alignment=TA_LEFT),
    "h1": _st("h1", fontName=BOLD, fontSize=16, leading=20, alignment=TA_LEFT, spaceBefore=4, spaceAfter=5),
    "h2": _st("h2", fontName=BOLD, fontSize=12, leading=15, alignment=TA_LEFT, spaceBefore=6, spaceAfter=4),
    "cap": _st("cap", fontName=BOLD, fontSize=10.5, leading=12.5, alignment=TA_CENTER, spaceBefore=3, spaceAfter=7),
    "tcap": _st("tcap", fontName=BOLD, fontSize=10.5, leading=12.5, alignment=TA_LEFT, spaceBefore=4, spaceAfter=3, keepWithNext=1),
    "cell": _st("cell", fontSize=9.3, leading=11, alignment=TA_LEFT, spaceAfter=0),
    "cellc": _st("cellc", fontSize=9.3, leading=11, alignment=TA_CENTER, spaceAfter=0),
    "cellh": _st("cellh", fontName=BOLD, fontSize=9.3, leading=11, alignment=TA_LEFT, spaceAfter=0),
    "bullet": _st("bullet", leftIndent=26, bulletIndent=12, spaceAfter=3),
    "num": _st("num", leftIndent=26, bulletIndent=8, spaceAfter=3),
    "pseudo": _st("pseudo", fontSize=10, leading=11.8, alignment=TA_LEFT, spaceAfter=0),
    "dash": _st("dash", fontSize=7, leading=8, alignment=TA_LEFT, spaceAfter=0),
    "code": ParagraphStyle("code", fontName=FONT, fontSize=8, leading=9.2),
    "cov": _st("cov", fontSize=12, leading=16, alignment=TA_CENTER, spaceAfter=0),
}
DASH = "-" * int(CW / stringWidth("-", FONT, 7))


# ------------------------------------------------------------- flowables
class TocLine(Flowable):
    def __init__(self, text, page):
        super().__init__(); self.text, self.page = text, page; self.height = 24

    def wrap(self, aw, ah):
        return aw, self.height

    def draw(self):
        c = self.canv
        c.setFont(FONT, 12); c.drawString(14, 8, self.text)
        c.setFont(BOLD, 12); pw = stringWidth(str(self.page), BOLD, 12)
        c.drawString(CW - pw, 8, str(self.page))
        tw = stringWidth(self.text, FONT, 12) + 16
        dots = ".  " if False else "."
        dw = stringWidth(".", FONT, 12)
        n = int((CW - pw - 4 - tw) / dw)
        c.setFont(FONT, 12); c.drawString(tw, 8, "." * max(n, 0))


class Heading(Paragraph):
    """Paragraph that tells the document which TOC entry it is."""
    def __init__(self, text, style, key=None):
        super().__init__(text, style); self._toc = key


def P(t, s="body"):
    return Paragraph(t, S[s])


def H1(t, key=None):
    return Heading(t, S["h1"], key)


def H2(t):
    return Paragraph(t, S["h2"])


def bullets(items):
    return [Paragraph(t, S["bullet"], bulletText="\u2022") for t in items]


def numbered(items):
    return [Paragraph(t, S["num"], bulletText=f"{i}.") for i, t in enumerate(items, 1)]


def tbl(rows, widths, caption=None, center_cols=(), header=True, size=None):
    data = []
    for r, row in enumerate(rows):
        line = []
        for c, v in enumerate(row):
            st = S["cellh"] if (header and r == 0) else (S["cellc"] if c in center_cols else S["cell"])
            line.append(Paragraph(str(v), st))
        data.append(line)
    scale = CW / sum(widths) if sum(widths) > CW else 1
    t = Table(data, colWidths=[w * scale for w in widths], repeatRows=1 if header else 0)
    sty = [("GRID", (0, 0), (-1, -1), 0.5, colors.black), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
           ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
           ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    if header:
        sty.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9E2F3")))
    t.setStyle(TableStyle(sty))
    out = [t, Spacer(1, 6)]
    return ([Paragraph(caption, S["tcap"])] + out) if caption else out


def fig(path, width, caption=None):
    w0, h0 = PILImage.open(path).size
    h = width * h0 / w0
    if h > 540:
        width, h = width * 540 / h, 540
    items = [Image(str(path), width=width, height=h)]
    if caption:
        items.append(Paragraph(caption, S["cap"]))
    return [KeepTogether(items)]


def fig_row(paths, width_each, captions, main_caption=None):
    cells = []
    for p, c in zip(paths, captions):
        w0, h0 = PILImage.open(p).size
        cells.append([Image(str(p), width=width_each, height=width_each * h0 / w0),
                      Paragraph(c, ParagraphStyle("c2", parent=S["cap"], fontSize=9.5, leading=11))])
    t = Table([cells], colWidths=[CW / len(paths)] * len(paths))
    t.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    items = [t]
    if main_caption:
        items.append(Paragraph(main_caption, S["cap"]))
    return [KeepTogether(items)]


def pseudo(lines, caption, explain):
    """lines: list of (indent, markup). Reproduces the dashed-rule algorithm blocks."""
    block = []
    for ind, txt in lines:
        st = ParagraphStyle("pl", parent=S["pseudo"], leftIndent=22 + ind * 16 + 16, firstLineIndent=-16)
        block.append(Paragraph(txt, st))
    block.append(Paragraph(DASH, S["dash"]))
    block.append(Paragraph(caption, ParagraphStyle("pc", parent=S["cap"], spaceBefore=4, spaceAfter=3)))
    out = [KeepTogether(block)] if len(lines) < 38 else block
    out.append(Paragraph(explain, S["body"]))
    out.append(Paragraph(DASH, S["dash"]))
    out.append(Spacer(1, 5))
    return out


def code(text):
    return Preformatted(text.rstrip("\n"), S["code"])


# ----------------------------------------------------------- document
def _decorate(canv, doc, number=True):
    canv.saveState()
    canv.setLineWidth(1.6); canv.setStrokeColor(colors.black)
    canv.rect(25, 22, W - 50, H - 47)
    lw = 356
    canv.drawImage(str(LOGO), (W - lw) / 2, H - 46 - lw * 234 / 1179, width=lw, height=lw * 234 / 1179)
    if number:
        canv.setFont(FONT, 10); canv.drawRightString(W - 40, 29, str(doc.page))
    canv.restoreState()


class ReportDoc(BaseDocTemplate):
    def __init__(self, path, toc_pages, **kw):
        super().__init__(path, pagesize=(W, H), **kw)
        self.toc_pages = toc_pages
        self.found = {}
        top = H - 128
        fr = Frame(LEFT, 60, CW, top - 60, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, id="f")
        self.addPageTemplates([
            PageTemplate("cover", frames=[fr], onPage=lambda c, d: _decorate(c, d, False)),
            PageTemplate("normal", frames=[fr], onPage=lambda c, d: _decorate(c, d, True)),
        ])

    def afterFlowable(self, f):
        k = getattr(f, "_toc", None)
        if k:
            self.found[k] = self.page
