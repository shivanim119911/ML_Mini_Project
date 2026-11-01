"""Layout helpers that reproduce the AISSMS IOIT mini-project report format."""
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Flowable, Frame, HRFlowable, Image, KeepTogether,
                                PageBreak, PageTemplate, Paragraph, Preformatted, Spacer, Table,
                                TableStyle)

HERE = Path(__file__).resolve().parent
LOGO = HERE / "assets" / "logo-000.jpg"
FD = "/usr/share/fonts/truetype/liberation/"
for n, f in [("Times", "LiberationSerif-Regular"), ("Times-Bold", "LiberationSerif-Bold"),
             ("Times-Italic", "LiberationSerif-Italic"), ("Times-BoldItalic", "LiberationSerif-BoldItalic")]:
    pdfmetrics.registerFont(TTFont(n, FD + f + ".ttf"))
pdfmetrics.registerFontFamily("Times", normal="Times", bold="Times-Bold", italic="Times-Italic",
                              boldItalic="Times-BoldItalic")

W, H = A4
LM = 56
FRAME_W = W - 2 * LM
TOP = 135          # distance from top edge to start of content
BOT = 52
HEAD_BLUE = colors.HexColor("#d9e2f3")


def S(name, **kw):
    base = dict(fontName="Times", fontSize=10.5, leading=14.2, alignment=TA_JUSTIFY, spaceAfter=5)
    base.update(kw)
    return ParagraphStyle(name, **base)


ST = {
    "body": S("body"),
    "bodyL": S("bodyL", alignment=TA_LEFT),
    "h1": S("h1", fontName="Times-Bold", fontSize=16, leading=20, alignment=TA_LEFT, spaceBefore=2, spaceAfter=5),
    "h2": S("h2", fontName="Times-Bold", fontSize=12.5, leading=16, alignment=TA_LEFT, spaceBefore=6, spaceAfter=4),
    "h3": S("h3", fontName="Times-Bold", fontSize=10.5, leading=14, alignment=TA_LEFT, spaceBefore=4, spaceAfter=3),
    "bullet": S("bullet", leftIndent=24, bulletIndent=10, spaceAfter=2.5),
    "num": S("num", leftIndent=26, bulletIndent=8, spaceAfter=3),
    "cap": S("cap", fontName="Times-Bold", alignment=TA_CENTER, spaceBefore=3, spaceAfter=6),
    "tcap": S("tcap", fontName="Times-Bold", alignment=TA_LEFT, spaceBefore=4, spaceAfter=3),
    "cell": S("cell", fontSize=9, leading=11.2, alignment=TA_LEFT, spaceAfter=0),
    "cellB": S("cellB", fontName="Times-Bold", fontSize=9, leading=11.2, alignment=TA_LEFT, spaceAfter=0),
    "cellC": S("cellC", fontSize=9, leading=11.2, alignment=TA_CENTER, spaceAfter=0),
    "pseudo": S("pseudo", fontSize=9.4, leading=12, alignment=TA_LEFT, spaceAfter=0),
    "center": S("center", alignment=TA_CENTER, fontSize=12, leading=18, spaceAfter=8),
    "centerB": S("centerB", alignment=TA_CENTER, fontSize=12, leading=18, spaceAfter=8, fontName="Times-Bold"),
    "code": ParagraphStyle("code", fontName="Times", fontSize=8.2, leading=9.6),
    "note": S("note", fontName="Times-Italic", fontSize=9.6, leading=12.5, alignment=TA_JUSTIFY),
}


class H1(Paragraph):
    """Section heading that reports itself to the TOC."""
    def __init__(self, text, key):
        super().__init__(text, ST["h1"])
        self.toc_key = key


class DotRow(Flowable):
    def __init__(self, text, page):
        super().__init__()
        self.text, self.page = text, page

    def wrap(self, aw, ah):
        self.aw = aw
        return aw, 22

    def draw(self):
        c = self.canv
        c.setFont("Times", 12)
        c.drawString(12, 6, self.text)
        c.setFont("Times-Bold", 12)
        c.drawRightString(self.aw, 6, str(self.page))
        x0 = 12 + pdfmetrics.stringWidth(self.text, "Times", 12) + 3
        x1 = self.aw - pdfmetrics.stringWidth(str(self.page), "Times-Bold", 12) - 3
        c.setFont("Times", 12)
        dots = "." * max(0, int((x1 - x0) / pdfmetrics.stringWidth(".", "Times", 12)))
        c.drawString(x0, 6, dots)


class ReportDoc(BaseDocTemplate):
    def __init__(self, path, pages):
        super().__init__(str(path), pagesize=A4, leftMargin=LM, rightMargin=LM,
                         topMargin=TOP, bottomMargin=BOT, title="ML Lab Mini Project Report",
                         author="AISSMS IOIT")
        self.found = {}
        fr = Frame(LM, BOT, FRAME_W, H - TOP - BOT, id="f", leftPadding=0, rightPadding=0,
                   topPadding=0, bottomPadding=0)
        self.addPageTemplates([PageTemplate(id="p", frames=[fr], onPage=self.decorate)])

    def decorate(self, c, doc):
        c.saveState()
        c.setLineWidth(1.6)
        c.rect(26, 26, W - 52, H - 52)
        lw = 360
        lh = lw * 234 / 1179
        c.drawImage(str(LOGO), (W - lw) / 2, H - 56 - lh, lw, lh)
        if doc.page > 1:
            c.setFont("Times", 10.5)
            c.drawRightString(W - 38, 31, str(doc.page))
        c.restoreState()

    def afterFlowable(self, fl):
        if isinstance(fl, H1):
            self.found[fl.toc_key] = self.page


def P(text, style="body"):
    return Paragraph(text, ST[style])


def bullets(items, style="bullet"):
    return [Paragraph(t, ST[style], bulletText="\u2022") for t in items]


def numbered(items):
    return [Paragraph(t, ST["num"], bulletText=f"{i}.") for i, t in enumerate(items, 1)]


def img(path, width, max_h=330):
    w, h = PILImage.open(path).size
    sc = width / w
    if h * sc > max_h:
        sc = max_h / h
    return Image(str(path), w * sc, h * sc)


def figure(path, caption, width=330, max_h=330):
    return KeepTogether([img(path, width, max_h), Paragraph(caption, ST["cap"])])


def table(rows, colw, caption=None, head=True, center_cols=(), fs=None):
    data = []
    for r, row in enumerate(rows):
        out = []
        for c, v in enumerate(row):
            st = "cellB" if (head and r == 0) else ("cellC" if c in center_cols else "cell")
            out.append(Paragraph(str(v), ST[st]))
        data.append(out)
    t = Table(data, colWidths=colw, repeatRows=1 if head else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.5, colors.black), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
             ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    if head:
        style.append(("BACKGROUND", (0, 0), (-1, 0), HEAD_BLUE))
    t.setStyle(TableStyle(style))
    items = [Paragraph(caption, ST["tcap"])] if caption else []
    return items + [t, Spacer(1, 6)]


def dash():
    return HRFlowable(width="100%", thickness=0.6, color=colors.black, dash=(3, 2), spaceBefore=2, spaceAfter=3)


def pseudo(lines, caption):
    """lines: list of (indent_level, html_text)."""
    out = [dash()]
    for ind, t in lines:
        out.append(Paragraph(t, ParagraphStyle("p", parent=ST["pseudo"], leftIndent=14 + 16 * ind)))
    out += [dash(), Paragraph(caption, ST["cap"])]
    return out


def code_block(text):
    return Preformatted(text, ST["code"])


def note_box(text):
    t = Table([[Paragraph(text, ST["note"])]], colWidths=[FRAME_W])
    t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.8, colors.black),
                           ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f3f3f3")),
                           ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                           ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    return [t, Spacer(1, 6)]
