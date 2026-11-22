# -*- coding: utf-8 -*-
"""Build an editable PPTX mechanism figure imitating the CURE layout.

Layout (13.33x7.5 in, 16:9):
  outer dashed frame "RSP"
  row 1 : inputs -> frozen backbone -> x1 -> gated projection -> output
  row 2 : (a) calibration estimators   (b) gated projection detail
All elements are native PowerPoint shapes, so the file stays fully editable.
"""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

INK = RGBColor(0x1F, 0x29, 0x33)
BLUE = RGBColor(0xDB, 0xEA, 0xFE)
BLUE_E = RGBColor(0x3B, 0x6E, 0xA5)
GREEN = RGBColor(0xDC, 0xFC, 0xE7)
GREEN_E = RGBColor(0x3F, 0x7D, 0x5A)
ORANGE = RGBColor(0xFF, 0xED, 0xD5)
ORANGE_E = RGBColor(0xB4, 0x65, 0x2A)
GREY = RGBColor(0xF3, 0xF4, 0xF6)
GREY_E = RGBColor(0x6B, 0x72, 0x80)
YELLOW = RGBColor(0xFE, 0xF9, 0xC3)
YELLOW_E = RGBColor(0xA3, 0x8A, 0x2A)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)


def box(slide, x, y, w, h, text, fill, line, size=11, bold=False, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
        dashed=False, align=PP_ALIGN.CENTER):
    sh = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = fill
    sh.line.color.rgb = line; sh.line.width = Pt(1.0)
    if dashed:
        from pptx.enum.dml import MSO_LINE_DASH_STYLE
        sh.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    tf = sh.text_frame
    tf.word_wrap = True
    tf.margin_left = Pt(3); tf.margin_right = Pt(3)
    tf.margin_top = Pt(2); tf.margin_bottom = Pt(2)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run(); run.text = text
    run.font.size = Pt(size); run.font.bold = bold
    run.font.color.rgb = INK; run.font.name = "Times New Roman"
    return sh


def arrow(slide, x, y, w, h, color=INK):
    sh = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = color
    sh.line.color.rgb = color
    return sh


def label(slide, x, y, w, h, text, size=9, color=GREY_E, italic=False, align=PP_ALIGN.LEFT):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = Pt(0); tf.margin_top = Pt(0)
    p = tf.paragraphs[0]; p.alignment = align
    r = p.add_run(); r.text = text
    r.font.size = Pt(size); r.font.color.rgb = color; r.font.italic = italic
    r.font.name = "Times New Roman"
    return tb


def main():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = WHITE

    # outer dashed frame
    frame = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.12), Inches(0.12),
                                   Inches(13.09), Inches(7.26))
    frame.fill.background()
    frame.line.color.rgb = GREY_E
    frame.line.width = Pt(1.1)
    from pptx.enum.dml import MSO_LINE_DASH_STYLE
    frame.line.dash_style = MSO_LINE_DASH_STYLE.DASH

    label(slide, 0.30, 0.18, 6.0, 0.35, "Reference-Pair Self-Calibrated Temporal Projection (RSP)", size=13, color=INK)

    # ---------------- row 1: main pipeline ----------------
    # inputs
    box(slide, 0.30, 0.78, 2.35, 2.42, "", GREY, GREY_E)
    label(slide, 0.42, 0.84, 2.1, 0.3, "Observed inputs (never $F_t$)", size=9, color=GREY_E, align=PP_ALIGN.CENTER)
    box(slide, 0.44, 1.14, 2.06, 0.52, "reference pair $(F_p,\\,C_p)$", WHITE, BLUE_E, size=9)
    box(slide, 0.44, 1.74, 2.06, 0.52, "reference pair $(F_n,\\,C_n)$", WHITE, BLUE_E, size=9)
    box(slide, 0.44, 2.34, 2.06, 0.52, "coarse at prediction date $C_t$", WHITE, BLUE_E, size=9)

    # backbone
    box(slide, 2.95, 1.16, 2.05, 1.62, "Frozen STFDiff backbone\n1 denoising call (1 NFE)", BLUE, BLUE_E, size=11, bold=True)
    arrow(slide, 2.68, 1.85, 0.22, 0.20)
    label(slide, 2.86, 2.82, 2.3, 0.3, "all five condition tensors", size=8)

    # x1
    box(slide, 5.20, 1.40, 1.35, 1.12, "$x_1$\nfirst-step\nfine output", WHITE, BLUE_E, size=10)
    arrow(slide, 5.02, 1.85, 0.16, 0.20)

    # gated projection (b)
    box(slide, 6.80, 1.18, 2.70, 1.60,
        "(b) Mismatch-gated projection\n$g=\\dfrac{1}{1+\\bar{|M|}/s}$\n"
        "$x_{\\mathrm{out}}=x_1-\\lambda\\,g\\,(\\mathrm{LP}(x_1)-T)$",
        YELLOW, YELLOW_E, size=9.5, bold=False)
    arrow(slide, 6.58, 1.85, 0.18, 0.20)
    label(slide, 6.70, 1.02, 2.9, 0.25, "T from (a); s frozen on validation", size=8)

    # output
    box(slide, 9.78, 1.42, 1.55, 1.12, "fused fine\nimage $\\hat F_t$", WHITE, BLUE_E, size=10)
    arrow(slide, 9.52, 1.85, 0.22, 0.20)

    # calibration feeds gate
    arrow(slide, 8.15, 3.05, 0.18, 0.34, color=ORANGE_E)

    # properties strip
    label(slide, 11.45, 1.20, 1.75, 0.95,
          "1 NFE\n0 new params\nno fine GT\nfrozen backbone", size=9, color=GREY_E)

    # ---------------- row 2: two panels ----------------
    # (a) estimators
    box(slide, 0.30, 3.62, 6.05, 3.52, "", WHITE, GREY_E, dashed=True)
    label(slide, 0.44, 3.68, 5.5, 0.30, "(a) Per-sample calibration estimators (closed form)", size=11, color=GREY_E)

    rows = [("OLS per band", "closed form, 12 scalars", "reference"),
            ("TLS / Deming", "errors-in-variables", "GSC 2026"),
            ("Huber IRLS", "robust to change outliers", "DCDP 2026"),
            ("Ridge to identity", "variance reduction", "Guidance 2026"),
            ("Span-normalised", "anchor vs prediction span", "WaFT 2026")]
    y0 = 3.98
    for i, (name, desc, src) in enumerate(rows):
        yy = y0 + i * 0.58
        box(slide, 0.52, yy, 2.28, 0.46, name, GREEN, GREEN_E, size=9.5)
        label(slide, 2.92, yy + 0.06, 1.95, 0.34, desc, size=8.5, color=INK)
        label(slide, 4.95, yy + 0.06, 1.25, 0.34, src, size=7.5, color=GREY_E, italic=True)

    label(slide, 0.52, 6.86, 5.6, 0.25,
          "fit on the sample's own anchors:  $\\mathrm{LP}(F_n-F_p)\\approx a\\,\\mathrm{LP}(C_n-C_p)+b$",
          size=8.5, color=GREY_E)

    # (b) gate detail
    box(slide, 6.62, 3.62, 6.45, 3.52, "", WHITE, GREY_E, dashed=True)
    label(slide, 6.76, 3.68, 6.0, 0.30, "(b) Gated projection detail", size=11, color=GREY_E)

    box(slide, 6.86, 4.02, 1.62, 0.70, "residual\n$M=\\mathrm{LP}(x_1)-T$", ORANGE, ORANGE_E, size=9)
    arrow(slide, 8.56, 4.26, 0.20, 0.20, color=ORANGE_E)
    box(slide, 8.86, 4.02, 1.55, 0.70, "gate\n$g=1/(1+\\bar{|M|}/s)$", ORANGE, ORANGE_E, size=9)
    arrow(slide, 10.49, 4.26, 0.20, 0.20, color=ORANGE_E)
    box(slide, 10.79, 4.02, 1.90, 0.70, "project\n$x_{\\mathrm{out}}=x_1-\\lambda g M$", YELLOW, YELLOW_E, size=9)

    box(slide, 6.86, 5.00, 2.55, 0.62, "mismatch $\\bar{|M|}$ small\n$\\Rightarrow$ trust the correction", WHITE, GREY_E, size=8.5)
    box(slide, 9.70, 5.00, 3.00, 0.62, "mismatch $\\bar{|M|}$ large\n$\\Rightarrow$ shrink toward $x_1$", WHITE, GREY_E, size=8.5)

    label(slide, 6.86, 5.86, 5.9, 0.80,
          "Only the low-frequency temporal residual is corrected; high-frequency detail produced by\n"
          "the backbone is untouched. The gate introduces no learned parameter: s is a scalar frozen\n"
          "on the validation split, in the spirit of conformal risk control.",
          size=8.5, color=GREY_E)

    out = Path("paper/figures/RSP_mechanism.pptx")
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out))
    print("saved", out)


if __name__ == "__main__":
    main()
