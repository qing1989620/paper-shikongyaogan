"""Build a native PPTX and matching vector PDF from the reviewed layout spec."""

import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_CONNECTOR
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Pt


FIG = Path(__file__).resolve().parents[1] / "source" / "figures"


def color(hex_string):
    return RGBColor.from_string(hex_string.lstrip("#"))


def editable_deck(spec):
    prs = Presentation()
    canvas = spec["canvas_points"]
    prs.slide_width, prs.slide_height = Pt(canvas["width"]), Pt(canvas["height"])
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    for region in spec["regions"]:
        for op in region:
            if op["type"] == "add_textbox":
                node = slide.shapes.add_textbox(*(Pt(op[k]) for k in ("left", "top", "width", "height")))
                node.name = op["name"]
                if op.get("fill_color"):
                    node.fill.solid()
                    node.fill.fore_color.rgb = color(op["fill_color"])
                    node.line.color.rgb = color(op["line_color"])
                    node.line.width = Pt(op["line_width"])
                frame = node.text_frame
                frame.text = op["text"]
                frame.word_wrap = op["word_wrap"]
                frame.vertical_anchor = MSO_ANCHOR.MIDDLE
                for side in ("top", "bottom", "left", "right"):
                    setattr(frame, f"margin_{side}", Pt(op["margin_" + side]))
                for paragraph in frame.paragraphs:
                    paragraph.alignment = PP_ALIGN.CENTER if op.get("alignment") == "center" else PP_ALIGN.LEFT
                    paragraph.space_after = Pt(0)
                    paragraph.space_before = Pt(0)
                    paragraph.font.name = op["font_name"]
                    paragraph.font.size = Pt(op["font_size"])
                    paragraph.font.bold = op.get("bold", False)
                    paragraph.font.color.rgb = color(op["font_color"])
            else:
                line = slide.shapes.add_connector(
                    MSO_CONNECTOR.STRAIGHT, Pt(op["begin_x"]), Pt(op["begin_y"]),
                    Pt(op["end_x"] - op["end_clearance"]), Pt(op["end_y"]),
                )
                line.name = op["name"]
                line.line.color.rgb = color(op["line_color"])
                line.line.width = Pt(op["line_width"])
                if op["line_dash"] == "dash":
                    line.line.dash_style = MSO_LINE_DASH_STYLE.DASH
                arrow = OxmlElement("a:tailEnd")
                arrow.set("type", "triangle")
                line._element.spPr.ln.append(arrow)
                style = line._element.find(qn("p:style"))
                style.find(qn("a:effectRef")).set("idx", "0")
    prs.save(FIG / "STF_bridge_editable.pptx")


def vector_figure(spec):
    size = spec["canvas_points"]
    width = size["width"]
    top, bottom = size["crop_top"], size["crop_bottom"]
    fig = plt.figure(figsize=(width / 72, (bottom - top) / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, width), ylim=(bottom, top))
    ax.axis("off")
    for region in spec["regions"]:
        for op in region:
            if op["type"] == "add_line":
                ax.add_patch(FancyArrowPatch(
                    (op["begin_x"], op["begin_y"]),
                    (op["end_x"] - op["end_clearance"], op["end_y"]),
                    arrowstyle="-|>", mutation_scale=13, shrinkA=0, shrinkB=0,
                    color=op["line_color"], lw=op["line_width"],
                    linestyle="--" if op["line_dash"] == "dash" else "-",
                ))
                continue
            x, y, w, h = (op[k] for k in ("left", "top", "width", "height"))
            center = op.get("alignment") == "center"
            if op.get("fill_color"):
                ax.add_patch(Rectangle((x, y), w, h, facecolor=op["fill_color"],
                                       edgecolor=op["line_color"], lw=op["line_width"]))
            ax.text(x + w / 2 if center else x, y + h / 2, op["text"],
                    ha="center" if center else "left", va="center", linespacing=1.18,
                    color=op["font_color"], size=op["font_size"], family="Arial",
                    weight="bold" if op.get("bold") else "normal")
    fig.savefig(FIG / "bridge.pdf", facecolor="white")
    fig.savefig(FIG / "bridge.png", dpi=220, facecolor="white")
    plt.close(fig)


def main():
    mpl.rcParams["pdf.fonttype"] = 42
    spec = json.loads((FIG / "bridge_design.json").read_text(encoding="utf-8"))
    editable_deck(spec)
    vector_figure(spec)


if __name__ == "__main__":
    main()
