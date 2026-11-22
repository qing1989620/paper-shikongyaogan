# -*- coding: utf-8 -*-
"""Two-panel mechanism figure for the gated RSP (paper version)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "serif", "font.size": 8,
                     "mathtext.fontset": "dejavuserif"})
INK = "#1f2933"; BLUE = "#dbeafe"; BLUE_E = "#3b6ea5"
GREEN = "#dcfce7"; GREEN_E = "#3f7d5a"
ORANGE = "#ffedd5"; ORANGE_E = "#b4652a"
GREY = "#f3f4f6"; GREY_E = "#6b7280"
YEL = "#fef9c3"; YEL_E = "#a38a2a"


def box(ax, x, y, w, h, text, fc, ec, fs=6.8, bold=False, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.015",
                                linewidth=0.9, edgecolor=ec, facecolor=fc,
                                linestyle=ls, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=INK, zorder=3, linespacing=1.3,
            fontweight="bold" if bold else "normal")


def arr(ax, p1, p2, color=INK, ls="-", lw=0.8):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=7,
                                 linewidth=lw, color=color, linestyle=ls,
                                 zorder=1, shrinkA=1.5, shrinkB=1.5))


fig, ax = plt.subplots(figsize=(7.05, 3.55))
ax.set_xlim(0, 10.4); ax.set_ylim(0, 5.2); ax.axis("off")

# frame
ax.add_patch(FancyBboxPatch((0.05, 0.05), 10.3, 5.1,
             boxstyle="round,pad=0.01,rounding_size=0.03",
             linewidth=0.9, edgecolor=GREY_E, facecolor="none", linestyle=(0, (4, 3))))
ax.text(0.22, 4.94, "Reference-Pair Self-Calibrated Temporal Projection (RSP)",
        fontsize=8.2, color=INK, fontweight="bold")

# ---- row 1 ----------------------------------------------------------------
ax.text(0.95, 4.62, "observed inputs", ha="center", fontsize=6.2, color=GREY_E)
box(ax, 0.14, 3.98, 1.62, 0.50, "reference pair $(F_p,C_p)$", GREY, GREY_E, fs=6.2)
box(ax, 0.14, 3.42, 1.62, 0.50, "reference pair $(F_n,C_n)$", GREY, GREY_E, fs=6.2)
box(ax, 0.14, 2.86, 1.62, 0.50, "coarse $C_t$", GREY, GREY_E, fs=6.4)

box(ax, 2.10, 3.26, 1.62, 0.86, "frozen STFDiff\nbackbone\n1 denoising call", BLUE, BLUE_E, fs=6.6)
for yy in (4.23, 3.67, 3.11):
    arr(ax, (1.78, yy), (2.08, min(max(yy, 3.35), 4.05)))
box(ax, 4.02, 3.36, 0.86, 0.66, "$x_1$\nfirst-step", "#ffffff", BLUE_E, fs=6.6)
arr(ax, (3.74, 3.69), (4.00, 3.69))

box(ax, 5.22, 3.24, 1.92, 0.92,
    "(b) gate\n$g=1/(1+\\bar{|M|}/s)$\n$x_{\\mathrm{out}}=x_1-\\lambda gM$", YEL, YEL_E, fs=6.3)
arr(ax, (4.90, 3.69), (5.20, 3.69))
box(ax, 7.46, 3.36, 1.30, 0.66, "fused fine\n$\\hat F_t$", "#ffffff", BLUE_E, fs=6.8)
arr(ax, (7.16, 3.69), (7.44, 3.69))
ax.text(8.95, 3.88, "1 NFE\n0 new params\nno fine GT", fontsize=6.0, color=GREY_E, va="top")

# ---- panel (a) ------------------------------------------------------------
ax.add_patch(FancyBboxPatch((0.14, 0.16), 4.86, 2.42,
             boxstyle="round,pad=0.008,rounding_size=0.02",
             linewidth=0.9, edgecolor=GREY_E, facecolor="none", linestyle=(0, (4, 3))))
ax.text(0.26, 2.40, "(a) per-sample calibration (closed form)", fontsize=7.4, color=GREY_E)
rows = [("OLS per band", "12 scalars  ·  PRISM'26"),
        ("TLS / Deming", "errors-in-variables  ·  GSC'26"),
        ("Huber IRLS", "robust to outliers  ·  DCDP'26"),
        ("Ridge to identity", "variance reduction  ·  GW'26")]
for i, (nm, sub) in enumerate(rows):
    yy = 1.92 - i * 0.44
    box(ax, 0.26, yy, 1.55, 0.34, nm, GREEN, GREEN_E, fs=6.2)
    ax.text(1.90, yy + 0.17, sub, fontsize=5.8, color=GREY_E, va="center")
ax.text(0.26, 0.30, "$\\mathrm{LP}(F_n-F_p)\\approx a\\,\\mathrm{LP}(C_n-C_p)+b$   on the sample's own anchors",
        fontsize=6.0, color=GREY_E)

# ---- panel (b) ------------------------------------------------------------
ax.add_patch(FancyBboxPatch((5.22, 0.16), 5.02, 2.42,
             boxstyle="round,pad=0.008,rounding_size=0.02",
             linewidth=0.9, edgecolor=GREY_E, facecolor="none", linestyle=(0, (4, 3))))
ax.text(5.34, 2.40, "(b) gated projection detail", fontsize=7.4, color=GREY_E)
box(ax, 5.36, 1.74, 1.36, 0.50, "residual\n$M=\\mathrm{LP}(x_1)-T$", ORANGE, ORANGE_E, fs=6.0)
arr(ax, (6.76, 1.99), (6.94, 1.99), color=ORANGE_E)
box(ax, 6.98, 1.74, 1.22, 0.50, "gate\ng", ORANGE, ORANGE_E, fs=6.4)
arr(ax, (8.24, 1.99), (8.42, 1.99), color=ORANGE_E)
box(ax, 8.46, 1.74, 1.58, 0.50, "project\n$x_1-\\lambda gM$", YEL, YEL_E, fs=6.2)
ax.text(5.36, 1.30, "small $\\bar{|M|}$ → trust the correction", fontsize=5.9, color=GREY_E)
ax.text(5.36, 1.02, "large $\\bar{|M|}$ → shrink toward $x_1$", fontsize=5.9, color=GREY_E)
ax.text(5.36, 0.52, "only the low-frequency temporal residual is corrected;\n"
                    "high-frequency detail from the backbone is untouched",
        fontsize=5.9, color=GREY_E, style="italic")

# arrow from (a) to gate
arr(ax, (4.72, 2.24), (5.60, 3.20), color=ORANGE_E, ls="--")
ax.text(4.70, 2.62, "$T$", fontsize=6.0, color=ORANGE_E)

fig.tight_layout(pad=0.12)
out = Path("paper/figures/method_rsp.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
print("saved", out)
