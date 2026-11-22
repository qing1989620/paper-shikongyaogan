# -*- coding: utf-8 -*-
"""Schematic of the RSP pipeline (publication-style vector figure)."""
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


def box(ax, x, y, w, h, text, fc, ec, fs=7.2, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.010,rounding_size=0.018",
                 linewidth=0.9, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=INK, zorder=3,
            fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(ax, p1, p2, color=INK, ls="-", lw=0.9):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=8,
                 linewidth=lw, color=color, linestyle=ls, zorder=1,
                 shrinkA=2, shrinkB=2))


fig, ax = plt.subplots(figsize=(7.0, 3.2))
ax.set_xlim(0, 10.2); ax.set_ylim(0, 4.2); ax.axis("off")

# ---- inputs: three stacked boxes -------------------------------------------
ax.text(0.82, 4.02, "observed inputs", ha="center", fontsize=6.6, color=GREY_E)
box(ax, 0.10, 3.18, 1.42, 0.62, "reference pair\n$(F_p,\\,C_p)$", GREY, GREY_E, fs=6.8)
box(ax, 0.10, 2.46, 1.42, 0.62, "reference pair\n$(F_n,\\,C_n)$", GREY, GREY_E, fs=6.8)
box(ax, 0.10, 1.74, 1.42, 0.62, "coarse at $t$\n$C_t$", GREY, GREY_E, fs=6.8)

# ---- backbone ---------------------------------------------------------------
box(ax, 1.98, 2.42, 1.52, 1.02, "frozen STFDiff\nbackbone\n1 denoising call", BLUE, BLUE_E, fs=7.0)
for yy in (3.49, 2.77, 2.05):
    arrow(ax, (1.54, yy), (1.96, min(max(yy, 2.5), 3.35)))

box(ax, 3.86, 2.52, 0.92, 0.82, "$x_1$\nfirst-step\noutput", "#ffffff", BLUE_E, fs=6.9)
arrow(ax, (3.52, 2.93), (3.84, 2.93))

# ---- self-calibration -------------------------------------------------------
box(ax, 1.98, 0.62, 1.52, 0.92, "self-calibration\nclosed-form OLS\n12 scalars", GREEN, GREEN_E, fs=6.9)
arrow(ax, (2.74, 2.40), (2.74, 1.56))
ax.text(2.82, 1.98, "reference pairs", fontsize=5.9, color=GREEN_E)

# ---- target -----------------------------------------------------------------
box(ax, 3.86, 0.62, 1.72, 0.92, "low-frequency target\n$T=\\frac{1}{2}(T_p+T_n)$\nfrom both anchors", ORANGE, ORANGE_E, fs=6.7)
arrow(ax, (3.52, 1.08), (3.84, 1.08))
arrow(ax, (4.62, 2.50), (4.62, 1.56), color=ORANGE_E)
ax.text(4.70, 2.02, "$C_t$", fontsize=6.4, color=ORANGE_E)

# ---- projection -------------------------------------------------------------
box(ax, 5.92, 2.40, 2.05, 1.02,
    "temporal projection\n$x_{\\mathrm{out}}=x_1-\\lambda\\,[\\mathrm{LP}(x_1)-T]$",
    "#fef9c3", "#a38a2a", fs=6.8)
arrow(ax, (4.80, 2.93), (5.90, 2.93))
arrow(ax, (5.60, 1.08), (6.90, 2.38), color=ORANGE_E, ls="--")
ax.text(6.05, 1.55, "$T$", fontsize=6.4, color=ORANGE_E)

# ---- output -----------------------------------------------------------------
box(ax, 8.28, 2.52, 1.30, 0.82, "fused fine\nimage $\\hat F_t$", "#ffffff", BLUE_E, fs=7.0)
arrow(ax, (7.99, 2.93), (8.26, 2.93))

ax.text(0.10, 0.28,
        "no prediction-date fine GT   ·   no backbone update   ·   no new network parameters   ·   1 NFE",
        fontsize=7.2, color=INK)
ax.text(0.10, 0.05,
        "high frequencies are untouched; only the low-frequency temporal residual is projected",
        fontsize=6.5, color=GREY_E, style="italic")

fig.tight_layout(pad=0.15)
out = Path("paper/figures/method_rsp.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
print("saved", out)
