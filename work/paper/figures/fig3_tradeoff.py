# -*- coding: utf-8 -*-
"""Figure 3: diverging bars of per-model val/test change vs the base backbone.

Style follows gallery diverging stacked bars (plot-is-all-you-need, essd-17-965
family): bars grow from a central zero line, red = degradation, blue = gain,
one pair of bars per model. Real data: paper/figures/data/tradeoff.json.

Usage: python fig3_tradeoff.py [en|zh]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = json.load(open(ROOT / "paper/figures/data/tradeoff.json", encoding="utf-8"))
LANG = sys.argv[1] if len(sys.argv) > 1 else "en"

L = {
    "en": {"title": "The stability–generalisation trade-off",
           "val": "Stable (val)", "test": "Flood (test)",
           "ylab": "RMSE change vs base (%)", "note": "negative = better"},
    "zh": {"title": "稳定性–泛化性权衡",
           "val": "稳定期（验证）", "test": "洪水期（测试）",
           "ylab": "相对 base 的 RMSE 变化（%）", "note": "负值 = 更好"},
}[LANG]

names = list(DATA.keys())
nv = len(names)
val = np.array([DATA[n]["val"] for n in names])
test = np.array([DATA[n]["test"] for n in names])

plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
                     "font.size": 8, "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(5.6, 4.6))
fig.subplots_adjust(left=0.24, right=0.96, top=0.89, bottom=0.12)

yy = np.arange(nv)
h = 0.36
b1 = ax.barh(yy - h / 2, val, h, color="#e36c73", edgecolor="#b4444c", lw=0.5,
             label=L["val"], zorder=3)
b2 = ax.barh(yy + h / 2, test, h, color="#4f8ef7", edgecolor="#2f6cc4", lw=0.5,
             label=L["test"], zorder=3)
for bars in (b1, b2):
    for rect in bars:
        w = rect.get_width()
        if abs(w) < 0.01:
            continue
        ax.text(w + (0.5 if w >= 0 else -0.5), rect.get_y() + rect.get_height() / 2,
                f"{w:+.1f}", va="center", ha="left" if w >= 0 else "right",
                fontsize=6.2, color="#333333")

ax.axvline(0, color="#333333", lw=1.0, zorder=4)
ax.set_yticks(yy, names, fontsize=7.5)
ax.invert_yaxis()
ax.set_xlabel(L["ylab"], fontweight="bold")
ax.set_title(L["title"], fontweight="bold", fontsize=10)
ax.grid(axis="x", color="#e8e8e8", lw=0.5)
ax.set_axisbelow(True)
ax.tick_params(length=0)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.legend(loc="upper right", frameon=False, fontsize=7.5, bbox_to_anchor=(1.0, 0.98))
ax.text(0.99, 0.01, L["note"], transform=ax.transAxes, ha="right", va="bottom",
        fontsize=6.5, color="#888888", style="italic")

out = ROOT / "paper/figures" / f"fig3_tradeoff_{LANG}"
fig.savefig(out.with_suffix(".png"), dpi=300, facecolor="white", bbox_inches="tight")
fig.savefig(out.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
print("saved", out)
