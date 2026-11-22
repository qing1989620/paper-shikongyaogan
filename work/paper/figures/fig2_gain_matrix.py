# -*- coding: utf-8 -*-
"""Figure 2: bubble matrix of inference-variant gains (15 variants x 4 splits).

Layout/palette follow gallery/bubble-matrix-upper (plot-is-all-you-need):
bubble size and colour encode the relative RMSE change of each inference-time
variant on each split. Red = worse, blue = better; the size/colour legend sits
below the matrix, outside the data area.

Usage: python fig2_gain_matrix.py [en|zh]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[2]
DATA = json.load(open(ROOT / "paper/figures/data/gain_matrix.json", encoding="utf-8"))
LANG = sys.argv[1] if len(sys.argv) > 1 else "en"

SPLITS = ["LGC-val", "LGC-test", "CIA-val", "CIA-test"]
if LANG == "zh":
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    SPLIT_LABELS = ["LGC\n验证", "LGC\n测试", "CIA\n验证", "CIA\n测试"]
else:
    SPLIT_LABELS = ["LGC\nval", "LGC\ntest", "CIA\nval", "CIA\ntest"]
YLAB = {"en": "Inference-time variant", "zh": "推理期变体"}[LANG]
NOTE = {"en": "Relative RMSE change (%)", "zh": "相对 RMSE 变化（%）"}[LANG]
if LANG == "zh":
    matplotlib.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]

names = list(DATA.keys())
n = len(names)
M = np.full((n, len(SPLITS)), np.nan)
for r, nm in enumerate(names):
    for c, sp in enumerate(SPLITS):
        if sp in DATA[nm]:
            M[r, c] = DATA[nm][sp]

CMAP = LinearSegmentedColormap.from_list(
    "blue_white_red", ["#2a2ad0", "#9aa0ee", "#f7f4f4", "#f09a8c", "#c8141e"])
lim = max(abs(np.nanmin(M)), abs(np.nanmax(M)))
norm = TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim)
size = lambda v: 10 + 240 * abs(v) / lim

plt.rcParams.update({"font.family": ["Arial", "Helvetica", "DejaVu Sans"],
                     "font.size": 7, "pdf.fonttype": 42, "axes.linewidth": 0.8})
fig, ax = plt.subplots(figsize=(4.0, 5.4))
fig.subplots_adjust(left=0.30, right=0.98, top=0.96, bottom=0.24)

for r in range(n):
    for c in range(len(SPLITS)):
        v = M[r, c]
        if np.isnan(v):
            continue
        ax.scatter(c, r, s=size(v), c=[CMAP(norm(v))],
                   edgecolors="#333333", linewidths=0.4, zorder=3)

ax.set_xlim(-0.6, len(SPLITS) - 0.4)
ax.set_ylim(n - 0.4, -0.6)
ax.set_xticks(range(len(SPLITS)), SPLIT_LABELS, fontsize=7)
ax.set_yticks(range(n), names, fontsize=6.5)
ax.grid(color="#e4e4e4", lw=0.5)
ax.set_axisbelow(True)
ax.tick_params(length=2.5)
# size/colour legend below the matrix, outside the data area
for k, t in enumerate([-lim, -lim / 2, 0, lim / 2, lim]):
    xx = 0.75 + 0.82 * k
    ax.scatter(xx, n + 1.35, s=size(t), c=[CMAP(norm(t))], edgecolors="#333333",
               linewidths=0.4, zorder=3, clip_on=False)
    ax.text(xx, n + 2.35, f"{t:+.0f}", ha="center", fontsize=6, clip_on=False)
ax.text(len(SPLITS) / 2 - 0.5, n + 3.2, NOTE, ha="center", va="center",
        fontsize=6.2, clip_on=False)

out = ROOT / "paper/figures" / f"fig2_gain_matrix_{LANG}"
fig.savefig(out.with_suffix(".png"), dpi=300, facecolor="white", bbox_inches="tight")
fig.savefig(out.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
print("saved", out)
