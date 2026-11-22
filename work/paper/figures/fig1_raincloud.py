# -*- coding: utf-8 -*-
"""Figure 1: raincloud plot of patch-level RMSE for every trained backbone.

Layout/marks/palette follow gallery/raincloud-median-badges (plot-is-all-you-need).
Real data: results/*_lgc_{val,test}.csv -> paper/figures/data/rmse_dist.json.

Usage: python fig1_raincloud.py [en|zh]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parents[2]
DATA = json.load(open(ROOT / "paper/figures/data/rmse_dist.json", encoding="utf-8"))
import matplotlib as _mpl
LANG = sys.argv[1] if len(sys.argv) > 1 else "en"
if LANG == "zh":
    _mpl.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    _mpl.rcParams["axes.unicode_minus"] = False

ORDER = ["base", "lf", "hf", "bandw", "delsup", "full", "delta", "gate", "reinject"]
SPLIT_FILTER = sys.argv[2] if len(sys.argv) > 2 else "test"   # one split per figure
HUES = ["#0a6aa8", "#1489c4", "#12a5c8", "#3dbfe0", "#e61e71", "#8a7ab5", "#2eb1d0", "#f48b8b", "#74b3f0"]

L = {
    "en": {"ylabel": "RMSE (reflectance, [0,1])", "val": "Stable (val)",
           "test": "Flood (test)", "valtag": "stable", "testtag": "flood"},
    "zh": {"ylabel": "RMSE（反射率，[0,1]）", "val": "稳定期（验证）",
           "test": "洪水期（测试）", "valtag": "稳定", "testtag": "洪水"},
}[LANG]

_FF = (["Microsoft YaHei", "SimHei", "DejaVu Sans"] if LANG == "zh"
       else ["Arial", "Helvetica", "DejaVu Sans"])
plt.rcParams.update({"font.family": _FF, "font.size": 7, "pdf.fonttype": 42})
if LANG == "zh":
    plt.rcParams["axes.unicode_minus"] = False
fig, ax = plt.subplots(figsize=(3.45, 2.9))
fig.subplots_adjust(left=0.17, right=0.98, top=0.96, bottom=0.22)

x = 0.0
xt, xl = [], []
rng = np.random.default_rng(7)
for i, key in enumerate(ORDER):
    hue = HUES[i % len(HUES)]
    v = np.array(DATA[f"{key}|{SPLIT_FILTER}"]["values"]) / 2.0
    yy = np.linspace(v.min(), v.max(), 180)
    dens = gaussian_kde(v)(yy)
    ax.fill_betweenx(yy, x + 0.05, x + 0.05 + dens / dens.max() * 0.52,
                     color=hue, alpha=0.85, lw=0, zorder=2)
    ax.scatter(x - 0.10 - rng.random(len(v)) * 0.30, v, s=2.4, color=hue,
               linewidths=0, alpha=0.8, zorder=3)
    med = float(np.median(v))
    ax.scatter(x + 0.05, med, s=70, fc="white", ec="black", lw=0.8, zorder=4)
    ax.text(x + 0.05, med, f"{med*1000:.0f}", ha="center", va="center",
            fontsize=4.6, color="#d0021b", fontweight="bold", zorder=5)
    xt.append(x); xl.append(DATA[f"{key}|{SPLIT_FILTER}"]["name"])
    x += 1.0

ax.set_xticks(xt)
ax.set_xticklabels(xl, rotation=15, ha="right", fontsize=6.5)
ax.set_xlim(-0.65, x - 0.35)
ax.set_ylabel(L["ylabel"], fontsize=7)
ax.tick_params(length=2, labelsize=6.5)
ax.grid(axis="y", color="#ececec", lw=0.4)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color("#9a9a9a"); ax.spines[side].set_linewidth(0.7)
ax.text(0.02, 0.97, L[f"{SPLIT_FILTER}tag"], transform=ax.transAxes,
        fontsize=6.5, style="italic", color="#555555", va="top")

out = ROOT / "paper/figures" / f"fig1_raincloud_{LANG}"
fig.savefig(out.with_suffix(".png"), dpi=300, facecolor="white")
fig.savefig(out.with_suffix(".pdf"), facecolor="white")
print("saved", out)
