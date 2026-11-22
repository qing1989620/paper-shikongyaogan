"""Draw paper figures using the frozen validation summaries."""

import csv
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "evidence"
FIGURES = ROOT / "source" / "figures"
INK = "#263C42"
GRID = "#e6edf0"
COLORS = {"base": "#6589CC", "i02_bridge": "#499082", "i01_sensor": "#9271AC"}


def read(name):
    with (DATA / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    for edge in ("left", "bottom"):
        ax.spines[edge].set_color("#a6b7bf")
        ax.spines[edge].set_linewidth(.65)
    ax.tick_params(color="#607781", labelcolor=INK, width=.6, length=2.5)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(FIGURES / (name + ".pdf"), facecolor="white")
    fig.savefig(FIGURES / (name + ".png"), dpi=300, facecolor="white")
    plt.close(fig)


def trajectory_and_gain(locale):
    zh = locale == "zh"
    labels = {
        "base": "基线" if zh else "Base",
        "i02_bridge": "I02 桥" if zh else "I02 bridge",
        "i01_sensor": "I01 算子" if zh else "I01 sensor",
    }
    fig, (curve, effects) = plt.subplots(
        1, 2, figsize=(7.0, 2.68), gridspec_kw={"width_ratios": [1.32, 1.0]},
    )
    fig.subplots_adjust(left=.082, right=.978, top=.865, bottom=.23, wspace=.42)
    scores = {row["method"]: row for row in read("scores.csv")}
    curves = read("curves.csv")
    for method, ls, symbol in zip(labels, ("-", "-.", "--"), ("o", "D", "s")):
        rows = [r for r in curves if r["method"] == method]
        epochs = [int(row["epoch"]) for row in rows]
        rmse = np.array([float(row["rmse"]) for row in rows]) * 1000
        curve.plot(epochs, rmse, color=COLORS[method], ls=ls, lw=1.2, label=labels[method])
        best = int(scores[method]["best_epoch"])
        curve.scatter(best, rmse[best - 1], s=23, color=COLORS[method], marker=symbol,
                      edgecolor="white", linewidth=.45, zorder=4)
    curve.axvline(40, color="#a6b7bf", lw=.8, ls=":")
    curve.set(xlim=(.5, 40.7), xticks=[1, 10, 20, 30, 40],
              xlabel="训练轮次" if zh else "Training epoch",
              ylabel="验证 RMSE (×10⁻³)" if zh else "Validation RMSE (×10⁻³)")
    curve.set_title("A   训练轨迹" if zh else "A   Training trajectories", loc="left", fontweight="bold", pad=7)
    curve.grid(axis="y", color=GRID, lw=.65)
    curve.legend(loc="upper center", bbox_to_anchor=(.48, -.2), ncol=3,
                 frameon=False, handlelength=1.5, columnspacing=.95, fontsize=7.7)

    rows = sorted(read("paired_gain.csv"), key=lambda r: -float(r["mean_gain"]))
    effects.axvline(0, color="#859ba5", lw=.8, ls="--")
    for index, row in enumerate(rows):
        method = row["method"]
        x, lo, hi = (1000 * float(row[k]) for k in ("mean_gain", "lo95", "hi95"))
        color = COLORS.get(method, "#667d88")
        marker = {"i02_bridge": "D", "i01_sensor": "s"}.get(method, "o")
        effects.errorbar(x, index, xerr=[[x - lo], [hi - x]], fmt=marker,
                         color=color, lw=1.25, capsize=2, markersize=4)
    effects.set_yticks(range(len(rows)), [r["method"][:3].upper() for r in rows])
    effects.invert_yaxis()
    effects.set_xlabel("配对 RMSE 改善 (×10⁻³)" if zh else "Paired RMSE gain (×10⁻³)")
    effects.set_title("B   最佳检查点" if zh else "B   Best checkpoints", loc="left", fontweight="bold", pad=7)
    effects.grid(axis="y", color=GRID, lw=.65)
    for ax in (curve, effects):
        style_axes(ax)
    save(fig, "validation_dynamics_" + locale)


def stratified(locale):
    zh = locale == "zh"
    rows = {r["stratum"]: r for r in read("coarse_strata.csv") if r["method"] == "i02_bridge"}
    order = ("low", "mid", "high")
    values = [float(rows[k]["gain_pct"]) for k in order]
    fig, ax = plt.subplots(figsize=(3.37, 1.72))
    fig.subplots_adjust(left=.18, right=.97, bottom=.27, top=.93)
    ax.bar(
        range(3), values, width=.58, color=["#B5DCCE", "#B5DCCE", "#C7B9DB"],
        edgecolor=["#499082", "#499082", "#9271AC"], linewidth=.9,
    )
    ax.axhline(0, color="#859ba5", lw=.8)
    for x, value in enumerate(values):
        sign = np.sign(value)
        ax.text(x, value + sign * .36, f"{value:+.2f}%", ha="center",
                va="bottom" if value > 0 else "top", fontsize=8, color=INK)
    ax.set(
        ylim=(-7.6, 10.0), yticks=[-5, 0, 5, 10],
        xticks=range(3), xticklabels=["低", "中", "高"] if zh else ["Low", "Middle", "High"],
        ylabel="RMSE 改善 (%)" if zh else "RMSE gain (%)",
        xlabel="粗观测偏离三分位" if zh else "Coarse-deviation tertile",
    )
    style_axes(ax)
    ax.grid(axis="y", color=GRID, lw=.6)
    save(fig, "coarse_strata_" + locale)


def main():
    font_manager.fontManager.addfont(r"C:\Windows\Fonts\msyh.ttc")
    mpl.rcParams.update({
        "font.family": ["DejaVu Sans", "Microsoft YaHei"], "font.size": 8.2,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK,
        "pdf.fonttype": 42, "axes.unicode_minus": True,
    })
    FIGURES.mkdir(parents=True, exist_ok=True)
    for lang in ("en", "zh"):
        trajectory_and_gain(lang)
        stratified(lang)


if __name__ == "__main__":
    main()
