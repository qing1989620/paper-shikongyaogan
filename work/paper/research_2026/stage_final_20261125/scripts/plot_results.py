"""Generate bilingual vector figures from the verified stage evidence."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PACKAGE = Path(__file__).resolve().parents[1]
FIGURES = PACKAGE / "source" / "figures"
BLUE, GREEN, PURPLE = "#6589CC", "#499082", "#9271AC"
PALE_BLUE, PALE_GREEN = "#B7CAE3", "#B5DCCE"
INK, GRID = "#263C42", "#e2e7eb"


def read(name):
    with (PACKAGE / "evidence" / (name + ".csv")).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#a8b5bc")
    ax.set_axisbelow(True)
    ax.tick_params(color="#83939d", length=2.5, pad=2)


def save(fig, name, language):
    fig.savefig(FIGURES / f"{name}_{language}.pdf", bbox_inches="tight", pad_inches=.05)
    fig.savefig(FIGURES / f"{name}_{language}.png", dpi=300, bbox_inches="tight", pad_inches=.05)
    plt.close(fig)


def gains(language, scores):
    en = language == "en"
    methods = [r["method"] for r in scores if r["seed"] == "42" and r["method"] != "base"]
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.6), sharex=True, sharey=True, layout="constrained")
    y = np.arange(len(methods))
    for letter, seed, ax in zip(("A", "B"), ("42", "123"), axes):
        by_method = {row["method"]: row for row in scores if row["seed"] == seed}
        ax.axhspan(8.54, 9.46, color="#eaf3f0", zorder=0)
        ax.axvline(0, linewidth=.95, color="#52626b", linestyle=(0, (2.5, 2)), zorder=1)
        for offset, name, color, marker, legend in (
            (-.17, "best_gain_pct", BLUE, "o", "Best checkpoint" if en else "最佳检查点"),
            (.17, "final_gain_pct", GREEN, "D", "Epoch 40" if en else "第 40 轮"),
        ):
            ax.scatter([float(by_method[m][name]) for m in methods], y + offset, color=color,
                       marker=marker, s=17, linewidth=.4, edgecolor="white", label=legend, zorder=3)
        ax.set_yticks(y, [name[:3].upper() for name in methods])
        ax.set_ylim(len(methods) - .5, -.5)
        ax.set_xlim(-12.6, 10)
        ax.set_xticks([-12, -8, -4, 0, 4, 8])
        ax.set_title(f"{letter}   " + (f"Seed {seed}" if en else f"种子 {seed}"), loc="left", weight="bold")
        ax.grid(axis="y", color=GRID, linewidth=.42)
        style(ax)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper right", ncol=2, frameon=False)
    fig.supxlabel("Paired RMSE reduction (%)" if en else "相对同种子基线的 RMSE 降幅（%）", fontsize=9)
    save(fig, "gains", language)


def trajectory(language, curves):
    en = language == "en"
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.66), sharex=True, sharey=True, layout="constrained")
    for letter, seed, ax in zip(("A", "B"), ("42", "123"), axes):
        for method, name, color, line in (
            ("base", "Base" if en else "基线", INK, "-"),
            ("i10_unmixing", "I10", GREEN, "-"),
            ("i03_frequency", "I03", BLUE, "--"),
            ("i14_masked_ema", "I14", PURPLE, ":"),
        ):
            run = [r for r in curves if r["phase"] == "replication" and r["seed"] == seed and r["method"] == method]
            epoch = np.array([int(row["epoch"]) for row in run])
            rmse = np.array([float(row["val_rmse"]) for row in run])
            ax.plot(epoch, rmse, color=color, linewidth=1.05, linestyle=line, label=name)
            k = np.argmin(rmse)
            ax.scatter(epoch[k], rmse[k], color=color, s=19, edgecolor="white", linewidth=.4, zorder=5)
        ax.axvline(40, linewidth=.75, color="#818e96", linestyle=":")
        ax.set_title(f"{letter}   " + (f"Seed {seed}" if en else f"种子 {seed}"), loc="left", weight="bold")
        ax.set_xticks((1, 10, 20, 30, 40))
        ax.set_xlim(.5, 40.7)
        ax.grid(axis="y", color=GRID, linewidth=.55)
        style(ax)
    axes[0].set_ylabel("Validation RMSE" if en else "验证 RMSE")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper right", ncol=4, frameon=False)
    fig.supxlabel("Epoch" if en else "训练轮次", fontsize=9)
    save(fig, "trajectory", language)


def diagnostics(language, intervals, strata):
    en = language == "en"
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.55), layout="constrained")
    a, b = axes
    a.axvline(0, color="#52626b", linewidth=.8, linestyle=":")
    for i, row in enumerate(intervals):
        delta, lo, hi = [1000 * float(row[k]) for k in ("mean_gain", "lo95", "hi95")]
        a.errorbar(delta, i, xerr=[[delta - lo], [hi - delta]], fmt=("o", "D")[i],
                   color=(BLUE, GREEN)[i], markersize=5, capsize=3, linewidth=1.1)
    a.set_yticks([0, 1], [(f"Seed {seed}" if en else f"种子 {seed}") for seed in (42, 123)])
    a.set_ylim(1.65, -.65)
    a.set_xlabel((r"Base minus I10 RMSE ($10^{-3}$)" if en else r"配对 RMSE 差（$10^{-3}$）"))
    a.set_title("A   " + ("Within-scene paired effects" if en else "场景内配对效应"), loc="left", weight="bold")
    a.grid(axis="x", color=GRID, linewidth=.52)
    x = np.arange(3)
    for i, seed in enumerate(("42", "123")):
        by_level = {row["stratum"]: float(row["gain_pct"]) for row in strata if row["seed"] == seed}
        b.bar(x + (i - .5) * .30, [by_level[key] for key in ("low", "mid", "high")], width=.28,
              color=(PALE_BLUE, PALE_GREEN)[i], edgecolor=(BLUE, GREEN)[i], linewidth=.8,
              label=f"Seed {seed}" if en else f"种子 {seed}")
    b.axhline(0, color="#52626b", linewidth=.8, linestyle=":")
    b.set_xticks(x, ("Low", "Mid", "High") if en else ("低", "中", "高"))
    b.set_xlabel("Coarse temporal deviation" if en else "粗观测时间偏离")
    b.set_ylabel("RMSE reduction (%)" if en else "RMSE 降幅（%）")
    b.set_title("B   " + ("I10 by deviation tertile" if en else "I10 的粗观测偏离分层"), loc="left", weight="bold")
    b.grid(axis="y", color=GRID, linewidth=.52)
    b.legend(frameon=False, loc="best", fontsize=7.4)
    for ax in axes:
        style(ax)
    save(fig, "diagnostics", language)


def main():
    scores = [r for r in read("scores") if r["phase"] == "replication"]
    curves = read("curves")
    intervals = [r for r in read("pair_intervals") if r["method"] == "i10_unmixing"]
    strata = [r for r in read("coarse_strata") if r["method"] == "i10_unmixing"]
    for language in ("en", "zh"):
        plt.rcParams.update({"font.family": "Arial" if language == "en" else "Microsoft YaHei",
                             "font.size": 8.5, "axes.labelcolor": INK, "text.color": INK,
                             "xtick.color": INK, "ytick.color": INK, "axes.unicode_minus": False,
                             "pdf.fonttype": 42})
        gains(language, scores)
        trajectory(language, curves)
        diagnostics(language, intervals, strata)


if __name__ == "__main__":
    main()
