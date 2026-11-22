# -*- coding: utf-8 -*-
"""Qualitative comparison figure: GT / fixed-1 / RSP / error maps on one patch."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from eval_methods import (gaussian_blur_np, fit_selfcal, anchor_target,  # noqa: E402
                          project, rmse)


def false_color(x, bands=(3, 2, 1)):
    img = x[..., list(bands)]
    img = (img + 1) / 2
    lo, hi = np.percentile(img, 2), np.percentile(img, 98)
    return np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=0, help="0 = largest gain")
    args = ap.parse_args()

    files = sorted(Path(args.patches, args.split).glob("*.npz"))
    cache = Path(args.cache_dir)
    recs = []
    for fp in files:
        z = np.load(fp)
        f1 = z["f1"].astype(np.float32)
        x1 = np.load(cache / f"{fp.stem}.npz")["x1"].transpose(1, 2, 0)
        c0 = z["c0"].astype(np.float32); c1 = z["c1"].astype(np.float32); c3 = z["c3"].astype(np.float32)
        f0 = z["f0"].astype(np.float32); f3 = z["f3"].astype(np.float32)
        a, b, _ = fit_selfcal(f0, c0, f3, c3, 2.0)
        T = anchor_target(f0, c0, c1, a, b, 2.0, f3, c3, "prev")
        out = project(x1, T, 0.2, 2.0)
        recs.append((rmse(out, f1) - rmse(x1, f1), fp))
    recs.sort(key=lambda t: t[0])
    fp = recs[args.rank][1]

    z = np.load(fp)
    f1 = z["f1"].astype(np.float32)
    f0 = z["f0"].astype(np.float32); f3 = z["f3"].astype(np.float32)
    c0 = z["c0"].astype(np.float32); c1 = z["c1"].astype(np.float32); c3 = z["c3"].astype(np.float32)
    x1 = np.load(cache / f"{fp.stem}.npz")["x1"].transpose(1, 2, 0)
    a, b, _ = fit_selfcal(f0, c0, f3, c3, 2.0)
    T = anchor_target(f0, c0, c1, a, b, 2.0, f3, c3, "prev")
    out = project(x1, T, 0.2, 2.0)

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    titles = [f"Ground truth", f"fixed-1 (RMSE {rmse(x1, f1):.4f})",
              f"RSP (RMSE {rmse(out, f1):.4f})", "|error| fixed-1 / RSP"]
    axes[0].imshow(false_color(f1)); axes[0].set_title(titles[0], fontsize=9)
    axes[1].imshow(false_color(x1)); axes[1].set_title(titles[1], fontsize=9)
    axes[2].imshow(false_color(out)); axes[2].set_title(titles[2], fontsize=9)
    err1 = np.abs(x1 - f1).mean(2); erro = np.abs(out - f1).mean(2)
    both = np.concatenate([err1, erro], axis=1)
    im = axes[3].imshow(both, cmap="inferno", vmin=0, vmax=np.percentile(both, 99))
    axes[3].set_title(titles[3], fontsize=9)
    axes[3].axvline(err1.shape[1] - 0.5, color="w", lw=0.8)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.02)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=220, bbox_inches="tight")
    print("saved", args.out, "patch", fp.stem)


if __name__ == "__main__":
    main()
