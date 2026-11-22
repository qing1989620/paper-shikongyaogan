# -*- coding: utf-8 -*-
"""Paired significance tests + bootstrap CI over per-patch CSV results.

Usage:
  python tools/stats_tests.py --csv results/lgc_val.csv \
      --pair rsp_bi_avg_l0.5 fixed1 --pair src_proj_l0.5 fixed1
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def paired_test(a, b, n_boot=5000, seed=42):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    try:
        from scipy.stats import wilcoxon
        stat, p = wilcoxon(a, b)
    except Exception:
        stat, p = float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "n": int(len(d)),
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "delta": float(d.mean()),
        "rel_pct": float(100 * d.mean() / b.mean()) if b.mean() != 0 else float("nan"),
        "wilcoxon_stat": float(stat), "p": float(p),
        "ci95": [float(lo), float(hi)],
        "win_rate": float((d < 0).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--metric", default="rmse")
    ap.add_argument("--pair", nargs=2, action="append", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    rows = load(args.csv)
    res = {}
    for a, b in args.pair:
        ca, cb = f"{a}_{args.metric}", f"{b}_{args.metric}"
        if ca not in rows[0] or cb not in rows[0]:
            print(f"skip {a} vs {b}: columns missing")
            continue
        res[f"{a}_vs_{b}"] = paired_test([float(r[ca]) for r in rows],
                                         [float(r[cb]) for r in rows])
    print(json.dumps(res, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
