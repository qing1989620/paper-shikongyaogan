# -*- coding: utf-8 -*-
"""Aggregate every trained/checkpoint variant into one locked result table.

Collects all results/*_lgc_{val,test}.csv and the 20-variant leaderboards,
computes paper-scale metrics and relative deltas vs the fixed-1 baseline of
each checkpoint, and writes results/FINAL_ALL_EXPERIMENTS.csv plus a Markdown
report ready to be pasted into the paper.

Run this only AFTER all training chains have completed.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

METRICS = ("rmse", "mae", "psnr", "ssim", "sam", "ergas", "cc")
MODELS = ["released", "base", "lf", "hf", "bandw", "delsup", "full", "delta", "gate", "reinject"]


def load(path: Path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def summarise(rows):
    m = {k: float(np.mean([float(r[f"fixed1_{k}"]) for r in rows])) for k in METRICS}
    aux = {}
    for tag in ("src_proj_l0.5", "rsp_prev_l0.3", "official10"):
        if f"{tag}_rmse" in rows[0]:
            aux[tag] = 100 * (np.mean([float(r[f"{tag}_rmse"]) for r in rows]) / m["rmse"] - 1)
    return m, aux


def main():
    out_rows = []
    for model in MODELS:
        rec = {"model": model}
        for split in ("val", "test"):
            fname = "released" if model == "released" else model
            p = Path(f"results/{fname}_lgc_{split}.csv")
            if model == "released" and not p.exists():
                p = Path(f"results/lgc_{split}.csv")
            if not p.exists():
                rec[f"{split}_status"] = "missing"
                continue
            rows = load(p)
            m, aux = summarise(rows)
            # convert to paper scale
            rec[f"{split}_status"] = "ok"
            rec[f"{split}_n"] = len(rows)
            rec[f"{split}_rmse"] = m["rmse"] / 2
            rec[f"{split}_ssim"] = m["ssim"]
            rec[f"{split}_sam"] = m["sam"]
            rec[f"{split}_cc"] = m["cc"]
            for tag, v in aux.items():
                rec[f"{split}_{tag}_rel"] = v
        out_rows.append(rec)

    out = Path("results/FINAL_ALL_EXPERIMENTS.csv")
    keys = sorted({k for r in out_rows for k in r})
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(out_rows)

    md = ["# 全部实验汇总（自动聚合，paper 口径 [0,1]）", "",
          "| 模型 | val fixed-1 | test fixed-1 | val 源校Δ% | test 源校Δ% | test 自校Δ% |",
          "|---|---|---|---|---|---|"]
    for r in out_rows:
        if r.get("val_status") != "ok":
            md.append(f"| {r['model']} | 待跑 | | | | |")
            continue
        md.append(
            f"| {r['model']} | {r['val_rmse']:.5f} | {r.get('test_rmse', float('nan')):.5f} "
            f"| {r.get('val_src_proj_l0.5_rel', float('nan')):+.1f} "
            f"| {r.get('test_src_proj_l0.5_rel', float('nan')):+.1f} "
            f"| {r.get('test_rsp_prev_l0.3_rel', float('nan')):+.1f} |"
        )
    Path("results/FINAL_ALL_EXPERIMENTS.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print("saved", out)


if __name__ == "__main__":
    main()
