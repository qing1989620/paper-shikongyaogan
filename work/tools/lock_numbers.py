# -*- coding: utf-8 -*-
"""Lock the final numbers: select lambda on val, report the same lambda on test.

All RMSE/MAE/PSNR are converted to the paper's [0,1] reflectance domain:
our tensors live in [-1,1], so RMSE_paper = RMSE_raw/2, MAE_paper = MAE_raw/2,
PSNR_paper = PSNR_raw + 20*log10(2) (= +6.0206 dB).
Writes FINAL_NUMBERS_LOCKED.csv as the single source of truth.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_summary(p: Path) -> dict:
    return json.loads(p.read_text())


def pick_lambda(val: dict, family_prefix: str, metric="rmse"):
    cands = {k: v for k, v in val.items() if k.startswith(family_prefix) and k != "_meta"}
    if not cands:
        return None, None
    best = min(cands.items(), key=lambda kv: kv[1][metric])
    return best[0], best[1]


def conv(metrics: dict, paper=True):
    out = dict(metrics)
    if paper:
        for k in ("rmse", "mae"):
            out[k] = out[k] / 2.0
        out["psnr"] = out["psnr"] + 20 * np.log10(2.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default="LGC")
    args = ap.parse_args()

    val = load_summary(Path(args.val))
    test = load_summary(Path(args.test))

    rows = []
    # fixed baselines
    for name in ["official10", "fixed1", "fixed2"]:
        if name in test:
            m = conv(test[name])
            rows.append({"dataset": args.dataset, "split": "test", "method": name,
                         "lambda": "", "selected_on": "n/a", **m})
    # families selected on val
    for family, label in [("rsp_bi_avg", "RSP bi-avg"), ("rsp_prev", "RSP prev-only"),
                          ("src_proj", "source-cal projection")]:
        best_key, best_val = pick_lambda(val, family)
        if best_key is None or best_key not in test:
            continue
        lam = best_key.split("_l")[-1]
        m = conv(test[best_key])
        rows.append({"dataset": args.dataset, "split": "test", "method": label,
                     "lambda": lam, "selected_on": f"val {best_val['rmse']:.5f}",
                     **m})
        # also record val at the same lambda for reference
        vm = conv(val[best_key])
        rows.append({"dataset": args.dataset, "split": "val", "method": label,
                     "lambda": lam, "selected_on": "same", **vm})

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    keys = ["dataset", "split", "method", "lambda", "selected_on",
            "rmse", "mae", "psnr", "ssim", "sam", "ergas", "cc"]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

    # console table with deltas vs fixed1
    base = next((r for r in rows if r["method"] == "fixed1" and r["split"] == "test"), None)
    print(f"{'method':26s} {'RMSE':>8s} {'d%':>7s} {'SSIM':>7s} {'SAM':>6s} {'PSNR':>7s} {'CC':>7s}")
    for r in rows:
        if r["split"] != "test":
            continue
        d = (r["rmse"] / base["rmse"] - 1) * 100 if base else float("nan")
        print(f"{r['method']:26s} {r['rmse']:8.5f} {d:+7.2f} {r['ssim']:7.4f} "
              f"{r['sam']:6.2f} {r['psnr']:7.2f} {r['cc']:7.4f}")
    print("saved", out)


if __name__ == "__main__":
    main()
