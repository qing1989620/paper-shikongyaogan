# -*- coding: utf-8 -*-
"""Ablation sweep for RSP using cached first-step predictions.

Sweeps calibration method, target-estimation form, low-pass sigma, change gate
and projection step size on a validation split. Writes a sorted CSV.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from eval_methods import (gaussian_blur_np, fit_selfcal, anchor_target,  # noqa: E402
                          project, rmse, ssim, sam, ergas, psnr, mae, cc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    files = sorted(Path(args.patches, args.split).glob("*.npz"))
    if args.limit:
        files = files[: args.limit]
    cache = Path(args.cache_dir)

    methods = ["identity", "scale", "ols"]
    forms = ["prev", "bi_avg"]
    sigmas = [1.0, 2.0, 3.0]
    gates = [0.0, 0.5]
    lams = [0.0, 0.2, 0.3, 0.5, 0.75, 1.0]

    agg = {}
    for fp in files:
        z = np.load(fp)
        f0 = z["f0"].astype(np.float32); f1 = z["f1"].astype(np.float32); f3 = z["f3"].astype(np.float32)
        c0 = z["c0"].astype(np.float32); c1 = z["c1"].astype(np.float32); c3 = z["c3"].astype(np.float32)
        x1 = np.load(cache / f"{fp.stem}.npz")["x1"].transpose(1, 2, 0)
        base = {"rmse": rmse(x1, f1), "ssim": ssim(x1, f1), "sam": sam(x1, f1),
                "ergas": ergas(x1, f1), "psnr": psnr(x1, f1), "mae": mae(x1, f1), "cc": cc(x1, f1)}
        key = ("fixed1", "none", "none", 0, 0.0)
        agg.setdefault(key, []).append(base)

        for sigma in sigmas:
            for gate in gates:
                for m in methods:
                    a, b, r2 = fit_selfcal(f0, c0, f3, c3, sigma, gate_q=gate if gate > 0 else 0.0, method=m)
                    for form in forms:
                        T = anchor_target(f0, c0, c1, a, b, sigma, f3, c3, form)
                        for lam in lams:
                            p = project(x1, T, lam, sigma)
                            key = (f"rsp_{m}_{form}", "bi" if form == "bi_avg" else "prev",
                                   round(sigma, 1), gate, lam)
                            agg.setdefault(key, []).append({
                                "rmse": rmse(p, f1), "ssim": ssim(p, f1), "sam": sam(p, f1),
                                "ergas": ergas(p, f1), "psnr": psnr(p, f1), "mae": mae(p, f1), "cc": cc(p, f1)})

    rows = []
    for key, vals in agg.items():
        row = {"method": key[0], "form": key[1], "sigma": key[2], "gate": key[3], "lam": key[4],
               "n": len(vals)}
        for mk in ["rmse", "ssim", "sam", "ergas", "psnr", "mae", "cc"]:
            row[mk] = float(np.mean([v[mk] for v in vals]))
        rows.append(row)
    rows.sort(key=lambda r: r["rmse"])
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    for r in rows[:12]:
        print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})
    print("saved", out)


if __name__ == "__main__":
    main()
