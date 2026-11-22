# -*- coding: utf-8 -*-
"""Diagnose the structural bias of the trained STFDiff backbone.

Question: does the network's first-step output encode the *true* coarse temporal
change, or does it drift? We measure, per patch:
  dF_pred = LP(x1) - LP(F_prev)          (predicted fine temporal change)
  dC_true = LP(C_pred) - LP(C_prev)      (observed coarse temporal change)
and report correlation, scale ratio (a*), residual bias (b*), and the residual
of an OLS fit dF_pred ~ a*dC_true + b. A well-calibrated backbone should have
a≈1, b≈0 and high correlation. Any systematic deviation is exactly what our
inference-time low-frequency replacement was repairing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from eval_methods import gaussian_blur_np  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--patches", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    rows = []
    for fp in sorted(Path(args.patches, args.split).glob("*.npz")):
        z = np.load(fp)
        f0 = z["f0"].astype(np.float32); f1 = z["f1"].astype(np.float32)
        c0 = z["c0"].astype(np.float32); c1 = z["c1"].astype(np.float32)
        x1 = np.load(Path(args.cache) / f"{fp.stem}.npz")["x1"].transpose(1, 2, 0)
        dF = gaussian_blur_np(x1 - f0, 2.0)
        dC = gaussian_blur_np(c1 - c0, 2.0)
        # also the GT fine change, to separate "network bias" from "sensor gap"
        dFgt = gaussian_blur_np(f1 - f0, 2.0)
        rows.append((dC, dF, dFgt))

    def fit(x, y):
        X = np.stack([x.ravel(), np.ones(x.size)], 1)
        sol, *_ = np.linalg.lstsq(X, y.ravel(), rcond=None)
        pred = X @ sol
        ss_res = ((y.ravel() - pred) ** 2).sum()
        ss_tot = ((y.ravel() - y.mean()) ** 2).sum()
        corr = float(np.corrcoef(x.ravel(), y.ravel())[0, 1])
        return float(sol[0]), float(sol[1]), corr, float(1 - ss_res / max(ss_tot, 1e-12))

    print(f"== {args.label or args.cache}  n={len(rows)}")
    ac, bc, rc, r2c = [], [], [], []
    ag, bg, rg, r2g = [], [], [], []
    for dC, dF, dFgt in rows:
        a, b, r, r2 = fit(dC, dF)
        ac.append(a); bc.append(b); rc.append(r); r2c.append(r2)
        a, b, r, r2 = fit(dC, dFgt)
        ag.append(a); bg.append(b); rg.append(r); r2g.append(r2)
    q = lambda v, p: float(np.percentile(v, p))
    print(f"  predicted change vs coarse: a={np.mean(ac):.3f} "
          f"[{q(ac,5):.2f},{q(ac,95):.2f}]  b={np.mean(bc):+.4f}  "
          f"corr={np.mean(rc):.3f}  R2={np.mean(r2c):.3f}")
    print(f"  GT fine change  vs coarse: a={np.mean(ag):.3f} "
          f"[{q(ag,5):.2f},{q(ag,95):.2f}]  b={np.mean(bg):+.4f}  "
          f"corr={np.mean(rg):.3f}  R2={np.mean(r2g):.3f}")
    print(f"  |a_pred - a_GT| mean={np.mean(np.abs(np.array(ac)-np.array(ag))):.3f}")


if __name__ == "__main__":
    main()
