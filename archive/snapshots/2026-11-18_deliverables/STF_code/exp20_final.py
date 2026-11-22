# -*- coding: utf-8 -*-
"""Final twenty literature-anchored variants (E01-E20) for RSP.

Each variant is an inference-time operator on the frozen backbone's first-step
output; none requires retraining, so all twenty are actually run.

Provenance of each idea is recorded in VARIANT_META and in
work/INNOVATION_20.md. Lambda is selected on the validation split and then
applied unchanged to the test split.
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
from eval_methods import rmse, mae, psnr, ssim, sam, ergas, cc  # noqa: E402
from exp20 import (lp, fit_ols, fit_huber, fit_tls, fit_deming, guided_filter,
                   haar_ll, target_from, prep_patch)  # noqa: E402


# --------------------------------------------------------------- new operators
def hist_match(src_low, ref):
    """Rank-based histogram matching of `ref` onto the distribution of `src_low`."""
    out = np.empty_like(ref)
    for k in range(ref.shape[-1]):
        s = np.sort(src_low[..., k].ravel())
        r = ref[..., k].ravel()
        # map each ref value to the src quantile at the same rank
        ranks = np.argsort(np.argsort(r)) / max(len(r) - 1, 1)
        idx = np.clip((ranks * (len(s) - 1)).astype(np.int64), 0, len(s) - 1)
        out[..., k] = s[idx].reshape(ref.shape[:2])
    return out


def ot_affine(src_low, ref):
    """Per-band mean/std alignment (linear optimal-transport / colour transfer)."""
    out = np.empty_like(ref)
    for k in range(ref.shape[-1]):
        s = src_low[..., k]; r = ref[..., k]
        ss = s.std() + 1e-6; rs = r.std() + 1e-6
        out[..., k] = (r - r.mean()) / rs * ss + s.mean()
    return out


def conformal_gate(M, lam, scale):
    """Per-sample shrinkage of lambda by the observed mismatch (risk control)."""
    e = float(np.mean(np.abs(M)))
    return lam / (1.0 + e / max(scale, 1e-6))


# --------------------------------------------------------------- variant registry
def build(lam, stats):
    """stats: dict of per-split calibration constants (e.g. median mismatch)."""
    s = stats
    V = []
    V.append(("E01_fixed1", lambda p: p["x1"]))
    V.append(("E02_identity", lambda p: _proj(p, np.ones(6), np.zeros(6), lam, form="prev")))
    V.append(("E03_ols_prev", lambda p: _proj(p, p["a_ols"], p["b_ols"], lam, form="prev")))
    V.append(("E04_ols_bi", lambda p: _proj(p, p["a_ols"], p["b_ols"], lam, form="bi")))
    V.append(("E05_tls", lambda p: _proj(p, p["a_tls"], p["b_tls"], lam, form="prev")))
    V.append(("E06_deming", lambda p: _proj(p, p["a_dem"], p["b_dem"], lam, form="prev")))
    V.append(("E07_huber", lambda p: _proj(p, p["a_hub"], p["b_hub"], lam, form="prev")))
    V.append(("E08_ridge", lambda p: _proj(p, p["a_rid"], p["b_rid"], lam, form="prev")))
    V.append(("E09_wls", lambda p: _proj(p, p["a_wls"], p["b_wls"], lam, form="prev")))
    V.append(("E10_span_norm", lambda p: _proj(p, p["a_ols"], p["b_ols"], lam, form="bi",
                                                span=p["span_ratio"])))
    V.append(("E11_invvar_bi", lambda p: _proj(p, p["a_ols"], p["b_ols"], lam, form="bi",
                                                invvar=True)))
    V.append(("E12_hist_match", lambda p: _hist(p, lam)))
    V.append(("E13_ot_affine", lambda p: _ot(p, lam)))
    V.append(("E14_conformal_gate", lambda p: _gate(p, lam, s.get("median_mismatch", 0.02))))
    V.append(("E15_haar", lambda p: _proj(p, p["a_ols"], p["b_ols"], lam, kind="haar")))
    V.append(("E16_guided", lambda p: _proj(p, p["a_ols"], p["b_ols"], lam,
                                             kind="guided", guide=p["f0"])))
    V.append(("E17_cascade", lambda p: _proj(p, p["a_ols"], p["b_ols"], lam, kind="cascade")))
    V.append(("E18_perband_lam", lambda p: _perband(p, lam)))
    V.append(("E19_pixel_adaptive", lambda p: _pixel(p, lam)))
    V.append(("E20_iter2", lambda p: _iter2(p, lam)))
    return V


def _target(p, a, b, form="prev", span=1.0, invvar=False):
    return target_from(p["f0"], p["c0"], p["c1"], a, b, 2.0,
                       p["f3"] if form != "prev" else None,
                       p["c3"] if form != "prev" else None, form, invvar=invvar, span=span)


def _proj(p, a, b, lam, form="prev", kind="gauss", guide=None, span=1.0, invvar=False):
    T = _target(p, a, b, form, span, invvar)
    if lam == 0:
        return p["x1"]
    if kind == "gauss":
        base = lp(p["x1"], 2.0)
    elif kind == "guided":
        base = guided_filter(guide if guide is not None else p["x1"], p["x1"])
    elif kind == "haar":
        base = haar_ll(p["x1"])
    elif kind == "cascade":
        base = lp(lp(p["x1"], 4.0), 2.0)
    else:
        raise ValueError(kind)
    return p["x1"] - lam * (base - T)


def _hist(p, lam):
    T = _target(p, p["a_ols"], p["b_ols"])
    base = lp(p["x1"], 2.0)
    A = hist_match(base, T)
    return p["x1"] - lam * (base - A)


def _ot(p, lam):
    T = _target(p, p["a_ols"], p["b_ols"])
    base = lp(p["x1"], 2.0)
    A = ot_affine(base, T)
    return p["x1"] - lam * (base - A)


def _gate(p, lam, scale):
    T = _target(p, p["a_ols"], p["b_ols"])
    M = lp(p["x1"], 2.0) - T
    return p["x1"] - conformal_gate(M, lam, scale) * M


def _perband(p, lam):
    T = _target(p, p["a_ols"], p["b_ols"])
    return p["x1"] - (lam * p["lam_k"])[None, None, :] * (lp(p["x1"], 2.0) - T)


def _pixel(p, lam):
    T = _target(p, p["a_ols"], p["b_ols"])
    return p["x1"] - (lam * p["w_pix"])[..., None] * (lp(p["x1"], 2.0) - T)


def _iter2(p, lam):
    T = _target(p, p["a_ols"], p["b_ols"])
    y = p["x1"] - lam * (lp(p["x1"], 2.0) - T)
    return y - lam * (lp(y, 2.0) - T)


VARIANT_META = {
    "E01_fixed1": ("none", "Baseline: one denoising call, no correction"),
    "E02_identity": ("RSAC prior-init (J. Remote Sens. 2026)", "Frequency separation: reuse coarse change as-is, a=1,b=0"),
    "E03_ols_prev": ("PRISM arXiv 2608.15037", "Closed-form affine from reference pairs (ours)"),
    "E04_ols_bi": ("Exact & calibrated diffusion recon. arXiv 2607.12937", "Two-anchor estimate, midpoint"),
    "E05_tls": ("GSC arXiv 2512.20108", "Total least squares / orthogonal regression"),
    "E06_deming": ("GSC arXiv 2512.20108", "Errors-in-variables with noise ratio"),
    "E07_huber": ("DCDP ICCP 2026", "Robust Huber affine"),
    "E08_ridge": ("Guidance weights arXiv 2610.03503", "Ridge shrinkage toward identity"),
    "E09_wls": ("Patch Forcing CVPR 2026", "Change-weighted least squares"),
    "E10_span_norm": ("WaFT TGRS 2026", "Slope normalised by anchor/prediction span"),
    "E11_invvar_bi": ("Uncertainty-aware TTA IGARSS 2026", "Inverse-variance anchor weighting"),
    "E12_hist_match": ("OT / histogram matching (dir. 6)", "Rank-based histogram transfer"),
    "E13_ot_affine": ("OT / colour transfer (dir. 6)", "Mean-std transport of low frequencies"),
    "E14_conformal_gate": ("Conformal prediction TGRS 2026", "Mismatch-gated projection strength"),
    "E15_haar": ("WaFT TGRS 2026 / HDW-SR CVPR 2026", "Haar low-pass projection"),
    "E16_guided": ("HDW-SR CVPR 2026", "Guided-filter projection, edge preserving"),
    "E17_cascade": ("HDW-SR CVPR 2026", "Two-scale cascade low-pass"),
    "E18_perband_lam": ("Guidance weights arXiv 2610.03503", "Per-band lambda proportional to slope"),
    "E19_pixel_adaptive": ("Patch Forcing CVPR 2026", "Per-pixel lambda from local change"),
    "E20_iter2": ("PC-DAPS arXiv 2610.31592", "Two-step fixed-point projection"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lams", default="0.2,0.3,0.5,0.75")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    files = sorted(Path(args.patches, args.split).glob("*.npz"))
    if args.limit:
        files = files[: args.limit]
    cache = Path(args.cache)
    lams = [float(v) for v in args.lams.split(",")]

    # first pass: collect per-patch stats needed by gated variants
    preps = []
    mism = []
    for fp in files:
        z = np.load(fp)
        p = prep_patch(z)
        p["x1"] = np.load(cache / f"{fp.stem}.npz")["x1"].transpose(1, 2, 0)
        p["f1"] = z["f1"].astype(np.float32)
        T = _target(p, p["a_ols"], p["b_ols"])
        mism.append(float(np.mean(np.abs(lp(p["x1"], 2.0) - T))))
        preps.append((fp.stem, p))
    stats = {"median_mismatch": float(np.median(mism))}
    print(f"split={args.split} n={len(preps)} median_mismatch={stats['median_mismatch']:.5f}", flush=True)

    rows = []
    for i, (stem, p) in enumerate(preps):
        f1 = p["f1"]
        for lam in lams:
            for name, fn in build(lam, stats):
                pred = fn(p)
                rows.append({"patch": stem, "lam": lam, "variant": name,
                             "rmse": rmse(pred, f1), "mae": mae(pred, f1), "psnr": psnr(pred, f1),
                             "ssim": ssim(pred, f1), "sam": sam(pred, f1),
                             "ergas": ergas(pred, f1), "cc": cc(pred, f1)})
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(preps)}", flush=True)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print("saved", out, len(rows), "rows", flush=True)


if __name__ == "__main__":
    main()
