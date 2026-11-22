# -*- coding: utf-8 -*-
"""Twenty inference-time variants for RSP, evaluated on cached first-step outputs.

Every variant is a *runnable* inference-time operator: it consumes the frozen
backbone's first-step output x1 plus the sample's own observations, and returns
a corrected image. No retraining is involved, so all twenty can be evaluated
honestly on CPU.

Protocol: for each variant the projection strength lambda is selected on the
validation split and then applied unchanged to the test split.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from eval_methods import rmse, mae, psnr, ssim, sam, ergas, cc  # noqa: E402

SIGMAS = (1.0, 2.0, 3.0, 4.0)


# ----------------------------------------------------------------- primitives
def lp(x, sigma):
    return gaussian_filter(x, sigma=(sigma, sigma, 0), mode="reflect")


def fit_ols(dC, dF, weights=None, prior=(1.0, 0.0), ridge=0.0):
    """Per-band affine fit with optional weights and ridge toward a prior."""
    a = np.empty(dC.shape[-1]); b = np.empty(dC.shape[-1])
    for k in range(dC.shape[-1]):
        x = dC[..., k].ravel().astype(np.float64)
        y = dF[..., k].ravel().astype(np.float64)
        if weights is not None:
            w = weights[..., k].ravel().astype(np.float64)
        else:
            w = np.ones_like(x)
        gate = np.abs(x) >= np.quantile(np.abs(x), 0.5)
        x, y, w = x[gate], y[gate], w[gate]
        X = np.stack([x, np.ones_like(x)], 1)
        Xw = X * w[:, None]
        A = Xw.T @ X
        if ridge > 0:
            A = A + ridge * np.eye(2)
            rhs = Xw.T @ y + ridge * np.array([prior[0], prior[1]])
        else:
            rhs = Xw.T @ y
        try:
            sol = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            sol = np.array([prior[0], prior[1]])
        a[k], b[k] = sol[0], sol[1]
    return a, b


def fit_huber(dC, dF, iters=3, c=1.345):
    a, b = fit_ols(dC, dF)
    for _ in range(iters):
        r = dF - (a * dC + b)
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-8
        w = np.clip(np.abs(r) / (c * s), None, 1.0)
        w = 1.0 / np.maximum(w, 1e-3)
        a, b = fit_ols(dC, dF, weights=w)
    return a, b


def fit_tls(dC, dF):
    """Total least squares (orthogonal regression) per band."""
    a = np.empty(dC.shape[-1]); b = np.empty(dC.shape[-1])
    for k in range(dC.shape[-1]):
        x = dC[..., k].ravel().astype(np.float64)
        y = dF[..., k].ravel().astype(np.float64)
        gate = np.abs(x) >= np.quantile(np.abs(x), 0.5)
        x, y = x[gate], y[gate]
        mx, my = x.mean(), y.mean()
        X = np.stack([x - mx, y - my], 1)
        cov = X.T @ X / max(len(x) - 1, 1)
        evals, evecs = np.linalg.eigh(cov)
        v = evecs[:, 0]
        slope = v[1] / v[0] if abs(v[0]) > 1e-12 else 1.0
        a[k] = slope
        b[k] = my - slope * mx
    return a, b


def fit_deming(dC, dF, ratio=0.5):
    """Errors-in-variables regression with noise ratio lambda = var(e_y)/var(e_x)."""
    a = np.empty(dC.shape[-1]); b = np.empty(dC.shape[-1])
    for k in range(dC.shape[-1]):
        x = dC[..., k].ravel().astype(np.float64)
        y = dF[..., k].ravel().astype(np.float64)
        gate = np.abs(x) >= np.quantile(np.abs(x), 0.5)
        x, y = x[gate], y[gate]
        mx, my = x.mean(), y.mean()
        sxx = ((x - mx) ** 2).mean(); syy = ((y - my) ** 2).mean()
        sxy = ((x - mx) * (y - my)).mean()
        num = syy - ratio * sxx
        den = 2 * sxy
        slope = num / den + np.sign(den or 1) * np.sqrt((num / (den or 1e-12)) ** 2 + ratio) if den != 0 else 1.0
        slope = float(np.clip(slope, 0.1, 3.0))
        a[k] = slope; b[k] = my - slope * mx
    return a, b


def guided_filter(I, p, radius=8, eps=1e-3):
    """Edge-preserving filter of p guided by I (single-channel guide, per band)."""
    out = np.empty_like(p)
    k = 2 * radius + 1
    for c in range(p.shape[-1]):
        g = I[..., c] if I.shape[-1] > 1 else I[..., 0]
        mean_I = uniform_filter(g, k); mean_p = uniform_filter(p[..., c], k)
        corr_I = uniform_filter(g * g, k); corr_Ip = uniform_filter(g * p[..., c], k)
        var_I = corr_I - mean_I ** 2
        cov_Ip = corr_Ip - mean_I * mean_p
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I
        out[..., c] = uniform_filter(a, k) * g + uniform_filter(b, k)
    return out


def haar_ll(x):
    """One-level Haar low-pass (LL) implemented as a separable [1,1]/2 filter."""
    from scipy.ndimage import convolve1d
    y = convolve1d(x, np.array([0.5, 0.5]), axis=0, mode="reflect")
    y = convolve1d(y, np.array([0.5, 0.5]), axis=1, mode="reflect")
    return y


def proj(x1, T, lam, kind="gauss", sigma=2.0, guide=None, ema=1.0):
    if lam == 0:
        return x1
    if kind == "gauss":
        base = lp(x1, sigma)
    elif kind == "guided":
        base = guided_filter(guide if guide is not None else x1, x1, radius=8, eps=1e-3)
    elif kind == "haar":
        base = haar_ll(x1)
    elif kind == "cascade":
        base = lp(lp(x1, 4.0), 2.0)
    else:
        raise ValueError(kind)
    return x1 - lam * (base - T)


def target_from(f0, c0, c1, a, b, sigma=2.0, f3=None, c3=None, form="prev",
                invvar=False, span=1.0):
    Tp = lp(f0, sigma) + a * lp(c1 - c0, sigma) * span + b
    if form == "prev" or f3 is None:
        return Tp
    Tn = lp(f3, sigma) - a * lp(c3 - c1, sigma) * span - b
    if form == "next":
        return Tn
    if invvar:
        # weight each anchor by the inverse of its anchor-pair temporal change
        e_p = np.mean(np.abs(lp(c1 - c0, sigma)), axis=(0, 1))
        e_n = np.mean(np.abs(lp(c3 - c1, sigma)), axis=(0, 1))
        w = (1.0 / np.maximum(e_p, 1e-6)) / (1.0 / np.maximum(e_p, 1e-6) + 1.0 / np.maximum(e_n, 1e-6))
        return w * Tp + (1 - w) * Tn
    return 0.5 * (Tp + Tn)


# ----------------------------------------------------------------- variants
def make_variants(lam):
    """Return list of (name, callable(prep) -> image)."""
    V = []
    V.append(("01_fixed1", lambda p: p["x1"]))
    V.append(("02_identity", lambda p: proj(p["x1"], target_from(p["f0"], p["c0"], p["c1"],
              np.ones(6), np.zeros(6), 2.0), lam)))
    V.append(("03_ols_prev", lambda p: proj(p["x1"], target_from(p["f0"], p["c0"], p["c1"],
              p["a_ols"], p["b_ols"], 2.0), lam)))
    V.append(("04_ols_bi", lambda p: proj(p["x1"], target_from(p["f0"], p["c0"], p["c1"],
              p["a_ols"], p["b_ols"], 2.0, p["f3"], p["c3"], "bi"), lam)))
    V.append(("05_next_only", lambda p: proj(p["x1"], target_from(p["f0"], p["c0"], p["c1"],
              p["a_ols"], p["b_ols"], 2.0, p["f3"], p["c3"], "next"), lam)))
    V.append(("06_invvar_bi", lambda p: proj(p["x1"], target_from(p["f0"], p["c0"], p["c1"],
              p["a_ols"], p["b_ols"], 2.0, p["f3"], p["c3"], "bi", invvar=True), lam)))
    V.append(("07_span_norm", lambda p: proj(p["x1"], target_from(p["f0"], p["c0"], p["c1"],
              p["a_ols"], p["b_ols"], 2.0, p["f3"], p["c3"], "bi", span=p["span_ratio"]), lam)))
    V.append(("08_huber", lambda p: proj(p["x1"], target_from(p["f0"], p["c0"], p["c1"],
              p["a_hub"], p["b_hub"], 2.0), lam)))
    V.append(("09_tls", lambda p: proj(p["x1"], target_from(p["f0"], p["c0"], p["c1"],
              p["a_tls"], p["b_tls"], 2.0), lam)))
    V.append(("10_deming", lambda p: proj(p["x1"], target_from(p["f0"], p["c0"], p["c1"],
              p["a_dem"], p["b_dem"], 2.0), lam)))
    V.append(("11_wls", lambda p: proj(p["x1"], target_from(p["f0"], p["c0"], p["c1"],
              p["a_wls"], p["b_wls"], 2.0), lam)))
    V.append(("12_ridge", lambda p: proj(p["x1"], target_from(p["f0"], p["c0"], p["c1"],
              p["a_rid"], p["b_rid"], 2.0), lam)))
    V.append(("13_scale_only", lambda p: proj(p["x1"], target_from(p["f0"], p["c0"], p["c1"],
              p["a_scl"], np.zeros(6), 2.0), lam)))
    V.append(("14_perband_lam", lambda p: p["x1"] - (lam * p["lam_k"])[None, None, :] *
              (lp(p["x1"], 2.0) - target_from(p["f0"], p["c0"], p["c1"], p["a_ols"], p["b_ols"], 2.0))))
    V.append(("15_pixel_adaptive", lambda p: p["x1"] - (lam * p["w_pix"])[..., None] *
              (lp(p["x1"], 2.0) - target_from(p["f0"], p["c0"], p["c1"], p["a_ols"], p["b_ols"], 2.0))))
    V.append(("16_guided", lambda p: proj(p["x1"], target_from(p["f0"], p["c0"], p["c1"],
              p["a_ols"], p["b_ols"], 2.0), lam, kind="guided", guide=p["f0"])))
    V.append(("17_haar", lambda p: proj(p["x1"], target_from(p["f0"], p["c0"], p["c1"],
              p["a_ols"], p["b_ols"], 2.0), lam, kind="haar")))
    V.append(("18_cascade", lambda p: proj(p["x1"], target_from(p["f0"], p["c0"], p["c1"],
              p["a_ols"], p["b_ols"], 2.0), lam, kind="cascade")))
    V.append(("19_iter2", lambda p: _iter2(p, lam)))
    V.append(("20_huber_proj", lambda p: _huber_proj(p, lam)))
    return V


def _iter2(p, lam):
    T = target_from(p["f0"], p["c0"], p["c1"], p["a_ols"], p["b_ols"], 2.0)
    y = proj(p["x1"], T, lam)
    return proj(y, T, lam)


def _huber_proj(p, lam, clip=0.05):
    T = target_from(p["f0"], p["c0"], p["c1"], p["a_ols"], p["b_ols"], 2.0)
    M = lp(p["x1"], 2.0) - T
    M = np.clip(M, -clip, clip)
    return p["x1"] - lam * M


def prep_patch(z):
    f0 = z["f0"].astype(np.float32); f1 = z["f1"].astype(np.float32); f3 = z["f3"].astype(np.float32)
    c0 = z["c0"].astype(np.float32); c1 = z["c1"].astype(np.float32); c3 = z["c3"].astype(np.float32)
    dF = lp(f3 - f0, 2.0); dC = lp(c3 - c0, 2.0)
    a_ols, b_ols = fit_ols(dC, dF)
    a_hub, b_hub = fit_huber(dC, dF)
    a_tls, b_tls = fit_tls(dC, dF)
    a_dem, b_dem = fit_deming(dC, dF)
    w = np.abs(dC)
    a_wls, b_wls = fit_ols(dC, dF, weights=w)
    a_rid, b_rid = fit_ols(dC, dF, ridge=50.0, prior=(1.0, 0.0))
    # scale only: closed form through origin
    a_scl = np.array([float((dC[..., k].ravel() * dF[..., k].ravel()).sum() /
                            max((dC[..., k].ravel() ** 2).sum(), 1e-9)) for k in range(6)])
    # per-band lambda proportional to fitted slope (normalised), clipped
    lam_k = np.clip(a_ols / max(a_ols.mean(), 1e-6), 0.3, 1.8)
    # per-pixel weight from local coarse change magnitude (normalised)
    m = np.abs(lp(c1 - c0, 2.0)).mean(2)
    w_pix = np.clip(m / (np.percentile(m, 90) + 1e-6), 0.0, 1.5)
    return dict(x1=None, f0=f0, f1=f1, f3=f3, c0=c0, c1=c1, c3=c3,
                a_ols=a_ols, b_ols=b_ols, a_hub=a_hub, b_hub=b_hub,
                a_tls=a_tls, b_tls=b_tls, a_dem=a_dem, b_dem=b_dem,
                a_wls=a_wls, b_wls=b_wls, a_rid=a_rid, b_rid=b_rid,
                a_scl=a_scl, lam_k=lam_k, w_pix=w_pix, span_ratio=1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lams", default="0.2,0.3,0.5,0.75")
    args = ap.parse_args()
    files = sorted(Path(args.patches, args.split).glob("*.npz"))
    cache = Path(args.cache)
    lams = [float(v) for v in args.lams.split(",")]
    rows = []
    for i, fp in enumerate(files):
        z = np.load(fp)
        p = prep_patch(z)
        p["x1"] = np.load(cache / f"{fp.stem}.npz")["x1"].transpose(1, 2, 0)
        f1 = p["f1"]
        for lam in lams:
            for name, fn in make_variants(lam):
                pred = fn(p)
                rows.append({"patch": fp.stem, "lam": lam, "variant": name,
                             **{f"m_{k}": v for k, v in
                                dict(rmse=rmse(pred, f1), mae=mae(pred, f1), psnr=psnr(pred, f1),
                                     ssim=ssim(pred, f1), sam=sam(pred, f1), ergas=ergas(pred, f1),
                                     cc=cc(pred, f1)).items()}})
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(files)}", flush=True)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print("saved", out, len(rows), "rows")


if __name__ == "__main__":
    main()
