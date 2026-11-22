"""Explicit reflectance metrics for the new benchmark (not wired into legacy runs).

Inputs are HWC reflectance tensors, normally in [0,1]. No automatic scaling or
clipping. The ERGAS h/l factor must be supplied by the frozen protocol.
SSIM uses an 11-pixel Gaussian window, sigma=1.5, population covariance,
data_range=1 and valid interior centers. Requires only NumPy.
"""
from __future__ import annotations

import math
import numpy as np

METRIC_VERSION = "reflectance_v1"


def from_minus_one_one(x):
    return (np.asarray(x, dtype=np.float64) + 1.0) / 2.0


def _blur11(a):
    g = np.exp(-np.arange(-5, 6, dtype=np.float64)**2 / (2*1.5**2))
    g /= g.sum()
    for axis in (0, 1):
        pads = [(0, 0)] * a.ndim
        pads[axis] = (5, 5)
        padded = np.pad(a, pads, mode="reflect")
        out = np.zeros_like(a, dtype=np.float64)
        for i, weight in enumerate(g):
            sl = [slice(None)] * a.ndim
            sl[axis] = slice(i, i+a.shape[axis])
            out += weight * padded[tuple(sl)]
        a = out
    return a


def _ssim(pred, gt, mask):
    if min(pred.shape[:2]) < 11:
        raise ValueError("SSIM requires at least 11 pixels per side")
    u, v = _blur11(pred), _blur11(gt)
    vp = np.maximum(_blur11(pred*pred) - u*u, 0)
    vg = np.maximum(_blur11(gt*gt) - v*v, 0)
    cov = _blur11(pred*gt) - u*v
    values = ((2*u*v + 0.01**2)*(2*cov + 0.03**2)
              / ((u*u+v*v+0.01**2)*(vp+vg+0.03**2)))
    # Every pixel in the local 11x11 neighborhood must be valid.
    centers = mask.copy()
    if not mask.all():
        padded = np.pad(mask, 5, mode="constant", constant_values=False)
        for dy in range(11):
            for dx in range(11):
                centers &= padded[dy:dy+mask.shape[0], dx:dx+mask.shape[1]]
    centers[:5] = centers[-5:] = False
    centers[:, :5] = centers[:, -5:] = False
    if not centers.any():
        raise ValueError("No valid SSIM window centers")
    return float(values[centers].mean()), int(centers.sum())


def compute_metrics(pred, gt, *, ergas_ratio, mask=None):
    """Compute all metrics on the same supplied valid-pixel mask.

    SAM excludes zero-norm GT spectra. A zero prediction with a nonzero GT is
    assigned pi/2, not silently dropped. Undefined constant-band CC and
    zero-mean-band ERGAS return NaN and are accompanied by explicit counts.
    """
    pred, gt = np.asarray(pred, dtype=np.float64), np.asarray(gt, dtype=np.float64)
    if pred.shape != gt.shape or pred.ndim != 3:
        raise ValueError("Expected identical HWC arrays")
    if not math.isfinite(ergas_ratio) or ergas_ratio <= 0:
        raise ValueError("An explicit positive ERGAS h/l ratio is required")
    m = np.ones(gt.shape[:2], dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if m.shape != gt.shape[:2] or not m.any():
        raise ValueError("Invalid or empty mask")
    if not np.isfinite(pred[m]).all() or not np.isfinite(gt[m]).all():
        raise ValueError("Non-finite values on the evaluation mask")
    # Masked-out values must not contaminate local SSIM calculations.
    p = np.where(m[..., None], pred, 0.0)
    y = np.where(m[..., None], gt, 0.0)
    pp, yy = pred[m], gt[m]
    error = pp-yy
    mse = float(np.mean(error**2))
    band_mse = np.mean(error**2, axis=0)
    means = yy.mean(axis=0)
    ergas_valid = np.abs(means) > 1e-12
    ergas = float(100*ergas_ratio*np.sqrt(np.mean(band_mse/means**2))) if ergas_valid.all() else float("nan")
    pc, yc = pp-pp.mean(axis=0), yy-yy.mean(axis=0)
    cc_den = np.sqrt((pc**2).sum(axis=0)*(yc**2).sum(axis=0))
    cc_valid = cc_den > 1e-12
    correlation = float(np.mean((pc*yc).sum(axis=0)/cc_den)) if cc_valid.all() else float("nan")
    norm_gt, norm_pred = np.linalg.norm(yy, axis=1), np.linalg.norm(pp, axis=1)
    sam_valid = norm_gt > 1e-12
    den = norm_gt*norm_pred
    cos = np.divide((pp*yy).sum(axis=1), den, out=np.zeros_like(den), where=den > 1e-24)
    angles = np.arccos(np.clip(cos[sam_valid], -1, 1))
    sam_rad = float(angles.mean()) if len(angles) else float("nan")
    structural, ssim_n = _ssim(p, y, m)
    return {"rmse": math.sqrt(mse), "mae": float(np.abs(error).mean()),
            "psnr_db": -10*math.log10(mse) if mse > 0 else float("inf"),
            "ssim": structural, "sam_rad": sam_rad, "sam_deg": math.degrees(sam_rad),
            "ergas": ergas, "cc": correlation, "n_valid_pixels": int(m.sum()),
            "n_sam_pixels": int(sam_valid.sum()), "n_ssim_centers": ssim_n,
            "undefined_cc_bands": int((~cc_valid).sum()),
            "undefined_ergas_bands": int((~ergas_valid).sum())}
