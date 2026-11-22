# -*- coding: utf-8 -*-
"""Reference-pair self-calibrated temporal projection (Stage4 core).

Calibration: fit per-band affine LP(F_next − F_prev) ≈ a·LP(C_next − C_prev) + b
from the SAMPLE'S OWN dual reference anchors — no prediction-date GT, no target
fine labels, no backbone update.
"""
import numpy as np
import torch


def fit_anchor_calib(blur, f_prev, c_prev, f_next, c_next, method='ols',
                     gate_q=0.5):
    """Per-band (a, b, R²) from a sample's own anchor pair.

    method: 'ols' (a+b) | 'scale' (b=0) | 'median' | 'mad' | 'identity'.
    Pixels in the bottom `gate_q` quantile of |ΔC| are excluded (low-change gate).
    """
    dF = blur(f_next - f_prev)
    dC = blur(c_next - c_prev)
    a, b, r2 = [], [], []
    for k in range(dF.shape[1]):
        x = dC[0, k].flatten().cpu().numpy().astype(np.float64)
        y = dF[0, k].flatten().cpu().numpy().astype(np.float64)
        gate = np.abs(x) >= np.quantile(np.abs(x), gate_q)
        x, y = x[gate], y[gate]
        if method == 'ols':
            aa, bb = np.polyfit(x, y, 1)
            pred = aa * x + bb
        elif method == 'scale':
            aa = float(np.sum(x * y) / max(np.sum(x * x), 1e-12)); bb = 0.0
            pred = aa * x
        elif method == 'median':
            mask = np.abs(x) > 1e-4
            aa = float(np.median(y[mask] / x[mask])) if mask.sum() > 10 else 1.0
            bb = 0.0; pred = aa * x
        elif method == 'mad':
            aa = float(np.median(np.abs(y - np.median(y))) /
                       max(np.median(np.abs(x - np.median(x))), 1e-6))
            bb = 0.0; pred = aa * x
        elif method == 'identity':
            aa, bb = 1.0, 0.0; pred = x
        else:
            raise ValueError(method)
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2.append(1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0)
        a.append(float(aa)); b.append(float(bb))
    return a, b, r2


def anchor_target_estimate(blur, a, b, f_prev, c_prev, c_t,
                           f_next=None, c_next=None, form='bi_avg'):
    """Low-frequency target estimate T from the prediction-date coarse change.

    prev   : T = LP(f_prev) + a·LP(c_t − c_prev) + b
    bi_avg : mean of prev and next-anchor estimates
             T_next = LP(f_next) − a·LP(c_next − c_t) − b
    """
    a_t = torch.as_tensor(a, dtype=torch.float32, device=f_prev.device).reshape(1, -1, 1, 1)
    b_t = torch.as_tensor(b, dtype=torch.float32, device=f_prev.device).reshape(1, -1, 1, 1)
    T_prev = blur(f_prev) + a_t * blur(c_t - c_prev) + b_t
    if form == 'prev' or f_next is None:
        return T_prev
    T_next = blur(f_next) - a_t * blur(c_next - c_t) - b_t
    return 0.5 * (T_prev + T_next)


def selfcal_projection(x1, blur, f_prev, c_prev, c_t, f_next, c_next,
                       lam: float, method: str = 'ols', form: str = 'bi_avg',
                       gate_q: float = 0.5):
    """Full Stage4 operation. Returns (x_out, a, b, r2_mean). GT-free."""
    a, b, r2 = fit_anchor_calib(blur, f_prev, c_prev, f_next, c_next,
                                method=method, gate_q=gate_q)
    T = anchor_target_estimate(blur, a, b, f_prev, c_prev, c_t,
                               f_next, c_next, form=form)
    M = blur(x1) - T
    x_out = x1 if lam == 0 else x1 - float(lam) * M
    return x_out, a, b, float(np.mean(r2))
