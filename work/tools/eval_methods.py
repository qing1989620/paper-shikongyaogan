# -*- coding: utf-8 -*-
"""Evaluate fixed-step / source-calibrated / self-calibrated projection methods.

Runs on prepared patch .npz files. Writes per-patch CSV + summary JSON so that
every reported number traces back to a raw run file.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from backbone_stfdiff import build, first_two_steps, official_10  # noqa: E402


# ------------------------------------------------------------------ metrics
def to01(x):
    return (x + 1.0) / 2.0


def rmse(x, y):
    return float(np.sqrt(np.mean((x - y) ** 2)))


def mae(x, y):
    return float(np.mean(np.abs(x - y)))


def psnr(x, y):
    r = rmse(to01(x), to01(y))
    return float(20 * np.log10(1.0 / max(r, 1e-8)))


def cc(x, y):
    vals = []
    for k in range(x.shape[-1]):
        a, b = x[..., k].ravel(), y[..., k].ravel()
        a = a - a.mean(); b = b - b.mean()
        den = np.sqrt((a * a).sum() * (b * b).sum())
        vals.append(float((a * b).sum() / den) if den > 1e-12 else 0.0)
    return float(np.mean(vals))


def sam(x, y):
    a = x.reshape(-1, x.shape[-1]); b = y.reshape(-1, y.shape[-1])
    num = (a * b).sum(1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    cos = np.clip(num / np.maximum(den, 1e-8), -1, 1)
    return float(np.degrees(np.arccos(cos)).mean())


def ergas(x, y):
    # computed in [0,1] reflectance domain so the per-band mean is physical
    x1, y1 = to01(x), to01(y)
    vals = []
    for k in range(x1.shape[-1]):
        m = max(float(np.mean(y1[..., k])), 1e-3)
        vals.append((rmse(x1[..., k], y1[..., k]) / m) ** 2)
    return float(100 * np.sqrt(np.mean(vals)))


def ssim(x, y):
    """Simple SSIM with 11x11 Gaussian window, per band, mean."""
    from scipy.ndimage import gaussian_filter
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    out = []
    for k in range(x.shape[-1]):
        a, b = x[..., k], y[..., k]
        mu_a = gaussian_filter(a, 1.5); mu_b = gaussian_filter(b, 1.5)
        sa = gaussian_filter(a * a, 1.5) - mu_a ** 2
        sb = gaussian_filter(b * b, 1.5) - mu_b ** 2
        sab = gaussian_filter(a * b, 1.5) - mu_a * mu_b
        num = (2 * mu_a * mu_b + C1) * (2 * sab + C2)
        den = (mu_a ** 2 + mu_b ** 2 + C1) * (sa + sb + C2)
        out.append(float((num / den).mean()))
    return float(np.mean(out))


ALL_METRICS = {"rmse": rmse, "mae": mae, "psnr": psnr, "cc": cc,
               "sam": sam, "ergas": ergas, "ssim": ssim}


def evaluate(x, y):
    return {k: f(x, y) for k, f in ALL_METRICS.items()}


# ------------------------------------------------------------------ helpers
def gaussian_blur_np(x, sigma):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(x, sigma=(sigma, sigma, 0), mode="reflect")


def fit_selfcal(f0, c0, f3, c3, sigma=2.0, gate_q=0.5, method="ols"):
    """Per-band affine LP(F3-F0) ~ a*LP(C3-C0)+b from the sample's own anchors."""
    dF = gaussian_blur_np(f3 - f0, sigma)
    dC = gaussian_blur_np(c3 - c0, sigma)
    a, b, r2 = [], [], []
    for k in range(dF.shape[-1]):
        x = dC[..., k].ravel().astype(np.float64)
        y = dF[..., k].ravel().astype(np.float64)
        gate = np.abs(x) >= np.quantile(np.abs(x), gate_q)
        x, y = x[gate], y[gate]
        if method == "ols":
            aa, bb = np.polyfit(x, y, 1)
            pred = aa * x + bb
        elif method == "scale":
            aa = float((x * y).sum() / max((x * x).sum(), 1e-12)); bb = 0.0
            pred = aa * x
        elif method == "identity":
            aa, bb = 1.0, 0.0; pred = x
        else:
            raise ValueError(method)
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2.append(1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0)
        a.append(float(aa)); b.append(float(bb))
    return np.array(a), np.array(b), float(np.mean(r2))


def anchor_target(f0, c0, c1, a, b, sigma=2.0, f3=None, c3=None, form="bi_avg"):
    T_prev = gaussian_blur_np(f0, sigma) + a * gaussian_blur_np(c1 - c0, sigma) + b
    if form == "prev" or f3 is None:
        return T_prev
    T_next = gaussian_blur_np(f3, sigma) - a * gaussian_blur_np(c3 - c1, sigma) - b
    return 0.5 * (T_prev + T_next)


def project(x1, T, lam, sigma=2.0):
    if lam == 0:
        return x1
    M = gaussian_blur_np(x1, sigma) - T
    return x1 - lam * M


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--lams", default="0.2,0.3,0.5,0.75")
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--arch", default="base")
    ap.add_argument("--calib", default="")
    ap.add_argument("--cache-dir", default="")
    args = ap.parse_args()

    files = sorted(Path(args.patches, args.split).glob("*.npz"))
    if args.limit:
        files = files[: args.limit]
    print(f"{len(files)} patches from {args.split}")
    d = build(args.ckpt, args.device, args.arch)
    lams = [float(v) for v in args.lams.split(",")]

    calib = json.loads(Path(args.calib).read_text()) if args.calib else None
    rows = []
    t_start = time.time()
    for i, fp in enumerate(files):
        z = np.load(fp)
        f0, f1, f3 = z["f0"].astype(np.float32), z["f1"].astype(np.float32), z["f3"].astype(np.float32)
        c0, c1, c3 = z["c0"].astype(np.float32), z["c1"].astype(np.float32), z["c3"].astype(np.float32)
        tens = lambda a: torch.from_numpy(a).permute(2, 0, 1)[None].to(args.device)
        t0, t1, t3 = tens(f0), tens(f1), tens(f3)
        u0, u1, u3 = tens(c0), tens(c1), tens(c3)

        tic = time.time()
        cache_dir = Path(args.cache_dir) if args.cache_dir else None
        cpath = cache_dir / f"{fp.stem}.npz" if cache_dir else None
        if cpath and cpath.exists():
            zz = np.load(cpath)
            x1 = torch.from_numpy(zz["x1"])[None]
            x2 = torch.from_numpy(zz["x2"])[None]
            x10 = torch.from_numpy(zz["x10"])[None]
            t_few = t_ten = 0.0
        else:
            x1, x2 = first_two_steps(d, u0, u1, u3, t0, t3, t1, seed=42 + i)
            t_few = time.time() - tic
            tic = time.time()
            x10 = official_10(d, u0, u1, u3, t0, t3, t1, seed=42 + i)
            t_ten = time.time() - tic
            if cpath:
                cpath.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cpath, x1=x1[0].cpu().numpy(), x2=x2[0].cpu().numpy(), x10=x10[0].cpu().numpy())

        preds = {
            "official10": x10[0].permute(1, 2, 0).cpu().numpy(),
            "fixed1": x1[0].permute(1, 2, 0).cpu().numpy(),
            "fixed2": x2[0].permute(1, 2, 0).cpu().numpy(),
        }
        # source-calibrated projection (Stage3), if calibration provided
        if calib is not None:
            al = np.array(calib["al"], dtype=np.float32).reshape(1, 1, -1)
            bl = np.array(calib["bl"], dtype=np.float32).reshape(1, 1, -1)
            T_src = gaussian_blur_np(f0, args.sigma) + al * gaussian_blur_np(c1 - c0, args.sigma) + bl
            for lam in lams:
                preds[f"src_proj_l{lam}"] = project(preds["fixed1"], T_src, lam, args.sigma)
        # self-calibrated projection (RSP)
        a, b, r2 = fit_selfcal(f0, c0, f3, c3, args.sigma)
        for form in ["prev", "bi_avg"]:
            T = anchor_target(f0, c0, c1, a, b, args.sigma, f3, c3, form)
            for lam in lams:
                preds[f"rsp_{form}_l{lam}"] = project(preds["fixed1"], T, lam, args.sigma)

        row = {"patch": fp.stem, "r2": r2, "t_few": t_few, "t_ten": t_ten}
        for name, p in preds.items():
            m = evaluate(p, f1)
            for k, v in m.items():
                row[f"{name}_{k}"] = v
        rows.append(row)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(files)} elapsed {time.time()-t_start:.0f}s", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader(); w.writerows(rows)

    # summary
    methods = sorted({k.rsplit("_", 1)[0] for k in rows[0] if k.split("_")[0] in
                      ("official10", "fixed1", "fixed2", "src", "rsp")})
    summary = {}
    for m in methods:
        cols = [k for k in rows[0] if k.startswith(m + "_") and k.split("_")[-1] in ALL_METRICS]
        summary[m] = {c.split("_")[-1]: float(np.mean([r[c] for r in rows])) for c in cols}
    summary["_meta"] = {"n": len(rows), "split": args.split,
                        "mean_t_few": float(np.mean([r["t_few"] for r in rows])),
                        "mean_t_ten": float(np.mean([r["t_ten"] for r in rows]))}
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: {kk: round(vv, 5) for kk, vv in v.items()} for k, v in summary.items() if k != "_meta"}, indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()
