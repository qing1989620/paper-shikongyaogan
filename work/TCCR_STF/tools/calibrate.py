# -*- coding: utf-8 -*-
"""Fit TCCR calibration (12 scalars + lambda + tau) on a domain's TRAIN/VAL split.

Never uses the test split. Backbone is frozen throughout.
Usage:
  python tools/calibrate.py --ckpt best_rmse.pth --data_root data/lgc_bsq \
      --out calibration/lgc.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from tccr.core import TCCRConfig, gaussian_blur  # noqa: E402

# The diffusion backbone wrapper lives with the training code; in the release it
# is expected at src/tccr/backbone.py (thin wrapper around the STFDiff repo).
from tccr.backbone import build_backbone, ddim_step1_and_optional_step2  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data_root', required=True, help='contains train/ and val/ Group dirs')
    ap.add_argument('--out', required=True)
    ap.add_argument('--n_train_pairs', type=int, default=200)
    ap.add_argument('--n_val_patches', type=int, default=48)
    ap.add_argument('--blur_sigma', type=float, default=2.0)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    root = Path(args.data_root)
    device = 'cuda'

    d = build_backbone(args.ckpt, device)
    from exp_data import SYYPatchDataset, fixed_subset  # STFDiff data contract
    train_patches = fixed_subset(SYYPatchDataset(str(root / 'train')),
                                 args.n_train_pairs, seed=7)
    val_patches = fixed_subset(SYYPatchDataset(str(root / 'val')),
                               args.n_val_patches, seed=42)
    pairs2 = make_pairs = None
    from tccr.backbone import first_step, second_step  # noqa: E402
    t0, t1 = first_step(d)

    # ---- 1) D_obs affine: blur(fine_ref_prev) -> coarse_ref_prev (per band) ----
    XA, XB = [], []
    for s in train_patches:
        f0 = s['fine_img_01'][None].to(device)
        c0 = s['coarse_img_01'][None].to(device)
        bf = gaussian_blur(f0, args.blur_sigma)
        XA.append(bf.reshape(6, -1).permute(1, 0).cpu())
        XB.append(c0.reshape(6, -1).permute(1, 0).cpu())
    A = torch.cat(XA).numpy(); B = torch.cat(XB).numpy()
    a, b = [], []
    for k in range(6):
        aa, bb = np.polyfit(A[:, k], B[:, k], 1)
        a.append(float(aa)); b.append(float(bb))

    # ---- 2) temporal-change affine: LP(c1-c0) -> LP(f1-f0) (per band) ----
    AL, BL = [], []
    for s in train_patches:
        f0 = s['fine_img_01'][None].to(device); f1 = s['fine_img_02'][None].to(device)
        c0 = s['coarse_img_01'][None].to(device); c1 = s['coarse_img_02'][None].to(device)
        lf = gaussian_blur(f1 - f0, args.blur_sigma).reshape(6, -1).permute(1, 0).cpu()
        lc = gaussian_blur(c1 - c0, args.blur_sigma).reshape(6, -1).permute(1, 0).cpu()
        AL.append(lf); BL.append(lc)
    AL = torch.cat(AL).numpy(); BL = torch.cat(BL).numpy()
    al, bl = [], []
    for k in range(6):
        aa, bb = np.polyfit(BL[:, k], AL[:, k], 1)
        al.append(float(aa)); bl.append(float(bb))

    cfg = TCCRConfig(blur_sigma=args.blur_sigma, a=tuple(a), b=tuple(b),
                     al=tuple(al), bl=tuple(bl))

    # ---- 3) collect val predictions once; grid-search (lambda, tau, direction) ----
    al_t = torch.as_tensor(al, device=device).reshape(1, 6, 1, 1)
    bl_t = torch.as_tensor(bl, device=device).reshape(1, 6, 1, 1)
    cache = []
    for i, s in enumerate(val_patches):
        c0 = s['coarse_img_01'][None].to(device); c1 = s['coarse_img_02'][None].to(device)
        c3 = s['coarse_img_03'][None].to(device)
        f0 = s['fine_img_01'][None].to(device); f1 = s['fine_img_01'][None].to(device)
        f2 = s['fine_img_02'][None].to(device); f3 = s['fine_img_03'][None].to(device)
        torch.manual_seed(args.seed + i)
        x1, x2 = ddim_step1_and_optional_step2(d, c0, c1, c3, f1, f3, f2,
                                               seed=args.seed + i)
        M = gaussian_blur(x1 - f0, args.blur_sigma) - \
            (al_t * gaussian_blur(c1 - c0, args.blur_sigma) + bl_t)
        cache.append({'x1': x1, 'x2': x2, 'M': M, 'f2': f2})
    from s2_metrics import rmse_of  # thin helper
    rmse1 = float(np.mean([rmse_of(c['x1'], c['f2']) for c in cache]))

    lams = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75]
    quants = np.round(np.arange(0.1, 0.95, 0.05), 2)
    best = None
    ecs = np.array([e_change_of(c, cfg) for c in cache])
    for direction in ['high', 'low']:
        for q in quants:
            tau = float(np.quantile(ecs, q))
            for lam in lams:
                vals = []
                nfes = []
                for c in cache:
                    ec = e_change_of(c, cfg)
                    hi = ec > tau
                    do_correct = hi if direction == 'high' else (not hi)
                    if do_correct:
                        vals.append(rmse_of(temporal_projection(c['x1'], c['M'], lam),
                                            c['f2']))
                        nfes.append(1)
                    else:
                        vals.append(rmse_of(c['x2'], c['f2'])); nfes.append(2)
                key = (round(float(np.mean(vals)), 7), -float(np.mean(nfes)))
                if best is None or key < best[0]:
                    best = (key, lam, tau, direction, float(np.mean(vals)),
                            float(np.mean(nfes)))
    _, lam, tau, direction, rmse_tccr, avg_nfe = best

    calib = {
        'dataset': root.name, 'bands': 6,
        'a': a, 'b': b, 'al': al, 'bl': bl,
        'lambda': lam, 'tau': tau, 'direction': direction,
        'blur_sigma': args.blur_sigma,
        'fit_split': 'train (affines) + val (lambda/tau/direction)',
        'n_train_pairs': args.n_train_pairs, 'n_val_patches': args.n_val_patches,
        'protocol': 'STFDiff dual-anchor 6-input, BSQ semantics',
        'val_rmse_fixed1': rmse1, 'val_rmse_tccr': rmse_tccr, 'avg_nfe': avg_nfe,
        'version': 1,
    }
    Path(args.out).write_text(json.dumps(calib, indent=1))
    print(f"frozen: lam={lam} tau={tau:.5f} dir={direction} | "
          f"val fixed1={rmse1:.5f} tccr={rmse_tccr:.5f} avgNFE={avg_nfe:.2f}")
    print('saved →', args.out)


def e_change_of(c, cfg):
    return float(c['M'].abs().mean())


if __name__ == '__main__':
    main()
