# -*- coding: utf-8 -*-
"""Stage3 Step 6-8: Second-Step Residual Anatomy + Reference Leakage + Oracle。
对每 patch（val_stable 与 test_flood_legacy 各 48p，eta=0, seed42）:
  x1, x2 已有（router perpatch 有 RMSE；此处重推理拿张量以算 cos/频域分解）
  r = x2 - x1; e1 = F2 - x1
  alignment_total = cos(r, e1)
  r_L = LP(r), r_H = r - r_L; e1_L/e1_H 同理 → alignment_low/high
  leakage = cos(r_L, LP(F0 - x1))   [reference 方向]
  coarse_change_alignment = cos(r_L, LP(C1 - C0 up))
  true_change_alignment = cos(r_L, LP(F2 - F0))
  low/high freq error1/error2
  delta_RMSE = RMSE2 - RMSE1
  E_change@step1（Stage2 已有公式）
输出: 03_residual_anatomy/residual_anatomy.csv + 04_oracle/oracle_{binary,soft,frequency}.csv
Oracle soft: x_lambda = x1 + λ r, λ∈{0..1,0.1}; oracle λ per patch（analysis only）
Oracle frequency: x1 + λL rL + λH rH, grid 5×5（dev only）
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

STAGE1 = Path(r"D:\qing_zhuomian_\科研区\陈达沛论文\寻找baseline\experiments\FDP_Diff_20260908_1326")
S2 = Path(r"D:\qing_zhuomian_\科研区\陈达沛论文\寻找baseline\experiments\STAGE2_Sampling_20260908_1837")
S3 = Path(r"D:\qing_zhuomian_\科研区\陈达沛论文\寻找baseline\experiments\STAGE3_TCAR_20260909_0035")
ROOT = STAGE1.parent.parent  # 寻找baseline 根
sys.path.insert(0, str(STAGE1 / 'configs'))
sys.path.insert(0, str(S2 / 'configs'))
sys.path.insert(0, str(ROOT / 'work' / 'experiments'))
sys.path.insert(0, str(ROOT / 'work'))

from common import build_diffusion, build_metrics, compute_metrics  # noqa: E402
from exp_data import SYYPatchDataset, fixed_subset  # noqa: E402
from s2_eval import make_time_pairs, ObservationOp  # noqa: E402

CKPT = S3 / 'checkpoints' / 'bsq_baseline' / 'best_val.pth'
OP_JSON = S3 / 'results' / '07_proxy_bsq' / 'op_d_bsq.json'


def lp(x, op):
    """低频 = D 算子的 blur 部分（σ=2.0, 已在 op 内），用 op.blur"""
    return op.blur(x)


@torch.no_grad()
def run_split(d, op, metrics, data_root, split_tag, out_anatomy, out_oracle,
              device='cuda', n_patches=48, noise_seed=42):
    ds = SYYPatchDataset(str(data_root))
    patches = fixed_subset(ds, n_patches, seed=42)
    pairs2 = make_time_pairs(2)
    recs = []
    t0 = time.time()
    for i, s in enumerate(patches):
        c1 = s['coarse_img_01'][None].to(device)
        c2 = s['coarse_img_02'][None].to(device)
        c3 = s['coarse_img_03'][None].to(device)
        f1 = s['fine_img_01'][None].to(device)
        f2 = s['fine_img_02'][None].to(device)
        f3 = s['fine_img_03'][None].to(device)
        torch.manual_seed(noise_seed + i)
        x = torch.randn_like(f1)
        t0t, t1t = pairs2[0]
        tc = torch.full((x.shape[0],), t0t, device=device, dtype=torch.long)
        pred_noise, x1, *_ = d.model_predictions(c1, c2, c3, f1, x, f3, tc,
                                                 None, clip_x_start=True)
        # step2 DDIM 更新（eta=0）
        alpha = d.alphas_cumprod[t0t]
        alpha_next = d.alphas_cumprod[t1t] if t1t >= 0 else d.alphas_cumprod[0]
        if t1t < 0:
            x2 = x1
        else:
            c = (1 - alpha_next).sqrt()
            x = x1 * alpha_next.sqrt() + c * pred_noise
            tc2 = torch.full((x.shape[0],), t1t, device=device, dtype=torch.long)
            _, x2, *_ = d.model_predictions(c1, c2, c3, f1, x, f3, tc2, None,
                                            clip_x_start=True)
        m1 = compute_metrics(x1, f2, metrics, device=device)
        m2 = compute_metrics(x2, f2, metrics, device=device)

        r = x2 - x1
        e1 = f2 - x1
        rL, rH = lp(r, op), r - lp(r, op)
        e1L, e1H = lp(e1, op), e1 - lp(e1, op)

        def cos(a, b):
            a = a.reshape(-1).float()
            b = b.reshape(-1).float()
            na, nb = a.norm(), b.norm()
            return float((a * b).sum() / (na * nb)) if na > 1e-12 and nb > 1e-12 else 0.0

        d_ref = lp(f1 - x1, op)                     # reference direction LP(F0 - x1)
        # coarse change: C1 - C0
        c0 = s['coarse_img_01'][None].to(device)
        coarse_change = lp(c1 - c0, op)
        true_change = lp(f2 - f1, op)

        err1_l = float((e1L ** 2).mean().sqrt())
        e2 = f2 - x2
        e2L = lp(e2, op)
        err2_l = float((e2L ** 2).mean().sqrt())
        err2_h = float(((e2 - e2L) ** 2).mean().sqrt())
        err1_h = float((e1H ** 2).mean().sqrt())

        e_change1 = op.d_change(x1, f1, c1, c2)

        recs.append({
            'patch': i, 'scene': split_tag,
            'E_change': e_change1,
            'RMSE_1': float(m1['RMSE']), 'RMSE_2': float(m2['RMSE']),
            'delta_RMSE': float(m2['RMSE']) - float(m1['RMSE']),
            'alignment_total': cos(r, e1),
            'alignment_low': cos(rL, e1L),
            'alignment_high': cos(rH, e1H),
            'reference_leakage': cos(rL, d_ref),
            'coarse_change_alignment': cos(rL, coarse_change),
            'true_change_alignment': cos(rL, true_change),
            'lowfreq_error1': err1_l, 'lowfreq_error2': err2_l,
            'highfreq_error1': err1_h, 'highfreq_error2': err2_h,
        })
        if (i + 1) % 16 == 0:
            print(f'[{split_tag}] {i+1}/{len(patches)} ({time.time()-t0:.0f}s)', flush=True)

    out_anatomy.mkdir(parents=True, exist_ok=True)
    with open(out_anatomy / f'residual_anatomy_{split_tag}.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)

    # ---- Oracle binary & soft & frequency（analysis-only，用 GT）----
    ora = []
    for r in recs:
        ora.append({'patch': r['patch'], 'scene': split_tag,
                    'RMSE_1': r['RMSE_1'], 'RMSE_2': r['RMSE_2'],
                    'oracle_binary': min(r['RMSE_1'], r['RMSE_2'])})
    # soft oracle 需要逐 λ RMSE → 重算需要 x1/x2 缓存；此处用线性插值近似记录结构，
    # 精确 soft oracle 在 oracle_soft.py 里做（需要缓存 x1,x2 张量）。
    out_oracle.mkdir(parents=True, exist_ok=True)
    with open(out_oracle / f'oracle_binary_{split_tag}.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(ora[0].keys()))
        w.writeheader()
        w.writerows(ora)
    print(f'[{split_tag}] anatomy+oracle done → {out_anatomy}', flush=True)
    return recs


@torch.no_grad()
def main():
    device = 'cuda'
    d = build_diffusion()
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    d.load_state_dict(ck['model'])
    d = d.to(device).eval()
    op = ObservationOp(str(OP_JSON))
    metrics = build_metrics()

    val_root = S3 / '02_reproducibility' / 'bsq_data' / 'syy9_val'
    test_root = S3 / '02_reproducibility' / 'bsq_data' / 'syy9_test'
    run_split(d, op, metrics, val_root, 'bsq_val',
              S3 / 'results' / '03_residual_anatomy', S3 / 'results' / '04_oracle')
    run_split(d, op, metrics, test_root, 'bsq_flood',
              S3 / 'results' / '03_residual_anatomy', S3 / 'results' / '04_oracle')


if __name__ == '__main__':
    main()
