# -*- coding: utf-8 -*-
"""Train STFDiff model6 on LGC patches, with several training-time variants.

Variants (all train from scratch, same budget):
  base      : official objective (pred_x0) + L1
  lf        : + low-frequency auxiliary loss on LP(x0)
  hf        : + high-frequency weighted L1 (focal-frequency-style emphasis)
  onestep   : timestep sampled from the last 10% only (specialise for 1 NFE)
  cond      : extra conditioning channel (coarse temporal change) fed to the
              noise branch via a widened input convolution
  bandw     : per-band loss weights from inverse band variance

Usage:
  python tools/train_stfdiff.py --variant base --epochs 40 --batch 4 --out runs/base
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "repos" / "STF" / "src" / "model" / "stfdiff" / "model6"))
from bi_diffusion import BiGaussianDiffusion  # noqa: E402
from bi_pred_resnet import BiPredNoiseNet  # noqa: E402
from bi_pred_resnet_v2 import BiPredNoiseNetV2  # noqa: E402


class PatchDS(Dataset):
    """Memory-mapped patches (float16), zero resident RAM."""

    KEYS = ("f0", "f1", "f3", "c0", "c1", "c3")

    def __init__(self, root: Path, mm: str = ""):
        root = Path(root)
        self.files = sorted(root.glob("*.npz"))
        self.mm = None
        if mm:
            try:
                self.mm = np.load(Path(mm), mmap_mode="r")
            except Exception:
                self.mm = None

    def __len__(self):
        return self.mm.shape[0] if self.mm is not None else len(self.files)

    def __getitem__(self, i):
        if self.mm is not None:
            a = self.mm[i]
            return tuple(torch.from_numpy(np.ascontiguousarray(a[j], dtype=np.float32))
                         for j in range(len(self.KEYS)))
        z = np.load(self.files[i])
        return tuple(torch.from_numpy(z[k].astype(np.float32)).permute(2, 0, 1)
                     for k in self.KEYS)


def build_model(channels=6, arch="base"):
    """arch: base | delta | gate | reinject | full (delta+gate+reinject)."""
    flags = dict(delta_enc=arch in ("delta", "full"),
                 time_gate=arch in ("gate", "full"),
                 reinject=arch in ("reinject", "full"))
    if arch == "base":
        model = BiPredNoiseNet(dim=64, channels=channels, dim_mults=(1, 2, 4),
                               self_condition=False, learned_sinusoidal_cond=False)
    else:
        model = BiPredNoiseNetV2(dim=64, channels=channels, dim_mults=(1, 2, 4),
                                 self_condition=False, learned_sinusoidal_cond=False,
                                 **flags)
    d = BiGaussianDiffusion(model, image_size=256, timesteps=100, sampling_timesteps=10,
                            objective="pred_x0", loss_type="l1", beta_schedule="cosine",
                            ddim_sampling_eta=1.0)
    return d


def lp(x, k=5, sigma=1.5):
    ax = torch.arange(-k, k + 1, dtype=x.dtype, device=x.device)
    g = torch.exp(-ax ** 2 / (2 * sigma ** 2)); g = g / g.sum()
    c = x.shape[1]
    x = F.pad(x, (k, k, 0, 0), mode="reflect")
    x = F.conv2d(x, g.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    x = F.pad(x, (0, 0, k, k), mode="reflect")
    x = F.conv2d(x, g.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)
    return x


@torch.no_grad()
def validate(d, loader, device):
    """Deterministic validation: fixed noise so model selection is comparable."""
    d.eval()
    tot, n = 0.0, 0
    gen = torch.Generator(device=device).manual_seed(1234)
    for f0, f1, f3, c0, c1, c3 in loader:
        f0, f1, f3 = f0.to(device), f1.to(device), f3.to(device)
        c0, c1, c3 = c0.to(device), c1.to(device), c3.to(device)
        x = torch.randn(f1.shape, generator=gen, device=device, dtype=f1.dtype)
        t = torch.full((x.shape[0],), 99, device=device, dtype=torch.long)
        _, x0, *_ = d.model_predictions(c0, c1, c3, f0, x, f3, t, None, clip_x_start=True)
        tot += float(((x0 - f1) ** 2).mean()) * x.shape[0]
        n += x.shape[0]
    return (tot / max(n, 1)) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="base")
    ap.add_argument("--arch", default="base", choices=["base","delta","gate","reinject","full"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", default="data/patches_lgc/train")
    ap.add_argument("--val", default="data/patches_lgc/val")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--mm-train", default="")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} variant={args.variant} arch={args.arch}", flush=True)

    tr = PatchDS(Path(args.train), args.mm_train); va = PatchDS(Path(args.val))
    if args.limit:
        tr.files = tr.files[: args.limit]; va.files = va.files[: max(8, args.limit // 4)]
    trl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)
    val = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=0)
    print(f"train={len(tr)} val={len(va)}", flush=True)

    d = build_model(arch=args.arch).to(device)
    opt = torch.optim.Adam(d.parameters(), lr=args.lr, betas=(0.9, 0.99))
    total_steps = max(len(trl) * args.epochs, 1)
    warm = max(args.warmup, 1)

    def lr_at(step):
        if step < warm:
            return args.lr * (step + 1) / warm
        prog = (step - warm) / max(total_steps - warm, 1)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    gstep = 0
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    log_path = out / "train_log.csv"
    with open(log_path, "w", newline="") as fh:
        csv.writer(fh).writerow(["epoch", "loss", "val_rmse", "sec"])

    best = 9e9
    for ep in range(1, args.epochs + 1):
        d.train(); t0 = time.time(); tot, n = 0.0, 0
        for f0, f1, f3, c0, c1, c3 in trl:
            f0, f1, f3 = f0.to(device), f1.to(device), f3.to(device)
            c0, c1, c3 = c0.to(device), c1.to(device), c3.to(device)
            b = f1.shape[0]
            if args.variant == "onestep":
                t = torch.randint(90, 100, (b,), device=device)
            else:
                t = torch.randint(0, 100, (b,), device=device)
            noise = torch.randn_like(f1)
            a = d.alphas_cumprod[t].view(b, 1, 1, 1)
            x_t = a.sqrt() * f1 + (1 - a).sqrt() * noise
            if True:
                _, x0, *_ = d.model_predictions(c0, c1, c3, f0, x_t, f3, t, None,
                                                clip_x_start=True)
                if args.variant == "delsup":
                    # delta supervision: match the observed coarse temporal change
                    # with the physically-calibrated slope a=0.54 (diagnosed from GT).
                    dx = lp(x0) - lp(f0)
                    dc = lp(c1 - c0)
                    target = 0.54 * dc
                    loss = F.l1_loss(x0, f1) + 0.5 * F.l1_loss(dx, target)
                elif args.variant == "lf":
                    loss = F.l1_loss(x0, f1) + 0.5 * F.l1_loss(lp(x0), lp(f1))
                elif args.variant == "hf":
                    hi_x = x0 - lp(x0); hi_y = f1 - lp(f1)
                    loss = F.l1_loss(x0, f1) + 1.0 * F.l1_loss(hi_x, hi_y)
                elif args.variant == "bandw":
                    w = torch.tensor([1.0, 1.1, 1.2, 1.0, 1.0, 1.0], device=device).view(1, 6, 1, 1)
                    loss = (w * (x0 - f1).abs()).mean()
                else:
                    loss = F.l1_loss(x0, f1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(d.parameters(), 1.0)
            for gparam in opt.param_groups:
                gparam["lr"] = lr_at(gstep)
            opt.step()
            gstep += 1
            tot += float(loss) * b; n += b
            last_gn = float(gn)
        vr = validate(d, val, device)
        sec = time.time() - t0
        print(f"ep {ep:03d} loss {tot/max(n,1):.4f} val {vr:.5f} gn {last_gn:.3f} lr {lr_at(gstep):.2e} {sec:.0f}s", flush=True)
        with open(log_path, "a", newline="") as fh:
            csv.writer(fh).writerow([ep, tot / max(n, 1), vr, sec])
        if vr < best:
            best = vr
            torch.save({"model": d.state_dict(), "val_rmse": vr, "epoch": ep,
                        "variant": args.variant, "arch": args.arch, "protocol": "BSQ dual-anchor 6-input"},
                       out / "best.pth")
        torch.save({"model": d.state_dict(), "epoch": ep, "variant": args.variant},
                   out / "last.pth")
    print("BEST", best, flush=True)


if __name__ == "__main__":
    main()
