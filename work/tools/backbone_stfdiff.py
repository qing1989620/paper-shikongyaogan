# -*- coding: utf-8 -*-
"""Build the STFDiff model6 backbone and load the released TCCR/Stage4 checkpoint.

Validates the whole inference path (no data needed) so that once patches are
prepared we can run fixed-1 / fixed-2 / projection / RSP immediately.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "repos" / "STF" / "src" / "model" / "stfdiff" / "model6"
sys.path.insert(0, str(MODEL_DIR))

from bi_diffusion import BiGaussianDiffusion  # noqa: E402
from bi_pred_resnet import BiPredNoiseNet  # noqa: E402
from bi_pred_resnet_v2 import BiPredNoiseNetV2  # noqa: E402


def build(ckpt_path: str | Path, device: str = "cpu", arch: str = "base"):
    if arch == "base":
        model = BiPredNoiseNet(dim=64, channels=6, dim_mults=(1, 2, 4),
                               self_condition=False, learned_sinusoidal_cond=False)
    else:
        model = BiPredNoiseNetV2(dim=64, channels=6, dim_mults=(1, 2, 4),
                                 self_condition=False, learned_sinusoidal_cond=False,
                                 delta_enc=arch in ("delta", "full"),
                                 time_gate=arch in ("gate", "full"),
                                 reinject=arch in ("reinject", "full"))
    diffusion = BiGaussianDiffusion(
        model,
        image_size=256,
        timesteps=100,
        sampling_timesteps=10,
        objective="pred_x0",
        loss_type="l1",
        beta_schedule="cosine",
        ddim_sampling_eta=1.0,
    )
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = ck["model"] if "model" in ck else ck
    missing, unexpected = diffusion.load_state_dict(sd, strict=False)
    # betas / alphas buffers are recomputed identically; only report real params
    real_missing = [k for k in missing if not k.startswith(("betas", "alphas", "sqrt", "log_"))]
    real_unexpected = [k for k in unexpected]
    print(f"loaded: missing={len(real_missing)} unexpected={len(real_unexpected)}")
    if real_missing:
        print("  missing sample:", real_missing[:5])
    if real_unexpected:
        print("  unexpected sample:", real_unexpected[:5])
    meta = {k: v for k, v in ck.items() if k != "model"} if "model" in ck else {}
    print("meta:", meta)
    return diffusion.to(device).eval()


@torch.no_grad()
def first_two_steps(d, c0, c1, c3, f0, f3, shape_ref, seed=42):
    """Official 2-step DDIM (eta=0): t=99 -> x1, t=49 -> x2."""
    dev = shape_ref.device
    torch.manual_seed(seed)
    x = torch.randn_like(shape_ref)
    t0, t1 = 99, 49
    tc = torch.full((x.shape[0],), t0, device=dev, dtype=torch.long)
    pn, x1, *_ = d.model_predictions(c0, c1, c3, f0, x, f3, tc, None, clip_x_start=True)
    a1 = d.alphas_cumprod[t1]
    x = x1 * a1.sqrt() + (1 - a1).sqrt() * pn
    tc2 = torch.full((x.shape[0],), t1, device=dev, dtype=torch.long)
    pn2, x2, *_ = d.model_predictions(c0, c1, c3, f0, x, f3, tc2, None, clip_x_start=True)
    return x1, x2


@torch.no_grad()
def official_10(d, c0, c1, c3, f0, f3, shape_ref, seed=42):
    """Official DDIM 10 steps, eta=1.0 (matching STFDiff defaults)."""
    dev = shape_ref.device
    torch.manual_seed(seed)
    x = torch.randn_like(shape_ref)
    times = torch.linspace(-1, d.num_timesteps - 1, steps=d.sampling_timesteps + 1)
    times = list(reversed(times.int().tolist()))
    pairs = list(zip(times[:-1], times[1:]))
    for t, tn in pairs:
        tc = torch.full((x.shape[0],), int(t), device=dev, dtype=torch.long)
        pn, x0, *_ = d.model_predictions(c0, c1, c3, f0, x, f3, tc, None, clip_x_start=True)
        if tn < 0:
            x = x0
            continue
        a, an = d.alphas_cumprod[int(t)], d.alphas_cumprod[int(tn)]
        eta = d.ddim_sampling_eta
        sigma = eta * ((1 - a / an) * (1 - an) / (1 - a)).sqrt()
        c = (1 - an - sigma ** 2).sqrt()
        noise = torch.randn_like(x)
        x = x0 * an.sqrt() + c * pn + sigma * noise
    return x


if __name__ == "__main__":
    ck = ROOT / "TCCR_STF" / "checkpoints" / "tccr_stfdiff_lgc_bsq_best.pth"
    d = build(ck, "cpu")
    B, C, H, W = 1, 6, 256, 256
    c0 = torch.randn(B, C, H, W) * 0.1
    c1 = torch.randn(B, C, H, W) * 0.1
    c3 = torch.randn(B, C, H, W) * 0.1
    f0 = torch.randn(B, C, H, W) * 0.1
    f3 = torch.randn(B, C, H, W) * 0.1
    x1, x2 = first_two_steps(d, c0, c1, c3, f0, f3, f0, seed=42)
    print("x1", tuple(x1.shape), float(x1.mean()), float(x1.std()))
    print("x2", tuple(x2.shape), float(x2.mean()), float(x2.std()))
    y = official_10(d, c0, c1, c3, f0, f3, f0, seed=42)
    print("y10", tuple(y.shape), float(y.mean()), float(y.std()))
    print("PARAMS(M)", sum(p.numel() for p in d.parameters()) / 1e6)
