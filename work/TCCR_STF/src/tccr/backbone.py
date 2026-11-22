# -*- coding: utf-8 -*-
"""Thin backbone wrapper: adapts the STFDiff BiGaussianDiffusion to TCCR.

In this release the STFDiff sources are expected at repos/STF (see
docs/DATA_PREPARATION.md). The wrapper isolates all backbone-specific details so
that swapping backbones later only requires reimplementing this file.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

# --- STFDiff import bootstrap (adjust STF_REPO if placed differently) --------
_HERE = Path(__file__).resolve().parent
CANDIDATES = [
    _HERE.parents[3] / 'repos' / 'STF',
    Path(__file__).resolve().parents[4] / 'repos' / 'STF',
]
for c in CANDIDATES:
    if (c / 'src' / 'model').exists():
        STF_REPO = c
        break
else:
    STF_REPO = Path(__file__).resolve().parents[4] / 'repos' / 'STF'

sys.path.insert(0, str(STF_REPO / 'src' / 'model' / 'stfdiff' / 'model6'))
sys.path.insert(0, str(STF_REPO / 'src'))


def build_backbone(ckpt_path: str, device: str = 'cuda'):
    """Load BiGaussianDiffusion with the official dual-anchor config."""
    from common import build_diffusion  # provided by STFDiff tooling
    d = build_diffusion()
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    d.load_state_dict(ck['model'])
    return d.to(device).eval()


@torch.no_grad()
def first_step(d):
    """Return the (t_first, t_next) pair of the official 2-step schedule."""
    # official uniform linspace over T=100 with 2 steps: [99, 49]
    ts = np_linspace_steps(100, 2)
    return int(ts[0]), int(ts[1])


def np_linspace_steps(T: int, n: int):
    import numpy as np
    return np.linspace(T - 1, 0, n).round().astype(int)


@torch.no_grad()
def ddim_step1_and_optional_step2(d, c0, c1, c3, f1, f3, f2_shape_ref,
                                  eta: float = 0.0, seed: int = 42):
    """Run the official 2-step DDIM (eta=0). Returns (x1, x2).

    x2 is always computed here for calibration/benchmark purposes; the TCCR
    router decides at inference whether x2 is actually used.
    """
    dev = f1.device
    torch.manual_seed(seed)
    x = torch.randn_like(f2_shape_ref)
    t0, t1 = first_step(d)
    tc = torch.full((x.shape[0],), t0, device=dev, dtype=torch.long)
    pn, x1, *_ = d.model_predictions(c0, c1, c3, f1, x, f3, tc, None,
                                     clip_x_start=True)
    alpha_n = d.alphas_cumprod[t1]
    x = x1 * alpha_n.sqrt() + (1 - alpha_n).sqrt() * pn
    tc2 = torch.full((x.shape[0],), t1, device=dev, dtype=torch.long)
    _, x2, *_ = d.model_predictions(c0, c1, c3, f1, x, f3, tc2, None,
                                    clip_x_start=True)
    return x1, x2


@torch.no_grad()
def second_step_only(d, c0, c1, c3, f1, f3, f2_shape_ref, seed: int = 42):
    """Only the 2-step forward (for REFINE branch latencies)."""
    return ddim_step1_and_optional_step2(d, c0, c1, c3, f1, f3, f2_shape_ref, seed=seed)
