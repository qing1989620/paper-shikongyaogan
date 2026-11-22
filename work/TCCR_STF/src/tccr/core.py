# -*- coding: utf-8 -*-
"""TCCR-STF core: protocol types, BSQ loader, temporal projection, correct-or-refine.

Terminology (see docs/PROTOCOL.md):
  fine_ref_prev  = Landsat at reference date t1 (earlier)
  fine_target    = Landsat at prediction date (GT; never used at inference)
  fine_ref_next  = Landsat at reference date t3 (later)
  coarse_ref_prev / coarse_pred / coarse_ref_next = MODIS 25m counterparts
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


# ----------------------------------------------------------------------------- data
BANDS = 6
LGC_H, LGC_W = 2720, 3200


def read_lgc_int_bsq(path: str | Path, rows: int = LGC_H, cols: int = LGC_W,
                     bands: int = BANDS) -> np.ndarray:
    """Read an LGC `.int` file with the CORRECT BSQ layout.

    WARNING: LGC .int files are band-sequential (BSQ). Reading them as BIL
    silently produces row-strided garbage (see docs/DATA_PREPARATION.md).

    Returns uint16 HWC array (rows, cols, bands).
    """
    data = np.fromfile(path, dtype='<u2')
    assert data.size == rows * cols * bands, \
        f'{path}: {data.size} != {rows}*{cols}*{bands}'
    return data.reshape(bands, rows, cols).transpose(1, 2, 0)


def read_cia_landsat_bil(path: str | Path, rows: int, cols: int,
                         bands: int = BANDS) -> np.ndarray:
    """CIA Landsat .bil files are genuine BIL (verified by masked NCC)."""
    data = np.fromfile(path, dtype='<u2')
    assert data.size == rows * cols * bands
    return data.reshape(rows, bands, cols).transpose(0, 2, 1)


def read_cia_modis_bsq(path: str | Path, rows: int, cols: int,
                       bands: int = BANDS) -> np.ndarray:
    """CIA MODIS .int files are genuine BSQ (per official readme)."""
    data = np.fromfile(path, dtype='<u2')
    assert data.size == rows * cols * bands
    return data.reshape(bands, rows, cols).transpose(1, 2, 0)


def sanitize_reflectance(x: np.ndarray, max_dn: int = 10000) -> np.ndarray:
    """Clip anomalous DN (fill values up to 65535) to [0, max_dn]."""
    return np.clip(x, 0, max_dn).astype(np.uint16)


def normalize_minus_one_one(x: np.ndarray | torch.Tensor):
    """DN [0,10000] → [-1,1] (STFDiff convention)."""
    if isinstance(x, np.ndarray):
        return x.astype(np.float32) / 10000.0 * 2.0 - 1.0
    return x / 10000.0 * 2.0 - 1.0


# ------------------------------------------------------------------- projection
@dataclass
class TCCRConfig:
    """All method hyper-parameters in one place (frozen per-domain calibration)."""
    lambda_proj: float = 0.5      # projection step size
    tau_echange: float = 0.02178  # routing threshold on E_change
    direction: str = 'high'       # 'high': E_change > tau -> CORRECT
    blur_sigma: float = 2.0       # low-pass (Gaussian) kernel sigma for LP()
    a: tuple = ()                 # per-band D_obs affine scale (6,)
    b: tuple = ()                 # per-band D_obs affine offset (6,)
    al: tuple = ()                # per-band temporal-change affine scale (6,)
    bl: tuple = ()                # per-band temporal-change affine offset (6,)
    version: int = 1

    @classmethod
    def from_json(cls, path: str | Path) -> 'TCCRConfig':
        return cls(**json.loads(Path(path).read_text()))

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.__dict__, indent=1))


def gaussian_blur(x: torch.Tensor, sigma: float = 2.0) -> torch.Tensor:
    """Separable Gaussian low-pass. x: (B,C,H,W). Reflect padding."""
    import math
    k = int(math.ceil(sigma * 3))
    xs = torch.arange(-k, k + 1, dtype=x.dtype, device=x.device)
    g = torch.exp(-xs ** 2 / (2 * sigma ** 2))
    g = (g / g.sum()).reshape(1, 1, -1)
    c = x.shape[1]
    x = torch.nn.functional.pad(x, (k, k, 0, 0), mode='reflect')
    x = torch.nn.functional.conv2d(x, g.expand(c, 1, 1, -1), groups=c)
    x = torch.nn.functional.pad(x, (0, 0, k, k), mode='reflect')
    x = torch.nn.functional.conv2d(x, g.transpose(1, 2).expand(c, 1, -1, 1), groups=c)
    return x


def temporal_mismatch(x1: torch.Tensor, fine_ref_prev: torch.Tensor,
                      coarse_ref_prev: torch.Tensor, coarse_pred: torch.Tensor,
                      cfg: TCCRConfig) -> torch.Tensor:
    """Low-frequency temporal mismatch M (B,C,H,W).

    M = LP(x1 - fine_ref_prev) - (al * LP(coarse_pred - coarse_ref_prev) + bl)

    al/bl map the coarse temporal change into the fine-radiometric scale
    (fitted on the *training* split of the calibration domain).
    """
    d_fine_pred = gaussian_blur(x1 - fine_ref_prev, cfg.blur_sigma)
    d_coarse_obs = gaussian_blur(coarse_pred - coarse_ref_prev, cfg.blur_sigma)
    al = torch.as_tensor(cfg.al, dtype=x1.dtype, device=x1.device).reshape(1, -1, 1, 1)
    bl = torch.as_tensor(cfg.bl, dtype=x1.dtype, device=x1.device).reshape(1, -1, 1, 1)
    return d_fine_pred - (al * d_coarse_obs + bl)


def temporal_projection(x1: torch.Tensor, M: torch.Tensor,
                        lam: float) -> torch.Tensor:
    """x_corr = x1 - lam * M. With lam == 0 this is EXACTLY x1 (unit-tested)."""
    if lam == 0:
        return x1
    return x1 - float(lam) * M


def e_change(x1: torch.Tensor, fine_ref_prev: torch.Tensor,
             coarse_ref_prev: torch.Tensor, coarse_pred: torch.Tensor,
             cfg: TCCRConfig) -> float:
    """No-GT routing certificate: mean |LP(x1 - f0) - (al*LP(c1 - c0) + bl)|."""
    M = temporal_mismatch(x1, fine_ref_prev, coarse_ref_prev, coarse_pred, cfg)
    return float(M.abs().mean())


def route(x1: torch.Tensor, x2: torch.Tensor | None, M: torch.Tensor,
          e_change: float, lam: float, cfg: TCCRConfig) -> tuple[torch.Tensor, str, int]:
    """Correct-or-Refine decision.

    Returns (output_tensor, action, nfe_used).
    CORRECT  -> x1 - lam*M   (nfe=1)
    REFINE   -> x2 (second DDIM step already computed by caller)  (nfe=2)
    """
    hi = e_change > cfg.tau_echange
    do_correct = hi if cfg.direction == 'high' else (not hi)
    if do_correct:
        return temporal_projection(x1, M, lam), 'CORRECT', 1
    assert x2 is not None, 'REFINE branch requires the second-step prediction'
    return x2, 'REFINE', 2
