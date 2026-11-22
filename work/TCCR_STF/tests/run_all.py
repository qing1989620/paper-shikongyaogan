# -*- coding: utf-8 -*-
"""Unit tests for TCCR-STF core (run: python -m pytest tests/ -q or python tests/run_all.py)."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from tccr.core import (TCCRConfig, e_change, gaussian_blur, read_lgc_int_bsq,
                       route, temporal_mismatch, temporal_projection)


def test_projection_lambda0_is_identity():
    cfg = TCCRConfig()
    x1 = torch.randn(1, 6, 32, 32)
    M = torch.randn_like(x1)
    out = temporal_projection(x1, M, lam=0.0)
    assert torch.equal(out, x1), 'lambda=0 must be bit-exact identity'


def test_projection_direction():
    x1 = torch.zeros(1, 6, 8, 8)
    M = torch.ones_like(x1)
    out = temporal_projection(x1, M, lam=0.5)
    assert torch.allclose(out, torch.full_like(x1, -0.5))


def test_router_correct_branch():
    cfg = TCCRConfig(tau_echange=0.5, direction='high')
    x1 = torch.zeros(1, 6, 8, 8)
    M = torch.ones_like(x1)
    x2 = torch.ones_like(x1)
    out, action, nfe = route(x1, x2, M, e_change=0.9, lam=0.5, cfg=cfg)
    assert action == 'CORRECT' and nfe == 1
    assert torch.allclose(out, torch.full_like(x1, -0.5))


def test_router_refine_branch():
    cfg = TCCRConfig(tau_echange=0.5, direction='high')
    x1 = torch.zeros(1, 6, 8, 8)
    x2 = torch.ones_like(x1)
    out, action, nfe = route(x1, x2, M=torch.zeros_like(x1), e_change=0.1,
                             lam=0.5, cfg=cfg)
    assert action == 'REFINE' and nfe == 2
    assert torch.equal(out, x2)


def test_temporal_mismatch_shape_and_zero():
    cfg = TCCRConfig(al=(1.0,) * 6, bl=(0.0,) * 6)
    x = torch.randn(1, 6, 16, 16)
    # identical inputs -> zero mismatch (blur of zero is zero)
    M = temporal_mismatch(x, x, x, x, cfg)
    assert M.shape == x.shape
    assert float(M.abs().max()) < 1e-6


def test_gaussian_blur_preserves_constant():
    x = torch.full((1, 3, 16, 16), 0.7)
    out = gaussian_blur(x, sigma=2.0)
    assert torch.allclose(out, x, atol=1e-5)


def test_bsq_loader_shape():
    """Requires a real LGC .int file passed via env var; skipped otherwise."""
    import os
    p = os.environ.get('TCCR_TEST_LGC_INT')
    if not p or not Path(p).exists():
        print('skip test_bsq_loader_shape (set TCCR_TEST_LGC_INT)')
        return
    a = read_lgc_int_bsq(p)
    assert a.shape == (2720, 3200, 6) and a.dtype == np.uint16
    # BSQ layout: per-band means must DIFFER (BIL mis-read makes them identical)
    means = [float(a[:, :, k].mean()) for k in range(6)]
    assert len(set(means)) > 1, 'per-band means identical -> wrong (BIL) layout!'




def test_selfcal_perfect_synthetic():
    """ΔF = 2·ΔC synthetic: OLS must recover a≈2, b≈0."""
    import sys as _s
    _s.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    from tccr.self_calibration import fit_anchor_calib
    blur = lambda x: x  # identity LP for synthetic test
    torch.manual_seed(0)
    dC = torch.randn(1, 6, 64, 64) * 0.1 + 0.3   # coarse change, nonzero mean
    dF = 2.0 * dC                                 # perfect a=2, b=0
    a, b, r2 = fit_anchor_calib(blur, torch.zeros_like(dC), torch.zeros_like(dC),
                                dF, dC, method='ols')
    assert abs(a[0] - 2.0) < 1e-4 and abs(b[0]) < 1e-6 and r2[0] > 0.999


def test_selfcal_low_change_fallback():
    """ΔC≈0 must fall back gracefully (a≈1 identity), no NaN."""
    import sys as _s
    _s.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    from tccr.self_calibration import fit_anchor_calib
    blur = lambda x: x
    torch.manual_seed(1)
    dC = torch.randn(1, 6, 64, 64) * 1e-9
    dF = torch.randn(1, 6, 64, 64)
    a, b, r2 = fit_anchor_calib(blur, torch.zeros_like(dC), torch.zeros_like(dC),
                                dF, dC, method='ols')
    assert all(np.isfinite(v) for v in a + b)


def test_no_target_gt_in_selfcal():
    """self_calibration module must not accept any fine_target argument."""
    import inspect, sys as _s
    _s.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    import tccr.self_calibration as sc
    for name in ['fit_anchor_calib', 'anchor_target_estimate', 'selfcal_projection']:
        sig = inspect.signature(getattr(sc, name))
        for pname in sig.parameters:
            assert 'target' not in pname.lower() or 'estimate' in name, \
                f'{name} must not take a fine_target parameter'


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f'PASS {fn.__name__}')
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f'FAIL {fn.__name__}: {e}')
    print(f'{len(fns) - failed}/{len(fns)} passed')
    sys.exit(1 if failed else 0)
