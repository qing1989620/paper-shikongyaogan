# TCCR-STF

**Temporal-Consistency Correct-or-Refine** sampling for efficient diffusion-based
remote-sensing spatiotemporal fusion (STF).

> TCCR-STF improves diffusion-based remote-sensing spatiotemporal fusion by using
> coarse temporal observations to correct low-frequency temporal mismatch and
> adaptively decide whether a second diffusion refinement step is needed.

Base model: **STFDiff** model6 / `BiGaussianDiffusion`
(Huang et al., Information Fusion 111 (2024) 102505, [repo](https://github.com/prowDIY/STF)).

## Results (LGC dataset, correct BSQ semantics, dual-anchor 6-input protocol)

| Method | val RMSE ↓ | avg NFE | Notes |
|---|---|---|---|
| Fixed-1 (1 diffusion step) | 0.02086 | 1 | strongest fixed baseline |
| Fixed-2 (2 steps, default sched) | 0.01982 | 2 | |
| Projection only (λ=0.5) | 0.01907 | 1 | accuracy core |
| **TCCR-STF (full)** | **0.01906** | **1.25** | accuracy + adaptive NFE |
| Oracle min(1,2) | 0.01981 | — | upper bound reference |

Statistical significance (paired Wilcoxon, 48 patches):
TCCR vs fixed-1 Δ=+0.0018, p≈0, 95%CI [+0.0014, +0.0022].

Cross-domain: CIA zero-shot with LGC-frozen calibration does **not** transfer
(−0.8%); the 12 scalar calibration coefficients are domain-specific and require a
lightweight re-fit on the target domain (see Limitations).

## Method in one picture

```
                ┌── conditions (never the GT) ─────────────────────────┐
noise x_T ──► 1st denoise (t=99) ──► x1
                                        │
        M = LP(x1 − fine_ref_prev) − (al·LP(coarse_pred − coarse_ref_prev) + bl)
        E_change = mean|M|
                                        │
              E_change > τ ──► CORRECT:  x = x1 − λ·M      (0 extra NFE)
              E_change ≤ τ ──► REFINE :  x = 2nd DDIM step  (+1 NFE)
```

- **Temporal Projection** = the CORRECT branch: a one-line low-frequency residual
  correction guided by the *coarse temporal change actually observed* on the
  prediction date. This is the accuracy core.
- **Correct-or-Refine routing** = a no-GT, no-parameter certificate that picks
  between the correction and a second diffusion step. This is the efficiency /
  adaptive-compute core.

## ⚠️ Data semantics — read before anything else

**LGC dataset `.int` files are band-sequential (BSQ):**
`(bands, rows=2720, cols=3200)`, uint16, little-endian, 6 bands.
Reading them as BIL produces silently corrupted (row-strided) images —
per-band means become identical, which is the quick self-check.

- Use `src/tccr/core.py::read_lgc_int_bsq`.
- CIA Landsat `.bil` files are genuine **BIL**; CIA MODIS `.int` are **BSQ**
  (verified by masked cross-sensor NCC). Use the dedicated readers.
- All reflectance DNs are clipped to [0, 10000]; normalization is `DN/10000*2−1`.

## Protocol (dual-anchor, 6 inputs — official STFDiff syy_setting-9)

```
t_prev(d1)          t_pred(d2)          t_next(d3)
fine_ref_prev       [fine_target=GT]    fine_ref_next        (Landsat 30m)
coarse_ref_prev     coarse_pred         coarse_ref_next      (MODIS 25m)
                        ▲ prediction-date only real observation
```

- `fine_target` is **never** an input: training uses it only for noising + loss;
  inference starts from pure Gaussian noise. (Verified in
  `docs/PROTOCOL.md`; TCCR inference is fully GT-free.)
- Full tensor-level audit: `docs/PROTOCOL.md` and
  `PROTOCOL_AUDIT_FINAL.md` in the experiment archive.

## Quick Start (Windows PowerShell)

```powershell
# 1. environment
python -m venv .venv; .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. prepare data (expects raw LGC .int + CIA archives; see docs/DATA_PREPARATION.md)
python tools\prepare_data.py --lgc_raw <path\to\LGC> --cia_raw <path\to\CIA> --out data

# 3. train the backbone (correct BSQ semantics, early stopping)
python tools\train.py --train_root data\lgc_bsq\train --val_root data\lgc_bsq\val `
    --seed 42 --out checkpoints\formal_seed42

# 4. fit calibration (12 scalars + lambda + tau on TRAIN/VAL only)
python tools\calibrate.py --ckpt checkpoints\formal_seed42\best_rmse.pth `
    --data_root data\lgc_bsq --out calibration\lgc.json

# 5-7. run fixed-1 / TCCR / evaluate
python tools\infer.py --ckpt checkpoints\formal_seed42\best_rmse.pth `
    --data_root data\lgc_bsq\val --method tccr --calibration calibration\lgc.json `
    --out results\lgc_val_tccr.csv
python tools\evaluate.py --results results\ --out results\final_summary.csv

# 8. unit tests
python tests\run_all.py
```

All commands also run under bash with forward slashes.

## Repository layout

```
TCCR_STF/
├── README.md            ← you are here
├── requirements.txt
├── configs/             ← experiment configs (JSON)
├── src/tccr/            ← core method (loader, projection, router)
├── tools/               ← train / calibrate / infer / evaluate / benchmark CLIs
├── scripts/             ← one-click batch scripts
├── tests/               ← unit tests (run_all.py)
├── calibration/         ← frozen per-domain calibration JSONs
├── docs/                ← PROTOCOL / DATA_PREPARATION / MODEL_CARD / EXPERIMENTS
├── results/             ← final_summary.csv (locked numbers)
├── checkpoints/         ← optional weights (see MODEL_CARD.md)
└── legacy_debug/        ← DO NOT USE FOR TRAINING (historical BIL mis-read)
```

## Limitations (honest)

1. **Cross-domain zero-shot calibration does not transfer** (CIA −0.8% with
   LGC-frozen scalars). The method *framework* transfers; the 12 scalar
   coefficients (6-band affine for the coarse→fine radiometric mapping) plus λ
   require a lightweight re-fit on ~dozens of target-domain reference pairs.
2. Evaluated on one diffusion STF backbone (STFDiff model6). Transfer to other
   diffusion STF architectures is expected (the method only needs
   "first prediction + coarse temporal observation") but not yet verified.
3. The dual-anchor 6-input protocol is the official STFDiff setting and carries
   more information than the minimal 3-input variant; results should not be
   compared across protocols.

## Citation

```bibtex
@article{huang2024stfdiff,
  title={STFDiff: Remote sensing image spatiotemporal fusion with diffusion models},
  author={Huang, He and He, Wei and Zhang, Hongyan and Xia, Yu and Zhang, Liangpei},
  journal={Information Fusion}, volume={111}, pages={102505}, year={2024}
}
```

TCCR-STF manuscript in preparation (title candidates in `docs/EXPERIMENTS.md`).
