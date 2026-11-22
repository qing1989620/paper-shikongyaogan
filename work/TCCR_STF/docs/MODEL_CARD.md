# MODEL_CARD — TCCR-STF release checkpoint

## Backbone

| Field | Value |
|---|---|
| Class | `BiGaussianDiffusion` (wrapper) over `BiPredNoiseNet` |
| Source | STFDiff official repo, `src/model/stfdiff/model6/` |
| Architecture | Dual-stream U-Net noise predictor, model6 config |
| Diffusion T | 100 |
| Objective | `pred_x0` (data prediction) |
| Official sampler | DDIM, 10 steps uniform, η=1.0 |
| TCCR sampler | DDIM η=0, 1–2 steps (t=99 → optional t=49) |
| Condition tensors | 5 (see docs/PROTOCOL.md) |
| Training loss | L1 (+self-conditioning per official config) |
| Input size | 256×256×6 per tensor |

## Release checkpoint (to be finalized after formal training converges)

| Field | Value |
|---|---|
| File | `checkpoints/tccr_stfdiff_lgc_bsq_best.pth` |
| Training data | LGC train triples, **correct BSQ semantics** (see docs/DATA_PREPARATION.md) |
| Protocol | dual-anchor 6-input (official syy_setting-9) |
| Seed | 42 (formal run); seed 123 secondary |
| Selection | best val RMSE (48-patch fixed subset), early stopping patience=20 |
| Val RMSE | see FINAL_NUMBERS_LOCKED.csv (authoritative) |
| SHA256 | see SHA256SUMS.txt |

## Calibration files

| File | Domain | Contents |
|---|---|---|
| `calibration/lgc.json` | LGC | a/b (D_obs affine), al/bl (temporal affine), λ, τ, direction |
| `calibration/cia.json` | CIA | in-domain re-fit (only if in-domain experiment succeeds) |

Calibration = 12 affine scalars + λ + τ + direction. No neural parameters.
Backbone is frozen; calibration takes minutes on one GPU.
