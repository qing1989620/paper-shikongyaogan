# PROTOCOL — TCCR-STF input/output contract

## Canonical terminology (use these names everywhere)

| Symbol | Code name | Sensor / resolution | Date role | Enters model forward? | Enters TCCR? |
|---|---|---|---|---|---|
| `fine_ref_prev` | `fine_img_01` | Landsat 30m | reference date d1 (earlier) | ✅ | ✅ (as F0 in M) |
| `fine_target` | `fine_img_02` | Landsat 30m | **prediction date** | ❌ (training: noising+loss only; inference: shape only) | ❌ (evaluation only) |
| `fine_ref_next` | `fine_img_03` | Landsat 30m | reference date d3 (later) | ✅ | ❌ |
| `coarse_ref_prev` | `coarse_img_01` | MODIS 25m | d1 | ✅ | ✅ (as C0 in M) |
| `coarse_pred` | `coarse_img_02` | MODIS 25m | **prediction date** | ✅ | ✅ (as C1 in M) |
| `coarse_ref_next` | `coarse_img_03` | MODIS 25m | d3 | ✅ | ❌ |

## The one-line protocol statement

**STFDiff official dual-anchor protocol (syy_setting-9), 6 condition inputs +
1 GT:**
`{fine_ref_prev, fine_ref_next, coarse_ref_prev, coarse_pred, coarse_ref_next}`
→ predict `fine_target`.

`fine_target` is the ground truth: it enters training only through the forward
noising (`q_sample`) and the loss target, and at inference the process starts
from pure Gaussian noise. Verified at tensor level — see
`PROTOCOL_AUDIT_FINAL.md` in the experiment archive.

## Relation to the minimal "F0, C0, C1 → F1" description

The minimal description corresponds to
`fine_ref_prev=F0, coarse_ref_prev=C0, coarse_pred=C1, fine_target=F1`.
The official protocol **additionally** uses `fine_ref_next` / `coarse_ref_next`.
This is an information-protocol difference (not just naming): results under the
dual-anchor protocol must not be presented as minimal-protocol results.

TCCR itself (projection + routing) only consumes
`x1, fine_ref_prev, coarse_ref_prev, coarse_pred` — none of which is GT — so the
method is expected to transfer to the minimal protocol, pending experiments.

## Diffusion sampling contract

- Sampler: DDIM, η=0 (deterministic), `pred_x0` parameterization, T=100.
- Official NFE: 10 (uniform linspace). TCCR uses 1 or 2 NFE:
  - step 1: t=99 → x1
  - optional step 2: t=49 → x2 (used only when the router says REFINE)
- Clip x-start: on (official setting).

## TCCR formulas

```
M        = LP(x1 − fine_ref_prev) − (al ⊙ LP(coarse_pred − coarse_ref_prev) + bl)
E_change = mean(|M|)
CORRECT  (E_change > τ, direction='high'):  F_hat = x1 − λ·M        (NFE=1)
REFINE   (E_change ≤ τ):               F_hat = x2                    (NFE=2)
LP = separable Gaussian blur, σ=2.0, reflect padding
al, bl ∈ R^6 : per-band affine fitted on the calibration domain's TRAIN split
λ, τ         : frozen on the calibration domain's VAL split
```
