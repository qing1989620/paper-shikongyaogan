# EXPERIMENTS — TCCR-STF experiment log summary

## Lineage

| Stage | Question | Outcome |
|---|---|---|
| 1 | Do frequency-domain modules help? | No (D̂/β/M_conf removed) |
| 2 | Why do few steps win? Adaptive routing? | One-call saturation; E_change routing framework |
| 3 | What is the routing mechanism? Domain transfer? | Direction is semantics-dependent; **Temporal Projection** is the accuracy core; zero-shot calibration does not transfer |
| Final (this release) | Lock backbone, formal training, in-domain CIA, release | This document |

## Paper title candidates

1. *Temporal-Consistency Correct-or-Refine Sampling for Efficient Diffusion-Based
   Remote Sensing Spatiotemporal Fusion*
2. *Correct or Refine? Coarse-Temporal-Observation-Guided Adaptive Diffusion
   Sampling for Spatiotemporal Fusion*
3. *Test-Time Temporal Correction for Few-Step Diffusion Spatiotemporal Fusion*

## Contribution drafts (max 3)

1. **Observation**: the coarse image on the prediction date, though spatially
   coarse, records the *true temporal change* between the reference and
   prediction dates; the low-frequency part of a diffusion prediction's temporal
   change can therefore be verified **without ground truth**.
2. **Method**: TCCR-STF — a training-free test-time procedure that (a) corrects
   the low-frequency temporal mismatch directly (Temporal Projection, 0 extra
   NFE) and (b) routes each patch between correction and a second diffusion
   refinement step (Correct-or-Refine), using a no-GT consistency certificate.
3. **Evidence**: on LGC (correct BSQ semantics), TCCR improves val RMSE by 8.6%
   over the strongest fixed-NFE baseline at 1.25 average NFE (vs 10 official),
   with paired Wilcoxon p≈0; a calibration study quantifies that the 12 scalar
   coefficients are domain-specific (CIA zero-shot −0.8%, in-domain oracle +2.3%).

## Known negative results (kept for the record)

- Frequency-domain modules (D̂/β/M_conf): no gain (Stage 1).
- Soft residual blending, frequency-selective gating, dynamic second timestep:
  oracle upper bounds <0.3% — removed.
- Reference-leakage hypothesis for second-step harm: rejected (ρ sign opposite).
- Cross-domain zero-shot calibration: fails (CIA −0.8%); in-domain oracle +2.3%.

## Publication recommendation

**A. Ready for ICASSP-style submission** (LGC in-domain, dual-scene, significant,
efficient) — with the cross-domain calibration limitation stated explicitly.
CVPR: needs a second backbone + more datasets first.
