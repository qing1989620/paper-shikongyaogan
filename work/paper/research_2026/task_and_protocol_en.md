# Working manuscript sections — task and experimental design

Status: research draft, 19 November 2026. The method and result-bearing abstract will be written after candidate selection and controlled replication. This text does not assert superiority over external methods.

## Problem formulation

We study dual-anchor spatiotemporal fusion of multispectral satellite images. Let F_0 and F_3 denote fine-resolution reference images acquired before and after a target date, and let C_0, C_t, and C_3 denote the corresponding coarse-sensor observations resampled to the common evaluation grid. Given these five observations and their acquisition times, the model estimates the unavailable fine-resolution image F_t. The target image is used for supervised training and evaluation, but is excluded from inference conditioning, initialization, calibration, and reliability estimation.

Two sources of variation are relevant to this problem. Differences between a same-date coarse/fine pair can reflect sensor response, spatial degradation, and registration mismatch. Differences across dates also contain genuine land-surface change. Our research hypotheses examine whether explicitly modeling these distinct roles improves fusion accuracy and robustness. The hypotheses are tested against simpler parameterizations and matched-capacity controls; their usefulness is not assumed from architectural complexity alone.

## Relationship to recent work

Recent work provides several useful design principles while also narrowing the novelty space. Progressive spectral restoration separates global appearance, coarse structure, and fine detail [J01]. Physics-guided fusion and explicit mixture models motivate constrained observation mappings [J05, J06]. Boundary-aware representations and temporally guided structural constraints address spatial transitions [J02, J03]. Uncertainty-aware transport and correlation-aware restoration suggest alternatives to isotropic generative geometry [C01, C02].

Direct spatiotemporal-fusion predecessors already include temporal feature alignment, optical-flow guidance, frequency-separated fusion, and prior-guided flow matching. Consequently, the presence of alignment, a wavelet transform, a gate, or a bridge is not by itself our contribution. For the selected model, the final manuscript must identify a specific unresolved assumption, define the proposed estimator, and demonstrate its distinct effect through removal, replacement, and intervention experiments.

Citation identifiers refer to the verified primary-source register in `docs/LITERATURE_2026_VERIFIED.md`; bibliography details and full-text checks must be completed for the final selected mechanism.

## Experimental design

We separate development screening from confirmatory evaluation. In development screening, independently implemented candidates share the same initialization checkpoint, training samples, optimizer settings, update budget, and underlying random-number realizations. Each candidate is evaluated with three fixed inference-noise realizations whose seeds are derived from stable sample identifiers. Metrics are computed separately for each generated prediction and then aggregated; averaging predictions before scoring is a different estimator and is not used to conceal sampling variability.

The shared starting model is retained as an additional control. A gain over a continued-training baseline that has degraded relative to its initialization is insufficient: a promoted candidate must also improve on the stronger starting control. The short development schedule is used to prioritize compute, not to establish final model rankings or conclusively reject mechanisms requiring substantial retraining.

All fourteen directions undergo paired training from scratch for 40 epochs with three training seeds; the baseline receives the same budget. Screening determines execution order rather than replacing full training. Candidates are selected only from development validation after all runs finish, followed by mechanism-specific ablations, matched-capacity and compute controls, and external evaluation. In-domain CIA results are reported separately from transfer of an LGC-trained model to CIA. Training-date overlap and additional pretraining data are recorded for all external checkpoints. Comparisons using author weights with different training recipes are reported separately; the formal main benchmark also requires a matched training protocol.

## Metrics and uncertainty

All predictions and references are explicitly mapped to the reflectance domain before evaluation. RMSE is the primary selection measure; MAE, PSNR, SSIM, spectral angle, ERGAS, and bandwise correlation provide complementary evidence. The SSIM window, spectral-angle units, ERGAS scale factor, valid-pixel mask, and scene reconstruction policy are fixed and disclosed. Patch-average RMSE, pooled pixel RMSE, and date-average scene RMSE are distinct quantities and are labeled accordingly.

Uncertainty estimates respect the dependence structure of the data. Repeated diffusion noise realizations are not independent acquisition events, and neighboring patches do not provide independent temporal replications. We therefore retain per-date results and use date/spatial-block analyses where supported by the available acquisitions. A validation split with a single target date supports scene-specific development observations, not a general claim of cross-event significance.

## Planned evidence for the final paper

The result-bearing paper will require a complete common-protocol comparison with competitive external models, component and intervention ablations, multiple training seeds, qualitative comparisons under the same display transform, and measurements of latency, peak memory, and additional computation. Model-only inference cost and end-to-end scene reconstruction cost will be distinguished. Negative results and material trade-offs will remain visible when interpreting the selected mechanism.
