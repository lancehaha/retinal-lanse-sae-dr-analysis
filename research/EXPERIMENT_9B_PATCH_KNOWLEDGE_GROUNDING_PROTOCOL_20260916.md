# Experiment 9B: Patch-SAE/Transformer Knowledge Grounding

Status: **frozen before scientific execution**  
Freeze date: 2026-09-16 (Asia/Shanghai)  
Scope: exploratory knowledge development and validation; IDRiD is not an untouched external validation set.

## Question and claim boundary

Can lesion-mask-grounded knowledge attached to frozen Patch-SAE feature identities improve referable-DR prediction and produce faithful, location-specific reports relative to the original Patch-SAE representation and a shuffled-knowledge control?

RETFound and the Experiment 8 K=64, dictionary-size 16,384 Patch-SAE remain frozen. The experiment reuses Experiment 8's label-free selected 64 feature IDs, 2,000-patient EyePACS pilot cohorts, and seeds 4106-4108. It does not alter or resume Experiment 7C and does not edit Experiment 9A artifacts.

## Frozen inputs

- IDRiD segmentation images: official 54-image training split for knowledge discovery and official 27-image testing split for knowledge validation.
- Lesion masks: microaneurysms, haemorrhages, hard exudates, and soft exudates. Missing mask files mean an all-zero mask for that lesion class.
- Optic-disc masks are measured only as an anatomy/confounding signal and can never ground a DR-lesion claim.
- Frozen RETFound `final_patch_prepool` tokens, shape `[image,196,1024]`, extracted with `retinal-field-crop-v1+retfound-cfp-channel-zscore-v1`.
- Frozen Experiment 8 K=64 Patch-SAE checkpoint and train-only normalization.
- Frozen 64 feature IDs, in the exact order recorded by Experiment 8 feature-card selection.
- Frozen Experiment 8 patient cohorts: 1,600 SAE-train patients, 200 development patients, and 200 pilot-holdout patients.

Every input path, file hash, seed, environment version, and frozen threshold is written to `specification_freeze.json` before analysis.

## Exact image and mask geometry

The image retinal-field crop is the bounding box of original RGB pixels whose maximum channel is greater than 8, exactly matching the existing preprocessing implementation. Images are then resized to 224x224 with bilinear interpolation and normalized per image/per channel as before.

Masks are never independently cropped or interpolated. Original-resolution positive mask pixels are projected through the image-derived crop and 224x224 resize into the row-major 14x14 token grid. A patch's binary overlap is true if any projected source mask pixel is positive. Fractional occupancy is the number of positive source pixels mapped to the patch divided by the number of source pixels mapped to that patch. This direct pixel projection preserves a one-pixel microaneurysm. Invalid retinal background is derived from the original image threshold and projected by the same mapping.

Extraction is accepted only if mean pooling each `[196,1024]` token array through RETFound `fc_norm` reproduces the corresponding global embedding within maximum absolute error `2e-5`.

## Grounding analysis

For each of 64 features and each lesion class, patch activation is the feature's ReLU SAE activation (zero when it is not in that patch's Top-64 entries). Lesion-positive patches are compared with same-image lesion-negative patches matched by quadrant and radial-distance bin. Radial bins are fixed from 14x14 patch-center distance to the grid center using edges `[0, 0.25, 0.50, 0.75, 1.01]` after division by the maximum grid-center distance. Negative candidates must:

- contain no pixel from any of the four lesion masks;
- contain no optic-disc pixel;
- have at least 50% valid retinal pixels;
- match the positive patch's quadrant and radial bin.

Within each image and stratum, all positives are retained and up to the same number of negatives are selected by SHA-256 order with seed 4106. Images/strata without both classes are excluded. Metrics pool the matched patch rows, while uncertainty resamples images as clusters.

Training statistics use 2,000 image-clustered bootstrap resamples. The one-sided training p-value is `(1 + count(AUROC_bootstrap <= 0.5)) / (B + 1)`. Benjamini-Hochberg correction is applied jointly across all 256 feature-lesion tests. Validation uses 2,000 image-clustered bootstrap resamples and reports AUROC, AUPRC, prevalence, AUPRC lift, and percentile intervals.

A feature-lesion pair is **supported** only when all frozen conditions hold:

1. training BH q-value `< 0.05`;
2. validation AUROC `>= 0.65`;
3. validation AUPRC lift `>= 2.0`;
4. training and validation AUROC are both greater than 0.5.

It is **strong** only when supported and the validation AUROC bootstrap lower bound is `> 0.5` and the validation AUPRC-lift lower bound is `> 1.0`. Thresholds will not be relaxed.

## Frozen knowledge vector

Each feature receives a six-dimensional vector:

1. microaneurysm association;
2. haemorrhage association;
3. hard-exudate association;
4. soft-exudate association;
5. confidence;
6. optic-disc/confounder association.

For a supported lesion pair, its association is `clip(2 * (validation_AUROC - 0.5), 0, 1)`; unsupported lesion entries are zero. Confidence is the maximum, over supported lesion pairs, of `clip(min((AUROC-0.5)/0.5, (lift-1)/1), 0, 1) * (1-q)`. The optic-disc score is `clip(2 * (optic_disc_validation_AUROC - 0.5), -1, 1)`. A feature is grounded only if at least one of its four lesion pairs is supported; otherwise its entire six-dimensional vector, including the optic-disc score, is zero. The vector file is frozen and hashed before prediction training.

## Prediction comparison

The endpoint is EyePACS referable DR (`grade >= 2`). Models use the original Experiment 8 patient split, seeds 4106-4108, weighted BCE, AdamW (`lr=1e-4`, `weight_decay=1e-4`), batch size 16, gradient clipping 1.0, maximum 50 epochs, and patience 8. Median selected epoch across seeds is frozen per arm; models are reinitialized and fitted on SAE-train plus development patients before the pilot holdout is accessed.

Arms:

1. `dense_mlp`: frozen global RETFound embedding with the Experiment 8 MLP reference.
2. `original_patch`: original full Patch-SAE Transformer with frozen decoder geometry, learned zero-initialized identity residual, global token, patch positions, and activation values.
3. `knowledge_patch`: the same Transformer backbone, global token, patch positions, activation values, and training procedure; feature identity enters only through the frozen six-dimensional knowledge vector projected to model dimension.
4. `shuffled_knowledge_patch`: identical to the knowledge arm, but feature-to-vector mapping is shuffled once with seed 9206.

The knowledge mapping is the only difference between arms 3 and 4. RETFound, the SAE, knowledge vectors, and the shuffled mapping remain frozen. Prediction success requires the knowledge arm to improve holdout AUROC by at least 0.01 over both original and shuffled arms, with each paired patient-cluster bootstrap 95% interval entirely above zero.

## Grounded reports and faithfulness

Reports are generated only from the frozen numerical prediction and frozen grounded-feature table. A cited entry must be grounded, active at the stated row-major 14x14 location, and have a positive independent deletion effect on referable-DR probability. Target labels, raw images, and uncited feature hypotheses are excluded from every report prompt. Prompts and raw responses are retained. If the local Qwen service/model is unavailable, the deterministic structured report remains the auditable primary report and Qwen rendering is marked unavailable rather than fabricated.

For every cited active entry, its activation is independently set to zero and the probability change is recorded. Cited effects are compared with same-image random active entries matched by activation quartile and patch radial bin, sampled with seed 9206. Uncertainty uses patient-clustered bootstrap resampling.

Interpretability success requires all of:

- at least 8 of 64 grounded features;
- support spans at least 2 lesion classes;
- at least 95% of evaluated reports satisfy the report schema and citation rules;
- mean cited deletion effect minus matched-random effect has a 95% patient-bootstrap interval entirely above zero.

Prediction and interpretability outcomes are reported independently. If grounding fails, the experiment completes as a negative result; no lesion label or report citation is invented.

## Execution and fail-closed rules

Stages are `freeze`, `smoke`, `extract`, `ground`, `predict`, `report`, and `finalize`. A five-image smoke test must pass before scientific extraction. RETFound extraction, Transformer training, and optional Qwen rendering acquire the shared external-drive GPU lock. CPU mask projection and statistical analysis may proceed without it.

Tests cover image/mask matching, exact crop projection, one-pixel microaneurysm preservation, global-vs-patch separation, cohort identity/disjointness, grouped bootstrap behavior, BH correction, fixed shuffling, schemas, checkpoint reload equivalence, and absence of target labels from prompts.

The run fails closed on any source hash mismatch, wrong token/decoder orientation, preprocessing drift, missing selected feature, non-finite value, cohort mismatch, patient overlap, incomplete prediction array, checkpoint reload mismatch, output outside the Experiment 9B directory, or target-label leakage into report prompts.

## Required artifacts

- specification and environment freeze;
- five-image smoke result;
- 81-image token/mask extraction evidence and arrays;
- frozen feature-grounding statistics and knowledge vectors;
- training histories, checkpoints, per-seed and ensemble predictions;
- patient-bootstrap comparisons and success decisions;
- deterministic reports, optional Qwen prompts/responses, deletion-faithfulness arrays;
- machine-readable final result and concise results document.
