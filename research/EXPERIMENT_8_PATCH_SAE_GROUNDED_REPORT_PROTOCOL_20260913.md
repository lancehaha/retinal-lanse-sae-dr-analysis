# Experiment 8: Patch-SAE Transformer with Feature-Grounded DR Reports

## Scope and status

This is a gated local-development pilot. It uses 2,000 deterministically selected EyePACS training patients only. It does not access the locked EyePACS test cohort, IDRiD, or any external test set. Results are feasibility evidence only and cannot support a clinical-utility or lesion-identity claim.

All large artifacts are written under `/Volumes/Elements/Retina-LanSE/experiments/experiment_8_patch_sae_grounded_report_pilot_20260913/`.

## Cohorts and leakage controls

Patient IDs in the EyePACS training manifest are hash-ranked before labels are loaded. The first 2,000 patients are partitioned into 1,600 SAE/model-training, 200 development, and 200 pilot-holdout patients. Patient overlap is a hard error. SAE fitting and feature-card hypothesis prompts are label-blind. The prediction specification is written before pilot-holdout targets and predictions are materialized.

The runner accepts only the training global-embedding run and rejects non-external output locations. There are no command-line inputs for locked test or IDRiD.

## Local representation

Frozen RETFound-CFP is evaluated at 224 x 224. Its final 196 patch tokens are saved in fixed row-major 14 x 14 order as float16 `[image, 196, 1024]` shards. For every extraction batch, the runner applies the existing final-mean-patch pooling and requires numerical agreement with the pre-existing global `final_mean_patch` embedding. This guards against extracting a different model layer or token ordering.

One patch SAE is trained on patch vectors from only the 1,600 SAE-training patients:

- input dimension 1,024; dictionary 16,384; Top-K 16;
- AdamW, learning rate 3e-4, weight decay 1e-5, batch size 1,024, 20 epochs;
- train-only normalization, decoder-column normalization, deterministic dead-feature resampling, seed 4106.

Gates are reconstruction explained variance at least 0.85, dead-feature fraction at most 0.25, Top-K at most 16, and exact checkpoint reload equivalence.

## Transformer arms

Each patch pools its 16 active SAE entries with learned attention. An entry combines a frozen 128-dimensional PCA projection of its normalized SAE decoder direction, an activation-strength projection, and a L2-regularized feature-specific residual. The final sequence is `[CLS] + global dense token + 196 local patch tokens`.

The Transformer has dimension 128, four pre-layer-normalized layers, eight heads, 512-wide feed-forward blocks, GELU, and dropout 0.1. 2D row/column position embeddings are present only in positional arms. RETFound and SAE parameters are frozen.

Predeclared binary and five-grade arms are:

1. Dense RETFound logistic baseline.
2. Dense RETFound MLP baseline.
3. Dense plus local activation-value Transformer without decoder geometry.
4. Dense plus decoder-geometry Transformer without location.
5. Full dense plus decoder geometry plus x/y Transformer.
6. Full Transformer with decoder geometry randomly permuted across SAE feature IDs.

Binary heads use weighted BCE and five-grade heads use class-balanced cross-entropy. Three seeds (4106-4108) select development epochs with early stopping, the median epoch is frozen per arm, and models are retrained on the 1,800 non-holdout patients. The holdout is then evaluated once. If a training/development cohort lacks a grade, five-grade outcomes are explicitly unavailable.

The primary outcome is referable-DR (grade >=2) AUROC. Secondary outcomes include AUPRC, Brier score, calibration, threshold metrics, and five-grade macro metrics. Paired patient-cluster bootstrap intervals compare full versus dense MLP, no-position, and randomized geometry.

## Feature cards and reports

The 64 features with highest label-free activation mass and at least 100 training-patch activations are eligible. A card receives 32 high-activation discovery patches, then new patient-disjoint groups of 32 high and 32 low patches for validation. Prompts omit labels, grades, patient IDs, and activation values. Validation is delivered as two 16-high/16-low requests to avoid incomplete local-VLM JSON; each malformed reply receives at most one JSON-format repair call with no new evidence. A card passes only when high presence is at least 0.60 and high-minus-low presence is at least 0.30.

Every pilot-holdout image receives a deterministic structured report. Only passed cards may be considered. Each candidate patch-feature entry is independently deleted from the frozen full Transformer. A feature is cited only if deleting it lowers the ensemble referable-DR probability. Its causal score change and its normalized 14 x 14 box are retained. Matched random active-entry deletions form the faithfulness control. A report passes the aggregate faithfulness check only when cited entries have a larger mean deletion effect than random entries.

Qwen receives the frozen structured report alone, never the target image or outcome. Its narrative is separate from the numerical score. Each statement must map to a report feature ID. Failed or malformed responses are retained, not silently dropped.

## Expansion gate

Move to publication-scale work only if extraction and SAE health checks pass, at least 48/64 cards pass contrastive validation, cited deletions beat random deletions, and the full Transformer is not more than 0.03 AUROC below the dense MLP on pilot holdout. Expansion requires five SAE seeds, five model seeds, locked internal testing, clean IDRiD evaluation, and specialist review.
