# Experiment 6: feature-card DR prediction

**Protocol date:** 2026-09-12  
**Status:** implemented; numerical and LLM execution are separate resumable phases.

## Question and fixed inputs

This experiment asks whether the 256 fixed K=1024, seed-4106 SAE feature cards
can predict diabetic-retinopathy grade and whether their natural-language
hypotheses add information beyond numerical activation/association evidence.
RETFound and the SAE remain frozen. The existing patient-level EyePACS cohorts
and clean 102-image IDRiD test are reused without resplitting.

The primary endpoint is referable DR (grade >=2) AUROC. Five-grade quadratic
weighted kappa is co-primary. Feature semantics remain same-model visual
hypotheses and are not lesion labels.

## Train-only association model

For each card, `log1p` activation is summarized per patient using the maximum
activation and maximum grade. Spearman association, a 2,000-resample
patient-cluster interval, and Benjamini-Hochberg q-value determine whether the
trend supports increasing grade, decreasing grade, or is mixed/uncertain.
Image-level activation rate and positive-activation strength are also reported
for every grade.

Positive activations are divided into four training-derived quantile bins; zero
is a fifth bin. Dirichlet smoothing with alpha=1 estimates
`P(grade | feature, activation_bin)`. For each target, the top 16 active cards
contribute capped log-likelihood ratios to a direct five-grade evidence score.
Only its temperature is fitted on a patient-disjoint 10% development subset.

## Arms and controls

1. Dense RETFound class-balanced logistic baseline.
2. Direct top-16 card-evidence likelihood rule.
3. Learned class-balanced probe on the identity-preserving 256-card activation vector.
4. Qwen with numerical evidence and the real card descriptions.
5. Qwen with the same evidence but no descriptions.
6. Qwen with descriptions permuted using seed 9206.

The primary set contains all 256 cards. A conservative sensitivity analysis
contains the 251 cards passing the bounded two-round protocol. The five
ten-round rescue cards are separately identified. Numeric models retain all
feature identities. Exact-description pairs 3322/7388 and 15184/12428 are
grouped in prompts so repeated language is not counted as independent evidence.

The LLM receives no target image, patient identity, or target label. One
deterministic JSON-repair call is allowed without new evidence; remaining
failures are retained, and full-cohort metrics replace their probabilities with
the EyePACS training prior. For tractable local execution, up to eight
independent cases share one text-only request; case IDs and outputs remain
separate. Confirmatory LLM evaluation uses the locked internal test and clean
IDRiD. Validation remains available for numerical diagnostics but is not spent
on the fixed LLM prompt.

## Statistics, success, and outputs

Report AUROC, AUPRC, Brier score, ECE, threshold metrics, macro-AUROC,
macro-AUPRC, QWK, macro-F1, ordinal MAE, completion rate, and 2,000 paired
patient-cluster bootstrap differences. IDRiD uses image-examination proxy
resampling. Card sufficiency requires direct evidence to be non-inferior to
dense within 0.03 AUROC and 0.05 QWK. Semantic benefit requires at least 0.02
AUROC gain over both description-free and shuffled-description controls with
Holm-adjusted confidence intervals excluding zero.

Implementation: `scripts/run_feature_card_dr_prediction.py`  
Heavy output:
`/Volumes/Elements/Retina-LanSE/experiments/experiment_6_feature_card_prediction_20260912/`

Every output records input hashes, card tiers, association tables, prompts,
responses, parsing failures, predictions, metrics, and the frozen specification.
