# Experiment 5B results: contrastive SAE feature-hypothesis validation

Date: 2026-09-10 to 2026-09-11  
Primary artifact: `/Volumes/Elements/Retina-LanSE/experiments/experiment_5b_contrastive_feature_validation_20260910/full_run_final_v2/`

## Objective

Test whether each existing high-activation SAE feature hypothesis describes a visual
pattern that is more consistently present in high-activation images than in low-activation
controls. This is a semantic consistency experiment, not a DR classifier or lesion-
identity validation.

## Frozen design

- SAE checkpoint: K=1024, dictionary size 16,384, seed 4106; RETFound was not retrained.
- Existing 32 strongest images were discovery-only.
- Initial validation used 32 new high-activation and 32 new low-positive-activation images.
- Sampling was patient-disjoint where feasible and exact paths/patient IDs were retained.
- Qwen2.5-VL 3B saw images and a text hypothesis only; labels, grades, patient IDs, and
  activation values were hidden.
- Each 32+32 round was sent as two 16+16 requests to prevent response truncation.
- A feature passed when at least 24 judgments existed in each group, high pattern-present
  rate was at least 0.60, and the high-minus-low rate was at least 0.30.
- A failed, valid hypothesis received exactly one revision using the original discovery
  highs plus a new low-activation sample, followed by fresh high/low re-validation.

## Results

| Outcome | Features |
|---|---:|
| Prior completed hypotheses evaluated | 143 |
| Cards without a completed prior hypothesis (unavailable) | 113 |
| Initial validation parsed successfully | 141 |
| Initial pass | 84 |
| Initial valid failure and one revision attempted | 57 |
| Revised validation passed | 56 |
| Final failed or unavailable | 3 |
| Maximum revisions per feature | 1 |

Among the 141 initially parsed validations, mean high pattern-present rate was 0.770,
mean low rate was 0.202, and mean contrast was 0.567. Among the 57 revised validations,
mean high rate was 0.982, mean low rate was 0.015, and mean contrast was 0.967. These
descriptive rates are not independent clinical evidence: the same local VLM generated and
validated the hypotheses, and revisions were conditioned on the initial failure.

## Interpretation and limitations

The procedure demonstrates that many hypotheses can be made contrastive against low-
activation controls under a fixed, bounded protocol. It does not establish that the
features represent diabetic-retinopathy lesions, are DR-specific, or generalize clinically.
The high revised-pass rate is especially vulnerable to same-model confirmation and must
not be presented as independent semantic validation. Independent ophthalmologist review
and lesion-grounded analyses remain required.

## Reproducibility evidence

- Machine-readable per-feature records: `contrastive_feature_validation.jsonl`.
- Aggregate summary: `contrastive_feature_validation_summary.json`.
- Checkpoint SHA-256: `03a7e71e63eb82d26239b0b625ddb54b9b2068fc7794df790a895e1a74bb2cc7`.
- Train-only normalization SHA-256: `6349d0599e31c31afda32b19d2951d4c194eb4b4818c3efe5fbf308a0c87ca06`.
- Labels were absent from all hypothesis and validation prompts.
