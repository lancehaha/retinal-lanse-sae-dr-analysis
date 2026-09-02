# Experiment 1: frozen RETFound global SAE reconstruction

Version: 1.1. This protocol defines the active no-AutoResearch experiment and must be hashed before the EyePACS internal-test split is opened. Version 1.1 records the label-free data-hygiene amendment made after the near-duplicate gate failed and before validation extraction or model selection began.

## Scope and claim

This experiment asks whether a Top-K sparse autoencoder can reconstruct frozen RETFound-CFP whole-image embeddings more faithfully than matched label-free reconstruction baselines while maintaining a healthy sparse feature population. It is a representation experiment, not a lesion-localization, diagnostic-performance, clinical-safety, or external-generalization experiment. Those claims require the later local-tile and externally annotated experiments in `PUBLICATION_PROTOCOL.md`.

The RETFound encoder is frozen. No clinical label is read for SAE training, hyperparameter selection, baseline fitting, or internal-test evaluation. AutoResearch is not invoked.

All registered extraction and SAE runs use CPU FP32. A bounded 834-train/88-validation software check completed the identical 4x, K=8, one-epoch configuration in 0.210 seconds on CPU versus 1.457 seconds on MPS; both passed exact checkpoint-reload equivalence and their validation MSE differed by approximately 2.1e-8. CPU is therefore retained as the measured execution choice, and no cross-device equivalence claim is made beyond this software check.

## Cohort and split

- Source dataset: complete EyePACS/Kaggle DR image corpus, 88,702 images.
- Unit of isolation: numeric patient identifier shared by left/right eyes.
- Initial deterministic split: 71,134 train, 8,804 validation, 8,764 internal test.
- The first high-specificity audit confirmed four disjoint train-to-holdout near-duplicate pairs. Before validation extraction, clinical-label access, or model selection, apply the fixed connected-component policy `train > validation > internal_test`: retain patients in the highest-priority split and exclude every image of lower-priority patients; never reassign a patient. This excludes four holdout patients/eight images, leaves training unchanged, and yields the publication cohort of 88,694 images/44,347 patients: 71,134/8,800/8,760 images and 35,567/4,400/4,380 patients in train/validation/internal test. Freeze both the failed source audit and the remediation report as provenance.
- Train/validation/internal-test patient sets must be disjoint.
- Exact cross-split duplicate count must be zero before extraction.
- Because 64-bit dHash is low-specificity on visually similar fundus fields, run a second label-free near-duplicate gate using a 255-bit DCT perceptual hash and 13 disjoint locality-sensitive bands. Thirteen bands guarantee that every pair with Hamming distance at most 12 shares an exact band and enters candidate verification. Confirmation requires Hamming distance at most 8 plus normalized pixel correlation at least 0.995 (including horizontal-flip comparison). Require zero confirmed cross-split near duplicates before validation/model selection.
- Dataset manifest and every embedding shard are SHA-256 protected.

The internal test is a label-free representation test. It is separate from the later locked clinical external tests (IDRiD and FGADR), which remain unopened until candidate neurons, hypotheses, thresholds, and multiplicity rules are frozen.

## Representations

Evaluate two preregistered 1,024-dimensional frozen RETFound outputs independently:

1. final mean-patch pooling;
2. final CLS-token pooling.

The primary representation is mean-patch. CLS is the preregistered pooling ablation. Train-only per-dimension z-score normalization is fitted separately for each representation.

## SAE search and selection

For each representation, screen the full Cartesian grid:

- dictionary expansion: 4x, 8x, 16x;
- Top-K: 8, 16, 32;
- learning rate: 1e-4, 3e-4, 1e-3;
- dead-feature resampling: disabled, enabled.

This yields 54 single-seed screening runs per representation and 108 total. Retain three Pareto finalists independently for each representation using validation MSE, explained variance, cosine similarity, dead-feature fraction, and saturation. Rerun each finalist with five fixed seeds for 50 epochs: 4106, 4107, 4108, 4109, and 4110 (30 finalist runs total).

Reject a run if checkpoint reload is not equivalent, the observed active-feature maximum exceeds K, or validation dead-feature fraction exceeds 0.5. Among feasible finalists, select the smallest K within one standard error of the best mean validation explained variance, then prefer lower dead-feature fraction, smaller expansion, and lower MSE in that order.

Before the full search, repeat one bounded real-embedding configuration twice from independent initialization processes with the same fixed seed. Require exact equality of model tensors, optimizer state, checkpoint metadata apart from artifact location, normalization contents, training curves, deterministic metrics, gates, and run signature. Hash this passing report and the training sources into the freeze manifest.

## Label-free baselines

Fit on normalized training embeddings only and evaluate on the identical target images:

1. train-mean reconstruction;
2. K largest centered input coordinates;
3. PCA with K retained coefficients.

K is inherited from the frozen SAE selection. The PCA fit and all baseline reports are hashed before internal-test access.

## Endpoints

Primary representation endpoint: total explained variance of the selected mean-patch SAE on the 8,760-image decontaminated internal test, reported separately for all five seeds and as mean, standard deviation, standard error, minimum, and maximum.

Primary paired inference: per-image MSE averaged across the five fixed mean-patch SAE seeds versus the matched-K PCA baseline. Average the five SAE errors within image first, then use a deterministic paired patient-cluster bootstrap with 2,000 resamples and a two-sided 95% percentile confidence interval. This is the sole confirmatory baseline contrast in Experiment 1.

Secondary endpoints:

- per-dimension MSE;
- cosine reconstruction similarity;
- mean and maximum active features;
- dead-feature and saturated-feature fractions;
- SAE-minus-baseline differences for per-image MSE and cosine similarity.

For all paired SAE-baseline differences, use the same deterministic paired patient-cluster bootstrap. Resample patients, never individual eyes. Report every seed separately as a stability diagnostic and also report the within-image five-seed average. CLS, cosine, train-mean, coordinate, and per-seed contrasts are secondary; interpret their intervals descriptively without additional dichotomous significance claims.

## Freeze and access policy

Before internal-test extraction, freeze and hash:

- dataset manifest;
- this protocol and the executable configuration;
- both representation selections and all ten selected checkpoints;
- every screening/finalist metric and training curve plus both normalization artifacts; unselected checkpoint hashes are bound through their frozen metric records and verified during the completion audit;
- validation baseline reports and PCA artifacts;
- all analysis source files used by extraction, training, selection, and evaluation.

Internal-test extraction must recompute and verify the analysis bundle. Any missing or modified file fails closed. The final evaluator refuses to overwrite an existing locked-test report. A rerun is allowed only to reuse already-created embedding shards after a documented software interruption; it must not change the frozen selections or analysis.

## Interpretation boundary

Success supports only the claim that a selected sparse dictionary reconstructs frozen retinal representations under the prespecified sparsity and baseline comparisons. It does not establish that individual features represent lesions, that reconstruction preserves DR predictions, or that the method generalizes across datasets or cameras. Negative results and failed dead-feature gates remain reportable outcomes.
