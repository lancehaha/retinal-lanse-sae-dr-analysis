# Experiment 7C: Three SAE Feature-View Transformers

## Status and claim boundary

This protocol is implemented but the full experiment is not yet a result. It is an exploratory follow-up because the EyePACS locked test and IDRiD were examined in earlier experiments. Feature-card descriptions are same-model visual hypotheses, not clinician-confirmed lesions.

## Fixed question

Do three alternative representations of an activated SAE feature—its feature-card hypothesis, its decoder direction, or its categorical identity—improve SAE-only prediction relative to the same activation-only Set Transformer?

Every arm uses the same 251 features whose final Experiment 5B status is `passed_initial` or `revised_and_passed`. All other dictionary features are removed. The final hypothesis version is the card text. RETFound dense embeddings, activation ranks, rank groups, and positional embeddings are excluded from model input.

## Representations and model

- **Card + activation:** frozen `sentence-transformers/all-MiniLM-L6-v2` sentence embedding, reduced from 384 to 128 dimensions by label-free PCA across the 251 cards.
- **Decoder + activation:** frozen normalized SAE decoder column, reduced from 1,024 to 128 dimensions by label-free PCA across the same 251 features.
- **ID + activation:** learned 128-dimensional categorical embedding over the compact 251-ID mapping.
- **Controls:** activation-only logistic regression over the 251 fixed coordinates and an activation-only Set Transformer.

Each active token receives train-only standardized `log1p(activation)` and log within-image activation share through a shared 2–64–128 MLP and learned gate. `[CLS]` and the unordered active tokens enter four pre-layer-normalized Transformer layers with dimension 128, eight heads, FFN width 512, GELU, and dropout 0.1. Input is serialized by feature ID for reproducibility, but random token permutation must leave evaluation-mode predictions unchanged.

## Training and evaluation

The frozen K=1024, dictionary-size 16,384, seed-4106 SAE, patient cohorts, 100% and deterministic 10% label budgets, and seeds 4106–4110 match Experiment 7. Binary and five-grade models use weighted BCE and class-balanced cross-entropy. A patient-disjoint inner 90/10 split selects epochs with patience 8 and a 50-epoch cap; the median epoch is frozen before five reinitialized final fits. Batch size is 16 with four-step accumulation.

The primary family contains three full-label EyePACS locked-test AUROC comparisons: card, decoder, and ID arms against activation-only Transformer. Each uses 2,000 patient-cluster bootstrap resamples and Holm correction. An adjusted interval above zero defines statistical usefulness; a difference of at least 0.01 is the practical threshold. EyePACS validation, five-grade metrics, 10%-label results, IDRiD, pairwise arm comparisons, and frozen representation-mapping shuffles are secondary.

## Execution and evidence

`scripts/run_sae_feature_view_transformers.py` has three stages:

1. `prepare` freezes the 251 card texts, text-model revision/cache hash, decoder directions, PCA banks, selected IDs, and provenance without loading labels.
2. `smoke` checks masking, token-order invariance, gradients, and model contracts without scientific data.
3. `run` freezes the complete specification before evaluation, trains resumably, verifies checkpoint reloads, and writes predictions, comparisons, machine-readable results, and publication tables under the external experiment directory.

A completed claim requires all three requested arms, both controls, both tasks, both label budgets, five seeds, all evaluation datasets, mapping-shuffle controls, and complete hashes. Partial or failed runs remain implementation evidence only.
