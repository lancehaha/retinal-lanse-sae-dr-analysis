# Experiment 10: Image + Patch + SAE-Card Qwen Pilot

## Purpose

This exploratory, paired pilot tests whether local Qwen2.5-VL:3B referable-DR estimates improve when a fundus image is accompanied by eight high-activation Patch-SAE crops and pre-existing Patch-SAE card context. It reuses the frozen Experiment 8 K=64 artifacts and its already accessed 162-image/81-patient pilot holdout. It is not a fresh clinical or external validation.

## Frozen inputs and card boundary

RETFound, the K=64 Patch-SAE, activation traces, raw-dense MLP outputs, and patient cohort assignment are frozen. No encoder, SAE, card, or numerical classifier is retrained. Only the 34 cards with a completed contrastive judgment are shown. All 34 failed that contrastive semantic test; the 30 cards whose validator output was malformed are excluded. Therefore card wording is explicitly unvalidated exploratory context, not a lesion or diagnostic claim.

Feature/grade summaries are fit from `sae_train` patients only. Holdout target labels are not read until every raw Qwen response is saved. The shuffled control preserves each event's activation, coordinate, and train-only association statistic but maps its text to a different feature using fixed seed 92010.

## Arms and endpoints

Each image is evaluated in four deterministic-temperature-zero arms: image-only; image plus eight-patch contact sheet; image plus patches plus real cards; and image plus patches plus shuffled cards. Qwen returns five grade probabilities and a referable probability. The evaluated referable score is always the auditable sum of its grade-2 through grade-4 probabilities; any disagreement with Qwen's separately returned referable field is retained. One formatting-only repair is permitted. Failed responses are recorded and receive the train-only grade prior in conservative all-image results.

The primary endpoint is referable-DR AUROC. Secondary endpoints are AUPRC, Brier score, five-grade quadratic weighted kappa, macro-F1, ordinal MAE, and completion rate. The frozen Experiment 8 raw-dense MLP is the numerical reference. All comparisons use 2,000 paired patient-cluster bootstrap resamples: patches minus image-only, real cards minus patches, real cards minus shuffled cards, and real cards minus raw dense MLP.

## Gates and outputs

Before the full run, five images must confirm 14x14 patch-coordinate alignment, unique feature selection, label-free prompts, correct real-card mapping, text-only shuffling, strict JSON parsing, and external-drive-only output. All outputs remain under `/Volumes/Elements/Retina-LanSE/experiments/experiment_10_image_patch_card_qwen_pilot_20260917/`: specification/hash receipts, training-only association tables, selection manifests, contact sheets, response checkpoints, raw and repaired JSON, metrics, paired intervals, and a CSV headed by the raw-dense baseline.
