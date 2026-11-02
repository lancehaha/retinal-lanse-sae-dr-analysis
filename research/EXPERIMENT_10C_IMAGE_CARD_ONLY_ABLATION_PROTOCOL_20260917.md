# Experiment 10C: Image + SAE-card-only Qwen ablation

## Purpose

Test whether existing SAE-card evidence adds predictive information to the full fundus image without supplying the eight selected retinal patch crops. This is a distinct paired ablation of Experiment 10B and does not modify its frozen four-arm specification or outputs.

## Frozen inputs

- The 162-image, 81-patient Experiment 8 pilot holdout.
- The frozen raw-dense RETFound MLP prediction file.
- The frozen K=64 Patch-SAE activation traces.
- The same top-eight unique activated feature events per image.
- The 34 existing cards with non-empty hypotheses. Each previously failed contrastive semantic validation, so all wording is unvalidated exploratory context.
- Training-patient-only activation/grade association summaries.

No RETFound encoder, Patch-SAE, card, numerical prediction model, or association estimate is retrained.

## Qwen input and output

Qwen2.5-VL:3B receives only the full retinal image plus the real card text, event location/rank, activation percentile, and training-only association context. It receives no patch contact sheet, holdout label, patient identifier, or target outcome.

It must return `predicted_grade` in 0–4 and a scalar `referable_risk_0_to_100`. AUROC is computed from risk divided by 100; QWK uses the integer grade. One formatting-only repair is permitted and records no extra evidence.

## Gates and analysis

- A fresh five-image smoke must yield five valid outputs before full inference.
- The full 162-image prediction set must be complete before any holdout labels are opened for metrics.
- Compare the image-plus-real-card-only score with the frozen raw-dense MLP using the same 2,000 patient-cluster bootstrap procedure.
- All outputs, prompts, raw responses, parsed responses, and source hashes remain on the external Retina-LanSE drive.

This is exploratory evidence about a local VLM prompt and unvalidated SAE-card text. It is not lesion confirmation, a clinical model, or an independent validation study.
