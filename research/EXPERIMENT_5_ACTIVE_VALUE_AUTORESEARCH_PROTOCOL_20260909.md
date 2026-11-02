# Experiment 5: Active-value fusion with AutoResearch feature explanations

**Status:** implementation complete; preflight gates passed; Stage 1 numerical
execution is in progress. Heavy outputs are restricted to
`/Volumes/Elements/Retina-LanSE/experiments/experiment_5_active_value_autoresearch_20260909/`.

## Stage 1: active-value numerical test

The RETFound encoder and five completed K=1024 SAE checkpoints (seeds
4106–4110) are frozen. For each image, nonzero SAE `(value, feature_id)` pairs
are sorted by descending value with feature ID as a deterministic tie-breaker.
Only the 1,024 values are supplied to the numerical model, zero-padded when
needed; the paired feature IDs are stored separately for later explanations.

The three matched arms are:

```text
dense:                1,024 RETFound values
active_values:        1,024 sorted SAE activation values
dense_plus_active:    [dense ; active_values] = 2,048 values
```

The classifier is class-balanced logistic regression with `C=1`, `lbfgs`,
`max_iter=2000`, fixed `tol=1e-3`, and train-only standardization. The tolerance
is predeclared as a CPU-execution setting; solver family, regularization, and
iteration cap remain unchanged. Primary analysis uses 100% of
the labeled EyePACS training cohort; 10% patient-level sampling is secondary.
Validation, locked internal testing, and clean 102-image IDRiD testing follow
the Experiment 4 cohorts. The primary endpoint is binary referable DR (grade
≥2) AUROC; AUPRC, Brier, ECE, threshold metrics, and five-grade metrics are
secondary. Five replicate probabilities are averaged, and paired dense-versus-
fusion differences use 2,000 patient/image-examination bootstrap resamples.

The numerical stage is frozen before AutoResearch. AutoResearch cannot modify
these classifiers or thresholds.

## Stage 2: K=1024 AutoResearch feature meaning

The semantic analysis uses the predeclared K=1024, seed=4106 checkpoint. All
71,134 unlabeled EyePACS training embeddings are encoded. The top 256 features
by global activation mass, subject to at least 50 activations, receive cards
with 32 deterministic high-activation examples each. Hypothesis prompts hide
clinical labels and prohibit DR inference.

Local Ollama `qwen2.5-vl:3b` generates one visual hypothesis, evidence strength,
alternative explanations, nuisance flags, and uncertainty per card. After the
hypotheses are frozen, all training images are scored computationally for
activation frequency and DR/non-DR association. Labels are used only for this
audit and recipe selection; no feature is called DR-specific.

Four prespecified activation recipes are compared on a patient-disjoint
development split and selected on validation: weighted activation sum,
non-DR-penalized sum, top-16 positive-evidence vote, and LLM-evidence-weighted
sum. The selected recipe is then evaluated once on locked internal test and
clean IDRiD.

## Stage 3: explanations and exploratory LLM prediction

For each internal-test and IDRiD image, all 1,024 feature IDs and values are
retained in the machine-readable trace. Only the top 16 active features are
verbalized. Qwen2.5-VL receives feature-card hypotheses, activation strengths,
and verification statistics, but not the target image, hidden label, or test
outcome. It returns structured observations, compatible/conflicting evidence,
uncertainty, an explanation, and an independent five-grade probability vector.

The explanation is post-hoc and cannot change the numerical prediction. The
independent LLM prediction is exploratory and is scored separately. These
outputs describe feature-card hypotheses, not direct lesion confirmation or
specialist validation.

## Files and gates

Implementation files are `scripts/run_active_value_fusion_1024.py`,
`scripts/run_autoresearch_feature_meaning_k1024.py`, and
`scripts/explain_retinal_prediction_k1024.py`. Required gates include exact
1,024-value sorting/padding, no feature IDs in classifier inputs, checkpoint
reload and patient-split checks, hidden labels in hypothesis prompts, external
output-path enforcement, resumable LLM JSONL, and explicit failure records.
