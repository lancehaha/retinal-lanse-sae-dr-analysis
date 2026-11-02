# Experiment 2: Dense RETFound + SAE Feature Fusion

Date: 2026-09-06

## Objective

Test whether the dense RETFound representation and the sparse SAE
representation contain complementary information for referable diabetic
retinopathy (DR).

The hypothesis is that the dense branch preserves broad image detail and
global context, while the SAE branch emphasizes a small number of interpretable
activation patterns. The primary question is whether combining them improves
DR prediction over the dense RETFound baseline.

This experiment tests numerical complementarity first. It does not include an
LLM explanation branch in the primary analysis, so any improvement can be
attributed to the representations rather than prompt wording or VLM behavior.

## Endpoint and data partitions

- Endpoint: referable DR, EyePACS grade >= 2.
- Train/validation/test partitions remain patient-disjoint.
- The dense encoder and SAE remain frozen.
- All normalization, feature selection, regularization selection, calibration,
  and thresholds are fitted using training data only.
- The validation partition is used for prespecified model selection and
  diagnostics.
- The locked internal test is opened once after the protocol is frozen.
- IDRiD is reserved for a separate one-time external evaluation and is not
  used to select the fusion design.

## Representation branches

Let `z` be the 1,024-dimensional dense RETFound embedding and `a` be the
16,384-dimensional SAE activation vector.

### Arm A: Dense baseline

Train the existing class-balanced linear probe on `z`. This is the primary
reference model.

### Arm B: SAE reconstruction baseline

Decode `a` into the reconstructed embedding `z_hat`, then train the same
linear-probe family on `z_hat`. This measures how much predictive information
survives SAE compression and reconstruction.

### Arm C: Compact sparse baseline

Use the prespecified 64 SAE features from the current training-only numerical
policy. Standardize their activation values using training statistics and fit
the same linear-probe family. This is a fairer predictive baseline than the
one-dimensional weighted score.

### Arm D: Early feature fusion (primary fusion arm)

Concatenate the dense embedding and compact sparse activations:

`[z ; a_64]`

Fit one regularized linear probe on this vector. The dense branch supplies
image detail; the sparse branch supplies feature-focused evidence.

### Arm E: Dense plus reconstructed embedding

Concatenate:

`[z ; z_hat]`

This tests whether reconstruction provides complementary information or is
mostly redundant with the original dense embedding.

### Arm F: Late score fusion

Combine the calibrated dense-probe probability and the fixed numerical SAE
score using a two-input logistic meta-model fitted on training data only:

`p_fused = logistic(w_dense * p_dense + w_sae * score_sae + b)`

This tests complementarity without exposing a high-dimensional sparse vector
to the fusion model.

## Leakage and model-selection controls

1. The SAE checkpoint, selected feature IDs, and activation aggregation rule
   are frozen before fusion evaluation.
2. The 64 compact SAE features are selected from training data only.
3. Standardization parameters are computed separately on training data only.
4. Regularization strength is selected by patient-group cross-validation inside
   the training partition, not by the locked test.
5. The validation partition may select the primary fusion arm only according
   to the predeclared rule below; it cannot change the feature definition.
6. No IDRiD labels, clinician labels, VLM outputs, or external data are used
   for model selection.

## Primary comparison and success rule

The primary comparison is Arm D versus Arm A on the locked internal test.

Report:

- AUROC and AUPRC;
- paired patient-cluster bootstrap confidence intervals for the difference;
- Brier score and calibration curve;
- sensitivity, specificity, PPV, and NPV at the train-only operating threshold;
- performance by EyePACS grade and image-quality subgroup where available.

Before opening the locked test, define a practically meaningful improvement as
an absolute AUROC increase of at least 0.03. A strong positive result requires
both:

1. fusion AUROC improvement >= 0.03; and
2. the paired 95% confidence interval for the AUROC difference excludes zero.

A smaller positive difference is reported as an incremental signal, not a
large performance improvement. If the confidence interval includes zero, the
experiment provides no reliable evidence that SAE features add predictive
information. If fusion decreases performance, the SAE branch is currently
redundant or noisy for prediction.

## Complementarity diagnostics

In addition to the primary AUROC comparison, record:

- correlation between dense and SAE scores;
- the fraction of dense-model errors corrected by fusion;
- the fraction of dense-model correct predictions made worse by fusion;
- coefficient stability for the SAE branch across patient-group bootstrap
  resamples;
- performance when the SAE branch is randomly permuted within the test
  evaluation protocol as a negative control;
- ablation of the SAE branch from the trained fusion model.

The SAE branch should be considered complementary only if its addition improves
held-out performance and produces stable, nonzero contribution—not merely
because its coefficients are nonzero on the training set.

## Analysis sequence

### Phase 1: Training-only construction

Generate `z`, `a`, `z_hat`, and the compact `a_64` branch from existing frozen
artifacts. Fit all candidate probes and the late-fusion meta-model using the
training partition with patient-group cross-validation for regularization.

### Phase 2: Validation selection

Evaluate all six arms on validation. Select the primary fusion arm using the
predeclared AUROC/AUPRC rule, record the complete candidate table, and freeze
the specification. Do not inspect the locked test during this phase.

### Phase 3: Locked internal test

Run the frozen selected models once on the locked test. Compute paired
patient-bootstrap intervals and the complementarity diagnostics. This is the
main result for Experiment 2.

### Phase 4: External validation

After the internal specification is frozen, apply the same selected fusion
model to the clean IDRiD external test without retraining, retuning, or prompt
changes. Report DR-grade/referable performance separately from lesion or
localization evidence.

## Optional second experiment: explanation fusion

Only if numerical fusion shows a reproducible gain, run a separate exploratory
study in which the VLM receives the image, the dense-model score, and SAE
feature cards. Compare:

1. image only;
2. image + dense score;
3. image + SAE cards;
4. image + dense score + SAE cards.

This is a new ablation and must not replace the numerical fusion result. LLM
feature descriptions remain hypotheses until blinded retinal-specialist review.

## Expected interpretations

- **Dense + SAE improves substantially:** the SAE branch contains useful
  information not fully captured by the dense probe.
- **Dense + SAE improves slightly:** the SAE is complementary but adds limited
  predictive information.
- **No improvement:** the current SAE features mostly re-express information
  already available to the dense branch, or the current compact feature set
  and scoring rule are insufficient.
- **Performance decreases:** the SAE branch introduces nuisance variation or
  overfitting; it should not be used for the predictive model without a revised
  protocol.

## Current status

The frozen EyePACS fusion experiment has been executed. Results are recorded in
`research/EXPERIMENT_2_DENSE_SAE_FUSION_RESULTS_20260906.md`. The dense-plus-SAE
fusion did not produce a meaningful improvement over the dense baseline. The
IDRiD external fusion evaluation remains pending.
