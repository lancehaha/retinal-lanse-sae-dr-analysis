# Current Retina-LanSE Experiment Design and Results

Date: 2026-09-06

Status: internal frozen-representation experiment completed; external IDRiD
evaluation prepared but not yet executed.

## Research question

Can sparse autoencoder (SAE) activations derived from a frozen retinal encoder
retain and expose information associated with referable diabetic retinopathy
(DR), and does adding SAE-derived feature context improve a vision-language
model (VLM)?

The endpoint used in the EyePACS experiments is binary referable DR:
EyePACS grade >= 2. This is a DR endpoint, not a direct diagnosis of diabetes.

## Fixed representation pipeline

- Dataset: EyePACS, with patient-disjoint train, validation, and locked-test
  partitions.
- Encoder: pretrained RETFound-CFP, frozen; no encoder fine-tuning.
- Representation: 1,024-dimensional final mean-patch embedding.
- SAE: 16,384 dictionary features with Top-K-32 sparsity. SAE training uses
  label-free training embeddings.
- Heavy data, checkpoints, embeddings, and outputs are stored on the external
  drive under `/Volumes/Elements/Retina-LanSE`.
- The locked internal test was not used for feature selection, threshold
  selection, prompt selection, or model fitting.

## Completed analysis arms

### Dense frozen-embedding baseline

A class-balanced linear probe was fitted on the dense 1,024-dimensional
RETFound embedding. This is the predictive reference, not an SAE method.

### SAE reconstruction probe

The SAE sparse code was decoded back into a reconstructed 1,024-dimensional
embedding. A linear probe was then fitted on that reconstruction. This tests
whether the SAE preserves predictive information after decoding.

### Direct sparse-code probe

A linear probe was fitted directly on the 16,384-dimensional sparse activation
vector. This tests the predictive signal available without reconstruction.

### Numerical AutoResearch SAE score

AutoResearch compared prespecified activation aggregation rules. The selected
policy was a weighted mean-difference score using 64 features selected from the
labeled training cohort only. It is an association score, not a claim that
the selected features are DR-specific or always active in DR images. A train-
only calibration and threshold were recorded; validation AUROC/AUPRC evaluate
the fixed score ranking.

### VLM ablation

The same frozen local VLM was run in two input conditions:

1. retinal image only;
2. retinal image plus SAE-derived feature cards.

No task-specific VLM fine-tuning was performed. The comparison tests whether
feature-card context changes VLM DR inference.

### Feature atlas and hypothesis generation

Highly activating images were collected for prominent SAE features. A VLM
generated visual hypotheses and nuisance alternatives. These are exploratory
descriptions; they are not clinician-validated feature identities.

## Results on the internal validation cohort

The validation cohort contains 3,476 images from 1,738 patients, including
678 referable-DR images.

| Arm | AUROC | AUPRC | Interpretation |
|---|---:|---:|---|
| Dense frozen RETFound linear probe | 0.837 | 0.590 | Strongest current predictive reference |
| SAE reconstruction linear probe | 0.795 | 0.542 | SAE reconstruction retains substantial signal |
| Direct sparse-code linear probe | 0.637 | 0.274 | Above chance but substantially weaker |
| Numerical AutoResearch SAE score | 0.716 | 0.471 | Moderate DR-related association |

The numerical SAE score is therefore below the dense representation: AUROC
0.716 versus 0.837. The reconstructed embedding performs better (0.795),
which suggests that the main loss comes from the simple one-dimensional
activation score rather than complete loss of information in the SAE.

The VLM ablation used a separate 1,024-image balanced matched cohort:

| VLM condition | AUROC | AUPRC | Interpretation |
|---|---:|---:|---|
| Image only | 0.500 | 0.500 | Random-level result on this ablation cohort |
| Image + active SAE cards | 0.485 | 0.490 | No demonstrated improvement |

The paired AUROC difference (feature cards minus image only) was approximately
-0.0147, with the recorded confidence interval spanning zero. The current
data do not support an improvement claim for SAE cards.

Label-free reconstruction on the locked internal test was also measured, but
these are not DR classification metrics:

| SAE representation | Explained variance | Cosine similarity |
|---|---:|---:|
| Mean-patch | 0.9223 | 0.9572 |
| CLS | 0.9736 | 0.9856 |

## Interpretation and limits

1. The SAE reconstructs the frozen retinal representation well.
2. The sparse representation contains measurable DR-related information.
3. The current additive top-64 score is weaker than a dense linear probe.
4. The current feature-selection rule measures average association; it does
   not establish uniqueness, causality, lesion identity, or absence from
   normal images.
5. AutoResearch generated candidate feature hypotheses and compared scoring
   recipes, but it did not produce clinician-validated semantic identities for
   the SAE features.
6. The VLM feature-card ablation has not shown an accuracy benefit.
7. These results are not clinical validation and do not establish diabetes
   diagnosis performance.

## External validation status

Official IDRiD files have been extracted and audited on the external drive.
The raw manifest contains 516 labeled images: 413 official training images
and 103 official testing images. One exact duplicate occurs across the
official partitions with conflicting labels; it is excluded from the primary
clean external test, leaving 102 testing images. The frozen IDRiD accuracy
evaluation has not yet been run, so no external AUROC/AUPRC is reported here.

## Dense+SAE fusion follow-up

The first numerical fusion experiment has now been completed. On the locked
internal test, dense RETFound achieved AUROC 0.8397, while dense plus the
compact 64-feature SAE branch achieved AUROC 0.8388. The paired difference was
-0.0009 (95% CI -0.0037 to 0.0020), so the fusion did not improve performance.
Full arm results are recorded in
`research/EXPERIMENT_2_DENSE_SAE_FUSION_RESULTS_20260906.md`.

Plain-language result: for the current implementation, SAE does not help much
with DR prediction. It may still support interpretability and feature analysis,
but no predictive-accuracy improvement has been demonstrated.

## Recommended next experiment

The next rigorous extension should keep the dense and SAE baselines fixed and
evaluate all prominent activations rather than only the largest group-level
DR/non-DR differences. For each feature, record activation prevalence in both
groups, nuisance correlations, conditional effects, and pairwise interactions.
Feature descriptions should be treated as hypotheses until blinded retinal-
specialist review is completed. IDRiD must remain a one-time frozen external
evaluation with no retuning.

## Source artifacts

- Consolidated results: `/Volumes/Elements/Retina-LanSE/reports/RETINA_LANSE_CONSOLIDATED_REPORT_20260905.md`
- Feasibility comparison: `research/FEASIBILITY_SINGLE_SAE_REPORT.md`
- Results summary: `research/RESULTS_SUMMARY.md`
- Numeric SAE diagnostics: `/Volumes/Elements/Retina-LanSE/experiments/numeric_sae_diagnostics/numeric_sae_diagnostics.json`
- Feature atlas: `/Volumes/Elements/Retina-LanSE/experiments/feature_atlas_autoresearch/feature_atlas_summary.json`
