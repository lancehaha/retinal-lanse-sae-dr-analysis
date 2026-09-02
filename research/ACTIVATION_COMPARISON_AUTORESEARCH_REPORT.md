# Activation-comparison AutoResearch pilot

This is a LanSE-inspired comparison experiment, not a new SAE training run. The 16,384-feature Top-K-32 SAE and frozen RETFound encoder were held fixed. AutoResearch searched four prespecified ways to construct an image-level activation score from the sparse retinal feature vector.

## Search protocol

- Search data: labeled training images only.
- Split: deterministic patient-disjoint 80/20 split inside the labeled training run.
- Development fit: 22,518 images / 11,259 patients.
- Development evaluation: 5,612 images / 2,806 patients.
- Candidate recipes: binary log-rate ratio, binary rate difference, weighted mean difference, and weighted standardized difference; each used the top 64 features with at least 50 training activations.
- Selection: highest development AUROC, AUPRC as tie-breaker.
- Locked internal test: not accessed.

## Result

The selected recipe was `weighted_mean_difference_top64`.

| Recipe | Development AUROC | Development AUPRC |
|---|---:|---:|
| Binary log-rate ratio | 0.666 | 0.422 |
| Binary rate difference | 0.706 | 0.477 |
| **Weighted mean difference** | **0.711** | **0.477** |
| Weighted standardized difference | 0.707 | 0.452 |

On the previously inspected validation pilot, the selected activation score reached AUROC **0.716** (patient-bootstrap 95% CI **0.686–0.745**) and AUPRC **0.471** (CI **0.421–0.523**).

This result is evidence that a train-derived combination of sparse activation patterns separates referable-DR groups better than the earlier unstructured direct sparse probe (AUROC 0.637). It remains exploratory because the validation set was previously inspected and no locked-test result exists.

## Interpretation boundary

The search changed only the data-comparison rule. It did not change the SAE, encoder, labels, or patient split. The resulting score is an activation-based retinal-group comparison, not a validated diagnostic model. Strong feature claims still require blinded ophthalmologist review and a fresh holdout evaluation after the analysis specification is frozen.

Machine-readable output: `/Volumes/Elements/Retina-LanSE/experiments/activation_comparison_autoresearch/activation_comparison_autoresearch.json`.
