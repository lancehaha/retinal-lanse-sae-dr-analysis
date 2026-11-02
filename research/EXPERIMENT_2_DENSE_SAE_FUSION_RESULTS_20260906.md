# Experiment 2 Results: Dense RETFound + SAE Fusion

Date: 2026-09-06

Status: complete for the frozen EyePACS internal experiment.

## Execution contract

- RETFound encoder: frozen.
- SAE checkpoint: frozen Top-K-32, 16,384 features.
- Endpoint: referable DR, EyePACS grade >= 2.
- Patient-disjoint train, validation, and locked internal-test cohorts.
- SAE feature selection, normalization, model fitting, and late fusion were
  performed using training data only.
- The fusion runner used the same class-balanced logistic-probe family for all
  arms, with `C=1` and train-only standardization. Comparisons below are paired
  within this run.
- Patient-cluster bootstrap intervals use 2,000 resamples.

## Cohorts

| Cohort | Images | Patients | Referable-DR images |
|---|---:|---:|---:|
| Train | 28,130 | 14,065 | 5,527 |
| Validation | 3,476 | 1,738 | 678 |
| Locked internal test | 3,518 | 1,759 | 667 |

## Results

| Arm | Validation AUROC | Validation AUPRC | Locked-test AUROC | Locked-test AUPRC |
|---|---:|---:|---:|---:|
| Dense RETFound | 0.8393 | 0.6543 | 0.8397 | 0.6653 |
| SAE reconstruction | 0.7877 | 0.5651 | 0.7847 | 0.5830 |
| Compact SAE features (64) | 0.7272 | 0.5122 | 0.7286 | 0.5252 |
| Dense + compact SAE features | 0.8399 | 0.6574 | 0.8388 | 0.6649 |
| Dense + SAE reconstruction | 0.8286 | 0.6443 | 0.8278 | 0.6462 |
| Late dense-score + SAE-score fusion | 0.8395 | 0.6488 | 0.8399 | 0.6608 |

## Primary comparison

The prespecified primary comparison was dense + compact SAE features versus
dense RETFound alone on the locked internal test.

- Dense baseline AUROC: **0.8397**
- Dense + SAE AUROC: **0.8388**
- Difference: **-0.0009**
- 95% patient-bootstrap CI: **[-0.0037, 0.0020]**
- Dense + SAE AUPRC difference: **-0.0004**
- 95% patient-bootstrap CI for AUPRC difference: **[-0.0053, 0.0049]**

The confidence intervals include zero and the change is far below the
predeclared 0.03 AUROC threshold for a practically meaningful improvement.

Late score fusion was similarly neutral:

- AUROC difference: **+0.0002**, 95% CI **[-0.0010, 0.0013]**
- AUPRC difference: **-0.0045**, 95% CI **[-0.0103, 0.0011]**

Concatenating the dense embedding with the SAE reconstruction reduced
performance (AUROC difference **-0.0118**, 95% CI **[-0.0212, -0.0033]**).

## Interpretation

The current SAE branch does not provide measurable predictive information beyond
the dense RETFound embedding on the locked EyePACS test. The two branches may
look conceptually different—dense image detail versus sparse feature emphasis—
but the current SAE features are either largely redundant with the dense
representation or too noisy after selection to improve classification.

This does not show that SAE features are useless for interpretation. It shows
that the current numerical fusion design does not improve DR prediction. The
SAE reconstruction and compact sparse arms remain weaker than the dense
baseline, and the LLM explanation branch must not be presented as a predictive
gain based on this result.

### Plain-language conclusion

For the current implementation, **SAE does not help much with DR prediction**:
adding SAE features to the dense RETFound model produced essentially no change
in AUROC or AUPRC. SAE remains useful for studying and visualizing activation
patterns, but this experiment does not support claiming that SAE improves
diagnostic accuracy.

## External status

The IDRiD external evaluation remains a separate frozen Stage 6 experiment and
has not been run from this fusion result. No external or clinical-validation
claim is made here.

## Artifact

Machine-readable output:

`/Volumes/Elements/Retina-LanSE/experiments/experiment_2_dense_sae_fusion_20260906/dense_sae_fusion_results.json`
