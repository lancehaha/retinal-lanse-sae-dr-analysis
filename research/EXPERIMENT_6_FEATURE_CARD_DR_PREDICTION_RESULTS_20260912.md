# Experiment 6: Feature-Card DR Prediction Results

Status: **complete**. All Qwen predictions were generated locally with the frozen K=1024, seed-4106 SAE and fixed feature-card evidence. RETFound, SAE, card associations, prompts, and downstream numerical specifications were frozen before the LLM ablation.

## Numerical arms

| Arm | Locked AUROC | Locked AUPRC | Locked QWK | IDRiD AUROC | IDRiD QWK |
|---|---:|---:|---:|---:|---:|
| dense | 0.8402 | 0.6653 | 0.4311 | 0.8095 | 0.3985 |
| direct_card_evidence | 0.6875 | 0.3787 | 0.2733 | 0.6162 | 0.0389 |
| direct_card_evidence_251 | 0.6888 | 0.3792 | 0.2750 | 0.6345 | 0.0389 |
| learned_card_probe | 0.7954 | 0.5625 | 0.3618 | 0.6569 | 0.2415 |

## Qwen semantic-text controls

| Arm | Locked completion | Locked AUROC | Locked AUPRC | Locked QWK | IDRiD completion | IDRiD AUROC | IDRiD AUPRC | IDRiD QWK |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| real_descriptions | 0.980 | 0.5407 | 0.2158 | 0.0099 | 1.000 | 0.4567 | 0.5978 | -0.0469 |
| description_free | 0.982 | 0.5011 | 0.1920 | -0.0060 | 1.000 | 0.3514 | 0.5450 | 0.0181 |
| shuffled_descriptions | 0.973 | 0.5557 | 0.2221 | 0.0251 | 1.000 | 0.4961 | 0.6349 | 0.0213 |

## Semantic comparison

| Comparison | Locked AUROC difference | 95% CI | Holm-adjusted p |
|---|---:|---:|---:|
| real_descriptions_vs_description_free | +0.0396 | [+0.0023, +0.0765] | 0.0680 |
| real_descriptions_vs_shuffled_descriptions | -0.0149 | [-0.0397, +0.0083] | 0.2150 |

Interpretation: this is an association-based, same-model feature-card analysis. It does not establish lesion identity, causality, DR specificity, clinical deployment utility, or equivalence to ophthalmologist review.
