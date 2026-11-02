# Additive versus pairwise activation AutoResearch

Status: complete bounded numerical search; no LLM used.

The frozen RETFound representation and 16,384-feature Top-K-32 SAE were held fixed. AutoResearch compared the existing weighted additive activation score with two prespecified pairwise co-activation rules. Feature pairs were restricted to the 64 highest train-derived mean-difference features, required at least 20 development activations, and were evaluated on a patient-disjoint 80/20 split inside labeled training data. The locked internal test was not accessed.

| Recipe | Development AUROC | Development AUPRC |
|---|---:|---:|
| Weighted mean difference, top 64 | **0.744** | **0.515** |
| Pair co-activation rate, top 128 | 0.708 | 0.451 |
| Pair activation product, top 128 | 0.688 | 0.355 |

The additive recipe remained selected. On the previously inspected validation pilot it reached AUROC 0.716 (patient-bootstrap 95% CI 0.686–0.745) and AUPRC 0.471 (CI 0.421–0.523). In this bounded search, pairwise interactions did not improve the activation score. This is not evidence that all interactions are unhelpful; it only covers the preregistered 64-feature/128-pair rules and should not be expanded after looking at the locked test.

Machine-readable output: `/Volumes/Elements/Retina-LanSE/experiments/activation_interaction_autoresearch/activation_interaction_autoresearch.json`.
