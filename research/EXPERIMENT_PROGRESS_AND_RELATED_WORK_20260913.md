# Retina-LanSE: experiment progress and related work

**Recorded:** 2026-09-13  
**Clinical endpoint:** binary referable diabetic retinopathy (DR), EyePACS grade >= 2.  
**Main metric:** AUROC (higher is better; 0.50 is chance).  
**Representation boundary:** RETFound-CFP is frozen. SAEs are trained only on frozen RETFound embeddings. No result below establishes clinical deployment, lesion identity, or causal feature meaning.

## Current state

- **Numerical experiments 2--7:** complete and auditable.
- **SAE semantic cards:** 256 cards have auditable same-model Qwen hypothesis/contrast records. This is a semantic-consistency study, not expert validation.
- **External evaluation:** clean official IDRiD test, 102 image-examination proxy units; it is reported separately from patient-level EyePACS testing.
- **Main conclusion so far:** an SAE can reconstruct and sparsify the frozen retinal embedding, but no SAE/card method has shown a reliable improvement over raw dense RETFound for binary referable-DR AUROC on both locked EyePACS and IDRiD.

## Completed numerical experiments

The values in this table are full-label, locked EyePACS AUROC. A dash means that experiment did not run IDRiD. Direct comparisons should be made **within an experiment**, because SAE checkpoints and prediction-head specifications differ slightly across experiments.

| Experiment | Representation / prediction method | Locked EyePACS AUROC | IDRiD AUROC | Result relative to matched raw-dense baseline |
|---|---|---:|---:|---|
| 2 | Raw dense RETFound, logistic probe | 0.8397 | -- | baseline |
| 2 | SAE reconstruction | 0.7847 | -- | worse |
| 2 | Compact SAE code, selected 64 coordinates | 0.7286 | -- | worse |
| 2 | Dense + compact SAE features | 0.8388 | -- | -0.0009; neutral |
| 2 | Dense + SAE reconstruction | 0.8278 | -- | worse |
| 2 | Late dense-score + SAE-score fusion | 0.8399 | -- | +0.0002; neutral |
| 4 | Raw dense RETFound, matched sparse-compatible probe | 0.8408 | 0.8087 | baseline |
| 4 | K=1024 full SAE sparse code | 0.8286 | 0.7623 | worse |
| 4 | Dense + K=1024 full sparse code | 0.8341 | 0.7721 | worse |
| 4 | K=1024 SAE reconstruction | 0.8439 | 0.7908 | small internal increase; worse externally |
| 5 | Raw dense RETFound, matched logistic probe | 0.8402 | 0.8095 | baseline |
| 5 | Sorted top-1024 activation magnitudes only | 0.5506 | 0.4412 | much worse |
| 5 | Dense + sorted top-1024 activation magnitudes | 0.8396 | 0.8124 | neutral |
| 6 | Direct feature-card evidence rule | 0.6875 | 0.6162 | worse |
| 6 | Learned 256-card identity-preserving activation probe | 0.7954 | 0.6569 | worse |
| 6 | Qwen with real feature-card descriptions | 0.5407 | 0.4567 | much worse |
| 6 | Qwen without descriptions | 0.5011 | 0.3514 | near chance / worse |
| 6 | Qwen with shuffled descriptions | 0.5557 | 0.4961 | much worse |
| 7 | Dense + sorted values logistic | 0.8358 | 0.8042 | worse |
| 7 | Value-only MLP | 0.8405 | 0.8315 | near-neutral internally; highest external point estimate |
| 7 | Transformer with feature identity, no rank position | 0.8450 | 0.8095 | +0.0048 internally; not significant |
| 7 | Transformer with rank position, no feature identity | 0.8434 | 0.8217 | small internal increase |
| 7 | Full feature-identity + rank-position Transformer | 0.8450 | 0.8168 | +0.0048; primary AUROC criterion not met |

### Experiment 3: patient-level label-efficiency test

This experiment asked whether dense + compact SAE helps when DR labels are scarce. Values are five-replicate ensemble AUROC.

| EyePACS label budget | Dense locked AUROC | Dense + SAE locked AUROC | Dense IDRiD AUROC | Dense + SAE IDRiD AUROC |
|---:|---:|---:|---:|---:|
| 1% | 0.7756 | 0.7710 | 0.7326 | 0.7216 |
| 5% | 0.7956 | 0.7937 | 0.8225 | 0.8405 |
| 10% | 0.8203 | 0.8200 | 0.8079 | 0.7945 |
| 25% | 0.8276 | 0.8250 | 0.7456 | 0.7436 |
| 50% | 0.8375 | 0.8380 | 0.8132 | 0.8075 |
| 100% | 0.8397 | 0.8388 | 0.8067 | 0.8087 |

At the prespecified 10% internal comparison, the fusion-minus-dense AUROC difference was -0.00035 (95% CI -0.00608 to 0.00535). No label budget showed a supported fusion improvement.

### Experiment 4: genuine SAE activation-capacity sweep

Dense baseline was 0.8408 locked-test AUROC. Increasing K created genuinely denser codes and raised reconstruction fidelity, but not predictive fusion performance.

| Per-image Top-K | Raw SAE code AUROC | Dense + SAE code AUROC | SAE reconstruction AUROC |
|---:|---:|---:|---:|
| 32 | 0.7714 | 0.8070 | 0.8058 |
| 64 | 0.8085 | 0.8267 | 0.8247 |
| 128 | 0.8097 | 0.8251 | 0.8314 |
| 256 | 0.8107 | 0.8238 | 0.8381 |
| 512 | 0.8165 | 0.8276 | 0.8410 |
| 1024 | 0.8286 | 0.8341 | 0.8439 |

### Important secondary result: five-grade grading

The full identity--position Transformer improved five-grade quadratic weighted kappa (QWK) on locked EyePACS from 0.4311 (dense logistic) to 0.4962. Its binary AUROC increase was only +0.0048, 95% CI -0.0056 to +0.0151, so it cannot be claimed as a successful binary-DR diagnostic improvement. The ordinal-grading signal is a justified follow-up, not a clinical claim.

## Semantic / AutoResearch progress

1. A K=1024 SAE (seed 4106) generated 256 feature cards from high-activation retinal examples.
2. Qwen generated feature hypotheses without DR labels in the prompts.
3. A contrastive check used fresh high-activation and low-activation image groups. It tested whether each stated pattern occurs more often in high than low activations.
4. The original bounded two-round study plus recovery produced auditable records for all 256 requested cards. A separate ten-round rescue was used for five initially unresolved cards and is explicitly exploratory.
5. The cards are useful for inspecting learned activation patterns and describing numerical predictions, but they have **not** been confirmed by ophthalmologists and must not be called verified lesions.

## Closest related image-based medical SAE projects

| Project | Modality and method | Why it is relevant to Retina-LanSE | Important difference |
|---|---|---|---|
| [SAIL: Sparse Autoencoders for Interpretable Medical Image Representation Learning](https://arxiv.org/abs/2603.23794) | Frozen BiomedParse/DINOv3 embeddings -> Matryoshka/BatchTopK SAE; VLM feature descriptions and language-driven retrieval on 909,873 CT/MRI slices | **Closest overall design.** It turns dense image-foundation-model embeddings into sparse features, auto-describes them with a VLM, and tests sparse fingerprints. | Normal-anatomy CT/MRI and retrieval-focused; it does not test retinal DR diagnosis or a dense-versus-SAE fusion advantage. |
| [Mammo-SAE](https://arxiv.org/abs/2507.15227) | Patch-level SAE on Mammo-CLIP mammography features; latent concept probes, spatial maps, and intervention | **Closest disease-image interpretability analogue.** It identifies breast concepts, investigates confounders, and tests interventions. | Mammogram patches with mass/calcification labels and spatial ground truth; our current SAE is whole-image retinal embedding based, without lesion labels. |
| [SAE-Rad: An X-Ray Is Worth 15 Features](https://arxiv.org/abs/2410.03334) | Chest X-ray ViT activations -> SAE features -> language descriptions -> radiology report generation | Closest analogue to our feature-card-to-language explanation path. | Report generation rather than DR classification; it uses report-derived language supervision. |
| [PICASSO: Dissecting and directing pathology foundation models](https://pmc.ncbi.nlm.nih.gov/articles/PMC13307983/) | Pathology-foundation-model patch embeddings -> Top-K SAE concept dictionary; concept visualisation and manipulation | Strong precedent for dictionary features as interpretable visual concepts and for testing feature interventions. | Histology patches at very large scale; not fundus images or patient-level DR screening. |
| [Sparse coding for retinal bright-lesion discrimination](https://pubmed.ncbi.nlm.nih.gov/25935125/) | Classical image-patch sparse dictionary + linear SVM for normal/drusen/exudates images | **Closest retinal precedent.** It shows sparse dictionary methods have been used on fundus images before. | Classical sparse coding, 828 images, lesion-type discrimination; not a modern frozen foundation model plus mechanistic SAE. |
| [Sparse autoencoder + XAI for ulcerative-colitis diagnosis](https://pubmed.ncbi.nlm.nih.gov/40488164/) | Sparse autoencoder feature extraction followed by a classifier and Grad-CAM on endoscopic images | Relevant as an image-diagnosis SAE plus explanation example. | Conventional supervised image SAE, not sparse decomposition of a frozen foundation-model embedding; its reported accuracy is not directly comparable. |

## Positioning for a paper

The defensible novelty is not “SAE improves DR diagnosis.” The evidence does not support that claim. A stronger, accurate position is:

> We evaluate whether a Top-K sparse dictionary can convert frozen RETFound retinal embeddings into auditable activation features; we test reconstruction, label efficiency, capacity, direct feature-card prediction, language-based explanations, and supervised identity/rank fusion under patient-level and external evaluation.

The most relevant follow-up from the literature is a **patch-level retinal SAE plus intervention study**, modeled after Mammo-SAE/PICASSO: use retinal patch tokens or image regions, link activations to known lesion annotations where licensed data provide them, and test whether turning a feature down changes the DR prediction. This directly addresses the gap between whole-image activation cards and lesion-level evidence.

## Source reports

- `research/EXPERIMENT_2_DENSE_SAE_FUSION_RESULTS_20260906.md`
- `research/EXPERIMENT_3_LABEL_EFFICIENCY_EXTERNAL_RESULTS_20260907.md`
- `research/EXPERIMENT_4_SAE_TOPK_CAPACITY_RESULTS_20260909.md`
- `research/EXPERIMENT_6_FEATURE_CARD_DR_PREDICTION_RESULTS_20260912.md`
- `research/EXPERIMENT_7_SUPERVISED_IDENTITY_RANK_FUSION_RESULTS_20260912.md`
- External Experiment 5 record: `/Volumes/Elements/Retina-LanSE/experiments/experiment_5_active_value_autoresearch_20260909/stage1_active_value_fusion.json`
