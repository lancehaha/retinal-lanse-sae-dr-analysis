# Retina-LanSE publication protocol

Version: 0.1, written before access to the clinical test sets.

All heavy data and experiment artifacts are stored under `/Volumes/Elements/Retina-LanSE` according to `configs/storage.yaml`. The system disk is limited to source code, small manifests, and reports.

## 1. Claim boundary

The paper will claim that sparse features can audit a frozen retinal foundation model. It will not claim diagnostic safety, biological causality, or that a neuron is a unique disease mechanism. The prior synthetic run is software evidence only and is excluded from all scientific tables.

LanSE already reports language-grounded sparse encoders in medical imaging, including CXR-LanSE. The novelty here is therefore not “SAEs for medicine.” The proposed contribution is the combination of:

1. frozen retina-specific foundation representations;
2. separate whole-image and high-resolution tile dictionaries;
3. direct lesion-mask localization;
4. lesion-versus-matched-control occlusion;
5. feature matching and revalidation across seeds;
6. external dataset/device validation.

## 2. Data roles

| Role | Preferred dataset | Scale and use | Access constraint |
|---|---|---|---|
| Unlabelled SAE fitting | EyePACS/Kaggle DR | Up to 88,702 images; patient-paired filenames; labels ignored for SAE fitting | 88.29 GB; Kaggle rules prohibit redistribution |
| Optional training diversity | APTOS 2019 | 3,662 labelled training images; labels ignored if included in SAE training | 10.22 GB competition data; rules must be accepted |
| Clinical development | DDR | 13,673 images from 9,598 patients; 757 lesion-annotated images | Use official release and preserve official split |
| External mask test 1 | IDRiD | 516 graded images; 81 lesion-mask images | CC BY 4.0; 81 MA/EX, 80 HE, 40 SE |
| External mask test 2 | FGADR Seg-set | 1,842 pixel-annotated images | Non-commercial research agreement; no redistribution |
| External grade/quality test | APTOS and optionally Messidor-2 | Cross-dataset downstream fidelity and camera robustness | Messidor-2 has 1,748 images but no official DR ground truth |

The primary feasible corpus is EyePACS-only SAE fitting with DDR development and locked IDRiD plus FGADR external evaluation. Do not use repackaged third-party dataset mirrors: they obscure licence terms, preprocessing, duplicates, and patient grouping.

## 3. Leakage controls

1. Hash every raw file and preserve the original dataset split.
2. Group EyePACS left/right images by the numeric subject identifier. No patient's images or tiles may cross splits.
3. Tiles inherit the parent image split; tile-level random splitting is prohibited.
4. Run exact SHA-256 and perceptual-hash duplicate detection within and across datasets before any split is opened.
5. Fit crop thresholds and embedding normalization on the SAE training split only.
6. Ignore all clinical labels while training and selecting the reconstruction SAE.
7. Use DDR development labels/masks for candidate screening and thresholds only.
8. Keep IDRiD and FGADR test manifests encrypted/read-only or permission-locked until the analysis code, neuron set, hypotheses, thresholds, and multiplicity plan are frozen.
9. Record that RETFound-CFP was pretrained partly on EyePACS. Use the base pretrained checkpoint, not a checkpoint fine-tuned on any evaluation dataset.
10. Never generate a label-derived caption and later present the same label as independent evidence of unsupervised discovery.

## 4. Encoder and embedding extraction

Use the gated official `RETFound_mae_natureCFP` ViT-Large checkpoint (1,024-dimensional encoder representation) under its CC BY-NC 4.0 licence. The encoder stays frozen throughout SAE training.

Before the large run:

- reproduce the official feature-vector notebook on ten approved images;
- record repository commit, checkpoint SHA-256, model licence, PyTorch/CUDA versions, preprocessing code hash, and exact pooling/layer;
- compare final CLS and final mean-patch pooling as two preregistered representations;
- benchmark 1,000 images to project wall time/storage before launching the full extraction;
- store embeddings in sharded, checksummed files with record-level metadata.

Main global input: full retinal field cropped by a versioned deterministic procedure and resized according to the official RETFound preprocessing.

Main local input: original-resolution 448-pixel tiles, 50% overlap, at least 80% retinal coverage. A 224-pixel tile setting is a scale ablation. Tile coordinates remain in original-image pixels. Cap or sample tiles only by a predeclared spatial rule, never by lesion labels.

## 5. Model-selection experiment

### Global SAE

Input dimension is 1,024. Search on EyePACS train/validation embeddings only:

- dictionary expansion: 4x, 8x, 16x;
- Top-K: 8, 16, 32;
- learning rate: 1e-4, 3e-4, 1e-3;
- pooling: CLS and mean-patch;
- AdamW, normalized decoder columns, train-only embedding normalization;
- dead-feature resampling enabled as a preregistered variant.

Use a two-stage search independently for each representation: screen 54 configurations for mean-patch and 54 for CLS (108 screening runs total), each with one seed. For each representation, retain three Pareto-optimal configurations using validation MSE, explained variance, cosine reconstruction, dead-feature fraction, and saturation, then rerun its three finalists with five fixed seeds. The clinical labels remain inaccessible during this choice.

### Local SAE

Train a separate dictionary on EyePACS tiles. Screen 448/224-pixel tiles, K in {16, 32}, and expansion in {4x, 8x}. Select using the same label-free validation criteria plus activation stability under small clinically preserving transformations.

### Fixed selection rule

Reject any run with failed reproducibility, more than the specified K activations, or more than 50% dead features on the large validation pool. Among remaining models, keep the smallest K model within one standard error of the best validation explained variance, then prefer lower dead-feature fraction. Freeze this rule before the search.

## 6. Baselines

All comparisons use the same split and activation/information budget where applicable.

1. Mean reconstruction and K-coordinate residual baseline.
2. PCA reconstruction with the same number of retained coefficients.
3. Sparse dictionary learning/K-SVD or sparse coding with matched K.
4. Dense autoencoder with matched dictionary width, to isolate the effect of sparsity.
5. DINOv2 encoder plus the same SAE, to isolate retina-specific pretraining.
6. Original versus reconstructed RETFound embeddings under a frozen linear DR probe, measuring downstream fidelity.
7. RETFound RELPROP/attention attribution for spatial localization.
8. Supervised lesion classifier or concept bottleneck as an explicitly labelled upper bound, not an unsupervised peer.
9. Reconstruction-only SAE versus rare-lesion transcoder as a secondary ablation.

## 7. Frozen interpretation protocol

Predefine the four primary concepts: hemorrhages, hard exudates, soft exudates, and microaneurysms. Microaneurysm claims are allowed only if visibility after preprocessing is verified by two reviewers on a blinded sample.

For each candidate neuron, using development data only:

- 50 highest activating examples;
- 20 medium-activation examples sampled from the middle activation quantiles;
- 50 matched negative controls, matched by dataset, camera when available, grade, and image quality;
- at least three hypotheses: intended lesion, plausible anatomical alternative, and technical confounder;
- automatic screening for borders, optic disc, fovea, vessels, brightness, blur, compression, watermark, and dataset source.

Limit confirmatory testing to a frozen candidate set: at most 25 global and 25 local neurons per lesion, selected without seeing locked-test labels. Save rejected and negative hypotheses.

## 8. Endpoints

### Representation endpoints

- per-dimension MSE and total explained variance;
- cosine reconstruction similarity;
- active features per sample, activation-magnitude distribution;
- dead and near-constant feature fractions;
- linear-probe prediction agreement and probability shift using original versus reconstructed embeddings.

### Image-level lesion association

- AUROC and AUPRC;
- sensitivity/specificity at a DDR-validation threshold;
- calibration plot for any aggregated concept score;
- patient-clustered 95% confidence intervals.

### Spatial localization

- pointing-game accuracy;
- tile-level average precision;
- Dice and IoU at a DDR-validation threshold;
- distance from maximum activation to the nearest lesion;
- lesion-size-stratified results, particularly for microaneurysms.

### Intervention evidence

For each mask-positive image, compare neuron activation on the original image, lesion inpainting/texture replacement, and matched random-region controls. Black rectangles are prohibited as the only intervention. Primary statistic: paired lesion-minus-control activation drop.

### Stability and confounding

- match neurons across five SAE seeds by decoder cosine similarity plus overlapping top-activation examples;
- report concept performance in every seed, not only the best seed;
- stratify by dataset/camera, image quality, lesion size, and DR severity;
- perturb brightness, contrast, blur, JPEG compression, small rotations, and small crops within clinically preserving ranges;
- quantify source predictability from each retained neuron and mark source-dominated features as confounded.

## 9. Statistical analysis

1. The patient/examination is the resampling unit; never bootstrap tiles independently.
2. Report 95% patient-cluster bootstrap confidence intervals with at least 2,000 resamples for AUROC, AUPRC, localization, and occlusion effects.
3. Use paired bootstrap differences for model/baseline comparisons on the same images.
4. Use a paired permutation test or Wilcoxon signed-rank test for lesion-versus-control occlusion drops after inspecting symmetry; report effect size and confidence interval.
5. Control the false-discovery rate at q=0.05 with Benjamini-Hochberg across the frozen neuron-concept family.
6. Report all five seeds individually and their mean/SD. Do not treat seeds as independent patients.
7. Give prevalence and positive/negative counts with every AUROC/AUPRC result.
8. Do not use test-set thresholds, neuron selection, or subgroup decisions.
9. Label analyses underpowered when a lesion has fewer than 100 positive external cases; soft-exudate and IDRiD-only analyses will likely be exploratory.
10. Publish a complete analysis manifest so every table cell maps to a checkpoint, split hash, and script invocation.

## 10. Expert evaluation

Recruit at least three ophthalmology reviewers. The confirmatory set contains a balanced, randomized, blinded sample of retained neuron descriptions, high activations, medium activations, and controls. Reviewers independently score description correctness, clinical meaningfulness, specificity, and visible confounding before adjudication.

Report raw reviewer decisions, percent agreement, Fleiss' kappa with patient-cluster bootstrap CI, and adjudicated results separately. No external multimodal API may receive retinal images without explicit governance approval.

## 11. Stage gates

1. **Access gate:** accepted dataset/model terms and completed data cards.
2. **Extraction gate:** ten-image RETFound parity, hashes, and 1,000-image throughput benchmark.
3. **Leakage gate:** zero cross-split patient or duplicate violations.
4. **SAE gate:** deterministic checkpoints, Top-K invariant, healthy feature population, and baseline comparison.
5. **Freeze gate:** final SAEs, candidate neurons, hypotheses, thresholds, and analysis code hashed before test access.
6. **External gate:** IDRiD/FGADR evaluated exactly once unless a logged software defect invalidates the run.
7. **Clinical gate:** blinded reviews complete and disagreements retained.
8. **Paper gate:** claims match registered endpoints, including negative results and failed features.

## 12. Figures and tables planned before results

- Figure 1: dual-scale pipeline and locked data flow.
- Figure 2: reconstruction/sparsity Pareto frontier across configurations.
- Figure 3: top/control examples and local activation maps for preregistered lesions.
- Figure 4: external association, localization, occlusion, and robustness forest plots.
- Figure 5: cross-seed feature matching and confounder failure cases.
- Table 1: datasets, access terms, patients/images, devices, roles, and split hashes.
- Table 2: representation and downstream-fidelity baselines.
- Table 3: confirmatory lesion results with multiplicity-adjusted q-values.
- Table 4: clinician agreement and adjudication.

## 13. Authoritative sources checked

- LanSE v4 and its CXR extension: https://arxiv.org/abs/2508.18236
- RETFound paper: https://www.nature.com/articles/s41586-023-06555-x
- RETFound code: https://github.com/rmaphoh/RETFound
- Official gated checkpoint: https://huggingface.co/YukunZhou/RETFound_mae_natureCFP
- EyePACS/Kaggle DR data: https://www.kaggle.com/c/diabetic-retinopathy-detection/data
- DDR paper: https://doi.org/10.1016/j.ins.2019.06.011
- DDR release: https://github.com/nkicsl/DDR-dataset
- IDRiD: https://idrid.grand-challenge.org/Data/
- FGADR: https://csyizhou.github.io/FGADR/
- APTOS: https://www.kaggle.com/c/aptos2019-blindness-detection/data
- Messidor-2: https://www.adcis.net/en/third-party/messidor2/
