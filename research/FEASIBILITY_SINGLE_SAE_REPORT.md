# Single-SAE retinal feasibility result

## Decision

The fixed Top-K SAE works as a representation-learning feasibility result, but the current sparse code is not yet a competitive referable-DR classifier.

- It reconstructs frozen RETFound mean-patch embeddings well: validation explained variance 0.9109, cosine similarity 0.9506, MSE 0.0892, with exactly 32 active features and 7.93% dead features.
- Its reconstructed 1,024-dimensional embeddings retain substantial referable-DR signal: AUROC 0.7949 (95% patient-bootstrap CI 0.7707–0.8182) and AUPRC 0.5415 (0.4909–0.5924).
- A direct linear probe on the 16,384-dimensional Top-K-32 codes is above chance but weak: AUROC 0.6368 (0.6150–0.6613) and AUPRC 0.2739 (0.2445–0.3057).
- Frozen dense RETFound remains stronger: AUROC 0.8368 (0.8143–0.8586) and AUPRC 0.5904 (0.5388–0.6393).
- The paired patient-bootstrap AUROC gap is -0.0419 (95% CI -0.0565 to -0.0285) for SAE reconstruction versus dense RETFound, and -0.2000 (-0.2236 to -0.1772) for direct sparse codes versus dense RETFound.

This supports a bounded conclusion: the SAE learns sparse retinal structure and preserves much, but not all, clinical signal through reconstruction. It does not support a claim that this SAE improves diagnosis or matches the dense encoder.

## Fixed protocol

- Encoder: pretrained RETFound CFP, frozen; `final_mean_patch`, 1,024 dimensions.
- SAE: one already-trained configuration; 16,384 features, Top-K 32, learning rate 3e-4, dead-feature resampling, seed 4106. No AutoResearch and no configuration selection from clinical labels.
- Label-free SAE training population: 71,134 EyePACS training-split images; clinical labels were not used to train the SAE.
- Clinical probe cohort: 28,130 labeled training images from 14,065 patients and 3,476 labeled validation images from 1,738 disjoint patients.
- Endpoint: referable diabetic retinopathy, EyePACS grade >=2; validation has 678 positive images.
- Probe: fixed class-balanced SGD logistic linear probe with train-fitted standardization and no hyperparameter tuning.
- Comparators: dense RETFound (1,024 dimensions), PCA-32, SAE reconstruction (1,024 dimensions), and direct Top-K sparse activations (16,384 dimensions).
- Uncertainty: 1,000 patient-cluster bootstrap resamples.
- Locked internal test: not accessed.

## Qualitative feature check

Four features were selected using training data only by the largest positive/negative activation-rate ratio, requiring at least 50 training activations. Their validation activation sets contained 88.9%–93.3% referable-DR images. Visual inspection shows coherent severe-retinopathy patterns, especially hard exudates for feature 2855 and exudate/vascular or treated-retina appearances for other features. These are exploratory image-level associations, not clinician-validated lesion concepts.

## Next bounded experiment

Before considering a wider search, repeat only this fixed configuration across three additional seeds and replace the generic sparse SGD probe with a prespecified, convergence-checked sparse logistic probe. Keep feature selection train-only, add blinded retinal-specialist labels for a small feature set, and reserve the locked internal test until that analysis specification is frozen.

## Artifacts

- Machine-readable results: `/Volumes/Elements/Retina-LanSE/experiments/feasibility_single_sae/feasibility_results.json`
- Feature montage: `/Volumes/Elements/Retina-LanSE/experiments/feasibility_single_sae/top_disease_associated_features.jpg`
- SAE checkpoint: `/Volumes/Elements/Retina-LanSE/experiments/paper_global_sae_screen_mean_patch/exp16_k32_lr3em04_resample1/checkpoint.pt`
- Evaluation source: `scripts/run_single_sae_feasibility.py`
