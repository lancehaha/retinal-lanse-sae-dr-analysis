# Experiment 3: Label-Efficiency and External Generalization

**Status:** preregistered implementation executed with frozen representations; result artifact is written to the external experiment directory after completion.

## Question and hypothesis

This experiment tests whether sparse autoencoder (SAE) features become useful when fewer diabetic-retinopathy (DR) labels are available, despite the preceding full-data dense+SAE fusion comparison showing no gain. The hypothesis is that dense RETFound embeddings retain broad retinal information, while selected SAE activations may improve sample efficiency or external robustness.

## Data and frozen representations

- EyePACS labeled training, validation, and locked internal-test cohorts use the existing patient-level manifests and frozen `final_mean_patch` RETFound embeddings (1,024 dimensions).
- The RETFound checkpoint and the SAE checkpoint are frozen. Neither representation is retrained at any label budget.
- SAE activations and reconstructions are computed with the frozen normalization and frozen SAE checkpoint. The SAE dictionary has 16,384 features; the compact downstream arm uses 64 features selected within each training subset.
- External validation uses the official IDRiD testing partition. The raw 103-image testing partition is cleaned by excluding one exact SHA-256 duplicate of an official training image with conflicting metadata, yielding a clean 102-image test. Raw archives and the original manifest are preserved.

## Label budgets and sampling

The EyePACS labeled training cohort is subsampled at 1%, 5%, 10%, 25%, 50%, and 100% of patients. Each budget has five deterministic replicates with seeds `4106 + 1000*budget_index + replicate`. All images belonging to selected patients are retained. Sampling is stratified by patient maximum grade and attempts to retain all five image grades whenever the budget permits. Validation and locked internal-test cohorts remain fixed. Feature selection, standardization, and prediction heads are refit separately within every replicate.

## Prediction arms

1. **Dense:** class-balanced logistic probe on the 1,024-dimensional RETFound embedding.
2. **Compact SAE:** class-balanced logistic probe on 64 SAE activations selected by training-subset binary mean difference (minimum ten training activations).
3. **Dense + SAE (primary):** one probe on `[1,024 dense ; 64 SAE]` (1,088 dimensions).
4. **SAE reconstruction (secondary):** one probe on the 1,024-dimensional SAE reconstruction.

Every probe uses class-balanced logistic regression with `C=1`, `lbfgs`, `max_iter=2000`; standardization is fitted on that replicate's training subset only. The 100%-label models are trained only on EyePACS and then evaluated on clean IDRiD without tuning.

## Endpoints

The primary endpoint is binary referable DR, defined as EyePACS/IDRiD grade >=2. For validation, locked internal test, and clean IDRiD test, report AUROC, AUPRC, Brier score, and patient/image-examination-cluster bootstrap 95% intervals. Sensitivity, specificity, PPV, NPV, and calibration error are reserved for the locked reporting table when threshold provenance is available; no test-derived threshold is used.

Five-grade DR (grades 0--4) is secondary: macro-AUROC, macro-AUPRC, quadratic weighted kappa, macro-F1, and ordinal mean absolute error. A replicate is marked unavailable when its training subset cannot contain all five classes.

## External unit and leakage controls

EyePACS bootstrap resampling is patient-clustered. IDRiD has no verified patient identifiers in the official files, so external resampling is reported as an image-examination proxy and is not described as patient-level generalization. IDRiD labels are not used for tuning or feature selection. Locked-test and external results are opened only after model specifications are fixed. Extraction-evidence, manifest, label, normalization, SAE, and checkpoint hashes are stored in the machine-readable output.

## Statistical analysis and success criteria

For each budget and arm, five replicate probabilities are retained and averaged for the main ensemble comparison; individual replicate metrics are also retained. Paired bootstrap differences use identical evaluation images and 2,000 cluster resamples. Secondary budget comparisons are interpreted with Holm correction.

- **Internal label-efficiency success:** at 10%, dense+SAE exceeds dense AUROC by >=0.03 and the paired 95% interval excludes zero.
- **External success:** on clean IDRiD testing, the 100%-label dense+SAE model exceeds dense AUROC with an interval excluding zero.
- Smaller positive differences are incremental signals. Neutral fusion across budgets supports the conclusion that the current SAE does not improve full-data prediction or label efficiency.

## Scope boundary

This phase does not include retinal-specialist review, lesion grounding, or claims that a feature corresponds to a named lesion. Those require a separate interpretability experiment.

## Outputs

Heavy outputs are written only to:

`/Volumes/Elements/Retina-LanSE/experiments/experiment_3_label_efficiency_external_20260907/`

The runner writes `label_efficiency_external_benchmark.json`, containing cohort counts, sampling seeds, selected feature IDs, model/preprocessing provenance, validation/internal-test/IDRiD metrics, paired intervals, unavailable-five-class records, and exact input hashes.
