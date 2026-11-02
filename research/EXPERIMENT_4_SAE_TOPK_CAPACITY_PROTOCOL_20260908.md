# Experiment 4: SAE activation-capacity sweep

**Status:** implementation complete; smoke gate passed; full execution is in
progress on the external drive.

## Objective

Test whether the neutral Experiment 3 fusion result was caused by the SAE's
per-image activation budget. The frozen RETFound-CFP encoder is held fixed while
new unsupervised SAEs are trained with Top-K values **32, 64, 128, 256, 512,
1024**. Top-K=32 is the matched reference. The 16,384-feature dictionary and
1,024-dimensional input remain fixed.

## Label-free SAE training

Each Top-K value is trained for five deterministic seeds (4106--4110) on the
71,134 full EyePACS train embeddings. The existing train-only normalization is
reused. Training uses AdamW, learning rate 3e-4, weight decay 1e-5, batch size
256, 20 epochs, decoder-column normalization, and deterministic dead-feature
resampling. No DR labels are used for SAE fitting or reconstruction selection.
Validation reconstruction is monitored on the existing validation embedding
run. Every checkpoint records reconstruction MSE, explained variance, cosine
similarity, mean/max active count, dead/saturated fractions, hashes, elapsed time,
and reload-determinism gates.

## Downstream representations and models

The SAE branch uses the **full 16,384-coordinate sparse code**, preserving
dictionary-feature identity. A Top-K=K model produces at most K nonzero values
per image; increasing K therefore changes the code itself rather than appending
duplicate or magnitude-sorted values. The four arms are:

1. frozen dense RETFound embedding (1,024 values);
2. full sparse SAE code (16,384 coordinates);
3. dense plus full sparse code;
4. SAE-reconstructed 1,024-dimensional embedding.

For every arm and Top-K, a class-balanced logistic probe uses `C=1`, `saga`,
`max_iter=2000`, fixed `tol=1e-2`, and train-only standardization. The
tolerance is predeclared as a CPU-execution setting after the pilot timing
check; it does not change the solver family, regularization, or iteration cap.
Sparse branches use
`StandardScaler(with_mean=False)`. The same head is used across the sweep; the
absolute scores are not compared with Experiment 3's `lbfgs` probes.

## Data and endpoints

Downstream models use 100% of the labeled EyePACS training cohort as the primary
analysis and a fixed 10% patient-level subset as a secondary low-label analysis.
Validation and the locked internal test remain fixed. The clean external IDRiD
testing cohort contains 102 images after excluding the documented exact duplicate;
its labels are unseen until evaluation. The primary endpoint is binary referable
DR (grade >=2). Secondary endpoints are AUPRC, Brier, 10-bin ECE, sensitivity,
specificity, PPV, NPV, and five-grade macro-AUROC, macro-AUPRC, quadratic weighted
kappa, macro-F1, and ordinal MAE.

Five SAE/downstream seeds are retained per Top-K. Ensemble probabilities are the
mean over seeds. EyePACS uncertainty uses 2,000 patient-cluster bootstrap
resamples; IDRiD uses image-examination proxy resampling because verified patient
IDs are unavailable. Paired dense+SAE minus dense differences use identical test
images. Holm correction covers the five K>32 internal comparisons; K=32 remains
the matched reference.

## Success criteria and scope

- Internal success: any K>32 fusion improves locked-test AUROC over dense by at
  least 0.03 with a Holm-adjusted paired 95% CI excluding zero.
- External success: the prespecified K=1024 fusion improves clean-IDRiD AUROC over
  dense with a paired CI excluding zero.
- Reconstruction gains without DR gains are reported as representation-fidelity
  gains only. Neutral/worse fusion across K supports the conclusion that the
  current limitation is not merely the previous 64-column bottleneck.

No lesion-semantic, specialist-review, or clinical-deployment claim is in scope.

## Reproducibility and storage

The runner is `scripts/run_sae_topk_capacity_sweep.py`. Heavy outputs, checkpoints,
curves, partial state, and final JSON are written only under:

`/Volumes/Elements/Retina-LanSE/experiments/experiment_4_sae_topk_capacity_sweep_20260908/`

The result JSON contains all Top-K/seed records, input hashes, model settings,
bootstrap intervals, Holm-adjusted comparisons, failure/unavailable records, and
timings. A one-epoch small-data smoke gate must pass before the full run.
