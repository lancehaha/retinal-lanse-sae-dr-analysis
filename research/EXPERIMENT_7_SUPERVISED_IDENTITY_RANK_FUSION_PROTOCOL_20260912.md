# Experiment 7: supervised identity-rank fusion

**Protocol date:** 2026-09-12  
**Status:** implemented; smoke verification precedes the full five-seed run.

## Question and representation

This experiment tests whether supervised nonlinear fusion of frozen RETFound
embeddings and frozen K=1024 SAE activations improves DR prediction, and whether
activation-rank positional embeddings contribute beyond feature identity.

Every image supplies 1,024 dense values and up to 1,024 sorted
`(feature_id, activation)` pairs. Feature ID is categorical and enters only
through an embedding table; it is never treated as a scalar measurement.

The model uses one dense token and 64 SAE tokens. Each SAE token attention-pools
16 consecutive activation ranks. Entry representations combine an activation
projection, optional 16,384-entry feature-identity embedding, and optional
within-group rank embedding. Optional group-rank embeddings locate the 64 SAE
tokens. `[CLS; dense; 64 SAE]` therefore has 66 tokens.

The fixed encoder has dimension 128, four pre-layer-normalized Transformer
layers, eight heads, feed-forward width 512, GELU, and dropout 0.1.

## Arms and training

1. Dense logistic baseline.
2. Dense plus sorted activation-value logistic baseline.
3. Two-layer value-only MLP without IDs or positions.
4. Identity-only Transformer without rank embeddings.
5. Position-only Transformer without feature identities.
6. Full identity-position Transformer.

RETFound and SAE checkpoints are frozen. Primary training uses all EyePACS
labels; the existing deterministic 10% patient subset is secondary. Five seeds
(4106--4110) are used. Binary and five-grade models are trained separately with
weighted BCE and class-balanced cross-entropy.

AdamW uses learning rate `1e-4`, weight decay `1e-4`, batch size 64, gradient
clip 1.0, and a 50-epoch cap. A patient-disjoint inner 90/10 split selects an
epoch with patience eight. The median epoch across five seeds is frozen for
each arm/task/budget, after which all five seeds are reinitialized and trained
on the full available training subset. Official validation, locked test, and
IDRiD do not select architecture or training duration.

## Statistics, interpretation, and outputs

Report the Experiment 6 binary and five-grade metrics, individual seeds,
ensembles, checkpoint hashes, and 2,000 paired patient-cluster bootstrap
differences. Holm correction covers the full model versus dense, value-only
MLP, no-position, and no-identity arms.

Overall success requires at least +0.03 locked-test AUROC over dense with an
adjusted interval excluding zero. Positional contribution requires at least
+0.01 AUROC over the no-position model with an adjusted interval excluding
zero. Identity contribution requires an adjusted interval above zero versus
the no-identity model. Improvement without a positional difference is
attributed to nonlinear fusion or identity, not positional embedding.

Implementation: `scripts/run_supervised_identity_rank_fusion.py`  
Heavy output:
`/Volumes/Elements/Retina-LanSE/experiments/experiment_7_supervised_identity_rank_fusion_20260912/`

Required gates cover trace ordering/alignment, padding invariance, categorical
feature IDs, patient separation, train-only scaling, checkpoint reload,
five-seed provenance, frozen specifications, and external-drive-only outputs.
This remains research prediction, not lesion confirmation or clinical deployment.
