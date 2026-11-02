# Experiment 7: Supervised Identity–Rank Fusion Results

Status: **complete**. RETFound and the K=1024, seed-4106 SAE remained frozen. Five neural seeds (4106–4110), both prespecified label budgets, both prediction tasks, all ablations, and 2,000 patient-cluster bootstrap comparisons completed. Heavy checkpoints and predictions are stored on the external drive.

## Primary 100%-label results

| Arm | Locked AUROC | Locked AUPRC | Locked QWK | IDRiD AUROC | IDRiD QWK |
|---|---:|---:|---:|---:|---:|
| Dense logistic | 0.8402 | 0.6653 | 0.4311 | 0.8095 | 0.3985 |
| Dense + sorted values logistic | 0.8358 | 0.6612 | 0.4354 | 0.8042 | 0.4415 |
| Value-only MLP | 0.8405 | 0.6613 | 0.4715 | **0.8315** | 0.4951 |
| Identity, no rank position | 0.8450 | **0.6696** | 0.4599 | 0.8095 | 0.5088 |
| Rank position, no identity | 0.8434 | 0.6649 | 0.4689 | 0.8217 | 0.5102 |
| Full identity–position Transformer | **0.8450** | 0.6676 | **0.4962** | 0.8168 | **0.5161** |

The full model improved locked-test AUROC over dense by only **0.0048** (95% CI −0.0056 to 0.0151; Holm-adjusted p=0.688). It therefore failed the predeclared overall-fusion requirement of an AUROC gain of at least 0.03 with an adjusted interval excluding zero.

The full model's AUROC was effectively identical to the identity-only model (difference 0.00004, 95% CI −0.00269 to 0.00260) and only 0.00159 above the position-only model (95% CI −0.00049 to 0.00374). The prespecified positional and identity AUROC-contribution criteria therefore also failed.

The five-grade result is more encouraging: full-model locked-test QWK was 0.4962 versus 0.4311 for dense, a difference of 0.0651 (95% CI 0.0399 to 0.0909). This supports improved ordinal grading in this cohort, but it is distinct from the failed primary binary-AUROC claim.

## Secondary 10%-label results

| Arm | Locked AUROC | Locked AUPRC | Locked QWK | IDRiD AUROC |
|---|---:|---:|---:|---:|
| Dense logistic | 0.7516 | 0.5238 | 0.3779 | **0.6805** |
| Dense + sorted values logistic | 0.7507 | 0.5229 | 0.3882 | 0.6296 |
| Value-only MLP | 0.7929 | 0.5688 | 0.3334 | 0.6056 |
| Identity, no rank position | 0.7907 | 0.5761 | 0.3523 | 0.6756 |
| Rank position, no identity | 0.7922 | 0.5807 | **0.4052** | 0.6565 |
| Full identity–position Transformer | **0.7956** | **0.5827** | 0.3714 | 0.6435 |

On the internal locked test, the full model improved over dense by **0.0441 AUROC** (95% CI 0.0241 to 0.0660; Holm-adjusted p<0.001). This is a real internal low-label signal. However, the full model did not significantly exceed the value-only MLP, identity-only Transformer, or position-only Transformer, so the gain cannot be attributed specifically to feature identity or positional embeddings.

The 10%-label signal did not transfer to IDRiD: full-model AUROC was 0.6435 versus 0.6805 for dense. It must therefore be reported as internal label-efficiency evidence, not external generalization.

## Predeclared conclusion

- Overall 100%-label fusion success: **not met**.
- Positional contribution criterion: **not met**.
- Feature-identity contribution criterion: **not met**.
- Exploratory/internal 10%-label improvement over dense: **observed**, but not externally reproduced.
- Five-grade QWK improvement at 100% labels: **observed** and suitable for follow-up, without claiming lesion identity or clinical deployment.

Machine-readable record: `/Volumes/Elements/Retina-LanSE/experiments/experiment_7_supervised_identity_rank_fusion_20260912/experiment_7_supervised_identity_rank_fusion.json`

Publication table: `/Volumes/Elements/Retina-LanSE/experiments/experiment_7_supervised_identity_rank_fusion_20260912/experiment_7_results_table.md`
