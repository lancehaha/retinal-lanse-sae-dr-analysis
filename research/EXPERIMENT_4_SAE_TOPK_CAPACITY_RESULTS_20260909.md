# Experiment 4: SAE activation-capacity sweep — results

**Status:** complete. The six Top-K settings (32, 64, 128, 256, 512, 1024)
were each trained with five seeds (4106–4110). All 30 checkpoints passed the
Top-K budget and reload gates. Heavy artifacts and the machine-readable result
are stored on the external drive at
`/Volumes/Elements/Retina-LanSE/experiments/experiment_4_sae_topk_capacity_sweep_20260908/sae_topk_capacity_sweep.json`.

## Primary internal locked-test result (100% labels)

The dense RETFound baseline was AUROC **0.8408** (AUPRC 0.6667). Fusion was
below the dense baseline at every activation capacity:

| Top-K | Sparse-code AUROC | Dense + sparse AUROC | Fusion ΔAUROC | SAE-reconstruction AUROC |
|---:|---:|---:|---:|---:|
| 32 | 0.7714 | 0.8070 | −0.0338 | 0.8058 |
| 64 | 0.8085 | 0.8267 | −0.0140 | 0.8247 |
| 128 | 0.8097 | 0.8251 | −0.0157 | 0.8314 |
| 256 | 0.8107 | 0.8238 | −0.0170 | 0.8381 |
| 512 | 0.8165 | 0.8276 | −0.0131 | 0.8410 |
| 1024 | 0.8286 | 0.8341 | **−0.0067** | **0.8439** |

For the prespecified K=1024 primary comparison, the paired bootstrap
95% CI for fusion minus dense was **[−0.0166, 0.0036]**, p=0.209. Therefore it
does not meet the predeclared internal success criterion (an improvement of at
least 0.03 with a CI excluding zero). Holm-adjusted comparisons for K=64–512
were also negative; K=1024 was not significant.

## Secondary low-label result (10% labels)

The dense baseline was AUROC **0.8164**. At K=1024, sparse code reached 0.7731,
fusion 0.7880, and SAE reconstruction 0.8249. Thus increasing capacity did not
produce a fusion gain in the low-label setting either.

## Clean IDRiD external test (102 images)

The dense baseline was AUROC **0.8087**. At the prespecified K=1024 setting,
sparse code reached 0.7623, fusion 0.7721, and SAE reconstruction 0.7908. The
paired fusion-minus-dense AUROC difference was **−0.0366**, with bootstrap 95%
CI **[−0.1108, 0.0356]**, p=0.308. The external success criterion was not met.

## Reconstruction capacity

Validation explained variance increased monotonically with Top-K, from 0.9112
(K=32) to 0.9901 (K=1024), while mean reconstruction MSE decreased from 0.0888
to 0.00995. Mean active features matched the requested budget (32, 64, 128,
256, 512, and approximately 1024), confirming that larger K values created
genuinely denser codes rather than appended zero coordinates.

## Conclusion

Larger activation budgets substantially improved SAE reconstruction fidelity and
made the sparse-code probe less weak, but they did **not** improve DR prediction
over the frozen dense RETFound embedding. The primary fusion result remains
neutral-to-worse internally and externally. This experiment therefore rejects
the explanation that the previous neutral result was caused only by a 64-feature
activation bottleneck; the limitation persists with a full 16,384-coordinate
code and up to 1,024 active features per image.

This conclusion concerns prediction only. It does not make lesion-semantic or
clinician-validation claims about individual SAE features.
