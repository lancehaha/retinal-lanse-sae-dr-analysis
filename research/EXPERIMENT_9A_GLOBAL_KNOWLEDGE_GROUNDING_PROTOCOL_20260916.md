# Experiment 9A: Whole-Image Knowledge-Grounded Cards

Status: frozen before execution on 2026-09-16.

## Question

Can any of the existing 256 frozen, global K=1024 SAE cards be supported by
pixel-level IDRiD lesion masks, and does restricting additive card evidence to
those supported cards improve exploratory EyePACS DR prediction or make its
explanations safer?

RETFound (`final_mean_patch`) and the K=1024, seed-4106 SAE remain frozen.
No encoder, SAE, feature-card hypothesis, or EyePACS split is retrained or
changed. IDRiD segmentation data are knowledge-development data: after this
experiment they must not be called an untouched external validation cohort.

## IDRiD grounding

The official segmentation partitions are 54 training and 27 testing images.
For microaneurysms, haemorrhages, hard exudates, and soft exudates, derive
image-level presence and normalized mask area. Optic-disc area is retained as
an anatomical/confounder audit only.

For every card-lesion pair, on the 54-image development partition compute
activation-vs-presence AUROC/AUPRC, activation-vs-area Spearman association,
2,000 image bootstrap intervals, and BH correction of the area-association
p-values. Freeze the orientation from development and measure the same
quantities on the 27-image segmentation test partition.

A pair is `supported` only when all of the following hold:

- development BH q < 0.05;
- test AUROC >= 0.65;
- test AUPRC/prevalence >= 1.5;
- development and test activation orientation agree.

It is `strong` only when the lower 95% bootstrap AUROC is > 0.5 and the lower
95% bootstrap AUPRC-lift is > 1.0. Thresholds are never relaxed. Every card,
including unsupported cards, receives a `knowledge_card_v1` record. A global
association is never described as a spatial lesion location, lesion count,
causal relation, or ICDR grade.

IDRiD mask source: <https://idrid.grand-challenge.org/Data/>. ICDR
relationship context: <https://diabetesjournals.org/care/article-abstract/26/9/2653/22508>
and <https://iris.who.int/bitstream/handle/10665/334224/9789290227946-eng.pdf>.

## Prediction arms

On the previously examined EyePACS locked internal-test cohort (explicitly
exploratory):

1. frozen dense RETFound logistic reference;
2. existing all-card direct additive evidence;
3. direct evidence restricted to supported grounded cards;
4. seed-9206 size-matched random-card control;
5. Qwen using original cards;
6. Qwen using knowledge-grounded cards;
7. Qwen using shuffled lesion knowledge.

The additive evidence remains decomposable per active card. Qwen receives no
target image, target label, patient identity, or individual training example,
and may cite only active cards. Qwen work uses a shared GPU lock. If no
mapping satisfies the frozen support rule, arms 5--7 are explicitly
unavailable rather than manufacturing lesion language; the experiment ends as
a negative result.

## Success and boundaries

Prediction success requires grounded knowledge to improve AUROC by >= 0.02
over both original and shuffled controls with Holm-adjusted paired bootstrap
interval above zero. Interpretability success requires <=5% unsupported lesion
claims, faithful grounded citations, and AUROC non-inferiority within -0.03.

Outputs are restricted to:

`/Volumes/Elements/Retina-LanSE/experiments/experiment_9a_global_knowledge_grounding_20260916/`

They include segmentation matching, frozen embeddings/activations, all
feature-lesion statistics, cards, specification hash, exact input hashes,
prediction arrays, metrics, failures, and final report. Tests cover matching,
partition boundaries, BH/bootstrap behavior, deterministic shuffling, schema,
frozen SAE reload, global-input shape, and label-free LLM prompts.
