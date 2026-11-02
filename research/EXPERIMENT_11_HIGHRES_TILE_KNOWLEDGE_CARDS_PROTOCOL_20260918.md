# Experiment 11: High-resolution Tile-SAE Knowledge-Grounded Evidence Cards

## Why this is a new representation experiment

Experiment 9A found zero supported mappings from the frozen global SAE cards to
the IDRiD lesion masks. Experiment 9B found zero supported mappings from the
frozen 224x224 RETFound 14x14 Patch-SAE features to the same masks. The
predeclared grounding gates were not relaxed and no lesion language was
attached to either dictionary.

Experiment 11 therefore does not retrofit medical claims onto those failed
features. It retains frozen RETFound but learns a new, label-free SAE over
original-resolution local tiles. This follows the high-resolution tile design
already specified in `PUBLICATION_PROTOCOL.md`.

## Representation and data

- Frozen RETFound-CFP, applied independently to local tiles after its existing
  224x224 input preprocessing.
- The original retinal field is cropped once with `retinal-field-crop-v1`.
- Extract 448-pixel tiles with a 224-pixel stride (50% overlap), retaining only
  tiles with at least 80% retinal coverage. Tile coordinates remain in original
  image pixels.
- Initial exploratory cohort: the existing Experiment 8 patient-disjoint
  1,600/200/200 EyePACS split. No labels are read by tiling, normalization,
  SAE training, dictionary selection, or lesion-feature candidate selection.
- Train one Tile-SAE on `sae_train` tiles only: input 1,024, dictionary 16,384,
  Top-K 32, AdamW lr 3e-4, weight decay 1e-5, batch 1,024, 20 epochs,
  decoder normalization, deterministic dead-feature resampling, seed 4106.

## Gates

The preflight must prove deterministic tile coordinates, 448/224 geometry,
80% retinal-coverage filtering, tile-to-source-pixel mapping, no patient
overlap, external-drive-only output, frozen RETFound checkpoint parity, and
label absence through SAE fitting. The SAE requires explained variance >=0.85,
dead-feature fraction <=25%, and max active features <=32.

## Knowledge grounding and score experiment

After the Tile-SAE is frozen, feature--lesion grounding uses IDRiD lesion masks
at tile level with matched same-image lesion-negative tiles. Every association
is trained on the official IDRiD segmentation training split and tested on its
official testing split, with image-cluster bootstrap intervals and BH control.
Only supported mappings may receive lesion knowledge-base text. This makes
IDRiD knowledge-development data, not untouched external DR validation.

The prediction score is numerical. A train-only class-balanced L2 logistic model
uses dense RETFound values plus per-feature tile activation summaries. Its
conditional card contribution is retained per image. Knowledge channels are
added only in a real-knowledge arm and compared with a fixed shuffled-feature
knowledge control. Qwen receives the frozen score, high-resolution crops,
cards, and knowledge only to return a constrained explanation; it cannot alter
the score.

## Scope and success criteria

This is exploratory because the Experiment 8 pilot holdout has already been
accessed. The primary numerical comparison is real knowledge versus both
conditional numeric cards and shuffled knowledge. A knowledge benefit requires
an AUROC increase of at least 0.02 with paired patient-bootstrap 95% interval
above zero. A separate report-faithfulness gate requires supported mappings,
high-resolution evidence, and cited-feature deletion effects larger than
matched random active-feature deletions. No clinical or lesion-identity claim
is made without these gates and independent specialist review.

All heavy outputs must remain under:

`/Volumes/Elements/Retina-LanSE/experiments/experiment_11_highres_tile_knowledge_cards_20260918/`
