# Experiment 5C: ten-round extended feature rescue

Date: 2026-09-11

## Purpose

Re-evaluate the five cards that did not obtain a passing hypothesis under the
original bounded two-round protocol: 9180, 3322, 7388, 15184, and 12428. This
is an exploratory extension. It does not overwrite or retroactively change the
original two-round result.

## Frozen inputs

- Frozen K=1024 SAE, dictionary size 16,384, seed 4106.
- Frozen RETFound embeddings and train-only normalization.
- 71,134 unlabeled EyePACS training embeddings.
- Local Qwen2.5-VL 3B through Ollama, temperature 0.
- Original 32 high-activation discovery images for each card.

## Ten-round procedure

For each card and each of ten rounds:

1. Use the fixed original high-activation discovery images and a new
   low-activation control group to generate one contrastive hypothesis.
2. Use fresh high- and low-activation groups for validation; groups are
   patient-disjoint where feasible and are not reused between rounds.
3. Ask Qwen for strict JSON judgments of every image. Labels, grades, patient
   IDs, activation values, and all other feature cards are hidden.
4. Apply the original single-round criterion: at least 24 valid judgments per
   group, high pattern-present rate at least 0.60, and high-minus-low contrast
   at least 0.30.

No hypothesis or card from another feature is included in a card's prompt.
Afterward, exact duplicate hypothesis text is flagged as an audit field only;
it cannot alter pass/fail status.

## Stability endpoint

A card is labeled `stable_after_10_rounds` only if all ten rounds complete and
at least eight rounds pass. A card with fewer than eight passing rounds is
`not_stable_after_10_rounds`. Every generation failure, malformed response,
sample manifest, raw response, and prompt hash is retained.

The locked internal test, IDRiD, and any clinical endpoint are excluded from
discovery, hypothesis generation, and validation.

## Interpretation boundary

This tests repeatability of a same-model visual hypothesis under fresh
high/low samples. It does not establish lesion identity, DR specificity,
causality, or clinical utility. Independent retinal-specialist review remains
required.
