# Experiment 5B: Contrastive SAE feature-hypothesis validation

This is a label-blind semantic follow-up to Experiment 5. The original 32 highest-
activation images for each feature are discovery-only and are never counted as validation.

## Sampling and hypothesis versions

For each eligible feature with a completed prior hypothesis, the initial validation uses
fresh high-activation ranks 33--64 and 32 deterministic low-activation controls. Low
controls are selected from the lowest positive activations (and use zero-activation images
only if fewer than 32 positive examples exist), so the test is not merely an active-versus-
absent check. Where feasible, discovery, high-validation, low-validation, revision, and
re-validation groups are patient-disjoint; exact paths and patient IDs are recorded for
audit. The validator sees two numbered contact sheets and returns a JSON
judgment for each image. A feature passes only when at least 24 images per group are
valid, high pattern-present rate is at least 0.60, and the high-minus-low rate is at least
0.30. This is a predeclared contrast criterion, not a clinical endpoint.

If the initial validation fails, exactly one revised hypothesis is permitted. It uses the
original discovery highs plus a new low-activation sample that was not used in either
initial validation group. Re-validation then uses new high and low samples disjoint from
all prior validation samples. No validation image is used to write a revised hypothesis.
Features with failed or malformed VLM responses remain failed/unavailable; they are not
silently discarded or repeatedly retried.

## Blinding and leakage controls

Labels, grades, patient IDs, activation values, and disease status are excluded from every
LLM prompt. Only the image groups and frozen text hypothesis are shown. The script loads
only the frozen K=1024, dictionary-size-16,384 SAE and the label-free training embedding
run. Locked-test and IDRiD inputs are not accepted. All outputs are constrained to
`/Volumes/Elements/Retina-LanSE/`.

## Interpretation

This tests whether a proposed visual pattern is more common in high-activation than low-
activation images. It does not prove lesion identity, DR specificity, causality, or clinical
utility. Same-model VLM agreement is an exploratory semantic check; independent retinal-
specialist review remains required for publication claims.

## Outputs

`contrastive_feature_validation.jsonl` stores every sample path, feature, hypothesis version,
raw response, validation rates, and failure. `contrastive_feature_validation_summary.json`
stores counts and input hashes. A smoke run tests deterministic tie-breaking, padding, and
the one-revision gate without loading images or labels.
For operational reliability, each 32+32 validation round is sent as two 16+16 requests
to the local Qwen model and then concatenated by image number. This preserves the planned
32-image groups while avoiding output truncation observed in the 3B model.
