# Experiment 5B recovery: unparsed feature-card hypotheses

This bounded follow-up processes the 113 feature cards whose original Qwen hypothesis
response was unparsed. It does not modify the completed 143-card experiment.

For each card, Qwen receives the original 32 strongest-activation discovery images plus
32 newly sampled low-activation controls. Labels, grades, patient IDs, activation values,
and disease status are hidden. The model returns a concise contrastive hypothesis in a
small JSON schema. A malformed response is retried once with the same images and then
recorded as unavailable.

Recovered hypotheses are validated on fresh 32-high/32-low groups, sent as two 16+16
requests to avoid output truncation. The same pass criterion is used: at least 24 valid
judgments per group, high pattern-present rate >=0.60, and high-minus-low rate >=0.30.
One revision is allowed after a valid initial failure, using the original highs and a new
low group, followed by a fresh re-validation. No unlimited hypothesis search is allowed.

All outputs remain under `/Volumes/Elements/Retina-LanSE/experiments/` and are stored
separately from the original results. The procedure is a same-model VLM consistency check,
not independent lesion validation or a DR performance result.
