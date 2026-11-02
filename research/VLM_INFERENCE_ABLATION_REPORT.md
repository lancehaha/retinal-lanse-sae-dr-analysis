# Blinded VLM inference ablation

This ablation tests whether a vision-language model can add useful image-level
severity information to the numerical SAE analysis. The VLM receives only one
fundus image and a fixed prompt; the EyePACS grade is attached after the response
is persisted. The prompt therefore contains no DR label, patient identifier, or
split-specific prevalence information.

The executable runner is `scripts/run_vlm_inference_ablation.py`. It selects a
deterministic balanced sample from the validation split, records raw and parsed
responses, and computes AUROC/AUPRC only over successfully parsed probabilities.
All outputs are required to remain under `/Volumes/Elements/Retina-LanSE`.

This is a model-behaviour pilot, not a clinical validation. A paper-level claim
requires a prespecified larger sample, calibration analysis, subgroup analysis,
blinded ophthalmologist review, and one locked external evaluation after all
prompts and selection rules are frozen.

## Run record

The current execution record is maintained in `research/EXPERIMENT_EXECUTION_LOG.md`.
When a run completes, its external `vlm_inference_summary.json` path, model tag,
sample size, parse failures, and metrics must be copied into that log. If the VLM
returns malformed JSON or times out, the image remains in the audit trail and is
not silently dropped from the requested sample count.

## Completed pilot

The local Qwen2.5-VL 3B pilot used eight deterministic, balanced images from the
validation split. All eight responses parsed, but every response returned
`referable_probability = 0.3`, giving AUROC 0.500 and AUPRC 0.500. This is a
negative pilot result: the model was not tuned, thresholded, or selectively
reported after inspecting the labels. Full per-image responses are preserved at
`/Volumes/Elements/Retina-LanSE/experiments/vlm_inference_ablation_qwen25vl_pilot_v2/vlm_predictions.jsonl`.

A matched 64-image run used the same fixed prompt and in-memory 560-pixel
preprocessing. Image-only inference again produced AUROC/AUPRC 0.500/0.500.
Adding active grounded feature-card context produced AUROC/AUPRC 0.475/0.522,
with 1.47 grounded cards per image on average. Matching outputs are under
`/Volumes/Elements/Retina-LanSE/experiments/vlm_inference_ablation_qwen25vl_64/`
and `/Volumes/Elements/Retina-LanSE/experiments/vlm_featurecard_ablation_qwen25vl_64/`.
The paired patient-bootstrap comparison is preserved at
`/Volumes/Elements/Retina-LanSE/experiments/vlm_ablation_comparison_qwen25vl_64/`:
feature-card minus image-only AUROC difference -0.025 (95% CI -0.153 to 0.104)
and AUPRC difference +0.022 (CI -0.046 to 0.130).

## Larger matched arm

To reduce the instability of the 64-image pilot, a deterministic balanced
128-image validation arm was run with the same prompt, model, preprocessing,
feature-card set, and hidden-label protocol. The image-only arm completed
128/128 responses with AUROC/AUPRC 0.500/0.500. The matched image-plus-card arm
completed 128/128 responses, averaged 1.51 grounded cards per image, and reached
AUROC 0.469 and AUPRC 0.503. The paired patient-bootstrap difference (card minus
image-only) was -0.031 AUROC (95% CI -0.122 to 0.058) and +0.003 AUPRC (CI
-0.043 to 0.073), across 127 patients. These intervals include no improvement;
the result remains an internal balanced pilot, not a clinical validation.

Outputs are preserved at:

- `/Volumes/Elements/Retina-LanSE/experiments/vlm_inference_ablation_qwen25vl_128/`
- `/Volumes/Elements/Retina-LanSE/experiments/vlm_featurecard_ablation_qwen25vl_128/`
- `/Volumes/Elements/Retina-LanSE/experiments/vlm_ablation_comparison_qwen25vl_128/`

Post-response calibration and subgroup diagnostics are preserved at
`/Volumes/Elements/Retina-LanSE/experiments/vlm_ablation_diagnostics_qwen25vl_128/`.
The image-plus-card arm's patient-bootstrap AUROC interval was 0.378--0.552
and its five-bin ECE was 0.393, versus 0.500 and 0.200 for image-only. These
calibration values are descriptive on a deliberately balanced internal sample;
they must not be interpreted as clinical calibration.

Both runners now checkpoint each response to a `.partial.jsonl` file so a
single slow or malformed image cannot erase earlier evidence. The checkpointed
256-image repetition completed 256/256 responses in both arms. Image-only
inference remained at AUROC/AUPRC 0.500/0.500; active feature-card context
reached 0.446/0.480 with 1.47 grounded cards per image. The paired
feature-card-minus-image-only difference was -0.054 AUROC (95% CI -0.117 to
0.009) and -0.020 AUPRC (CI -0.045 to 0.029), across 243 patients. The
corresponding feature-card AUROC bootstrap interval was 0.385--0.507 and its
overall Brier/ECE were 0.449/0.416. This larger arm still remains a balanced
internal pilot, not a clinical validation estimate.

Outputs are preserved at:

- `/Volumes/Elements/Retina-LanSE/experiments/vlm_inference_ablation_qwen25vl_256/`
- `/Volumes/Elements/Retina-LanSE/experiments/vlm_featurecard_ablation_qwen25vl_256/`
- `/Volumes/Elements/Retina-LanSE/experiments/vlm_ablation_comparison_qwen25vl_256/`
- `/Volumes/Elements/Retina-LanSE/experiments/vlm_ablation_diagnostics_qwen25vl_256/`

## 1,024-image matched arm

A deterministic balanced 1,024-image validation arm was completed with the
same hidden-label protocol. Image-only Qwen2.5-VL inference returned
AUROC/AUPRC **0.500/0.500**. The provenance-valid SAE-card arm used the
16,384-feature checkpoint that generated the feature atlas, completed all
1,024 responses, averaged **1.423 grounded cards/image**, and returned
AUROC/AUPRC **0.485/0.490**. Across 808 patients, the paired feature-card
minus image-only differences were **-0.0147 AUROC** (5,000-patient-bootstrap
95% CI **-0.0478 to 0.0171**) and **-0.0102 AUPRC** (CI **-0.0272 to
0.0127**). The card arm's patient-bootstrap AUROC CI was **0.452--0.517**;
overall Brier/ECE were **0.436/0.403**. These results provide no evidence of
an accuracy gain and remain an internal balanced pilot, not clinical
validation.

An initial 1,024-image attempt was rejected before analysis: it used the
4,096-feature final reconstruction checkpoint while the grounded atlas used
16,384 features, yielding zero mapped cards. The rejected summary and the
accepted-run decision are recorded in
`/Volumes/Elements/Retina-LanSE/experiments/vlm_ablation_provenance_audit_qwen25vl_1024/provenance_audit.json`.

Accepted outputs are preserved at:

- `/Volumes/Elements/Retina-LanSE/experiments/vlm_inference_ablation_qwen25vl_1024/`
- `/Volumes/Elements/Retina-LanSE/experiments/vlm_featurecard_ablation_qwen25vl_1024_atlas16384/`
- `/Volumes/Elements/Retina-LanSE/experiments/vlm_ablation_comparison_qwen25vl_1024/`
- `/Volumes/Elements/Retina-LanSE/experiments/vlm_ablation_diagnostics_qwen25vl_1024/`

## Card-selection and prompt-framing control

The first random-card control used wording that explicitly called the cards
“distractors.” It reached AUROC/AUPRC 0.542/0.564, but that result is
framing-sensitive and is not interpreted as a benefit. To remove this confound,
the active and random arms were rerun on the same 128 images with identical
neutral wording: “Feature-card hypotheses provided as additional context.”

Under the neutral prompt, active SAE-selected cards reached AUROC/AUPRC
0.470/0.508, while random cards reached 0.518/0.547. The paired active-minus-
random differences were -0.048 AUROC (95% CI -0.156 to 0.058) and -0.039 AUPRC
(CI -0.125 to 0.054). Relative to image-only, neutral active cards changed
AUROC by -0.030 (CI -0.122 to 0.059), whereas neutral random cards changed it
by +0.018 (CI -0.066 to 0.101). All arms used the same card-count distribution
(mean 1.51 cards/image; 32/128 images had zero cards). These overlapping
intervals provide no evidence that SAE-selected card identity improves VLM
prediction; they instead expose sensitivity to prompt framing and the need for
clinician-validated semantics.

Neutral-control outputs are preserved at:

- `/Volumes/Elements/Retina-LanSE/experiments/vlm_active_neutral_ablation_qwen25vl_128/`
- `/Volumes/Elements/Retina-LanSE/experiments/vlm_random_neutral_ablation_qwen25vl_128/`
- `/Volumes/Elements/Retina-LanSE/experiments/vlm_active_neutral_comparison_qwen25vl_128/`
- `/Volumes/Elements/Retina-LanSE/experiments/vlm_random_neutral_comparison_qwen25vl_128/`
- `/Volumes/Elements/Retina-LanSE/experiments/vlm_active_vs_random_neutral_comparison_qwen25vl_128/`
- `/Volumes/Elements/Retina-LanSE/experiments/vlm_active_neutral_diagnostics_qwen25vl_128/`
- `/Volumes/Elements/Retina-LanSE/experiments/vlm_random_neutral_diagnostics_qwen25vl_128/`
