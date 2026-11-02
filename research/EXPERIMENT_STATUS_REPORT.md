# Retina-LanSE current experiment status

Date: 2026-09-11

## Executive result

The frozen retinal representation and SAE contain measurable referable-DR signal. A label-agnostic feature atlas and bounded AutoResearch comparison of additive and pairwise activation rules are complete; the additive weighted-mean activation rule remained best in the tested search. Local Qwen2.5-VL 3B grounding and blinded VLM ablations have now run. In the completed 1,024-image balanced arm, image-only inference returned AUROC/AUPRC 0.500/0.500, while atlas-compatible active feature-card context produced 0.485/0.490; the paired AUROC difference was -0.0147 (95% CI -0.0478 to 0.0171). These larger exploratory controls provide no evidence that SAE-selected card identity improves prediction. The separate label-free global reconstruction experiment has passed its freeze, one-time internal-test evaluation, and completion audit.

## Completed evidence

| Component | Evidence |
|---|---|
| EyePACS preparation | `/Volumes/Elements/Retina-LanSE/datasets/eyepacs/preparation_evidence.json`; 35,126 train and 53,576 test images |
| Fixed SAE feasibility | `/Volumes/Elements/Retina-LanSE/experiments/feasibility_single_sae/`; validation pilot, not final test |
| Four-recipe activation search | `/Volumes/Elements/Retina-LanSE/experiments/activation_comparison_autoresearch/` |
| Label-agnostic feature atlas | `/Volumes/Elements/Retina-LanSE/experiments/feature_atlas_autoresearch/`; 71,134 embeddings, 6,763 eligible features, 512 cards |
| Pairwise activation search | `/Volumes/Elements/Retina-LanSE/experiments/activation_interaction_autoresearch/`; additive rule selected |
| Fixed numerical diagnostics | `/Volumes/Elements/Retina-LanSE/experiments/numeric_sae_diagnostics/`; 28,130 labeled train and 3,476 labeled validation images; AUROC 0.716 (95% CI 0.687–0.744), AUPRC 0.471 (CI 0.421–0.522) |
| Local VLM feature grounding | `/Volumes/Elements/Retina-LanSE/experiments/feature_grounding_qwen25vl_expanded16/`; 16/16 structured hypotheses; labels hidden during generation |
| Post-hoc feature audit | `/Volumes/Elements/Retina-LanSE/experiments/feature_grounding_verification_qwen25vl_expanded16/`; 12 near-neutral, 4 negative, 0 clearly positive association cards |
| Blinded clinician packet | `/Volumes/Elements/Retina-LanSE/experiments/clinician_review_packet_qwen25vl_expanded16/`; 16 cards, 64 images, labels and activation values excluded |
| Same-model hypothesis consistency | `/Volumes/Elements/Retina-LanSE/experiments/feature_hypothesis_verification_qwen25vl_expanded16/`; 64/64 responses parsed, 100% pattern-present; treated as circular consistency only, not semantic validation |
| Blinded image-only VLM ablation | `/Volumes/Elements/Retina-LanSE/experiments/vlm_inference_ablation_qwen25vl_128/`; 128 images; AUROC/AUPRC 0.500/0.500 (8- and 64-image pilots also retained) |
| Blinded image+feature-card VLM ablation | `/Volumes/Elements/Retina-LanSE/experiments/vlm_featurecard_ablation_qwen25vl_128/`; 128 matched images; AUROC/AUPRC 0.469/0.503; 1.51 grounded cards/image on average |
| Paired VLM comparison | `/Volumes/Elements/Retina-LanSE/experiments/vlm_ablation_comparison_qwen25vl_128/`; AUROC difference -0.031 (95% CI -0.122 to 0.058), AUPRC difference +0.003 (CI -0.043 to 0.073) |
| Larger 256-image VLM ablation | `/Volumes/Elements/Retina-LanSE/experiments/vlm_ablation_comparison_qwen25vl_256/`; active-card AUROC/AUPRC 0.446/0.480 versus image-only 0.500/0.500; paired AUROC difference -0.054 (95% CI -0.117 to 0.009) across 243 patients |
| 1,024-image VLM ablation | `/Volumes/Elements/Retina-LanSE/experiments/vlm_ablation_comparison_qwen25vl_1024/`; atlas-compatible active-card AUROC/AUPRC 0.485/0.490 versus image-only 0.500/0.500; paired AUROC difference -0.0147 (95% CI -0.0478 to 0.0171) across 808 patients |
| Contrastive SAE feature-hypothesis validation | `/Volumes/Elements/Retina-LanSE/experiments/experiment_5b_contrastive_feature_validation_20260910/`; all 256 requested cards now have records. Original 143-card run: 84 initial passes, 56 revised passes, 3 final failures/unavailable. Recovery of the 113 previously unparsed cards: 99 initial passes, 12 revised passes, 2 final failures; coverage audit has no missing/duplicate IDs and zero sample-audit errors. Same-model VLM consistency only; no lesion or clinical-validity claim. |
| Ten-round extended feature rescue | `/Volumes/Elements/Retina-LanSE/experiments/experiment_5c_extended_feature_rescue_10rounds_20260911/`; cards 9180, 3322, 7388, 15184, and 12428 each completed 10/10 rounds and passed the >=8/10 stability rule. Two exact duplicate hypothesis groups were flagged. Exploratory same-model consistency only; original two-round failures remain unchanged. |
| VLM calibration/subgroups | `/Volumes/Elements/Retina-LanSE/experiments/vlm_ablation_diagnostics_qwen25vl_1024/`; fixed Brier/ECE, patient bootstrap, grade/quality/uncertainty/card-count strata; earlier 128/256 diagnostics retained |
| Random-card and neutral-framing controls | `/Volumes/Elements/Retina-LanSE/experiments/vlm_active_neutral_comparison_qwen25vl_128/`, `/Volumes/Elements/Retina-LanSE/experiments/vlm_random_neutral_comparison_qwen25vl_128/`, `/Volumes/Elements/Retina-LanSE/experiments/vlm_active_vs_random_neutral_comparison_qwen25vl_128/`; no active-selection benefit detected |
| Clinician review handoff | `/Volumes/Elements/Retina-LanSE/experiments/clinician_review_handoff_qwen25vl_expanded16/clinician_review_handoff.zip`; 64 images, 16 cards, and three blinded forms; scoring status `pending_clinician_reviews` (0/192 rows complete) |
| Frozen internal-test reconstruction | `/Volumes/Elements/Retina-LanSE/reports/EXPERIMENT_1_RESULTS.md`; mean-patch and CLS five-seed evaluations complete on 8,760 images/4,380 patients; label-free only |
| Reproducibility checks | 20 repository unit tests pass; new runners compile successfully |
| Release manifest and package audit | `/Volumes/Elements/Retina-LanSE/experiments/retina_lanse_release_manifest_20260905.json` and `/Volumes/Elements/Retina-LanSE/experiments/release_audit_20260905.json`; hashes, statuses, and cross-artifact checks pass |

## Interpretation

The atlas deliberately includes prominent features that are common in DR and non-DR images. It does not claim that a feature is a unique DR lesion. The numerical AutoResearch score is a group-association result; it is not a diagnostic probability and does not establish biological causality. The VLM’s feature descriptions are similarly unvalidated hypotheses; two cards in the first run were truncated at the context limit, so the shorter four-example rerun is the parseable evidence.

## Completion audit

### Requirement status

| Requirement | Evidence/status |
|---|---|
| External-drive storage and EyePACS decontamination | Complete; release manifest verifies heavy outputs are external-drive-only. |
| Frozen encoder and reproducible SAE | Complete for the current mean-patch SAE; encoder was not trained or fine-tuned. |
| Label-agnostic feature atlas and AutoResearch search | Complete bounded search; selected additive rule and 512-card atlas are recorded. |
| Image-only versus feature-card ablation | Complete internal 1,024-image matched arm; no reliable improvement demonstrated. |
| Calibration and subgroup diagnostics | Complete exploratory post-response analysis; balanced-sample caveat retained. |
| Card-selection/framing control | Complete neutral-wording active-versus-random comparison; result is null and framing-sensitive. |
| Contrastive feature-hypothesis validation | Complete for all 256 requested cards, including bounded recovery of the 113 previously unparsed cards; consolidated coverage/sample audit passes and all raw responses are on the external drive. Same-model consistency only; independent specialist validation remains pending. |
| Extended hypothesis rescue | Complete as an exploratory 10-round extension for the five requested cards; all 50 generations parsed and all five met the fixed stability rule. This does not establish lesion semantics or clinical validity. |
| Clinician review preparation | Complete blinded archive and scoring validator; actual reviewer judgments are still missing. |
| Independent semantic feature validation | Pending; blinded packet is ready, but at least three ophthalmology reviewers have not submitted scores. |
| One-time external clinical evaluation (protocol Stage 6) | Gated; official IDRiD/FGADR access is not present. |
| Stage 6 gate audit | Complete as a fail-closed check; no external inference performed. | `/Volumes/Elements/Retina-LanSE/experiments/stage6_external_gate_idrid_20260905.json` |
| Locked internal test | Complete once after the label-free freeze; completion audit passed for both representations. This remains a reconstruction result, not a clinical DR test. |

### External validation access gate

The official access pages were checked. IDRiD directs researchers to IEEE
Dataport for the complete labeled set; FGADR requires a signed non-commercial
research-use agreement and an emailed request before a download link is issued.
Neither archive is present in this workspace, so no external clinical result is
claimed. The exact gate and required evidence are recorded in
`research/EXTERNAL_VALIDATION_ACCESS_GATE.md`.

### Next actions

1. Expand blinded grounding beyond the 16-card pilot, with a prespecified image budget and clinician adjudication; no card is accepted as a lesion concept from the current VLM output alone.
2. Treat the 1,024-image arm as exploratory and seek a clinician-reviewed, externally governed validation sample before making any clinical inference.
3. Freeze feature cards, prompts, thresholds, and multiplicity rules.
4. Obtain approved labeled external data and evaluate the frozen clinical policy once under canonical protocol Stage 6, then publish the complete audit trail.

The Stage 6 checker is `scripts/check_stage6_external_gate.py`. Its current
record is `blocked_external_access`: the IDRiD directory, approved access
record, and verified manifest are absent, and the independent clinician packet
is still 0/192 complete. This is an intentional stop before any external
inference, not a failed model result.

Once the gate passes, the one-time evaluator is
`scripts/run_stage6_external_evaluation.py`. It is frozen to the train-derived
feature set/threshold and will refuse to overwrite or repeat an external run.

The EyePACS internal test has been consumed once for the label-free reconstruction experiment. The canonical external clinical test remains gated. If an external model is used, image-governance approval and explicit endpoint configuration are required; no images have been sent externally in the current run.

The exact handoff and data requirements are summarized in
`research/NEXT_REQUIRED_INPUTS.md`.

The consolidated reader-facing report is
`/Volumes/Elements/Retina-LanSE/reports/RETINA_LANSE_CONSOLIDATED_REPORT_20260905.md`;
it is generated from the audited JSON artifacts rather than hand-entered
metrics.
