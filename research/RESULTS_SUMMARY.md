# Retina-LanSE internal results summary

Date: 2026-09-05

This is the concise interpretation of the executed internal experiment. It is a
feasibility and ablation package, not a clinical validation or deployment claim.

## What was fixed

- RETFound-CFP encoder: frozen; no encoder fine-tuning.
- SAE: 16,384 dictionary features, Top-K 32, fixed checkpoint.
- Primary endpoint: referable DR, EyePACS grade >=2.
- Patient-level grouping and bootstrap resampling.
- Heavy data, checkpoints, predictions, and manifests: external drive only at
  `/Volumes/Elements/Retina-LanSE`.
- Locked internal test: opened once after the protocol freeze; labels were not used.

## Main numerical results

| Arm | Images | Patients | AUROC | AUPRC | Interpretation |
|---|---:|---:|---:|---:|---|
| Dense frozen RETFound linear probe | 3,476 validation | 1,738 | 0.837 | 0.590 | Feasibility reference, not locked-test evidence |
| SAE reconstruction linear probe | 3,476 validation | 1,738 | 0.795 | — | Reconstruction retains measurable signal |
| Numerical AutoResearch score | 3,476 validation | 1,738 | 0.716 | 0.471 | Fixed additive activation association |
| VLM image-only | 128 balanced | 127 | 0.500 | 0.500 | Local Qwen2.5-VL 3B, labels hidden |
| VLM + active SAE cards, original wording | 128 balanced | 127 | 0.469 | 0.503 | No reliable improvement |
| VLM + active SAE cards, neutral wording | 128 balanced | 127 | 0.470 | 0.508 | No reliable improvement |
| VLM + random cards, neutral wording | 128 balanced | 127 | 0.518 | 0.547 | Null-control fluctuation; not a positive claim |
| VLM + active SAE cards, larger pilot | 256 balanced | 243 | 0.446 | 0.480 | Below image-only 0.500/0.500; no improvement |
| VLM + active SAE cards, 1,024-image matched arm | 1,024 balanced | 808 | 0.485 | 0.490 | Atlas-compatible checkpoint; paired difference remains null |
| Frozen mean-patch SAE reconstruction | 8,760 internal-test | 4,380 | EV 0.922 | cosine 0.957 | Five-seed label-free reconstruction; not a DR classifier |
| Frozen CLS SAE reconstruction | 8,760 internal-test | 4,380 | EV 0.974 | cosine 0.986 | Five-seed label-free reconstruction; not a DR classifier |

The neutral active-minus-random comparison was AUROC -0.048 (95% CI
-0.156 to 0.058) and AUPRC -0.039 (CI -0.125 to 0.054). The random-card
fluctuation demonstrates why card identity and prompt framing must be tested
together; it does not show that random text is clinically useful.

## Experiment 3: label efficiency and external generalization (2026-09-07)

Frozen RETFound and frozen SAE were evaluated with five patient-level replicates
at each EyePACS label budget. The table reports ensemble AUROC; “Fusion − dense”
is the paired difference on the same evaluation images.

| Labels | Dense internal | Dense+SAE internal | Fusion − dense | Dense IDRiD | Dense+SAE IDRiD | Fusion − dense |
|---:|---:|---:|---:|---:|---:|---:|
| 1% | 0.7756 | 0.7710 | −0.0046 | 0.7326 | 0.7216 | −0.0110 |
| 5% | 0.7956 | 0.7937 | −0.0019 | 0.8225 | 0.8405 | +0.0179 |
| 10% | 0.8203 | 0.8200 | −0.0004 | 0.8079 | 0.7945 | −0.0134 |
| 25% | 0.8276 | 0.8250 | −0.0026 | 0.7456 | 0.7436 | −0.0020 |
| 50% | 0.8375 | 0.8380 | +0.0004 | 0.8132 | 0.8075 | −0.0057 |
| 100% | 0.8397 | 0.8388 | −0.0009 | 0.8067 | 0.8087 | +0.0020 |

Primary conclusions: at 10% labels, AUROC difference was −0.00035 (95% CI
−0.00608 to 0.00535); on clean 102-image IDRiD at 100% labels, the difference
was +0.00204 (95% CI −0.01788 to 0.02138). Neither prespecified success
criterion was met. See the [Experiment 3 protocol](EXPERIMENT_3_LABEL_EFFICIENCY_EXTERNAL_PROTOCOL_20260907.md)
and [full results](EXPERIMENT_3_LABEL_EFFICIENCY_EXTERNAL_RESULTS_20260907.md).

## Published benchmark context

Published models provide useful context, but not a direct leaderboard comparison:

| Reference | Training/evaluation setup | Reported result | Comparison with ours |
|---|---|---:|---|
| RETFound (Nature, 2023) | RETFound fine-tuned on APTOS-2019, externally evaluated on IDRiD | AUROC **0.822** | Same broad dataset direction, but the encoder was fine-tuned and the task/reporting protocol differs |
| Gulshan et al. (JAMA, 2016) | Deep network trained on 128,175 expert-graded images; EyePACS-1 clinical validation | AUROC **0.991**; high-specificity sensitivity 90.3%, specificity 98.1% | Clinical upper reference, not comparable to our frozen RETFound probe or labels |
| Our Experiment 3 | EyePACS labels, frozen RETFound + logistic head, clean IDRiD testing | AUROC **0.8067** dense; **0.8087** dense+SAE | Valid paired SAE comparison; not a claim of state-of-the-art performance |

RETFound's paper reports AUROC 0.822 on IDRiD when fine-tuned on APTOS-2019 and describes AUROC/AUPR as its principal metrics ([Nature RETFound paper](https://www.nature.com/articles/s41586-023-06555-x)). The earlier EyePACS-1 clinical study used a much larger expert-graded development cohort and reported AUROC 0.991 ([JAMA EyePACS-1 study](https://jamanetwork.com/journals/jama/fullarticle/2588763)). These references show that our dense score is plausible for a frozen representation assay, but they cannot establish superiority or inferiority without matching training labels, fine-tuning, preprocessing, endpoint, and test cohort.

For a stronger paper claim, add a conventional same-protocol comparator (for example, frozen ImageNet EfficientNet/ResNet) and, if compute permits, a fully fine-tuned RETFound upper-bound arm. The current experiment already has the correct dense baseline for the narrower question “does SAE improve this frozen RETFound representation?”

## Experiment 5B: contrastive feature-hypothesis validation

The label-blind follow-up evaluated 143 of 256 cards with completed prior Qwen hypotheses
using fresh high- and low-activation controls; the other 113 cards are explicitly recorded
as unavailable. Eighty-four passed the initial contrast criterion; 57 received one
predeclared revision, of which 56 passed re-validation and 3 remained failed/unavailable.
The mean initial high-minus-low pattern-present contrast was 0.567;
the revised mean was 0.967. These are same-model VLM consistency rates, not independent
semantic validation or DR performance. Full per-feature records and hashes are in the
[Experiment 5B results](EXPERIMENT_5B_CONTRASTIVE_FEATURE_VALIDATION_RESULTS_20260910.md)
and the external output directory.

The bounded recovery then reran all 113 previously unparsed cards using the
contrastive high/low procedure. All 113 hypotheses became parseable; 99 passed
initial validation, 14 received the single permitted revision, 12 revised
hypotheses passed, and 2 remained failed. Together with the original 143-card
run, all 256 requested cards now have auditable records. This remains a
same-model Qwen consistency result, not lesion validation.

[Recovery results](EXPERIMENT_5B_UNPARSED_CARD_RECOVERY_RESULTS_20260911.md)

An exploratory ten-round extension then re-evaluated the five cards that had
failed twice or remained unparsed in the original two-round procedure. All five
completed ten rounds and met the fixed >=8/10 stability criterion, although two
exact duplicate hypothesis groups were detected (3322/7388 and 15184/12428).
This extension is reported separately and does not alter the predeclared
two-round result.

[Ten-round rescue results](EXPERIMENT_5C_EXTENDED_FEATURE_RESCUE_10ROUNDS_RESULTS_20260911.md)

## What the experiment supports

1. A frozen retinal embedding plus SAE reconstruction contains measurable
   referable-DR signal on the available development labels.
2. A label-agnostic activation atlas and bounded AutoResearch search are
   reproducible.
3. The current local VLM does not demonstrate a reliable gain from SAE feature
   cards in the matched internal ablation.
4. The generated card descriptions remain hypotheses; no feature has a
   clinically validated lesion meaning.

## What is still required

- At least three independent ophthalmology reviewers must return the blinded
  64-image feature packet before any semantic claim is made.
- The handoff is prepared at
  `/Volumes/Elements/Retina-LanSE/experiments/clinician_review_templates_qwen25vl_expanded16/`;
  the validator currently reports 0/192 completed reviewer rows.
- Licensed IDRiD and FGADR archives must be obtained through their official
  access terms for the canonical Stage 6 external evaluation.
- The fail-closed Stage 6 gate has been executed and recorded
  `blocked_external_access` at
  `/Volumes/Elements/Retina-LanSE/experiments/stage6_external_gate_idrid_20260905.json`;
  no external inference was performed.
- The current locked internal test is a label-free representation result; it
  does not substitute for the external clinical test.

## Evidence index

- Full report: `/Volumes/Elements/Retina-LanSE/reports/EXPERIMENT_1_RESULTS.md`
- Consolidated reader-facing report: `/Volumes/Elements/Retina-LanSE/reports/RETINA_LANSE_CONSOLIDATED_REPORT_20260905.md`
- Exploratory AutoResearch/VLM report: `research/EXPERIMENT_REPORT_TO_DATE.md`
- Execution ledger: `research/EXPERIMENT_EXECUTION_LOG.md`
- Status and completion audit: `research/EXPERIMENT_STATUS_REPORT.md`
- External access gate: `research/EXTERNAL_VALIDATION_ACCESS_GATE.md`
- Release manifest: `/Volumes/Elements/Retina-LanSE/experiments/retina_lanse_release_manifest_20260905.json`
- Full frozen SAE audit: `/Volumes/Elements/Retina-LanSE/experiments/experiment_1_full_audit_20260905.json`
- 1,024-image paired VLM comparison: `/Volumes/Elements/Retina-LanSE/experiments/vlm_ablation_comparison_qwen25vl_1024/`
- 1,024-image provenance audit: `/Volumes/Elements/Retina-LanSE/experiments/vlm_ablation_provenance_audit_qwen25vl_1024/`
- Final package audit: `/Volumes/Elements/Retina-LanSE/experiments/release_audit_20260905.json` (`release_audit_passed`)
- Stage 6 gate record: `/Volumes/Elements/Retina-LanSE/experiments/stage6_external_gate_idrid_20260905.json`
- Remaining external inputs: `research/NEXT_REQUIRED_INPUTS.md`
