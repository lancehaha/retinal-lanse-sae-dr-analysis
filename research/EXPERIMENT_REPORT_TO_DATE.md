# Retina-LanSE experiment report to date

Date: 2026-09-05

## Scope

This report records the executed, reproducible portion of the retinal SAE study.
The encoder is a frozen RETFound-CFP representation (1,024 dimensions); only
the Top-K SAE and downstream analysis are trained or searched. All large data,
weights, embeddings, checkpoints, and response logs are on the external drive
under `/Volumes/Elements/Retina-LanSE`.

The decontaminated EyePACS publication cohort contains 88,694 images from
44,347 patients: 71,134 train, 8,800 validation, and 8,760 locked internal-test
images. Clinical referable DR means EyePACS grade >=2. Patient identity is the
grouping unit throughout.

### Protocol-stage mapping

The canonical `research/PUBLICATION_PROTOCOL.md` defines Stage 6 as the
one-time external evaluation gate: IDRiD/FGADR are evaluated only after the
analysis code, features, hypotheses, thresholds, and multiplicity rules are
frozen. The ablation is a separate comparison experiment that tests numerical
SAE scoring, AutoResearch feature selection, image-only VLM inference, and
image-plus-feature-card VLM inference under matched splits. The execution
ledger uses compact evidence-row numbering, where row 6 is software
verification; that row number should not be read as protocol Stage 6.

## Executed stages and findings

1. **Representation and SAE feasibility.** A fixed 16,384-feature, Top-K-32
   SAE was evaluated without changing the encoder. The exploratory validation
   AUROC values were 0.837 for dense RETFound embeddings, 0.795 for SAE
   reconstruction, 0.695 for PCA-32, and 0.637 for the raw sparse code. These
   are pilot values, not a locked clinical estimate.

2. **Numerical activation AutoResearch.** Four prespecified activation rules
   were compared on a patient-disjoint development split. The weighted mean
   difference over the top 64 eligible features was selected. A subsequent
   bounded interaction search tested pairwise co-activation and product rules;
   neither exceeded the additive rule. The selected rule reached development
   AUROC 0.744 and the previously inspected validation pilot reached AUROC
   0.716 (patient-bootstrap 95% CI 0.686--0.745) and AUPRC 0.471 (CI
   0.421--0.523).

   Fixed-policy diagnostics were then recomputed on all 28,130 labeled train
   images and 3,476 labeled validation images. Using only train-derived score
   weights, validation AUROC was 0.716 (2,000-patient-bootstrap 95% CI
   0.687--0.744) and AUPRC 0.471 (CI 0.421--0.522). Mean score increased by
   reference grade (0.171 for grade 0, 0.190 for grade 1, 0.307 for grade 2,
   0.670 for grade 3, and 1.003 for grade 4). Train-only Youden thresholding
   gave validation sensitivity 0.558 and specificity 0.766; these operating
   points are descriptive and were not used to open the locked test.

3. **Label-agnostic feature atlas.** All 71,134 train embeddings and all
   16,384 SAE features were audited. 6,763 features met the minimum activation
   count and 512 feature cards were written with 32 high-activation examples
   each. Labels were not used to rank cards; label-side DR rates are stored only
   as an audit field.

4. **Language grounding pilot.** A local Qwen2.5-VL 3B model was given four
   high-activation images for each of four feature cards, with DR labels and
   patient information hidden. All four responses parsed as JSON. The model
   produced uncertain visual hypotheses, including vessel/disc and macular
   patterns, but these are not clinically validated concepts. The post-hoc
   audit found three near-neutral activation associations and one negative
   association; no card was clearly positive.

   The run was expanded to 16 cards with two examples per card. All 16
   responses parsed. The expanded post-hoc audit found 12 near-neutral, four
   negative, and zero clearly positive activation associations. The repeated
   vessel-oriented descriptions and universal uncertainty flag show why these
   cards must not be treated as validated retinal lesion concepts.

   A separate same-model consistency check asked the VLM whether each proposed
   pattern was visible in the 64 packet images. It parsed 64/64 responses and
   marked all 64 present. Because the same model generated and checked the
   hypotheses, this 100% result is circular consistency evidence only and is
   explicitly not semantic validation.

5. **LLM inference ablation pilot.** The same local model received eight
   deterministic, balanced validation images one at a time. Reference labels
   were attached only after each response was persisted. Every response returned
   referable probability 0.3, yielding AUROC 0.500 and AUPRC 0.500. Therefore
   this model did not improve prediction in the pilot, and the result was not
   tuned or selectively removed. An image-plus-active-feature-card arm was then
   run on the same eight images. It had 1.25 grounded cards per image on average
   and reached AUROC 0.125/AUPRC 0.500. This is also a negative pilot, not a
   claim that feature cards are intrinsically harmful.

   The matching image-only and image-plus-card arms were repeated on 64 balanced
   validation images using deterministic in-memory resizing to 560 pixels. The
   image-only arm remained at AUROC/AUPRC 0.500/0.500; the feature-card arm
   reached 0.475/0.522 with 1.47 grounded cards per image on average. These
   numbers are still exploratory balanced-sample pilots, not clinical estimates.
   A paired patient-bootstrap comparison estimated a feature-card minus
   image-only AUROC difference of -0.025 (95% CI -0.153 to 0.104) and AUPRC
   difference of +0.022 (CI -0.046 to 0.130).

   A larger deterministic balanced 128-image repetition completed 128/128
   responses in both arms. Image-only remained at AUROC/AUPRC 0.500/0.500;
   image-plus-feature-card reached 0.469/0.503 with 1.51 grounded cards per
   image. The paired difference was -0.031 AUROC (95% CI -0.122 to 0.058) and
   +0.003 AUPRC (CI -0.043 to 0.073), across 127 patients.

   A checkpointed 256-image repetition then completed 256/256 responses in both
   arms on the same balanced validation design. Image-only inference remained
   at AUROC/AUPRC 0.500/0.500, while active feature-card context reached
   0.446/0.480 with 1.47 grounded cards per image. Across 243 patients, the
   paired feature-card-minus-image-only difference was -0.054 AUROC (95% CI
   -0.117 to 0.009) and -0.020 AUPRC (CI -0.045 to 0.029). This strengthens the
   negative pilot evidence but remains an internal balanced VLM experiment, not
   clinical validation.

   To control for prompt framing, active and random feature-card arms were
   rerun with identical neutral wording on the same 128 images and matched
   card-count distribution. Active cards reached AUROC/AUPRC 0.470/0.508;
   random cards reached 0.518/0.547. The paired active-minus-random difference
   was -0.048 AUROC (95% CI -0.156 to 0.058) and -0.039 AUPRC (CI -0.125 to
   0.054). This control provides no evidence that SAE-selected card identity
   improves prediction and exposes sensitivity to prompt framing.

   A deterministic balanced 1,024-image repetition completed both image-only
   and atlas-compatible active-card arms. Image-only inference remained at
   AUROC/AUPRC 0.500/0.500; the active-card arm reached 0.485/0.490 with
   1.423 grounded cards per image. Across 808 patients, the paired differences
   were -0.0147 AUROC (95% CI -0.0478 to 0.0171) and -0.0102 AUPRC (CI
   -0.0272 to 0.0127), using 5,000 patient-bootstrap resamples. A first
   1,024-image attempt used a 4,096-feature checkpoint with a 16,384-feature
   grounded atlas and was rejected because it produced zero mapped cards; the
   provenance audit records the rejection and the corrected rerun.

6. **Software verification.** The project virtual environment passed all 23
   repository unit tests. New runners compile successfully. AutoResearch and
   VLM pilots did not access the locked test.

7. **Frozen label-free internal test.** After the protocol freeze and a
   regenerated same-seed repeatability check, the locked internal split was
   opened once for the preregistered representation experiment. Mean-patch
   reconstruction over 8,760 images/4,380 patients achieved five-seed mean
   explained variance 0.9223 and cosine similarity 0.9572; CLS achieved 0.9736
   and 0.9856, respectively. Relative to matched-K PCA, the seed-averaged MSE
   difference was -0.1412 (95% CI -0.1438 to -0.1387) for mean-patch and
   -0.0489 (CI -0.0495 to -0.0482) for CLS. These are label-free reconstruction
   results, not DR AUROC or clinical validation.

## Artifact index

- Execution ledger: `research/EXPERIMENT_EXECUTION_LOG.md`
- Current status and gates: `research/EXPERIMENT_STATUS_REPORT.md`
- Feature atlas: `/Volumes/Elements/Retina-LanSE/experiments/feature_atlas_autoresearch/`
- Grounding responses: `/Volumes/Elements/Retina-LanSE/experiments/feature_grounding_qwen25vl_pilot_v3/`
- Expanded grounding responses: `/Volumes/Elements/Retina-LanSE/experiments/feature_grounding_qwen25vl_expanded16/`
- Grounding audit: `/Volumes/Elements/Retina-LanSE/experiments/feature_grounding_verification_qwen25vl_pilot_v3/`
- VLM predictions: `/Volumes/Elements/Retina-LanSE/experiments/vlm_inference_ablation_qwen25vl_pilot_v2/`
- VLM feature-card predictions: `/Volumes/Elements/Retina-LanSE/experiments/vlm_featurecard_ablation_qwen25vl_pilot/`
- Matching 64-image VLM predictions: `/Volumes/Elements/Retina-LanSE/experiments/vlm_inference_ablation_qwen25vl_64/` and `/Volumes/Elements/Retina-LanSE/experiments/vlm_featurecard_ablation_qwen25vl_64/`
- Paired VLM comparison: `/Volumes/Elements/Retina-LanSE/experiments/vlm_ablation_comparison_qwen25vl_64/`
- Matching 128-image VLM predictions: `/Volumes/Elements/Retina-LanSE/experiments/vlm_inference_ablation_qwen25vl_128/` and `/Volumes/Elements/Retina-LanSE/experiments/vlm_featurecard_ablation_qwen25vl_128/`
- Paired 128-image VLM comparison: `/Volumes/Elements/Retina-LanSE/experiments/vlm_ablation_comparison_qwen25vl_128/`
- 128-image VLM calibration/subgroup diagnostics: `/Volumes/Elements/Retina-LanSE/experiments/vlm_ablation_diagnostics_qwen25vl_128/`
- Matching 256-image VLM predictions and diagnostics: `/Volumes/Elements/Retina-LanSE/experiments/vlm_inference_ablation_qwen25vl_256/`, `/Volumes/Elements/Retina-LanSE/experiments/vlm_featurecard_ablation_qwen25vl_256/`, `/Volumes/Elements/Retina-LanSE/experiments/vlm_ablation_comparison_qwen25vl_256/`, and `/Volumes/Elements/Retina-LanSE/experiments/vlm_ablation_diagnostics_qwen25vl_256/`
- Neutral active/random card controls: `/Volumes/Elements/Retina-LanSE/experiments/vlm_active_neutral_ablation_qwen25vl_128/`, `/Volumes/Elements/Retina-LanSE/experiments/vlm_random_neutral_ablation_qwen25vl_128/`, and paired comparisons under `/Volumes/Elements/Retina-LanSE/experiments/vlm_active_vs_random_neutral_comparison_qwen25vl_128/`
- Clinician review templates and validator: `/Volumes/Elements/Retina-LanSE/experiments/clinician_review_templates_qwen25vl_expanded16/` and `/Volumes/Elements/Retina-LanSE/experiments/clinician_review_scoring_qwen25vl_expanded16/`
- Blinded clinician packet: `/Volumes/Elements/Retina-LanSE/experiments/clinician_review_packet_qwen25vl_expanded16/`
- Reviewer-ready blinded archive: `/Volumes/Elements/Retina-LanSE/experiments/clinician_review_handoff_qwen25vl_expanded16/clinician_review_handoff.zip` (64 images, 16 cards, three forms; labels, activation values, and analysis key excluded)
- Same-model hypothesis check: `/Volumes/Elements/Retina-LanSE/experiments/feature_hypothesis_verification_qwen25vl_expanded16/`
- Numerical interaction search: `/Volumes/Elements/Retina-LanSE/experiments/activation_interaction_autoresearch/`
- Numerical diagnostics: `/Volumes/Elements/Retina-LanSE/experiments/numeric_sae_diagnostics/`
- 256-image VLM ablation: `/Volumes/Elements/Retina-LanSE/experiments/vlm_ablation_comparison_qwen25vl_256/` and `/Volumes/Elements/Retina-LanSE/experiments/vlm_ablation_diagnostics_qwen25vl_256/`
- 1,024-image VLM ablation: `/Volumes/Elements/Retina-LanSE/experiments/vlm_ablation_comparison_qwen25vl_1024/` and `/Volumes/Elements/Retina-LanSE/experiments/vlm_ablation_diagnostics_qwen25vl_1024/`
- 1,024-image provenance audit: `/Volumes/Elements/Retina-LanSE/experiments/vlm_ablation_provenance_audit_qwen25vl_1024/provenance_audit.json`
- Frozen internal-test report: `/Volumes/Elements/Retina-LanSE/reports/EXPERIMENT_1_RESULTS.md`
- Completion audit: `/Volumes/Elements/Retina-LanSE/experiments/experiment_1_completion_audit.json`
- Release manifest: `/Volumes/Elements/Retina-LanSE/experiments/retina_lanse_release_manifest_20260905.json`
- Final package audit: `/Volumes/Elements/Retina-LanSE/experiments/release_audit_20260905.json` (`release_audit_passed`)
- Stage 6 gate record: `/Volumes/Elements/Retina-LanSE/experiments/stage6_external_gate_idrid_20260905.json` (`blocked_external_access`; no inference performed)

## What is not yet a publishable clinical result

The current work is a rigorously logged label-free reconstruction and
feasibility/ablation package, not a final diagnostic validation. The remaining
requirements are at least three ophthalmologist reviewers, a larger blinded
feature-card grounding set, and the canonical Stage 6 evaluation on approved
labeled external clinical tests after prompts, features, thresholds, and
multiplicity rules are frozen. The matched image-plus-feature-card inference,
calibration, subgroup, and neutral-card controls are already complete as
exploratory internal ablations. The EyePACS internal test is unlabeled for
this study and therefore cannot provide a DR AUROC.

The Stage 6 gate checker records the missing external data/access manifest and
the pending 0/192 clinician-review rows without attempting a download or model
run. This is a completion record for the local experiment, not an external
clinical result.
