# Retina-LanSE

Retina-LanSE is a research-only sparse-autoencoder (SAE) pipeline for auditing retinal image embeddings. The synthetic **Phase A** smoke test remains only a software check; the repository also contains a bounded retinal feasibility study and an activation-comparison AutoResearch runner.

## Repository scope

This repository contains source code and small configuration files for three related research tracks:

| Track | Main code | Purpose |
| --- | --- | --- |
| Retina-LanSE | `retina_lanse/`, `scripts/run_global_sae_*.py` | Audit sparse features of frozen RETFound retinal embeddings. |
| RetinaVLM–OLIVES | `scripts/retinavlm_*.py`, `scripts/retinavlm_*.sbatch`, `scripts/fit_retinavlm_olives_jlens_map.py` | Fit and audit task-conditioned decoder readouts, then compare matched training arms. |
| FundusExpert J-Lens | `fundusexpert_jlens/` | Separate fundus report-supervision feasibility study; see its own README. |

No licensed retinal images, patient-level label files, model checkpoints, extracted activations, or cluster outputs are included. Obtain datasets and upstream models from their official sources under their respective terms. Local paths in some scripts describe the original research environment; adjust them for your own installation. No clinical deployment or clinical-utility claim is supported by this repository.

### RetinaVLM–OLIVES code path

The J-Lens workflow is implemented as separate, auditable steps:

1. Verify the official split and image files with `scripts/audit_retinavlm_official_six_biomarkers.py` and `scripts/audit_retinavlm_olives_hf_images.py`.
2. Fit a task-conditioned map with `scripts/fit_retinavlm_olives_jlens_map.py` and the cluster wrapper `scripts/retinavlm_jlens_map.sbatch`. The fitter requires a released RetinaVLM model, aligned adapter, image directories, and a patient-disjoint selection plan supplied outside Git.
3. Apply the unchanged statistical checks with `scripts/audit_retinavlm_olives_map_candidate.py`, then assess answer-state interventions with `scripts/retinavlm_map_intervention_v3.sbatch`. Only fully accepted maps may be finalized by `scripts/finalize_retinavlm_olives_map_v3.py`.
4. Compare any accepted map loss against matched baselines with `scripts/train_retinavlm_olives_abc.py` and the corresponding evaluation scripts. A map gate by itself is not evidence of downstream benefit.

The current seven-finding v3 gate accepted five maps (B1 using alternate layers, B3, B4, B5, and SRF). B2 and B6 remain failed candidates. A broader B2/B6 layer sweep on already inspected training patients is exploratory; a promising layer pair would require a new prospectively fixed protocol and independent expert-labeled patients before eligibility. The original validation and test patients are not used for map selection.

The synthetic Phase A remains a mechanics-only check and is not clinical evidence. The active publication experiment started from the complete externally stored EyePACS corpus (88,702 images). A label-free pre-selection near-duplicate gate excluded four holdout patients/eight images, yielding a patient-disjoint publication cohort of 88,694 images (71,134/8,800/8,760 train/validation/internal test) and a checksum-pinned, frozen RETFound-CFP encoder. The encoder is not fine-tuned during this experiment.

## Run the verified Phase A experiment

```bash
python3 -m pip install -e .
python3 scripts/run_phase_a.py --config configs/smoke_global_sae.yaml
```

The repository-wide test suite also imports optional research packages such as scikit-learn and OpenCV. Install the dependencies for the track you are testing before running `python3 -m unittest discover -s tests -v`; the minimal `pyproject.toml` covers the Phase A core only.

The run creates a patient-level split manifest, cached embeddings, a checkpoint containing the optimizer/configuration/hashes/seed/curves, metrics, and the versioned SQLite neuron-registry schema.

Key artifacts:

- `artifacts/data/synthetic_fundus/manifest.jsonl`
- `artifacts/embeddings/smoke_global_embeddings.pt`
- `artifacts/checkpoints/smoke_global_sae.pt`
- `artifacts/experiments/phase_a_global_sae_smoke/metrics.json`
- `artifacts/experiments/phase_a_global_sae_smoke/training_curves.csv`
- `artifacts/neuron_registry.sqlite`

## Data contract for a real campaign

Every cached whole-image record carries its relative path, patient, laterality, immutable patient-level split, dataset/source partition, dimensions, format, byte count, SHA-256, perceptual hash, and camera availability. The frozen RETFound adapter uses the official upstream architecture, reports state-dictionary compatibility, hashes the gated checkpoint, and fails closed on unapproved architecture drift.

Before any clinical claim, obtain datasets through their official access routes, record licences/data cards, validate RETFound-CFP extraction, preregister thresholds, and lock the test set. The activation-comparison and VLM runs are exploratory train/validation-pilot evidence and do not make a clinical-utility claim. The separate label-free global reconstruction experiment has completed its authorized one-time EyePACS internal-test evaluation; its audited report is `/Volumes/Elements/Retina-LanSE/reports/EXPERIMENT_1_RESULTS.md`.

## Bounded retinal feasibility and activation comparison

The fixed feasibility evaluation is implemented by `scripts/run_single_sae_feasibility.py`. It compares dense RETFound embeddings, PCA-32, SAE reconstruction, and raw Top-K sparse activations on a patient-disjoint EyePACS train/validation cohort. The report and montage are written externally under `/Volumes/Elements/Retina-LanSE/experiments/feasibility_single_sae`.

The LanSE-inspired activation comparison is implemented by `scripts/run_activation_comparison_autoresearch.py`. It keeps the SAE checkpoint fixed and searches four prespecified activation-comparison recipes on an 80/20 patient split inside labeled training data. The current pilot result is written externally under `/Volumes/Elements/Retina-LanSE/experiments/activation_comparison_autoresearch`.

The label-agnostic feature-atlas stage is implemented by `scripts/build_feature_atlas_autoresearch.py`. It audits every SAE feature on all 71,134 training embeddings, writes the most prominent feature cards with high-activation image paths, and attaches DR/non-DR statistics without using labels for feature ranking or language grounding. Its external output is `/Volumes/Elements/Retina-LanSE/experiments/feature_atlas_autoresearch`; most cards remain `hypothesis_status=pending_llm_or_clinician_review`, while a prespecified 16-card subset now has local VLM hypotheses recorded separately.

`scripts/run_feature_grounding.py` is the language-grounding runner. It supports an OpenAI-compatible or local Ollama vision-language endpoint, hides DR labels from the prompt, and writes structured hypotheses plus raw responses to an external output directory. The executed local pilots are under `/Volumes/Elements/Retina-LanSE/experiments/feature_grounding_qwen25vl_expanded16`; external endpoints still require explicit image-governance approval.

`scripts/run_vlm_inference_ablation.py` runs a blinded image-level VLM pilot on a balanced, patient-grouped development sample. It records model responses before attaching reference grades for AUROC/AUPRC, so the label cannot leak into the prompt. It checkpoints each response under the external output directory. This is an exploratory ablation and does not substitute for clinician adjudication or a locked external clinical test.

`scripts/run_vlm_featurecard_ablation.py` repeats that pilot with the active SAE feature-card hypotheses included as optional context. `scripts/build_clinician_review_packet.py` creates a label-blinded 64-image review packet for the 16-card grounding run; its output contains no DR labels or activation values. The complete execution ledger and evidence-bounded report are `research/EXPERIMENT_EXECUTION_LOG.md` and `research/EXPERIMENT_REPORT_TO_DATE.md`; the compact reader-facing summary is `research/RESULTS_SUMMARY.md`. The final package audit is `/Volumes/Elements/Retina-LanSE/experiments/release_audit_20260905.json`, and the consolidated report is `/Volumes/Elements/Retina-LanSE/reports/RETINA_LANSE_CONSOLIDATED_REPORT_20260905.md`.

`scripts/prepare_clinician_review_templates.py` creates three independent reviewer forms from that packet, and `scripts/score_clinician_reviews.py` validates returned forms and computes Fleiss' kappa and pairwise agreement. The current external-drive scoring artifact is intentionally `pending_clinician_reviews` until human judgments are returned.

`scripts/compare_vlm_ablations.py` computes the paired patient-bootstrap difference between the image-only and feature-card arms; the completed 1,024-image comparison is stored under `/Volumes/Elements/Retina-LanSE/experiments/vlm_ablation_comparison_qwen25vl_1024` (the 64-, 128-, and 256-image pilots are retained separately). The 1,024-image provenance audit rejects any feature-card run whose SAE dictionary does not match the grounded atlas.

`scripts/run_feature_hypothesis_verification.py` performs a label-blinded same-model consistency check on the clinician packet. Its result is explicitly circular and cannot substitute for independent ophthalmologist adjudication.

`scripts/run_numeric_sae_diagnostics.py` recomputes the selected numerical score on all labeled train/validation images, reports grade-stratified scores, train-only threshold operating points, calibration, and patient-cluster bootstrap intervals. Its evidence is `/Volumes/Elements/Retina-LanSE/experiments/numeric_sae_diagnostics`.

`scripts/run_activation_interaction_autoresearch.py` extends the numerical search with two prespecified pairwise co-activation rules. In the current pilot, neither pairwise rule beat the weighted additive score; details are in `research/ACTIVATION_INTERACTION_AUTORESEARCH_REPORT.md`.

## Publication-scale study

The complete paper design is specified in `research/PUBLICATION_PROTOCOL.md`. The currently executable label-free global reconstruction study is frozen separately in `research/EXPERIMENT_1_GLOBAL_SAE_PROTOCOL.md`, so its internal-test evidence cannot be confused with the later lesion-localization and external-clinical experiments. Access and compute prerequisites are tracked in `research/ACCESS_AND_RESOURCE_GATE.md`.

All publication-scale datasets, model weights, embeddings, tiles, checkpoints, and experiment tensors are routed to `/Volumes/Elements/Retina-LanSE`. Run `python3 scripts/check_external_storage.py` before any download or extraction; it fails closed if the external mount, directory layout, or free-space requirement is missing.

The executable sequence is:

```bash
# Requires accepted Kaggle competition rules.
/Volumes/Elements/Retina-LanSE/runtime/venv/bin/python scripts/download_eyepacs.py

# Requires a Hugging Face read token cached under external HF_HOME.
HF_HOME=/Volumes/Elements/Retina-LanSE/model_weights/huggingface \
  /Volumes/Elements/Retina-LanSE/runtime/venv/bin/python scripts/download_retfound_checkpoint.py

# After official archives are extracted under the external dataset root.
/Volumes/Elements/Retina-LanSE/runtime/venv/bin/python scripts/build_eyepacs_manifest.py \
  --dataset-root /Volumes/Elements/Retina-LanSE/datasets/eyepacs \
  --output /Volumes/Elements/Retina-LanSE/datasets/eyepacs/manifest.jsonl
```

`scripts/extract_retfound_embeddings.py` writes resumable, hashed external shards for either `final_mean_patch` or `final_cls`. `scripts/compare_embedding_runs.py` implements the repeatability and FP32/AMP parity gate.

Publication-scale global SAE execution uses:

- `scripts/run_global_sae_search.py` for the 54-configuration, single-seed, label-free screen per representation;
- `scripts/run_global_sae_finalists.py` for three finalists across five fixed seeds and the frozen one-standard-error selection rule;
- `scripts/run_representation_baselines.py` for frozen train-mean, matched-K coordinate, and matched-K PCA baselines;
- `scripts/freeze_internal_test_access.py` to bind the dataset, protocol, configuration, selections, checkpoints, baselines, and analysis source hashes before test access;
- `scripts/run_locked_internal_test_evaluation.py` for the single-access five-seed internal-test evaluation;
- `scripts/run_post_train_pipeline_terminal.zsh` to execute the complete sequence for mean-patch and CLS pooling.

Passing `--limit-configs` or `--epochs` to the screen always marks its output `partial_software_check`; such an artifact cannot be accepted by the finalist runner or used as paper evidence.

Internal-test extraction fails closed until both label-free selections and validation baselines are frozen. It then verifies the complete analysis-code hash bundle before reading the 8,760-image decontaminated internal-test split. The two selected five-seed SAEs are evaluated once alongside their frozen baselines. The activation-comparison AutoResearch runner is separate and refuses any locked-test path.

At any point, inspect the artifact-derived progress without changing the run:

```bash
/Volumes/Elements/Retina-LanSE/runtime/venv/bin/python scripts/report_experiment_1_status.py
```

The canonical Stage 6 in `research/PUBLICATION_PROTOCOL.md` is the one-time
labeled external evaluation (IDRiD/FGADR). It is distinct from the exploratory
ablation comparisons and remains gated until licensed data and clinical review
prerequisites are satisfied.

`scripts/check_stage6_external_gate.py` records that gate without downloading
or evaluating external images. The current blocked record is
`/Volumes/Elements/Retina-LanSE/experiments/stage6_external_gate_idrid_20260905.json`;
it will be rerun only after approved access metadata, a verified patient/label
manifest, and completed independent review are present.

After that gate passes, `scripts/run_stage6_external_evaluation.py` performs the
single frozen image-level evaluation. It reuses the train-derived SAE feature
set and threshold, refuses internal-test paths and output overwrite, and writes
patient-clustered AUROC/AUPRC plus all input hashes. The current invocation was
blocked before inference because the external directory and review rows are
still absent.

For an officially obtained IDRiD/FGADR archive with a supplied grade CSV,
`scripts/prepare_stage6_image_manifest.py` verifies image/label coverage,
patient IDs, dimensions, perceptual hashes, and SHA-256 values before the gate
can be rerun. It performs no download and accepts no unofficial mirror.
The required folder/CSV/access-record contract is documented in
`research/STAGE6_DATA_INTAKE_TEMPLATE.md`.
