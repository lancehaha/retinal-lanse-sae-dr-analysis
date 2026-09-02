# Retina-LanSE

Retina-LanSE is a research-only sparse-autoencoder (SAE) pipeline for auditing retinal image embeddings. The synthetic **Phase A** smoke test remains only a software check; the repository also contains a bounded retinal feasibility study and an activation-comparison AutoResearch runner.

The synthetic Phase A remains a mechanics-only check and is not clinical evidence. The active publication experiment started from the complete externally stored EyePACS corpus (88,702 images). A label-free pre-selection near-duplicate gate excluded four holdout patients/eight images, yielding a patient-disjoint publication cohort of 88,694 images (71,134/8,800/8,760 train/validation/internal test) and a checksum-pinned, frozen RETFound-CFP encoder. The encoder is not fine-tuned during this experiment.

## Run the verified Phase A experiment

```bash
python3 scripts/run_phase_a.py --config configs/smoke_global_sae.yaml
python3 -m unittest discover -s tests -v
```

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

Before any clinical claim, obtain datasets through their official access routes, record licences/data cards, validate RETFound-CFP extraction, preregister thresholds, and lock the test set. The current activation-comparison run is exploratory train/validation-pilot evidence; it does not access the locked internal test and is not a clinical-utility claim.

## Bounded retinal feasibility and activation comparison

The fixed feasibility evaluation is implemented by `scripts/run_single_sae_feasibility.py`. It compares dense RETFound embeddings, PCA-32, SAE reconstruction, and raw Top-K sparse activations on a patient-disjoint EyePACS train/validation cohort. The report and montage are written externally under `/Volumes/Elements/Retina-LanSE/experiments/feasibility_single_sae`.

The LanSE-inspired activation comparison is implemented by `scripts/run_activation_comparison_autoresearch.py`. It keeps the SAE checkpoint fixed and searches four prespecified activation-comparison recipes on an 80/20 patient split inside labeled training data. The current pilot result is written externally under `/Volumes/Elements/Retina-LanSE/experiments/activation_comparison_autoresearch`.

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
