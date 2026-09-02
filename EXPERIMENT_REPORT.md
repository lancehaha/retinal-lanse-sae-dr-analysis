# Retina-LanSE Phase A Experiment Report

## Outcome

The reproducible global Top-K SAE smoke test completed successfully on 2026-08-27. All six predeclared software/representation gates passed. This is a synthetic, training-only pilot and provides **no clinical evidence**.

## Experimental setup

| Item | Value |
|---|---:|
| Fictional patients | 36 |
| Images | 72 |
| Train / validation / locked test images | 56 / 12 / 4 |
| Embedding dimension | 64 |
| Dictionary width | 256 (4x) |
| Top-K budget | 8 |
| Epochs | 100 |
| Seed | 4106 |

Both synthetic eyes from a fictional patient remain in one split. Embeddings are standardized using train-split statistics only. The four synthetic test images are generated and locked but are not evaluated in this Phase A run.

## Results

| Metric | Untrained SAE | Sparse coordinate baseline (K=8) | Trained SAE |
|---|---:|---:|---:|
| Validation MSE | 1.2296 | 0.5927 | **0.3609** |
| Explained variance | -0.3412 | 0.3535 | **0.6064** |
| Cosine similarity | 0.0193 | 0.6585 | **0.8057** |
| Maximum active features | 8 | 8 | 8 |

Training MSE was 0.0444 and training explained variance was 0.9556. Validation had no saturated features, but 79.7% of dictionary features were dead on the tiny validation split. This indicates that the pipeline works while feature coverage is not established. A real campaign needs substantially more diverse embeddings and explicit dead-feature handling.

## Phase A gates

- Frozen encoder: passed.
- Exactly K or fewer active neurons per sample: passed (maximum 8).
- Reconstruction improved during training: passed.
- Better validation reconstruction than a same-budget sparse coordinate baseline: passed.
- Checkpoint saved and reloaded with optimizer, hashes, configuration, seed, and curves: passed.
- Evaluation repeated deterministically from the same checkpoint: passed; metrics and curves were also byte-identical across a full rerun.

## Integrity evidence

- Patient-level split hash: `735b6c03f8685f30fd3a32b3d4860dfec355f2ad6855828b06d0a6eab282e6ae`
- Frozen encoder hash: `5209a7ce4624e0b4c5789ba97a24650d3ce71b84d84fd3b1ce54e8c96f1bf2b8`
- The exact current checkpoint and embedding hashes are recorded in `artifacts/experiments/phase_a_global_sae_smoke/metrics.json`.
- Four unit tests passed, covering split determinism/leakage, Top-K enforcement, encoder freezing/determinism, and registry initialization.
- The SQLite registry contains all nine planned entity tables.

## Deliberately deferred

RETFound-CFP, approved retinal images, lesion labels/masks, local SAE localization, lesion occlusion, AutoResearch interpretation, and clinician review were not run because their required approved assets are absent. The loader fails closed rather than silently substituting a clinical encoder. These are later plan milestones, not evidence produced by this pilot.

