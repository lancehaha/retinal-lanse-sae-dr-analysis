# Experiment 7C stop receipt

- Stop requested by the user: 2026-09-16 13:12 CST.
- Managed terminal session: `93527`.
- Stop mechanism: keyboard interrupt (`Ctrl-C`).
- Confirmed process exit code: `130` (`KeyboardInterrupt`).
- Interrupted work unit: full-label (`1.00`), five-grade task, ID Transformer, seed `4107`; this incomplete fit was not added to the progress ledger.
- Final experiment artifact: absent. No final evaluation result or scientific conclusion was produced.
- Persistent progress ledger: `/Volumes/Elements/Retina-LanSE/experiments/experiment_7c_sae_feature_view_transformers_20260914/training_progress.json`.
- Ledger last modified: 2026-09-16 12:44:34 CST.
- Ledger SHA-256: `51afd3d1d97445155c42ed764d292532ea5c2de51a03ba9ee5d10afccef6ae5e`.
- Completed development fits: `36 / 80`.

## Completed development fits and selected epochs

| Label budget | Task | Arm | Completed seeds and selected epochs |
|---|---|---|---|
| 1.00 | binary | activation Transformer | 4106: 1; 4107: 27; 4108: 43; 4109: 34; 4110: 33 |
| 1.00 | binary | card Transformer | 4106: 31; 4107: 33; 4108: 46; 4109: 27; 4110: 40 |
| 1.00 | binary | decoder Transformer | 4106: 17; 4107: 14; 4108: 25; 4109: 25; 4110: 23 |
| 1.00 | binary | ID Transformer | 4106: 16; 4107: 14; 4108: 14; 4109: 12; 4110: 22 |
| 1.00 | five-grade | activation Transformer | 4106: 8; 4107: 15; 4108: 28; 4109: 20; 4110: 26 |
| 1.00 | five-grade | card Transformer | 4106: 45; 4107: 21; 4108: 40; 4109: 38; 4110: 48 |
| 1.00 | five-grade | decoder Transformer | 4106: 22; 4107: 22; 4108: 32; 4109: 19; 4110: 27 |
| 1.00 | five-grade | ID Transformer | 4106: 22 |

## Not completed

- Remaining 44 development fits, including ID Transformer five-grade seeds 4107-4110 and every 10% label-budget fit.
- Median-epoch final refits.
- Per-seed and ensemble predictions for all required evaluation datasets.
- Bootstrap comparisons, mapping-shuffle controls, result JSON, and publication-ready results table.

This receipt records progress only. Development metrics and selected epochs are not final test results and must not be reported as completed Experiment 7C findings.
