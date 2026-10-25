# FundusExpert hard-exudate state-supervision study

This package implements the **method study** described in the user protocol. It does not contain patient data, FundusExpert weights, DDR images, or clinical results. Keep those assets on the external research drive. It uses the released FundusExpert checkpoint for F and as the initialization for B, C, and D.

## Current provenance and status (2026-10-12)

- Official FundusExpert source: `MeteorElf/FundusExpert`, cloned at commit `aaec4370a2f3e35994955ffc5ea901af66b3c02e` in `/Volumes/Elements/Retina-LanSE/experiments/fundusexpert_jlens_20261012/upstream/FundusExpert`.
- Official checkpoint: `MeteorElf/FundusExpert`, pinned HF revision `53bca09462f02c812cf56144105a21c7ae7bfc7b`; the four safetensors shards and all 16 runtime support files are in `/Volumes/Elements/Retina-LanSE/experiments/fundusexpert_jlens_20261012/checkpoint`. The index and shard extents parse successfully.
- Official DDR repository: `nkicsl/DDR-dataset`, cloned at commit `225d56e2f49dc767896e3d38a01071cfadef4271` under the same `upstream` directory. Its images are CC BY-NC-SA 4.0. The ten-part official ZIP passed `unzip -tq`; the 757-image segmentation subset and grading text files are extracted under `ddr_extracted`.
- The reproducible manifest builder is `python -m fundusexpert_jlens.build_ddr --root .../ddr_extracted/DDR-dataset --output .../ddr_manifest.jsonl --audit .../ddr_audit.json`. It keeps the official segmentation partitions, joins grades by exact filename, and conservatively excludes cross-partition perceptual-hash candidates from training or validation. The resulting frozen manifest has 742 images: 369 train, 148 validation, and 225 test. Public DDR provides no patient IDs; image-level resampling is a study limitation.
- Local Mac has 16 GB RAM, PyTorch with MPS, and no CUDA. The saved `ubuntu-heshi` SSH host at `192.168.1.5` timed out on 2026-10-12. Thus **no model preflight, training, or test evaluation has run**.
- A Python 3.11 tokenizer-check environment with Transformers 4.47.1 and SentencePiece 0.1.99 exists at `.../venv311`. The CUDA PyTorch, InternVL, `timm`, and PEFT runtime must be installed on the GPU host.

## Fixed data contract

The official DDR builder verifies a class-specific EX mask for every segmentation image. The original DDR paper says all 757 images were annotated for the four lesion classes and reports 486 EX-positive images, matching the archive exactly. Empty EX masks are treated as dataset-level negatives within this selected subset. The location target is the image-relative quadrants containing at least 5% of annotated EX pixels; it does not infer anatomical left/right. Grade labels are joined by exact filename from the grading text files regardless of those files' different partition assignments. Two test images have no compatible grade and retain `unknown`. The sample target contact sheet is `.../ddr_target_audit.jpg`.

```bash
python3 -m fundusexpert_jlens.build_ddr --root /Volumes/Elements/Retina-LanSE/experiments/fundusexpert_jlens_20261012/ddr_extracted/DDR-dataset --output /Volumes/Elements/Retina-LanSE/experiments/fundusexpert_jlens_20261012/ddr_manifest.jsonl --audit /Volumes/Elements/Retina-LanSE/experiments/fundusexpert_jlens_20261012/ddr_audit.json
```

Do not mix prior IDRiD experiments into the locked test. The JSON report target uses the same fixed prompt for all arms and has `hard_exudates`, `location`, and `dr_grade` fields. The manifest and audit should be treated as versioned input to every model run.

## Access and environment

On a CUDA host with enough VRAM for the 8B checkpoint plus activations, make the verified external-drive checkpoint available. The original artifacts are already downloaded. In the commands below, `UPSTREAM_SRC` is the cloned FundusExpert repository's `src` directory:

```bash
python -m fundusexpert_jlens.run preflight --model /path/to/checkpoint --upstream /path/to/upstream/FundusExpert/src --manifest /path/to/ddr_manifest.jsonl --out /path/to/preflight
```

Install a compatible CUDA PyTorch build and the upstream InternVL dependencies; their reference environment pins Transformers 4.47.1, `timm`, `einops`, `sentencepiece`, `peft`, and `accelerate`. Verify `torch.cuda.is_available()` and a one-image F generation before training. Do not use the broad upstream requirements file blindly; it includes unrelated segmentation and benchmark stacks.

## Study sequence

1. **Preflight.** Run `python -m fundusexpert_jlens.run preflight --model MODEL --upstream UPSTREAM_SRC --manifest MANIFEST --out PREFLIGHT --layer 16 --term present`. The word `exudate` is three tokens in FundusExpert's tokenizer, so it cannot serve as the planned single-token readout. The task-conditioned J-Lens variant takes the gradient of the future `present` status-token logit after a fixed, label-free `{"hard_exudates":"` prefix with respect to the pre-answer layer state. The readout state itself is computed before report text. It calibrates cosine scores on verified training labels. Validation AUC must be at least 0.60, and lesion-mask blurs must change the score more than size-matched background blurs on at least six verified positive images. Both gates are recorded in `calibration.json`; D refuses to train if either fails. These thresholds are feasibility criteria, not claimed clinical performance. Inspect the perturbation pairs and verify the mask class before accepting the gate.
2. **Train B/C/D.** Run `python -m fundusexpert_jlens.run train --arm B|C|D --seed 4106|4107|4108 --model MODEL --upstream UPSTREAM_SRC --manifest MANIFEST --out RUN_DIR` with the same `--epochs`, `--lr`, image policy, and LoRA rank for each arm. C adds a learned linear lesion probe at the same state; D uses the frozen scalar calibration and periodically refreshes the lens direction while logging direction cosine. D also takes `--calibration PREFLIGHT/calibration.json`. Tune `--loss-weight` on validation for C and D; B uses zero auxiliary loss. Save each candidate in a separate directory. F is the untouched checkpoint.
3. **Validation prediction.** For each candidate, run `python -m fundusexpert_jlens.run predict --arm ARM --seed SEED --adapter RUN_DIR/adapter --split validation --model MODEL --upstream UPSTREAM_SRC --manifest MANIFEST --out PRED_DIR`. F prediction omits `--adapter`. Use `freeze.py` with candidates in the form `arm:seed:weight:prediction_path:adapter_path`, including exactly one F reference prediction. Set recall and grade noninferiority margins to 0.05 absolute before any test prediction. It first excludes candidates below F minus those margins, then selects one candidate per arm and seed and writes a lock with the manifest hash, parser, margins, and chosen adapters.
4. **Locked test.** Generate test predictions once for F and all nine selected B/C/D models. Run `python -m fundusexpert_jlens.evaluate --manifest MANIFEST --lock LOCK --prediction F:4106:F_PRED --prediction B:4106:B_PRED ... --out RESULTS`. Repeat prediction flags for all arm/seed pairs. The evaluator requires identical test IDs, reports per-seed unsupported claims per image, lesion precision/recall, exact location agreement with the mask-derived descriptions, grade accuracy where compatible grades exist, invalid JSON count, and paired image-level bootstrap intervals. Patient-level intervals are unavailable from the public DDR files. Disputed free-text or location judgments require blinded reviewer adjudication outside this automated parser. Run available Fundus-MMBench tasks separately as a capability check; they are not yet wired into this package.

The automatic `j_lens_specific_benefit` flag is deliberately strict: it requires D versus both B and C for all three seeds, with the claim-rate interval below zero, recall/grade lower intervals above the frozen margins, and no higher invalid-report rate. Report all numeric results even if this flag is false. A lower error rate does not establish a unique causal reasoning chain.

## Technical limits

- This is a **task-conditioned, source-to-future-position Jacobian lens variant**. It is not the paper's full vocabulary-wide average Jacobian method. The interpretation is influence on the future `present` status logit in the fixed report format, not direct lesion detection or a spatial map.
- The scalar calibrator is fit only on training data. Refreshed directions can drift relative to that calibration, so D records cosine drift and requires separate validation selection.
- The code uses one case per optimizer step and an identical LoRA policy for trainable arms. It is a research reference runner, not a distributed training system. GPU memory and runtime have not been validated.
- Exact location agreement is only meaningful for reviewed location labels; it is not a pixel localization score. The primary unsupported-claim count excludes unknown negatives, and recall is reported to expose silence.

## Sources

[FundusExpert paper](https://arxiv.org/html/2507.17539) · [official code](https://github.com/MeteorElf/FundusExpert) · [checkpoint](https://huggingface.co/MeteorElf/FundusExpert) · [official DDR](https://github.com/nkicsl/DDR-dataset) · [SituationSupervision](https://arxiv.org/html/2212.10012) · [J-Lens](https://arxiv.org/html/2607.15495)
