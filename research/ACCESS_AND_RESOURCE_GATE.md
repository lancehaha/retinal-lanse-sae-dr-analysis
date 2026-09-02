# Access and resource gate

No large-scale or clinical experiment may start until each item is evidenced.

## User-controlled access

- [x] Kaggle account accepted the EyePACS Diabetic Retinopathy Detection competition rules. The official 88.29 GB Kaggle bundle was downloaded directly to the external drive; all 15 inner files match the locked official byte inventory. Multipart reconstruction and CRC-checked extraction produced exactly 35,126 train and 53,576 test images. Evidence is stored at `/Volumes/Elements/Retina-LanSE/datasets/eyepacs/preparation_evidence.json`.
- [x] External storage selected: `/Volumes/Elements/Retina-LanSE` on the 4 TB external volume. Heavy artifacts are forbidden on the system disk by `configs/storage.yaml`.
- [x] Hugging Face account has accepted the gated `RETFound_mae_natureCFP` conditions. Live browser evidence on 2026-08-27 showed “You have been granted access to this model.” A download token must remain outside the repository.
- [x] Official Hugging Face metadata is pinned to repository revision `556830f78214f0e8da35af965292ac5a3180ac47`, file `RETFound_mae_natureCFP.pth`, and reported repository storage 3,952,489,221 bytes. The checkpoint downloader writes a SHA-256 evidence record under the external model directory.
- [x] Hugging Face CLI device authorization completed for `lancehaha`; the renewable credential is stored only under `/Volumes/Elements/Retina-LanSE/model_weights/huggingface`.
- [x] The official 3,952,489,221-byte checkpoint is downloaded externally and verified as SHA-256 `e1e4f66a1b792eeb6e2efaf158f33be35c8255f36b3d17ed67cd5129da246485`.
- [ ] DDR is downloaded from the official release and its licence/terms are recorded.
- [ ] IDRiD is downloaded from IEEE DataPort/official challenge and CC BY 4.0 attribution is recorded.
- [ ] FGADR research-use agreement is accepted if it will serve as the second external mask test.
- [ ] Data governance confirms where images, manifests, and derived activation examples may be stored and viewed.
- [ ] At least three ophthalmology reviewers are available, or the paper scope is explicitly reduced before test access.

## Compute gate

- [x] An isolated Python 3.11 environment and the official `rmaphoh/RETFound` repository are staged under `/Volumes/Elements/Retina-LanSE/runtime`; package caches and build temporaries are redirected to the external drive.
- [x] The upstream ViT-L architecture imports and produces `(1, 1, 1024)` `forward_features` output on a deterministic zero-input parity probe. The publication adapter flattens this to `(N, 1024)`, freezes every parameter, hashes the checkpoint, and fails closed on state-dictionary drift.
- [x] The real gated checkpoint loads into the frozen `final_mean_patch` encoder. The audited non-strict keys are limited to the absent classification/pooling heads and unused full-MAE decoder tensors expected by the official feature-extraction path; unrelated unexpected keys still fail closed.
- [ ] Linux GPU environment reproduces the official RETFound dependency versions.
- [ ] The prepared `ubuntu-heshi` GPU host is reachable. A read-only SSH probe on 2026-08-28 produced no banner or command output and was interrupted rather than changing remote state.
- [ ] GPU has enough memory for frozen ViT-L inference at the chosen batch size; automatic mixed precision is validated against FP32 on 100 images.
- [ ] 1,000-image benchmark records images/second, peak GPU memory, host RAM, cache size, and failure rate.
- [x] A 1,000-image CPU FP32 benchmark completed with zero failures at 7.804 images/s, batch size 4, two workers, about 4.33 GB peak process RSS, and a 4,159,776-byte hashed embedding shard. Evidence is stored under `/Volumes/Elements/Retina-LanSE/embeddings/technical_1000_cpu_fp32`.
- [x] Projected full global embedding extraction fits the storage and CPU budget: 88,702 images project to about 3.2 hours at the measured 7.804 images/s, subject to I/O variance. Local-tile extraction remains separately gated.
- [ ] Checkpoint, embedding, and raw-data backup strategy is tested.

## Repository gate

- [x] Real datasets, model/SAE checkpoints, embedding tensors, archives, and credentials are excluded from Git. The repository contains only external-path configuration references and small provenance reports; `.gitignore` blocks common credential names and publication-scale file types/directories.
- [x] Publication configs reference the external-only storage policy.
- [x] Raw retinal images are never committed or redistributed; JPEG/TIFF images and dataset/archive directories are ignored, and the acquired EyePACS files remain outside the repository under `/Volumes/Elements/Retina-LanSE`.
- [x] The full 88,702-image dataset manifest is complete and includes patient grouping, SHA-256, perceptual hash, source partition, camera availability, dimensions, and deterministic split. It contains 44,351 bilateral patients: 71,134 train, 8,804 validation, and 8,764 locked internal-test images.
- [x] EyePACS manifest tooling deterministically groups left/right images by patient, records dimensions/format/bytes/SHA-256/perceptual hash/source, and assigns the preregistered patient-level 80/10/10 train/validation/internal-test split. Camera metadata remains unavailable from the Kaggle distribution and is recorded as unavailable rather than inferred.
- [x] Full-manifest leakage audit found zero inconsistent patient splits, zero non-bilateral patients, zero exact duplicate groups, and zero exact cross-split duplicates. dHash collisions are retained as a low-specificity near-duplicate screening signal, not treated as proof of identity.
- [x] The high-specificity 255-bit DCT-pHash audit subsequently confirmed four disjoint train-to-holdout near-duplicate pairs before validation extraction or model selection. The label-free version-1.1 remediation excluded every image of the four lower-priority holdout patients, reassigned no patient, left all 71,134 training images unchanged, and produced the hashed 88,694-image publication manifest (71,134/8,800/8,760 images; 35,567/4,400/4,380 patients). The failed audit and remediation report remain preserved as provenance.
- [x] Extraction tooling writes deterministic external-only shards, supports both preregistered `final_mean_patch` and `final_cls` representations, records checkpoint/manifest/shard hashes and peak memory, and supports exact-repeat plus FP32/AMP comparison reports.
- [x] The current synthetic checkpoint is excluded from scientific comparisons. Publication search/finalist entry points require completed external RETFound embedding-run evidence and reject the Phase-A synthetic payload format.
- [x] Pipeline access to the EyePACS internal-test split is technically locked until an external authorization manifest freezes both mean-patch and CLS selections plus the dataset, analysis-code, config, and protocol hashes. A live refusal probe without that manifest exited before model loading with code 2.

## First executable milestone

Once the EyePACS rules gate, model-token gate, and a GPU are available:

1. install the official RETFound environment;
2. extract embeddings from exactly ten approved images;
3. compare output shapes/checksums across two runs;
4. benchmark 1,000 images;
5. only then schedule the 88,702-image extraction.

Milestone status: steps 1-4 passed. The 10-image FP32 repeats were byte-identical with shape `[10, 1024]`, maximum absolute error 0, and minimum cosine similarity 1.0. Train-split extraction may now proceed in resumable shards; validation and locked internal-test artifacts remain separated.
