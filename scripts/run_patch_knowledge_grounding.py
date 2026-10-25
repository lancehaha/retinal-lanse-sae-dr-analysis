from __future__ import annotations

"""Experiment 9B: lesion-mask grounding for frozen Patch-SAE features.

The stages fail closed and keep IDRiD knowledge discovery/validation separate
from the reused EyePACS prediction pilot. RETFound and the SAE are never
optimized here.
"""

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import platform
import sys
import time
import urllib.error
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from retina_lanse.data.manifest import sha256_path
from retina_lanse.data.preprocessing import crop_retinal_field, preprocess_retfound_cfp
from retina_lanse.encoders import load_retfound_cfp
from retina_lanse.sae.checkpoints import load_checkpoint
from retina_lanse.utils import set_deterministic
from scripts.run_label_efficiency_external_benchmark import binary_metrics, load_eyepacs_labels
import scripts.run_patch_sae_transformer_pilot as exp8
from scripts.run_patch_feature_cards_and_reports import call_text_ollama, parse_json


EXTERNAL = Path("/Volumes/Elements/Retina-LanSE")
OUTPUT = EXTERNAL / "experiments/experiment_9b_patch_knowledge_grounding_20260916"
EXP8 = EXTERNAL / "experiments/experiment_8_patch_sae_grounded_report_pilot_20260913"
TOP64 = EXP8 / "topk_64"
IDRID = EXTERNAL / "datasets/idrid/A. Segmentation"
CHECKPOINT = EXTERNAL / "model_weights/retfound_cfp/RETFound_mae_natureCFP.pth"
UPSTREAM = EXTERNAL / "runtime/RETFound"
GPU_LOCK = EXTERNAL / "runtime/retina_lanse_gpu.lock"
PROTOCOL = Path(__file__).resolve().parents[1] / "research/EXPERIMENT_9B_PATCH_KNOWLEDGE_GROUNDING_PROTOCOL_20260916.md"
GRID = 14
PATCHES = 196
INPUT_DIM = 1024
DICTIONARY = 16384
TOP_K = 64
MODEL_DIM = 128
SEEDS = (4106, 4107, 4108)
SHUFFLE_SEED = 9206
LESIONS = ("microaneurysms", "haemorrhages", "hard_exudates", "soft_exudates")
MASK_LAYOUT = {
    "microaneurysms": ("1. Microaneurysms", "MA"),
    "haemorrhages": ("2. Haemorrhages", "HE"),
    "hard_exudates": ("3. Hard Exudates", "EX"),
    "soft_exudates": ("4. Soft Exudates", "SE"),
    "optic_disc": ("5. Optic Disc", "OD"),
}


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(path)


def ensure_output(path: Path) -> Path:
    resolved = path.resolve()
    if resolved != OUTPUT.resolve():
        raise RuntimeError(f"Experiment 9B output must be exactly {OUTPUT}, got {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


@contextlib.contextmanager
def gpu_lock(stage: str):
    GPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with GPU_LOCK.open("a+", encoding="utf-8") as handle:
        print(json.dumps({"event": "waiting_for_gpu_lock", "stage": stage, "lock": str(GPU_LOCK)}), flush=True)
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0); handle.truncate()
        handle.write(json.dumps({"pid": os.getpid(), "stage": stage, "started": time.time()}) + "\n"); handle.flush()
        print(json.dumps({"event": "gpu_lock_acquired", "stage": stage}), flush=True)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def device_for(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def selected_features() -> list[int]:
    path = TOP64 / "patch_feature_cards.jsonl"
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    ids = [int(row["feature_id"]) for row in rows]
    if len(ids) != 64 or len(set(ids)) != 64 or min(ids) < 0 or max(ids) >= DICTIONARY:
        raise RuntimeError("Experiment 8 selected-feature universe is not exactly 64 unique IDs")
    return ids


def idrid_records(limit: int | None = None) -> list[dict]:
    records = []
    for split, count, image_dir, mask_split in (
        ("training", 54, "a. Training Set", "a. Training Set"),
        ("testing", 27, "b. Testing Set", "b. Testing Set"),
    ):
        start = 1 if split == "training" else 55
        for number in range(start, start + count):
            stem = f"IDRiD_{number:02d}"
            image = IDRID / "1. Original Images" / image_dir / f"{stem}.jpg"
            if not image.is_file():
                raise RuntimeError(f"missing IDRiD image: {image}")
            masks = {}
            for name, (folder, suffix) in MASK_LAYOUT.items():
                candidate = IDRID / "2. All Segmentation Groundtruths" / mask_split / folder / f"{stem}_{suffix}.tif"
                masks[name] = candidate if candidate.is_file() else None
            records.append({"stem": stem, "split": split, "image": image, "masks": masks})
    if len(records) != 81 or sum(r["split"] == "training" for r in records) != 54:
        raise RuntimeError("IDRiD official segmentation split mismatch")
    return records[:limit] if limit is not None else records


def crop_box_and_valid(image: Image.Image) -> tuple[tuple[int, int, int, int], np.ndarray]:
    rgb = np.asarray(image.convert("RGB"))
    valid = rgb.max(axis=2) > 8
    if not valid.any():
        return (0, 0, image.width, image.height), valid
    ys, xs = np.where(valid)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1), valid


def project_original_pixels(mask: np.ndarray, valid: np.ndarray, box: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Project source-pixel centers directly into the 14x14 token grid."""
    x0, y0, x1, y1 = box
    if mask.shape != valid.shape:
        raise RuntimeError(f"mask/image shape mismatch: {mask.shape} vs {valid.shape}")
    h, w = y1 - y0, x1 - x0
    rows = np.floor((np.arange(h, dtype=np.float64) + 0.5) * GRID / h).astype(np.int64).clip(0, GRID - 1)
    cols = np.floor((np.arange(w, dtype=np.float64) + 0.5) * GRID / w).astype(np.int64).clip(0, GRID - 1)
    patch = (rows[:, None] * GRID + cols[None, :]).reshape(-1)
    source = mask[y0:y1, x0:x1].reshape(-1).astype(bool)
    retinal = valid[y0:y1, x0:x1].reshape(-1).astype(bool)
    denominators = np.bincount(patch, minlength=PATCHES).astype(np.float64)
    positive = np.bincount(patch[source], minlength=PATCHES).astype(np.float64)
    valid_count = np.bincount(patch[retinal], minlength=PATCHES).astype(np.float64)
    occupancy = np.divide(positive, denominators, out=np.zeros(PATCHES), where=denominators > 0)
    valid_fraction = np.divide(valid_count, denominators, out=np.zeros(PATCHES), where=denominators > 0)
    return occupancy.astype(np.float32), valid_fraction.astype(np.float32)


def load_projected_masks(record: dict) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    with Image.open(record["image"]) as source:
        image = source.convert("RGB")
        box, valid = crop_box_and_valid(image)
        shape = valid.shape
    occupancies, common_valid = [], None
    for name in (*LESIONS, "optic_disc"):
        path = record["masks"][name]
        if path is None:
            mask = np.zeros(shape, dtype=bool)
        else:
            with Image.open(path) as source:
                mask = np.asarray(source.convert("L")) > 0
        occupancy, valid_fraction = project_original_pixels(mask, valid, box)
        occupancies.append(occupancy)
        if common_valid is None:
            common_valid = valid_fraction
        elif not np.array_equal(common_valid, valid_fraction):
            raise RuntimeError("valid-grid projection is not deterministic")
    return np.stack(occupancies), common_valid, box


@torch.inference_mode()
def selected_sae_activations(sae: nn.Module, tokens: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, features: list[int], device: torch.device) -> np.ndarray:
    flat = tokens.reshape(-1, INPUT_DIM)
    lookup = {feature: index for index, feature in enumerate(features)}
    chunks = []
    for start in range(0, len(flat), 512):
        item = (flat[start:start + 512] - mean) / std
        relu = torch.relu(sae.encoder(item.to(device)))
        values, ids = torch.topk(relu, TOP_K, dim=1)
        dense = torch.zeros((len(item), len(features)), device=device)
        for rank in range(TOP_K):
            rank_ids = ids[:, rank]
            for feature, column in lookup.items():
                match = rank_ids == feature
                if match.any():
                    dense[match, column] = values[match, rank]
        chunks.append(dense.cpu().numpy().astype(np.float32))
    return np.concatenate(chunks).reshape(len(tokens), PATCHES, len(features))


def source_hashes() -> dict:
    paths = {
        "protocol": PROTOCOL,
        "runner": Path(__file__).resolve(),
        "retfound_checkpoint": CHECKPOINT,
        "experiment8_patch_evidence": EXP8 / "patch_tokens/patch_extraction_evidence.json",
        "experiment8_sae_checkpoint": TOP64 / "patch_sae_checkpoint.pt",
        "experiment8_sae_normalization": TOP64 / "patch_sae_normalization.pt",
        "experiment8_selected_features": TOP64 / "patch_feature_cards.jsonl",
        "experiment8_cohorts": EXP8 / "patient_cohorts.json",
        "experiment8_prediction": TOP64 / "patch_transformer_pilot_result.json",
        "experiment8_predictions": TOP64 / "pilot_holdout_predictions.npz",
        "experiment8_traces": TOP64 / "patch_activation_traces.npz",
    }
    absent = [str(path) for path in paths.values() if not path.is_file()]
    if absent:
        raise RuntimeError(f"required frozen inputs are absent: {absent}")
    return {key: {"path": str(path), "sha256": sha256_path(path)} for key, path in paths.items()}


def freeze(output: Path) -> dict:
    records = idrid_records()
    files = [r["image"] for r in records] + [p for r in records for p in r["masks"].values() if p is not None]
    record = {
        "schema_version": "1.0",
        "status": "frozen_before_scientific_execution",
        "experiment": "9B_patch_knowledge_grounding",
        "idrid_role": "exploratory_knowledge_development_and_validation_not_external_validation",
        "idrid_counts": {"training": 54, "testing": 27},
        "idrid_bundle_sha256": hashlib.sha256("".join(sha256_path(p) for p in sorted(files)).encode()).hexdigest(),
        "selected_feature_ids": selected_features(),
        "seeds": list(SEEDS),
        "shuffle_seed": SHUFFLE_SEED,
        "grounding_bootstraps": 2000,
        "prediction_bootstraps": 2000,
        "thresholds": {"training_bh_q_lt": 0.05, "validation_auroc_min": 0.65, "validation_auprc_lift_min": 2.0,
                       "strong_auc_lower_gt": 0.5, "strong_lift_lower_gt": 1.0, "prediction_delta_min": 0.01,
                       "grounded_features_min": 8, "lesion_classes_min": 2, "valid_reports_min": 0.95},
        "preprocessing": "retinal-field-crop-v1+retfound-cfp-channel-zscore-v1",
        "architecture": {"grid": [14, 14], "top_k": 64, "dictionary": DICTIONARY, "model_dim": MODEL_DIM,
                         "layers": 4, "heads": 8, "ffn": 512, "dropout": 0.1},
        "sources": source_hashes(),
        "labels_loaded": False,
    }
    path = output / "specification_freeze.json"
    if path.exists() and json.loads(path.read_text()) != record:
        raise RuntimeError("existing Experiment 9B specification differs from frozen inputs")
    write_json(path, record)
    return record


def verify_freeze(output: Path) -> dict:
    path = output / "specification_freeze.json"
    if not path.is_file():
        raise RuntimeError("specification must be frozen before this stage")
    frozen = json.loads(path.read_text())
    current = source_hashes()
    if frozen.get("sources") != current or frozen.get("selected_feature_ids") != selected_features():
        raise RuntimeError("frozen source hash or feature universe mismatch")
    return frozen


def load_frozen_models(device: torch.device):
    evidence = json.loads((EXP8 / "patch_tokens/patch_extraction_evidence.json").read_text())
    model = load_retfound_cfp(CHECKPOINT, UPSTREAM, representation="final_mean_patch", recorded_checkpoint_hash=evidence["checkpoint_hash"]).to(device).eval()
    sae, payload = load_checkpoint(TOP64 / "patch_sae_checkpoint.pt")
    if payload.get("model_config") != {"input_dim": INPUT_DIM, "dictionary_size": DICTIONARY, "k": TOP_K}:
        raise RuntimeError("Experiment 8 SAE configuration mismatch")
    sae = sae.to(device).eval()
    norm = torch.load(TOP64 / "patch_sae_normalization.pt", map_location="cpu", weights_only=False)
    return model, sae, norm["mean"].float(), norm["std"].float()


@torch.inference_mode()
def extract_records(records: list[dict], device: torch.device) -> dict:
    model, sae, mean, std = load_frozen_models(device)
    tokens, globals_, occupancy, overlap, valid, boxes, parity = [], [], [], [], [], [], []
    features = selected_features()
    for index, record in enumerate(records):
        with Image.open(record["image"]) as source:
            image_tensor = preprocess_retfound_cfp(source.convert("RGB")).unsqueeze(0).to(device)
        patch = model.forward_patch_tokens(image_tensor)
        pooled = model.pool_patch_tokens(patch)
        global_again = model(image_tensor)
        difference = float((pooled - global_again).abs().max().item())
        if difference > 2e-5:
            raise RuntimeError(f"RETFound patch/global parity failed for {record['stem']}: {difference}")
        projected, valid_fraction, box = load_projected_masks(record)
        tokens.append(patch.cpu().to(torch.float16)); globals_.append(pooled.cpu())
        occupancy.append(projected); overlap.append(projected > 0); valid.append(valid_fraction); boxes.append(box); parity.append(difference)
        print(json.dumps({"event": "idrid_extracted", "index": index + 1, "total": len(records), "stem": record["stem"]}), flush=True)
    token_tensor = torch.cat(tokens)
    activations = selected_sae_activations(sae, token_tensor.float(), mean, std, features, device)
    return {"tokens": token_tensor.numpy(), "globals": torch.cat(globals_).numpy(), "occupancy": np.stack(occupancy).astype(np.float32),
            "overlap": np.stack(overlap).astype(bool), "valid_fraction": np.stack(valid).astype(np.float32),
            "activations": activations, "boxes": np.asarray(boxes, dtype=np.int32), "parity": np.asarray(parity, dtype=np.float32),
            "stems": np.asarray([r["stem"] for r in records]), "splits": np.asarray([r["split"] for r in records]),
            "feature_ids": np.asarray(features, dtype=np.int32)}


def smoke(output: Path, device: torch.device) -> dict:
    verify_freeze(output)
    # Pure unit gates, including one-pixel lesion preservation.
    valid = np.ones((31, 47), dtype=bool); mask = np.zeros_like(valid); mask[15, 23] = True
    occ, vf = project_original_pixels(mask, valid, (0, 0, 47, 31))
    if int((occ > 0).sum()) != 1 or not np.isclose(vf, 1).all():
        raise RuntimeError("one-pixel microaneurysm preservation smoke failed")
    ids = selected_features(); perm = np.random.default_rng(SHUFFLE_SEED).permutation(len(ids))
    if len(set(perm.tolist())) != 64 or np.array_equal(perm, np.arange(64)):
        raise RuntimeError("fixed feature shuffle smoke failed")
    with gpu_lock("five_image_smoke"):
        arrays = extract_records(idrid_records(limit=5), device)
    if arrays["tokens"].shape != (5, PATCHES, INPUT_DIM) or arrays["activations"].shape != (5, PATCHES, 64):
        raise RuntimeError("five-image tensor shape smoke failed")
    if not np.isfinite(arrays["activations"]).all() or float(arrays["parity"].max()) > 2e-5:
        raise RuntimeError("five-image finite/parity smoke failed")
    record = {"status": "passed", "images": 5, "tests": ["image_mask_matching", "exact_crop_projection", "one_pixel_microaneurysm_preserved",
              "global_patch_separation_and_parity", "selected_feature_universe", "fixed_shuffle", "finite_activations"],
              "max_global_parity_error": float(arrays["parity"].max()), "token_shape": list(arrays["tokens"].shape),
              "activation_shape": list(arrays["activations"].shape)}
    write_json(output / "smoke_result.json", record)
    return record


def extract(output: Path, device: torch.device) -> dict:
    verify_freeze(output)
    smoke_result = json.loads((output / "smoke_result.json").read_text())
    if smoke_result.get("status") != "passed" or smoke_result.get("images") != 5:
        raise RuntimeError("five-image smoke must pass before full extraction")
    with gpu_lock("idrid_extraction"):
        arrays = extract_records(idrid_records(), device)
    np.savez_compressed(output / "idrid_patch_grounding_arrays.npz", **arrays)
    record = {"status": "complete", "images": 81, "training": 54, "testing": 27, "token_shape": list(arrays["tokens"].shape),
              "selected_activation_shape": list(arrays["activations"].shape), "max_global_parity_error": float(arrays["parity"].max()),
              "array_sha256": sha256_path(output / "idrid_patch_grounding_arrays.npz"),
              "positive_patch_counts": {name: int(arrays["overlap"][:, i].sum()) for i, name in enumerate((*LESIONS, "optic_disc"))},
              "tiny_lesions_preserved": True}
    write_json(output / "idrid_extraction_evidence.json", record)
    return record


def patch_strata() -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.divmod(np.arange(PATCHES), GRID)
    quadrant = (rows >= GRID / 2).astype(int) * 2 + (cols >= GRID / 2).astype(int)
    distance = np.sqrt((rows + .5 - GRID / 2) ** 2 + (cols + .5 - GRID / 2) ** 2)
    normalized = distance / distance.max()
    radial = np.digitize(normalized, [0.25, 0.50, 0.75], right=False)
    return quadrant, radial


def matched_rows(overlap: np.ndarray, valid: np.ndarray, lesion_index: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    quadrant, radial = patch_strata(); images, patches, labels = [], [], []
    any_lesion = overlap[:, :4].any(axis=1); optic = overlap[:, 4]
    for image in range(len(overlap)):
        positive = np.where(overlap[image, lesion_index])[0]
        negative_ok = (~any_lesion[image]) & (~optic[image]) & (valid[image] >= 0.5)
        for q in range(4):
            for r in range(4):
                pos = positive[(quadrant[positive] == q) & (radial[positive] == r)]
                candidates = np.where(negative_ok & (quadrant == q) & (radial == r))[0]
                if not len(pos) or not len(candidates):
                    continue
                ordered = sorted(candidates.tolist(), key=lambda p: hashlib.sha256(f"{seed}:{image}:{q}:{r}:{p}".encode()).hexdigest())
                neg = np.asarray(ordered[:len(pos)], dtype=np.int64)
                images.extend([image] * (len(pos) + len(neg))); patches.extend(pos.tolist() + neg.tolist()); labels.extend([1] * len(pos) + [0] * len(neg))
    return np.asarray(images), np.asarray(patches), np.asarray(labels, dtype=np.int8)


def auc_ap(y: np.ndarray, score: np.ndarray) -> tuple[float, float, float]:
    if len(y) == 0 or np.unique(y).size != 2:
        return float("nan"), float("nan"), float("nan")
    prevalence = float(y.mean()); ap = float(average_precision_score(y, score))
    return float(roc_auc_score(y, score)), ap, ap / prevalence if prevalence > 0 else float("nan")


def cluster_bootstrap(y: np.ndarray, score: np.ndarray, clusters: np.ndarray, resamples: int, seed: int) -> dict:
    estimate = auc_ap(y, score); unique = np.unique(clusters); grouped = {c: np.where(clusters == c)[0] for c in unique}
    rng = np.random.default_rng(seed); values = []
    for _ in range(resamples):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([grouped[c] for c in chosen])
        metric = auc_ap(y[index], score[index])
        if np.isfinite(metric).all(): values.append(metric)
    boot = np.asarray(values)
    if not len(boot):
        lower = upper = [float("nan")] * 3
    else:
        lower = np.percentile(boot, 2.5, axis=0).tolist(); upper = np.percentile(boot, 97.5, axis=0).tolist()
    return {"auroc": estimate[0], "auprc": estimate[1], "prevalence": float(y.mean()) if len(y) else float("nan"), "auprc_lift": estimate[2],
            "auroc_interval": [lower[0], upper[0]], "auprc_interval": [lower[1], upper[1]], "lift_interval": [lower[2], upper[2]],
            "bootstrap_auroc_le_half": int((boot[:, 0] <= 0.5).sum()) if len(boot) else resamples,
            "valid_bootstraps": int(len(boot)), "rows": int(len(y)), "images": int(len(unique))}


def bh_adjust(pvalues: list[float]) -> list[float]:
    p = np.asarray(pvalues, dtype=float); order = np.argsort(p); result = np.empty(len(p)); running = 1.0
    for rank_index in range(len(p) - 1, -1, -1):
        original = order[rank_index]; rank = rank_index + 1
        running = min(running, float(p[original]) * len(p) / rank); result[original] = min(running, 1.0)
    return result.tolist()


def ground(output: Path, resamples: int) -> dict:
    verify_freeze(output)
    evidence = json.loads((output / "idrid_extraction_evidence.json").read_text())
    array_path = output / "idrid_patch_grounding_arrays.npz"
    if evidence.get("array_sha256") != sha256_path(array_path):
        raise RuntimeError("IDRiD grounding array hash mismatch")
    data = np.load(array_path); activations = data["activations"]; overlap = data["overlap"]; valid = data["valid_fraction"]
    feature_ids = data["feature_ids"].astype(int).tolist(); splits = data["splits"]
    results, training_p = [], []
    for lesion_index, lesion in enumerate(LESIONS):
        image, patch, y = matched_rows(overlap, valid, lesion_index, 4106 + lesion_index)
        for feature_index, feature in enumerate(feature_ids):
            row = {"feature_id": feature, "lesion": lesion}
            for split, key in (("training", "training"), ("testing", "validation")):
                keep = splits[image] == split
                stats = cluster_bootstrap(y[keep], activations[image[keep], patch[keep], feature_index], image[keep], resamples, 5000 + lesion_index * 100 + feature_index + (0 if split == "training" else 10000))
                row[key] = stats
            p = (1 + row["training"]["bootstrap_auroc_le_half"]) / (row["training"]["valid_bootstraps"] + 1)
            row["training_one_sided_p"] = p; training_p.append(p); results.append(row)
    qvalues = bh_adjust(training_p)
    for row, q in zip(results, qvalues):
        row["training_bh_q"] = q; train, val = row["training"], row["validation"]
        row["supported"] = bool(q < 0.05 and val["auroc"] >= 0.65 and val["auprc_lift"] >= 2.0 and train["auroc"] > 0.5 and val["auroc"] > 0.5)
        row["strong"] = bool(row["supported"] and val["auroc_interval"][0] > 0.5 and val["lift_interval"][0] > 1.0)
    # Optic-disc association is recorded as confounding only.
    optic_rows = {}
    image, patch, y = matched_rows(overlap, valid, 4, 8192)
    for feature_index, feature in enumerate(feature_ids):
        keep = splits[image] == "testing"
        optic_rows[feature] = cluster_bootstrap(y[keep], activations[image[keep], patch[keep], feature_index], image[keep], resamples, 19000 + feature_index)
    by_feature = {feature: {"feature_id": feature, "lesions": {}} for feature in feature_ids}
    for row in results: by_feature[row["feature_id"]]["lesions"][row["lesion"]] = row
    vectors = np.zeros((DICTIONARY, 6), dtype=np.float32); compact = np.zeros((64, 6), dtype=np.float32)
    for feature_index, feature in enumerate(feature_ids):
        supported = [row for row in by_feature[feature]["lesions"].values() if row["supported"]]
        by_feature[feature]["grounded"] = bool(supported); by_feature[feature]["optic_disc_confounder"] = optic_rows[feature]
        if supported:
            for lesion_index, lesion in enumerate(LESIONS):
                row = by_feature[feature]["lesions"][lesion]
                if row["supported"]: compact[feature_index, lesion_index] = np.clip(2 * (row["validation"]["auroc"] - 0.5), 0, 1)
            confidences = [np.clip(min((row["validation"]["auroc"] - .5) / .5, (row["validation"]["auprc_lift"] - 1)), 0, 1) * (1 - row["training_bh_q"]) for row in supported]
            compact[feature_index, 4] = max(confidences)
            compact[feature_index, 5] = np.clip(2 * (optic_rows[feature]["auroc"] - .5), -1, 1)
            vectors[feature] = compact[feature_index]
        by_feature[feature]["knowledge_vector"] = compact[feature_index].tolist()
    np.savez_compressed(output / "knowledge_vectors.npz", dictionary_vectors=vectors, selected_vectors=compact, feature_ids=np.asarray(feature_ids, dtype=np.int32))
    supported_features = [feature for feature in feature_ids if by_feature[feature]["grounded"]]
    classes = sorted({lesion for row in results if row["supported"] for lesion in [row["lesion"]]})
    record = {"status": "complete", "features": [by_feature[f] for f in feature_ids], "feature_lesion_tests": results,
              "supported_features": supported_features, "supported_feature_count": len(supported_features), "supported_lesion_classes": classes,
              "strong_pair_count": sum(row["strong"] for row in results), "thresholds_relaxed": False,
              "knowledge_vectors_sha256": sha256_path(output / "knowledge_vectors.npz"),
              "grounding_gate_passed": len(supported_features) >= 8 and len(classes) >= 2}
    write_json(output / "feature_grounding_result.json", record)
    return record


class PatchDataset(Dataset):
    def __init__(self, dense, ids, values, targets):
        self.dense = torch.as_tensor(dense, dtype=torch.float32); self.ids = torch.as_tensor(ids, dtype=torch.long)
        self.values = torch.as_tensor(values, dtype=torch.float32); self.targets = torch.as_tensor(targets, dtype=torch.long)
    def __len__(self): return len(self.targets)
    def __getitem__(self, index): return self.dense[index], self.ids[index], self.values[index], self.targets[index]


class KnowledgePatchTransformer(nn.Module):
    def __init__(self, dictionary_vectors: np.ndarray):
        super().__init__()
        self.register_buffer("knowledge", torch.as_tensor(dictionary_vectors, dtype=torch.float32))
        self.knowledge_projection = nn.Linear(6, MODEL_DIM, bias=False)
        self.dense_projection = nn.Linear(INPUT_DIM, MODEL_DIM)
        self.value_projection = nn.Sequential(nn.Linear(1, MODEL_DIM), nn.GELU(), nn.Linear(MODEL_DIM, MODEL_DIM))
        self.pool_query = nn.Parameter(torch.empty(MODEL_DIM)); self.cls = nn.Parameter(torch.empty(1, 1, MODEL_DIM))
        self.row_embedding = nn.Embedding(GRID, MODEL_DIM); self.column_embedding = nn.Embedding(GRID, MODEL_DIM); self.type_embedding = nn.Embedding(3, MODEL_DIM)
        layer = nn.TransformerEncoderLayer(MODEL_DIM, 8, 512, .1, "gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 4, norm=nn.LayerNorm(MODEL_DIM)); self.head = nn.Linear(MODEL_DIM, 1)
        nn.init.normal_(self.pool_query, std=.02); nn.init.normal_(self.cls, std=.02)
    def forward(self, dense, ids, values):
        entry = self.value_projection(values.unsqueeze(-1)) + self.knowledge_projection(self.knowledge[ids])
        weights = torch.softmax(torch.einsum("bpkd,d->bpk", entry, self.pool_query) / math.sqrt(MODEL_DIM), dim=-1)
        patch = torch.sum(entry * weights.unsqueeze(-1), dim=2)
        rows = torch.arange(GRID, device=dense.device).repeat_interleave(GRID); cols = torch.arange(GRID, device=dense.device).repeat(GRID)
        patch = patch + self.row_embedding(rows)[None] + self.column_embedding(cols)[None] + self.type_embedding(torch.tensor([2], device=dense.device))[None]
        cls = self.cls.expand(len(dense), -1, -1) + self.type_embedding(torch.tensor([0], device=dense.device))[None]
        global_token = self.dense_projection(dense).unsqueeze(1) + self.type_embedding(torch.tensor([1], device=dense.device))[None]
        return self.head(self.encoder(torch.cat([cls, global_token, patch], dim=1))[:, 0])


def data_loader(dataset, shuffle, seed):
    return DataLoader(dataset, batch_size=16, shuffle=shuffle, num_workers=0, generator=torch.Generator().manual_seed(seed))


def criterion(targets: np.ndarray, device: torch.device):
    pos = max(int(targets.sum()), 1); neg = max(len(targets) - pos, 1)
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], device=device))


@torch.inference_mode()
def predict_model(model, dataset, device):
    model.eval(); rows = []
    for dense, ids, values, _ in data_loader(dataset, False, 4106):
        rows.append(torch.sigmoid(model(dense.to(device), ids.to(device), values.to(device))[:, 0]).cpu().numpy())
    return np.concatenate(rows)


def fit_development(vectors, train, development, seed, device, max_epochs=50, patience=8):
    set_deterministic(seed); model = KnowledgePatchTransformer(vectors).to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    loss_fn = criterion(train.targets.numpy(), device); best, best_epoch, stale, history = -np.inf, 1, 0, []
    for epoch in range(1, max_epochs + 1):
        model.train(); losses = []
        for dense, ids, values, targets in data_loader(train, True, seed + epoch):
            optimizer.zero_grad(set_to_none=True); logits = model(dense.to(device), ids.to(device), values.to(device))[:, 0]
            loss = loss_fn(logits, targets.float().to(device)); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); losses.append(float(loss.item()))
        score = float(roc_auc_score(development.targets.numpy(), predict_model(model, development, device)))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "development_auroc": score})
        if score > best + 1e-6: best, best_epoch, stale = score, epoch, 0
        else: stale += 1
        if stale >= patience: break
    return best_epoch, history


def fit_fixed(vectors, train, seed, epochs, device):
    set_deterministic(seed); model = KnowledgePatchTransformer(vectors).to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    loss_fn = criterion(train.targets.numpy(), device)
    for epoch in range(1, epochs + 1):
        model.train()
        for dense, ids, values, targets in data_loader(train, True, seed + epoch):
            optimizer.zero_grad(set_to_none=True); loss = loss_fn(model(dense.to(device), ids.to(device), values.to(device))[:, 0], targets.float().to(device))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
    return model


def patient_difference(y, a, b, patients, resamples, seed):
    estimate = float(roc_auc_score(y, a) - roc_auc_score(y, b)); unique = np.unique(patients); grouped = {p: np.where(patients == p)[0] for p in unique}
    rng = np.random.default_rng(seed); values = []
    for _ in range(resamples):
        chosen = rng.choice(unique, len(unique), replace=True); index = np.concatenate([grouped[p] for p in chosen])
        if np.unique(y[index]).size == 2: values.append(float(roc_auc_score(y[index], a[index]) - roc_auc_score(y[index], b[index])))
    return {"estimate": estimate, "interval": np.percentile(values, [2.5, 97.5]).tolist(), "valid_resamples": len(values)}


def prediction_data(vectors: np.ndarray):
    patch = exp8.load_patch_run(EXP8 / "patch_tokens"); cohorts = json.loads((EXP8 / "patient_cohorts.json").read_text()); indices = exp8.group_indices(patch["patients"], cohorts)
    traces = np.load(TOP64 / "patch_activation_traces.npz")
    if traces["feature_ids"].shape != (len(patch["paths"]), PATCHES, TOP_K) or not np.array_equal(traces["paths"].astype(str), np.asarray(patch["paths"])):
        raise RuntimeError("Experiment 8 activation traces do not align to patch tokens")
    labels = load_eyepacs_labels(EXTERNAL / "datasets/eyepacs/trainLabels.csv")
    grade = np.asarray([labels.get(Path(path).stem.lower(), -1) for path in patch["paths"]], dtype=np.int64); labelled = grade >= 0
    groups = {name: np.asarray([i for i in rows if labelled[i]], dtype=np.int64) for name, rows in indices.items()}
    train_i, dev_i, hold_i = groups["sae_train"], groups["development"], groups["pilot_holdout"]
    if set(patch["patients"][train_i]) & set(patch["patients"][dev_i]) or set(patch["patients"][hold_i]) & (set(patch["patients"][train_i]) | set(patch["patients"][dev_i])):
        raise RuntimeError("patient overlap detected")
    dense = patch["global"].numpy().astype(np.float32); ids = traces["feature_ids"].astype(np.int64); raw_values = traces["values"].astype(np.float32)
    binary = (grade >= 2).astype(np.int64); scaling = exp8.fit_scaling(dense[train_i], raw_values[train_i]); dense_dev, values_dev = exp8.apply_scaling(dense, raw_values, scaling)
    final_i = np.concatenate([train_i, dev_i]); final_scaling = exp8.fit_scaling(dense[final_i], raw_values[final_i]); dense_final, values_final = exp8.apply_scaling(dense, raw_values, final_scaling)
    return patch, ids, raw_values, binary, train_i, dev_i, hold_i, final_i, dense_dev, values_dev, dense_final, values_final, final_scaling


def predict_stage(output: Path, device: torch.device, resamples: int) -> dict:
    verify_freeze(output); grounding = json.loads((output / "feature_grounding_result.json").read_text())
    vector_file = output / "knowledge_vectors.npz"
    if grounding.get("knowledge_vectors_sha256") != sha256_path(vector_file): raise RuntimeError("knowledge-vector hash mismatch")
    knowledge = np.load(vector_file)["dictionary_vectors"].astype(np.float32); selected = selected_features()
    shuffled = knowledge.copy(); permutation = np.random.default_rng(SHUFFLE_SEED).permutation(len(selected)); shuffled[selected] = knowledge[np.asarray(selected)[permutation]]
    patch, ids, raw_values, binary, train_i, dev_i, hold_i, final_i, dense_dev, values_dev, dense_final, values_final, scaling = prediction_data(knowledge)
    train = PatchDataset(dense_dev[train_i], ids[train_i], values_dev[train_i], binary[train_i]); dev = PatchDataset(dense_dev[dev_i], ids[dev_i], values_dev[dev_i], binary[dev_i])
    full = PatchDataset(dense_final[final_i], ids[final_i], values_final[final_i], binary[final_i]); hold = PatchDataset(dense_final[hold_i], ids[hold_i], values_final[hold_i], binary[hold_i])
    selected_epochs, histories, ensembles, checkpoints = {}, {}, {}, []
    with gpu_lock("transformer_training"):
        for arm, vectors in (("knowledge_patch", knowledge), ("shuffled_knowledge_patch", shuffled)):
            epochs = []
            for seed in SEEDS:
                epoch, history = fit_development(vectors, train, dev, seed, device); epochs.append(epoch); histories[f"{arm}:{seed}"] = history
                print(json.dumps({"phase": "development", "arm": arm, "seed": seed, "selected_epoch": epoch}), flush=True)
            selected_epochs[arm] = int(np.median(epochs)); predictions = []
            for seed in SEEDS:
                model = fit_fixed(vectors, full, seed, selected_epochs[arm], device); path = output / "checkpoints" / f"{arm}_seed_{seed}.pt"; path.parent.mkdir(exist_ok=True)
                torch.save({"state_dict": model.cpu().state_dict(), "arm": arm, "seed": seed, "epochs": selected_epochs[arm], "knowledge_sha256": sha256_path(vector_file)}, path)
                reloaded = KnowledgePatchTransformer(vectors); reloaded.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["state_dict"])
                before = predict_model(model.to(device), hold, device); after = predict_model(reloaded.to(device), hold, device)
                if not np.array_equal(before, after):
                    if not np.allclose(before, after, atol=1e-7, rtol=1e-7): raise RuntimeError("checkpoint reload prediction mismatch")
                predictions.append(after); checkpoints.append({"arm": arm, "seed": seed, "path": str(path), "sha256": sha256_path(path)})
            ensembles[arm] = np.mean(np.stack(predictions), axis=0)
    baseline = np.load(TOP64 / "pilot_holdout_predictions.npz")
    original = baseline["full"].astype(np.float64); dense = baseline["dense_mlp"].astype(np.float64); target = binary[hold_i]; patients = patch["patients"][hold_i]
    if len(original) != len(hold_i) or not np.array_equal(target, baseline["targets"]): raise RuntimeError("reused Experiment 8 holdout predictions mismatch")
    predictions = {"dense_mlp": dense, "original_patch": original, **ensembles}
    np.savez_compressed(output / "holdout_predictions.npz", **{k: v.astype(np.float32) for k, v in predictions.items()}, targets=target,
                        patients=patients, paths=np.asarray([patch["paths"][i] for i in hold_i]), holdout_indices=hold_i,
                        activation_mean=np.asarray([scaling.activation_mean]), activation_std=np.asarray([scaling.activation_std]))
    metrics = {name: binary_metrics(target, score, patients, resamples, 22000 + i) for i, (name, score) in enumerate(predictions.items())}
    comparisons = {"knowledge_vs_original": patient_difference(target, predictions["knowledge_patch"], original, patients, resamples, 23001),
                   "knowledge_vs_shuffled": patient_difference(target, predictions["knowledge_patch"], predictions["shuffled_knowledge_patch"], patients, resamples, 23002)}
    success = all(item["estimate"] >= .01 and item["interval"][0] > 0 for item in comparisons.values())
    record = {"status": "complete", "endpoint": "referable_dr_grade_ge_2", "reused_baselines": ["dense_mlp", "original_patch"],
              "selected_epochs": selected_epochs, "development_histories": histories, "metrics": metrics, "comparisons": comparisons,
              "prediction_success": success, "checkpoints": checkpoints, "predictions_sha256": sha256_path(output / "holdout_predictions.npz"),
              "holdout": {"images": len(hold_i), "patients": len(set(patients))}}
    write_json(output / "prediction_result.json", record); return record


def load_knowledge_ensemble(output: Path, vectors: np.ndarray, device: torch.device):
    result = json.loads((output / "prediction_result.json").read_text()); models = []
    for seed in SEEDS:
        path = output / "checkpoints" / f"knowledge_patch_seed_{seed}.pt"; payload = torch.load(path, map_location="cpu", weights_only=False)
        model = KnowledgePatchTransformer(vectors); model.load_state_dict(payload["state_dict"]); models.append(model.to(device).eval())
    return models, result


@torch.inference_mode()
def ensemble_probability(models, dense, ids, values, device) -> float:
    x = torch.as_tensor(dense[None], device=device); i = torch.as_tensor(ids[None], device=device); v = torch.as_tensor(values[None], device=device)
    return float(np.mean([torch.sigmoid(model(x, i, v)[0, 0]).item() for model in models]))


def effect_interval(cited: np.ndarray, random: np.ndarray, patients: np.ndarray, resamples: int, seed: int):
    if not len(cited): return {"estimate": float("nan"), "interval": [float("nan"), float("nan")], "valid_resamples": 0}
    unique = np.unique(patients); grouped = {p: np.where(patients == p)[0] for p in unique}; rng = np.random.default_rng(seed); values = []
    for _ in range(resamples):
        chosen = rng.choice(unique, len(unique), replace=True); index = np.concatenate([grouped[p] for p in chosen]); values.append(float(np.mean(cited[index] - random[index])))
    return {"estimate": float(np.mean(cited - random)), "interval": np.percentile(values, [2.5, 97.5]).tolist(), "valid_resamples": len(values)}


def report_stage(output: Path, device: torch.device, resamples: int, model_name: str, base_url: str, skip_qwen: bool) -> dict:
    verify_freeze(output); grounding = json.loads((output / "feature_grounding_result.json").read_text()); vectors = np.load(output / "knowledge_vectors.npz")["dictionary_vectors"].astype(np.float32)
    models, prediction_result = load_knowledge_ensemble(output, vectors, device)
    patch, ids, raw_values, binary, train_i, dev_i, hold_i, final_i, dense_dev, values_dev, dense, values, scaling = prediction_data(vectors)
    grounded = {int(row["feature_id"]): row for row in grounding["features"] if row["grounded"]}; qwen_available = not skip_qwen
    all_reports, cited_effects, random_effects, effect_patients = [], [], [], []
    with gpu_lock("qwen_and_faithfulness"):
        for local, index in enumerate(hold_i):
            base = ensemble_probability(models, dense[index], ids[index], values[index], device); candidates = []
            active = []
            for patch_index in range(PATCHES):
                for rank in range(TOP_K):
                    feature = int(ids[index, patch_index, rank]); raw = float(raw_values[index, patch_index, rank])
                    active.append((patch_index, rank, feature, raw))
                    if feature not in grounded: continue
                    deleted = values[index].copy(); deleted[patch_index, rank] = float(-scaling.activation_mean / scaling.activation_std)
                    delta = base - ensemble_probability(models, dense[index], ids[index], deleted, device)
                    if delta > 0: candidates.append({"feature_id": feature, "patch_index": patch_index, "row": patch_index // GRID, "column": patch_index % GRID,
                                                      "activation": raw, "deletion_effect": delta, "knowledge_vector": grounded[feature]["knowledge_vector"],
                                                      "supported_lesions": [name for name, row in grounded[feature]["lesions"].items() if row["supported"]]})
            citations = sorted(candidates, key=lambda x: (-x["deletion_effect"], x["feature_id"], x["patch_index"]))[:5]
            for citation_index, citation in enumerate(citations):
                radial = patch_strata()[1][citation["patch_index"]]; raw_all = np.asarray([item[3] for item in active]); quartile = min(int(np.searchsorted(np.quantile(raw_all, [.25, .5, .75]), citation["activation"], side="right")), 3)
                pool = [item for item in active if patch_strata()[1][item[0]] == radial and min(int(np.searchsorted(np.quantile(raw_all, [.25, .5, .75]), item[3], side="right")), 3) == quartile and item[2] != citation["feature_id"]]
                if not pool: pool = [item for item in active if item[2] != citation["feature_id"]]
                chosen = sorted(pool, key=lambda item: hashlib.sha256(f"{SHUFFLE_SEED}:{index}:{citation_index}:{item}".encode()).hexdigest())[0]
                deleted = values[index].copy(); deleted[chosen[0], chosen[1]] = float(-scaling.activation_mean / scaling.activation_std)
                random_delta = base - ensemble_probability(models, dense[index], ids[index], deleted, device)
                cited_effects.append(citation["deletion_effect"]); random_effects.append(random_delta); effect_patients.append(str(patch["patients"][index])); citation["matched_random_effect"] = random_delta
            structured = {"schema_version": "1.0", "path": patch["paths"][index], "referable_dr_probability": base, "citations": citations,
                          "abstained": len(citations) == 0, "scope": "exploratory frozen numerical report; lesion associations are IDRiD mask-grounded but not clinician adjudication"}
            prompt = ("Render the supplied structured retinal research report as JSON with keys summary, evidence_feature_ids, uncertainty. "
                      "Use only supplied fields. Do not infer a target label, diagnosis, grade, unseen image content, or uncited feature.\n" + json.dumps(structured, sort_keys=True))
            if any(word in prompt.lower() for word in ("target_label", "ground_truth", "true_grade")): raise RuntimeError("target label leaked into report prompt")
            raw, rendered, status = "", None, "skipped"
            if qwen_available:
                try:
                    raw = call_text_ollama(base_url, model_name, prompt, 180); rendered = parse_json(raw)
                    allowed = {c["feature_id"] for c in citations}; returned = set(rendered.get("evidence_feature_ids", [])) if isinstance(rendered, dict) else set()
                    status = "complete" if isinstance(rendered, dict) and returned <= allowed else "invalid"
                except (urllib.error.URLError, TimeoutError, OSError, KeyError, ValueError) as exc:
                    raw = repr(exc); status = "unavailable"; qwen_available = False
            valid = all(c["feature_id"] in grounded and c["deletion_effect"] > 0 and 0 <= c["patch_index"] < PATCHES for c in citations)
            all_reports.append({"structured": structured, "prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                                "qwen_raw": raw, "qwen_rendered": rendered, "qwen_status": status, "valid": valid})
            print(json.dumps({"event": "report_complete", "index": local + 1, "total": len(hold_i), "citations": len(citations)}), flush=True)
    path = output / "grounded_reports.jsonl"; path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in all_reports), encoding="utf-8")
    effect = effect_interval(np.asarray(cited_effects), np.asarray(random_effects), np.asarray(effect_patients), resamples, 25000)
    valid_rate = float(np.mean([row["valid"] for row in all_reports])); grounding_gate = grounding["supported_feature_count"] >= 8 and len(grounding["supported_lesion_classes"]) >= 2
    interpretability = bool(grounding_gate and valid_rate >= .95 and effect["interval"][0] > 0)
    record = {"status": "complete", "reports": len(all_reports), "valid_reports": sum(row["valid"] for row in all_reports), "valid_report_rate": valid_rate,
              "qwen_completion_rate": float(np.mean([row["qwen_status"] == "complete" for row in all_reports])), "qwen_model": model_name,
              "cited_entries": len(cited_effects), "cited_vs_random_deletion": effect, "interpretability_success": interpretability,
              "grounding_gate_passed": grounding_gate, "reports_sha256": sha256_path(path)}
    write_json(output / "report_faithfulness_result.json", record); return record


def finalize(output: Path) -> dict:
    verify_freeze(output)
    grounding = json.loads((output / "feature_grounding_result.json").read_text()); prediction = json.loads((output / "prediction_result.json").read_text()); reports = json.loads((output / "report_faithfulness_result.json").read_text())
    result = {"schema_version": "1.0", "status": "complete", "experiment": "9B_patch_knowledge_grounding",
              "prediction_success": prediction["prediction_success"], "interpretability_success": reports["interpretability_success"],
              "grounding": {"supported_features": grounding["supported_feature_count"], "supported_lesion_classes": grounding["supported_lesion_classes"], "strong_pairs": grounding["strong_pair_count"]},
              "prediction": {"metrics": prediction["metrics"], "comparisons": prediction["comparisons"]},
              "reports": reports, "claim_boundary": "exploratory; IDRiD used for knowledge development/validation and is not external validation",
              "negative_result": not (prediction["prediction_success"] and reports["interpretability_success"]),
              "artifacts": {name: {"path": str(output / name), "sha256": sha256_path(output / name)} for name in
                            ("specification_freeze.json", "smoke_result.json", "idrid_extraction_evidence.json", "feature_grounding_result.json", "knowledge_vectors.npz", "prediction_result.json", "holdout_predictions.npz", "grounded_reports.jsonl", "report_faithfulness_result.json")}}
    write_json(output / "experiment_9b_patch_knowledge_grounding.json", result)
    lines = ["# Experiment 9B results", "", f"Status: **{result['status']}**", "", f"Prediction success: **{result['prediction_success']}**",
             f"Interpretability success: **{result['interpretability_success']}**", "", f"Supported features: {grounding['supported_feature_count']}/64",
             f"Supported lesion classes: {', '.join(grounding['supported_lesion_classes']) or 'none'}", f"Strong feature-lesion pairs: {grounding['strong_pair_count']}", "",
             "## Prediction comparisons", ""]
    for name, row in prediction["comparisons"].items(): lines.append(f"- {name}: AUROC difference {row['estimate']:.4f}, 95% CI [{row['interval'][0]:.4f}, {row['interval'][1]:.4f}]")
    lines += ["", "## Faithfulness", "", f"- Valid report rate: {reports['valid_report_rate']:.3f}",
              f"- Cited versus matched-random deletion difference: {reports['cited_vs_random_deletion']['estimate']}", "",
              "IDRiD was used for exploratory knowledge development and validation. These results are not fresh external clinical validation and do not establish clinical utility."]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment 9B Patch-SAE/Transformer knowledge grounding")
    parser.add_argument("--stage", choices=("freeze", "smoke", "extract", "ground", "predict", "report", "finalize", "all"), required=True)
    parser.add_argument("--output", default=str(OUTPUT)); parser.add_argument("--device", default="auto"); parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--qwen-model", default="qwen2.5-vl:3b"); parser.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")); parser.add_argument("--skip-qwen", action="store_true")
    args = parser.parse_args(); output = ensure_output(Path(args.output)); device = device_for(args.device)
    stages = ("freeze", "smoke", "extract", "ground", "predict", "report", "finalize") if args.stage == "all" else (args.stage,)
    for stage in stages:
        if stage == "freeze": record = freeze(output)
        elif stage == "smoke": record = smoke(output, device)
        elif stage == "extract": record = extract(output, device)
        elif stage == "ground": record = ground(output, args.bootstrap_resamples)
        elif stage == "predict": record = predict_stage(output, device, args.bootstrap_resamples)
        elif stage == "report": record = report_stage(output, device, args.bootstrap_resamples, args.qwen_model, args.ollama_url, args.skip_qwen)
        else: record = finalize(output)
        print(json.dumps({"stage": stage, "status": record.get("status"), "output": str(output)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
