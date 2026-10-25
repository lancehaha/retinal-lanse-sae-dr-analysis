from __future__ import annotations

"""Experiment 11 post-SAE stages: tile grounding, numerical cards, reports.

This runner is deliberately fail-closed.  It requires the label-free,
high-resolution Tile-SAE produced by ``run_highres_tile_sae_preflight.py``.
EyePACS grades are read only by the explicit ``predict`` / ``report`` stages.
IDRiD segmentation masks are used only to test tile-level associations and do
not make any feature a clinically verified lesion detector.
"""

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import sys
import time
import urllib.error
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from retina_lanse.data.manifest import sha256_path
from retina_lanse.encoders import load_retfound_cfp
from retina_lanse.sae.checkpoints import load_checkpoint
from scripts.run_global_sae_search import choose_device, require_external
from scripts.run_label_efficiency_external_benchmark import (
    binary_difference,
    binary_metrics,
    load_eyepacs_labels,
)
from scripts.run_patch_feature_cards_and_reports import call_ollama, parse_json, repair_json_once
import scripts.run_patch_sae_transformer_pilot as exp8
from scripts.run_highres_tile_sae_preflight import (
    CHECKPOINT,
    DATASET,
    DICTIONARY,
    EXP8,
    INPUT_DIM,
    MANIFEST,
    OUTPUT,
    PROTOCOL,
    TILE_SIZE,
    TOP_K,
    UPSTREAM,
    field_crop_and_box,
    preprocess_tile,
    selected_records,
    tile_plan,
)


IDRID = Path("/Volumes/Elements/Retina-LanSE/datasets/idrid/A. Segmentation")
GPU_LOCK = Path("/Volumes/Elements/Retina-LanSE/runtime/retina_lanse_gpu.lock")
RUNNER = Path(__file__).resolve()
LABELS = Path("/Volumes/Elements/Retina-LanSE/datasets/eyepacs/trainLabels.csv")
COHORTS = EXP8 / "patient_cohorts.json"
FEATURE_COUNT = 64
SEED = 4106
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


def ensure_output(value: str | Path) -> Path:
    path = require_external(Path(value), "Experiment 11 output").resolve()
    if path != OUTPUT.resolve():
        raise RuntimeError(f"Experiment 11 output must be exactly {OUTPUT}; got {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextlib.contextmanager
def gpu_lock(stage: str):
    GPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with GPU_LOCK.open("a+", encoding="utf-8") as handle:
        print(json.dumps({"event": "waiting_for_gpu_lock", "stage": stage}), flush=True)
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0); handle.truncate(); handle.write(json.dumps({"pid": os.getpid(), "stage": stage, "started": time.time()}) + "\n"); handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def idrid_records(limit: int | None = None) -> list[dict]:
    records: list[dict] = []
    for split, count, image_dir, mask_split, start in (
        ("training", 54, "a. Training Set", "a. Training Set", 1),
        ("testing", 27, "b. Testing Set", "b. Testing Set", 55),
    ):
        for number in range(start, start + count):
            stem = f"IDRiD_{number:02d}"
            image = IDRID / "1. Original Images" / image_dir / f"{stem}.jpg"
            if not image.is_file():
                raise RuntimeError(f"missing official IDRiD image: {image}")
            masks = {}
            for name, (folder, suffix) in MASK_LAYOUT.items():
                item = IDRID / "2. All Segmentation Groundtruths" / mask_split / folder / f"{stem}_{suffix}.tif"
                masks[name] = item if item.is_file() else None
            records.append({"stem": stem, "split": split, "image": image, "masks": masks})
    if len(records) != 81 or sum(row["split"] == "training" for row in records) != 54:
        raise RuntimeError("official IDRiD segmentation split mismatch")
    return records[:limit] if limit is not None else records


def sha_paths(paths: dict[str, Path]) -> dict:
    absent = [str(path) for path in paths.values() if not path.is_file()]
    if absent:
        raise RuntimeError(f"required input absent: {absent}")
    return {key: {"path": str(path), "sha256": sha256_path(path)} for key, path in paths.items()}


def source_hashes(output: Path, *, include_selection: bool = False) -> dict:
    paths = {
        "protocol": PROTOCOL, "post_sae_runner": RUNNER,
        "preflight_runner": Path(__file__).with_name("run_highres_tile_sae_preflight.py"),
        "tile_extraction": output / "tile_extraction_evidence.json",
        "tile_sae_result": output / "tile_sae_result.json",
        "tile_sae_checkpoint": output / "tile_sae_checkpoint.pt",
        "tile_sae_normalization": output / "tile_sae_normalization.pt",
        "experiment8_cohorts": COHORTS, "retfound_checkpoint": CHECKPOINT,
    }
    if include_selection:
        paths["tile_feature_selection"] = output / "tile_feature_selection.json"
    return sha_paths(paths)


def require_healthy_sae(output: Path) -> dict:
    result_path = output / "tile_sae_result.json"
    if not result_path.is_file():
        raise RuntimeError("high-resolution Tile-SAE has not completed")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "complete" or result.get("labels_loaded"):
        raise RuntimeError("high-resolution Tile-SAE is not a completed label-free healthy run")
    if not (output / "tile_extraction_evidence.json").is_file():
        raise RuntimeError("tile extraction evidence is absent")
    return result


def load_sae(output: Path, device: torch.device):
    require_healthy_sae(output)
    sae, payload = load_checkpoint(output / "tile_sae_checkpoint.pt")
    config = payload.get("model_config", {})
    if config != {"input_dim": INPUT_DIM, "dictionary_size": DICTIONARY, "k": TOP_K}:
        raise RuntimeError(f"unexpected Tile-SAE configuration: {config}")
    norm = torch.load(output / "tile_sae_normalization.pt", map_location="cpu", weights_only=False)
    if norm.get("labels_loaded"):
        raise RuntimeError("tile normalization is label-contaminated")
    return sae.to(device).eval(), norm["mean"].float(), norm["std"].float().clamp_min(1e-6)


@torch.inference_mode()
def topk_codes(sae, vectors: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, device: torch.device, batch: int = 512):
    values_all, ids_all = [], []
    for start in range(0, len(vectors), batch):
        normalized = ((vectors[start:start + batch] - mean) / std).to(device)
        relu = torch.relu(sae.encoder(normalized))
        values, ids = torch.topk(relu, TOP_K, dim=1)
        values_all.append(values.cpu()); ids_all.append(ids.cpu())
    return torch.cat(values_all), torch.cat(ids_all)


def tile_shards(output: Path):
    evidence = json.loads((output / "tile_extraction_evidence.json").read_text(encoding="utf-8"))
    if evidence.get("status") != "complete" or evidence.get("labels_loaded"):
        raise RuntimeError("tile extraction evidence is incomplete or label-contaminated")
    for item in evidence["shards"]:
        path = Path(item["path"])
        if not path.is_file() or sha256_path(path) != item["sha256"]:
            raise RuntimeError(f"tile shard missing or hash-mismatched: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        yield payload


def select(output: Path, device: torch.device) -> dict:
    """Rank 64 dictionary IDs by label-free activation mass on SAE-training tiles."""
    require_healthy_sae(output)
    sae, mean, std = load_sae(output, device)
    mass = torch.zeros(DICTIONARY, dtype=torch.float64); count = torch.zeros(DICTIONARY, dtype=torch.int64)
    train_tiles = 0
    with gpu_lock("label_free_tile_feature_selection"):
        for payload in tile_shards(output):
            keep = [index for index, group in enumerate(payload["cohorts"]) if group == "sae_train"]
            if not keep:
                continue
            offsets = payload["tile_offsets"].tolist()
            pieces = [payload["tile_embeddings"][offsets[index]:offsets[index + 1]].float() for index in keep]
            vectors = torch.cat(pieces); values, ids = topk_codes(sae, vectors, mean, std, device)
            mass.scatter_add_(0, ids.reshape(-1), values.reshape(-1).double())
            count.scatter_add_(0, ids.reshape(-1), torch.ones_like(ids.reshape(-1), dtype=torch.int64))
            train_tiles += len(vectors)
    eligible = torch.where(count >= 100)[0]
    if len(eligible) < FEATURE_COUNT:
        raise RuntimeError(f"only {len(eligible)} dictionary features have >=100 label-free activations")
    order = sorted(eligible.tolist(), key=lambda feature: (-float(mass[feature]), int(feature)))
    feature_ids = order[:FEATURE_COUNT]
    record = {"status": "complete", "labels_loaded": False, "selection": "top activation mass on SAE-training tiles only",
              "feature_ids": feature_ids, "feature_count": len(feature_ids), "sae_training_tiles": train_tiles,
              "statistics": [{"feature_id": int(feature), "activation_mass": float(mass[feature]), "activation_count": int(count[feature])} for feature in feature_ids],
              "sources": source_hashes(output)}
    write_json(output / "tile_feature_selection.json", record)
    return record


def freeze(output: Path) -> dict:
    selection = json.loads((output / "tile_feature_selection.json").read_text(encoding="utf-8"))
    if selection.get("status") != "complete" or selection.get("labels_loaded") or len(selection.get("feature_ids", [])) != FEATURE_COUNT:
        raise RuntimeError("label-free 64-feature selection must complete before freezing Experiment 11 grounding")
    inputs = source_hashes(output, include_selection=True)
    files = [row["image"] for row in idrid_records()] + [path for row in idrid_records() for path in row["masks"].values() if path is not None]
    record = {"schema_version": "1.0", "status": "frozen_before_grounding", "experiment": "11_highres_tile_knowledge_cards",
              "representation": {"source_tile_px": TILE_SIZE, "retfound_input_px": 224, "dictionary": DICTIONARY, "top_k": TOP_K},
              "selected_feature_ids": selection["feature_ids"], "selection_labels_loaded": False,
              "idrid_role": "feature-grounding development and held-out mask validation; not external DR validation",
              "idrid_counts": {"training": 54, "testing": 27},
              "idrid_bundle_sha256": hashlib.sha256("".join(sha256_path(path) for path in sorted(files)).encode()).hexdigest(),
              "thresholds": {"train_bh_q_lt": .05, "validation_auroc_min": .65, "validation_lift_min": 2.0, "minimum_grounded_features": 8, "minimum_lesion_classes": 2},
              "seeds": {"grounding": SEED, "knowledge_shuffle": SHUFFLE_SEED, "prediction": SEED},
              "sources": inputs, "eyepacs_grades_loaded": False}
    path = output / "knowledge_grounding_specification.json"
    if path.is_file() and json.loads(path.read_text(encoding="utf-8")) != record:
        raise RuntimeError("Experiment 11 grounding specification would change after freeze")
    write_json(path, record)
    return record


def verify_freeze(output: Path) -> dict:
    path = output / "knowledge_grounding_specification.json"
    if not path.is_file():
        raise RuntimeError("grounding specification has not been frozen")
    frozen = json.loads(path.read_text(encoding="utf-8"))
    if frozen.get("sources") != source_hashes(output, include_selection=True):
        raise RuntimeError("a frozen Experiment 11 source hash changed")
    return frozen


def tile_mask_labels(record: dict, plans: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Return tile-level lesion/optic indicators directly from original-resolution masks."""
    with Image.open(record["image"]) as source:
        # Segmentation masks are in the original image coordinate frame, as are
        # ``source_box`` values emitted by the high-resolution tiler.
        image_shape = (source.height, source.width)
    masks = []
    for name in (*LESIONS, "optic_disc"):
        source = record["masks"][name]
        if source is None:
            masks.append(np.zeros(image_shape, dtype=bool))
        else:
            with Image.open(source) as item:
                mask = np.asarray(item.convert("L")) > 0
            if mask.shape != image_shape:
                raise RuntimeError(f"mask/image mismatch for {record['stem']} {name}: {mask.shape} != {image_shape}")
            masks.append(mask)
    labels, coverage = [], []
    for plan in plans:
        x0, y0, x1, y1 = plan["source_box"]
        # Source box is in full image coordinates; translate it into original mask coordinates.
        lesion = [bool(mask[y0:y1, x0:x1].any()) for mask in masks]
        labels.append(lesion); coverage.append(float(plan["retinal_coverage"]))
    return np.asarray(labels, dtype=bool), np.asarray(coverage, dtype=np.float32)


@torch.inference_mode()
def idrid_extract(records: list[dict], output: Path, device: torch.device) -> dict:
    frozen = verify_freeze(output); sae, mean, std = load_sae(output, device)
    selected = [int(value) for value in frozen["selected_feature_ids"]]; lookup = {value: index for index, value in enumerate(selected)}
    encoder = load_retfound_cfp(CHECKPOINT, UPSTREAM, representation="final_mean_patch").to(device).eval()
    activation_rows, label_rows, coverage_rows, centers, offsets, stems, splits, source_boxes = [], [], [], [], [0], [], [], []
    with gpu_lock("idrid_highres_tile_extraction"):
        for index, record in enumerate(records):
            with Image.open(record["image"]) as source:
                field, _, plans = tile_plan(source.convert("RGB"))
                tensors = torch.stack([preprocess_tile(field, plan) for plan in plans])
            embedding = []
            for start in range(0, len(tensors), 16):
                embedding.append(encoder(tensors[start:start + 16].to(device)).float().cpu())
            vectors = torch.cat(embedding)
            values, ids = topk_codes(sae, vectors, mean, std, device)
            selected_activation = torch.zeros((len(vectors), len(selected)), dtype=torch.float32)
            for rank in range(TOP_K):
                for feature, column in lookup.items():
                    hit = ids[:, rank] == feature
                    if hit.any(): selected_activation[hit, column] = values[hit, rank]
            labels, coverage = tile_mask_labels(record, plans)
            boxes = np.asarray([plan["source_box"] for plan in plans], dtype=np.int32)
            # Normalized centres are used only to match controls within an image.
            with Image.open(record["image"]) as source:
                width, height = source.size
            centre = np.stack([(boxes[:, 0] + boxes[:, 2]) / (2 * width), (boxes[:, 1] + boxes[:, 3]) / (2 * height)], axis=1)
            activation_rows.append(selected_activation.numpy()); label_rows.append(labels); coverage_rows.append(coverage); centers.append(centre.astype(np.float32)); source_boxes.append(boxes)
            offsets.append(offsets[-1] + len(vectors)); stems.append(record["stem"]); splits.append(record["split"])
            print(json.dumps({"event": "idrid_tile_extracted", "index": index + 1, "total": len(records), "tiles": len(vectors)}), flush=True)
    return {"activations": np.concatenate(activation_rows), "lesion_labels": np.concatenate(label_rows), "coverage": np.concatenate(coverage_rows),
            "centres": np.concatenate(centers), "source_boxes": np.concatenate(source_boxes), "image_offsets": np.asarray(offsets, dtype=np.int64),
            "stems": np.asarray(stems), "splits": np.asarray(splits), "feature_ids": np.asarray(selected, dtype=np.int32),
            "retfound_checkpoint_hash": encoder.checkpoint_hash}


def ground_smoke(output: Path, device: torch.device) -> dict:
    verify_freeze(output)
    records = idrid_records(limit=3); arrays = idrid_extract(records, output, device)
    expected = sum(np.diff(arrays["image_offsets"]))
    if arrays["activations"].shape != (expected, FEATURE_COUNT) or arrays["lesion_labels"].shape != (expected, 5):
        raise RuntimeError("IDRiD high-resolution tile extraction shape smoke failed")
    if not np.isfinite(arrays["activations"]).all() or not np.isfinite(arrays["centres"]).all():
        raise RuntimeError("IDRiD high-resolution tile extraction produced non-finite values")
    record = {"status": "passed", "images": len(records), "tiles": int(expected), "selected_activation_shape": list(arrays["activations"].shape),
              "mask_shape": list(arrays["lesion_labels"].shape), "eyepacs_grades_loaded": False,
              "tests": ["original_resolution_tile_mask_overlap", "variable_tile_offsets", "selected_feature_alignment", "finite_activations"]}
    write_json(output / "knowledge_grounding_smoke.json", record); return record


def ground_extract(output: Path, device: torch.device) -> dict:
    verify_freeze(output)
    smoke = json.loads((output / "knowledge_grounding_smoke.json").read_text(encoding="utf-8"))
    if smoke.get("status") != "passed":
        raise RuntimeError("high-resolution grounding smoke must pass first")
    arrays = idrid_extract(idrid_records(), output, device)
    path = output / "idrid_highres_tile_grounding_arrays.npz"; np.savez_compressed(path, **arrays)
    record = {"status": "complete", "images": 81, "training_images": 54, "testing_images": 27, "tiles": int(len(arrays["activations"])),
              "selected_activation_shape": list(arrays["activations"].shape), "array_sha256": sha256_path(path),
              "retfound_checkpoint_hash": arrays["retfound_checkpoint_hash"], "eyepacs_grades_loaded": False}
    write_json(output / "idrid_highres_tile_extraction_evidence.json", record); return record


def matched_rows(labels: np.ndarray, centres: np.ndarray, offsets: np.ndarray, lesion_index: int, seed: int):
    images: list[int] = []; tiles: list[int] = []; targets: list[int] = []
    for image in range(len(offsets) - 1):
        start, end = int(offsets[image]), int(offsets[image + 1]); local = labels[start:end]
        positive = np.where(local[:, lesion_index])[0]
        negative = np.where((~local[:, :4].any(axis=1)) & (~local[:, 4]))[0]
        if not len(positive) or not len(negative):
            continue
        # Nearest unselected normal tile in the same image is the strongest available spatial control.
        used: set[int] = set()
        for pos in positive.tolist():
            distance = ((centres[start + negative] - centres[start + pos]) ** 2).sum(axis=1)
            ranked = sorted(range(len(negative)), key=lambda j: (float(distance[j]), hashlib.sha256(f"{seed}:{image}:{pos}:{int(negative[j])}".encode()).hexdigest()))
            candidate = next((int(negative[j]) for j in ranked if int(negative[j]) not in used), None)
            if candidate is None:
                continue
            used.add(candidate); images.extend([image, image]); tiles.extend([start + pos, start + candidate]); targets.extend([1, 0])
    return np.asarray(images), np.asarray(tiles), np.asarray(targets, dtype=np.int8)


def auc_lift(y: np.ndarray, score: np.ndarray) -> tuple[float, float, float]:
    if len(y) == 0 or np.unique(y).size != 2:
        return float("nan"), float("nan"), float("nan")
    ap = float(average_precision_score(y, score)); prevalence = float(y.mean())
    return float(roc_auc_score(y, score)), ap, ap / prevalence


def cluster_bootstrap(y: np.ndarray, score: np.ndarray, images: np.ndarray, resamples: int, seed: int) -> dict:
    estimate = auc_lift(y, score); groups = {value: np.where(images == value)[0] for value in np.unique(images)}; keys = list(groups)
    values = []; rng = np.random.default_rng(seed)
    for _ in range(resamples):
        choice = rng.choice(keys, len(keys), replace=True); index = np.concatenate([groups[value] for value in choice])
        item = auc_lift(y[index], score[index])
        if np.isfinite(item).all(): values.append(item)
    boot = np.asarray(values)
    lo, hi = (np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)) if len(boot) else (np.full(3, np.nan), np.full(3, np.nan))
    return {"auroc": estimate[0], "auprc": estimate[1], "auprc_lift": estimate[2], "prevalence": float(y.mean()) if len(y) else float("nan"),
            "auroc_interval": [float(lo[0]), float(hi[0])], "lift_interval": [float(lo[2]), float(hi[2])],
            "one_sided_auroc_p": float((1 + int((boot[:, 0] <= .5).sum())) / (len(boot) + 1)) if len(boot) else 1.0,
            "valid_bootstraps": int(len(boot)), "images": len(keys), "rows": len(y)}


def bh_adjust(pvalues: list[float]) -> list[float]:
    p = np.asarray(pvalues, dtype=float); order = np.argsort(p); result = np.empty(len(p)); running = 1.0
    for rank_index in range(len(p) - 1, -1, -1):
        original, rank = order[rank_index], rank_index + 1
        running = min(running, float(p[original]) * len(p) / rank); result[original] = min(1., running)
    return result.tolist()


def ground(output: Path, resamples: int) -> dict:
    frozen = verify_freeze(output); evidence = json.loads((output / "idrid_highres_tile_extraction_evidence.json").read_text(encoding="utf-8"))
    array_path = output / "idrid_highres_tile_grounding_arrays.npz"
    if evidence.get("array_sha256") != sha256_path(array_path):
        raise RuntimeError("IDRiD high-resolution grounding array hash mismatch")
    data = np.load(array_path); activations, labels, centres, offsets = data["activations"], data["lesion_labels"], data["centres"], data["image_offsets"]
    splits, feature_ids = data["splits"], data["feature_ids"].astype(int).tolist(); tests, pvalues = [], []
    tile_image = np.concatenate([np.full(offsets[i + 1] - offsets[i], i, dtype=np.int64) for i in range(len(offsets) - 1)])
    for lesion_index, lesion in enumerate(LESIONS):
        image, rows, target = matched_rows(labels, centres, offsets, lesion_index, SEED + lesion_index)
        for feature_index, feature in enumerate(feature_ids):
            row = {"feature_id": feature, "lesion": lesion}
            for split, key in (("training", "training"), ("testing", "validation")):
                keep = splits[image] == split
                row[key] = cluster_bootstrap(target[keep], activations[rows[keep], feature_index], image[keep], resamples, 5000 + lesion_index * 100 + feature_index + (10000 if split == "testing" else 0))
            pvalues.append(row["training"]["one_sided_auroc_p"]); tests.append(row)
    for row, q in zip(tests, bh_adjust(pvalues)):
        row["training_bh_q"] = q; train, valid = row["training"], row["validation"]
        row["supported"] = bool(q < .05 and valid["auroc"] >= .65 and valid["auprc_lift"] >= 2.0 and train["auroc"] > .5)
        row["strong"] = bool(row["supported"] and valid["auroc_interval"][0] > .5 and valid["lift_interval"][0] > 1.0)
    by_feature = {feature: {"feature_id": feature, "lesions": {}} for feature in feature_ids}
    for item in tests: by_feature[item["feature_id"]]["lesions"][item["lesion"]] = item
    vectors = np.zeros((DICTIONARY, 5), dtype=np.float32)
    for feature in feature_ids:
        support = [item for item in by_feature[feature]["lesions"].values() if item["supported"]]
        vector = np.zeros(5, dtype=np.float32)
        for index, lesion in enumerate(LESIONS):
            item = by_feature[feature]["lesions"][lesion]
            if item["supported"]: vector[index] = float(np.clip(2 * (item["validation"]["auroc"] - .5), 0., 1.))
        if support: vector[4] = float(max(np.clip((item["validation"]["auroc"] - .5) / .5, 0., 1.) * (1 - item["training_bh_q"]) for item in support))
        by_feature[feature]["grounded"] = bool(support); by_feature[feature]["knowledge_vector"] = vector.tolist(); vectors[feature] = vector
    vector_path = output / "highres_tile_knowledge_vectors.npz"; np.savez_compressed(vector_path, dictionary_vectors=vectors, feature_ids=np.asarray(feature_ids, dtype=np.int32))
    supported = [feature for feature in feature_ids if by_feature[feature]["grounded"]]; classes = sorted({item["lesion"] for item in tests if item["supported"]})
    record = {"status": "complete", "feature_lesion_tests": tests, "features": [by_feature[feature] for feature in feature_ids],
              "supported_features": supported, "supported_feature_count": len(supported), "supported_lesion_classes": classes,
              "strong_pair_count": int(sum(item["strong"] for item in tests)), "grounding_gate_passed": len(supported) >= 8 and len(classes) >= 2,
              "knowledge_vectors_sha256": sha256_path(vector_path), "eyepacs_grades_loaded": False}
    write_json(output / "highres_tile_feature_grounding.json", record); return record


def image_feature_aggregates(output: Path, device: torch.device) -> dict:
    """Produce [image,64] log-sum activation evidence without loading EyePACS grades."""
    frozen = verify_freeze(output); selected = [int(item) for item in frozen["selected_feature_ids"]]; lookup = {item: index for index, item in enumerate(selected)}
    sae, mean, std = load_sae(output, device); rows = []
    paths: list[str] = []; patients: list[str] = []; cohorts: list[str] = []; boxes: list[np.ndarray] = []
    all_activations: list[np.ndarray] = []; tile_offsets = [0]
    with gpu_lock("label_free_image_tile_aggregate"):
        for payload in tile_shards(output):
            offsets = payload["tile_offsets"].tolist(); code_values, code_ids = topk_codes(sae, payload["tile_embeddings"].float(), mean, std, device)
            dense = torch.zeros((len(code_values), FEATURE_COUNT), dtype=torch.float32)
            for rank in range(TOP_K):
                for feature, column in lookup.items():
                    hit = code_ids[:, rank] == feature
                    if hit.any(): dense[hit, column] = code_values[hit, rank]
            for index in range(len(payload["relative_paths"])):
                start, end = offsets[index], offsets[index + 1]
                current = dense[start:end].numpy().astype(np.float32)
                rows.append(np.log1p(current.sum(0)))
                paths.append(payload["relative_paths"][index]); patients.append(str(payload["patient_ids"][index])); cohorts.append(payload["cohorts"][index]); boxes.append(payload["source_boxes"][start:end].numpy())
                all_activations.append(current); tile_offsets.append(tile_offsets[-1] + len(current))
    result = {"aggregate": np.stack(rows).astype(np.float32), "paths": np.asarray(paths), "patients": np.asarray(patients), "cohorts": np.asarray(cohorts),
              "tile_activations": np.concatenate(all_activations), "tile_offsets": np.asarray(tile_offsets, dtype=np.int64),
              "source_boxes": np.concatenate(boxes), "feature_ids": np.asarray(selected, dtype=np.int32)}
    if not (len(result["paths"]) + 1 == len(result["tile_offsets"]) and result["tile_activations"].shape[0] == result["source_boxes"].shape[0] == result["tile_offsets"][-1]):
        raise RuntimeError("image/tile evidence alignment failed")
    return result


def fit_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray):
    mean = train_x.mean(0, keepdims=True); std = train_x.std(0, keepdims=True); std[std < 1e-6] = 1.
    model = LogisticRegression(C=1., class_weight="balanced", solver="lbfgs", max_iter=2000, random_state=SEED)
    model.fit((train_x - mean) / std, train_y)
    return model, mean, std, model.predict_proba((test_x - mean) / std)[:, 1]


def predict(output: Path, device: torch.device, resamples: int) -> dict:
    frozen = verify_freeze(output); grounding = json.loads((output / "highres_tile_feature_grounding.json").read_text(encoding="utf-8"))
    if not grounding.get("grounding_gate_passed"):
        raise RuntimeError("knowledge-card prediction is blocked: high-resolution lesion-grounding gate failed")
    aggregate = image_feature_aggregates(output, device)
    evidence_path = output / "highres_tile_image_card_evidence.npz"
    np.savez_compressed(evidence_path, **aggregate)
    # Grades become available only here, after all representation and grounding choices are frozen.
    labels = load_eyepacs_labels(LABELS); grade = np.asarray([labels.get(Path(path).stem.lower(), -1) for path in aggregate["paths"]])
    if (grade < 0).any(): raise RuntimeError("an Experiment 8 cohort image lacks an EyePACS grade")
    patch = exp8.load_patch_run(EXP8 / "patch_tokens")
    dense_lookup = {path: vector.numpy().astype(np.float32) for path, vector in zip(patch["paths"], patch["global"])}
    if set(aggregate["paths"]) != set(dense_lookup): raise RuntimeError("high-resolution tile and frozen dense paths do not align")
    dense = np.stack([dense_lookup[path] for path in aggregate["paths"]])
    groups = {name: np.where(aggregate["cohorts"] == name)[0] for name in ("sae_train", "development", "pilot_holdout")}
    train, dev, hold = groups["sae_train"], groups["development"], groups["pilot_holdout"]
    if set(aggregate["patients"][train]) & set(aggregate["patients"][dev]) or set(aggregate["patients"][hold]) & (set(aggregate["patients"][train]) | set(aggregate["patients"][dev])):
        raise RuntimeError("Experiment 11 patient separation failed")
    final = np.concatenate([train, dev]); target = (grade >= 2).astype(np.int64); cards = aggregate["aggregate"]
    knowledge = np.load(output / "highres_tile_knowledge_vectors.npz")["dictionary_vectors"][aggregate["feature_ids"]]
    knowledge_channels = cards @ knowledge
    shuffled = knowledge[np.random.default_rng(SHUFFLE_SEED).permutation(FEATURE_COUNT)]; shuffled_channels = cards @ shuffled
    arms = {"dense_retfound": dense, "dense_plus_numeric_cards": np.concatenate([dense, cards], axis=1),
            "dense_plus_numeric_cards_plus_grounded_knowledge": np.concatenate([dense, cards, knowledge_channels], axis=1),
            "dense_plus_numeric_cards_plus_shuffled_knowledge": np.concatenate([dense, cards, shuffled_channels], axis=1)}
    models, predictions, contributions = {}, {}, {}
    for name, matrix in arms.items():
        model, mean, std, score = fit_predict(matrix[final], target[final], matrix[hold]); models[name] = {"mean": mean, "std": std, "coef": model.coef_[0], "intercept": model.intercept_}; predictions[name] = score
        if name == "dense_plus_numeric_cards":
            contribution = ((cards[hold] - mean[:, 1024:]) / std[:, 1024:]) * model.coef_[0, 1024:]
            contributions = {"feature_ids": aggregate["feature_ids"].tolist(), "per_image_conditional_logit_contribution": contribution.astype(np.float32).tolist()}
    metrics = {name: binary_metrics(target[hold], score, aggregate["patients"][hold], resamples, 23000 + index) for index, (name, score) in enumerate(predictions.items())}
    comparisons = {name: binary_difference(target[hold], score, predictions["dense_retfound"], aggregate["patients"][hold], roc_auc_score, resamples, 24000 + index) for index, (name, score) in enumerate(predictions.items()) if name != "dense_retfound"}
    prediction_path = output / "highres_tile_card_predictions.npz"; np.savez_compressed(prediction_path, **{name: value.astype(np.float32) for name, value in predictions.items()},
        targets=target[hold], paths=aggregate["paths"][hold], patients=aggregate["patients"][hold], hold_indices=hold)
    model_path = output / "highres_tile_card_models.npz"
    np.savez_compressed(model_path, **{f"{name}_{field}": value.astype(np.float32) for name, values in models.items() for field, value in values.items()})
    record = {"status": "complete", "endpoint": "exploratory referable DR, EyePACS grade >=2; previously accessed pilot holdout", "eyepacs_grades_loaded": True,
              "arms": list(arms), "metrics": metrics, "paired_auroc_differences_vs_dense": comparisons,
              "conditional_card_contributions": contributions, "prediction_sha256": sha256_path(prediction_path),
              "image_tile_evidence_sha256": sha256_path(evidence_path), "model_parameters_sha256": sha256_path(model_path),
              "holdout": {"images": int(len(hold)), "patients": int(len(set(aggregate["patients"][hold]))), "paths_sha256": hashlib.sha256("\n".join(aggregate["paths"][hold]).encode()).hexdigest()},
              "knowledge_mapping": "only tile features that passed held-out IDRiD mask association; shuffled control permutes only feature-to-knowledge mapping"}
    write_json(output / "highres_tile_card_prediction_result.json", record); return record


def probability(raw: np.ndarray, mean: np.ndarray, std: np.ndarray, coefficient: np.ndarray, intercept: np.ndarray) -> float:
    logit = float(((raw - mean[0]) / std[0]).dot(coefficient) + intercept.reshape(-1)[0])
    return float(1.0 / (1.0 + np.exp(-np.clip(logit, -35., 35.))))


def save_crop(relative_path: str, box: np.ndarray, destination: Path) -> Path:
    if destination.is_file():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = DATASET / relative_path
    if not source.is_file():
        raise RuntimeError(f"source image for report crop is absent: {source}")
    with Image.open(source) as image:
        rgb = image.convert("RGB"); x0, y0, x1, y1 = [int(value) for value in box]
        crop = rgb.crop((max(0, x0), max(0, y0), min(rgb.width, x1), min(rgb.height, y1)))
        if crop.width < 1 or crop.height < 1:
            raise RuntimeError(f"invalid report crop: {relative_path} {box.tolist()}")
        crop.resize((448, 448), Image.Resampling.LANCZOS).save(destination, format="JPEG", quality=90)
    return destination


def patient_mean_difference(cited: np.ndarray, random: np.ndarray, patients: np.ndarray, resamples: int) -> dict:
    if not len(cited):
        return {"estimate": float("nan"), "interval": [float("nan"), float("nan")], "valid_resamples": 0}
    groups = {patient: np.where(patients == patient)[0] for patient in np.unique(patients)}; keys = list(groups); rng = np.random.default_rng(25000); values = []
    for _ in range(resamples):
        sample = rng.choice(keys, len(keys), replace=True); index = np.concatenate([groups[patient] for patient in sample])
        values.append(float(np.mean(cited[index] - random[index])))
    return {"estimate": float(np.mean(cited - random)), "interval": np.percentile(values, [2.5, 97.5]).tolist(), "valid_resamples": len(values)}


def report(output: Path, resamples: int, model_name: str, base_url: str, skip_qwen: bool) -> dict:
    """Produce score-locked card reports; optional Qwen can narrate but not score."""
    verify_freeze(output)
    grounding = json.loads((output / "highres_tile_feature_grounding.json").read_text(encoding="utf-8"))
    if not grounding.get("grounding_gate_passed"):
        raise RuntimeError("structured reports are blocked by the high-resolution grounding gate")
    prediction = json.loads((output / "highres_tile_card_prediction_result.json").read_text(encoding="utf-8"))
    evidence_path = output / "highres_tile_image_card_evidence.npz"
    model_path = output / "highres_tile_card_models.npz"
    if prediction.get("image_tile_evidence_sha256") != sha256_path(evidence_path) or prediction.get("model_parameters_sha256") != sha256_path(model_path):
        raise RuntimeError("report inputs do not match the frozen numerical prediction")
    evidence = np.load(evidence_path, allow_pickle=False); model = np.load(model_path, allow_pickle=False)
    paths, patients, cohorts = evidence["paths"].astype(str), evidence["patients"].astype(str), evidence["cohorts"].astype(str)
    feature_ids, cards, offsets, tile_values, boxes = evidence["feature_ids"].astype(int), evidence["aggregate"], evidence["tile_offsets"], evidence["tile_activations"], evidence["source_boxes"]
    dense_patch = exp8.load_patch_run(EXP8 / "patch_tokens")
    dense_lookup = {path: vector.numpy().astype(np.float32) for path, vector in zip(dense_patch["paths"], dense_patch["global"])}
    feature_grounding = {int(item["feature_id"]): item for item in grounding["features"] if item["grounded"]}
    hold = np.where(cohorts == "pilot_holdout")[0]
    if set(paths[hold]) != set(np.load(output / "highres_tile_card_predictions.npz")["paths"].astype(str)):
        raise RuntimeError("report holdout paths do not match frozen prediction paths")
    mean, std = model["dense_plus_numeric_cards_mean"], model["dense_plus_numeric_cards_std"]
    coefficient, intercept = model["dense_plus_numeric_cards_coef"], model["dense_plus_numeric_cards_intercept"]
    report_root = output / "highres_tile_reports"; crop_root = report_root / "crops"; entries = []; cited_effects = []; random_effects = []; effect_patients = []
    qwen_available = not skip_qwen
    for local_index, image_index in enumerate(hold):
        raw = np.concatenate([dense_lookup[paths[image_index]], cards[image_index]])
        base = probability(raw, mean, std, coefficient, intercept)
        normalized_card = (cards[image_index] - mean[0, INPUT_DIM:]) / std[0, INPUT_DIM:]
        conditional = normalized_card * coefficient[INPUT_DIM:]
        candidates = []
        for feature_column in np.argsort(-conditional, kind="stable"):
            feature = int(feature_ids[feature_column])
            if feature not in feature_grounding or cards[image_index, feature_column] <= 0:
                continue
            deleted = raw.copy(); deleted[INPUT_DIM + feature_column] = 0.
            delta = base - probability(deleted, mean, std, coefficient, intercept)
            if delta <= 0:
                continue
            start, end = int(offsets[image_index]), int(offsets[image_index + 1])
            tile_index = start + int(np.argmax(tile_values[start:end, feature_column]))
            crop = save_crop(paths[image_index], boxes[tile_index], crop_root / f"{local_index:04d}_f{feature}_tile{tile_index - start}.jpg")
            grounded = feature_grounding[feature]
            candidates.append({"feature_id": feature, "conditional_logit_contribution": float(conditional[feature_column]),
                               "image_activation_logsum": float(cards[image_index, feature_column]), "max_tile_activation": float(tile_values[tile_index, feature_column]),
                               "source_box": boxes[tile_index].astype(int).tolist(), "crop_path": str(crop), "deletion_score_change": float(delta),
                               "knowledge_vector": grounded["knowledge_vector"],
                               "supported_mask_associations": [name for name, item in grounded["lesions"].items() if item["supported"]],
                               "uncertainty": "Mask-grounded association, not specialist-verified lesion identity."})
            if len(candidates) == 5:
                break
        citations = candidates
        for citation_index, citation in enumerate(citations):
            candidate_columns = [column for column, feature in enumerate(feature_ids) if column != int(np.where(feature_ids == citation["feature_id"])[0][0]) and cards[image_index, column] > 0]
            if not candidate_columns:
                continue
            random_column = sorted(candidate_columns, key=lambda value: hashlib.sha256(f"{SHUFFLE_SEED}:{image_index}:{citation_index}:{value}".encode()).hexdigest())[0]
            deleted = raw.copy(); deleted[INPUT_DIM + random_column] = 0.
            random_delta = base - probability(deleted, mean, std, coefficient, intercept)
            citation["matched_random_feature_id"] = int(feature_ids[random_column]); citation["matched_random_deletion_score_change"] = float(random_delta)
            cited_effects.append(citation["deletion_score_change"]); random_effects.append(random_delta); effect_patients.append(patients[image_index])
        structured = {"schema_version": "1.0", "relative_path": paths[image_index], "referable_dr_probability": base,
                      "numerical_model": "frozen dense RETFound plus high-resolution tile-card logistic probe", "supporting_evidence": citations,
                      "conflicting_or_normal_evidence": [], "abstained": len(citations) == 0,
                      "scope_warning": "Feature associations are segmentation-mask grounded but are not clinically verified lesion or diagnosis claims."}
        prompt = ("Write a cautious retinal-image research narrative from the supplied locked structured report and its visual crops. "
                  "The numerical probability is fixed and you must not alter it. Every observation must cite one supplied feature_id. "
                  "Do not infer an unseen label, diagnosis, grade, or a clinically confirmed lesion. Return JSON only with keys summary, evidence_feature_ids, uncertainty.\n"
                  + json.dumps(structured, sort_keys=True))
        raw_qwen, parsed, qwen_status = "", None, "skipped"
        if qwen_available:
            try:
                raw_qwen = call_ollama(base_url, model_name, prompt, [[Path(item["crop_path"]) for item in citations]], 180)
                parsed = parse_json(raw_qwen)
                if parsed is None:
                    raw_qwen = repair_json_once(base_url, model_name, raw_qwen, 60); parsed = parse_json(raw_qwen)
                returned = set(parsed.get("evidence_feature_ids", [])) if isinstance(parsed, dict) else set()
                qwen_status = "complete" if isinstance(parsed, dict) and returned <= {item["feature_id"] for item in citations} else "invalid"
            except (urllib.error.URLError, TimeoutError, OSError, KeyError, ValueError) as exc:
                raw_qwen = repr(exc); qwen_status = "unavailable"; qwen_available = False
        entries.append({"structured": structured, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "qwen_raw": raw_qwen, "qwen_parsed": parsed, "qwen_status": qwen_status,
                        "valid": all(item["deletion_score_change"] > 0 and item["feature_id"] in feature_grounding for item in citations)})
        print(json.dumps({"event": "highres_tile_report_complete", "index": local_index + 1, "total": len(hold), "citations": len(citations)}), flush=True)
    report_path = report_root / "structured_reports.jsonl"; report_root.mkdir(parents=True, exist_ok=True); report_path.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in entries), encoding="utf-8")
    effect = patient_mean_difference(np.asarray(cited_effects), np.asarray(random_effects), np.asarray(effect_patients), resamples)
    record = {"status": "complete", "reports": len(entries), "valid_reports": int(sum(item["valid"] for item in entries)),
              "valid_report_rate": float(np.mean([item["valid"] for item in entries])), "qwen_completion_rate": float(np.mean([item["qwen_status"] == "complete" for item in entries])),
              "qwen_model": model_name, "cited_vs_matched_random_deletion": effect,
              "faithfulness_gate_passed": bool(effect["interval"][0] > 0 and np.mean([item["valid"] for item in entries]) >= .95),
              "reports_sha256": sha256_path(report_path), "prediction_is_unchanged": True}
    write_json(output / "highres_tile_report_result.json", record); return record


def finalize(output: Path) -> dict:
    frozen = verify_freeze(output); grounding = json.loads((output / "highres_tile_feature_grounding.json").read_text(encoding="utf-8"))
    prediction = json.loads((output / "highres_tile_card_prediction_result.json").read_text(encoding="utf-8"))
    reports = json.loads((output / "highres_tile_report_result.json").read_text(encoding="utf-8"))
    outcome = {"schema_version": "1.0", "status": "complete", "experiment": "11_highres_tile_knowledge_cards",
               "grounding": {"gate_passed": grounding["grounding_gate_passed"], "supported_features": grounding["supported_feature_count"], "lesion_classes": grounding["supported_lesion_classes"]},
               "prediction": {"metrics": prediction["metrics"], "comparisons": prediction["paired_auroc_differences_vs_dense"]},
               "reports": reports,
               "claim_boundary": "Exploratory reused pilot holdout. Grounded associations are not specialist-verified lesion identity or clinical-deployment evidence."}
    write_json(output / "experiment_11_highres_tile_knowledge_cards.json", outcome)
    lines = ["# Experiment 11 results", "", "Status: **complete**", "", f"Grounded features: {grounding['supported_feature_count']}/64", "", "## Pilot AUROC", ""]
    for name, metric in prediction["metrics"].items(): lines.append(f"- {name}: {metric['auroc']['estimate']:.4f}")
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return outcome


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment 11 high-resolution tile knowledge-card stages")
    parser.add_argument("--stage", required=True, choices=("select", "freeze", "ground_smoke", "ground_extract", "ground", "predict", "report", "finalize", "all"))
    parser.add_argument("--output", default=str(OUTPUT)); parser.add_argument("--device", default="auto"); parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--qwen-model", default="qwen2.5-vl:3b"); parser.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")); parser.add_argument("--skip-qwen", action="store_true")
    args = parser.parse_args(); output = ensure_output(args.output); device = choose_device(args.device)
    stages = ("select", "freeze", "ground_smoke", "ground_extract", "ground", "predict", "report", "finalize") if args.stage == "all" else (args.stage,)
    for stage in stages:
        if stage == "select": record = select(output, device)
        elif stage == "freeze": record = freeze(output)
        elif stage == "ground_smoke": record = ground_smoke(output, device)
        elif stage == "ground_extract": record = ground_extract(output, device)
        elif stage == "ground": record = ground(output, args.bootstrap_resamples)
        elif stage == "predict": record = predict(output, device, args.bootstrap_resamples)
        elif stage == "report": record = report(output, args.bootstrap_resamples, args.qwen_model, args.ollama_url, args.skip_qwen)
        else: record = finalize(output)
        print(json.dumps({"stage": stage, "status": record.get("status"), "output": str(output)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
