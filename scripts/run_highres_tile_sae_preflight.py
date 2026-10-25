from __future__ import annotations

"""Experiment 11 high-resolution retinal-tile extraction and label-free SAE.

The prior 224-pixel whole-field Patch-SAE had no lesion-grounded features. This
runner intentionally changes the *local input scale*, never the frozen RETFound
encoder, and keeps all image labels out of tiling, normalisation, and SAE fit.
"""

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image

from retina_lanse.data.manifest import sha256_path
from retina_lanse.data.preprocessing import normalize_retfound_cfp
from retina_lanse.encoders import load_retfound_cfp
from retina_lanse.evaluation.metrics import evaluate_embeddings_batched
from retina_lanse.sae.checkpoints import load_checkpoint, save_checkpoint
from retina_lanse.sae.model import TopKSAE
from retina_lanse.sae.trainer import train_sae_batched
from retina_lanse.utils import set_deterministic
from scripts.extract_retfound_embeddings import read_jsonl
from scripts.run_global_sae_search import choose_device, require_external


EXTERNAL = Path("/Volumes/Elements/Retina-LanSE")
OUTPUT = EXTERNAL / "experiments/experiment_11_highres_tile_knowledge_cards_20260918"
EXP8 = EXTERNAL / "experiments/experiment_8_patch_sae_grounded_report_pilot_20260913"
COHORTS = EXP8 / "patient_cohorts.json"
MANIFEST = EXTERNAL / "datasets/eyepacs/manifest_publication_v1.jsonl"
DATASET = EXTERNAL / "datasets/eyepacs"
CHECKPOINT = EXTERNAL / "model_weights/retfound_cfp/RETFound_mae_natureCFP.pth"
UPSTREAM = EXTERNAL / "runtime/RETFound"
PROTOCOL = Path(__file__).resolve().parents[1] / "research/EXPERIMENT_11_HIGHRES_TILE_KNOWLEDGE_CARDS_PROTOCOL_20260918.md"

TILE_SIZE = 448
TILE_STRIDE = 224
MODEL_INPUT = 224
MIN_RETINAL_COVERAGE = 0.80
INPUT_DIM = 1024
DICTIONARY = 16_384
TOP_K = 32
SEED = 4106


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


def field_crop_and_box(image: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int], np.ndarray]:
    rgb = np.asarray(image.convert("RGB"))
    valid = rgb.max(axis=2) > 8
    if not valid.any():
        raise RuntimeError("image has no retinal pixels above threshold")
    ys, xs = np.where(valid)
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    return image.convert("RGB").crop(box), box, valid[box[1] : box[3], box[0] : box[2]]


def starts(length: int) -> list[int]:
    if length < TILE_SIZE:
        return [0]
    result = list(range(0, length - TILE_SIZE + 1, TILE_STRIDE))
    last = length - TILE_SIZE
    if result[-1] != last:
        result.append(last)
    return result


def tile_plan(image: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int], list[dict]]:
    """Return retained 448px tile boxes in original-image coordinates.

    Coverage is computed from retinal pixels in the original field crop. The
    crop is never independently re-cropped after a tile is selected.
    """
    field, field_box, valid = field_crop_and_box(image)
    retained: list[dict] = []
    for y in starts(field.height):
        for x in starts(field.width):
            y1, x1 = min(y + TILE_SIZE, field.height), min(x + TILE_SIZE, field.width)
            coverage = float(valid[y:y1, x:x1].sum()) / float(TILE_SIZE * TILE_SIZE)
            if coverage < MIN_RETINAL_COVERAGE:
                continue
            retained.append({
                "field_box": (x, y, x + TILE_SIZE, y + TILE_SIZE),
                "source_box": (field_box[0] + x, field_box[1] + y, field_box[0] + x + TILE_SIZE, field_box[1] + y + TILE_SIZE),
                "retinal_coverage": coverage,
            })
    if not retained:
        raise RuntimeError(f"no tile satisfies {MIN_RETINAL_COVERAGE:.2f} retinal coverage")
    return field, field_box, retained


def preprocess_tile(field: Image.Image, tile: dict) -> torch.Tensor:
    # PIL pads only if a source box reaches beyond the original field. This is
    # deterministic and preserves the tile's original coordinate frame.
    crop = field.crop(tile["field_box"]).resize((MODEL_INPUT, MODEL_INPUT), Image.Resampling.BILINEAR)
    array = np.asarray(crop, dtype=np.float32) / 255.0
    return normalize_retfound_cfp(torch.from_numpy(array).permute(2, 0, 1).contiguous())


def selected_records(manifest: Path, cohorts: Path) -> list[dict]:
    cohort = json.loads(cohorts.read_text(encoding="utf-8"))
    groups = cohort.get("patients", {})
    expected = {name: set(map(str, groups.get(name, []))) for name in ("sae_train", "development", "pilot_holdout")}
    if set(expected) != set(groups) or any(len(values) == 0 for values in expected.values()):
        raise RuntimeError("Experiment 8 cohort structure is invalid")
    if len(set().union(*expected.values())) != sum(map(len, expected.values())):
        raise RuntimeError("Experiment 8 patient groups overlap")
    lookup = {patient: group for group, patients in expected.items() for patient in patients}
    records = [row for row in read_jsonl(manifest) if row.get("split") == "train" and str(row.get("patient_id")) in lookup]
    observed = {str(row["patient_id"]) for row in records}
    if observed != set(lookup):
        missing = sorted(set(lookup) - observed)
        raise RuntimeError(f"cohort patients absent from manifest: {missing[:3]}")
    for row in records:
        row["cohort"] = lookup[str(row["patient_id"])]
    return records


def source_hashes() -> dict:
    inputs = {"protocol": PROTOCOL, "runner": Path(__file__).resolve(), "cohorts": COHORTS, "manifest": MANIFEST, "retfound_checkpoint": CHECKPOINT}
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"required Experiment 11 input missing: {missing}")
    return {name: {"path": str(path), "sha256": sha256_path(path)} for name, path in inputs.items()}


def smoke(output: Path, manifest: Path, dataset: Path, cohorts: Path) -> dict:
    # Synthetic geometry: 896px field gives 3 positions per dimension with the
    # required 50% overlap. No clinical labels are read.
    synthetic = Image.fromarray(np.full((896, 896, 3), 255, dtype=np.uint8), mode="RGB")
    _, box, plans = tile_plan(synthetic)
    actual = selected_records(manifest, cohorts)[:5]
    counts, source_boxes = [], []
    for record in actual:
        source = dataset / record["relative_path"]
        if not source.is_file():
            raise RuntimeError(f"missing cohort image: {source}")
        with Image.open(source) as image:
            field, _, tiles = tile_plan(image)
            tensor = preprocess_tile(field, tiles[0])
        if tensor.shape != (3, MODEL_INPUT, MODEL_INPUT) or not torch.isfinite(tensor).all():
            raise RuntimeError("tile preprocessing smoke failed")
        counts.append(len(tiles)); source_boxes.append(tiles[0]["source_box"])
    record = {
        "status": "smoke_passed",
        "labels_loaded": False,
        "synthetic_field_box": box,
        "synthetic_tile_count": len(plans),
        "synthetic_first_boxes": [item["source_box"] for item in plans[:4]],
        "tile_size_px": TILE_SIZE,
        "tile_stride_px": TILE_STRIDE,
        "minimum_retinal_coverage": MIN_RETINAL_COVERAGE,
        "real_images": len(actual),
        "real_tile_counts": counts,
        "real_first_source_boxes": source_boxes,
        "patient_groups_checked": [item["cohort"] for item in actual],
        "source_hashes": source_hashes(),
    }
    if len(plans) != 9 or any(count < 1 for count in counts):
        raise RuntimeError("tile geometry smoke invariant failed")
    write_json(output / "tile_preflight_smoke.json", record)
    return record


def extraction_specification(output: Path, records: list[dict], device: str) -> dict:
    record = {
        "schema_version": "1.0",
        "status": "frozen_before_tile_extraction",
        "experiment": "11_highres_tile_knowledge_cards",
        "labels_loaded": False,
        "records": len(records),
        "patients": len({str(row["patient_id"]) for row in records}),
        "tile": {"source_size_px": TILE_SIZE, "stride_px": TILE_STRIDE, "overlap_fraction": 0.5, "minimum_retinal_coverage": MIN_RETINAL_COVERAGE, "model_input_px": MODEL_INPUT},
        "retfound": {"frozen": True, "representation": "final_mean_patch", "input_dim": INPUT_DIM},
        "sae_plan": {"label_free": True, "dictionary": DICTIONARY, "top_k": TOP_K, "seed": SEED, "epochs": 20},
        "device_requested": device,
        "source_hashes": source_hashes(),
        "image_quality_policy": {
            "on_no_eligible_tile": "exclude_image_and_record_reason",
            "minimum_retinal_coverage_remains": MIN_RETINAL_COVERAGE,
        },
    }
    path = output / "tile_extraction_specification.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        stable_keys = ("status", "experiment", "labels_loaded", "records", "patients", "tile", "retfound", "sae_plan", "device_requested")
        if any(existing.get(key) != record.get(key) for key in stable_keys):
            raise RuntimeError("existing tile extraction specification differs from frozen inputs")
        old_hashes, new_hashes = existing.get("source_hashes", {}), record.get("source_hashes", {})
        stable_hashes = ("protocol", "cohorts", "manifest", "retfound_checkpoint")
        if any(old_hashes.get(key) != new_hashes.get(key) for key in stable_hashes):
            raise RuntimeError("existing tile extraction source inputs differ from frozen inputs")
        if old_hashes.get("runner") != new_hashes.get("runner"):
            write_json(output / "tile_extraction_recovery_amendment.json", {
                "status": "quality-exclusion-recovery",
                "reason": "A source image had no tile at the prespecified 0.80 coverage threshold; the runner now records exclusion instead of aborting.",
                "threshold_changed": False,
                "original_specification_sha256": sha256_path(path),
                "original_runner": old_hashes.get("runner"),
                "recovery_runner": new_hashes.get("runner"),
                "labels_loaded": False,
            })
        return existing
    write_json(path, record)
    return record


def extract(args, output: Path) -> dict:
    records = selected_records(Path(args.manifest), Path(args.cohorts))
    if args.max_images is not None:
        if args.max_images < 1:
            raise ValueError("max-images must be positive")
        records = records[: args.max_images]
    spec = extraction_specification(output, records, args.device)
    device = choose_device(args.device)
    model = load_retfound_cfp(Path(args.checkpoint), Path(args.upstream), representation="final_mean_patch").to(device).eval()
    shard_root = output / "tile_embeddings"; shard_root.mkdir(parents=True, exist_ok=True)
    evidence, computed, reused, total_tiles = [], 0, 0, 0
    exclusions: list[dict] = []
    all_patient_ids: list[str] = []
    for start in range(0, len(records), args.shard_images):
        chunk = records[start : start + args.shard_images]
        end = start + len(chunk) - 1
        target = shard_root / f"tiles_{start:06d}_{end:06d}.pt"
        expected_paths = [row["relative_path"] for row in chunk]
        if target.is_file():
            payload = torch.load(target, map_location="cpu", weights_only=False)
            requested = payload.get("requested_relative_paths", payload.get("relative_paths"))
            if requested != expected_paths or payload.get("checkpoint_hash") != model.checkpoint_hash:
                raise RuntimeError(f"existing tile shard is incompatible: {target}")
            tile_count = int(payload["tile_embeddings"].shape[0]); reused += len(payload["relative_paths"])
            exclusions.extend(payload.get("exclusions", []))
        else:
            all_embeddings, offsets, boxes, coverage, fields = [], [0], [], [], []
            valid_paths, valid_patients, valid_cohorts, shard_exclusions = [], [], [], []
            with torch.inference_mode():
                for record in chunk:
                    source = Path(args.dataset) / record["relative_path"]
                    if not source.is_file():
                        raise RuntimeError(f"missing cohort image: {source}")
                    try:
                        with Image.open(source) as image:
                            field, field_box, plans = tile_plan(image)
                            tensors = torch.stack([preprocess_tile(field, tile) for tile in plans])
                    except RuntimeError as error:
                        if "no tile satisfies" not in str(error) and "no retinal pixels" not in str(error):
                            raise
                        shard_exclusions.append({
                            "relative_path": record["relative_path"],
                            "patient_id": str(record["patient_id"]),
                            "cohort": record["cohort"],
                            "reason": str(error),
                        })
                        continue
                    rows = []
                    for batch_start in range(0, len(tensors), args.batch_size):
                        batch = tensors[batch_start : batch_start + args.batch_size].to(device)
                        rows.append(model(batch).float().cpu())
                    embedding = torch.cat(rows)
                    if embedding.ndim != 2 or embedding.shape[1] != INPUT_DIM or not torch.isfinite(embedding).all():
                        raise RuntimeError("frozen RETFound tile extraction produced invalid embeddings")
                    all_embeddings.append(embedding); offsets.append(offsets[-1] + len(embedding))
                    boxes.extend(tile["source_box"] for tile in plans); coverage.extend(tile["retinal_coverage"] for tile in plans); fields.append(field_box)
                    valid_paths.append(record["relative_path"]); valid_patients.append(str(record["patient_id"])); valid_cohorts.append(record["cohort"])
            tile_embeddings = torch.cat(all_embeddings) if all_embeddings else torch.empty((0, INPUT_DIM), dtype=torch.float32)
            payload = {
                "schema_version": "1.0", "checkpoint_hash": model.checkpoint_hash, "representation": model.representation,
                "tile_size_px": TILE_SIZE, "tile_stride_px": TILE_STRIDE, "minimum_retinal_coverage": MIN_RETINAL_COVERAGE,
                "requested_relative_paths": expected_paths, "relative_paths": valid_paths, "patient_ids": valid_patients, "cohorts": valid_cohorts,
                "exclusions": shard_exclusions,
                "tile_offsets": torch.tensor(offsets, dtype=torch.int64), "tile_embeddings": tile_embeddings.half(),
                "source_boxes": torch.tensor(boxes, dtype=torch.int32), "retinal_coverage": torch.tensor(coverage, dtype=torch.float32), "field_boxes": torch.tensor(fields, dtype=torch.int32),
            }
            partial = target.with_suffix(".pt.partial"); torch.save(payload, partial); partial.replace(target)
            tile_count = int(payload["tile_embeddings"].shape[0]); computed += len(valid_paths); exclusions.extend(shard_exclusions)
        all_patient_ids.extend(payload["patient_ids"])
        total_tiles += tile_count
        evidence.append({"path": str(target), "requested_images": len(chunk), "valid_images": len(payload["relative_paths"]), "excluded_images": len(payload.get("exclusions", [])), "tiles": tile_count, "sha256": sha256_path(target), "bytes": target.stat().st_size})
        print(json.dumps({"event": "tile_shard_complete", "processed_images": start + len(chunk), "total_images": len(records), "tiles": total_tiles, "excluded_images": len(exclusions), "action": "reused" if target.is_file() and computed == 0 else "computed"}, sort_keys=True), flush=True)
    exclusion_path = output / "tile_image_exclusions.json"
    write_json(exclusion_path, {"status": "complete", "labels_loaded": False, "minimum_retinal_coverage": MIN_RETINAL_COVERAGE, "excluded_images": exclusions})
    valid_paths = []
    for entry in evidence:
        payload = torch.load(entry["path"], map_location="cpu", weights_only=False)
        valid_paths.extend(payload["relative_paths"])
    result = {"status": "complete", "labels_loaded": False, "requested_images": len(records), "images": len(valid_paths), "requested_patients": len({str(row["patient_id"]) for row in records}), "patients": len(set(all_patient_ids)), "tiles": total_tiles,
              "excluded_images": len(exclusions), "exclusions_sha256": sha256_path(exclusion_path), "computed_images": computed, "reused_images": reused, "device": str(device), "checkpoint_hash": model.checkpoint_hash, "specification_sha256": sha256_path(output / "tile_extraction_specification.json"), "shards": evidence}
    write_json(output / "tile_extraction_evidence.json", result)
    return result


def load_tiles(root: Path) -> dict:
    evidence = json.loads((root / "tile_extraction_evidence.json").read_text(encoding="utf-8"))
    if evidence.get("status") != "complete" or evidence.get("labels_loaded"):
        raise RuntimeError("tile extraction evidence is incomplete or label-contaminated")
    tensors, paths, patients, cohorts, tile_cohorts = [], [], [], [], []
    for entry in evidence["shards"]:
        path = Path(entry["path"])
        if not path.is_file() or sha256_path(path) != entry["sha256"]:
            raise RuntimeError(f"tile shard missing or hash-mismatched: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        embeddings = payload["tile_embeddings"].float()
        offsets = payload["tile_offsets"].to(torch.int64)
        image_cohorts = payload["cohorts"]
        if len(offsets) != len(image_cohorts) + 1 or int(offsets[-1]) != len(embeddings):
            raise RuntimeError(f"tile/image offset alignment is invalid: {path}")
        counts = offsets[1:] - offsets[:-1]
        tensors.append(embeddings); paths.extend(payload["relative_paths"]); patients.extend(payload["patient_ids"]); cohorts.extend(image_cohorts)
        tile_cohorts.extend(group for group, count in zip(image_cohorts, counts.tolist()) for _ in range(int(count)))
    return {"tiles": torch.cat(tensors), "paths": paths, "patients": np.asarray(patients), "cohorts": np.asarray(cohorts), "tile_cohorts": np.asarray(tile_cohorts), "evidence": evidence}


def train_sae(args, output: Path) -> dict:
    loaded = load_tiles(output)
    if len(loaded["tile_cohorts"]) != len(loaded["tiles"]):
        raise RuntimeError("tile cohort expansion is not aligned with tile embeddings")
    train = loaded["tiles"][loaded["tile_cohorts"] == "sae_train"]
    development = loaded["tiles"][loaded["tile_cohorts"] == "development"]
    if not len(train) or not len(development):
        raise RuntimeError("Tile-SAE requires non-empty train and development tile banks")
    mean, std = train.mean(0), train.std(0, unbiased=False).clamp_min(1e-6)
    normalized_train, normalized_dev = (train - mean) / std, (development - mean) / std
    device = choose_device(args.device); set_deterministic(SEED)
    model = TopKSAE(INPUT_DIM, DICTIONARY, TOP_K).to(device)
    history, optimizer = train_sae_batched(model, normalized_train, normalized_dev, epochs=20, batch_size=1024, evaluation_batch_size=1024, learning_rate=3e-4, weight_decay=1e-5, seed=SEED, device=device, resample_dead=True, resample_warmup_epochs=5, resample_interval_epochs=5)
    metrics = evaluate_embeddings_batched(model, normalized_dev, batch_size=1024, device=device)
    gates = {"explained_variance_min": 0.85, "dead_feature_fraction_max": 0.25, "max_active_features_max": TOP_K}
    passed = metrics["explained_variance"] >= gates["explained_variance_min"] and metrics["dead_feature_fraction"] <= gates["dead_feature_fraction_max"] and metrics["max_active_features"] <= TOP_K
    result = {"status": "complete" if passed else "failed_health_gate", "labels_loaded": False, "metrics": metrics, "gates": gates, "history": history, "tile_extraction_sha256": sha256_path(output / "tile_extraction_evidence.json")}
    write_json(output / "tile_sae_result.json", result)
    if not passed:
        raise RuntimeError(f"high-resolution Tile-SAE health gate failed: {metrics}")
    torch.save({"mean": mean, "std": std, "labels_loaded": False, "source": "sae_train_tiles"}, output / "tile_sae_normalization.pt")
    save_checkpoint(output / "tile_sae_checkpoint.pt", model.cpu(), optimizer, {"seed": SEED, "history": history, "metrics": metrics, "labels_loaded": False, "tile_extraction_sha256": sha256_path(output / "tile_extraction_evidence.json")})
    reloaded, _ = load_checkpoint(output / "tile_sae_checkpoint.pt")
    if any(not torch.equal(value, reloaded.state_dict()[key]) for key, value in model.cpu().state_dict().items()):
        raise RuntimeError("Tile-SAE checkpoint reload mismatch")
    return result


def extraction_smoke(args, output: Path) -> dict:
    """Exercise frozen RETFound on one real image without freezing the full run."""
    smoke_root = output / "encoder_extraction_smoke"
    original_limit = args.max_images
    args.max_images = 1
    try:
        result = extract(args, smoke_root)
    finally:
        args.max_images = original_limit
    if result["images"] != 1 or result["tiles"] < 1 or result["labels_loaded"]:
        raise RuntimeError("tile encoder smoke invariant failed")
    result["status"] = "smoke_passed"
    write_json(output / "tile_encoder_smoke.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment 11 high-resolution Tile-SAE preflight")
    parser.add_argument("--stage", choices=("smoke", "extract_smoke", "extract", "sae"), required=True)
    parser.add_argument("--output", default=str(OUTPUT)); parser.add_argument("--manifest", default=str(MANIFEST)); parser.add_argument("--dataset", default=str(DATASET)); parser.add_argument("--cohorts", default=str(COHORTS))
    parser.add_argument("--checkpoint", default=str(CHECKPOINT)); parser.add_argument("--upstream", default=str(UPSTREAM)); parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16); parser.add_argument("--shard-images", type=int, default=32); parser.add_argument("--max-images", type=int)
    args = parser.parse_args(); output = ensure_output(args.output)
    for name in ("manifest", "dataset", "cohorts", "checkpoint", "upstream"):
        setattr(args, name, str(require_external(Path(getattr(args, name)), name)))
    if args.stage == "smoke":
        print(json.dumps(smoke(output, Path(args.manifest), Path(args.dataset), Path(args.cohorts)), sort_keys=True)); return 0
    if args.stage == "extract_smoke":
        print(json.dumps(extraction_smoke(args, output), sort_keys=True)); return 0
    if args.stage == "extract":
        print(json.dumps(extract(args, output), sort_keys=True)); return 0
    print(json.dumps(train_sae(args, output), sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
