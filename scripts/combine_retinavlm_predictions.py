"""Combine complete, hash-bound A/B/C/D split predictions by matched seed."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from retinavlm_released_model import SPECIALIST_REVISION
from retinavlm_verified_images import load_verified_manifest, select_split, sha256


FIELDS = ["source_image_path", "image_sha256", "arm", "seed", "answer"]


def run(*, manifest: Path, image_audit: Path, split: str,
        run_dirs: list[Path], output_csv: Path, bundle_receipt: Path,
        selection_lock: Path | None = None) -> dict:
    if split not in {"validation", "test"} or not run_dirs:
        raise ValueError("Use validation/test and at least one prediction run")
    if output_csv.exists() or bundle_receipt.exists():
        raise FileExistsError("Combined predictions or bundle receipt already exists")
    rows = load_verified_manifest(manifest, image_audit)
    manifest_digest = sha256(manifest)
    image_audit_digest = sha256(image_audit)
    selected = select_split(rows, split, manifest_digest=manifest_digest,
                            selection_lock=selection_lock)
    expected = {row["source_image_path"]: row["sha256"] for row in selected}
    lock = None
    if split == "test":
        lock = json.loads(selection_lock.read_text(encoding="utf-8"))
        if (lock.get("status") != "validation_selection_frozen"
                or lock.get("manifest_sha256") != manifest_digest
                or lock.get("image_audit_sha256") != image_audit_digest
                or lock.get("selected_configuration", {}).get("released_revision") != SPECIALIST_REVISION):
            raise ValueError("Test lock does not match manifest and released model")
    seen_keys = set()
    combined = []
    source_receipts = []
    for directory in run_dirs:
        prediction_csv = directory / "predictions.csv"
        receipt_path = directory / "prediction_receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        arm, seed = receipt.get("arm"), receipt.get("seed")
        if arm not in {"A", "B", "C", "D"} or not isinstance(seed, int) or seed < 0:
            raise ValueError(f"Invalid prediction arm/seed: {directory}")
        key = (arm, seed)
        if key in seen_keys:
            raise ValueError(f"Duplicate prediction run: {key}")
        seen_keys.add(key)
        if (receipt.get("status") != "complete_verified_split_predictions"
                or receipt.get("split") != split
                or receipt.get("scans") != len(expected)
                or receipt.get("manifest_sha256") != manifest_digest
                or receipt.get("image_audit_sha256") != image_audit_digest
                or receipt.get("released_revision") != SPECIALIST_REVISION
                or receipt.get("predictions_sha256") != sha256(prediction_csv)):
            raise ValueError(f"Prediction receipt is incomplete or changed: {key}")
        if split == "test":
            if receipt.get("selection_lock_sha256") != sha256(selection_lock):
                raise ValueError(f"Prediction run used a different test lock: {key}")
            expected_checkpoint = (None if arm == "A" else lock["selected_configuration"]["checkpoints"].get(f"{arm}/{seed}"))
            if receipt.get("checkpoint_sha256") != expected_checkpoint:
                raise ValueError(f"Prediction run used an unselected checkpoint: {key}")
        elif arm == "A" and receipt.get("checkpoint_sha256") is not None:
            raise ValueError("Released A arm must have no trained checkpoint")
        with prediction_csv.open(newline="", encoding="utf-8") as file:
            records = list(csv.DictReader(file))
        if len(records) != len(expected) or not records or not set(FIELDS).issubset(records[0]):
            raise ValueError(f"Prediction run does not cover the full split: {key}")
        paths = set()
        for row in records:
            path = row["source_image_path"]
            if (path in paths or path not in expected or row["image_sha256"] != expected[path]
                    or row["arm"] != arm or row["seed"] != str(seed)
                    or row["answer"] not in {"present", "absent"}):
                raise ValueError(f"Invalid or repeated prediction row: {key} {path}")
            paths.add(path)
            combined.append({field: row[field] for field in FIELDS})
        if paths != set(expected):
            raise ValueError(f"Incomplete split for {key}")
        source_receipts.append({"arm": arm, "seed": seed,
                                "receipt_path": str(receipt_path.resolve()),
                                "receipt_sha256": sha256(receipt_path),
                                "predictions_path": str(prediction_csv.resolve()),
                                "predictions_sha256": sha256(prediction_csv),
                                "checkpoint_sha256": receipt.get("checkpoint_sha256")})
    seeds = {seed for arm, seed in seen_keys if arm == "D"}
    required_keys = {(arm, seed) for seed in seeds for arm in "ABCD"}
    if not seeds or seen_keys != required_keys:
        raise ValueError("Each selected seed needs one complete A/B/C/D prediction run")
    combined.sort(key=lambda row: (row["seed"], row["arm"], row["source_image_path"]))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_csv.with_suffix(output_csv.suffix + ".writing")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(combined)
    os.replace(temporary, output_csv)
    bundle = {"status": "complete_matched_prediction_bundle",
              "split": split, "manifest_sha256": manifest_digest,
              "image_audit_sha256": image_audit_digest,
              "selection_lock_sha256": sha256(selection_lock) if selection_lock else None,
              "released_revision": SPECIALIST_REVISION,
              "patients": len({row["Patient_ID"] for row in selected}),
              "scans_per_arm_seed": len(expected),
              "matched_seeds": sorted(seeds),
              "combined_predictions_path": str(output_csv.resolve()),
              "combined_predictions_sha256": sha256(output_csv),
              "source_runs": sorted(source_receipts, key=lambda row: (row["seed"], row["arm"])),
              "test_split_accessed": split == "test"}
    bundle_receipt.parent.mkdir(parents=True, exist_ok=True)
    bundle_receipt.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    return bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-image-manifest", type=Path, required=True)
    parser.add_argument("--image-audit", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "test"], required=True)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--selection-lock", type=Path)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--bundle-receipt", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(
        manifest=args.official_image_manifest, image_audit=args.image_audit,
        split=args.split, run_dirs=args.run_dir, output_csv=args.output_csv,
        bundle_receipt=args.bundle_receipt, selection_lock=args.selection_lock,
    ), indent=2))
