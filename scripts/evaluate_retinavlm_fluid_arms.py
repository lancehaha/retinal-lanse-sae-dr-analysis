"""Evaluate matched RetinaVLM fluid-answer arms on a patient-disjoint split.

Predictions CSV columns: source_image_path, image_sha256, arm, seed, answer. Each arm/seed
must cover every scan in the requested split exactly once. Answers must be
exactly `present` or `absent`; missing, duplicate, or invalid responses fail.
The verified official-image manifest supplies labels and patient IDs, never
predictions. Its image-audit receipt is mandatory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


EXPECTED_ARCHIVE_MD5 = "e08f4198aa919d6900bd30525c137933"
EXPECTED_EXPERT_IMAGES = 9408


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def scores(rows: list[tuple[int, int]]) -> dict[str, float | int | None]:
    tp = sum(y == 1 and p == 1 for y, p in rows)
    tn = sum(y == 0 and p == 0 for y, p in rows)
    fp = sum(y == 0 and p == 1 for y, p in rows)
    fn = sum(y == 1 and p == 0 for y, p in rows)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    fpr = fp / (fp + tn) if fp + tn else None
    return {"n": len(rows), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
            "false_positive_rate": fpr}


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def paired_bootstrap(
    gold: dict[str, dict[str, str]],
    predictions: dict[tuple[str, str], dict[str, int]],
    arm_a: tuple[str, str],
    arm_b: tuple[str, str],
    replicates: int,
    seed: int,
) -> dict:
    patient_paths = defaultdict(list)
    trial_patients = defaultdict(set)
    for path, row in gold.items():
        patient = row["Patient_ID"]
        patient_paths[patient].append(path)
        trial_patients[row["trial"]].add(patient)
    patients = sorted(patient_paths)
    rng = random.Random(seed)
    deltas = {"false_positive_rate": [], "recall": [], "f1": []}
    for _ in range(replicates):
        sample = []
        for trial in sorted(trial_patients):
            group = sorted(trial_patients[trial])
            sample.extend(rng.choice(group) for _ in group)
        paths = [path for patient in sample for path in patient_paths[patient]]
        metrics_a = scores([(int(gold[path]["fluid"]), predictions[arm_a][path]) for path in paths])
        metrics_b = scores([(int(gold[path]["fluid"]), predictions[arm_b][path]) for path in paths])
        for metric in deltas:
            a, b = metrics_a[metric], metrics_b[metric]
            if a is not None and b is not None:
                deltas[metric].append(a - b)
    result = {}
    for metric, values in deltas.items():
        result[metric] = ({"ci95": [percentile(values, 0.025), percentile(values, 0.975)],
                           "valid_replicates": len(values)} if values else None)
    return {"arm_a": list(arm_a), "arm_b": list(arm_b),
            "patients": len(patients), "replicates": replicates,
            "trial_patient_counts": {trial: len(group) for trial, group in sorted(trial_patients.items())},
            "difference_a_minus_b": result}


def evaluate(manifest: Path, image_audit: Path, prediction_csv: Path, split: str,
             replicates: int, bootstrap_seed: int,
             selection_lock: Path | None = None,
             prediction_bundle: Path | None = None) -> dict:
    if replicates < 1:
        raise ValueError("At least one bootstrap replicate is required")
    audit = json.loads(image_audit.read_text(encoding="utf-8"))
    manifest_digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    image_audit_digest = file_sha256(image_audit)
    if (audit.get("status") != "official_images_matched_and_exact_duplicate_audited"
            or audit.get("source_archive_md5") != EXPECTED_ARCHIVE_MD5
            or audit.get("matched_images") != EXPECTED_EXPERT_IMAGES
            or audit.get("manifest_sha256") != manifest_digest
            or audit.get("near_duplicate_gate") != "reviewed_pass"
            or audit.get("fovea_crop_and_scanner_gate") != "reviewed_input_compatible_with_domain_shift"
            or audit.get("images_with_values_above_255", 0) != 0
            or audit.get("cross_split_decoded_pixel_duplicate_hashes") != 0
            or audit.get("conflicting_label_decoded_pixel_duplicate_hashes") != 0):
        raise ValueError("Official image audit does not certify this complete manifest")
    max_recall_drop = None
    bundle = None
    if split == "test":
        if selection_lock is None:
            raise ValueError("Test evaluation requires a frozen validation-selection lock")
        lock = json.loads(selection_lock.read_text(encoding="utf-8"))
        choice = lock.get("selected_configuration")
        if (lock.get("status") != "validation_selection_frozen"
                or lock.get("manifest_sha256") != manifest_digest
                or lock.get("image_audit_sha256") != image_audit_digest
                or not isinstance(choice, dict)):
            raise ValueError("Validation-selection lock does not match the official manifest")
        max_recall_drop = choice.get("max_recall_drop")
        if (not isinstance(max_recall_drop, (int, float))
                or isinstance(max_recall_drop, bool)
                or not 0 <= max_recall_drop <= 1):
            raise ValueError("Frozen maximum recall drop must be between 0 and 1")
        if prediction_bundle is None:
            raise ValueError("Test evaluation requires a matched prediction bundle")
        bundle = json.loads(prediction_bundle.read_text(encoding="utf-8"))
        if (bundle.get("status") != "complete_matched_prediction_bundle"
                or bundle.get("split") != "test"
                or bundle.get("manifest_sha256") != manifest_digest
                or bundle.get("image_audit_sha256") != image_audit_digest
                or bundle.get("selection_lock_sha256") != file_sha256(selection_lock)
                or bundle.get("combined_predictions_sha256") != file_sha256(prediction_csv)):
            raise ValueError("Matched test prediction bundle does not bind these inputs")
        source_runs = bundle.get("source_runs")
        if not isinstance(source_runs, list) or not source_runs:
            raise ValueError("Matched test prediction bundle has no source runs")
        for source in source_runs:
            receipt_path = Path(source["receipt_path"])
            source_prediction_path = Path(source["predictions_path"])
            if (file_sha256(receipt_path) != source["receipt_sha256"]
                    or file_sha256(source_prediction_path) != source["predictions_sha256"]):
                raise ValueError("A source prediction run has changed since bundling")
    all_gold = read_csv(manifest)
    required_gold = {"source_image_path", "Patient_ID", "trial", "fluid", "split", "sha256", "local_image_path"}
    if not all_gold or not required_gold.issubset(all_gold[0]):
        raise ValueError("Official manifest is missing required columns")
    if len(all_gold) != EXPECTED_EXPERT_IMAGES:
        raise ValueError("Official expert-image manifest row count differs from audit")
    patient_splits = defaultdict(set)
    patient_trials = defaultdict(set)
    all_paths = set()
    for row in all_gold:
        path = row["source_image_path"]
        if path in all_paths:
            raise ValueError(f"Repeated source path in official manifest: {path}")
        all_paths.add(path)
        patient_splits[row["Patient_ID"]].add(row["split"])
        patient_trials[row["Patient_ID"]].add(row["trial"])
    crossing = [patient for patient, splits in patient_splits.items() if len(splits) != 1]
    if crossing:
        raise ValueError(f"Patients cross splits: {len(crossing)}")
    crossing_trials = [patient for patient, trials in patient_trials.items() if len(trials) != 1]
    if crossing_trials:
        raise ValueError(f"Patients cross trials: {len(crossing_trials)}")
    gold = {}
    for row in all_gold:
        if row["split"] != split:
            continue
        path = row["source_image_path"]
        if path in gold or row["fluid"] not in {"0", "1"} or not row["Patient_ID"]:
            raise ValueError(f"Invalid or repeated official label: {path}")
        gold[path] = row
    if not gold:
        raise ValueError(f"No official labels for split {split}")
    predictions = defaultdict(dict)
    input_rows = read_csv(prediction_csv)
    required_prediction = {"source_image_path", "image_sha256", "arm", "seed", "answer"}
    if not input_rows or not required_prediction.issubset(input_rows[0]):
        raise ValueError("Predictions CSV is missing required columns")
    for row in input_rows:
        path = row["source_image_path"]
        key = (row["arm"], row["seed"])
        answer = row["answer"].strip().lower()
        if path not in gold or answer not in {"present", "absent"} or not all(key):
            raise ValueError(f"Out-of-split, invalid, or incomplete prediction: {key} {path}")
        if row["image_sha256"] != gold[path]["sha256"]:
            raise ValueError(f"Prediction image hash does not match official image: {key} {path}")
        if path in predictions[key]:
            raise ValueError(f"Duplicate prediction: {key} {path}")
        predictions[key][path] = int(answer == "present")
    expected = set(gold)
    for key, values in predictions.items():
        if set(values) != expected:
            raise ValueError(f"Incomplete predictions for {key}: missing {len(expected - set(values))}")
    if bundle is not None:
        source_keys = {(row["arm"], str(row["seed"])) for row in bundle["source_runs"]}
        if source_keys != set(predictions) or len(source_keys) != len(bundle["source_runs"]):
            raise ValueError("Bundle source runs do not match evaluated arm/seed predictions")
        for source in bundle["source_runs"]:
            arm, seed_text = source["arm"], str(source["seed"])
            expected_checkpoint = (None if arm == "A" else lock["selected_configuration"]["checkpoints"].get(f"{arm}/{seed_text}"))
            if source.get("checkpoint_sha256") != expected_checkpoint:
                raise ValueError("Bundle source checkpoint differs from validation selection")
    metrics = {}
    metrics_by_trial = {}
    if not any(key[0] == "D" for key in predictions):
        raise ValueError("No J-Lens arm D predictions supplied")
    if not any(key[0] == "A" for key in predictions):
        raise ValueError("No released-checkpoint arm A predictions supplied")
    for key in sorted(predictions):
        metrics["/".join(key)] = scores([
            (int(gold[path]["fluid"]), predictions[key][path]) for path in sorted(gold)
        ])
        metrics_by_trial["/".join(key)] = {
            trial: scores([
                (int(gold[path]["fluid"]), predictions[key][path])
                for path in sorted(gold) if gold[path]["trial"] == trial
            ])
            for trial in sorted({row["trial"] for row in gold.values()})
        }
    comparisons = []
    for key in sorted(predictions):
        if key[0] != "D":
            continue
        for baseline in ("B", "C"):
            control = (baseline, key[1])
            if control not in predictions:
                raise ValueError(f"Missing matched control {control} for {key}")
            comparison = paired_bootstrap(
                gold, predictions, key, control, replicates, bootstrap_seed
            )
            if max_recall_drop is not None:
                d_metrics = metrics["/".join(key)]
                control_metrics = metrics["/".join(control)]
                comparison["frozen_max_recall_drop"] = max_recall_drop
                comparison["recall_preserved_point_estimate"] = (
                    d_metrics["recall"] is not None
                    and control_metrics["recall"] is not None
                    and d_metrics["recall"] >= control_metrics["recall"] - max_recall_drop
                )
                comparison["false_positive_rate_lower_point_estimate"] = (
                    d_metrics["false_positive_rate"] is not None
                    and control_metrics["false_positive_rate"] is not None
                    and d_metrics["false_positive_rate"] < control_metrics["false_positive_rate"]
                )
            comparisons.append(comparison)
    return {"split": split, "source_manifest": str(manifest),
            "manifest_sha256": manifest_digest,
            "image_audit_sha256": image_audit_digest,
            "source_archive_md5": EXPECTED_ARCHIVE_MD5,
            "selection_lock_sha256": (hashlib.sha256(selection_lock.read_bytes()).hexdigest()
                                      if split == "test" else None),
            "prediction_bundle_sha256": file_sha256(prediction_bundle) if bundle else None,
            "predictions_file": str(prediction_csv),
            "predictions_sha256": hashlib.sha256(prediction_csv.read_bytes()).hexdigest(),
            "scans": len(gold),
            "patients": len({row["Patient_ID"] for row in gold.values()}),
            "metrics": metrics, "metrics_by_trial": metrics_by_trial,
            "paired_patient_bootstrap": comparisons}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("official_manifest", type=Path)
    parser.add_argument("--image-audit", type=Path, required=True)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--split", choices=["train", "validation", "test"], required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=4106)
    parser.add_argument("--selection-lock", type=Path)
    parser.add_argument("--prediction-bundle", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = evaluate(args.official_manifest, args.image_audit, args.predictions, args.split,
                      args.bootstrap_replicates, args.bootstrap_seed,
                      args.selection_lock, args.prediction_bundle)
    payload = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
