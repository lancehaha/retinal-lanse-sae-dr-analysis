"""Compare released and OLIVES-BYOL encoders with matched IRF linear probes.

Fit probes only on frozen training patients; evaluate on all frozen validation
patients. This measures linear IRF readability, not RetinaVLM report accuracy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


EXPECTED_ADAPTED_SHA256 = "c2e433c7d64d747b2b3f3a14f28cbd9c2d153cfb1b5270093b07b8b32c9ac50c"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_package(root: Path, split: str) -> tuple[list[dict[str, str]], dict]:
    receipt = json.loads((root / "receipt.json").read_text(encoding="utf-8"))
    status, manifest, manifest_key, expected_rows, expected_patients = {
        "train": ("frozen_olives_train_only_byol_package", "train_manifest.csv", "train_manifest_sha256", 6664, 61),
        "validation": ("frozen_olives_validation_only_irf_package", "validation_manifest.csv", "validation_manifest_sha256", 1470, 13),
    }[split]
    manifest_path = root / manifest
    if receipt["status"] != status or sha256(manifest_path) != receipt[manifest_key]:
        raise ValueError(f"{split} receipt or manifest mismatch")
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_rows or len({row["patient_key"] for row in rows}) != expected_patients:
        raise ValueError(f"Unexpected {split} cohort size")
    if len({row["source_image_path"] for row in rows}) != len(rows) or any(row["IRF"] not in {"0", "1"} for row in rows):
        raise ValueError(f"Unexpected {split} image path or IRF label")
    return rows, receipt


class ScanDataset:
    def __init__(self, root: Path, rows: list[dict[str, str]]):
        self.root = root / "images"
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        from PIL import Image
        from torchvision.transforms.functional import to_tensor

        row = self.rows[index]
        relative = row["source_image_path"].lstrip("/")
        if ".." in Path(relative).parts:
            raise ValueError("Unsafe image path")
        path = self.root / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Image missing or symlinked: {path}")
        with Image.open(path) as image:
            if image.mode not in {"L", "RGB"}:
                raise ValueError(f"Unexpected image mode: {path}")
            image = image.convert("L").resize((192, 192), Image.Resampling.BILINEAR)
            tensor = to_tensor(image)
        return tensor, int(row["IRF"]), row["patient_key"], row["source_image_path"]


def extract_features(models, root, rows, batch_size, workers):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(ScanDataset(root, rows), batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    stores = {name: [] for name in models}
    labels, patients, paths = [], [], []
    with torch.inference_mode():
        for batch_index, (images, y, patient, path) in enumerate(loader, start=1):
            images = images.cuda(non_blocking=True)
            for name, model in models.items():
                stores[name].append(model(images).float().cpu().numpy())
            labels.extend(y.tolist())
            patients.extend(patient)
            paths.extend(path)
            if batch_index % 25 == 0 or batch_index == len(loader):
                print(f"Embedded {min(batch_index * batch_size, len(rows))}/{len(rows)} scans", flush=True)
    return {name: np.concatenate(parts) for name, parts in stores.items()}, np.asarray(labels), patients, paths


def evaluate_probe(x_train, y_train, train_patients, x_val, y_val, val_patients):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    counts = Counter(train_patients)
    weights = np.asarray([len(y_train) / (len(counts) * counts[p]) for p in train_patients])
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(x_train)
    scaled_val = scaler.transform(x_val)
    classifier = LogisticRegression(C=1.0, solver="liblinear", max_iter=1000, random_state=42)
    classifier.fit(scaled_train, y_train, sample_weight=weights)
    scores = classifier.predict_proba(scaled_val)[:, 1]
    groups = defaultdict(list)
    for index, patient in enumerate(val_patients):
        groups[patient].append(index)
    patient_auc = {}
    for patient, indices in groups.items():
        if len(set(y_val[indices])) != 2:
            raise ValueError(f"Patient lacks both IRF classes: {patient}")
        patient_auc[patient] = float(roc_auc_score(y_val[indices], scores[indices]))
    return scores, {
        "patient_macro_auroc": float(np.mean(list(patient_auc.values()))),
        "pooled_scan_auroc": float(roc_auc_score(y_val, scores)),
        "pooled_scan_average_precision": float(average_precision_score(y_val, scores)),
        "patient_auroc": patient_auc,
        "train_feature_std_mean": float(np.mean(np.std(x_train, axis=0))),
        "train_feature_nonconstant_dims": int(np.count_nonzero(np.std(x_train, axis=0) > 1e-6)),
        "probe_iterations": int(classifier.n_iter_[0]),
    }


def run(args):
    import numpy as np
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    train_rows, train_receipt = read_package(args.train_dir, "train")
    val_rows, val_receipt = read_package(args.validation_dir, "validation")
    if train_receipt["split_manifest_sha256"] != val_receipt["split_manifest_sha256"]:
        raise ValueError("Train and validation packages use different frozen splits")
    if {r["patient_key"] for r in train_rows} & {r["patient_key"] for r in val_rows}:
        raise ValueError("Train/validation patient overlap")
    if {r["source_image_path"] for r in train_rows} & {r["source_image_path"] for r in val_rows}:
        raise ValueError("Train/validation image overlap")
    if sha256(args.adapted_checkpoint) != EXPECTED_ADAPTED_SHA256:
        raise ValueError("Adapted encoder checkpoint hash mismatch")

    sys.path.insert(0, str(args.authors_code))
    from models.ssl_config import ssl_backbone
    from safetensors import safe_open

    index_path = args.released_checkpoint / "model.safetensors.index.json"
    if sha256(index_path) != "65dfbceb982e991eaa66fa381aaeaf042e393dcfa1f16bad14da41ff2c1fe7a2":
        raise ValueError("Released Specialist checkpoint index differs")
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    prefix = "model.visual_encoder.model."
    baseline = ssl_backbone(channels_in=1)
    keys = set(baseline.state_dict())
    selected = {k[len(prefix):]: v for k, v in weight_map.items() if k.startswith(prefix)}
    if set(selected) != keys:
        raise ValueError("Released encoder state keys differ")
    state = {}
    for shard in sorted(set(selected.values())):
        with safe_open(args.released_checkpoint / shard, framework="pt", device="cpu") as handle:
            for key, source_shard in selected.items():
                if shard == source_shard:
                    state[key] = handle.get_tensor(prefix + key)
    baseline.load_state_dict(state, strict=True)
    adapted_file = torch.load(args.adapted_checkpoint, map_location="cpu")
    if adapted_file["format"] != "retinavlm_olives_byol_encoder_full_v1" or adapted_file["step"] != 2083:
        raise ValueError("Unexpected adapted encoder checkpoint")
    if adapted_file["metrics"]["split_manifest_sha256"] != train_receipt["split_manifest_sha256"]:
        raise ValueError("Adapted encoder trained on a different split")
    adapted = ssl_backbone(channels_in=1)
    adapted.load_state_dict(adapted_file["encoder_state_dict"], strict=True)
    models = {"released": baseline.cuda().eval(), "adapted_byol": adapted.cuda().eval()}
    print(json.dumps({"status": "encoder_irf_probe_start", "train": len(train_rows),
                      "validation": len(val_rows), "validation_patients": 13,
                      "gpu": torch.cuda.get_device_name(0)}), flush=True)
    train_features, y_train, train_patients, _ = extract_features(models, args.train_dir, train_rows,
                                                                 args.batch_size, args.workers)
    val_features, y_val, val_patients, val_paths = extract_features(models, args.validation_dir, val_rows,
                                                                    args.batch_size, args.workers)
    scores, results = {}, {}
    for name in models:
        scores[name], results[name] = evaluate_probe(train_features[name], y_train, train_patients,
                                                       val_features[name], y_val, val_patients)
        print(json.dumps({"encoder": name, "patient_macro_auroc": results[name]["patient_macro_auroc"],
                          "pooled_scan_auroc": results[name]["pooled_scan_auroc"]}), flush=True)
    patients = sorted(results["released"]["patient_auroc"])
    deltas = np.asarray([results["adapted_byol"]["patient_auroc"][p] -
                         results["released"]["patient_auroc"][p] for p in patients])
    rng = np.random.default_rng(42)
    bootstrap = deltas[rng.integers(0, len(deltas), size=(10000, len(deltas)))].mean(axis=1)
    summary = {
        "status": "encoder_irf_validation_probe_complete",
        "interpretation": "linear IRF readability from frozen encoder features; not VLM report accuracy",
        "train_images": len(train_rows), "train_patients": 61,
        "validation_images": len(val_rows), "validation_patients": 13, "test_images_used": 0,
        "split_manifest_sha256": train_receipt["split_manifest_sha256"],
        "adapted_encoder_sha256": EXPECTED_ADAPTED_SHA256,
        "preprocessing": "full_frame_resize_192_bilinear_grayscale_0_to_1",
        "probe": "StandardScaler and L2 logistic regression C=1, trained on training patients only, equal total weight per training patient",
        "validation_IRF_counts": dict(Counter(map(str, y_val))),
        "released": results["released"], "adapted_byol": results["adapted_byol"],
        "adapted_minus_released_patient_macro_auroc": float(deltas.mean()),
        "adapted_minus_released_patient_macro_auroc_bootstrap_95pct": [float(x) for x in np.quantile(bootstrap, [0.025, 0.975])],
        "patients_improved": int((deltas > 0).sum()),
        "patients_worsened": int((deltas < 0).sum()),
        "patients_tied": int((deltas == 0).sum()),
    }
    args.out_metrics.parent.mkdir(parents=True, exist_ok=True)
    args.out_metrics.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with args.out_predictions.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("source_image_path", "patient_key", "expert_IRF", "released_probe_score", "adapted_probe_score"))
        for path, patient, label, a, b in zip(val_paths, val_patients, y_val,
                                             scores["released"], scores["adapted_byol"]):
            writer.writerow((path, patient, int(label), float(a), float(b)))
    print(json.dumps({key: summary[key] for key in ("status", "adapted_minus_released_patient_macro_auroc",
                                                   "adapted_minus_released_patient_macro_auroc_bootstrap_95pct",
                                                   "patients_improved", "patients_worsened")}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--released-checkpoint", type=Path, required=True)
    parser.add_argument("--adapted-checkpoint", type=Path, required=True)
    parser.add_argument("--authors-code", type=Path, required=True)
    parser.add_argument("--out-metrics", type=Path, required=True)
    parser.add_argument("--out-predictions", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run(args)
