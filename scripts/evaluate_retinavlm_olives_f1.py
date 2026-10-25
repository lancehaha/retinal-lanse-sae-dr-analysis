"""Patient-disjoint frozen-encoder probes for OLIVES findings and PRIME DRSS.

The BYOL encoder and released encoder are frozen. Only small logistic probes are
fitted on training patients. Findings are scored per OCT B-scan; DRSS is scored
per eye/visit after averaging the features of that visit's B-scans.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

SPLIT_SHA = "28690d3ced5cdd3349100448b8f471cb56bb1690b0502c28148d055a266068fe"
JOIN_SHA = "68b5fd90074f6acd50f609683fc08977b0ac53d85547b906d8cda42150a66229"
ADAPTED_SHA = "c2e433c7d64d747b2b3f3a14f28cbd9c2d153cfb1b5270093b07b8b32c9ac50c"
INDEX_SHA = "65dfbceb982e991eaa66fa381aaeaf042e393dcfa1f16bad14da41ff2c1fe7a2"
FINDINGS = {"B1": "IR HRF", "B2": "Partially attached vitreous face",
            "B3": "Fully attached vitreous face", "B4": "Intraretinal fluid",
            "B5": "DRT/ME", "B6": "Vitreous debris", "SRF": "Subretinal fluid"}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class Images:
    def __init__(self, rows, root):
        self.rows, self.root = rows, root

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from PIL import Image
        from torchvision.transforms.functional import to_tensor
        row = self.rows[index]
        path = self.root / row["source_image_path"].lstrip("/")
        with Image.open(path) as image:
            if image.mode not in {"L", "RGB"}:
                raise ValueError(f"Unexpected image mode: {path}")
            image = image.convert("L").resize((192, 192), Image.Resampling.BILINEAR)
            return to_tensor(image)


def load_models(args):
    import torch
    from safetensors import safe_open

    sys.path.insert(0, str(args.authors_code))
    from models.ssl_config import ssl_backbone
    index_path = args.released_checkpoint / "model.safetensors.index.json"
    if sha256(index_path) != INDEX_SHA or sha256(args.adapted_checkpoint) != ADAPTED_SHA:
        raise ValueError("Encoder checkpoint provenance mismatch")
    weight_map = json.loads(index_path.read_text())["weight_map"]
    prefix = "model.visual_encoder.model."
    released = ssl_backbone(channels_in=1)
    selected = {k[len(prefix):]: v for k, v in weight_map.items() if k.startswith(prefix)}
    if set(selected) != set(released.state_dict()):
        raise ValueError("Released encoder tensor keys differ")
    state = {}
    for shard in sorted(set(selected.values())):
        with safe_open(args.released_checkpoint / shard, framework="pt", device="cpu") as f:
            for key, filename in selected.items():
                if filename == shard:
                    state[key] = f.get_tensor(prefix + key)
    released.load_state_dict(state, strict=True)
    checkpoint = torch.load(args.adapted_checkpoint, map_location="cpu")
    if (checkpoint["format"] != "retinavlm_olives_byol_encoder_full_v1"
            or checkpoint["step"] != 2083
            or checkpoint["metrics"]["split_manifest_sha256"] != SPLIT_SHA):
        raise ValueError("Adapted encoder provenance mismatch")
    adapted = ssl_backbone(channels_in=1)
    adapted.load_state_dict(checkpoint["encoder_state_dict"], strict=True)
    return {"released": released.cuda().eval(), "adapted_byol": adapted.cuda().eval()}


def embed(models, rows, image_root, batch_size, workers):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(Images(rows, image_root), batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    result = {name: [] for name in models}
    with torch.inference_mode():
        for index, images in enumerate(loader, 1):
            images = images.cuda(non_blocking=True)
            for name, model in models.items():
                result[name].append(model(images).float().cpu().numpy())
            if index % 25 == 0 or index == len(loader):
                print(f"Embedded {min(index * batch_size, len(rows))}/{len(rows)}", flush=True)
    return {name: np.concatenate(parts) for name, parts in result.items()}


def patient_weights(rows):
    import numpy as np
    counts = Counter(r["patient_key"] for r in rows)
    return np.array([1.0 / counts[r["patient_key"]] for r in rows])


def fit_binary(x_train, y_train, rows_train, x_val, x_test):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    valid = np.isin(y_train, [0, 1])
    if set(y_train[valid]) != {0, 1}:
        raise ValueError("Training labels lack a class")
    scaler = StandardScaler()
    xx = scaler.fit_transform(x_train[valid])
    weights = patient_weights([row for row, keep in zip(rows_train, valid) if keep])
    weights *= len(weights) / weights.sum()
    clf = LogisticRegression(C=1, solver="liblinear", max_iter=1000, random_state=42)
    clf.fit(xx, y_train[valid], sample_weight=weights)
    return (clf.predict_proba(scaler.transform(x_val))[:, 1],
            clf.predict_proba(scaler.transform(x_test))[:, 1])


def choose_threshold(y, score):
    import numpy as np
    from sklearn.metrics import f1_score
    good = np.isin(y, [0, 1])
    # A score equal to a positive example is a complete candidate threshold.
    candidates = np.unique(np.r_[0.5, score[good & (y == 1)]])
    choices = [(f1_score(y[good], score[good] >= t, zero_division=0),
                -abs(float(t) - 0.5), float(t)) for t in candidates]
    return max(choices)[2]


def binary_metrics(y, score, threshold):
    import numpy as np
    from sklearn.metrics import f1_score, confusion_matrix
    good = np.isin(y, [0, 1])
    yt, yp = y[good], score[good] >= threshold
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    return {"F1": float(f1_score(yt, yp, zero_division=0)), "threshold": float(threshold),
            "n": int(good.sum()), "positive": int(yt.sum()),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}


def aggregate_visits(rows, features, join):
    import numpy as np
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        entry = join.get(row["source_image_path"])
        if entry is not None:
            groups[entry["visit_key"]].append((i, entry))
    visits = []
    vectors = []
    for key, entries in sorted(groups.items()):
        indices = [i for i, _ in entries]
        labels = {int(e["DRSS"]) for _, e in entries}
        patients = {e["patient_key"] for _, e in entries}
        if len(labels) != 1 or len(patients) != 1 or len(indices) != 49:
            raise ValueError(f"Bad DRSS visit: {key}, {len(indices)} scans")
        visits.append({"visit_key": key, "patient_key": patients.pop(), "DRSS": labels.pop(), "scans": len(indices)})
        vectors.append(features[indices].mean(axis=0))
    return visits, np.stack(vectors)


def drss_metrics(y, pred, classes):
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    supported = sorted(set(y))
    return {"visits": len(y), "true_counts": dict(Counter(map(str, y))),
            "predicted_counts": dict(Counter(map(str, pred))),
            "accuracy": float(accuracy_score(y, pred)),
            "macro_F1_all_train_classes": float(f1_score(y, pred, labels=classes, average="macro", zero_division=0)),
            "macro_F1_test_supported_classes": float(f1_score(y, pred, labels=supported, average="macro", zero_division=0)),
            "weighted_F1": float(f1_score(y, pred, labels=classes, average="weighted", zero_division=0)),
            "class_order": classes, "confusion": confusion_matrix(y, pred, labels=classes).tolist()}


def run(args):
    import numpy as np
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    if not torch.cuda.is_available() or sha256(args.split_csv) != SPLIT_SHA or sha256(args.drss_join) != JOIN_SHA:
        raise ValueError("CUDA or frozen input provenance missing")
    all_rows = read_csv(args.split_csv)
    join_rows = read_csv(args.drss_join)
    join = {r["source_image_path"]: r for r in join_rows}
    if len(all_rows) != 9408 or len(join_rows) != 3822 or len(join) != len(join_rows):
        raise ValueError("Unexpected cohort size or duplicate DRSS image")
    by_split = {s: [r for r in all_rows if r["split"] == s] for s in ("train", "validation", "test")}
    expected = {"train": 6664, "validation": 1470, "test": 1274}
    if {s: len(rows) for s, rows in by_split.items()} != expected:
        raise ValueError("Unexpected frozen split")
    patients = {s: {r["patient_key"] for r in rows} for s, rows in by_split.items()}
    if any(patients[a] & patients[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Patient leakage across frozen splits")
    for s, rows in by_split.items():
        root = getattr(args, s + "_images")
        for row in rows:
            path = root / row["source_image_path"].lstrip("/")
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"Missing or linked image: {path}")
    models = load_models(args)
    embeddings = {s: embed(models, rows, getattr(args, s + "_images"), args.batch_size, args.workers)
                  for s, rows in by_split.items()}
    result = {"status": "olives_frozen_encoder_f1_complete", "split_sha256": SPLIT_SHA,
              "drss_join_sha256": JOIN_SHA, "adapted_encoder_sha256": ADAPTED_SHA,
              "interpretation": "Frozen encoder plus training-only linear probes; no adapter or LLM fine-tuning",
              "findings": {}, "DRSS": {}}
    predictions = []
    for finding, name in FINDINGS.items():
        ys = {s: np.array([int(r[finding]) if r[finding] in {"0", "1"} else -1
                            for r in rows]) for s, rows in by_split.items()}
        block = {"name": name, "encoders": {}}
        for model_name in models:
            val_score, test_score = fit_binary(embeddings["train"][model_name], ys["train"], by_split["train"],
                                               embeddings["validation"][model_name], embeddings["test"][model_name])
            threshold = choose_threshold(ys["validation"], val_score)
            block["encoders"][model_name] = {
                "validation": binary_metrics(ys["validation"], val_score, threshold),
                "test": binary_metrics(ys["test"], test_score, threshold)}
            for split, scores in (("validation", val_score), ("test", test_score)):
                predictions.extend({"task": finding, "encoder": model_name, "split": split,
                                    "source_image_path": r["source_image_path"], "patient_key": r["patient_key"],
                                    "label": r[finding], "score": float(score), "threshold": threshold}
                                   for r, score in zip(by_split[split], scores))
        result["findings"][finding] = block
        print(json.dumps({"finding": finding, "released_test_F1": block["encoders"]["released"]["test"]["F1"],
                          "adapted_test_F1": block["encoders"]["adapted_byol"]["test"]["F1"]}), flush=True)
    for model_name in models:
        visits, vectors = {}, {}
        for split in by_split:
            visits[split], vectors[split] = aggregate_visits(by_split[split], embeddings[split][model_name], join)
        if {s: len(v) for s, v in visits.items()} != {"train": 54, "validation": 12, "test": 12}:
            raise ValueError("DRSS visit join count changed")
        y = {s: np.array([v["DRSS"] for v in visits[s]]) for s in visits}
        scaler = StandardScaler()
        x = scaler.fit_transform(vectors["train"])
        weights = patient_weights(visits["train"])
        weights *= len(weights) / weights.sum()
        classifier = LogisticRegression(C=1, solver="lbfgs", max_iter=3000, random_state=42)
        classifier.fit(x, y["train"], sample_weight=weights)
        block = {"training_visits": len(y["train"]), "classes": list(map(int, classifier.classes_)),
                 "validation": {}, "test": {}}
        for split in ("validation", "test"):
            pred = classifier.predict(scaler.transform(vectors[split]))
            block[split] = drss_metrics(y[split], pred, list(map(int, classifier.classes_)))
            for visit, predicted in zip(visits[split], pred):
                predictions.append({"task": "DRSS_visit", "encoder": model_name, "split": split,
                                    "source_image_path": visit["visit_key"], "patient_key": visit["patient_key"],
                                    "label": visit["DRSS"], "score": int(predicted), "threshold": ""})
        result["DRSS"][model_name] = block
        print(json.dumps({"DRSS_encoder": model_name,
                          "test_macro_F1_all_train_classes": block["test"]["macro_F1_all_train_classes"]}), flush=True)
    result["DRSS"]["majority_baseline"] = {}
    majority = Counter(v["DRSS"] for v in visits["train"]).most_common(1)[0][0]
    for split in ("validation", "test"):
        yy = np.array([v["DRSS"] for v in visits[split]])
        result["DRSS"]["majority_baseline"][split] = drss_metrics(yy, np.full_like(yy, majority),
                                                                    list(map(int, classifier.classes_)))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "olives_f1_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.out_dir / "olives_f1_predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    print(json.dumps({"status": result["status"], "metrics": str(args.out_dir / "olives_f1_metrics.json"),
                      "predictions": str(args.out_dir / "olives_f1_predictions.csv")}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("split-csv", "drss-join", "train-images", "validation-images", "test-images",
                 "released-checkpoint", "adapted-checkpoint", "authors-code", "out-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    run(p.parse_args())
