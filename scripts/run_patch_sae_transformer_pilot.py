from __future__ import annotations

"""Experiment 8: label-free patch SAE and spatially grounded Transformer pilot.

Stages are deliberately separate.  ``select`` and ``sae`` never read the DR
label CSV.  ``predict`` writes a frozen specification before it materialises
pilot-holdout targets or predictions.
"""

import argparse
import copy
import hashlib
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from retina_lanse.data.embedding_shards import load_embedding_run
from retina_lanse.data.manifest import sha256_path
from retina_lanse.evaluation.metrics import evaluate_embeddings_batched
from retina_lanse.sae.checkpoints import load_checkpoint, save_checkpoint
from retina_lanse.sae.model import TopKSAE
from retina_lanse.sae.trainer import train_sae_batched
from retina_lanse.utils import set_deterministic
from scripts.extract_retfound_embeddings import read_jsonl
from scripts.run_global_sae_search import choose_device, require_external
from scripts.run_label_efficiency_external_benchmark import (
    binary_difference,
    binary_metrics,
    collect_eyepacs,
    fit_probe,
    fit_youden_threshold,
    load_eyepacs_labels,
    multiclass_metrics,
    predict_probe,
    threshold_metrics,
)


SEED = 4106
MODEL_SEEDS = (4106, 4107, 4108)
PATIENTS = 2000
GROUP_SIZES = {"sae_train": 1600, "development": 200, "pilot_holdout": 200}
GRID = 14
PATCHES = GRID * GRID
INPUT_DIM = 1024
DICTIONARY = 16384
TOP_K = 16
MODEL_DIM = 128
ARMS = ("dense_mlp", "value_only", "geometry_no_position", "full", "permuted_geometry")


def stable_rank(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def load_patch_run(root: Path) -> dict:
    evidence_path = root / "patch_extraction_evidence.json"
    if not evidence_path.is_file():
        raise RuntimeError(f"patch extraction evidence is absent: {evidence_path}")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence.get("status") != "complete" or evidence.get("failures") != 0:
        raise RuntimeError("patch extraction evidence is incomplete")
    token_rows, global_rows, paths, patients = [], [], [], []
    for record in evidence.get("shards", []):
        shard = Path(record["path"])
        if not shard.is_file() or sha256_path(shard) != record["sha256"]:
            raise RuntimeError(f"patch shard is missing or hash-mismatched: {shard}")
        payload = torch.load(shard, map_location="cpu", weights_only=False)
        tokens = payload.get("patch_tokens")
        if not torch.is_tensor(tokens) or tokens.ndim != 3 or tokens.shape[1:] != (PATCHES, INPUT_DIM):
            raise RuntimeError(f"invalid patch-token shape: {getattr(tokens, 'shape', None)}")
        if payload.get("grid_rows") != GRID or payload.get("grid_columns") != GRID:
            raise RuntimeError("patch grid is not 14x14")
        token_rows.append(tokens.float())
        global_rows.append(payload["pooled_global_embeddings"].float())
        paths.extend(payload["relative_paths"])
        patients.extend(str(item) for item in payload["patient_ids"])
    tokens = torch.cat(token_rows)
    globals_ = torch.cat(global_rows)
    if len(tokens) != int(evidence["images"]) or len(set(paths)) != len(paths):
        raise RuntimeError("patch run metadata does not match evidence")
    return {"tokens": tokens, "global": globals_, "paths": paths, "patients": np.asarray(patients), "evidence": evidence}


def create_cohorts(manifest: Path, output: Path) -> dict:
    records = [row for row in read_jsonl(manifest) if row["split"] == "train"]
    patients = sorted({str(row["patient_id"]) for row in records}, key=lambda value: stable_rank(SEED, value))
    if len(patients) < PATIENTS:
        raise RuntimeError("training manifest has fewer than 2,000 patients")
    chosen = patients[:PATIENTS]
    ordered = sorted(chosen, key=lambda value: stable_rank(SEED + 1, value))
    cursor = 0
    groups = {}
    for group, size in GROUP_SIZES.items():
        groups[group] = ordered[cursor : cursor + size]
        cursor += size
    if cursor != PATIENTS or len(set().union(*map(set, groups.values()))) != PATIENTS:
        raise RuntimeError("patient cohort construction failed")
    record = {
        "schema_version": "1.0",
        "status": "complete_label_blind_patient_selection",
        "selection_seed": SEED,
        "manifest": str(manifest),
        "manifest_sha256": sha256_path(manifest),
        "patients": groups,
        "counts": {name: len(values) for name, values in groups.items()},
        "labels_loaded": False,
    }
    path = output / "patient_cohorts.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != record:
            raise RuntimeError(f"existing cohorts are incompatible: {path}")
    else:
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def group_indices(patients: np.ndarray, cohorts: dict) -> dict[str, np.ndarray]:
    groups = {name: set(values) for name, values in cohorts["patients"].items()}
    result = {name: np.asarray([i for i, patient in enumerate(patients) if patient in selected], dtype=np.int64) for name, selected in groups.items()}
    seen = set()
    for name, indices in result.items():
        observed = {str(patients[i]) for i in indices}
        if not observed or observed != groups[name] or seen & observed:
            raise RuntimeError(f"patch images do not exactly cover the {name} cohort")
        seen.update(observed)
    return result


def normalise_patch_vectors(tokens: torch.Tensor, train_image_indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    train = tokens[torch.from_numpy(train_image_indices)].reshape(-1, INPUT_DIM)
    mean = train.mean(dim=0)
    std = train.std(dim=0, unbiased=False).clamp_min(1e-6)
    return mean, std


@torch.inference_mode()
def encode_topk(model: TopKSAE, tokens: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, device: torch.device, batch: int = 512) -> tuple[np.ndarray, np.ndarray]:
    ids, values = [], []
    flat = tokens.reshape(-1, INPUT_DIM)
    model = model.to(device).eval()
    for start in range(0, len(flat), batch):
        item = (flat[start : start + batch] - mean) / std
        activations = torch.relu(model.encoder(item.to(device)))
        value, index = torch.topk(activations, TOP_K, dim=1)
        ids.append(index.cpu().numpy().astype(np.int16))
        values.append(value.cpu().numpy().astype(np.float32))
    return np.concatenate(ids).reshape(len(tokens), PATCHES, TOP_K), np.concatenate(values).reshape(len(tokens), PATCHES, TOP_K)


def decoder_geometry(model: TopKSAE) -> torch.Tensor:
    """Frozen PCA projection of decoder directions, fitted only on the SAE dictionary."""
    directions = model.decoder.weight.detach().cpu().T.float()
    directions = directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-8)
    centered = directions - directions.mean(dim=0, keepdim=True)
    _, _, vectors = torch.pca_lowrank(centered, q=MODEL_DIM, center=False, niter=4)
    geometry = centered @ vectors[:, :MODEL_DIM]
    return geometry / geometry.norm(dim=1, keepdim=True).clamp_min(1e-8)


@dataclass
class Scaling:
    dense_mean: np.ndarray
    dense_std: np.ndarray
    activation_mean: float
    activation_std: float


def fit_scaling(dense: np.ndarray, values: np.ndarray) -> Scaling:
    mean = dense.mean(0, keepdims=True).astype(np.float32)
    std = dense.std(0, keepdims=True).astype(np.float32)
    std[std < 1e-6] = 1.0
    logged = np.log1p(values).reshape(-1)
    return Scaling(mean, std, float(logged.mean()), max(float(logged.std()), 1e-6))


def apply_scaling(dense: np.ndarray, values: np.ndarray, scaling: Scaling) -> tuple[np.ndarray, np.ndarray]:
    return ((dense - scaling.dense_mean) / scaling.dense_std).astype(np.float32), ((np.log1p(values) - scaling.activation_mean) / scaling.activation_std).astype(np.float32)


class PatchDataset(Dataset):
    def __init__(self, dense, ids, values, targets):
        self.dense = torch.as_tensor(dense, dtype=torch.float32)
        self.ids = torch.as_tensor(ids, dtype=torch.long)
        self.values = torch.as_tensor(values, dtype=torch.float32)
        self.targets = torch.as_tensor(targets)
    def __len__(self): return len(self.targets)
    def __getitem__(self, index): return self.dense[index], self.ids[index], self.values[index], self.targets[index]


class DenseMLP(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1024, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, output_dim))
    def forward(self, dense, ids, values):
        del ids, values
        return self.net(dense)


class PatchSAETransformer(nn.Module):
    def __init__(self, output_dim: int, geometry: torch.Tensor, *, use_geometry: bool, use_position: bool, permutation: torch.Tensor | None = None):
        super().__init__()
        self.use_geometry, self.use_position = use_geometry, use_position
        if permutation is not None:
            geometry = geometry[permutation]
        self.register_buffer("geometry", geometry)
        self.dense_projection = nn.Linear(1024, MODEL_DIM)
        self.value_projection = nn.Sequential(nn.Linear(1, MODEL_DIM), nn.GELU(), nn.Linear(MODEL_DIM, MODEL_DIM))
        self.identity_residual = nn.Embedding(DICTIONARY, MODEL_DIM) if use_geometry else None
        if self.identity_residual is not None:
            nn.init.zeros_(self.identity_residual.weight)
        self.pool_query = nn.Parameter(torch.empty(MODEL_DIM))
        self.cls = nn.Parameter(torch.empty(1, 1, MODEL_DIM))
        self.row_embedding = nn.Embedding(GRID, MODEL_DIM) if use_position else None
        self.column_embedding = nn.Embedding(GRID, MODEL_DIM) if use_position else None
        self.type_embedding = nn.Embedding(3, MODEL_DIM)
        layer = nn.TransformerEncoderLayer(d_model=MODEL_DIM, nhead=8, dim_feedforward=512, dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=4, norm=nn.LayerNorm(MODEL_DIM))
        self.head = nn.Linear(MODEL_DIM, output_dim)
        nn.init.normal_(self.pool_query, std=0.02); nn.init.normal_(self.cls, std=0.02)
    def forward(self, dense, ids, values):
        batch = dense.shape[0]
        entry = self.value_projection(values.unsqueeze(-1))
        if self.use_geometry:
            entry = entry + self.geometry[ids] + self.identity_residual(ids)
        scores = torch.einsum("bpkd,d->bpk", entry, self.pool_query) / math.sqrt(MODEL_DIM)
        weights = torch.softmax(scores, dim=-1)
        patch = torch.sum(entry * weights.unsqueeze(-1), dim=2)
        if self.use_position:
            rows = torch.arange(GRID, device=dense.device).repeat_interleave(GRID)
            columns = torch.arange(GRID, device=dense.device).repeat(GRID)
            patch = patch + self.row_embedding(rows)[None] + self.column_embedding(columns)[None]
        cls = self.cls.expand(batch, -1, -1) + self.type_embedding(torch.zeros(1, device=dense.device, dtype=torch.long))[None]
        global_token = self.dense_projection(dense).unsqueeze(1) + self.type_embedding(torch.ones(1, device=dense.device, dtype=torch.long))[None]
        patch = patch + self.type_embedding(torch.full((1,), 2, device=dense.device, dtype=torch.long))[None]
        sequence = torch.cat([cls, global_token, patch], dim=1)
        return self.head(self.encoder(sequence)[:, 0])
    def residual_penalty(self):
        return self.identity_residual.weight.square().mean() if self.identity_residual is not None else self.pool_query.square().sum() * 0.0


def build_model(arm: str, output_dim: int, geometry: torch.Tensor) -> nn.Module:
    if arm == "dense_mlp": return DenseMLP(output_dim)
    if arm == "value_only": return PatchSAETransformer(output_dim, geometry, use_geometry=False, use_position=True)
    if arm == "geometry_no_position": return PatchSAETransformer(output_dim, geometry, use_geometry=True, use_position=False)
    if arm == "full": return PatchSAETransformer(output_dim, geometry, use_geometry=True, use_position=True)
    if arm == "permuted_geometry":
        permutation = torch.randperm(DICTIONARY, generator=torch.Generator().manual_seed(SEED + 99))
        return PatchSAETransformer(output_dim, geometry, use_geometry=True, use_position=True, permutation=permutation)
    raise ValueError(arm)


def loader(dataset, batch, shuffle, seed):
    return DataLoader(dataset, batch_size=batch, shuffle=shuffle, num_workers=0, generator=torch.Generator().manual_seed(seed))


@torch.inference_mode()
def predict(model, dataset, device, task):
    rows = []
    model.eval()
    for dense, ids, values, _ in loader(dataset, 16, False, SEED):
        logits = model(dense.to(device), ids.to(device), values.to(device))
        rows.append((torch.sigmoid(logits[:, 0]) if task == "binary" else torch.softmax(logits, dim=1)).cpu().numpy())
    return np.concatenate(rows)


def loss_for(task: str, targets: np.ndarray, device: torch.device):
    if task == "binary":
        pos, neg = max(int(targets.sum()), 1), max(int(len(targets) - targets.sum()), 1)
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], device=device))
    counts = np.bincount(targets, minlength=5)
    weight = len(targets) / (5 * np.maximum(counts, 1))
    return nn.CrossEntropyLoss(weight=torch.tensor(weight, dtype=torch.float32, device=device))


def fit_development(arm, task, train, development, geometry, seed, device, max_epochs=50, patience=8):
    set_deterministic(seed); model = build_model(arm, 1 if task == "binary" else 5, geometry).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = loss_for(task, train.targets.numpy(), device); best, best_epoch, stale = -np.inf, 1, 0
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train(); total = 0.0
        for dense, ids, values, targets in loader(train, 16, True, seed + epoch):
            optimizer.zero_grad(set_to_none=True); logits = model(dense.to(device), ids.to(device), values.to(device))
            loss = criterion(logits[:, 0], targets.float().to(device)) if task == "binary" else criterion(logits, targets.long().to(device))
            if hasattr(model, "residual_penalty"): loss = loss + 1e-4 * model.residual_penalty()
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); total += float(loss.item())
        score = roc_auc_score(development.targets.numpy(), predict(model, development, device, task)) if task == "binary" else roc_auc_score(np.eye(5)[development.targets.numpy()], predict(model, development, device, task), average="macro", multi_class="ovr")
        history.append({"epoch": epoch, "loss": total, "development_score": float(score)})
        if score > best + 1e-6: best, best_epoch, stale = score, epoch, 0
        else: stale += 1
        if stale >= patience: break
    return best_epoch, history


def fit_fixed(arm, task, train, geometry, seed, epochs, device):
    set_deterministic(seed); model = build_model(arm, 1 if task == "binary" else 5, geometry).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4); criterion = loss_for(task, train.targets.numpy(), device)
    for epoch in range(1, epochs + 1):
        model.train()
        for dense, ids, values, targets in loader(train, 16, True, seed + epoch):
            optimizer.zero_grad(set_to_none=True); logits = model(dense.to(device), ids.to(device), values.to(device))
            loss = criterion(logits[:, 0], targets.float().to(device)) if task == "binary" else criterion(logits, targets.long().to(device))
            if hasattr(model, "residual_penalty"): loss = loss + 1e-4 * model.residual_penalty()
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
    return model


def smoke(output: Path) -> dict:
    ids = np.tile(np.arange(TOP_K, dtype=np.int16), (2, PATCHES, 1)); values = np.ones((2, PATCHES, TOP_K), dtype=np.float32)
    geometry = torch.randn(DICTIONARY, MODEL_DIM); model = build_model("full", 1, geometry)
    dense = torch.zeros((2, 1024)); result = model(dense, torch.from_numpy(ids).long(), torch.from_numpy(values))
    if result.shape != (2, 1): raise RuntimeError("Transformer smoke shape failed")
    record = {"status": "smoke_passed", "sequence_tokens": 198, "grid": [14, 14], "top_k": TOP_K, "feature_geometry_dimension": MODEL_DIM, "labels_loaded": False}
    (output / "patch_transformer_smoke.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def run_sae(args, output: Path) -> None:
    cohorts_path = require_external(Path(args.cohorts), "cohorts")
    patch = load_patch_run(require_external(Path(args.patch_run), "patch run")); cohorts = json.loads(cohorts_path.read_text())
    indices = group_indices(patch["patients"], cohorts); mean, std = normalise_patch_vectors(patch["tokens"], indices["sae_train"])
    train = ((patch["tokens"][torch.from_numpy(indices["sae_train"])] - mean) / std).reshape(-1, INPUT_DIM)
    dev = ((patch["tokens"][torch.from_numpy(indices["development"])] - mean) / std).reshape(-1, INPUT_DIM)
    device = choose_device(args.device); set_deterministic(SEED); model = TopKSAE(INPUT_DIM, DICTIONARY, TOP_K).to(device)
    history, optimizer = train_sae_batched(model, train, dev, epochs=20, batch_size=1024, evaluation_batch_size=1024, learning_rate=3e-4, weight_decay=1e-5, seed=SEED, device=device, resample_dead=True, resample_warmup_epochs=5, resample_interval_epochs=5)
    metrics = evaluate_embeddings_batched(model, dev, batch_size=1024, device=device)
    if metrics["max_active_features"] > TOP_K or metrics["explained_variance"] < 0.85 or metrics["dead_feature_fraction"] > 0.25:
        failure = {"status": "failed_health_gate", "stage": "patch_sae", "seed": SEED, "labels_loaded": False,
                   "metrics": metrics, "gates": {"explained_variance_minimum": 0.85, "dead_feature_fraction_maximum": 0.25, "top_k_maximum": TOP_K},
                   "reason": "one_or_more_patch_sae_health_gates_failed"}
        (output / "patch_sae_failure.json").write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        raise RuntimeError(f"Patch SAE health gate failed: {metrics}")
    normalization = {"mean": mean, "std": std, "source": "sae_train_patch_vectors", "labels_loaded": False}
    torch.save(normalization, output / "patch_sae_normalization.pt")
    save_checkpoint(output / "patch_sae_checkpoint.pt", model.cpu(), optimizer, {"seed": SEED, "history": history, "metrics": metrics, "patch_evidence_sha256": sha256_path(Path(args.patch_run) / "patch_extraction_evidence.json"), "cohorts_sha256": sha256_path(Path(args.cohorts)), "labels_loaded": False})
    loaded, _ = load_checkpoint(output / "patch_sae_checkpoint.pt")
    for key, value in model.cpu().state_dict().items():
        if not torch.equal(value, loaded.state_dict()[key]): raise RuntimeError("SAE checkpoint reload mismatch")
    (output / "patch_sae_result.json").write_text(json.dumps({"status": "complete", "metrics": metrics, "history": history, "labels_loaded": False}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "patch_sae_complete", "metrics": metrics}, sort_keys=True))


def run_prediction(args, output: Path) -> None:
    # The frozen specification is persisted before holdout predictions and target metrics are built.
    specification = {"experiment": 8, "endpoint": "referable_dr_grade_ge_2", "arms": ["dense_logistic", *ARMS], "model_seeds": list(MODEL_SEEDS), "locked_or_external_inputs_allowed": False, "holdout_access": "after_specification_freeze"}
    freeze = output / "prediction_specification_freeze.json"; freeze.write_text(json.dumps(specification, indent=2, sort_keys=True) + "\n")
    cohorts_path = require_external(Path(args.cohorts), "cohorts")
    patch = load_patch_run(require_external(Path(args.patch_run), "patch run")); cohorts = json.loads(cohorts_path.read_text()); indices = group_indices(patch["patients"], cohorts)
    global_run = load_embedding_run(require_external(Path(args.global_run), "global run"), expected_split="train", verify_sha256=True)
    dense_lookup = {path: row.numpy() for path, row in zip(global_run.relative_paths, global_run.embeddings)}
    if any(path not in dense_lookup for path in patch["paths"]): raise RuntimeError("global dense embeddings do not cover patch run")
    labels = load_eyepacs_labels(require_external(Path(args.labels), "labels"))
    labeled = np.asarray([Path(path).stem.lower() in labels for path in patch["paths"]])
    if not labeled.any(): raise RuntimeError("no labelled pilot images")
    model, payload = load_checkpoint(require_external(Path(args.sae_checkpoint), "patch SAE checkpoint"))
    if payload["model_config"] != {"input_dim": INPUT_DIM, "dictionary_size": DICTIONARY, "k": TOP_K}: raise RuntimeError("wrong patch SAE configuration")
    norm = torch.load(require_external(Path(args.normalization), "patch normalization"), map_location="cpu", weights_only=False)
    mean, std = norm["mean"], norm["std"]
    device = choose_device(args.device); ids, values = encode_topk(model, patch["tokens"], mean, std, device)
    np.savez_compressed(output / "patch_activation_traces.npz", feature_ids=ids, values=values, paths=np.asarray(patch["paths"]), patients=patch["patients"])
    geometry = decoder_geometry(model)
    torch.save({"geometry": geometry, "method": "decoder_direction_pca_128", "checkpoint_sha256": sha256_path(Path(args.sae_checkpoint))}, output / "decoder_geometry.pt")
    group_labeled = {name: np.asarray([i for i in item if labeled[i]], dtype=np.int64) for name, item in indices.items()}
    if not all(len(value) for value in group_labeled.values()): raise RuntimeError("a pilot cohort contains no labelled images")
    dense = np.stack([dense_lookup[path] for path in patch["paths"]]).astype(np.float32)
    grade = np.asarray([labels.get(Path(path).stem.lower(), -1) for path in patch["paths"]], dtype=np.int64)
    train_indices, dev_indices, holdout_indices = (group_labeled["sae_train"], group_labeled["development"], group_labeled["pilot_holdout"])
    if np.unique(grade[train_indices] >= 2).size != 2 or np.unique(grade[dev_indices] >= 2).size != 2:
        raise RuntimeError("binary development cohorts must each contain referable and non-referable images")
    scaling = fit_scaling(dense[train_indices], values[train_indices]); dense_scaled, values_scaled = apply_scaling(dense, values, scaling)
    binary = (grade >= 2).astype(np.int64)
    train_dataset = PatchDataset(dense_scaled[train_indices], ids[train_indices], values_scaled[train_indices], binary[train_indices])
    dev_dataset = PatchDataset(dense_scaled[dev_indices], ids[dev_indices], values_scaled[dev_indices], binary[dev_indices])
    selected = {}; histories = {}
    for arm in ARMS:
        epochs = []
        for seed in MODEL_SEEDS:
            epoch, history = fit_development(arm, "binary", train_dataset, dev_dataset, geometry, seed, device)
            epochs.append(epoch); histories[f"{arm}:{seed}"] = history
        selected[arm] = int(statistics.median(epochs))
    # Development is now incorporated only after all per-arm epochs are fixed.
    full_indices = np.concatenate([train_indices, dev_indices]); full_scaling = fit_scaling(dense[full_indices], values[full_indices]); dense_scaled, values_scaled = apply_scaling(dense, values, full_scaling)
    np.savez_compressed(output / "prediction_scaling.npz", dense_mean=full_scaling.dense_mean, dense_std=full_scaling.dense_std,
                        activation_mean=np.asarray([full_scaling.activation_mean], dtype=np.float32),
                        activation_std=np.asarray([full_scaling.activation_std], dtype=np.float32))
    full_train = PatchDataset(dense_scaled[full_indices], ids[full_indices], values_scaled[full_indices], binary[full_indices])
    holdout = PatchDataset(dense_scaled[holdout_indices], ids[holdout_indices], values_scaled[holdout_indices], binary[holdout_indices])
    neural_predictions = {}; checkpoints = []
    for arm in ARMS:
        predictions = []
        for seed in MODEL_SEEDS:
            fitted = fit_fixed(arm, "binary", full_train, geometry, seed, selected[arm], device)
            state = output / f"{arm}_binary_seed_{seed}.pt"; torch.save({"state_dict": fitted.cpu().state_dict(), "epochs": selected[arm], "arm": arm}, state)
            predictions.append(predict(fitted, holdout, device, "binary")); checkpoints.append({"arm": arm, "seed": seed, "path": str(state), "sha256": sha256_path(state)})
        neural_predictions[arm] = np.mean(np.stack(predictions), axis=0)
    dense_logistic = fit_probe(dense[full_indices], binary[full_indices], SEED, multiclass=False)
    logistic_prediction = predict_probe(dense_logistic, dense[holdout_indices])[:, 1]
    holdout_patients = patch["patients"][holdout_indices]; holdout_target = binary[holdout_indices]
    predictions = {"dense_logistic": logistic_prediction, **neural_predictions}
    metrics = {}
    for name, score in predictions.items():
        threshold = fit_youden_threshold(binary[full_indices], predict_probe(dense_logistic, dense[full_indices])[:, 1]) if name == "dense_logistic" else 0.5
        metrics[name] = {**binary_metrics(holdout_target, score, holdout_patients, args.bootstrap_resamples, SEED), **threshold_metrics(holdout_target, score, threshold)}
    comparisons = {f"full_vs_{name}": binary_difference(holdout_target, predictions["full"], predictions[name], holdout_patients, roc_auc_score, args.bootstrap_resamples, SEED + 500 + index) for index, name in enumerate(("dense_mlp", "geometry_no_position", "permuted_geometry"))}
    five_grade = {"status": "unavailable", "reason": "a required grade is absent from the training or development cohort"}
    required_grades = set(range(5))
    if set(grade[train_indices]) == required_grades and set(grade[dev_indices]) == required_grades:
        five_train = PatchDataset(dense_scaled[train_indices], ids[train_indices], values_scaled[train_indices], grade[train_indices])
        five_dev = PatchDataset(dense_scaled[dev_indices], ids[dev_indices], values_scaled[dev_indices], grade[dev_indices])
        five_selected, five_histories, five_predictions, five_checkpoints = {}, {}, {}, []
        for arm in ARMS:
            epochs = []
            for seed in MODEL_SEEDS:
                epoch, history = fit_development(arm, "five_grade", five_train, five_dev, geometry, seed, device)
                epochs.append(epoch); five_histories[f"{arm}:{seed}"] = history
            five_selected[arm] = int(statistics.median(epochs))
        five_full_train = PatchDataset(dense_scaled[full_indices], ids[full_indices], values_scaled[full_indices], grade[full_indices])
        five_holdout = PatchDataset(dense_scaled[holdout_indices], ids[holdout_indices], values_scaled[holdout_indices], grade[holdout_indices])
        for arm in ARMS:
            ensemble = []
            for seed in MODEL_SEEDS:
                fitted = fit_fixed(arm, "five_grade", five_full_train, geometry, seed, five_selected[arm], device)
                state = output / f"{arm}_five_grade_seed_{seed}.pt"
                torch.save({"state_dict": fitted.cpu().state_dict(), "epochs": five_selected[arm], "arm": arm, "task": "five_grade"}, state)
                ensemble.append(predict(fitted, five_holdout, device, "five_grade"))
                five_checkpoints.append({"arm": arm, "seed": seed, "path": str(state), "sha256": sha256_path(state)})
            five_predictions[arm] = np.mean(np.stack(ensemble), axis=0)
        if set(grade[holdout_indices]) == required_grades:
            five_metrics = {name: multiclass_metrics(grade[holdout_indices], probability) for name, probability in five_predictions.items()}
        else:
            five_metrics = {name: {"status": "unavailable", "reason": "pilot holdout lacks one or more grades"} for name in five_predictions}
        five_grade = {"status": "complete", "selected_epochs": five_selected, "development_histories": five_histories,
                      "metrics": five_metrics, "checkpoints": five_checkpoints, "probabilities": five_predictions}
        checkpoints.extend(five_checkpoints)
    arrays = {name: score.astype(np.float32) for name, score in predictions.items()}
    if five_grade["status"] == "complete":
        arrays.update({f"five_grade_{name}": probability.astype(np.float32) for name, probability in five_grade.pop("probabilities").items()})
    np.savez_compressed(output / "pilot_holdout_predictions.npz", **arrays, targets=holdout_target, grades=grade[holdout_indices], paths=np.asarray([patch["paths"][i] for i in holdout_indices]), patients=holdout_patients)
    result = {"status": "complete", "specification_sha256": sha256_path(freeze), "cohorts_sha256": sha256_path(Path(args.cohorts)), "patch_evidence_sha256": sha256_path(Path(args.patch_run) / "patch_extraction_evidence.json"), "sae_checkpoint_sha256": sha256_path(Path(args.sae_checkpoint)), "labels_sha256": sha256_path(Path(args.labels)), "selected_epochs": selected, "development_histories": histories, "holdout": {"images": len(holdout_indices), "patients": len(set(holdout_patients))}, "metrics": metrics, "five_grade": five_grade, "comparisons": comparisons, "checkpoints": checkpoints, "scope": "pilot holdout only; no locked EyePACS or IDRiD inputs were accepted"}
    (output / "patch_transformer_pilot_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "prediction_complete", "full_auroc": metrics["full"]["auroc"]["estimate"], "dense_mlp_auroc": metrics["dense_mlp"]["auroc"]["estimate"]}, sort_keys=True))


def main() -> int:
    global TOP_K
    parser = argparse.ArgumentParser(description="Experiment 8 patch-SAE spatial Transformer pilot")
    parser.add_argument("--stage", choices=("select", "sae", "predict", "smoke"), required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--manifest"); parser.add_argument("--cohorts")
    parser.add_argument("--patch-run"); parser.add_argument("--global-run"); parser.add_argument("--labels"); parser.add_argument("--sae-checkpoint"); parser.add_argument("--normalization")
    parser.add_argument("--device", default="auto"); parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--top-k", type=int, choices=(16, 32, 64), default=16,
                        help="Per-patch SAE activation budget. Use a distinct output directory for each capacity.")
    args = parser.parse_args(); TOP_K = args.top_k; output = require_external(Path(args.output), "output"); output.mkdir(parents=True, exist_ok=True)
    if args.stage == "smoke": print(json.dumps(smoke(output), sort_keys=True)); return 0
    if args.stage == "select":
        if not args.manifest: parser.error("select requires --manifest")
        create_cohorts(require_external(Path(args.manifest), "manifest"), output); print(json.dumps({"status": "cohorts_complete", "output": str(output / "patient_cohorts.json")})); return 0
    if args.stage == "sae":
        if not args.patch_run or not args.cohorts: parser.error("sae requires --patch-run and --cohorts")
        run_sae(args, output); return 0
    if not all((args.patch_run, args.cohorts, args.global_run, args.labels, args.sae_checkpoint, args.normalization)):
        parser.error("predict requires patch-run, cohorts, global-run, labels, sae-checkpoint, and normalization")
    run_prediction(args, output); return 0


if __name__ == "__main__":
    raise SystemExit(main())
