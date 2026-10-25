from __future__ import annotations

"""Experiment 7C: SAE-only, unordered feature-view Transformers.

The three primary arms differ only in how a selected SAE feature is represented:
its frozen feature-card text, its frozen decoder direction, or a learned categorical
ID.  All arms receive the same activation features.  Dense RETFound vectors, input
rank, rank groups, and positional embeddings never enter a model.
"""

import argparse
import copy
import csv
import hashlib
import json
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from retina_lanse.data.embedding_shards import load_embedding_run
from retina_lanse.data.manifest import sha256_path
from retina_lanse.sae.checkpoints import load_checkpoint
from retina_lanse.utils import set_deterministic
from scripts.run_feature_card_dr_prediction import fit_probe, predict_probe, qwk_difference, stable_dev_mask
from scripts.run_global_sae_search import require_external
from scripts.run_label_efficiency_external_benchmark import (
    binary_difference,
    binary_metrics,
    collect_eyepacs,
    collect_idrid,
    fit_youden_threshold,
    load_eyepacs_labels,
    load_idrid_manifest,
    multiclass_metrics,
    sample_patient_indices,
    threshold_metrics,
)


SEEDS = (4106, 4107, 4108, 4109, 4110)
FRACTIONS = ("1.00", "0.10")
DICTIONARY_SIZE = 16384
SELECTED_FEATURES = 251
TRACE_WIDTH = 1024
MODEL_DIM = 128
TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PRIMARY_ARMS = ("card_transformer", "decoder_transformer", "id_transformer")
NEURAL_ARMS = ("activation_transformer", *PRIMARY_ARMS)
ALL_ARMS = ("activation_logistic", *NEURAL_ARMS)
PASS_STATUSES = {"passed_initial", "revised_and_passed"}
SHUFFLE_SEED = 7510


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(path)).encode("utf-8"))
        with child.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def load_passed_cards(path: Path) -> tuple[np.ndarray, list[str], list[dict]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    passed = [row for row in rows if row.get("final_status") in PASS_STATUSES]
    if len(passed) != SELECTED_FEATURES:
        raise RuntimeError(f"expected exactly {SELECTED_FEATURES} passed cards; found {len(passed)}")
    if len({int(row["feature_id"]) for row in passed}) != SELECTED_FEATURES:
        raise RuntimeError("passed feature-card IDs are not unique")
    selected = []
    for row in passed:
        versions = row.get("hypothesis_versions")
        if not isinstance(versions, list) or not versions:
            raise RuntimeError(f"feature {row.get('feature_id')} has no hypothesis versions")
        text = str(versions[-1].get("hypothesis", "")).strip()
        if not text:
            raise RuntimeError(f"feature {row.get('feature_id')} has no final hypothesis text")
        selected.append((int(row["feature_id"]), text, str(row["final_status"])))
    selected.sort(key=lambda item: item[0])
    ids = np.asarray([item[0] for item in selected], dtype=np.int64)
    texts = [item[1] for item in selected]
    audit = [{"feature_id": item[0], "final_hypothesis": item[1], "final_status": item[2]} for item in selected]
    if np.any(ids < 0) or np.any(ids >= DICTIONARY_SIZE):
        raise RuntimeError("selected card ID is outside the SAE dictionary")
    return ids, texts, audit


def pca_bank(values: np.ndarray) -> tuple[np.ndarray, dict]:
    if values.shape[0] != SELECTED_FEATURES or not np.all(np.isfinite(values)):
        raise RuntimeError("PCA input must contain one finite row per selected feature")
    pca = PCA(n_components=MODEL_DIM, svd_solver="full")
    projected = pca.fit_transform(values.astype(np.float64)).astype(np.float32)
    norms = np.linalg.norm(projected, axis=1, keepdims=True)
    projected /= np.maximum(norms, 1e-8)
    return projected, {
        "components": pca.components_.astype(np.float32),
        "mean": pca.mean_.astype(np.float32),
        "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
    }


def prepare_representations(args: argparse.Namespace, output: Path) -> None:
    cards_path = require_external(Path(args.cards), "cards")
    checkpoint_path = require_external(Path(args.checkpoint), "checkpoint")
    cache = require_external(Path(args.text_model_cache), "text model cache")
    feature_ids, texts, card_audit = load_passed_cards(cards_path)
    sae, payload = load_checkpoint(checkpoint_path)
    if payload.get("model_config") != {"input_dim": 1024, "dictionary_size": DICTIONARY_SIZE, "k": 1024}:
        raise RuntimeError("Experiment 7C requires the frozen K=1024 seed-4106 SAE")
    decoder = sae.decoder.weight.detach().cpu().numpy()[:, feature_ids].T.astype(np.float32)
    decoder_norms = np.linalg.norm(decoder, axis=1)
    if not np.allclose(decoder_norms, 1.0, atol=1e-4):
        raise RuntimeError("selected SAE decoder columns are not normalized")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "representation preparation requires sentence-transformers; install it in the external Retina-LanSE runtime"
        ) from exc
    model = SentenceTransformer(TEXT_MODEL, cache_folder=str(cache), device="cpu")
    card_raw = np.asarray(model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True), dtype=np.float32)
    if card_raw.shape != (SELECTED_FEATURES, 384):
        raise RuntimeError(f"unexpected MiniLM card embedding shape: {card_raw.shape}")
    first = model._first_module() if hasattr(model, "_first_module") else None
    config = getattr(getattr(first, "auto_model", None), "config", None)
    revision = getattr(config, "_commit_hash", None)
    card_bank, card_pca = pca_bank(card_raw)
    decoder_bank, decoder_pca = pca_bank(decoder)

    banks_path = output / "feature_view_banks.npz"
    np.savez_compressed(
        banks_path,
        feature_ids=feature_ids,
        card_bank=card_bank,
        decoder_bank=decoder_bank,
        card_pca_components=card_pca["components"],
        card_pca_mean=card_pca["mean"],
        decoder_pca_components=decoder_pca["components"],
        decoder_pca_mean=decoder_pca["mean"],
    )
    record = {
        "schema_version": "1.0",
        "status": "complete",
        "selected_features": SELECTED_FEATURES,
        "feature_ids": feature_ids.tolist(),
        "cards": card_audit,
        "card_definition": "final hypothesis sentence; last hypothesis version wins",
        "text_model": TEXT_MODEL,
        "text_model_resolved_revision": revision,
        "text_model_cache": str(cache),
        "text_model_cache_sha256": sha256_tree(cache),
        "card_embedding_shape_before_pca": list(card_raw.shape),
        "card_pca_explained_variance_ratio_sum": card_pca["explained_variance_ratio_sum"],
        "decoder_shape_before_pca": list(decoder.shape),
        "decoder_pca_explained_variance_ratio_sum": decoder_pca["explained_variance_ratio_sum"],
        "banks": str(banks_path),
        "banks_sha256": sha256_path(banks_path),
        "cards_sha256": sha256_path(cards_path),
        "checkpoint_sha256": sha256_path(checkpoint_path),
        "sae_frozen": True,
        "representations_frozen": True,
        "labels_loaded": False,
    }
    metadata = output / "feature_view_banks.json"
    metadata.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": record["status"], "metadata": str(metadata), "banks": str(banks_path)}, indent=2))


def load_banks(path: Path, metadata_path: Path) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "complete" or metadata.get("selected_features") != SELECTED_FEATURES:
        raise RuntimeError("feature-view bank metadata is incomplete")
    if metadata.get("banks_sha256") != sha256_path(path):
        raise RuntimeError("feature-view bank hash mismatch")
    payload = np.load(path, allow_pickle=False)
    ids = payload["feature_ids"].astype(np.int64)
    banks = {"card_transformer": payload["card_bank"].astype(np.float32), "decoder_transformer": payload["decoder_bank"].astype(np.float32)}
    if ids.shape != (SELECTED_FEATURES,) or len(np.unique(ids)) != SELECTED_FEATURES:
        raise RuntimeError("invalid selected feature IDs")
    for name, bank in banks.items():
        if bank.shape != (SELECTED_FEATURES, MODEL_DIM) or not np.all(np.isfinite(bank)):
            raise RuntimeError(f"invalid frozen bank for {name}")
    return ids, banks, metadata


def selected_activation_matrix(feature_ids: np.ndarray, values: np.ndarray, selected_ids: np.ndarray) -> np.ndarray:
    if feature_ids.shape != values.shape or feature_ids.ndim != 2 or feature_ids.shape[1] != TRACE_WIDTH:
        raise RuntimeError("active trace must contain aligned [N,1024] feature IDs and values")
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise RuntimeError("activation values must be finite and nonnegative")
    lookup = np.full(DICTIONARY_SIZE, -1, dtype=np.int32)
    lookup[selected_ids] = np.arange(SELECTED_FEATURES, dtype=np.int32)
    matrix = np.zeros((len(values), SELECTED_FEATURES), dtype=np.float32)
    rows = np.arange(len(values))
    for column in range(TRACE_WIDTH):
        original = feature_ids[:, column]
        valid = (original >= 0) & (original < DICTIONARY_SIZE)
        compact = np.full(len(values), -1, dtype=np.int32)
        compact[valid] = lookup[original[valid]]
        keep = (compact >= 0) & (values[:, column] > 0)
        if np.any(keep):
            if np.any(matrix[rows[keep], compact[keep]] != 0):
                raise RuntimeError("duplicate selected feature in one activation trace")
            matrix[rows[keep], compact[keep]] = values[keep, column]
    if np.any(np.count_nonzero(matrix, axis=1) == 0):
        raise RuntimeError("an evaluated image has no active selected feature")
    return matrix


@dataclass
class ActivationScaling:
    mean: np.ndarray
    std: np.ndarray


def raw_activation_features(matrix: np.ndarray) -> np.ndarray:
    active = matrix > 0
    totals = matrix.sum(axis=1, keepdims=True)
    features = np.zeros((*matrix.shape, 2), dtype=np.float32)
    features[..., 0] = np.log1p(matrix)
    shares = np.zeros_like(matrix)
    np.divide(matrix, np.maximum(totals, 1e-12), out=shares, where=active)
    features[..., 1][active] = np.log(np.maximum(shares[active], 1e-12))
    return features


def fit_activation_scaling(matrix: np.ndarray) -> ActivationScaling:
    features = raw_activation_features(matrix)
    active = matrix > 0
    mean = features[active].mean(axis=0).astype(np.float32)
    std = features[active].std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return ActivationScaling(mean, std)


def apply_activation_scaling(matrix: np.ndarray, scaling: ActivationScaling) -> np.ndarray:
    features = raw_activation_features(matrix)
    active = matrix > 0
    features[active] = (features[active] - scaling.mean) / scaling.std
    features[~active] = 0.0
    return features.astype(np.float32)


class FeatureSetDataset(Dataset):
    def __init__(self, matrix: np.ndarray, activation_features: np.ndarray, targets: np.ndarray):
        active = matrix > 0
        compact = np.broadcast_to(np.arange(SELECTED_FEATURES, dtype=np.int64), matrix.shape).copy()
        compact[~active] = -1
        self.ids = torch.from_numpy(compact)
        self.activation_features = torch.from_numpy(np.asarray(activation_features, dtype=np.float32))
        self.targets = torch.from_numpy(np.asarray(targets))

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        return self.ids[index], self.activation_features[index], self.targets[index]


class SAEFeatureSetTransformer(nn.Module):
    """Permutation-invariant SAE-only Transformer; no rank or positional inputs."""

    def __init__(self, arm: str, output_dim: int, frozen_bank: np.ndarray | None = None):
        super().__init__()
        if arm not in NEURAL_ARMS:
            raise ValueError(f"unknown arm: {arm}")
        self.arm = arm
        if arm in {"card_transformer", "decoder_transformer"}:
            if frozen_bank is None or frozen_bank.shape != (SELECTED_FEATURES, MODEL_DIM):
                raise ValueError(f"{arm} requires a [{SELECTED_FEATURES},{MODEL_DIM}] frozen bank")
            self.register_buffer("frozen_bank", torch.from_numpy(frozen_bank.astype(np.float32)), persistent=True)
            self.id_embedding = None
        elif arm == "id_transformer":
            self.register_buffer("frozen_bank", torch.empty(0), persistent=False)
            self.id_embedding = nn.Embedding(SELECTED_FEATURES, MODEL_DIM)
        else:
            self.register_buffer("frozen_bank", torch.empty(0), persistent=False)
            self.id_embedding = None
        self.activation_projection = nn.Sequential(nn.Linear(2, 64), nn.GELU(), nn.Linear(64, MODEL_DIM))
        self.activation_gate = nn.Sequential(nn.Linear(2, 64), nn.GELU(), nn.Linear(64, MODEL_DIM))
        self.token_norm = nn.LayerNorm(MODEL_DIM)
        self.cls_token = nn.Parameter(torch.empty(1, 1, MODEL_DIM))
        layer = nn.TransformerEncoderLayer(
            d_model=MODEL_DIM,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=4, norm=nn.LayerNorm(MODEL_DIM))
        self.head = nn.Linear(MODEL_DIM, output_dim)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, compact_ids: torch.Tensor, activation_features: torch.Tensor, mapping: torch.Tensor | None = None) -> torch.Tensor:
        valid = compact_ids >= 0
        safe = compact_ids.clamp_min(0)
        if mapping is not None:
            safe = mapping[safe]
        if self.id_embedding is not None:
            base = self.id_embedding(safe)
        elif self.frozen_bank.numel():
            base = self.frozen_bank[safe]
        else:
            base = torch.zeros((*safe.shape, MODEL_DIM), dtype=activation_features.dtype, device=activation_features.device)
        activation = self.activation_projection(activation_features)
        gate = 1.0 + torch.sigmoid(self.activation_gate(activation_features))
        tokens = self.token_norm(base + activation) * gate
        tokens = tokens * valid.unsqueeze(-1)
        cls = self.cls_token.expand(len(tokens), -1, -1)
        sequence = torch.cat([cls, tokens], dim=1)
        padding = torch.cat([torch.zeros((len(tokens), 1), dtype=torch.bool, device=tokens.device), ~valid], dim=1)
        encoded = self.encoder(sequence, src_key_padding_mask=padding)
        return self.head(encoded[:, 0])


def build_model(arm: str, output_dim: int, banks: dict[str, np.ndarray]) -> SAEFeatureSetTransformer:
    return SAEFeatureSetTransformer(arm, output_dim, banks.get(arm))


def make_loader(dataset: FeatureSetDataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, generator=generator)


@torch.inference_mode()
def predict_model(model: nn.Module, dataset: FeatureSetDataset, *, batch_size: int, device: torch.device, task: str, mapping: torch.Tensor | None = None) -> np.ndarray:
    model.eval()
    rows = []
    mapping = mapping.to(device) if mapping is not None else None
    for ids, activations, _ in make_loader(dataset, batch_size, False, 0):
        logits = model(ids.to(device), activations.to(device), mapping=mapping)
        rows.append((torch.sigmoid(logits[:, 0]) if task == "binary" else torch.softmax(logits, dim=1)).cpu().numpy())
    return np.concatenate(rows, axis=0)


def monitor_score(task: str, targets: np.ndarray, probabilities: np.ndarray) -> float:
    if task == "binary":
        return float(roc_auc_score(targets, probabilities))
    return float(roc_auc_score(np.eye(5, dtype=np.int64)[targets], probabilities, average="macro", multi_class="ovr"))


def loss_function(task: str, targets: np.ndarray, device: torch.device) -> nn.Module:
    if task == "binary":
        positive = max(int(targets.sum()), 1)
        negative = max(int(len(targets) - positive), 1)
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negative / positive], dtype=torch.float32, device=device))
    counts = np.bincount(targets.astype(np.int64), minlength=5)
    weights = len(targets) / (5 * np.maximum(counts, 1))
    return nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))


def train_epochs(model: nn.Module, dataset: FeatureSetDataset, task: str, optimizer: torch.optim.Optimizer, *, device: torch.device, batch_size: int, accumulation: int, seed: int, epoch: int) -> float:
    model.train()
    loss_fn = loss_function(task, dataset.targets.numpy(), device)
    optimizer.zero_grad(set_to_none=True)
    total_loss, seen = 0.0, 0
    batches = make_loader(dataset, batch_size, True, seed + epoch)
    for step, (ids, activations, targets) in enumerate(batches, start=1):
        logits = model(ids.to(device), activations.to(device))
        loss = loss_fn(logits[:, 0], targets.float().to(device)) if task == "binary" else loss_fn(logits, targets.long().to(device))
        (loss / accumulation).backward()
        if step % accumulation == 0 or step == len(batches):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        total_loss += float(loss.item()) * len(targets)
        seen += len(targets)
    return total_loss / max(seen, 1)


def train_development(arm: str, task: str, train: FeatureSetDataset, dev: FeatureSetDataset, banks: dict[str, np.ndarray], *, seed: int, device: torch.device, batch_size: int, accumulation: int, max_epochs: int, patience: int) -> tuple[int, list[dict]]:
    set_deterministic(seed)
    model = build_model(arm, 1 if task == "binary" else 5, banks).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    best_score, best_epoch, stale = -np.inf, 1, 0
    best_state, history = None, []
    for epoch in range(1, max_epochs + 1):
        train_loss = train_epochs(model, train, task, optimizer, device=device, batch_size=batch_size, accumulation=accumulation, seed=seed, epoch=epoch)
        probabilities = predict_model(model, dev, batch_size=batch_size, device=device, task=task)
        score = monitor_score(task, dev.targets.numpy(), probabilities)
        history.append({"epoch": epoch, "train_loss": train_loss, "development_metric": score})
        if score > best_score + 1e-6:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("development training produced no checkpoint")
    return best_epoch, history


def train_fixed(arm: str, task: str, dataset: FeatureSetDataset, banks: dict[str, np.ndarray], *, epochs: int, seed: int, device: torch.device, batch_size: int, accumulation: int) -> nn.Module:
    set_deterministic(seed)
    model = build_model(arm, 1 if task == "binary" else 5, banks).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    for epoch in range(1, epochs + 1):
        train_epochs(model, dataset, task, optimizer, device=device, batch_size=batch_size, accumulation=accumulation, seed=seed, epoch=epoch)
    return model


def holm_adjust(records: dict[str, dict]) -> None:
    ordered = sorted(records, key=lambda key: records[key]["p_value"])
    running, total = 0.0, len(ordered)
    for rank, key in enumerate(ordered):
        adjusted = min(1.0, records[key]["p_value"] * (total - rank))
        running = max(running, adjusted)
        records[key]["holm_adjusted_p_value"] = running


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def smoke(device: torch.device) -> dict:
    rng = np.random.default_rng(SEEDS[0])
    n = 12
    matrix = np.zeros((n, SELECTED_FEATURES), dtype=np.float32)
    for row in range(n):
        columns = rng.choice(SELECTED_FEATURES, size=32, replace=False)
        matrix[row, columns] = rng.uniform(0.01, 3.0, size=len(columns))
    scaling = fit_activation_scaling(matrix[:8])
    features = apply_activation_scaling(matrix, scaling)
    targets = np.asarray([0, 1] * 6, dtype=np.int64)
    dataset = FeatureSetDataset(matrix, features, targets)
    banks = {
        "card_transformer": rng.normal(size=(SELECTED_FEATURES, MODEL_DIM)).astype(np.float32),
        "decoder_transformer": rng.normal(size=(SELECTED_FEATURES, MODEL_DIM)).astype(np.float32),
    }
    model = build_model("card_transformer", 1, banks).to(device).eval()
    ids = dataset.ids[:2].to(device)
    activations = dataset.activation_features[:2].to(device)
    permutation = torch.randperm(SELECTED_FEATURES, device=device)
    with torch.inference_mode():
        original = model(ids, activations)
        reordered = model(ids[:, permutation], activations[:, permutation])
        changed_padding = activations.clone()
        changed_padding[ids < 0] = 999.0
        padded = model(ids, changed_padding)
    if not torch.allclose(original, reordered, atol=1e-5, rtol=1e-5):
        raise RuntimeError("input-token permutation invariance failed")
    if not torch.allclose(original, padded, atol=1e-5, rtol=1e-5):
        raise RuntimeError("padding-mask invariance failed")
    train = FeatureSetDataset(matrix[:8], features[:8], targets[:8])
    dev = FeatureSetDataset(matrix[8:], features[8:], targets[8:])
    epoch, history = train_development("activation_transformer", "binary", train, dev, banks, seed=SEEDS[0], device=device, batch_size=4, accumulation=2, max_epochs=1, patience=1)
    counts = {arm: parameter_count(build_model(arm, 1, banks)) for arm in NEURAL_ARMS}
    return {
        "status": "smoke_passed",
        "device": str(device),
        "selected_features": SELECTED_FEATURES,
        "rank_inputs": False,
        "positional_embeddings": False,
        "dense_model_inputs": False,
        "permutation_invariant": True,
        "padding_mask_invariant": True,
        "gradient_accumulation_verified": True,
        "epoch": epoch,
        "history": history,
        "parameter_counts": counts,
    }


def run_experiment(args: argparse.Namespace, output: Path, device: torch.device) -> None:
    names = (
        "train_run", "validation_run", "internal_test_run", "idrid_run", "idrid_manifest",
        "labels", "active_trace", "normalization", "checkpoint", "banks", "banks_metadata",
    )
    missing = [name for name in names if not getattr(args, name)]
    if missing:
        raise RuntimeError("run stage is missing: " + ", ".join(missing))
    roots = {name: require_external(Path(getattr(args, name)), name.replace("_", " ")) for name in names}
    selected_ids, banks, bank_metadata = load_banks(roots["banks"], roots["banks_metadata"])
    if bank_metadata.get("checkpoint_sha256") != sha256_path(roots["checkpoint"]):
        raise RuntimeError("representation banks and SAE checkpoint do not match")
    _, checkpoint_payload = load_checkpoint(roots["checkpoint"])
    if checkpoint_payload.get("model_config") != {"input_dim": 1024, "dictionary_size": DICTIONARY_SIZE, "k": 1024}:
        raise RuntimeError("Experiment 7C requires the frozen K=1024 seed-4106 SAE")

    labels = load_eyepacs_labels(roots["labels"])
    idrid_manifest = load_idrid_manifest(roots["idrid_manifest"])
    train_run = load_embedding_run(roots["train_run"], expected_split="train", verify_sha256=True)
    validation_run = load_embedding_run(roots["validation_run"], expected_split="validation", verify_sha256=True)
    test_run = load_embedding_run(roots["internal_test_run"], expected_split="internal_test", verify_sha256=True)
    external_run = load_embedding_run(roots["idrid_run"], expected_split="external", verify_sha256=True)
    _, train_grades, train_patients, _ = collect_eyepacs(train_run, labels)
    _, validation_grades, validation_patients, _ = collect_eyepacs(validation_run, labels)
    _, test_grades, test_patients, _ = collect_eyepacs(test_run, labels)
    _, idrid_grades, idrid_patients, _ = collect_idrid(external_run, idrid_manifest)
    for left, right in ((train_patients, validation_patients), (train_patients, test_patients), (validation_patients, test_patients)):
        if set(left) & set(right):
            raise RuntimeError("EyePACS patient overlap detected")

    trace = np.load(roots["active_trace"], allow_pickle=False)
    trace_pairs = {
        "train": (trace["train_feature_ids"], trace["train_values"]),
        "validation": (trace["validation_feature_ids"], trace["validation_values"]),
        "locked_test": (trace["test_feature_ids"], trace["test_values"]),
        "idrid": (trace["idrid_feature_ids"], trace["idrid_values"]),
    }
    grades = {"train": train_grades, "validation": validation_grades, "locked_test": test_grades, "idrid": idrid_grades}
    patients = {"train": train_patients, "validation": validation_patients, "locked_test": test_patients, "idrid": idrid_patients}
    matrices = {}
    coverage = {}
    for split, pair in trace_pairs.items():
        if len(pair[0]) != len(grades[split]):
            raise RuntimeError(f"{split} trace and label rows are misaligned")
        matrices[split] = selected_activation_matrix(pair[0], pair[1], selected_ids)
        counts = np.count_nonzero(matrices[split], axis=1)
        coverage[split] = {"images": len(counts), "mean_active_selected": float(counts.mean()), "minimum_active_selected": int(counts.min()), "maximum_active_selected": int(counts.max())}

    fraction_indices = {"1.00": np.arange(len(train_grades), dtype=np.int64)}
    low_indices, low_patients = sample_patient_indices(train_patients, train_grades, 0.10, SEEDS[0])
    fraction_indices["0.10"] = low_indices
    specification = {
        "schema_version": "1.0",
        "experiment": "7C",
        "exploratory_reused_evaluation_sets": True,
        "selected_feature_ids": selected_ids.tolist(),
        "selected_feature_count": SELECTED_FEATURES,
        "feature_universe_rule": "final_status in {passed_initial,revised_and_passed}",
        "arms": list(ALL_ARMS),
        "primary_arms": list(PRIMARY_ARMS),
        "tasks": ["binary", "five_grade"],
        "fractions": list(FRACTIONS),
        "model": {"tokens_maximum_including_cls": SELECTED_FEATURES + 1, "dimension": MODEL_DIM, "layers": 4, "heads": 8, "ffn": 512, "dropout": 0.1, "rank_inputs": False, "positional_embeddings": False, "dense_model_inputs": False},
        "activation": {"features": ["train_standardized_log1p_raw", "train_standardized_log_within_image_share"], "projection": [2, 64, MODEL_DIM], "learned_gate": True},
        "training": {"optimizer": "AdamW", "learning_rate": 1e-4, "weight_decay": 1e-4, "batch_size": args.batch_size, "gradient_accumulation": args.gradient_accumulation, "effective_batch_size": args.batch_size * args.gradient_accumulation, "max_epochs": args.max_epochs, "patience": args.patience, "seeds": list(SEEDS)},
        "statistics": {"bootstrap_resamples": args.bootstrap_resamples, "primary_family": "three representation arms versus activation-only Transformer", "holm_correction": True, "practical_auroc_difference": 0.01},
        "coverage": coverage,
        "banks_metadata_sha256": sha256_path(roots["banks_metadata"]),
        "banks_sha256": sha256_path(roots["banks"]),
        "source_hashes": {name: sha256_path(path if path.is_file() else path / "extraction_evidence.json") for name, path in roots.items() if name not in {"banks", "banks_metadata"}},
    }
    freeze_path = output / "specification_freeze.json"
    if freeze_path.is_file():
        prior = json.loads(freeze_path.read_text(encoding="utf-8"))
        if prior != specification:
            raise RuntimeError("existing frozen specification differs from requested run")
    else:
        freeze_path.write_text(json.dumps(specification, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    split_order = ("train", "validation", "locked_test", "idrid")
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output / "training_progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.is_file() else {"schema_version": "1.0", "development": {}}
    result_fractions, selected_epochs = {}, {}
    started = time.time()

    for fraction in FRACTIONS:
        indices = fraction_indices[fraction]
        selected_patients = train_patients[indices]
        dev_mask = stable_dev_mask(selected_patients, seed=SEEDS[0], fraction=0.10)
        if not np.any(dev_mask) or not np.any(~dev_mask):
            raise RuntimeError("inner patient development split is empty")
        fit_indices, dev_indices = indices[~dev_mask], indices[dev_mask]
        fit_scaling = fit_activation_scaling(matrices["train"][fit_indices])
        fit_features = apply_activation_scaling(matrices["train"][fit_indices], fit_scaling)
        dev_features = apply_activation_scaling(matrices["train"][dev_indices], fit_scaling)
        development_epochs = defaultdict(lambda: defaultdict(list))
        histories = {}
        for task in ("binary", "five_grade"):
            fit_targets = (train_grades[fit_indices] >= 2).astype(np.int64) if task == "binary" else train_grades[fit_indices]
            dev_targets = (train_grades[dev_indices] >= 2).astype(np.int64) if task == "binary" else train_grades[dev_indices]
            fit_dataset = FeatureSetDataset(matrices["train"][fit_indices], fit_features, fit_targets)
            dev_dataset = FeatureSetDataset(matrices["train"][dev_indices], dev_features, dev_targets)
            for arm in NEURAL_ARMS:
                for seed in SEEDS:
                    key = f"{fraction}:{task}:{arm}:{seed}"
                    prior = progress["development"].get(key)
                    if prior and prior.get("max_epochs") == args.max_epochs and prior.get("patience") == args.patience:
                        epoch, history = int(prior["selected_epoch"]), prior["history"]
                    else:
                        epoch, history = train_development(
                            arm, task, fit_dataset, dev_dataset, banks,
                            seed=seed, device=device, batch_size=args.batch_size,
                            accumulation=args.gradient_accumulation, max_epochs=args.max_epochs, patience=args.patience,
                        )
                        progress["development"][key] = {"selected_epoch": epoch, "history": history, "max_epochs": args.max_epochs, "patience": args.patience}
                        progress_path.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                    development_epochs[task][arm].append(epoch)
                    histories[key] = history
                    print(json.dumps({"phase": "development", "fraction": fraction, "task": task, "arm": arm, "seed": seed, "selected_epoch": epoch}), flush=True)
        median_epochs = {task: {arm: int(statistics.median(values)) for arm, values in arm_rows.items()} for task, arm_rows in development_epochs.items()}
        selected_epochs[fraction] = {"per_seed": {task: dict(rows) for task, rows in development_epochs.items()}, "median": median_epochs}

        full_scaling = fit_activation_scaling(matrices["train"][indices])
        scaled = {split: apply_activation_scaling(matrices[split][indices] if split == "train" else matrices[split], full_scaling) for split in split_order}
        matrix_for_split = {split: matrices[split][indices] if split == "train" else matrices[split] for split in split_order}
        targets_for_split = {split: grades[split][indices] if split == "train" else grades[split] for split in split_order}
        patients_for_split = {split: patients[split][indices] if split == "train" else patients[split] for split in split_order}

        train_binary = (train_grades[indices] >= 2).astype(np.int64)
        logistic_input = np.log1p(matrices["train"][indices])
        logistic_binary = fit_probe(logistic_input, train_binary, multiclass=False, seed=SEEDS[0])
        logistic_grade = fit_probe(logistic_input, train_grades[indices], multiclass=True, seed=SEEDS[0])
        binary_predictions, grade_predictions = defaultdict(list), defaultdict(list)
        for split in split_order:
            source = np.log1p(matrix_for_split[split])
            binary_predictions["activation_logistic"].append(predict_probe(logistic_binary, source)[:, 1])
            grade_predictions["activation_logistic"].append(predict_probe(logistic_grade, source))

        replicate_metrics = defaultdict(list)
        shuffled_binary, shuffled_grade = defaultdict(list), defaultdict(list)
        mapping = torch.from_numpy(np.random.default_rng(SHUFFLE_SEED).permutation(SELECTED_FEATURES).astype(np.int64))
        for arm in NEURAL_ARMS:
            for seed in SEEDS:
                for task in ("binary", "five_grade"):
                    train_targets = train_binary if task == "binary" else train_grades[indices]
                    train_dataset = FeatureSetDataset(matrix_for_split["train"], scaled["train"], train_targets)
                    epochs = median_epochs[task][arm]
                    checkpoint_path = checkpoint_dir / f"fraction_{fraction}_{arm}_{task}_seed_{seed}.pt"
                    if checkpoint_path.is_file():
                        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                        if (payload.get("arm"), payload.get("task"), payload.get("seed"), payload.get("epochs")) != (arm, task, seed, epochs):
                            raise RuntimeError(f"incompatible resumable checkpoint: {checkpoint_path}")
                    else:
                        model = train_fixed(arm, task, train_dataset, banks, epochs=epochs, seed=seed, device=device, batch_size=args.batch_size, accumulation=args.gradient_accumulation)
                        payload = {
                            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                            "arm": arm, "task": task, "seed": seed, "epochs": epochs,
                            "model_specification": specification["model"],
                            "activation_scaling": {"mean": full_scaling.mean, "std": full_scaling.std},
                            "selected_feature_ids": selected_ids,
                            "specification_freeze_sha256": sha256_path(freeze_path),
                        }
                        torch.save(payload, checkpoint_path)
                        del model
                    reloaded = build_model(arm, 1 if task == "binary" else 5, banks).to(device)
                    reloaded.load_state_dict(payload["state_dict"])
                    for key, expected in payload["state_dict"].items():
                        if not torch.equal(expected.cpu(), reloaded.state_dict()[key].detach().cpu()):
                            raise RuntimeError(f"checkpoint reload mismatch: {checkpoint_path}:{key}")
                    for split in split_order:
                        split_targets = ((targets_for_split[split] >= 2).astype(np.int64) if task == "binary" else targets_for_split[split])
                        dataset = FeatureSetDataset(matrix_for_split[split], scaled[split], split_targets)
                        prediction = predict_model(reloaded, dataset, batch_size=args.batch_size, device=device, task=task)
                        (binary_predictions if task == "binary" else grade_predictions)[arm].append(prediction)
                        if arm in PRIMARY_ARMS and split in {"locked_test", "idrid"}:
                            shuffled = predict_model(reloaded, dataset, batch_size=args.batch_size, device=device, task=task, mapping=mapping)
                            (shuffled_binary if task == "binary" else shuffled_grade)[arm].append((split, seed, shuffled))
                    replicate_metrics[arm].append({"seed": seed, "task": task, "epochs": epochs, "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256_path(checkpoint_path), "parameters": parameter_count(reloaded)})
                    print(json.dumps({"phase": "final_fit", "fraction": fraction, "task": task, "arm": arm, "seed": seed, "epochs": epochs}), flush=True)
                    del reloaded
                    if device.type == "mps":
                        torch.mps.empty_cache()

        ensembles, arrays = {}, {}
        for arm in ALL_ARMS:
            if arm == "activation_logistic":
                binary_by_split = {split: binary_predictions[arm][index] for index, split in enumerate(split_order)}
                grade_by_split = {split: grade_predictions[arm][index] for index, split in enumerate(split_order)}
            else:
                binary_by_split = {
                    split: np.mean(np.stack([binary_predictions[arm][seed_index * 4 + split_index] for seed_index in range(len(SEEDS))]), axis=0)
                    for split_index, split in enumerate(split_order)
                }
                grade_by_split = {
                    split: np.mean(np.stack([grade_predictions[arm][seed_index * 4 + split_index] for seed_index in range(len(SEEDS))]), axis=0)
                    for split_index, split in enumerate(split_order)
                }
            threshold = fit_youden_threshold((targets_for_split["train"] >= 2).astype(np.int64), binary_by_split["train"])
            ensembles[arm] = {}
            for split in ("validation", "locked_test", "idrid"):
                binary_targets = (targets_for_split[split] >= 2).astype(np.int64)
                ensembles[arm][split] = {
                    "binary": {
                        **binary_metrics(binary_targets, binary_by_split[split], patients_for_split[split], args.bootstrap_resamples, SEEDS[0] + len(ensembles)),
                        **threshold_metrics(binary_targets, binary_by_split[split], threshold),
                    },
                    "five_grade": multiclass_metrics(targets_for_split[split], grade_by_split[split]),
                }
                arrays[f"{arm}_{split}_binary"] = binary_by_split[split].astype(np.float32)
                arrays[f"{arm}_{split}_grade"] = grade_by_split[split].astype(np.float32)
            arrays[f"{arm}_train_binary"] = binary_by_split["train"].astype(np.float32)

        shuffle_controls = {}
        for arm in PRIMARY_ARMS:
            shuffle_controls[arm] = {}
            for split in ("locked_test", "idrid"):
                binary_stack = [values for observed_split, _, values in shuffled_binary[arm] if observed_split == split]
                grade_stack = [values for observed_split, _, values in shuffled_grade[arm] if observed_split == split]
                if len(binary_stack) != len(SEEDS) or len(grade_stack) != len(SEEDS):
                    raise RuntimeError(f"incomplete shuffled mapping predictions: {arm}:{split}")
                binary_mean = np.mean(np.stack(binary_stack), axis=0)
                grade_mean = np.mean(np.stack(grade_stack), axis=0)
                binary_targets = (targets_for_split[split] >= 2).astype(np.int64)
                shuffle_controls[arm][split] = {
                    "binary_auroc": float(roc_auc_score(binary_targets, binary_mean)),
                    "binary_auroc_change_real_minus_shuffled": float(ensembles[arm][split]["binary"]["auroc"]["estimate"] - roc_auc_score(binary_targets, binary_mean)),
                    "five_grade": multiclass_metrics(targets_for_split[split], grade_mean),
                    "mapping_seed": SHUFFLE_SEED,
                }
                arrays[f"{arm}_{split}_binary_shuffled_mapping"] = binary_mean.astype(np.float32)
                arrays[f"{arm}_{split}_grade_shuffled_mapping"] = grade_mean.astype(np.float32)

        locked_binary_targets = (test_grades >= 2).astype(np.int64)
        primary_comparisons = {}
        for index, arm in enumerate(PRIMARY_ARMS):
            record = binary_difference(
                locked_binary_targets,
                arrays[f"{arm}_locked_test_binary"],
                arrays["activation_transformer_locked_test_binary"],
                test_patients,
                roc_auc_score,
                args.bootstrap_resamples,
                SEEDS[0] + 700 + index,
            )
            record["qwk"] = qwk_difference(
                test_grades,
                arrays[f"{arm}_locked_test_grade"],
                arrays["activation_transformer_locked_test_grade"],
                test_patients,
                args.bootstrap_resamples,
                SEEDS[0] + 800 + index,
            )
            primary_comparisons[f"{arm}_vs_activation_transformer"] = record
        holm_adjust(primary_comparisons)
        for record in primary_comparisons.values():
            record["statistically_useful"] = bool(record["ci_lower"] > 0 and record["holm_adjusted_p_value"] < 0.05)
            record["practically_meaningful"] = bool(record["difference"] >= 0.01)

        pairwise = {}
        for left_index, left in enumerate(PRIMARY_ARMS):
            for right in PRIMARY_ARMS[left_index + 1:]:
                pairwise[f"{left}_vs_{right}"] = binary_difference(
                    locked_binary_targets,
                    arrays[f"{left}_locked_test_binary"],
                    arrays[f"{right}_locked_test_binary"],
                    test_patients,
                    roc_auc_score,
                    args.bootstrap_resamples,
                    SEEDS[0] + 900 + len(pairwise),
                )

        external_consistency = {}
        for arm in PRIMARY_ARMS:
            internal_delta = primary_comparisons[f"{arm}_vs_activation_transformer"]["difference"]
            external_delta = ensembles[arm]["idrid"]["binary"]["auroc"]["estimate"] - ensembles["activation_transformer"]["idrid"]["binary"]["auroc"]["estimate"]
            external_consistency[arm] = {"locked_test_difference": internal_delta, "idrid_difference": external_delta, "same_direction": bool(np.sign(internal_delta) == np.sign(external_delta))}

        prediction_path = output / f"fraction_{fraction}_predictions.npz"
        np.savez_compressed(prediction_path, **arrays)
        result_fractions[fraction] = {
            "training_images": len(indices),
            "training_patients": len(set(train_patients[indices])),
            "inner_development_images": len(dev_indices),
            "activation_scaling": {"mean": full_scaling.mean.tolist(), "std": full_scaling.std.tolist()},
            "median_epochs": median_epochs,
            "ensembles": ensembles,
            "primary_comparisons": primary_comparisons,
            "pairwise_primary_arms": pairwise,
            "shuffled_mapping_controls": shuffle_controls,
            "external_consistency": external_consistency,
            "replicate_checkpoints": dict(replicate_metrics),
            "development_histories": histories,
            "prediction_artifact": str(prediction_path),
            "prediction_artifact_sha256": sha256_path(prediction_path),
        }

    result = {
        "schema_version": "1.0",
        "status": "complete",
        "experiment": "7C",
        "device": str(device),
        "encoder_frozen": True,
        "sae_frozen": True,
        "feature_card_and_decoder_banks_frozen": True,
        "dense_model_inputs": False,
        "rank_inputs": False,
        "positional_embeddings": False,
        "selected_feature_count": SELECTED_FEATURES,
        "selected_feature_ids": selected_ids.tolist(),
        "ten_percent_subset": {"images": len(low_indices), "patients": len(low_patients), "seed": SEEDS[0]},
        "specification": specification,
        "selected_epochs": selected_epochs,
        "fractions": result_fractions,
        "elapsed_seconds": time.time() - started,
        "specification_freeze_sha256": sha256_path(freeze_path),
        "scope_boundary": "Exploratory comparison on reused EyePACS locked-test and IDRiD sets; feature cards are same-model hypotheses, not confirmed lesions or clinical evidence.",
    }
    result_path = output / "experiment_7c_feature_view_transformers.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = []
    for fraction, fraction_record in result_fractions.items():
        for arm, split_rows in fraction_record["ensembles"].items():
            for split, metrics in split_rows.items():
                rows.append({
                    "fraction": fraction, "arm": arm, "split": split,
                    "auroc": metrics["binary"]["auroc"]["estimate"],
                    "auprc": metrics["binary"]["auprc"]["estimate"],
                    "qwk": metrics["five_grade"]["quadratic_weighted_kappa"],
                    "macro_auroc": metrics["five_grade"]["macro_auroc"],
                    "ordinal_mae": metrics["five_grade"]["ordinal_mae"],
                })
    csv_path = output / "experiment_7c_results_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    markdown = ["# Experiment 7C results", "", "Exploratory reused-test evaluation; feature-card text is not clinician validated.", "", "| Labels | Arm | Split | AUROC | AUPRC | QWK | Macro-AUROC | Ordinal MAE |", "|---:|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        markdown.append(f"| {row['fraction']} | {row['arm']} | {row['split']} | {row['auroc']:.4f} | {row['auprc']:.4f} | {row['qwk']:.4f} | {row['macro_auroc']:.4f} | {row['ordinal_mae']:.4f} |")
    (output / "experiment_7c_results_table.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "result": str(result_path), "elapsed_seconds": result["elapsed_seconds"]}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment 7C three SAE feature-view Transformers")
    parser.add_argument("--stage", choices=("prepare", "run", "smoke"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cards")
    parser.add_argument("--text-model-cache")
    parser.add_argument("--banks")
    parser.add_argument("--banks-metadata")
    parser.add_argument("--train-run")
    parser.add_argument("--validation-run")
    parser.add_argument("--internal-test-run")
    parser.add_argument("--idrid-run")
    parser.add_argument("--idrid-manifest")
    parser.add_argument("--labels")
    parser.add_argument("--active-trace")
    parser.add_argument("--normalization")
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    args = parser.parse_args()
    if args.batch_size < 1 or args.gradient_accumulation < 1:
        parser.error("batch size and gradient accumulation must be positive")
    if args.bootstrap_resamples < 100:
        parser.error("bootstrap-resamples must be at least 100")
    output = require_external(Path(args.output), "output")
    output.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    if args.stage == "smoke":
        record = smoke(device)
        (output / "smoke.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(record, indent=2, sort_keys=True))
        return 0
    if args.stage == "prepare":
        if not all((args.cards, args.checkpoint, args.text_model_cache)):
            parser.error("prepare requires --cards, --checkpoint, and --text-model-cache")
        prepare_representations(args, output)
        return 0
    run_experiment(args, output, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
