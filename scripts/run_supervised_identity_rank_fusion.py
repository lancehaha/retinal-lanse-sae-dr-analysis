from __future__ import annotations

"""Experiment 7: supervised dense/SAE fusion with identity and rank embeddings."""

import argparse
import copy
import csv
import hashlib
import json
import math
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
from sklearn.metrics import average_precision_score, roc_auc_score
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
TRACE_WIDTH = 1024
DICTIONARY_SIZE = 16384
GROUP_SIZE = 16
GROUPS = 64
MODEL_DIM = 128
NEURAL_ARMS = (
    "value_mlp",
    "identity_only_transformer",
    "position_only_transformer",
    "identity_position_transformer",
)
ALL_ARMS = ("dense_logistic", "dense_value_logistic", *NEURAL_ARMS)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def validate_trace(ids: np.ndarray, values: np.ndarray, expected_rows: int) -> None:
    if ids.shape != (expected_rows, TRACE_WIDTH) or values.shape != ids.shape:
        raise RuntimeError(f"expected aligned [N,{TRACE_WIDTH}] trace; got {ids.shape} and {values.shape}")
    if np.any(ids < -1) or np.any(ids >= DICTIONARY_SIZE):
        raise RuntimeError("feature IDs are outside [-1,16383]")
    if np.any(values < 0) or not np.all(np.isfinite(values)):
        raise RuntimeError("activation values must be finite and nonnegative")
    if np.any((ids < 0) != (values <= 0)):
        raise RuntimeError("padding IDs and zero activation values are not aligned")
    for row in range(min(expected_rows, 256)):
        valid = ids[row] >= 0
        row_values = values[row, valid]
        row_ids = ids[row, valid]
        expected = np.lexsort((row_ids, -row_values))
        if not np.array_equal(expected, np.arange(len(row_values))):
            raise RuntimeError("activation trace is not sorted by value then feature ID")


@dataclass
class Scaling:
    dense_mean: np.ndarray
    dense_std: np.ndarray
    active_mean: float
    active_std: float


def fit_scaling(dense: np.ndarray, ids: np.ndarray, values: np.ndarray) -> Scaling:
    dense_mean = dense.mean(axis=0, keepdims=True).astype(np.float32)
    dense_std = dense.std(axis=0, keepdims=True).astype(np.float32)
    dense_std[dense_std < 1e-6] = 1.0
    valid = ids >= 0
    logged = np.log1p(values[valid]).astype(np.float64)
    active_mean = float(logged.mean()) if len(logged) else 0.0
    active_std = float(logged.std()) if len(logged) else 1.0
    if active_std < 1e-6:
        active_std = 1.0
    return Scaling(dense_mean, dense_std, active_mean, active_std)


def apply_scaling(dense: np.ndarray, ids: np.ndarray, values: np.ndarray, scaling: Scaling) -> tuple[np.ndarray, np.ndarray]:
    dense_scaled = ((dense - scaling.dense_mean) / scaling.dense_std).astype(np.float32)
    logged = np.log1p(values).astype(np.float32)
    values_scaled = ((logged - scaling.active_mean) / scaling.active_std).astype(np.float32)
    values_scaled[ids < 0] = 0.0
    return dense_scaled, values_scaled


class FusionDataset(Dataset):
    def __init__(self, dense: np.ndarray, ids: np.ndarray, values: np.ndarray, targets: np.ndarray):
        self.dense = torch.from_numpy(np.asarray(dense, dtype=np.float32))
        self.ids = torch.from_numpy(np.asarray(ids, dtype=np.int64))
        self.values = torch.from_numpy(np.asarray(values, dtype=np.float32))
        self.targets = torch.from_numpy(np.asarray(targets))

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        return self.dense[index], self.ids[index], self.values[index], self.targets[index]


class ValueOnlyMLP(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2048, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim),
        )

    def forward(self, dense: torch.Tensor, feature_ids: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        del feature_ids
        return self.network(torch.cat([dense, values], dim=1))


class IdentityRankTransformer(nn.Module):
    """A 66-token fusion model; feature IDs enter only through nn.Embedding."""

    def __init__(self, output_dim: int, *, use_identity: bool, use_position: bool):
        super().__init__()
        self.use_identity = use_identity
        self.use_position = use_position
        self.padding_id = DICTIONARY_SIZE
        self.dense_projection = nn.Linear(1024, MODEL_DIM)
        self.value_projection = nn.Sequential(nn.Linear(1, MODEL_DIM), nn.GELU(), nn.Linear(MODEL_DIM, MODEL_DIM))
        self.feature_identity_embedding = nn.Embedding(DICTIONARY_SIZE + 1, MODEL_DIM, padding_idx=self.padding_id) if use_identity else None
        self.within_group_rank_embedding = nn.Embedding(GROUP_SIZE, MODEL_DIM) if use_position else None
        self.group_rank_embedding = nn.Embedding(GROUPS, MODEL_DIM) if use_position else None
        self.modality_embedding = nn.Embedding(3, MODEL_DIM)
        self.pool_query = nn.Parameter(torch.empty(MODEL_DIM))
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
        nn.init.normal_(self.pool_query, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, dense: torch.Tensor, feature_ids: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        batch = dense.shape[0]
        valid = (feature_ids >= 0) & (values != 0)
        safe_ids = torch.where(valid, feature_ids, torch.full_like(feature_ids, self.padding_id))
        entries = self.value_projection(values.unsqueeze(-1))
        if self.feature_identity_embedding is not None:
            entries = entries + self.feature_identity_embedding(safe_ids)
        if self.within_group_rank_embedding is not None:
            ranks = torch.arange(GROUP_SIZE, device=dense.device).repeat(GROUPS)
            entries = entries + self.within_group_rank_embedding(ranks)[None, :, :]
        entries = entries.reshape(batch, GROUPS, GROUP_SIZE, MODEL_DIM)
        grouped_valid = valid.reshape(batch, GROUPS, GROUP_SIZE)
        scores = torch.einsum("bgkd,d->bgk", entries, self.pool_query) / math.sqrt(MODEL_DIM)
        scores = scores.masked_fill(~grouped_valid, -1e9)
        weights = torch.softmax(scores, dim=-1) * grouped_valid.float()
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        group_tokens = torch.sum(entries * weights.unsqueeze(-1), dim=2)
        group_present = grouped_valid.any(dim=2)
        group_tokens = group_tokens * group_present.unsqueeze(-1)
        if self.group_rank_embedding is not None:
            group_ranks = torch.arange(GROUPS, device=dense.device)
            group_tokens = group_tokens + self.group_rank_embedding(group_ranks)[None, :, :] * group_present.unsqueeze(-1)

        cls = self.cls_token.expand(batch, -1, -1) + self.modality_embedding(torch.zeros(1, dtype=torch.long, device=dense.device))[None, :, :]
        dense_token = self.dense_projection(dense).unsqueeze(1) + self.modality_embedding(torch.ones(1, dtype=torch.long, device=dense.device))[None, :, :]
        group_tokens = group_tokens + self.modality_embedding(torch.full((1,), 2, dtype=torch.long, device=dense.device))[None, :, :]
        sequence = torch.cat([cls, dense_token, group_tokens], dim=1)
        padding_mask = torch.cat([torch.zeros((batch, 2), dtype=torch.bool, device=dense.device), ~group_present], dim=1)
        encoded = self.encoder(sequence, src_key_padding_mask=padding_mask)
        return self.head(encoded[:, 0])


def build_model(arm: str, output_dim: int) -> nn.Module:
    if arm == "value_mlp":
        return ValueOnlyMLP(output_dim)
    if arm == "identity_only_transformer":
        return IdentityRankTransformer(output_dim, use_identity=True, use_position=False)
    if arm == "position_only_transformer":
        return IdentityRankTransformer(output_dim, use_identity=False, use_position=True)
    if arm == "identity_position_transformer":
        return IdentityRankTransformer(output_dim, use_identity=True, use_position=True)
    raise ValueError(f"unknown neural arm: {arm}")


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def make_loader(dataset: FusionDataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, generator=generator)


@torch.inference_mode()
def predict_model(model: nn.Module, dataset: FusionDataset, *, batch_size: int, device: torch.device, task: str) -> np.ndarray:
    model.eval()
    rows = []
    for dense, ids, values, _ in make_loader(dataset, batch_size, False, 0):
        logits = model(dense.to(device), ids.to(device), values.to(device))
        if task == "binary":
            rows.append(torch.sigmoid(logits.squeeze(1)).cpu().numpy())
        else:
            rows.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(rows, axis=0)


def monitor_score(task: str, targets: np.ndarray, probabilities: np.ndarray) -> float:
    if task == "binary":
        return float(roc_auc_score(targets, probabilities))
    one_hot = np.eye(5, dtype=np.int64)[targets]
    return float(roc_auc_score(one_hot, probabilities, average="macro", multi_class="ovr"))


def train_development(
    arm: str,
    task: str,
    train_dataset: FusionDataset,
    dev_dataset: FusionDataset,
    *,
    seed: int,
    device: torch.device,
    batch_size: int,
    max_epochs: int,
    patience: int,
) -> tuple[int, list[dict]]:
    set_deterministic(seed)
    output_dim = 1 if task == "binary" else 5
    model = build_model(arm, output_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    targets = train_dataset.targets.numpy()
    if task == "binary":
        positive = max(int(targets.sum()), 1)
        negative = max(int(len(targets) - positive), 1)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negative / positive], dtype=torch.float32, device=device))
    else:
        counts = np.bincount(targets.astype(np.int64), minlength=5)
        weights = len(targets) / (5 * np.maximum(counts, 1))
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    best_score = -np.inf
    best_epoch = 1
    best_state = None
    history = []
    stale = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for dense, ids, values, batch_targets in make_loader(train_dataset, batch_size, True, seed + epoch):
            optimizer.zero_grad(set_to_none=True)
            logits = model(dense.to(device), ids.to(device), values.to(device))
            if task == "binary":
                loss = loss_fn(logits.squeeze(1), batch_targets.float().to(device))
            else:
                loss = loss_fn(logits, batch_targets.long().to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_targets)
            seen += len(batch_targets)
        dev_probabilities = predict_model(model, dev_dataset, batch_size=batch_size, device=device, task=task)
        score = monitor_score(task, dev_dataset.targets.numpy(), dev_probabilities)
        history.append({"epoch": epoch, "train_loss": total_loss / max(seen, 1), "development_metric": score})
        if score > best_score + 1e-6:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("development training produced no checkpoint")
    return best_epoch, history


def train_fixed_epochs(
    arm: str,
    task: str,
    dataset: FusionDataset,
    *,
    epochs: int,
    seed: int,
    device: torch.device,
    batch_size: int,
) -> nn.Module:
    set_deterministic(seed)
    model = build_model(arm, 1 if task == "binary" else 5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    targets = dataset.targets.numpy()
    if task == "binary":
        positive = max(int(targets.sum()), 1)
        negative = max(int(len(targets) - positive), 1)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negative / positive], dtype=torch.float32, device=device))
    else:
        counts = np.bincount(targets.astype(np.int64), minlength=5)
        weights = len(targets) / (5 * np.maximum(counts, 1))
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    for epoch in range(1, epochs + 1):
        model.train()
        for dense, ids, values, batch_targets in make_loader(dataset, batch_size, True, seed + epoch):
            optimizer.zero_grad(set_to_none=True)
            logits = model(dense.to(device), ids.to(device), values.to(device))
            loss = loss_fn(logits.squeeze(1), batch_targets.float().to(device)) if task == "binary" else loss_fn(logits, batch_targets.long().to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model


def holm_adjust(records: dict[str, dict]) -> None:
    ordered = sorted(records, key=lambda key: records[key]["p_value"])
    running = 0.0
    total = len(ordered)
    for rank, key in enumerate(ordered):
        adjusted = min(1.0, records[key]["p_value"] * (total - rank))
        running = max(running, adjusted)
        records[key]["holm_adjusted_p_value"] = running


def run_smoke(device: torch.device) -> dict:
    rng = np.random.default_rng(SEEDS[0])
    n = 32
    dense = rng.normal(size=(n, 1024)).astype(np.float32)
    ids = np.tile(np.arange(1024, dtype=np.int64), (n, 1))
    values = np.sort(rng.uniform(0.01, 2.0, size=(n, 1024)).astype(np.float32), axis=1)[:, ::-1].copy()
    ids[:, 900:] = -1
    values[:, 900:] = 0
    targets = np.asarray([0, 1] * (n // 2), dtype=np.int64)
    scaling = fit_scaling(dense[:24], ids[:24], values[:24])
    dense_scaled, values_scaled = apply_scaling(dense, ids, values, scaling)
    train = FusionDataset(dense_scaled[:24], ids[:24], values_scaled[:24], targets[:24])
    dev = FusionDataset(dense_scaled[24:], ids[24:], values_scaled[24:], targets[24:])
    epoch, history = train_development("identity_position_transformer", "binary", train, dev, seed=SEEDS[0], device=device, batch_size=8, max_epochs=1, patience=1)
    model = build_model("identity_position_transformer", 1).to(device).eval()
    with torch.inference_mode():
        original = model(train.dense[:2].to(device), train.ids[:2].to(device), train.values[:2].to(device))
        changed = train.values[:2].clone()
        changed[train.ids[:2] < 0] = 999.0
        masked = model(train.dense[:2].to(device), train.ids[:2].to(device), changed.to(device))
    if not torch.allclose(original, masked, atol=1e-5):
        raise RuntimeError("padding-mask invariance failed")
    counts = {arm: parameter_count(build_model(arm, 1)) for arm in NEURAL_ARMS}
    return {"status": "smoke_passed", "device": str(device), "epoch": epoch, "history": history, "sequence_tokens": 66, "padding_mask_invariant": True, "feature_ids_are_categorical_embeddings": True, "parameter_counts": counts}


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment 7 supervised identity/rank fusion")
    parser.add_argument("--train-run")
    parser.add_argument("--validation-run")
    parser.add_argument("--internal-test-run")
    parser.add_argument("--idrid-run")
    parser.add_argument("--idrid-manifest")
    parser.add_argument("--labels")
    parser.add_argument("--active-trace")
    parser.add_argument("--normalization")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--arms", nargs="+", choices=NEURAL_ARMS, default=list(NEURAL_ARMS))
    parser.add_argument("--fractions", nargs="+", choices=FRACTIONS, default=list(FRACTIONS))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = require_external(Path(args.output), "output")
    output.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    if args.smoke:
        smoke = run_smoke(device)
        (output / "smoke.json").write_text(json.dumps(smoke, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(smoke, indent=2, sort_keys=True))
        return 0
    required = {name: getattr(args, name.replace(" ", "_")) for name in ("train run", "validation run", "internal test run", "idrid run", "idrid manifest", "labels", "active trace", "normalization", "checkpoint")}
    missing = [name for name, value in required.items() if not value]
    if missing:
        parser.error("missing full-run inputs: " + ", ".join(missing))
    roots = {name: require_external(Path(value), name) for name, value in required.items()}
    if args.bootstrap_resamples < 100:
        parser.error("bootstrap-resamples must be at least 100")
    _, checkpoint_payload = load_checkpoint(roots["checkpoint"])
    if checkpoint_payload.get("model_config") != {"input_dim": 1024, "dictionary_size": 16384, "k": 1024}:
        raise RuntimeError("Experiment 7 requires the frozen K=1024 seed-4106 SAE")

    labels = load_eyepacs_labels(roots["labels"])
    idrid_manifest = load_idrid_manifest(roots["idrid manifest"])
    train_run = load_embedding_run(roots["train run"], expected_split="train", verify_sha256=True)
    val_run = load_embedding_run(roots["validation run"], expected_split="validation", verify_sha256=True)
    test_run = load_embedding_run(roots["internal test run"], expected_split="internal_test", verify_sha256=True)
    idrid_run = load_embedding_run(roots["idrid run"], expected_split="external", verify_sha256=True)
    train_dense, train_grades, train_patients, train_paths = collect_eyepacs(train_run, labels)
    val_dense, val_grades, val_patients, val_paths = collect_eyepacs(val_run, labels)
    test_dense, test_grades, test_patients, test_paths = collect_eyepacs(test_run, labels)
    idrid_dense, idrid_grades, idrid_patients, idrid_paths = collect_idrid(idrid_run, idrid_manifest)
    for left, right in ((train_patients, val_patients), (train_patients, test_patients), (val_patients, test_patients)):
        if set(left) & set(right):
            raise RuntimeError("EyePACS patient overlap detected")
    trace = np.load(roots["active trace"])
    trace_arrays = {
        "train": (trace["train_feature_ids"], trace["train_values"]),
        "validation": (trace["validation_feature_ids"], trace["validation_values"]),
        "locked_test": (trace["test_feature_ids"], trace["test_values"]),
        "idrid": (trace["idrid_feature_ids"], trace["idrid_values"]),
    }
    for split, expected in (("train", len(train_dense)), ("validation", len(val_dense)), ("locked_test", len(test_dense)), ("idrid", len(idrid_dense))):
        validate_trace(*trace_arrays[split], expected)

    specification = {
        "schema_version": "1.0",
        "experiment": 7,
        "model": {"tokens": 66, "model_dim": 128, "layers": 4, "heads": 8, "ffn": 512, "dropout": 0.1, "group_size": 16, "groups": 64},
        "training": {"optimizer": "AdamW", "learning_rate": 1e-4, "weight_decay": 1e-4, "batch_size": args.batch_size, "max_epochs": args.max_epochs, "patience": args.patience, "gradient_clip": 1.0, "seeds": list(SEEDS)},
        "arms": ["dense_logistic", "dense_value_logistic", *args.arms],
        "fractions": args.fractions,
        "locked_test_used_for_selection": False,
    }
    freeze_path = output / "specification_freeze.json"
    freeze_path.write_text(json.dumps(specification, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    dense_by_split = {"train": train_dense, "validation": val_dense, "locked_test": test_dense, "idrid": idrid_dense}
    grades_by_split = {"train": train_grades, "validation": val_grades, "locked_test": test_grades, "idrid": idrid_grades}
    patients_by_split = {"train": train_patients, "validation": val_patients, "locked_test": test_patients, "idrid": idrid_patients}
    path_by_split = {"train": train_paths, "validation": val_paths, "locked_test": test_paths, "idrid": idrid_paths}
    fraction_indices = {"1.00": np.arange(len(train_grades), dtype=np.int64)}
    low_indices, low_patients = sample_patient_indices(train_patients, train_grades, 0.10, SEEDS[0])
    fraction_indices["0.10"] = low_indices
    selected_epochs: dict[str, dict] = defaultdict(dict)
    result_fractions = {}
    prediction_artifacts = {}
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output / "training_progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.is_file() else {"schema_version": "1.0", "development": {}}
    started = time.time()

    for fraction in args.fractions:
        indices = fraction_indices[fraction]
        selected_patients = train_patients[indices]
        dev_local = stable_dev_mask(selected_patients, seed=SEEDS[0], fraction=0.10)
        if not np.any(dev_local) or not np.any(~dev_local):
            raise RuntimeError("inner patient development split is empty")
        fit_indices = indices[~dev_local]
        dev_indices = indices[dev_local]
        fit_scaler = fit_scaling(train_dense[fit_indices], trace_arrays["train"][0][fit_indices], trace_arrays["train"][1][fit_indices])
        fit_dense_scaled, fit_values_scaled = apply_scaling(train_dense[fit_indices], trace_arrays["train"][0][fit_indices], trace_arrays["train"][1][fit_indices], fit_scaler)
        dev_dense_scaled, dev_values_scaled = apply_scaling(train_dense[dev_indices], trace_arrays["train"][0][dev_indices], trace_arrays["train"][1][dev_indices], fit_scaler)
        development_epochs = defaultdict(lambda: defaultdict(list))
        histories = defaultdict(dict)
        for task in ("binary", "five_grade"):
            fit_targets = (train_grades[fit_indices] >= 2).astype(np.int64) if task == "binary" else train_grades[fit_indices]
            dev_targets = (train_grades[dev_indices] >= 2).astype(np.int64) if task == "binary" else train_grades[dev_indices]
            fit_dataset = FusionDataset(fit_dense_scaled, trace_arrays["train"][0][fit_indices], fit_values_scaled, fit_targets)
            dev_dataset = FusionDataset(dev_dense_scaled, trace_arrays["train"][0][dev_indices], dev_values_scaled, dev_targets)
            for arm in args.arms:
                for seed in SEEDS:
                    progress_key = f"{fraction}:{task}:{arm}:{seed}"
                    prior = progress["development"].get(progress_key)
                    if prior and prior.get("max_epochs") == args.max_epochs and prior.get("patience") == args.patience:
                        epoch = int(prior["selected_epoch"])
                        history = prior["history"]
                    else:
                        epoch, history = train_development(arm, "binary" if task == "binary" else "five_grade", fit_dataset, dev_dataset, seed=seed, device=device, batch_size=args.batch_size, max_epochs=args.max_epochs, patience=args.patience)
                        progress["development"][progress_key] = {"selected_epoch": epoch, "history": history, "max_epochs": args.max_epochs, "patience": args.patience}
                        progress_path.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                        print(json.dumps({"phase": "development", "fraction": fraction, "task": task, "arm": arm, "seed": seed, "selected_epoch": epoch}), flush=True)
                    development_epochs[task][arm].append(epoch)
                    histories[f"{task}:{arm}:{seed}"] = history
        median_epochs = {task: {arm: int(statistics.median(values)) for arm, values in arms.items()} for task, arms in development_epochs.items()}
        selected_epochs[fraction] = {"per_seed": {task: dict(arms) for task, arms in development_epochs.items()}, "median": median_epochs}

        full_scaler = fit_scaling(train_dense[indices], trace_arrays["train"][0][indices], trace_arrays["train"][1][indices])
        scaled = {}
        for split in dense_by_split:
            split_indices = indices if split == "train" else np.arange(len(dense_by_split[split]))
            dense_source = dense_by_split[split][split_indices] if split == "train" else dense_by_split[split]
            ids_source = trace_arrays[split][0][split_indices] if split == "train" else trace_arrays[split][0]
            values_source = trace_arrays[split][1][split_indices] if split == "train" else trace_arrays[split][1]
            scaled[split] = (*apply_scaling(dense_source, ids_source, values_source, full_scaler), ids_source)

        binary_target = (train_grades[indices] >= 2).astype(np.int64)
        dense_binary = fit_probe(train_dense[indices], binary_target, multiclass=False, seed=SEEDS[0])
        dense_multi = fit_probe(train_dense[indices], train_grades[indices], multiclass=True, seed=SEEDS[0])
        value_train = np.concatenate([train_dense[indices], trace_arrays["train"][1][indices]], axis=1)
        value_binary = fit_probe(value_train, binary_target, multiclass=False, seed=SEEDS[0])
        value_multi = fit_probe(value_train, train_grades[indices], multiclass=True, seed=SEEDS[0])
        binary_predictions = defaultdict(list)
        grade_predictions = defaultdict(list)
        for split in ("train", "validation", "locked_test", "idrid"):
            dense_source = train_dense[indices] if split == "train" else dense_by_split[split]
            value_source = trace_arrays["train"][1][indices] if split == "train" else trace_arrays[split][1]
            binary_predictions["dense_logistic"].append(predict_probe(dense_binary, dense_source)[:, 1])
            grade_predictions["dense_logistic"].append(predict_probe(dense_multi, dense_source))
            fused_source = np.concatenate([dense_source, value_source], axis=1)
            binary_predictions["dense_value_logistic"].append(predict_probe(value_binary, fused_source)[:, 1])
            grade_predictions["dense_value_logistic"].append(predict_probe(value_multi, fused_source))

        replicate_metrics = defaultdict(list)
        split_order = ("train", "validation", "locked_test", "idrid")
        for arm in args.arms:
            for seed in SEEDS:
                for task in ("binary", "five_grade"):
                    targets = binary_target if task == "binary" else train_grades[indices]
                    train_dataset = FusionDataset(scaled["train"][0], scaled["train"][2], scaled["train"][1], targets)
                    epochs = median_epochs[task][arm]
                    state_path = checkpoint_dir / f"fraction_{fraction}_{arm}_{task}_seed_{seed}.pt"
                    if state_path.is_file():
                        payload = torch.load(state_path, map_location="cpu", weights_only=False)
                        expected = (payload.get("arm"), payload.get("task"), payload.get("seed"), payload.get("epochs"))
                        if expected != (arm, task, seed, epochs):
                            raise RuntimeError(f"incompatible resumable checkpoint: {state_path}")
                    else:
                        model = train_fixed_epochs(arm, "binary" if task == "binary" else "five_grade", train_dataset, epochs=epochs, seed=seed, device=device, batch_size=args.batch_size)
                        payload = {"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "arm": arm, "task": task, "seed": seed, "epochs": epochs, "model_specification": specification["model"], "scaling": {"dense_mean": full_scaler.dense_mean, "dense_std": full_scaler.dense_std, "active_mean": full_scaler.active_mean, "active_std": full_scaler.active_std}}
                        torch.save(payload, state_path)
                        del model
                    reloaded = build_model(arm, 1 if task == "binary" else 5).to(device)
                    reloaded.load_state_dict(torch.load(state_path, map_location=device, weights_only=False)["state_dict"])
                    for key, expected_value in payload["state_dict"].items():
                        if not torch.equal(expected_value.cpu(), reloaded.state_dict()[key].detach().cpu()):
                            raise RuntimeError(f"checkpoint reload equivalence failed for {state_path}:{key}")
                    for split in split_order:
                        split_targets = ((grades_by_split[split] >= 2).astype(np.int64) if task == "binary" else grades_by_split[split]) if split != "train" else targets
                        dataset = FusionDataset(scaled[split][0], scaled[split][2], scaled[split][1], split_targets)
                        prediction = predict_model(reloaded, dataset, batch_size=args.batch_size, device=device, task="binary" if task == "binary" else "five_grade")
                        (binary_predictions if task == "binary" else grade_predictions)[arm].append(prediction)
                    replicate_metrics[arm].append({"seed": seed, "task": task, "epochs": epochs, "checkpoint": str(state_path), "checkpoint_sha256": sha256_path(state_path), "parameters": parameter_count(reloaded)})
                    print(json.dumps({"phase": "final_fit", "fraction": fraction, "task": task, "arm": arm, "seed": seed, "epochs": epochs, "checkpoint": str(state_path)}), flush=True)
                    del reloaded
                    if device.type == "mps":
                        torch.mps.empty_cache()

        # Logistic arms have four arrays; neural arms have five seeds x four split arrays.
        ensembles = {}
        arrays = {}
        for arm in ("dense_logistic", "dense_value_logistic", *args.arms):
            if arm in {"dense_logistic", "dense_value_logistic"}:
                binary_by_split = {split: binary_predictions[arm][i] for i, split in enumerate(split_order)}
                grade_by_split = {split: grade_predictions[arm][i] for i, split in enumerate(split_order)}
            else:
                binary_by_split = {split: np.mean(np.stack([binary_predictions[arm][seed_index * 4 + split_index] for seed_index in range(5)]), axis=0) for split_index, split in enumerate(split_order)}
                grade_by_split = {split: np.mean(np.stack([grade_predictions[arm][seed_index * 4 + split_index] for seed_index in range(5)]), axis=0) for split_index, split in enumerate(split_order)}
            threshold = fit_youden_threshold(binary_target, binary_by_split["train"])
            ensembles[arm] = {}
            for split in ("validation", "locked_test", "idrid"):
                binary_y = (grades_by_split[split] >= 2).astype(np.int64)
                ensembles[arm][split] = {
                    "binary": {**binary_metrics(binary_y, binary_by_split[split], patients_by_split[split], args.bootstrap_resamples, SEEDS[0] + len(ensembles[arm])), **threshold_metrics(binary_y, binary_by_split[split], threshold)},
                    "five_grade": multiclass_metrics(grades_by_split[split], grade_by_split[split]),
                }
                arrays[f"{arm}_{split}_binary"] = binary_by_split[split].astype(np.float32)
                arrays[f"{arm}_{split}_grade"] = grade_by_split[split].astype(np.float32)
            arrays[f"{arm}_train_binary"] = binary_by_split["train"].astype(np.float32)

        comparisons = {}
        full = "identity_position_transformer"
        if full in args.arms:
            test_targets = (test_grades >= 2).astype(np.int64)
            for reference in ("dense_logistic", "value_mlp", "identity_only_transformer", "position_only_transformer"):
                if reference not in ensembles:
                    continue
                name = f"{full}_vs_{reference}"
                comparisons[name] = binary_difference(test_targets, arrays[f"{full}_locked_test_binary"], arrays[f"{reference}_locked_test_binary"], test_patients, roc_auc_score, args.bootstrap_resamples, SEEDS[0] + 900)
                comparisons[name]["qwk"] = qwk_difference(test_grades, arrays[f"{full}_locked_test_grade"], arrays[f"{reference}_locked_test_grade"], test_patients, args.bootstrap_resamples, SEEDS[0] + 901)
            holm_adjust(comparisons)
        prediction_path = output / f"fraction_{fraction}_predictions.npz"
        np.savez_compressed(prediction_path, **arrays)
        result_fractions[fraction] = {"training_images": len(indices), "training_patients": len(set(train_patients[indices])), "inner_development_images": len(dev_indices), "median_epochs": median_epochs, "ensembles": ensembles, "comparisons": comparisons, "replicate_checkpoints": replicate_metrics, "prediction_artifact": str(prediction_path), "development_histories": histories}

    result = {
        "schema_version": "1.0",
        "status": "complete" if set(args.arms) == set(NEURAL_ARMS) and set(args.fractions) == set(FRACTIONS) else "complete_requested_subset",
        "experiment": 7,
        "device": str(device),
        "encoder_frozen": True,
        "sae_frozen": True,
        "sae_checkpoint_seed": 4106,
        "feature_ids_enter_as_scalar_values": False,
        "cohorts": {split: {"images": len(grades_by_split[split]), "units": len(set(patients_by_split[split]))} for split in grades_by_split},
        "ten_percent_subset": {"images": len(low_indices), "patients": len(low_patients), "seed": SEEDS[0]},
        "specification": specification,
        "selected_epochs": selected_epochs,
        "fractions": result_fractions,
        "success_criteria": {"full_vs_dense_minimum_auroc_gain": 0.03, "position_minimum_auroc_gain": 0.01, "identity_requires_adjusted_ci_above_zero": True},
        "elapsed_seconds": time.time() - started,
        "provenance": {name.replace(" ", "_") + "_sha256": sha256_path(path if path.is_file() else path / "extraction_evidence.json") for name, path in roots.items()},
        "specification_freeze_sha256": sha256_path(freeze_path),
        "scope_boundary": "This is supervised research prediction from frozen representations, not lesion confirmation or clinical deployment validation.",
    }
    result_path = output / "experiment_7_supervised_identity_rank_fusion.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    table_path = output / "experiment_7_results_table.csv"
    table_rows = []
    for fraction, fraction_row in result_fractions.items():
        for arm, split_rows in fraction_row["ensembles"].items():
            for split, record in split_rows.items():
                table_rows.append({"fraction": fraction, "arm": arm, "split": split, "auroc": record["binary"]["auroc"]["estimate"], "auprc": record["binary"]["auprc"]["estimate"], "qwk": record["five_grade"]["quadratic_weighted_kappa"], "macro_auroc": record["five_grade"]["macro_auroc"], "ordinal_mae": record["five_grade"]["ordinal_mae"]})
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)
    md = ["# Experiment 7 results", "", "| Labels | Arm | Split | AUROC | AUPRC | QWK | Macro-AUROC | Ordinal MAE |", "|---:|---|---|---:|---:|---:|---:|---:|"]
    for row in table_rows:
        md.append(f"| {row['fraction']} | {row['arm']} | {row['split']} | {row['auroc']:.4f} | {row['auprc']:.4f} | {row['qwk']:.4f} | {row['macro_auroc']:.4f} | {row['ordinal_mae']:.4f} |")
    (output / "experiment_7_results_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "result": str(result_path), "elapsed_seconds": result["elapsed_seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
