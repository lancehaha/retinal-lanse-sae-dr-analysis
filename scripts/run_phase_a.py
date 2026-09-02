from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image

from retina_lanse.autoresearch.registry import initialize_registry
from retina_lanse.config import load_config
from retina_lanse.data.preprocessing import preprocess_global
from retina_lanse.data.synthetic import generate_dataset
from retina_lanse.encoders import DeterministicSmokeEncoder
from retina_lanse.evaluation.metrics import evaluate_embeddings, sparse_coordinate_baseline
from retina_lanse.sae.checkpoints import load_checkpoint, save_checkpoint
from retina_lanse.sae.model import TopKSAE
from retina_lanse.sae.trainer import train_sae, write_curves
from retina_lanse.utils import set_deterministic, sha256_file, stable_hash, write_json


def run(config_path: str) -> dict:
    config = load_config(config_path)
    seed = int(config["seed"])
    set_deterministic(seed)
    data_cfg, enc_cfg, sae_cfg, paths = config["data"], config["encoder"], config["sae"], config["artifacts"]
    records = generate_dataset(
        data_cfg["root"], int(data_cfg["num_patients"]), int(data_cfg["images_per_patient"]),
        int(data_cfg["image_size"]), tuple(data_cfg["split_fractions"]), seed,
    )
    split_hash = stable_hash({"patient_id": r["patient_id"], "split": r["split"]} for r in records)
    encoder = DeterministicSmokeEncoder(int(enc_cfg["embedding_dim"]), int(enc_cfg["pool_size"]), int(enc_cfg["seed"]))
    by_split: dict[str, list[torch.Tensor]] = {"train": [], "validation": [], "test": []}
    embedding_records = []
    with torch.no_grad():
        for record in records:
            with Image.open(record["path"]) as image:
                tensor = preprocess_global(image, int(enc_cfg["input_size"])).unsqueeze(0)
            embedding = encoder(tensor).squeeze(0).cpu()
            by_split[record["split"]].append(embedding)
            embedding_records.append({
                **record,
                "encoder_name": enc_cfg["name"],
                "encoder_checkpoint_hash": encoder.checkpoint_hash,
                "preprocessing_version": data_cfg["preprocessing_version"],
            })
    raw_by_split = {key: torch.stack(value) for key, value in by_split.items()}
    normalization_mean = raw_by_split["train"].mean(dim=0)
    normalization_std = raw_by_split["train"].std(dim=0, unbiased=False).clamp_min(1e-3)
    normalized_by_split = {
        key: (value - normalization_mean) / normalization_std for key, value in raw_by_split.items()
    }
    embedding_payload = {
        "records": embedding_records,
        "embeddings": normalized_by_split,
        "embedding_normalization": {
            "method": "train_split_zscore",
            "mean": normalization_mean,
            "std": normalization_std,
        },
        "split_hash": split_hash,
        "encoder_checkpoint_hash": encoder.checkpoint_hash,
    }
    embedding_path = Path(paths["embeddings"])
    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embedding_payload, embedding_path)

    train_x = embedding_payload["embeddings"]["train"]
    validation_x = embedding_payload["embeddings"]["validation"]
    dictionary_size = int(enc_cfg["embedding_dim"]) * int(sae_cfg["dictionary_expansion"])
    model = TopKSAE(int(enc_cfg["embedding_dim"]), dictionary_size, int(sae_cfg["top_k"]))
    initial_validation = evaluate_embeddings(model, validation_x)
    history, optimizer = train_sae(
        model, train_x, validation_x, epochs=int(sae_cfg["epochs"]), batch_size=int(sae_cfg["batch_size"]),
        learning_rate=float(sae_cfg["learning_rate"]), weight_decay=float(sae_cfg["weight_decay"]), seed=seed,
    )
    write_curves(paths["curves"], history)
    metadata = {
        "experiment_name": config["experiment_name"], "config": config, "random_seed": seed,
        "data_split_hash": split_hash, "encoder_name": enc_cfg["name"],
        "encoder_checkpoint_hash": encoder.checkpoint_hash,
        "preprocessing_version": data_cfg["preprocessing_version"], "training_curves": history,
        "scope": "synthetic training-only pipeline smoke test; no clinical conclusions",
    }
    save_checkpoint(paths["checkpoint"], model, optimizer, metadata)
    checkpoint_hash = sha256_file(paths["checkpoint"])
    loaded_model, loaded_payload = load_checkpoint(paths["checkpoint"])
    before_reload = evaluate_embeddings(model, validation_x)
    after_reload_1 = evaluate_embeddings(loaded_model, validation_x)
    after_reload_2 = evaluate_embeddings(loaded_model, validation_x)
    deterministic_reload = after_reload_1 == after_reload_2 and before_reload == after_reload_1
    _, sparse_baseline = sparse_coordinate_baseline(
        validation_x, train_x.mean(dim=0, keepdim=True), int(sae_cfg["top_k"])
    )
    metrics = {
        "scope": metadata["scope"],
        "dataset": {"total_images": len(records), "split_images": {k: len(v) for k, v in by_split.items()},
                    "split_patients": {k: len({r['patient_id'] for r in records if r['split'] == k}) for k in by_split}},
        "hashes": {"data_split": split_hash, "encoder_checkpoint": encoder.checkpoint_hash,
                   "sae_checkpoint": checkpoint_hash, "embedding_cache": sha256_file(embedding_path)},
        "initial_validation": initial_validation,
        "same_budget_sparse_coordinate_baseline": sparse_baseline,
        "final": {
            "train": evaluate_embeddings(loaded_model, train_x),
            "validation": after_reload_1,
        },
        "gates": {
            "exact_top_k_or_fewer": after_reload_1["max_active_features"] <= int(sae_cfg["top_k"]),
            "reconstruction_improved": after_reload_1["mse"] < initial_validation["mse"],
            "better_than_same_budget_sparse_baseline": after_reload_1["mse"] < sparse_baseline["mse"],
            "checkpoint_saved_and_reloaded": loaded_payload["metadata"]["data_split_hash"] == split_hash,
            "deterministic_evaluation": deterministic_reload,
            "encoder_frozen": all(not p.requires_grad for p in encoder.parameters()),
        },
        "clinical_claims_authorized": False,
        "unavailable_requirements": ["approved retinal images", "validated RETFound-CFP checkpoint", "clinical labels/masks", "clinician review"],
    }
    write_json(paths["metrics"], metrics)
    initialize_registry(paths["registry"])
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke_global_sae.yaml")
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2, sort_keys=True))
