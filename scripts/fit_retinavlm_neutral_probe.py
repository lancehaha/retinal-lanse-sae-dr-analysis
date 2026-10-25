"""Fit frozen-state neutral-prompt linear probes for the J-Lens comparison.

Uses all official training patients, train-only standardization, and the same
released model, prompt, layers, and validation patients as the neutral lens.
The image encoder, visual adapter, and Llama decoder are never updated.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from fit_retinavlm_fluid_lens_candidate import (
    CANDIDATE_STATUSES, NEUTRAL_QUESTION, tensor_for_row,
)
from retinavlm_pre_answer import decoder_blocks
from retinavlm_released_model import SPECIALIST_REVISION, load_released_specialist
from retinavlm_verified_images import load_verified_manifest, select_split, sha256


def hidden_states_one_image(mini, image_tensor: torch.Tensor,
                            layers: list[int]) -> dict[int, np.ndarray]:
    blocks = decoder_blocks(mini)
    if not layers or len(set(layers)) != len(layers) or any(index < 0 or index >= len(blocks)
                                                        for index in layers):
        raise ValueError("Probe layers are invalid")
    captured = {}
    handles = []
    try:
        for index in layers:
            def hook(_module, _inputs, output, layer=index):
                hidden = output[0] if isinstance(output, tuple) else output
                if hidden.ndim != 3 or hidden.shape[0] != 1:
                    raise ValueError("Expected one pre-answer state per OCT image")
                captured[layer] = hidden[:, -1, :].detach().float().cpu().numpy()[0]
            handles.append(blocks[index].register_forward_hook(hook))
        samples = {"Image": image_tensor, "Question": [NEUTRAL_QUESTION], "Answer": [""]}
        with torch.inference_mode():
            embeddings, _, attention_mask = mini.form_input(
                samples, answer_preamble=[""], query=True, pad_side="left"
            )
            mini.llama_model(inputs_embeds=embeddings, attention_mask=attention_mask,
                             use_cache=False, return_dict=True)
        if set(captured) != set(layers):
            raise ValueError("Selected neutral decoder layers did not all execute")
        if any(not np.isfinite(vector).all() for vector in captured.values()):
            raise FloatingPointError("Neutral pre-answer state is nonfinite")
        return captured
    finally:
        for handle in handles:
            handle.remove()


def collect_states(model, rows: list[dict[str, str]], image_root: Path,
                   layers: list[int]) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    labels = []
    states = {layer: [] for layer in layers}
    for row in rows:
        vectors = hidden_states_one_image(model.model,
                                          tensor_for_row(model, row, image_root), layers)
        labels.append(int(row["fluid"]))
        for layer in layers:
            states[layer].append(vectors[layer])
    return np.asarray(labels, dtype=np.int64), {
        layer: np.stack(vectors).astype(np.float32, copy=False)
        for layer, vectors in states.items()
    }


def fit_probe(train: np.ndarray, labels: np.ndarray,
              validation: np.ndarray, seed: int) -> tuple[np.ndarray, dict]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    if (train.ndim != 2 or validation.ndim != 2 or train.shape[1] != validation.shape[1]
            or len(train) != len(labels) or len(set(labels.tolist())) != 2
            or not np.isfinite(train).all() or not np.isfinite(validation).all()):
        raise ValueError("Probe train/validation states are incomplete")
    scaler = StandardScaler()
    transformed_train = scaler.fit_transform(train)
    transformed_validation = scaler.transform(validation)
    classifier = LogisticRegression(solver="liblinear", C=1.0, max_iter=1000,
                                    random_state=seed)
    classifier.fit(transformed_train, labels)
    logits = classifier.decision_function(transformed_validation)
    if not np.isfinite(logits).all():
        raise FloatingPointError("Probe validation scores are nonfinite")
    return logits, {
        "standardizer_mean": scaler.mean_.tolist(),
        "standardizer_scale": scaler.scale_.tolist(),
        "coefficient": classifier.coef_[0].tolist(),
        "intercept": float(classifier.intercept_[0]),
        "coefficient_l2": float(np.linalg.norm(classifier.coef_[0])),
        "trainable_probe_parameters": train.shape[1] + 1,
        "solver": "liblinear", "regularization_C": 1.0,
        "max_iterations": 1000, "actual_iterations": int(classifier.n_iter_[0]),
    }


def run(*, model_dir: Path, authors_code: Path, manifest: Path, image_audit: Path,
        image_root: Path, neutral_candidate: Path, output: Path) -> dict:
    from sklearn.metrics import roc_auc_score

    if output.exists():
        raise FileExistsError("Neutral probe receipt already exists")
    rows = load_verified_manifest(manifest, image_audit)
    manifest_digest, audit_digest = sha256(manifest), sha256(image_audit)
    candidate = json.loads(neutral_candidate.read_text(encoding="utf-8"))
    layers = candidate.get("source_layers")
    if (candidate.get("status") != CANDIDATE_STATUSES["neutral"]
            or candidate.get("readout_mode") != "neutral"
            or candidate.get("question") != NEUTRAL_QUESTION
            or candidate.get("checkpoint_revision") != SPECIALIST_REVISION
            or candidate.get("manifest_sha256") != manifest_digest
            or candidate.get("image_audit_sha256") != audit_digest
            or candidate.get("test_split_accessed") is not False
            or not isinstance(layers, list) or not layers
            or len(set(layers)) != len(layers)
            or any(not isinstance(index, int) or isinstance(index, bool) or index < 0
                   for index in layers)):
        raise ValueError("Neutral candidate is not bound to this cohort and layer set")
    train_rows = sorted(select_split(rows, "train"), key=lambda row: row["source_image_path"])
    validation_rows = sorted(select_split(rows, "validation"),
                             key=lambda row: row["source_image_path"])
    model, _ = load_released_specialist(model_dir, authors_code)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    train_labels, train_states = collect_states(model, train_rows, image_root, layers)
    val_labels, val_states = collect_states(model, validation_rows, image_root, layers)
    if len(set(val_labels.tolist())) != 2:
        raise ValueError("Probe validation needs both fluid classes")
    probe_weights = {}
    val_scores = {}
    val_auc = {}
    for layer in layers:
        logits, weights = fit_probe(train_states[layer], train_labels,
                                    val_states[layer], int(candidate["seed"]))
        if not math.isfinite(weights["coefficient_l2"]) or weights["coefficient_l2"] == 0:
            raise ValueError("Probe is degenerate")
        probe_weights[str(layer)] = weights
        val_scores[str(layer)] = logits.tolist()
        val_auc[str(layer)] = float(roc_auc_score(val_labels, logits))
    result = {
        "status": "frozen_neutral_linear_probe_validation",
        "checkpoint_revision": SPECIALIST_REVISION,
        "manifest_sha256": manifest_digest,
        "image_audit_sha256": audit_digest,
        "neutral_candidate_sha256": sha256(neutral_candidate),
        "question": NEUTRAL_QUESTION,
        "source_layers": layers,
        "train_scans": len(train_rows), "validation_scans": len(validation_rows),
        "trained_components": "one independent linear probe per layer; released RetinaVLM frozen",
        "probe_weights": probe_weights,
        "validation_auc": val_auc,
        "validation_score_rows": [
            {"source_image_path": row["source_image_path"], "Patient_ID": row["Patient_ID"],
             "trial": row["trial"], "fluid": int(row["fluid"]),
             "scores": {str(layer): float(val_scores[str(layer)][position]) for layer in layers}}
            for position, row in enumerate(validation_rows)
        ],
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "test_split_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--authors-code", type=Path, required=True)
    parser.add_argument("--official-image-manifest", type=Path, required=True)
    parser.add_argument("--image-audit", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--neutral-candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(model_dir=args.model_dir, authors_code=args.authors_code,
                 manifest=args.official_image_manifest, image_audit=args.image_audit,
                 image_root=args.image_root, neutral_candidate=args.neutral_candidate,
                 output=args.output)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"probe_weights", "validation_score_rows"}}, indent=2))
