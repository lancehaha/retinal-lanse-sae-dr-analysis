"""Fit candidate single-token J-Lens directions from training OCT contexts.

Produces candidate directions and train-calibrated validation scores only.
It never emits the `real_model_lens_validated` receipt required by arm D; the
real-model Jacobian and image-control gates must be reviewed separately.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from retinavlm_binary_answer import continuation_ids, post_image_segment
from retinavlm_pre_answer import decoder_blocks
from retinavlm_projected_jacobian import (
    explicit_submatrix_check, mean_projected_directions, projected_score,
)
from retinavlm_released_model import SPECIALIST_REVISION, load_released_specialist
from retinavlm_verified_images import load_verified_manifest, open_verified_image, select_split, sha256


QUESTION = "Is intraretinal or subretinal fluid present in this OCT scan? Answer present or absent."
NEUTRAL_QUESTION = "Describe the retinal OCT image briefly."
READOUT_QUESTIONS = {"answer": QUESTION, "neutral": NEUTRAL_QUESTION}
CANDIDATE_STATUSES = {
    "answer": "candidate_requires_real_model_controls_before_D",
    "neutral": "neutral_candidate_requires_real_model_controls",
}


def balanced_contexts(rows: list[dict[str, str]], count_per_class: int, seed: int) -> list[dict[str, str]]:
    if count_per_class < 1:
        raise ValueError("At least one context per class is needed")
    by_label = defaultdict(list)
    for row in rows:
        by_label[int(row["fluid"])].append(row)
    if any(len(by_label[label]) < count_per_class for label in (0, 1)):
        raise ValueError("Not enough training scans in each fluid class")
    rng = random.Random(seed)
    selected = []
    for label in (0, 1):
        selected.extend(rng.sample(sorted(by_label[label], key=lambda row: row["source_image_path"]), count_per_class))
    rng.shuffle(selected)
    return selected


def tensor_for_row(model, row: dict[str, str], image_root: Path) -> torch.Tensor:
    image = open_verified_image(row, image_root)
    if image.mode != "L":
        raise ValueError(f"OCT image mode {image.mode!r} needs a documented preprocessing decision")
    return model.convert_any_image_to_normalized_tensor(image).unsqueeze(0).to(model.model.device)


def author_prompt(mini, question: str) -> str:
    prompt = mini.llm.build_prompt(mini.config, {"Question": question, "Answer": ""})
    marker = "encoding of a retinal OCT image"
    if prompt.count(marker) != 1:
        raise ValueError("Released author prompt has changed")
    return prompt.replace(marker, marker + " <Img><ImageHere></Img>", 1)


def score_one_image(mini, image_tensor: torch.Tensor,
                    directions: dict[int, torch.Tensor],
                    question: str = QUESTION) -> dict[int, float]:
    blocks = decoder_blocks(mini)
    captured: dict[int, torch.Tensor] = {}
    handles = []
    try:
        for index in directions:
            def hook(_module, _inputs, output, layer=index):
                hidden = output[0] if isinstance(output, tuple) else output
                captured[layer] = hidden[:, -1, :]
            handles.append(blocks[index].register_forward_hook(hook))
        samples = {"Image": image_tensor, "Question": [question], "Answer": [""]}
        with torch.inference_mode():
            embeddings, _, attention_mask = mini.form_input(
                samples, answer_preamble=[""], query=True, pad_side="left"
            )
            mini.llama_model(inputs_embeds=embeddings, attention_mask=attention_mask,
                             use_cache=False, return_dict=True)
            if set(captured) != set(directions):
                raise ValueError("Selected decoder layers were not all captured")
            return {index: float(projected_score(captured[index], direction).item())
                    for index, direction in directions.items()}
    finally:
        for handle in handles:
            handle.remove()


def score_text_only(mini, image_prompt: str,
                    directions: dict[int, torch.Tensor]) -> dict[int, float]:
    """Score the same chat question with no image marker or image embedding."""
    marker = " <Img><ImageHere></Img>"
    if image_prompt.count(marker) != 1:
        raise ValueError("Expected exactly one image marker in author prompt")
    text_prompt = image_prompt.replace(marker, "", 1)
    tokenizer = mini.get_tokenizer()
    tokens = tokenizer(text_prompt, return_tensors="pt", add_special_tokens=False)
    token_ids = tokens.input_ids.to(mini.device)
    embeddings = mini.embed_tokens(token_ids)
    attention_mask = torch.ones_like(token_ids)
    blocks = decoder_blocks(mini)
    captured: dict[int, torch.Tensor] = {}
    handles = []
    try:
        for index in directions:
            def hook(_module, _inputs, output, layer=index):
                hidden = output[0] if isinstance(output, tuple) else output
                captured[layer] = hidden[:, -1, :]
            handles.append(blocks[index].register_forward_hook(hook))
        with torch.inference_mode():
            mini.llama_model(inputs_embeds=embeddings, attention_mask=attention_mask,
                             use_cache=False, return_dict=True)
            if set(captured) != set(directions):
                raise ValueError("Selected text-only decoder layers were not captured")
            return {index: float(projected_score(captured[index], direction).item())
                    for index, direction in directions.items()}
    finally:
        for handle in handles:
            handle.remove()


def score_split(model, rows: list[dict[str, str]], image_root: Path,
                directions: dict[int, torch.Tensor],
                question: str = QUESTION) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    labels = []
    by_layer = {index: [] for index in directions}
    for row in rows:
        score = score_one_image(model.model, tensor_for_row(model, row, image_root),
                                directions, question)
        labels.append(int(row["fluid"]))
        for index in directions:
            by_layer[index].append(score[index])
    return np.asarray(labels, dtype=np.int64), {
        index: np.asarray(values, dtype=np.float64) for index, values in by_layer.items()
    }


def run(*, model_dir: Path, authors_code: Path, manifest: Path, image_audit: Path,
        image_root: Path, output: Path, seed: int, contexts_per_class: int,
        source_layers: list[int], target_layer: int,
        readout_mode: str = "answer") -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    if seed < 0 or not source_layers or len(set(source_layers)) != len(source_layers):
        raise ValueError("Expected nonnegative seed and distinct source layers")
    if readout_mode not in READOUT_QUESTIONS:
        raise ValueError("Unknown J-Lens readout mode")
    readout_question = READOUT_QUESTIONS[readout_mode]
    rows = load_verified_manifest(manifest, image_audit)
    train_rows = sorted(select_split(rows, "train"), key=lambda row: row["source_image_path"])
    validation_rows = sorted(select_split(rows, "validation"), key=lambda row: row["source_image_path"])
    contexts = balanced_contexts(train_rows, contexts_per_class, seed)
    model, _ = load_released_specialist(model_dir, authors_code)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    mini = model.model
    blocks = decoder_blocks(mini)
    if any(index < 0 or index >= target_layer for index in source_layers) or target_layer >= len(blocks):
        raise ValueError("Source layers must precede a valid target layer")
    tokenizer = mini.get_tokenizer()
    prompt = author_prompt(mini, readout_question)
    fluid_ids = continuation_ids(tokenizer, post_image_segment(prompt), "fluid")
    if len(fluid_ids) != 1:
        raise ValueError("Fluid is not one token in the exact author continuation context")
    fluid_id = fluid_ids[0]
    output_embeddings = mini.llama_model.get_output_embeddings()
    unembedding_row = output_embeddings.weight[fluid_id].detach()

    def forwards():
        for row in contexts:
            def forward(selected=row):
                tensor = tensor_for_row(model, selected, image_root)
                samples = {"Image": tensor, "Question": [readout_question], "Answer": [""]}
                embeddings, _, mask = mini.form_input(
                    samples, answer_preamble=[""], query=True, pad_side="left"
                )
                if samples["Input"][0] != prompt:
                    raise ValueError("Lens fit prompt differs from frozen author prompt")
                return mini.llama_model(inputs_embeds=embeddings, attention_mask=mask,
                                        use_cache=False, return_dict=True)
            yield forward

    directions, fitted_contexts = mean_projected_directions(
        forwards(), blocks, source_layers, target_layer, unembedding_row
    )
    control_tensor = tensor_for_row(model, contexts[0], image_root)
    blank_scores = score_one_image(mini, torch.zeros_like(control_tensor),
                                   directions, readout_question)
    text_only_scores = score_text_only(mini, prompt, directions)
    def control_forward():
        samples = {"Image": control_tensor, "Question": [readout_question], "Answer": [""]}
        embeddings, _, mask = mini.form_input(samples, answer_preamble=[""], query=True, pad_side="left")
        return mini.llama_model(inputs_embeds=embeddings, attention_mask=mask,
                                use_cache=False, return_dict=True)
    top_outputs = torch.topk(unembedding_row.float().abs(), k=min(4, unembedding_row.numel())).indices.tolist()
    jacobian_checks = {}
    for index in source_layers:
        top_inputs = torch.topk(directions[index].abs(), k=min(8, directions[index].numel())).indices.tolist()
        jacobian_checks[str(index)] = explicit_submatrix_check(
            control_forward, blocks, index, target_layer, unembedding_row,
            top_inputs, top_outputs,
        )
    train_labels, train_scores = score_split(model, train_rows, image_root,
                                            directions, readout_question)
    val_labels, val_scores = score_split(model, validation_rows, image_root,
                                        directions, readout_question)
    if len(set(train_labels)) != 2 or len(set(val_labels)) != 2:
        raise ValueError("Training and validation each need both fluid classes")
    candidates = {}
    for index in source_layers:
        if not np.isfinite(train_scores[index]).all() or not np.isfinite(val_scores[index]).all():
            raise FloatingPointError(f"Nonfinite J-Lens score at layer {index}")
        calibration = LogisticRegression(solver="lbfgs", max_iter=1000)
        calibration.fit(train_scores[index].reshape(-1, 1), train_labels)
        scale = float(calibration.coef_[0, 0])
        intercept = float(calibration.intercept_[0])
        if not math.isfinite(scale) or scale == 0 or not math.isfinite(intercept):
            raise ValueError(f"Degenerate calibration at layer {index}")
        val_probability = calibration.predict_proba(val_scores[index].reshape(-1, 1))[:, 1]
        positive_scores = val_scores[index][val_labels == 1]
        negative_scores = val_scores[index][val_labels == 0]
        rng = random.Random(seed + index)
        pair_count = min(32, len(positive_scores), len(negative_scores))
        positive_sample = rng.sample(list(positive_scores), pair_count)
        negative_sample = rng.sample(list(negative_scores), pair_count)
        pair_differences = np.asarray(positive_sample) - np.asarray(negative_sample)
        candidates[str(index)] = {
            "direction": directions[index].tolist(),
            "direction_l2": float(directions[index].norm()),
            "scale": scale, "intercept": intercept,
            "train_auc": float(roc_auc_score(train_labels, train_scores[index])),
            "validation_auc": float(roc_auc_score(val_labels, val_probability)),
            "validation_score_std": float(val_scores[index].std()),
            "validation_positive_mean": float(val_scores[index][val_labels == 1].mean()),
            "validation_negative_mean": float(val_scores[index][val_labels == 0].mean()),
            "same_prompt_opposite_label_pairs": pair_count,
            "same_prompt_positive_score_higher_fraction": float((pair_differences > 0).mean()),
            "same_prompt_median_absolute_score_change": float(np.median(np.abs(pair_differences))),
            "blank_image_score": blank_scores[index],
            "text_only_score": text_only_scores[index],
            "jacobian_submatrix_check": jacobian_checks[str(index)],
        }
    receipt = {
        "status": CANDIDATE_STATUSES[readout_mode],
        "checkpoint_revision": SPECIALIST_REVISION,
        "manifest_sha256": sha256(manifest),
        "image_audit_sha256": sha256(image_audit),
        "seed": seed,
        "fit_contexts_per_class": contexts_per_class,
        "fitted_contexts": fitted_contexts,
        "fit_context_source_paths": [row["source_image_path"] for row in contexts],
        "calibration_train_scans": len(train_rows),
        "validation_scans": len(validation_rows),
        "fluid_token_id": fluid_id,
        "question": readout_question,
        "readout_mode": readout_mode,
        "source_layers": source_layers,
        "target_layer": target_layer,
        "candidates": candidates,
        "validation_score_rows": [
            {"source_image_path": row["source_image_path"],
             "Patient_ID": row["Patient_ID"], "trial": row["trial"],
             "fluid": int(row["fluid"]),
             "scores": {str(index): float(val_scores[index][position]) for index in source_layers}}
            for position, row in enumerate(validation_rows)
        ],
        "image_signal_gate": "measurements_recorded_requires_review",
        "jacobian_submatrix_gate": "measurements_recorded_requires_review",
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "test_split_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--authors-code", type=Path, required=True)
    parser.add_argument("--official-image-manifest", type=Path, required=True)
    parser.add_argument("--image-audit", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=4106)
    parser.add_argument("--contexts-per-class", type=int, default=32)
    parser.add_argument("--source-layers", type=int, nargs="+", default=[8, 16, 24])
    parser.add_argument("--target-layer", type=int, default=31)
    parser.add_argument("--readout-mode", choices=sorted(READOUT_QUESTIONS), default="answer")
    args = parser.parse_args()
    print(json.dumps(run(
        model_dir=args.model_dir, authors_code=args.authors_code,
        manifest=args.official_image_manifest, image_audit=args.image_audit,
        image_root=args.image_root, output=args.output, seed=args.seed,
        contexts_per_class=args.contexts_per_class,
        source_layers=args.source_layers, target_layer=args.target_layer,
        readout_mode=args.readout_mode,
    ), indent=2))
