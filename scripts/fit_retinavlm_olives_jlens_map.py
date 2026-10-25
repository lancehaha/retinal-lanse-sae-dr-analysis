"""Fit and audit candidate OCT finding readouts on an aligned RetinaVLM.

Only train patients fit directions/calibration/probes. Validation patients
provide complete score tables and image controls. This does not read test data
or declare a candidate suitable for auxiliary training by itself.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from evaluate_retinavlm_adapter_olives import image_batch
from retinavlm_binary_answer import answer_token_ids
from retinavlm_pre_answer import decoder_blocks, pre_answer_forward
from retinavlm_projected_jacobian import answer_margin_directions_for_batch, projected_score
from retinavlm_released_model import load_released_specialist
from train_retinavlm_adapter_olives import (ADAPTED_SHA, QUESTIONS, SPLIT_SHA,
                                            install_encoder, load_rows, sha256)


def choose_balanced(rows, finding, per_class, seed, excluded=frozenset()):
    """Draw distinct scans, cycling through patients within each class."""
    rng = random.Random(seed)
    selected = []
    for label in (0, 1):
        by_patient = defaultdict(list)
        for row in rows:
            if row[finding] == str(label) and row["source_image_path"] not in excluded:
                by_patient[row["patient_key"]].append(row)
        if sum(map(len, by_patient.values())) < per_class:
            raise ValueError(f"Insufficient distinct {finding}={label} scans")
        for group in by_patient.values():
            rng.shuffle(group)
        patients = sorted(by_patient)
        rng.shuffle(patients)
        group_selected = []
        while len(group_selected) < per_class:
            available = [p for p in patients if by_patient[p]]
            rng.shuffle(available)
            for patient in available:
                if len(group_selected) == per_class:
                    break
                group_selected.append(by_patient[patient].pop())
        selected.extend(group_selected)
    rng.shuffle(selected)
    return selected


def choose_patient_disjoint_fit(rows, finding, direction_per_class,
                                calibration_per_class, seed):
    """Keep direction and calibration patients separate for both labels."""
    patients = sorted({row["patient_key"] for row in rows})
    rng = random.Random(seed)
    for _ in range(10000):
        rng.shuffle(patients)
        cut = len(patients) // 2
        first = set(patients[:cut])
        direction_pool = [row for row in rows if row["patient_key"] in first]
        calibration_pool = [row for row in rows if row["patient_key"] not in first]
        if all(sum(row[finding] == str(label) for row in direction_pool) >= direction_per_class
               and sum(row[finding] == str(label) for row in calibration_pool) >= calibration_per_class
               for label in (0, 1)):
            direction = choose_balanced(direction_pool, finding, direction_per_class, seed)
            calibration = choose_balanced(calibration_pool, finding, calibration_per_class, seed + 1)
            return direction, calibration
    raise ValueError(f"Cannot form patient-disjoint direction/calibration sets for {finding}")


def capture_scores(model, rows, image_root, finding, layers, directions,
                   absent_id, present_id, batch_size, *, blank=False,
                   retain_states=False):
    mini = model.model
    blocks = decoder_blocks(mini)
    scores = {layer: [] for layer in layers}
    states = {layer: [] for layer in layers} if retain_states else None
    margins = []
    for start in range(0, len(rows), batch_size):
        part = rows[start:start + batch_size]
        images = image_batch(model, part, image_root)
        if blank:
            images = torch.zeros_like(images)
        captured = {}
        handles = []
        try:
            for layer in layers:
                def hook(_module, _inputs, output, index=layer):
                    hidden = output[0] if isinstance(output, tuple) else output
                    captured[index] = hidden[:, -1, :]
                handles.append(blocks[layer].register_forward_hook(hook))
            samples = {"Image": images, "Question": [QUESTIONS[finding]] * len(part),
                       "Answer": [""] * len(part)}
            with torch.inference_mode():
                logits, _ = pre_answer_forward(mini, samples)
                if set(captured) != set(layers):
                    raise ValueError("Missing decoder layer activation")
                margins.extend((logits[:, present_id] - logits[:, absent_id]).float().cpu().tolist())
                for layer in layers:
                    hidden = captured[layer].float().cpu()
                    scores[layer].extend(projected_score(hidden, directions[layer]).tolist())
                    if retain_states:
                        states[layer].append(hidden.numpy())
        finally:
            for handle in handles:
                handle.remove()
        if (start // batch_size + 1) % 50 == 0 or start + len(part) == len(rows):
            print(json.dumps({"finding": finding, "blank": blank, "scored": start + len(part),
                              "total": len(rows)}), flush=True)
    arrays = {layer: np.asarray(values, dtype=np.float32) for layer, values in scores.items()}
    state_arrays = ({layer: np.concatenate(parts) for layer, parts in states.items()}
                    if retain_states else None)
    return np.asarray(margins, dtype=np.float32), arrays, state_arrays


def patient_bootstrap_auc(rows, labels, scores, seed, replicates=1000):
    from sklearn.metrics import roc_auc_score
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["patient_key"]].append(index)
    patients = sorted(groups)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        sample = rng.choice(patients, size=len(patients), replace=True)
        indices = [i for patient in sample for i in groups[patient]]
        if len(set(labels[indices])) == 2:
            values.append(float(roc_auc_score(labels[indices], scores[indices])))
    if not values:
        raise ValueError("Patient bootstrap contained no two-class draws")
    return [float(v) for v in np.quantile(values, [0.025, 0.975])], len(values)


def write_csv(path, rows):
    if not rows:
        raise ValueError("Empty result table")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite_difference_check(model, image, finding, layer, coordinate,
                            absent_id, present_id, epsilon=0.25):
    """Compare one actual-model VJP coordinate with a hidden-state edit."""
    mini = model.model
    block = decoder_blocks(mini)[layer]
    samples = {"Image": image, "Question": [QUESTIONS[finding]], "Answer": [""]}

    def margin(delta):
        def edit(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            updated = hidden.clone()
            updated[:, -1, coordinate] += delta
            return (updated, *output[1:]) if isinstance(output, tuple) else updated
        handle = block.register_forward_hook(edit)
        try:
            with torch.inference_mode():
                logits, _ = pre_answer_forward(mini, samples.copy())
                return float((logits[0, present_id] - logits[0, absent_id]).float())
        finally:
            handle.remove()

    return (margin(epsilon) - margin(-epsilon)) / (2 * epsilon)


def run(args):
    if args.finding not in QUESTIONS or args.contexts_per_class < 2 or args.calibration_per_class < 2:
        raise ValueError("Invalid finding or training sample count")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError("Map output directory must be empty")
    groups = load_rows(args.split_csv)
    checkpoint = torch.load(args.adapter_checkpoint, map_location="cpu")
    if (checkpoint.get("format") != "retinavlm_olives_adapter_alignment_v1"
            or checkpoint.get("variant") != "adapted_byol"
            or checkpoint.get("step") != 10000
            or checkpoint.get("split_sha256") != SPLIT_SHA
            or checkpoint.get("adapted_encoder_sha256") != ADAPTED_SHA):
        raise ValueError("Expected complete aligned BYOL-adapter checkpoint")
    model, _ = load_released_specialist(args.model_dir, args.authors_code)
    install_encoder(model, "adapted_byol", args.adapted_checkpoint)
    model.model.llama_proj.load_state_dict(checkpoint["adapter"], strict=True)
    del checkpoint
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    mini = model.model
    blocks = decoder_blocks(mini)
    layers = tuple(sorted(set(args.layers)))
    if len(layers) != len(args.layers) or not layers or layers[0] < 0 or layers[-1] >= len(blocks):
        raise ValueError("Selected layers are not distinct valid decoder layers")
    absent_id, present_id = answer_token_ids(mini.get_tokenizer(), QUESTIONS[args.finding])
    if args.selection_plan:
        plan = json.loads(args.selection_plan.read_text())
        if (plan.get("protocol") != "retinavlm_jlens_train_gate_v3"
                or plan.get("finding") != args.finding
                or plan.get("split_sha256") != SPLIT_SHA
                or plan.get("direction_per_class") != args.contexts_per_class
                or plan.get("calibration_per_class") != args.calibration_per_class):
            raise ValueError("Selection plan does not match this map fit")
        train_lookup = {row["source_image_path"]: row for row in groups["train"]}
        if len(train_lookup) != len(groups["train"]):
            raise ValueError("Duplicate training scan paths")
        selections = []
        for key in ("direction_paths", "calibration_paths", "gate_paths"):
            paths = plan[key]
            if len(paths) != len(set(paths)) or any(path not in train_lookup for path in paths):
                raise ValueError(f"Invalid {key} in selection plan")
            selections.append([train_lookup[path] for path in paths])
        direction_rows, calibration_rows, val_rows = selections
        cohorts = [{row["patient_key"] for row in cohort} for cohort in selections]
        if (any(cohorts[i] & cohorts[j] for i, j in ((0, 1), (0, 2), (1, 2)))
                or len(set().union(*(set(row["source_image_path"] for row in cohort)
                                      for cohort in selections))) != sum(map(len, selections))
                or any(sum(row[args.finding] == str(label) for row in direction_rows)
                       != args.contexts_per_class for label in (0, 1))
                or any(sum(row[args.finding] == str(label) for row in calibration_rows)
                       != args.calibration_per_class for label in (0, 1))):
            raise ValueError("Plan violates patient disjointness or fitting budgets")
        by_gate_patient = defaultdict(set)
        for row in val_rows:
            if row[args.finding] not in ("0", "1"):
                raise ValueError("Unknown label in gate cohort")
            by_gate_patient[row["patient_key"]].add(row[args.finding])
        if sum(labels == {"0", "1"} for labels in by_gate_patient.values()) < 6:
            raise ValueError("Too few mixed-label patients in gate cohort")
        gate_images = args.train_images
        map_eval_cohort = "held-out original training patients"
    elif args.patient_disjoint_fit:
        direction_rows, calibration_rows = choose_patient_disjoint_fit(
            groups["train"], args.finding, args.contexts_per_class,
            args.calibration_per_class, args.seed)
    else:
        direction_rows = choose_balanced(groups["train"], args.finding,
                                         args.contexts_per_class, args.seed)
        excluded = {r["source_image_path"] for r in direction_rows}
        calibration_rows = choose_balanced(groups["train"], args.finding,
                                           args.calibration_per_class, args.seed + 1, excluded)
    if not args.selection_plan:
        val_rows = [r for r in groups["validation"] if r[args.finding] in {"0", "1"}]
        gate_images = args.validation_images
        map_eval_cohort = "original validation patients"
    direction_sums = {layer: torch.zeros(mini.llama_model.config.hidden_size) for layer in layers}
    first_gradients = None
    first_image = None
    for index, row in enumerate(direction_rows, 1):
        images = image_batch(model, [row], args.train_images)
        def forward():
            samples = {"Image": images, "Question": [QUESTIONS[args.finding]], "Answer": [""]}
            logits, _ = pre_answer_forward(mini, samples)
            return logits
        gradients = answer_margin_directions_for_batch(
            forward, blocks, layers, absent_id, present_id)
        if index == 1:
            first_gradients = gradients
            first_image = images
        for layer in layers:
            direction_sums[layer] += gradients[layer][0]
        if index % 8 == 0 or index == len(direction_rows):
            print(json.dumps({"finding": args.finding, "gradient_context": index,
                              "total": len(direction_rows)}), flush=True)
    directions = {}
    direction_norms = {}
    for layer in layers:
        vector = direction_sums[layer] / len(direction_rows)
        norm = float(vector.norm())
        if not math.isfinite(norm) or norm <= 1e-9:
            raise ValueError(f"Degenerate direction at layer {layer}")
        directions[layer] = vector / norm
        direction_norms[str(layer)] = norm
    gradient_checks = {}
    for layer in layers:
        vector = first_gradients[layer][0]
        coordinate = int(vector.abs().argmax())
        analytic = float(vector[coordinate])
        finite_difference = finite_difference_check(
            model, first_image, args.finding, layer, coordinate,
            absent_id, present_id)
        gradient_checks[str(layer)] = {
            "coordinate": coordinate, "analytic": analytic,
            "finite_difference": finite_difference,
            "absolute_error": abs(analytic - finite_difference),
            "epsilon": 0.25,
        }

    _, train_raw, train_states = capture_scores(
        model, calibration_rows, args.train_images, args.finding, layers,
        directions, absent_id, present_id, args.batch_size, retain_states=True)
    val_margin, val_raw, val_states = capture_scores(
        model, val_rows, gate_images, args.finding, layers,
        directions, absent_id, present_id, args.batch_size, retain_states=True)
    control_rows = choose_balanced(val_rows, args.finding, min(args.control_per_class,
                                   min(sum(r[args.finding] == str(k) for r in val_rows)
                                       for k in (0, 1))), args.seed + 2)
    _, blank_raw, _ = capture_scores(
        model, control_rows, gate_images, args.finding, layers,
        directions, absent_id, present_id, args.batch_size, blank=True)
    train_y = np.asarray([int(r[args.finding]) for r in calibration_rows])
    val_y = np.asarray([int(r[args.finding]) for r in val_rows])
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    calibrations, layer_metrics, probe_states = {}, {}, {}
    for layer in layers:
        mean = float(train_raw[layer].mean())
        std = float(train_raw[layer].std())
        if std <= 1e-8:
            raise ValueError(f"Constant training scores at layer {layer}")
        calibration = LogisticRegression(class_weight="balanced", random_state=args.seed)
        calibration.fit(((train_raw[layer] - mean) / std).reshape(-1, 1), train_y)
        scale = float(calibration.coef_[0, 0] / std)
        intercept = float(calibration.intercept_[0] - scale * mean)
        calibrated = scale * val_raw[layer] + intercept
        auc = float(roc_auc_score(val_y, calibrated))
        ci, valid_replicates = patient_bootstrap_auc(val_rows, val_y, calibrated,
                                                    args.seed + layer)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(train_states[layer])
        probe = LogisticRegression(C=0.01, class_weight="balanced", max_iter=1000,
                                   random_state=args.seed)
        probe.fit(x_train, train_y)
        probe_score = probe.decision_function(scaler.transform(val_states[layer]))
        probe_states[str(layer)] = {"scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
                                    "coef": probe.coef_, "intercept": probe.intercept_}
        control_indices = {r["source_image_path"]: i for i, r in enumerate(val_rows)}
        real_controls = np.asarray([val_raw[layer][control_indices[r["source_image_path"]]]
                                    for r in control_rows])
        blank_change = np.abs(real_controls - blank_raw[layer])
        by_patient = defaultdict(lambda: {0: [], 1: []})
        for i, row in enumerate(val_rows):
            by_patient[row["patient_key"]][int(row[args.finding])].append(i)
        rng = random.Random(args.seed + layer)
        pair_deltas = []
        for patient, classes in sorted(by_patient.items()):
            if classes[0] and classes[1]:
                negative = rng.choice(classes[0])
                positive = rng.choice(classes[1])
                pair_deltas.append(float(calibrated[positive] - calibrated[negative]))
        calibrations[str(layer)] = {"scale": scale, "intercept": intercept,
                                    "training_mean": mean, "training_std": std}
        layer_metrics[str(layer)] = {
            "validation_auc": auc, "patient_bootstrap_auc_ci95": ci,
            "valid_bootstrap_replicates": valid_replicates,
            "linear_probe_validation_auc": float(roc_auc_score(val_y, probe_score)),
            "within_patient_opposite_label_pairs": len(pair_deltas),
            "positive_higher_pair_fraction": float(np.mean(np.asarray(pair_deltas) > 0))
                if pair_deltas else None,
            "median_absolute_blank_change": float(np.median(blank_change)),
            "control_scans": len(control_rows),
        }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"directions": directions, "probe_states": probe_states},
               args.out_dir / "map_and_probe.pt")
    validation_table = []
    for i, row in enumerate(val_rows):
        for layer in layers:
            calibration = calibrations[str(layer)]
            validation_table.append({
                "source_image_path": row["source_image_path"], "patient_key": row["patient_key"],
                "finding": args.finding, "label": int(row[args.finding]), "layer": layer,
                "raw_score": float(val_raw[layer][i]),
                "calibrated_logit": float(calibration["scale"] * val_raw[layer][i]
                                          + calibration["intercept"]),
                "final_answer_margin": float(val_margin[i]),
            })
    write_csv(args.out_dir / "validation_map_scores.csv", validation_table)
    control_table = []
    for i, row in enumerate(control_rows):
        index = next(j for j, item in enumerate(val_rows)
                     if item["source_image_path"] == row["source_image_path"])
        for layer in layers:
            control_table.append({"source_image_path": row["source_image_path"],
                                  "patient_key": row["patient_key"], "finding": args.finding,
                                  "label": int(row[args.finding]), "layer": layer,
                                  "real_score": float(val_raw[layer][index]),
                                  "blank_score": float(blank_raw[layer][i])})
    write_csv(args.out_dir / "image_controls.csv", control_table)
    result = {
        "status": "candidate_map_fitted_validation_pending_gate",
        "finding": args.finding, "question": QUESTIONS[args.finding],
        "source_layers": layers, "direction_contexts": len(direction_rows),
        "calibration_scans": len(calibration_rows), "validation_scans": len(val_rows),
        "direction_per_class": args.contexts_per_class,
        "calibration_per_class": args.calibration_per_class,
        "direction_patients": len({r["patient_key"] for r in direction_rows}),
        "calibration_patients": len({r["patient_key"] for r in calibration_rows}),
        "direction_calibration_patient_overlap": len(
            {r["patient_key"] for r in direction_rows}
            & {r["patient_key"] for r in calibration_rows}),
        "patient_disjoint_fit": args.patient_disjoint_fit or bool(args.selection_plan),
        "validation_patients": len({r["patient_key"] for r in val_rows}),
        "map_eval_cohort": map_eval_cohort,
        "selection_plan_sha256": sha256(args.selection_plan) if args.selection_plan else None,
        "inference_batch_size": args.batch_size,
        "test_accessed": False, "seed": args.seed,
        "split_sha256": SPLIT_SHA, "adapter_sha256": sha256(args.adapter_checkpoint),
        "adapted_encoder_sha256": ADAPTED_SHA,
        "answer_token_ids": {"absent": absent_id, "present": present_id},
        "direction_norms_before_unit_normalization": direction_norms,
        "actual_model_gradient_checks": gradient_checks,
        "calibration": calibrations, "layers": layer_metrics,
        "map_file_sha256": sha256(args.out_dir / "map_and_probe.pt"),
        "validation_rows_sha256": sha256(args.out_dir / "validation_map_scores.csv"),
        "controls_sha256": sha256(args.out_dir / "image_controls.csv"),
        "gpu": torch.cuda.get_device_name(0),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "finding": args.finding,
                      "layers": layer_metrics, "out_dir": str(args.out_dir)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("split-csv", "train-images", "validation-images", "model-dir",
                 "authors-code", "adapted-checkpoint", "adapter-checkpoint", "out-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--finding", choices=list(QUESTIONS), required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[8, 16, 24, 30])
    parser.add_argument("--contexts-per-class", type=int, default=16)
    parser.add_argument("--calibration-per-class", type=int, default=64)
    parser.add_argument("--patient-disjoint-fit", action="store_true")
    parser.add_argument("--selection-plan", type=Path)
    parser.add_argument("--control-per-class", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    run(parser.parse_args())
