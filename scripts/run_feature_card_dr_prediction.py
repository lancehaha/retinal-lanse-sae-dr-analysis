from __future__ import annotations

"""Experiment 6: prediction from fixed, language-grounded SAE feature cards."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import math
import os
import re
import sys
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, cohen_kappa_score, roc_auc_score

from retina_lanse.data.embedding_shards import load_embedding_run
from retina_lanse.data.manifest import sha256_path
from retina_lanse.sae.checkpoints import load_checkpoint
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
    patient_groups,
    threshold_metrics,
)


SEED = 4106
SHUFFLE_SEED = 9206
CARD_COUNT = 256
TOP_CARDS = 16
RESCUE_FEATURES = {9180, 3322, 7388, 15184, 12428}
DUPLICATE_DESCRIPTION_GROUPS = ((3322, 7388), (15184, 12428))
LLM_ARMS = ("real_descriptions", "description_free", "shuffled_descriptions")
LLM_PROMPT = """You are predicting diabetic-retinopathy grade from frozen SAE feature-card evidence.
The target retinal image, patient identity, and target label are hidden. Card descriptions are
uncertain same-model visual hypotheses, not verified lesions. Training-only association statistics
are supervised evidence, not causal evidence. Return strict compact JSON only:
{"grade_probabilities":[p0,p1,p2,p3,p4],"predicted_grade":0,"explanation":"...","uncertainty":"low|medium|high"}
The five probabilities must be finite, nonnegative, and sum to one. Do not invent image findings."""
REPAIR_PROMPT = """Repair the following response into strict JSON with exactly these keys:
{"grade_probabilities":[p0,p1,p2,p3,p4],"predicted_grade":0,"explanation":"...","uncertainty":"low|medium|high"}
Use only information already present. Do not add evidence. Return JSON only.

RESPONSE:
"""
BATCH_LLM_PROMPT = """You are predicting diabetic-retinopathy grade for several independent cases from
frozen SAE feature-card evidence. No target image, patient identity, or target label is shown. Card
descriptions are uncertain same-model hypotheses, not verified lesions. Return strict compact JSON:
{"predictions":[{"case_id":0,"grade_probabilities":[p0,p1,p2,p3,p4],"predicted_grade":0,"explanation":"short evidence summary","uncertainty":"low|medium|high"}]}
Return exactly one prediction for every supplied case_id. Probabilities must be finite, nonnegative,
and sum to one. Do not combine cases and do not invent image findings."""
BATCH_REPAIR_PROMPT = """Repair the response into strict JSON of the form
{"predictions":[{"case_id":0,"grade_probabilities":[p0,p1,p2,p3,p4],"predicted_grade":0,"explanation":"...","uncertainty":"low|medium|high"}]}.
Preserve every case_id and use only the original response. Do not add evidence. Return JSON only.

RESPONSE:
"""


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stable_dev_mask(patients: np.ndarray, seed: int = SEED, fraction: float = 0.10) -> np.ndarray:
    threshold = int(fraction * 2**64)
    return np.asarray(
        [
            int.from_bytes(hashlib.sha256(f"experiment6:{seed}:{patient}".encode()).digest()[:8], "big")
            < threshold
            for patient in patients
        ],
        dtype=bool,
    )


def card_matrix(feature_ids: np.ndarray, values: np.ndarray, card_ids: np.ndarray) -> np.ndarray:
    """Convert rank traces to a compact identity-preserving card activation matrix."""
    if feature_ids.shape != values.shape or feature_ids.ndim != 2 or feature_ids.shape[1] != 1024:
        raise ValueError("feature-ID/value traces must be aligned [N,1024] arrays")
    if len(np.unique(card_ids)) != CARD_COUNT:
        raise ValueError("feature cards must contain exactly 256 unique feature IDs")
    lookup = np.full(16384, -1, dtype=np.int32)
    lookup[card_ids] = np.arange(len(card_ids), dtype=np.int32)
    result = np.zeros((len(values), len(card_ids)), dtype=np.float32)
    rows = np.arange(len(values))
    for rank in range(values.shape[1]):
        ids = feature_ids[:, rank]
        valid_id = (ids >= 0) & (ids < len(lookup))
        columns = np.full(len(ids), -1, dtype=np.int32)
        columns[valid_id] = lookup[ids[valid_id]]
        keep = (columns >= 0) & (values[:, rank] > 0)
        if np.any(keep):
            result[rows[keep], columns[keep]] = values[keep, rank]
    return result


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 1.0
    for reverse_rank in range(len(p) - 1, -1, -1):
        index = order[reverse_rank]
        rank = reverse_rank + 1
        running = min(running, float(p[index]) * len(p) / rank)
        adjusted[index] = min(1.0, running)
    return adjusted


def patient_level_arrays(matrix: np.ndarray, grades: np.ndarray, patients: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
    groups = patient_groups(patients)
    ids = sorted(groups)
    patient_x = np.stack([matrix[groups[patient]].max(axis=0) for patient in ids])
    patient_y = np.asarray([grades[groups[patient]].max() for patient in ids], dtype=np.int64)
    return patient_x, patient_y, ids


def association_statistics(
    matrix: np.ndarray,
    grades: np.ndarray,
    patients: np.ndarray,
    card_ids: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> list[dict]:
    """Patient-level Spearman trends with a vectorized cluster bootstrap."""
    patient_x, patient_y, patient_ids = patient_level_arrays(np.log1p(matrix), grades, patients)
    x_rank = np.stack([rankdata(patient_x[:, j], method="average") for j in range(patient_x.shape[1])], axis=1).astype(np.float64)
    y_rank = rankdata(patient_y, method="average").astype(np.float64)
    x_rank -= x_rank.mean(axis=0, keepdims=True)
    y_rank -= y_rank.mean()
    denominator = np.sqrt(np.sum(x_rank * x_rank, axis=0) * np.sum(y_rank * y_rank))
    rho = np.divide(y_rank @ x_rank, denominator, out=np.zeros(x_rank.shape[1]), where=denominator > 0)

    rng = np.random.default_rng(seed)
    n = len(patient_ids)
    boot = np.empty((resamples, len(card_ids)), dtype=np.float32)
    # Sampling patient indices preserves both eyes through the patient-level maximum summary.
    for start in range(0, resamples, 32):
        stop = min(start + 32, resamples)
        for row in range(start, stop):
            sampled = rng.integers(0, n, size=n)
            ys = y_rank[sampled]
            xs = x_rank[sampled]
            ys = ys - ys.mean()
            xs = xs - xs.mean(axis=0, keepdims=True)
            den = np.sqrt(np.sum(xs * xs, axis=0) * np.sum(ys * ys))
            boot[row] = np.divide(ys @ xs, den, out=np.zeros(len(card_ids)), where=den > 0)
    p_values = 2.0 * np.minimum(np.mean(boot <= 0, axis=0), np.mean(boot >= 0, axis=0))
    q_values = bh_adjust(p_values)
    records = []
    for column, feature_id in enumerate(card_ids):
        lo, hi = np.quantile(boot[:, column], [0.025, 0.975])
        direction = "mixed_or_uncertain"
        if q_values[column] < 0.05 and lo > 0:
            direction = "supports_increasing_dr_grade"
        elif q_values[column] < 0.05 and hi < 0:
            direction = "supports_decreasing_dr_grade"
        grade_rows = []
        for grade in range(5):
            mask = grades == grade
            grade_rows.append(
                {
                    "grade": grade,
                    "images": int(mask.sum()),
                    "activation_rate": float(np.mean(matrix[mask, column] > 0)),
                    "mean_positive_activation": float(matrix[mask, column][matrix[mask, column] > 0].mean()) if np.any(matrix[mask, column] > 0) else 0.0,
                }
            )
        records.append(
            {
                "feature_id": int(feature_id),
                "patient_level_spearman_rho": float(rho[column]),
                "bootstrap_ci_lower": float(lo),
                "bootstrap_ci_upper": float(hi),
                "bootstrap_p_value": float(p_values[column]),
                "bh_q_value": float(q_values[column]),
                "association_direction": direction,
                "grade_statistics": grade_rows,
                "resampling_unit": "patient_max_grade_and_patient_max_activation",
            }
        )
    return records


def fit_evidence_tables(matrix: np.ndarray, grades: np.ndarray, alpha: float = 1.0) -> dict:
    priors = (np.bincount(grades, minlength=5).astype(np.float64) + alpha) / (len(grades) + 5 * alpha)
    thresholds = np.zeros((matrix.shape[1], 3), dtype=np.float32)
    grade_probs = np.zeros((matrix.shape[1], 5, 5), dtype=np.float64)
    for feature in range(matrix.shape[1]):
        positive = matrix[:, feature][matrix[:, feature] > 0]
        thresholds[feature] = np.quantile(positive, [0.25, 0.50, 0.75]) if len(positive) else 0.0
        bins = np.zeros(len(matrix), dtype=np.int64)
        active = matrix[:, feature] > 0
        bins[active] = 1 + np.searchsorted(thresholds[feature], matrix[active, feature], side="right")
        for bin_index in range(5):
            counts = np.bincount(grades[bins == bin_index], minlength=5).astype(np.float64)
            grade_probs[feature, bin_index] = (counts + alpha) / (counts.sum() + 5 * alpha)
    log_likelihood_ratio = np.clip(np.log(grade_probs / priors[None, None, :]), -3.0, 3.0)
    return {"priors": priors, "thresholds": thresholds, "grade_probabilities": grade_probs, "log_likelihood_ratio": log_likelihood_ratio}


def evidence_logits(matrix: np.ndarray, tables: dict, eligible: np.ndarray, top_cards: int = TOP_CARDS) -> np.ndarray:
    logits = np.tile(np.log(tables["priors"]), (len(matrix), 1))
    columns = np.flatnonzero(eligible)
    for row in range(len(matrix)):
        active = columns[matrix[row, columns] > 0]
        if len(active) > top_cards:
            order = np.lexsort((active, -matrix[row, active]))[:top_cards]
            active = active[order]
        for feature in active:
            bin_index = 1 + int(np.searchsorted(tables["thresholds"][feature], matrix[row, feature], side="right"))
            logits[row] += tables["log_likelihood_ratio"][feature, bin_index]
    return logits


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = logits / max(float(temperature), 1e-6)
    scaled -= scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def fit_temperature(logits: np.ndarray, grades: np.ndarray) -> float:
    candidates = np.geomspace(0.25, 4.0, 121)
    losses = []
    for temperature in candidates:
        probs = softmax(logits, float(temperature))
        losses.append(float(-np.log(np.clip(probs[np.arange(len(grades)), grades], 1e-12, 1.0)).mean()))
    return float(candidates[int(np.argmin(losses))])


def fit_probe(features: np.ndarray, targets: np.ndarray, *, multiclass: bool, seed: int = SEED):
    model = LogisticRegression(
        class_weight="balanced",
        C=1.0,
        solver="lbfgs",
        max_iter=2000,
        tol=1e-3,
        multi_class="multinomial" if multiclass else "auto",
        random_state=seed,
    )
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    model.fit((features - mean) / std, targets)
    return model, mean, std


def predict_probe(bundle, features: np.ndarray) -> np.ndarray:
    model, mean, std = bundle
    return model.predict_proba((features - mean) / std)


def parse_probability_json(text: str) -> dict | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(re.sub(r",\s*([}\]])", r"\1", text[start : end + 1]))
        except json.JSONDecodeError:
            return None
    if not isinstance(parsed, dict):
        return None
    raw = parsed.get("grade_probabilities")
    if not isinstance(raw, list) or len(raw) != 5:
        return None
    try:
        probs = np.asarray([float(x) for x in raw], dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(probs)) or np.any(probs < 0) or probs.sum() <= 0:
        return None
    probs /= probs.sum()
    parsed["grade_probabilities"] = probs.tolist()
    parsed["predicted_grade"] = int(np.argmax(probs))
    return parsed


def parse_batch_probability_json(text: str, expected: int) -> dict[int, dict] | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(re.sub(r",\s*([}\]])", r"\1", text[start : end + 1]))
        except json.JSONDecodeError:
            return None
    predictions = parsed.get("predictions") if isinstance(parsed, dict) else None
    if not isinstance(predictions, list):
        return None
    result = {}
    for row in predictions:
        if not isinstance(row, dict) or not isinstance(row.get("case_id"), int):
            continue
        normalized = parse_probability_json(json.dumps(row))
        if normalized is not None and 0 <= int(row["case_id"]) < expected:
            result[int(row["case_id"])] = normalized
    return result if len(result) == expected else None


def prediction_json_schema(expected_predictions: int) -> dict:
    return {
        "type": "object",
        "properties": {
            "predictions": {
                "type": "array",
                "minItems": expected_predictions,
                "maxItems": expected_predictions,
                "items": {
                    "type": "object",
                    "properties": {
                        "case_id": {"type": "integer", "minimum": 0, "maximum": expected_predictions - 1},
                        "grade_probabilities": {
                            "type": "array",
                            "minItems": 5,
                            "maxItems": 5,
                            "items": {"type": "number", "minimum": 0},
                        },
                        "predicted_grade": {"type": "integer", "minimum": 0, "maximum": 4},
                        "explanation": {"type": "string"},
                        "uncertainty": {"type": "string", "enum": ["low", "medium", "high"]},
                    },
                    "required": ["case_id", "grade_probabilities", "predicted_grade", "explanation", "uncertainty"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["predictions"],
        "additionalProperties": False,
    }


def call_ollama(base_url: str, model: str, prompt: str, timeout: int, context_length: int, expected_predictions: int) -> str:
    payload = {
        "model": model,
        "stream": False,
        "format": prediction_json_schema(expected_predictions),
        "options": {"temperature": 0, "num_ctx": context_length, "num_predict": 1536},
        "messages": [{"role": "user", "content": prompt}],
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return str(json.loads(response.read().decode())["message"]["content"])


def final_hypotheses(card_records: list[dict], rescue_records: list[dict]) -> tuple[dict[int, str], set[int]]:
    hypotheses: dict[int, str] = {}
    bounded_pass: set[int] = set()
    for row in card_records:
        feature = int(row["feature_id"])
        versions = row.get("hypothesis_versions", [])
        if versions and versions[-1].get("hypothesis"):
            hypotheses[feature] = str(versions[-1]["hypothesis"])
        if row.get("final_status") in {"passed_initial", "revised_and_passed"}:
            bounded_pass.add(feature)
    for row in rescue_records:
        feature = int(row["feature_id"])
        passing = [round_row for round_row in row.get("rounds", []) if round_row.get("validation", {}).get("pass")]
        if not passing:
            continue
        normalized = [str(item.get("hypothesis", "")).strip() for item in passing if item.get("hypothesis")]
        if normalized:
            counts = Counter(normalized)
            hypotheses[feature] = sorted(counts, key=lambda text: (-counts[text], normalized.index(text)))[0]
    if len(hypotheses) != CARD_COUNT or len(bounded_pass) != 251:
        raise RuntimeError(f"expected 256 hypotheses and 251 bounded-pass cards; got {len(hypotheses)} and {len(bounded_pass)}")
    return hypotheses, bounded_pass


def prompt_context(
    row: np.ndarray,
    card_ids: np.ndarray,
    tables: dict,
    associations: dict[int, dict],
    hypotheses: dict[int, str],
) -> tuple[list[list], list[int]]:
    active = np.flatnonzero(row > 0)
    if len(active) > TOP_CARDS:
        active = active[np.lexsort((card_ids[active], -row[active]))[:TOP_CARDS]]
    grouped: list[list[int]] = []
    consumed: set[int] = set()
    for group in DUPLICATE_DESCRIPTION_GROUPS:
        present = [int(np.where(card_ids == feature)[0][0]) for feature in group if feature in set(card_ids[active])]
        if present:
            grouped.append(present)
            consumed.update(present)
    grouped.extend([[int(index)] for index in active if int(index) not in consumed])
    rows = []
    used = []
    for group in grouped:
        feature_ids = [int(card_ids[index]) for index in group]
        used.extend(feature_ids)
        evidence = []
        for index in group:
            bin_index = 1 + int(np.searchsorted(tables["thresholds"][index], row[index], side="right"))
            evidence.append(
                [
                    int(card_ids[index]),
                    int(bin_index),
                    np.round(tables["grade_probabilities"][index, bin_index], 4).tolist(),
                    associations[int(card_ids[index])]["association_direction"],
                    round(associations[int(card_ids[index])]["patient_level_spearman_rho"], 4),
                ]
            )
        # Structured rows avoid embedding JSON as an escaped string. Descriptions are emitted once
        # per unique group in the batch-level catalog rather than repeated for every image.
        rows.append([feature_ids, evidence])
    return rows, used


def run_smoke() -> dict:
    ids = np.asarray([[5, 2, 1, -1] + [-1] * 1020], dtype=np.int32)
    values = np.asarray([[0.2, 0.9, 0.4, 0.0] + [0.0] * 1020], dtype=np.float32)
    card_ids = np.arange(256, dtype=np.int64)
    matrix = card_matrix(ids, values, card_ids)
    assert matrix.shape == (1, 256) and matrix[0, 5] == np.float32(0.2) and matrix[0, 2] == np.float32(0.9)
    tables = fit_evidence_tables(np.vstack([matrix] * 5), np.arange(5))
    probs = softmax(evidence_logits(matrix, tables, np.ones(256, dtype=bool)))
    parsed = parse_probability_json('{"grade_probabilities":[1,2,3,4,5]}')
    assert probs.shape == (1, 5) and parsed is not None and math.isclose(sum(parsed["grade_probabilities"]), 1.0)
    return {"status": "smoke_passed", "card_matrix_shape": [1, 256], "top_cards": 16, "labels_in_prompt": False}


def evaluate_arm(grades, patients, binary_scores, grade_probs, threshold, bootstrap, seed):
    binary_targets = (grades >= 2).astype(np.int64)
    return {
        "binary": {**binary_metrics(binary_targets, binary_scores, patients, bootstrap, seed), **threshold_metrics(binary_targets, binary_scores, threshold)},
        "five_grade": multiclass_metrics(grades, grade_probs),
    }


def qwk_difference(grades, left_probs, right_probs, patients, bootstrap, seed):
    groups = patient_groups(np.asarray(patients))
    patient_ids = sorted(groups)
    rng = np.random.default_rng(seed)
    values = []
    left = np.argmax(left_probs, axis=1)
    right = np.argmax(right_probs, axis=1)
    for _ in range(bootstrap):
        sampled = rng.integers(0, len(patient_ids), size=len(patient_ids))
        indices = np.concatenate([groups[patient_ids[i]] for i in sampled])
        values.append(float(cohen_kappa_score(grades[indices], left[indices], weights="quadratic") - cohen_kappa_score(grades[indices], right[indices], weights="quadratic")))
    return {"difference": float(cohen_kappa_score(grades, left, weights="quadratic") - cohen_kappa_score(grades, right, weights="quadratic")), "ci_lower": float(np.quantile(values, 0.025)), "ci_upper": float(np.quantile(values, 0.975)), "resamples": bootstrap}


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment 6 feature-card DR prediction")
    parser.add_argument("--train-run")
    parser.add_argument("--validation-run")
    parser.add_argument("--internal-test-run")
    parser.add_argument("--idrid-run")
    parser.add_argument("--idrid-manifest")
    parser.add_argument("--labels")
    parser.add_argument("--active-trace")
    parser.add_argument("--feature-cards")
    parser.add_argument("--card-records")
    parser.add_argument("--extended-rescue")
    parser.add_argument("--normalization")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="qwen2.5vl:3b")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--llm-batch-size", type=int, default=8)
    parser.add_argument("--llm-splits", nargs="+", choices=("validation", "locked_test", "idrid"), default=["locked_test", "idrid"])
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--association-bootstrap", type=int, default=2000)
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = require_external(Path(args.output), "output")
    output.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        smoke = run_smoke()
        (output / "smoke.json").write_text(json.dumps(smoke, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(smoke, indent=2))
        return 0
    required = {name: getattr(args, name.replace(" ", "_")) for name in ("train run", "validation run", "internal test run", "idrid run", "idrid manifest", "labels", "active trace", "feature cards", "card records", "extended rescue", "normalization", "checkpoint")}
    missing = [name for name, value in required.items() if not value]
    if missing:
        parser.error("missing full-run inputs: " + ", ".join(missing))
    roots = {name: require_external(Path(value), name) for name, value in required.items()}
    if args.bootstrap_resamples < 100 or args.association_bootstrap < 100:
        parser.error("bootstrap counts must be at least 100")

    checkpoint, checkpoint_payload = load_checkpoint(roots["checkpoint"])
    if checkpoint_payload.get("model_config") != {"input_dim": 1024, "dictionary_size": 16384, "k": 1024}:
        raise RuntimeError("Experiment 6 requires the frozen 1024x16384 Top-K-1024 SAE")
    del checkpoint
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

    cards = load_jsonl(roots["feature cards"])
    card_ids = np.asarray([int(row["feature_id"]) for row in cards], dtype=np.int64)
    if len(cards) != CARD_COUNT or len(np.unique(card_ids)) != CARD_COUNT:
        raise RuntimeError("Experiment 6 requires 256 unique feature cards")
    card_records = load_jsonl(roots["card records"])
    rescue_records = load_jsonl(roots["extended rescue"])
    hypotheses, bounded_pass = final_hypotheses(card_records, rescue_records)
    trace = np.load(roots["active trace"])
    split_trace_names = {
        "train": ("train_feature_ids", "train_values"),
        "validation": ("validation_feature_ids", "validation_values"),
        "locked_test": ("test_feature_ids", "test_values"),
        "idrid": ("idrid_feature_ids", "idrid_values"),
    }
    matrices = {split: card_matrix(trace[id_key], trace[value_key], card_ids) for split, (id_key, value_key) in split_trace_names.items()}
    expected_rows = {"train": len(train_dense), "validation": len(val_dense), "locked_test": len(test_dense), "idrid": len(idrid_dense)}
    if any(len(matrices[split]) != count for split, count in expected_rows.items()):
        raise RuntimeError("active trace rows do not match embedding cohorts")

    specification = {
        "schema_version": "1.0",
        "experiment": 6,
        "cards": 256,
        "bounded_pass_cards": 251,
        "top_cards_per_prediction": 16,
        "association_resamples": args.association_bootstrap,
        "evaluation_resamples": args.bootstrap_resamples,
        "llm_model": args.model,
        "llm_arms": list(LLM_ARMS),
        "llm_batch_size": args.llm_batch_size,
        "llm_splits": args.llm_splits,
        "locked_test_used_for_selection": False,
    }
    freeze_path = output / "specification_freeze.json"
    freeze_path.write_text(json.dumps(specification, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    associations = association_statistics(matrices["train"], train_grades, train_patients, card_ids, resamples=args.association_bootstrap, seed=SEED)
    association_by_id = {int(row["feature_id"]): row for row in associations}
    association_path = output / "card_associations.jsonl"
    with association_path.open("w", encoding="utf-8") as handle:
        for row in associations:
            row["hypothesis"] = hypotheses[int(row["feature_id"])]
            row["validation_tier"] = "bounded_two_round_pass" if int(row["feature_id"]) in bounded_pass else "ten_round_rescue"
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    dev_mask = stable_dev_mask(train_patients)
    fit_mask = ~dev_mask
    selection_tables = fit_evidence_tables(matrices["train"][fit_mask], train_grades[fit_mask])
    eligible_all = np.ones(CARD_COUNT, dtype=bool)
    eligible_conservative = np.asarray([int(feature) in bounded_pass for feature in card_ids], dtype=bool)
    fit_logits = evidence_logits(matrices["train"][dev_mask], selection_tables, eligible_all)
    temperature = fit_temperature(fit_logits, train_grades[dev_mask])
    # Once temperature is frozen, refit every train-derived table/head on all
    # available training patients before any held-out prediction is produced.
    tables = fit_evidence_tables(matrices["train"], train_grades)

    dense_binary = fit_probe(train_dense, (train_grades >= 2).astype(np.int64), multiclass=False)
    dense_multi = fit_probe(train_dense, train_grades, multiclass=True)
    card_features = np.log1p(matrices["train"])
    card_binary = fit_probe(card_features, (train_grades >= 2).astype(np.int64), multiclass=False)
    card_multi = fit_probe(card_features, train_grades, multiclass=True)

    train_predictions = {
        "dense": predict_probe(dense_binary, train_dense)[:, 1],
        "direct_card_evidence": softmax(evidence_logits(matrices["train"], tables, eligible_all), temperature)[:, 2:].sum(axis=1),
        "learned_card_probe": predict_probe(card_binary, card_features)[:, 1],
    }
    thresholds = {name: fit_youden_threshold((train_grades >= 2).astype(np.int64), scores) for name, scores in train_predictions.items()}
    split_payload = {
        "validation": (val_dense, val_grades, val_patients, val_paths),
        "locked_test": (test_dense, test_grades, test_patients, test_paths),
        "idrid": (idrid_dense, idrid_grades, idrid_patients, idrid_paths),
    }
    numeric_probabilities: dict[str, dict[str, dict[str, np.ndarray]]] = defaultdict(dict)
    metrics: dict[str, dict[str, dict]] = defaultdict(dict)
    for split, (dense, grades, patients, _) in split_payload.items():
        dense_bin = predict_probe(dense_binary, dense)[:, 1]
        dense_grade = predict_probe(dense_multi, dense)
        direct_grade = softmax(evidence_logits(matrices[split], tables, eligible_all), temperature)
        conservative_grade = softmax(evidence_logits(matrices[split], tables, eligible_conservative), temperature)
        card_bin = predict_probe(card_binary, np.log1p(matrices[split]))[:, 1]
        card_grade = predict_probe(card_multi, np.log1p(matrices[split]))
        arm_values = {
            "dense": (dense_bin, dense_grade),
            "direct_card_evidence": (direct_grade[:, 2:].sum(axis=1), direct_grade),
            "direct_card_evidence_251": (conservative_grade[:, 2:].sum(axis=1), conservative_grade),
            "learned_card_probe": (card_bin, card_grade),
        }
        for arm, (binary_scores, grade_probs) in arm_values.items():
            threshold = thresholds.get(arm, thresholds["direct_card_evidence"])
            metrics[arm][split] = evaluate_arm(grades, patients, binary_scores, grade_probs, threshold, args.bootstrap_resamples, SEED + len(metrics[arm]))
            numeric_probabilities[arm][split] = {"binary": binary_scores.astype(np.float32), "grade": grade_probs.astype(np.float32)}

    comparisons = {}
    target = (test_grades >= 2).astype(np.int64)
    dense_test = numeric_probabilities["dense"]["locked_test"]
    for arm in ("direct_card_evidence", "direct_card_evidence_251", "learned_card_probe"):
        candidate = numeric_probabilities[arm]["locked_test"]
        comparisons[arm + "_vs_dense"] = {
            "auroc": binary_difference(target, candidate["binary"], dense_test["binary"], test_patients, roc_auc_score, args.bootstrap_resamples, SEED + 300),
            "qwk": qwk_difference(test_grades, candidate["grade"], dense_test["grade"], test_patients, args.bootstrap_resamples, SEED + 301),
        }

    description_ids = card_ids.copy()
    rng = np.random.default_rng(SHUFFLE_SEED)
    shuffled = description_ids.copy()
    rng.shuffle(shuffled)
    real_map = {int(feature): hypotheses[int(feature)] for feature in card_ids}
    shuffled_map = {int(feature): hypotheses[int(source)] for feature, source in zip(description_ids, shuffled)}
    llm_path = output / "llm_predictions.jsonl"
    raw_batch_path = output / "llm_raw_batches.jsonl"
    prior_rows = {}
    if llm_path.is_file():
        for row in load_jsonl(llm_path):
            prior_rows[(row.get("arm"), row.get("split"), row.get("relative_path"))] = row
    llm_rows = []
    if not args.skip_llm:
        if args.llm_batch_size < 1:
            parser.error("llm-batch-size must be positive")
        base_url = args.base_url or os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")

        def predict_batch(batch: list[tuple[str, str, int, str]]) -> tuple[list[dict], dict]:
            arm, split = batch[0][0], batch[0][1]
            mapping = real_map if arm == "real_descriptions" else shuffled_map if arm == "shuffled_descriptions" else None
            cases = []
            bases = []
            description_catalog = {}
            for case_id, (_, _, index, path) in enumerate(batch):
                context, used = prompt_context(matrices[split][index], card_ids, tables, association_by_id, hypotheses)
                cases.append([case_id, context])
                for grouped_feature_ids, _ in context:
                    catalog_key = "+".join(str(feature) for feature in grouped_feature_ids)
                    description_catalog[catalog_key] = None if mapping is None else mapping[grouped_feature_ids[0]]
                bases.append({"arm": arm, "split": split, "relative_path": path, "target_image_shown": False, "target_label_in_prompt": False, "feature_ids": used})
            prompt_payload = {
                "description_catalog": description_catalog,
                "cases": cases,
            }
            prompt = (
                BATCH_LLM_PROMPT
                + "\n\nPAYLOAD SCHEMA: description_catalog maps grouped feature IDs to hypothesis text or null; "
                + "cases are [case_id, card_rows]; each card_row is [grouped_feature_ids, "
                + "[[feature_id,activation_bin,grade_probabilities,association_direction,rho],...]].\nPAYLOAD:\n"
                + json.dumps(prompt_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            )
            batch_id = hashlib.sha256(prompt.encode()).hexdigest()
            try:
                raw = call_ollama(base_url, args.model, prompt, args.timeout, args.context_length, len(batch))
                parsed = parse_batch_probability_json(raw, len(batch))
                repaired_raw = None
                # A repair call may fix syntax only. If the first response does not already contain
                # one probability record and case identifier per requested case, a repair model
                # would have to invent missing predictions without seeing the original evidence.
                repair_eligible = (
                    raw.count("grade_probabilities") >= len(batch)
                    and raw.count("case_id") >= len(batch)
                )
                if parsed is None and repair_eligible:
                    repaired_raw = call_ollama(base_url, args.model, BATCH_REPAIR_PROMPT + raw, args.timeout, args.context_length, len(batch))
                    parsed = parse_batch_probability_json(repaired_raw, len(batch))
                if parsed is None:
                    rows = [{**base, "status": "unresolved_parse_failure", "batch_id": batch_id} for base in bases]
                else:
                    rows = [{**base, "status": "complete", "batch_id": batch_id, "parsed_response": parsed[index]} for index, base in enumerate(bases)]
                return rows, {"batch_id": batch_id, "arm": arm, "split": split, "cases": len(batch), "prompt": prompt, "prompt_sha256": batch_id, "raw_response": raw, "repair_eligible": repair_eligible, "repair_response": repaired_raw, "status": "complete" if parsed is not None else "unresolved_parse_failure"}
            except (urllib.error.URLError, TimeoutError, OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
                rows = [{**base, "status": "request_failed", "batch_id": batch_id, "error": repr(exc)} for base in bases]
                return rows, {"batch_id": batch_id, "arm": arm, "split": split, "cases": len(batch), "prompt": prompt, "prompt_sha256": batch_id, "status": "request_failed", "error": repr(exc)}

        batches = []
        for arm in LLM_ARMS:
            for split in args.llm_splits:
                paths = split_payload[split][3]
                pending = []
                for index, path in enumerate(paths):
                    key = (arm, split, path)
                    if key in prior_rows:
                        llm_rows.append(prior_rows[key])
                    else:
                        pending.append((arm, split, index, path))
                batches.extend([pending[start : start + args.llm_batch_size] for start in range(0, len(pending), args.llm_batch_size)])
        with llm_path.open("w", encoding="utf-8") as handle:
            for row in llm_rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(predict_batch, batch) for batch in batches]
                with raw_batch_path.open("a", encoding="utf-8") as raw_handle:
                    total_batches = len(futures)
                    for completed_batches, future in enumerate(as_completed(futures), start=1):
                        rows, raw_record = future.result()
                        llm_rows.extend(rows)
                        for row in rows:
                            handle.write(json.dumps(row, sort_keys=True) + "\n")
                        raw_handle.write(json.dumps(raw_record, sort_keys=True) + "\n")
                        handle.flush()
                        raw_handle.flush()
                        if completed_batches == 1 or completed_batches % 25 == 0 or completed_batches == total_batches:
                            print(json.dumps({
                                "phase": "llm_prediction",
                                "completed_batches": completed_batches,
                                "total_batches": total_batches,
                                "completed_cases": len(llm_rows),
                            }), flush=True)

    llm_metrics = {}
    llm_comparisons = {}
    if llm_rows:
        prior = tables["priors"]
        scored_probabilities = defaultdict(dict)
        for arm in LLM_ARMS:
            llm_metrics[arm] = {}
            for split in args.llm_splits:
                _, grades, patients, paths = split_payload[split]
                by_path = {row["relative_path"]: row for row in llm_rows if row.get("arm") == arm and row.get("split") == split}
                grade_probs = np.stack([np.asarray(by_path[path]["parsed_response"]["grade_probabilities"], dtype=np.float64) if by_path.get(path, {}).get("status") == "complete" else prior for path in paths])
                binary_scores = grade_probs[:, 2:].sum(axis=1)
                scored_probabilities[arm][split] = {"binary": binary_scores, "grade": grade_probs}
                llm_metrics[arm][split] = {**evaluate_arm(grades, patients, binary_scores, grade_probs, float(prior[2:].sum()), args.bootstrap_resamples, SEED + 700), "completion_rate": float(sum(by_path.get(path, {}).get("status") == "complete" for path in paths) / len(paths)), "failure_fallback": "training_grade_prior"}
        if "locked_test" in args.llm_splits:
            target = (test_grades >= 2).astype(np.int64)
            for control in ("description_free", "shuffled_descriptions"):
                name = "real_descriptions_vs_" + control
                llm_comparisons[name] = binary_difference(target, scored_probabilities["real_descriptions"]["locked_test"]["binary"], scored_probabilities[control]["locked_test"]["binary"], test_patients, roc_auc_score, args.bootstrap_resamples, SEED + 750)
            ordered = sorted(llm_comparisons, key=lambda key: llm_comparisons[key]["p_value"])
            running = 0.0
            for rank, key in enumerate(ordered):
                running = max(running, min(1.0, llm_comparisons[key]["p_value"] * (len(ordered) - rank)))
                llm_comparisons[key]["holm_adjusted_p_value"] = running

    arrays = {}
    for arm, splits in numeric_probabilities.items():
        for split, values in splits.items():
            arrays[f"{arm}_{split}_binary"] = values["binary"]
            arrays[f"{arm}_{split}_grade"] = values["grade"]
    prediction_path = output / "numeric_predictions.npz"
    np.savez_compressed(prediction_path, **arrays)
    association_arrays = output / "card_evidence_tables.npz"
    np.savez_compressed(association_arrays, card_ids=card_ids, priors=tables["priors"], thresholds=tables["thresholds"], grade_probabilities=tables["grade_probabilities"], log_likelihood_ratio=tables["log_likelihood_ratio"], bounded_pass_mask=eligible_conservative)

    result = {
        "schema_version": "1.0",
        "status": "complete" if not args.skip_llm and llm_rows else "complete_numeric_llm_pending",
        "experiment": 6,
        "encoder_frozen": True,
        "sae_frozen": True,
        "semantic_checkpoint_seed": 4106,
        "cohorts": {split: {"images": len(payload[1]), "units": len(set(payload[2]))} for split, payload in {"train": (train_dense, train_grades, train_patients, train_paths), **split_payload}.items()},
        "cards": {"all": 256, "bounded_two_round_pass": 251, "ten_round_rescue": sorted(RESCUE_FEATURES), "duplicate_description_groups": [list(group) for group in DUPLICATE_DESCRIPTION_GROUPS]},
        "temperature": temperature,
        "numeric_metrics": metrics,
        "numeric_comparisons": comparisons,
        "llm_metrics": llm_metrics,
        "llm_comparisons": llm_comparisons,
        "success_criteria": {"card_auroc_noninferiority_margin": -0.03, "card_qwk_noninferiority_margin": -0.05, "semantic_minimum_auroc_gain": 0.02},
        "outputs": {"associations": str(association_path), "evidence_tables": str(association_arrays), "numeric_predictions": str(prediction_path), "llm_predictions": str(llm_path), "llm_raw_batches": str(raw_batch_path)},
        "provenance": {name.replace(" ", "_") + "_sha256": sha256_path(path if path.is_file() else path / "extraction_evidence.json") for name, path in roots.items()},
        "specification_freeze_sha256": sha256_path(freeze_path),
        "scope_boundary": "Card associations and Qwen explanations are research evidence, not lesion confirmation, causality, or clinical-deployment validation.",
    }
    result_path = output / "experiment_6_feature_card_prediction.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    table_rows = []
    for arm, split_rows in metrics.items():
        for split, record in split_rows.items():
            table_rows.append({"arm": arm, "split": split, "completion_rate": 1.0, "auroc": record["binary"]["auroc"]["estimate"], "auprc": record["binary"]["auprc"]["estimate"], "qwk": record["five_grade"]["quadratic_weighted_kappa"], "macro_auroc": record["five_grade"]["macro_auroc"], "ordinal_mae": record["five_grade"]["ordinal_mae"]})
    for arm, split_rows in llm_metrics.items():
        for split, record in split_rows.items():
            table_rows.append({"arm": "llm_" + arm, "split": split, "completion_rate": record["completion_rate"], "auroc": record["binary"]["auroc"]["estimate"], "auprc": record["binary"]["auprc"]["estimate"], "qwk": record["five_grade"]["quadratic_weighted_kappa"], "macro_auroc": record["five_grade"]["macro_auroc"], "ordinal_mae": record["five_grade"]["ordinal_mae"]})
    csv_path = output / "experiment_6_results_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)
    md = ["# Experiment 6 results", "", "| Arm | Split | Completion | AUROC | AUPRC | QWK | Macro-AUROC | Ordinal MAE |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in table_rows:
        md.append(f"| {row['arm']} | {row['split']} | {row['completion_rate']:.3f} | {row['auroc']:.4f} | {row['auprc']:.4f} | {row['qwk']:.4f} | {row['macro_auroc']:.4f} | {row['ordinal_mae']:.4f} |")
    (output / "experiment_6_results_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "result": str(result_path), "locked_test": {arm: values["locked_test"] for arm, values in metrics.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
