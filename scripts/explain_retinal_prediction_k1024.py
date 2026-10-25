from __future__ import annotations

"""Experiment 5 stage 3: feature-card explanations and exploratory LLM prediction."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from retina_lanse.data.embedding_shards import load_embedding_run
from retina_lanse.data.manifest import sha256_path
from scripts.run_global_sae_search import require_external
from scripts.run_label_efficiency_external_benchmark import (
    collect_eyepacs,
    collect_idrid,
    load_idrid_manifest,
    load_eyepacs_labels,
    multiclass_metrics,
)


PROMPT = """You are interpreting a frozen retinal SAE-based research prediction.
The target retinal image is not shown. You receive only feature-card hypotheses
and the activation strengths observed for this target. Hypotheses are uncertain
visual descriptions, not diagnoses. Do not invent lesions, do not mention hidden
ground-truth labels, and do not claim clinical validity.

Return JSON only:
{
  \"feature_observations\": [{\"feature_id\": 0, \"observation\": \"...\", \"support\": \"low/medium/high\"}],
  \"compatible_evidence\": [\"...\"],
  \"conflicting_evidence\": [\"...\"],
  \"uncertainty\": \"low/medium/high\",
  \"explanation\": \"one concise evidence-based explanation\",
  \"predicted_grade\": 0,
  \"grade_probabilities\": [0.2, 0.2, 0.2, 0.2, 0.2],
  \"referable_probability\": 0.0
}
"""


def parse_json(text: str) -> dict | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                cleaned = re.sub(r",\s*([}\]])", r"\1", text[start : end + 1])
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    return None
    return None


def coerce_json_object(parsed):
    """Extract the first useful object from common chat-model wrappers."""
    if isinstance(parsed, dict):
        for key in ("response", "output", "result", "answer"):
            nested = parsed.get(key)
            if isinstance(nested, str):
                nested = parse_json(nested)
            if isinstance(nested, dict):
                return nested
        return parsed
    if isinstance(parsed, str):
        nested = parse_json(parsed)
        if isinstance(nested, dict):
            return nested
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                return item
    return None


def call_ollama(base_url: str, model: str, prompt: str, timeout: int, context_length: int) -> str:
    payload = {"model": model, "stream": False, "format": "json", "options": {"temperature": 0, "num_ctx": context_length, "num_predict": 768}, "messages": [{"role": "user", "content": prompt + "\nReturn compact JSON only; use short strings and finish within 768 tokens."}]}
    request = urllib.request.Request(base_url.rstrip("/") + "/api/chat", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return str(json.loads(response.read().decode())["message"]["content"])


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def make_context(ids: np.ndarray, values: np.ndarray, hypotheses: dict[int, dict], verification: dict[int, dict], top_features: int) -> tuple[str, list[dict]]:
    selected = []
    for feature, value in zip(ids[:top_features], values[:top_features]):
        if int(feature) < 0:
            continue
        feature_id = int(feature)
        card = hypotheses.get(feature_id, {})
        audit = verification.get(feature_id, {})
        selected.append({"feature_id": feature_id, "activation": float(value), "hypothesis": card.get("hypothesis", "no grounded hypothesis"), "evidence_strength": card.get("evidence_strength", "unknown"), "activation_rate": audit.get("all_image_activation_count"), "dr_active_rate": audit.get("labeled_dr_activation_rate"), "non_dr_active_rate": audit.get("labeled_non_dr_activation_rate"), "nuisance_flags": card.get("nuisance_flags", [])})
    context = "\n".join(f"- feature {row['feature_id']}: activation={row['activation']:.6g}; hypothesis={row['hypothesis']}; evidence={row['evidence_strength']}; DR-active-rate={row['dr_active_rate']}; non-DR-active-rate={row['non_dr_active_rate']}; nuisance={row['nuisance_flags']}" for row in selected)
    return context, selected


def normalize_probability_schema(parsed: dict) -> tuple[float, list[float]]:
    """Accept only a complete schema, while tolerating JSON numeric strings."""
    probability = float(parsed["referable_probability"])
    grade_probs_raw = parsed["grade_probabilities"]
    if not isinstance(grade_probs_raw, list) or len(grade_probs_raw) != 5:
        raise ValueError("invalid grade probability vector")
    grade_probs = [float(value) for value in grade_probs_raw]
    if not all(np.isfinite(value) and value >= 0 for value in grade_probs):
        raise ValueError("non-finite or negative grade probability")
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("invalid referable probability")
    total = float(sum(grade_probs))
    if total <= 0:
        raise ValueError("zero grade probability mass")
    grade_probs = [value / total for value in grade_probs]
    return probability, grade_probs


def evaluate_predictions(rows: list[dict]) -> dict:
    valid = [row for row in rows if row.get("status") == "complete" and isinstance(row.get("parsed_response"), dict)]
    if not valid:
        return {"status": "no_complete_predictions", "images_completed": 0}
    targets = np.asarray([int(row["target"]) for row in valid], dtype=np.int64)
    probabilities = np.asarray([float(row["parsed_response"]["referable_probability"]) for row in valid], dtype=np.float64)
    result = {"images_completed": len(valid)}
    if len(np.unique(targets)) == 2:
        result.update({"auroc": float(roc_auc_score(targets, probabilities)), "auprc": float(average_precision_score(targets, probabilities))})
    grades = []
    grade_probs = []
    for row in valid:
        parsed = row["parsed_response"]
        gp = parsed.get("grade_probabilities")
        if isinstance(gp, list) and len(gp) == 5 and all(isinstance(x, (int, float)) for x in gp):
            grade_probs.append(np.asarray(gp, dtype=np.float64))
            grades.append(int(row["target_grade"]))
    if len(grade_probs) == len(valid):
        result["five_class"] = multiclass_metrics(np.asarray(grades, dtype=np.int64), np.stack(grade_probs))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Explain K=1024 active-value retinal predictions with feature cards")
    parser.add_argument("--stage1", required=True)
    parser.add_argument("--stage1-predictions", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--hypotheses", required=True)
    parser.add_argument("--verification", required=True)
    parser.add_argument("--test-run", required=True)
    parser.add_argument("--idrid-run", required=True)
    parser.add_argument("--idrid-manifest", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="qwen2.5-vl:3b")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--top-features", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--context-length", type=int, default=32768)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=4106)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.top_features != 16:
        parser.error("the preregistered explanation stage requires top 16 features")
    roots = {name: require_external(Path(value), name) for name, value in {"stage1": args.stage1, "stage1 predictions": args.stage1_predictions, "trace": args.trace, "hypotheses": args.hypotheses, "verification": args.verification, "test run": args.test_run, "IDRiD run": args.idrid_run, "IDRiD manifest": args.idrid_manifest, "labels": args.labels, "dataset root": args.dataset_root, "output": args.output}.items()}
    output = roots["output"]
    output.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        smoke = {"status": "smoke_passed", "top_features": 16, "target_image_shown_to_llm": False, "labels_hidden_from_prompt": True, "classifier_score_used_only_after_response": True}
        (output / "stage3_smoke.json").write_text(json.dumps(smoke, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(smoke, indent=2))
        return 0
    stage1 = json.loads(roots["stage1"].read_text(encoding="utf-8"))
    predictions = np.load(roots["stage1 predictions"])
    trace = np.load(roots["trace"])
    hypotheses = {int(row["feature_id"]): row.get("parsed_response", {}) for row in load_jsonl(roots["hypotheses"]) if row.get("status") == "complete"}
    verification = {int(row["feature_id"]): row for row in load_jsonl(roots["verification"])}
    labels = load_eyepacs_labels(roots["labels"])
    idrid_labels = load_idrid_manifest(roots["IDRiD manifest"])
    test_run = load_embedding_run(roots["test run"], expected_split="internal_test", verify_sha256=True)
    idrid_run = load_embedding_run(roots["IDRiD run"], expected_split="external", verify_sha256=True)
    _, test_grades, test_patients, test_paths = collect_eyepacs(test_run, labels)
    _, idrid_grades, idrid_patients, idrid_paths = collect_idrid(idrid_run, idrid_labels)
    split_data = {"locked_test": (trace["test_feature_ids"], trace["test_values"], test_paths, test_patients, test_grades, predictions["1.00_dense_plus_active_values_test"]), "idrid": (trace["idrid_feature_ids"], trace["idrid_values"], idrid_paths, idrid_patients, idrid_grades, predictions["1.00_dense_plus_active_values_idrid"])}
    result_path = output / "explanations.jsonl"
    partial = {}
    if result_path.is_file():
        for row in load_jsonl(result_path):
            partial[(row.get("split"), row.get("relative_path"))] = row
    rows = []
    base_url = args.base_url or os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
    def make_row(split: str, feature_ids: np.ndarray, feature_values: np.ndarray, path: str, patient: str, grade: int, numerical_score: float) -> dict:
        context, selected = make_context(feature_ids, feature_values, hypotheses, verification, args.top_features)
        prompt = PROMPT + "\nActivated feature-card evidence (the target image is hidden):\n" + (context or "No grounded feature cards were available.")
        base = {"split": split, "relative_path": path, "patient_id": str(patient), "target": int(grade >= 2), "target_grade": int(grade), "numerical_fusion_probability": float(numerical_score), "active_feature_ids": [int(x) for x in feature_ids if int(x) >= 0], "active_values": [float(x) for x in feature_values if float(x) > 0], "explained_feature_cards": selected, "labels_hidden_from_prompt": True}
        try:
            raw = call_ollama(base_url, args.model, prompt, args.timeout, args.context_length)
            parsed = coerce_json_object(parse_json(raw))
            if not isinstance(parsed, dict):
                raise ValueError("invalid LLM JSON object")
            probability, grade_probs = normalize_probability_schema(parsed)
            parsed = {**parsed, "referable_probability": probability, "grade_probabilities": grade_probs}
            return {**base, "status": "complete", "model": args.model, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "raw_response": raw, "parsed_response": parsed}
        except (urllib.error.URLError, TimeoutError, OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            return {**base, "status": "request_failed", "error": repr(exc)}

    with result_path.open("w", encoding="utf-8") as handle:
        pending = []
        for split, (ids, values, paths, patients, grades, numerical_scores) in split_data.items():
            for index, (feature_ids, feature_values, path, patient, grade, numerical_score) in enumerate(zip(ids, values, paths, patients, grades, numerical_scores)):
                key = (split, path)
                if key in partial and partial[key].get("status") == "complete":
                    row = partial[key]
                    rows.append(row)
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                    handle.flush()
                else:
                    pending.append((split, feature_ids.copy(), feature_values.copy(), path, patient, int(grade), float(numerical_score)))
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(make_row, *item) for item in pending]
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
    summary = {"schema_version": "1.0", "status": "complete_feature_explanations" if all(row.get("status") == "complete" for row in rows) else "partial_feature_explanations", "model": args.model, "top_features_explained": args.top_features, "target_image_shown_to_llm": False, "labels_hidden_from_prompt": True, "hypotheses_completed": len(hypotheses), "numerical_stage_status": stage1.get("status"), "splits": {split: {"images_requested": sum(row.get("split") == split for row in rows), "images_completed": sum(row.get("split") == split and row.get("status") == "complete" for row in rows), "metrics": evaluate_predictions([row for row in rows if row.get("split") == split])} for split in split_data}, "output_jsonl": str(result_path), "provenance": {"stage1_sha256": sha256_path(roots["stage1"]), "stage1_predictions_sha256": sha256_path(roots["stage1 predictions"]), "trace_sha256": sha256_path(roots["trace"]), "hypotheses_sha256": sha256_path(roots["hypotheses"]), "verification_sha256": sha256_path(roots["verification"]), "labels_sha256": sha256_path(roots["labels"])}, "caveat": "LLM explanations summarize feature-card hypotheses and activations; they are not direct lesion confirmation or specialist validation."}
    (output / "explanations_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
