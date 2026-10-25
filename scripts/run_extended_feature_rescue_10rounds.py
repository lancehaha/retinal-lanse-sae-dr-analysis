from __future__ import annotations

"""Ten-round exploratory rescue for the five Experiment 5B cards.

The original two-round result is never overwritten. Each card receives up to
ten isolated hypothesis-generation/contrastive-validation rounds. Prompts are
card-local: no hypothesis or feature card from another feature is supplied.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retina_lanse.data.embedding_shards import load_embedding_run
from retina_lanse.sae.checkpoints import load_checkpoint
from scripts.run_global_sae_search import require_external
from scripts.run_single_sae_feasibility import encode_csr
from scripts.run_contrastive_feature_hypothesis_validation import (
    DISCOVERY_N, call_ollama, parse_json, sample_rows, valid_reviews,
)


FEATURE_IDS = (9180, 3322, 7388, 15184, 12428)
ROUNDS = 10
GENERATION_PROMPT = """You are describing one sparse-autoencoder feature in retinal fundus photographs.
The first numbered contact sheet is the fixed HIGH-activation discovery group (32 images).
The second is a NEW LOW-activation control group (32 images). They are the only images you may
use. Labels, patient IDs, activation values, disease status, and all other feature cards are
hidden. Give one concise contrastive visual hypothesis: what visible pattern is more consistently
present in HIGH than LOW? Do not infer diabetic-retinopathy grade or prevalence. Return JSON only
with exactly these keys:
{"hypothesis":"one concise visual description","evidence":"visible contrast",
"alternatives":["possible nuisance explanation"],"confidence":"low/medium/high",
"uncertainty":"what remains uncertain"}
"""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hypothesis_text(parsed: dict | None) -> str:
    if not isinstance(parsed, dict):
        return ""
    return str({str(k).lower(): v for k, v in parsed.items()}.get("hypothesis", "")).strip()


def norm_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--train-run", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--normalization", required=True)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--feature-cards", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--feature-ids", nargs="+", type=int, default=list(FEATURE_IDS))
    p.add_argument("--model", default="qwen2.5vl:3b")
    p.add_argument("--base-url", default="http://127.0.0.1:11434")
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    out = require_external(Path(args.output), "output")
    out.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        result = {"status": "smoke_passed", "cards": list(FEATURE_IDS),
                  "rounds_per_card": ROUNDS, "validation_high": 32,
                  "validation_low": 32, "classifier_receives_feature_ids": False,
                  "cross_card_hypothesis_context": False, "labels_in_prompts": False,
                  "stable_rule": "10 rounds required and at least 8 complete passes"}
        (out / "extended_rescue_smoke.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2)); return 0

    roots = {k: require_external(Path(v), k) for k, v in {
        "train run": args.train_run, "checkpoint": args.checkpoint,
        "normalization": args.normalization, "dataset root": args.dataset_root,
        "feature cards": args.feature_cards}.items()}
    run = load_embedding_run(roots["train run"], expected_split="train", verify_sha256=True)
    model, ckpt = load_checkpoint(roots["checkpoint"])
    if ckpt.get("model_config") != {"input_dim": 1024, "dictionary_size": 16384, "k": 1024}:
        raise RuntimeError("checkpoint must be frozen K=1024 dictionary-size 16384")
    norm = torch.load(roots["normalization"], map_location="cpu", weights_only=False)
    dense = (run.embeddings - norm["mean"].float()) / norm["std"].float()
    codes = encode_csr(model, dense, batch_size=args.batch_size)
    cards = {int(x["feature_id"]): x for x in load_jsonl(roots["feature cards"])}
    missing = [fid for fid in args.feature_ids if fid not in cards]
    if missing:
        raise RuntimeError(f"feature cards missing requested IDs: {missing}")
    dataset = roots["dataset root"]
    paths, patients = run.relative_paths, run.patient_ids
    result_path = out / "extended_feature_rescue_10rounds.jsonl"
    done = {int(r["feature_id"]) for r in load_jsonl(result_path)} if result_path.exists() else set()
    records: list[dict] = load_jsonl(result_path) if result_path.exists() else []

    for fid in args.feature_ids:
        if fid in done:
            continue
        card = cards[fid]
        discovery = card.get("top_high_activation_examples", [])[:DISCOVERY_N]
        discovery_patients = {str(x.get("patient_id")) for x in discovery}
        discovery_paths = {str(x["relative_path"]) for x in discovery}
        high_paths = [dataset / x["relative_path"] for x in discovery if (dataset / x["relative_path"]).is_file()]
        excluded = set(discovery_paths)
        excluded_patients = set(discovery_patients)
        rec = {"feature_id": fid, "original_status": "failed_twice_or_unparsed",
               "rounds_requested": ROUNDS, "labels_hidden": True,
               "cross_card_hypothesis_context": False, "discovery_high": discovery,
               "rounds": [], "checkpoint_sha256": sha256(roots["checkpoint"]),
               "normalization_sha256": sha256(roots["normalization"])}
        for round_index in range(1, ROUNDS + 1):
            generation_low = sample_rows(codes, paths, patients, fid, excluded, False, 32,
                                         excluded_patients=excluded_patients)
            excluded.update(x["relative_path"] for x in generation_low)
            excluded_patients.update(str(x["patient_id"]) for x in generation_low)
            low_paths = [dataset / x["relative_path"] for x in generation_low if (dataset / x["relative_path"]).is_file()]
            prompt = GENERATION_PROMPT
            raw, parsed, hypothesis = "", None, ""
            generation_status = "request_failed"
            try:
                raw = call_ollama(args.base_url, args.model, prompt, high_paths, low_paths,
                                  args.timeout, 8192)
                parsed = parse_json(raw)
                hypothesis = hypothesis_text(parsed)
                generation_status = "complete" if hypothesis else "unparsed"
            except Exception as exc:
                raw = repr(exc)
            round_record = {"round": round_index, "generation_status": generation_status,
                            "generation_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                            "generation_raw_response": raw, "generation_parsed_response": parsed,
                            "hypothesis": hypothesis, "generation_low": generation_low,
                            "validation": None}
            if hypothesis:
                validation_patients = set(excluded_patients)
                validation_high = sample_rows(codes, paths, patients, fid, excluded, True, 32,
                                              excluded_patients=validation_patients)
                excluded.update(x["relative_path"] for x in validation_high)
                validation_patients.update(str(x["patient_id"]) for x in validation_high)
                validation_low = sample_rows(codes, paths, patients, fid, excluded, False, 32,
                                             excluded_patients=validation_patients)
                excluded.update(x["relative_path"] for x in validation_low)
                excluded_patients.update(str(x["patient_id"]) for x in validation_low)
                from scripts.recover_unparsed_feature_hypotheses_contrastive import validate_round
                round_record["validation"] = {
                    "samples": {"high": validation_high, "low": validation_low},
                    **validate_round(args.base_url, args.model, dataset, validation_high,
                                     validation_low, hypothesis, args.timeout, round_index)}
            rec["rounds"].append(round_record)
        complete = [r for r in rec["rounds"] if r.get("validation", {}).get("status") == "complete"]
        passed = [r for r in complete if r.get("validation", {}).get("pass") is True]
        rec["complete_validation_rounds"] = len(complete)
        rec["passing_rounds"] = len(passed)
        rec["stable_pass_8_of_10"] = bool(len(rec["rounds"]) == ROUNDS and len(passed) >= 8)
        rec["final_status"] = "stable_after_10_rounds" if rec["stable_pass_8_of_10"] else "not_stable_after_10_rounds"
        records.append(rec)
        with result_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(rec, sort_keys=True) + "\n")

    # Post-hoc duplicate audit. It never changes pass/fail status.
    records = load_jsonl(result_path)
    groups: dict[str, list[int]] = {}
    for rec in records:
        for rr in rec.get("rounds", []):
            h = norm_text(rr.get("hypothesis", ""))
            if h:
                groups.setdefault(h, []).append(int(rec["feature_id"]))
    duplicates = {k: sorted(set(v)) for k, v in groups.items() if len(set(v)) > 1}
    for rec in records:
        own_duplicate_ids: set[int] = set()
        for ids in duplicates.values():
            if int(rec["feature_id"]) in ids:
                own_duplicate_ids.update(ids)
        own_duplicate_ids.discard(int(rec["feature_id"]))
        rec["duplicate_hypothesis_feature_ids"] = sorted(own_duplicate_ids)
    with result_path.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec, sort_keys=True) + "\n")
    summary = {"schema_version": "1.0", "status": "complete",
               "features_requested": len(args.feature_ids), "features_processed": len(records),
               "rounds_per_feature": ROUNDS,
               "stable_after_10_rounds": sum(r.get("stable_pass_8_of_10") is True for r in records),
               "not_stable_after_10_rounds": sum(r.get("stable_pass_8_of_10") is not True for r in records),
               "hypothesis_generation_complete_rounds": sum(sum(x.get("generation_status") == "complete" for x in r.get("rounds", [])) for r in records),
               "cross_card_duplicate_groups": len(duplicates),
               "cross_card_hypothesis_context": False, "labels_in_prompts": False,
               "feature_ids": list(args.feature_ids), "model": args.model,
               "checkpoint_sha256": sha256(roots["checkpoint"]),
               "normalization_sha256": sha256(roots["normalization"]),
               "feature_cards_sha256": sha256(roots["feature cards"]),
               "output_jsonl": str(result_path)}
    (out / "extended_feature_rescue_10rounds_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
