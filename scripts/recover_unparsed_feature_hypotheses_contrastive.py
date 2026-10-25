from __future__ import annotations

"""Recover Experiment 5 hypotheses that were unparsed in the first run.

Each recovery hypothesis is generated contrastively from the original 32 high-
activation discovery images plus 32 new low-activation controls. The recovered
hypothesis is then evaluated on fresh 32-high/32-low groups using the same bounded
one-revision protocol as Experiment 5B.
"""

import argparse
import hashlib
import json
import os
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
    DISCOVERY_N, INITIAL_PROMPT, VERIFY_PROMPT, call_ollama, parse_json,
    sample_rows, valid_reviews,
)

RECOVERY_PROMPT = """You are describing one sparse-autoencoder feature in retinal fundus photographs.
The first numbered contact sheet is the original HIGH-activation discovery group (32 images).
The second is a NEW LOW-activation control group (32 images). Labels, patient IDs, activation
values, and disease status are hidden. Give one concise contrastive visual hypothesis: what
visible pattern is more consistently present in HIGH than LOW? Do not infer diabetic-retinopathy
grade or prevalence. Return JSON only with exactly these keys:
{\"hypothesis\":\"one concise visual description\",\"evidence\":\"visible contrast\",\"alternatives\":[\"...\"],\"confidence\":\"low/medium/high\",\"uncertainty\":\"...\"}
"""


def compact_hypothesis(parsed: dict | None) -> str:
    if not isinstance(parsed, dict):
        return ""
    lowered = {str(k).lower(): v for k, v in parsed.items()}
    return str(lowered.get("hypothesis", "")).strip()


def generation(base_url, model_name, prompt, high_paths, low_paths, timeout):
    raw = call_ollama(base_url, model_name, prompt, high_paths, low_paths, timeout, 8192)
    parsed = parse_json(raw)
    return compact_hypothesis(parsed), raw, parsed


def validate_round(base_url, model_name, dataset, high_items, low_items, hypothesis, timeout, version):
    all_high, all_low, raws, hashes = [], [], [], []
    # Keep the logical groups at 32+32, but use two 16+16 requests for reliable output.
    for start in range(0, 32, 16):
        hi, li = high_items[start:start + 16], low_items[start:start + 16]
        hp = [dataset / x["relative_path"] for x in hi if (dataset / x["relative_path"]).is_file()]
        lp = [dataset / x["relative_path"] for x in li if (dataset / x["relative_path"]).is_file()]
        n = min(len(hp), len(lp))
        if n == 0:
            return {"status": "request_failed", "version": version, "error": "no_image_files"}
        prompt = VERIFY_PROMPT.format(hypothesis=hypothesis, n=n)
        try:
            raw = call_ollama(base_url, model_name, prompt, hp[:n], lp[:n], timeout, 8192)
            parsed = valid_reviews(parse_json(raw), expected_n=n)
            raws.append(raw); hashes.append(hashlib.sha256(prompt.encode()).hexdigest())
            if parsed is None:
                return {"status": "unparsed", "version": version, "raw_responses": raws, "prompt_sha256": hashes}
            h, l = parsed
            for item in h: item["image_number"] += start
            for item in l: item["image_number"] += start
            all_high.extend(h); all_low.extend(l)
        except Exception as exc:
            return {"status": "request_failed", "version": version, "error": repr(exc), "raw_responses": raws, "prompt_sha256": hashes}
    hr = float(np.mean([x["pattern_present"] for x in all_high])); lr = float(np.mean([x["pattern_present"] for x in all_low]))
    return {"status": "complete", "version": version, "high_reviews": all_high, "low_reviews": all_low,
            "high_rate": hr, "low_rate": lr, "contrast": hr - lr,
            "pass": bool(hr >= .60 and hr - lr >= .30 and len(all_high) >= 24 and len(all_low) >= 24),
            "prompt_sha256": hashes, "raw_responses": raws}


def main() -> int:
    p = argparse.ArgumentParser(description="Recover unparsed Experiment 5 SAE hypotheses contrastively")
    p.add_argument("--train-run", required=True); p.add_argument("--checkpoint", required=True)
    p.add_argument("--normalization", required=True); p.add_argument("--dataset-root", required=True)
    p.add_argument("--prior-cards", required=True); p.add_argument("--prior-hypotheses", required=True)
    p.add_argument("--output", required=True); p.add_argument("--model", default="qwen2.5vl:3b")
    p.add_argument("--base-url", default=None); p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=1024); p.add_argument("--smoke", action="store_true")
    p.add_argument("--feature-start", type=int, default=0)
    p.add_argument("--feature-end", type=int, default=None)
    args = p.parse_args()
    if args.smoke:
        out = require_external(Path(args.output), "output"); out.mkdir(parents=True, exist_ok=True)
        result = {"status": "smoke_passed", "discovery_high": 32, "generation_low": 32,
                  "validation_high": 32, "validation_low": 32, "max_revisions": 1,
                  "labels_in_prompts": False, "classifier_receives_feature_ids": False}
        (out / "recovery_smoke.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2)); return 0
    roots = {k: require_external(Path(v), k) for k, v in {
        "train run": args.train_run, "checkpoint": args.checkpoint, "normalization": args.normalization,
        "dataset root": args.dataset_root, "prior cards": args.prior_cards,
        "prior hypotheses": args.prior_hypotheses, "output": args.output}.items()}
    out = roots["output"]; out.mkdir(parents=True, exist_ok=True)
    run = load_embedding_run(roots["train run"], expected_split="train", verify_sha256=True)
    model, ckpt = load_checkpoint(roots["checkpoint"])
    if ckpt.get("model_config") != {"input_dim": 1024, "dictionary_size": 16384, "k": 1024}:
        raise RuntimeError("checkpoint must be frozen K=1024 dictionary-size 16384")
    norm = torch.load(roots["normalization"], map_location="cpu", weights_only=False)
    dense = (run.embeddings - norm["mean"].float()) / norm["std"].float()
    codes = encode_csr(model, dense, batch_size=args.batch_size)
    cards = [json.loads(x) for x in roots["prior cards"].read_text().splitlines() if x.strip()]
    prior = {int(json.loads(x)["feature_id"]): json.loads(x) for x in roots["prior hypotheses"].read_text().splitlines() if x.strip()}
    cards = [c for c in cards if prior.get(int(c["feature_id"]), {}).get("status") == "unparsed"]
    cards = cards[args.feature_start:args.feature_end]
    paths, patients, dataset = run.relative_paths, run.patient_ids, roots["dataset root"]
    base_url = args.base_url or os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
    result_path = out / "recovered_contrastive_feature_validation.jsonl"
    records = []
    for card in cards:
        fid = int(card["feature_id"])
        discovery = card.get("top_high_activation_examples", [])[:DISCOVERY_N]
        discovery_paths = {str(x["relative_path"]) for x in discovery}
        discovery_patients = {str(x.get("patient_id")) for x in discovery}
        excluded = set(discovery_paths)
        generation_low = sample_rows(codes, paths, patients, fid, excluded, False, 32, excluded_patients=discovery_patients)
        excluded.update(x["relative_path"] for x in generation_low)
        gen_prompt = RECOVERY_PROMPT
        high_paths = [dataset / x["relative_path"] for x in discovery if (dataset / x["relative_path"]).is_file()]
        low_paths = [dataset / x["relative_path"] for x in generation_low if (dataset / x["relative_path"]).is_file()]
        h, raw, parsed = "", "", None
        gen_status = "request_failed"
        for attempt in range(2):
            try:
                h, raw, parsed = generation(base_url, args.model, gen_prompt, high_paths, low_paths, args.timeout)
                gen_status = "complete" if h else "unparsed"
            except Exception as exc:
                raw, parsed, gen_status = repr(exc), None, "request_failed"
            if h: break
            gen_prompt = RECOVERY_PROMPT + "\nBe concise. Output no prose outside the JSON object."
        rec = {"feature_id": fid, "global_prominence_rank": card.get("global_prominence_rank"),
               "status": gen_status, "generation_attempts": 2, "generation_prompt_sha256": hashlib.sha256(gen_prompt.encode()).hexdigest(),
               "generation_raw_response": raw, "generation_parsed_response": parsed,
               "discovery_high": discovery, "generation_low": generation_low,
               "hypothesis_versions": [{"version": 1, "source": "original_high_plus_new_low", "hypothesis": h}],
               "validation": [], "labels_hidden": True}
        if not h:
            rec["final_status"] = "hypothesis_unavailable"; records.append(rec)
        else:
            validation_patients = discovery_patients | {x["patient_id"] for x in generation_low}
            initial_high = sample_rows(codes, paths, patients, fid, excluded, True, 32, excluded_patients=validation_patients)
            excluded.update(x["relative_path"] for x in initial_high); validation_patients |= {x["patient_id"] for x in initial_high}
            initial_low = sample_rows(codes, paths, patients, fid, excluded, False, 32, excluded_patients=validation_patients)
            excluded.update(x["relative_path"] for x in initial_low); validation_patients |= {x["patient_id"] for x in initial_low}
            first = validate_round(base_url, args.model, dataset, initial_high, initial_low, h, args.timeout, 1)
            rec["validation"].append({"phase": "initial", "samples": {"high": initial_high, "low": initial_low}, **first})
            if first.get("status") == "complete" and not first.get("pass"):
                revision_low = sample_rows(codes, paths, patients, fid, excluded, False, 32, excluded_patients=validation_patients)
                excluded.update(x["relative_path"] for x in revision_low); validation_patients |= {x["patient_id"] for x in revision_low}
                rev_prompt = RECOVERY_PROMPT + "\nThis is a bounded revision. Reconsider the contrast using the same original HIGH images and this new LOW group."
                rev_h, rev_raw, rev_parsed = "", "", None
                try: rev_h, rev_raw, rev_parsed = generation(base_url, args.model, rev_prompt, high_paths, [dataset / x["relative_path"] for x in revision_low if (dataset / x["relative_path"]).is_file()], args.timeout)
                except Exception as exc: rev_raw = repr(exc)
                if rev_h:
                    rec["hypothesis_versions"].append({"version": 2, "source": "original_high_plus_new_revision_low", "hypothesis": rev_h, "raw_response": rev_raw, "parsed_response": rev_parsed})
                    fresh_high = sample_rows(codes, paths, patients, fid, excluded, True, 32, excluded_patients=validation_patients)
                    excluded.update(x["relative_path"] for x in fresh_high); validation_patients |= {x["patient_id"] for x in fresh_high}
                    fresh_low = sample_rows(codes, paths, patients, fid, excluded, False, 32, excluded_patients=validation_patients)
                    rec["validation"].append({"phase": "revalidation", "samples": {"high": fresh_high, "low": fresh_low}, **validate_round(base_url, args.model, dataset, fresh_high, fresh_low, rev_h, args.timeout, 2)})
            rec["final_status"] = "passed_initial" if first.get("pass") else ("revised_and_passed" if rec["validation"][-1].get("pass") else "failed_or_unavailable")
            rec["status"] = "complete"
            records.append(rec)
        with result_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(records[-1], sort_keys=True) + "\n")
    summary = {"schema_version": "1.0", "status": "complete", "cards_requested": len(cards),
               "cards_processed": len(records), "hypotheses_recovered": sum(r.get("status") == "complete" for r in records),
               "feature_start": args.feature_start, "feature_end": args.feature_end,
               "passed_initial": sum(r.get("final_status") == "passed_initial" for r in records),
               "revised": sum(len(r.get("hypothesis_versions", [])) > 1 for r in records),
               "revised_passed": sum(r.get("final_status") == "revised_and_passed" for r in records),
               "failed_or_unavailable": sum(r.get("final_status") not in {"passed_initial", "revised_and_passed"} for r in records),
               "labels_in_prompts": False, "max_revisions_per_feature": 1,
               "checkpoint_sha256": hashlib.sha256(roots["checkpoint"].read_bytes()).hexdigest(),
               "normalization_sha256": hashlib.sha256(roots["normalization"].read_bytes()).hexdigest(),
               "output_jsonl": str(result_path)}
    (out / "recovered_contrastive_feature_validation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
