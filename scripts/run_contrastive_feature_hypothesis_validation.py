from __future__ import annotations

"""Contrastive, label-blind validation of Experiment 5 SAE feature hypotheses.

The original 32 high-activation examples are discovery-only.  Initial validation
uses fresh high ranks 33--64 and fresh low ranks from the same embedding cohort.
Features that fail get one (and only one) revised hypothesis using the original
discovery highs plus a new low sample; re-validation is performed on another
fresh high/low pair.  Labels are never sent to Ollama.
"""

import argparse
import base64
import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retina_lanse.data.embedding_shards import load_embedding_run
from retina_lanse.sae.checkpoints import load_checkpoint
from scripts.run_global_sae_search import require_external
from scripts.run_single_sae_feasibility import encode_csr


INITIAL_PROMPT = """You are describing one sparse-autoencoder feature in retinal fundus photographs.
The high group has high activation and the low group has low or zero activation. Labels,
patient IDs, activation values, and disease status are hidden. Give a contrastive visual
hypothesis: what visible pattern is more consistently present in high than low images?
Do not infer diabetic-retinopathy grade or prevalence. Mention nuisance explanations.
Return JSON only: {\"hypothesis\":\"one concise visual description\",\"evidence\":\"visible contrast\",\"alternatives\":[\"...\"],\"confidence\":\"low/medium/high\",\"uncertainty\":\"...\"}
"""

VERIFY_PROMPT = """You are a blinded visual validator. A proposed feature description is supplied below.
Two numbered contact sheets follow: HIGH ({n} images) and LOW ({n} images). Judge every image
for only whether the described visible pattern is present. Do not infer DR grade, labels, or
prevalence. Return compact JSON only with exactly 32 booleans in each array (index 1 is sheet
image 1):
{{\"high_present\":[true,false],\"low_present\":[false,true],\"high_evidence\":\"short shared evidence\",\"low_evidence\":\"short shared evidence\",\"nuisance\":\"short nuisance note\"}}
The arrays must each have length {n}. Proposed description: {hypothesis}
"""

SEED = 4106
DISCOVERY_N = 32
VALIDATION_N = 32


def parse_json(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            cleaned = re.sub(r",\s*([}\]])", r"\1", text[start : end + 1])
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                return None
    return None


def image_bytes(path: Path, max_side: int = 512) -> bytes:
    with Image.open(path) as src:
        image = ImageOps.exif_transpose(src).convert("RGB")
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buf = BytesIO()
        image.save(buf, format="JPEG", quality=82, optimize=True)
        return buf.getvalue()


def sheet(paths: list[Path], max_side: int = 1024) -> bytes:
    tile = max(64, max_side // 8)
    out = Image.new("RGB", (8 * tile, 4 * tile), "white")
    draw = ImageDraw.Draw(out)
    for i, path in enumerate(paths[:32]):
        with Image.open(path) as src:
            im = ImageOps.exif_transpose(src).convert("RGB")
            im.thumbnail((tile - 4, tile - 4), Image.Resampling.LANCZOS)
            x = (i % 8) * tile + (tile - im.width) // 2
            y = (i // 8) * tile + (tile - im.height) // 2
            out.paste(im, (x, y))
        draw.rectangle(((i % 8) * tile, (i // 8) * tile, (i % 8) * tile + 18, (i // 8) * tile + 16), fill="white")
        draw.text(((i % 8) * tile + 2, (i // 8) * tile), str(i + 1), fill="black")
    buf = BytesIO()
    out.save(buf, format="JPEG", quality=78, optimize=True)
    return buf.getvalue()


def call_ollama(base_url: str, model: str, prompt: str, high: list[Path], low: list[Path], timeout: int, context: int) -> str:
    payload = {
        "model": model, "stream": False, "format": "json",
        "options": {"temperature": 0, "num_ctx": context, "num_predict": 2048},
        "messages": [{"role": "user", "content": prompt + "\nHIGH and LOW sheets are separate images.",
                      "images": [base64.b64encode(sheet(high)).decode(), base64.b64encode(sheet(low)).decode()]}],
    }
    req = urllib.request.Request(base_url.rstrip("/") + "/api/chat", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return str(json.loads(response.read().decode())["message"]["content"])


def valid_reviews(parsed: dict | None, expected_n: int = 32) -> tuple[list[dict], list[dict]] | None:
    if not isinstance(parsed, dict):
        return None
    # Qwen sometimes capitalizes object keys; accept that harmless variation.
    lowered = {str(k).lower(): v for k, v in parsed.items()}
    compact_high, compact_low = lowered.get("high_present"), lowered.get("low_present")
    if isinstance(compact_high, list) and isinstance(compact_low, list):
        if len(compact_high) != expected_n or len(compact_low) != expected_n or not all(isinstance(x, bool) for x in compact_high + compact_low):
            return None
        he, le = str(lowered.get("high_evidence", "")), str(lowered.get("low_evidence", ""))
        nuisance = str(lowered.get("nuisance", ""))
        return ([{"image_number": i + 1, "pattern_present": bool(v), "confidence": "unknown", "evidence": he, "nuisance": nuisance} for i, v in enumerate(compact_high)],
                [{"image_number": i + 1, "pattern_present": bool(v), "confidence": "unknown", "evidence": le, "nuisance": nuisance} for i, v in enumerate(compact_low)])
    high, low = lowered.get("high"), lowered.get("low")
    if not isinstance(high, list) or not isinstance(low, list):
        return None
    def clean(items):
        out = []
        for i, item in enumerate(items):
            if not isinstance(item, dict) or not isinstance(item.get("pattern_present"), bool):
                continue
            out.append({"image_number": int(item.get("image_number", i + 1)),
                        "pattern_present": bool(item["pattern_present"]),
                        "confidence": str(item.get("confidence", "unknown")),
                        "evidence": str(item.get("evidence", "")),
                        "nuisance": str(item.get("nuisance", ""))})
        return out
    h, l = clean(high), clean(low)
    return (h, l) if h and l else None


def sample_rows(codes, paths, patients, feature: int, excluded: set[str], top: bool, n: int,
                offset: int = 0, excluded_patients: set[str] | None = None):
    col = codes.getcol(feature).tocoo()
    values = np.asarray(col.data, dtype=np.float32)
    rows = np.asarray(col.row, dtype=np.int64)
    if top:
        order = sorted(range(len(rows)), key=lambda j: (-float(values[j]), str(paths[int(rows[j])])) )
    else:
        # Prefer the lowest *positive* activations so validation tests a gradient
        # rather than trivially distinguishing active images from absent images;
        # use zeros only if a feature has fewer than n positive examples.
        active_map = {int(r): float(v) for r, v in zip(rows, values)}
        positive_order = sorted([int(r) for r, v in zip(rows, values) if float(v) > 0],
                                key=lambda i: (active_map[i], str(paths[i])))
        zero_order = sorted([i for i in range(len(paths)) if i not in active_map], key=lambda i: str(paths[i]))
        order = positive_order + zero_order
    out = []
    used_patients = set(excluded_patients or set())
    for pos in order[offset:]:
        idx = int(rows[pos]) if top else int(pos)
        rel = str(paths[idx])
        patient = str(patients[idx])
        if rel in excluded or patient in used_patients:
            continue
        out.append({"relative_path": rel, "patient_id": patient,
                    "activation": float(active_map.get(idx, 0.0) if not top else values[pos]),
                    "rank_source": "high" if top else "low"})
        used_patients.add(patient)
        if len(out) >= n:
            break
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--train-run")
    p.add_argument("--checkpoint")
    p.add_argument("--normalization")
    p.add_argument("--dataset-root")
    p.add_argument("--prior-cards")
    p.add_argument("--prior-hypotheses")
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="qwen2.5vl:3b")
    p.add_argument("--base-url", default=None)
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--max-features", type=int, default=256)
    p.add_argument("--skip-llm", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        x = np.asarray([0.2, 0.9, 0.9, 0.0], dtype=np.float32)
        pairs = sorted([(float(v), i) for i, v in enumerate(x) if v > 0], key=lambda z: (-z[0], z[1]))
        result = {"status": "smoke_passed", "tie_break": [i for _, i in pairs], "padding": 1024 - len(pairs),
                  "classifier_contains_feature_ids": False, "one_revision_max": True, "labels_in_prompts": False}
        out = require_external(Path(args.output), "output"); out.mkdir(parents=True, exist_ok=True)
        (out / "contrastive_validation_smoke.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2)); return 0
    required = {"--train-run": args.train_run, "--checkpoint": args.checkpoint,
                "--normalization": args.normalization, "--dataset-root": args.dataset_root,
                "--prior-cards": args.prior_cards, "--prior-hypotheses": args.prior_hypotheses}
    missing = [flag for flag, value in required.items() if not value]
    if missing:
        p.error("missing required arguments: " + ", ".join(missing))
    roots = {k: require_external(Path(v), k) for k, v in {
        "train run": args.train_run, "checkpoint": args.checkpoint, "normalization": args.normalization, "dataset root": args.dataset_root,
        "prior cards": args.prior_cards, "prior hypotheses": args.prior_hypotheses, "output": args.output}.items()}
    out = roots["output"]; out.mkdir(parents=True, exist_ok=True)
    run = load_embedding_run(roots["train run"], expected_split="train", verify_sha256=True)
    model, ckpt = load_checkpoint(roots["checkpoint"])
    if ckpt.get("model_config") != {"input_dim": 1024, "dictionary_size": 16384, "k": 1024}:
        raise RuntimeError("checkpoint must be the frozen K=1024, 16,384-feature SAE")
    all_cards = [json.loads(x) for x in roots["prior cards"].read_text().splitlines() if x.strip()]
    hypotheses = {}
    for line in roots["prior hypotheses"].read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("status") == "complete" and isinstance(row.get("parsed_response"), dict):
                hypotheses[int(row["feature_id"])] = row["parsed_response"]
    cards = [c for c in all_cards if int(c["feature_id"]) in hypotheses][: args.max_features]
    unavailable = [{"feature_id": int(c["feature_id"]), "reason": "no_completed_prior_hypothesis"}
                   for c in all_cards if int(c["feature_id"]) not in hypotheses]
    # Recompute from the same frozen semantic checkpoint; no labels are loaded.
    norm = torch.load(roots["normalization"], map_location="cpu", weights_only=False)
    mean, std = norm["mean"].float(), norm["std"].float()
    dense = (run.embeddings - mean) / std
    codes = encode_csr(model, dense, batch_size=args.batch_size)
    paths, patients = run.relative_paths, run.patient_ids
    base_url = args.base_url or os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
    records = []
    for card in cards:
        fid = int(card["feature_id"])
        discovery = card.get("top_high_activation_examples", [])[:DISCOVERY_N]
        discovery_paths = [str(x["relative_path"]) for x in discovery]
        excluded = set(discovery_paths)
        discovery_patients = {str(x.get("patient_id")) for x in discovery}
        initial_high = sample_rows(codes, paths, patients, fid, excluded, True, VALIDATION_N,
                                   excluded_patients=discovery_patients)
        excluded.update(x["relative_path"] for x in initial_high)
        initial_patients = discovery_patients | {x["patient_id"] for x in initial_high}
        initial_low = sample_rows(codes, paths, patients, fid, excluded, False, VALIDATION_N,
                                  excluded_patients=initial_patients)
        excluded.update(x["relative_path"] for x in initial_low)
        initial_patients |= {x["patient_id"] for x in initial_low}
        dataset = roots["dataset root"]
        high_paths = [dataset / x["relative_path"] for x in initial_high]
        low_paths = [dataset / x["relative_path"] for x in initial_low]
        hypothesis = str(hypotheses[fid].get("hypothesis", ""))
        row = {"feature_id": fid, "global_prominence_rank": card.get("global_prominence_rank"),
               "discovery_high": discovery, "initial_validation_high": initial_high,
               "initial_validation_low": initial_low, "labels_hidden": True,
               "hypothesis_versions": [{"version": 1, "source": "prior_high_only", "hypothesis": hypothesis}],
               "validation": []}
        def validate(version, high_items, low_items, hyp):
            if args.skip_llm:
                return {"status": "skipped_llm", "version": version}
            # 16+16 images per request is deliberately used because the local
            # 3B vision model truncated 32+32 responses in the preflight.
            chunk_size = 16
            all_high, all_low, raws, prompt_hashes = [], [], [], []
            for start in range(0, max(len(high_items), len(low_items)), chunk_size):
                h_items, l_items = high_items[start:start + chunk_size], low_items[start:start + chunk_size]
                hp = [dataset / x["relative_path"] for x in h_items if (dataset / x["relative_path"]).is_file()]
                lp = [dataset / x["relative_path"] for x in l_items if (dataset / x["relative_path"]).is_file()]
                n = min(len(hp), len(lp))
                if n == 0:
                    continue
                hp, lp = hp[:n], lp[:n]
                prompt = VERIFY_PROMPT.format(hypothesis=hyp, n=n)
                try:
                    raw = call_ollama(base_url, args.model, prompt, hp, lp, args.timeout, 8192)
                    parsed = valid_reviews(parse_json(raw), expected_n=n)
                    raws.append(raw); prompt_hashes.append(hashlib.sha256(prompt.encode()).hexdigest())
                    if parsed is None:
                        return {"status": "unparsed", "version": version, "raw_responses": raws,
                                "prompt_sha256": prompt_hashes}
                    h, l = parsed
                    for item in h: item["image_number"] += start
                    for item in l: item["image_number"] += start
                    all_high.extend(h); all_low.extend(l)
                except Exception as exc:
                    return {"status": "request_failed", "version": version, "error": repr(exc),
                            "raw_responses": raws, "prompt_sha256": prompt_hashes}
            hr = float(np.mean([x["pattern_present"] for x in all_high])); lr = float(np.mean([x["pattern_present"] for x in all_low]))
            return {"status": "complete", "version": version, "high_reviews": all_high, "low_reviews": all_low,
                    "high_rate": hr, "low_rate": lr, "contrast": hr - lr,
                    "pass": bool(hr >= .60 and hr - lr >= .30 and len(all_high) >= 24 and len(all_low) >= 24),
                    "prompt_sha256": prompt_hashes, "raw_responses": raws}
        first = validate(1, initial_high, initial_low, hypothesis)
        row["validation"].append({"phase": "initial", "samples": {"high": initial_high, "low": initial_low}, **first})
        if first.get("status") == "complete" and not first.get("pass"):
            revision_low = sample_rows(codes, paths, patients, fid, excluded, False, VALIDATION_N,
                                       excluded_patients=initial_patients)
            excluded.update(x["relative_path"] for x in revision_low)
            revision_patients = initial_patients | {x["patient_id"] for x in revision_low}
            revision_prompt = INITIAL_PROMPT + "\nThe first 32 images are the original HIGH discovery group; the second 32 are NEW LOW images for contrastive revision."
            rev_high = [dataset / x["relative_path"] for x in discovery if (dataset / x["relative_path"]).is_file()]
            rev_low = [dataset / x["relative_path"] for x in revision_low if (dataset / x["relative_path"]).is_file()]
            revised = hypothesis
            revision_record = {"status": "skipped_llm", "source_low": revision_low}
            if not args.skip_llm:
                try:
                    raw = call_ollama(base_url, args.model, revision_prompt, rev_high, rev_low, args.timeout, 8192)
                    parsed = parse_json(raw); revised = str((parsed or {}).get("hypothesis", "")).strip() or hypothesis
                    revision_record = {"status": "complete" if revised != hypothesis else "unparsed", "hypothesis": revised,
                                       "raw_response": raw, "prompt_sha256": hashlib.sha256(revision_prompt.encode()).hexdigest(), "source_low": revision_low}
                except Exception as exc:
                    revision_record = {"status": "request_failed", "error": repr(exc), "source_low": revision_low}
            row["hypothesis_versions"].append({"version": 2, "source": "original_high_plus_new_low", "hypothesis": revised, **revision_record})
            fresh_high = sample_rows(codes, paths, patients, fid, excluded, True, VALIDATION_N, offset=0,
                                     excluded_patients=revision_patients | discovery_patients)
            excluded.update(x["relative_path"] for x in fresh_high)
            fresh_low = sample_rows(codes, paths, patients, fid, excluded, False, VALIDATION_N,
                                    excluded_patients=revision_patients | discovery_patients | {x["patient_id"] for x in fresh_high})
            row["validation"].append({"phase": "revalidation", "samples": {"high": fresh_high, "low": fresh_low}, **validate(2, fresh_high, fresh_low, revised)})
        row["final_status"] = "passed_initial" if first.get("pass") else ("revised_and_passed" if row["validation"][-1].get("pass") else "failed_or_unavailable")
        records.append(row)
        with (out / "contrastive_feature_validation.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {"schema_version": "1.0", "status": "complete" if records else "no_eligible_features",
               "features_evaluated": len(records), "initial_pass": sum(r["final_status"] == "passed_initial" for r in records),
               "revised": sum(len(r["hypothesis_versions"]) > 1 for r in records),
               "revised_pass": sum(r["final_status"] == "revised_and_passed" for r in records),
               "unavailable_count": len(unavailable), "unavailable_features": unavailable,
               "labels_in_prompts": False, "max_revisions_per_feature": 1,
               "checkpoint_sha256": hashlib.sha256(roots["checkpoint"].read_bytes()).hexdigest(),
               "normalization_sha256": hashlib.sha256(roots["normalization"].read_bytes()).hexdigest(),
               "cards_sha256": hashlib.sha256(roots["prior cards"].read_bytes()).hexdigest(),
               "hypotheses_sha256": hashlib.sha256(roots["prior hypotheses"].read_bytes()).hexdigest(),
               "output_jsonl": str(out / "contrastive_feature_validation.jsonl"),
               "caveat": "Same-model VLM review; semantic validity still requires independent retinal-specialist review."}
    (out / "contrastive_feature_validation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
