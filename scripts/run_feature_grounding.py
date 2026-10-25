from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_global_sae_search import require_external


PROMPT = """You are reviewing retinal fundus images to describe one sparse-autoencoder feature.
The images were selected because the same SAE feature has a high activation. The images may
contain diabetic retinopathy or may be normal; the clinical labels are intentionally hidden.
Describe the most specific visible visual pattern shared by all or most images. Do not infer a
DR grade, do not use disease prevalence, and do not claim a lesion that is not visibly supported.
If the images are heterogeneous, say so and provide the best alternative explanation.

Return JSON only with these fields:
{
  "hypothesis": "one concise visual description",
  "anatomical_region": "retina/vessel/optic-disc/macula/global/uncertain",
  "evidence_strength": "high/medium/low",
  "alternative_explanations": ["..."],
  "nuisance_flags": ["blur/brightness/camera/border/compression/none/other"],
  "uncertain": true
}
"""


def image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def call_openai(base_url: str, api_key: str, model: str, prompt: str, images: list[Path], timeout: int) -> str:
    content: list[dict] = [{"type": "text", "text": prompt}]
    content.extend({"type": "image_url", "image_url": {"url": image_data_url(path)}} for path in images)
    payload = {"model": model, "temperature": 0, "messages": [{"role": "user", "content": content}]}
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return str(body["choices"][0]["message"]["content"])


def call_ollama(base_url: str, model: str, prompt: str, images: list[Path], timeout: int, context_length: int) -> str:
    payload = {
        "model": model,
        "stream": False,
        "options": {"temperature": 0, "num_ctx": context_length},
        "messages": [{"role": "user", "content": prompt, "images": [
            base64.b64encode(path.read_bytes()).decode("ascii") for path in images
        ]}],
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return str(body["message"]["content"])


def parse_json_response(text: str) -> dict | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Ground retinal SAE feature cards with a vision-language model.")
    parser.add_argument("--atlas", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--provider", choices=("openai", "ollama"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--max-features", type=int, default=32)
    parser.add_argument("--max-examples", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--context-length", type=int, default=32768)
    args = parser.parse_args()

    atlas_path = require_external(Path(args.atlas), "feature atlas")
    dataset_root = require_external(Path(args.dataset_root), "retinal dataset")
    output = require_external(Path(args.output), "feature grounding output")
    output.mkdir(parents=True, exist_ok=True)
    if args.provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for provider=openai")
        base_url = args.base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    else:
        api_key = None
        base_url = args.base_url or os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")

    records = [json.loads(line) for line in atlas_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = records[: args.max_features]
    output_jsonl = output / "feature_grounding.jsonl"
    results = []
    for record in records:
        examples = []
        for item in record["top_high_activation_examples"]:
            path = dataset_root / item["relative_path"]
            if path.is_file():
                examples.append(path)
            if len(examples) >= args.max_examples:
                break
        if not examples:
            results.append({"feature_id": record["feature_id"], "status": "no_images_found"})
            continue
        prompt = PROMPT + f"\nFeature ID (do not interpret as a clinical label): {record['feature_id']}\n"
        try:
            if args.provider == "openai":
                response = call_openai(base_url, api_key, args.model, prompt, examples, args.timeout)
            else:
                response = call_ollama(base_url, args.model, prompt, examples, args.timeout, args.context_length)
            parsed_response = parse_json_response(response)
            result = {
                "feature_id": record["feature_id"],
                "status": "complete" if parsed_response is not None else "unparsed_response",
                "example_paths": [str(path) for path in examples],
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "raw_response": response,
                "parsed_response": parsed_response,
            }
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            result = {"feature_id": record["feature_id"], "status": "request_failed", "error": repr(exc)}
        results.append(result)

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, sort_keys=True) + "\n")
    summary = {
        "schema_version": "1.0",
        "status": "complete_language_grounding" if all(item.get("status") == "complete" for item in results) else "partial_language_grounding",
        "provider": args.provider,
        "model": args.model,
        "features_requested": len(records),
        "features_completed": sum(item.get("status") == "complete" for item in results),
        "features_unparsed": sum(item.get("status") == "unparsed_response" for item in results),
        "max_examples_per_feature": args.max_examples,
        "labels_hidden_from_prompt": True,
        "output_jsonl": str(output_jsonl),
    }
    (output / "feature_grounding_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
