"""Verify checkpoint, tokenizer, and fixed data before CUDA model work."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

from .protocol import PROMPT, audit_manifest, manifest_sha256, read_jsonl

SUPPORT = (
    "added_tokens.json", "config.json", "configuration_intern_vit.py",
    "configuration_internlm2.py", "configuration_internvl_chat.py",
    "conversation.py", "generation_config.json", "model.safetensors.index.json",
    "modeling_intern_vit.py", "modeling_internlm2.py",
    "modeling_internvl_chat.py", "special_tokens_map.json",
    "tokenization_internlm2.py", "tokenization_internlm2_fast.py",
    "tokenizer.model", "tokenizer_config.json",
)
READOUT_PREFIX = '{"hard_exudates":"'


def run(checkpoint: Path, manifest: Path, data_audit: Path) -> dict:
    from transformers import AutoConfig, AutoTokenizer

    missing = [name for name in SUPPORT if not (checkpoint / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint support files: {missing}")
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    expected = index["weight_map"]
    shards = sorted(set(expected.values()))
    actual: dict[str, str] = {}
    shard_info = {}
    for name in shards:
        path = checkpoint / name
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("rb") as handle:
            header_bytes = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_bytes))
        for tensor, info in header.items():
            if tensor == "__metadata__":
                continue
            if tensor in actual:
                raise ValueError(f"Tensor appears in multiple shards: {tensor}")
            actual[tensor] = name
        extent = max((info["data_offsets"][1] for key, info in header.items()
                      if key != "__metadata__"), default=0)
        if extent != path.stat().st_size - 8 - header_bytes:
            raise ValueError(f"Safetensors extent mismatch: {name}")
        shard_info[name] = {"bytes": path.stat().st_size,
                            "tensors": sum(k != "__metadata__" for k in header)}
    if actual != expected:
        raise ValueError("Shard tensor names do not match model.safetensors.index.json")

    config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True,
                                              use_fast=False, local_files_only=True)
    prefix_ids = tokenizer.encode(READOUT_PREFIX, add_special_tokens=False)
    term_ids = tokenizer.encode("present", add_special_tokens=False)
    joined = tokenizer.encode(READOUT_PREFIX + "present", add_special_tokens=False)
    if len(term_ids) != 1 or joined != prefix_ids + term_ids:
        raise ValueError("The fixed future status readout is not one token")
    image_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    if image_token_id is None or image_token_id == tokenizer.unk_token_id:
        raise ValueError("Missing FundusExpert image context token")

    rows = read_jsonl(manifest)
    data_summary = audit_manifest(rows)
    audit = json.loads(data_audit.read_text())
    if data_summary["images"] != audit["images"] or audit["original_ex_positive"] != 486:
        raise ValueError("Data manifest and audit disagree")
    return {
        "static_preflight_pass": True,
        "checkpoint_tensors": len(actual),
        "checkpoint_shards": shard_info,
        "config": {"model_type": config.model_type, "template": config.template,
                   "image_size": config.force_image_size,
                   "max_dynamic_patch": config.max_dynamic_patch,
                   "language_layers": config.llm_config.num_hidden_layers,
                   "language_hidden_size": config.llm_config.hidden_size},
        "tokenizer": {"class": type(tokenizer).__name__, "vocab_size": len(tokenizer),
                      "image_context_token_id": image_token_id,
                      "readout_prefix": READOUT_PREFIX, "readout_term": "present",
                      "readout_token_id": term_ids[0],
                      "exudate_token_count": len(tokenizer.encode(" exudate", add_special_tokens=False))},
        "data": data_summary,
        "manifest_sha256": manifest_sha256(manifest),
        "data_audit_sha256": hashlib.sha256(data_audit.read_bytes()).hexdigest(),
        "prompt": PROMPT,
        "cuda_model_preflight_run": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.checkpoint, args.manifest, args.data_audit)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
