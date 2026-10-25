"""Check single-token finding/answer words with RetinaVLM's exact Llama 3 tokenizer.

Requires approved access to meta-llama/Meta-Llama-3-8B-Instruct. Run on the
machine that will execute RetinaVLM; no model weights or OCT images are needed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
QUESTION = (
    "Is intraretinal or subretinal fluid present in this OCT scan? "
    "Answer present or absent."
)


def continuation_tokens(tokenizer, prefix: str, continuation: str) -> list[int]:
    """Return token IDs introduced by a continuation after the exact prefix."""
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    combined = tokenizer.encode(prefix + continuation, add_special_tokens=False)
    # Boundary merges can change the end of prefix tokenization. Mark such a
    # result invalid instead of silently treating it as a one-token target.
    if combined[: len(prefix_ids)] != prefix_ids:
        raise ValueError(f"Tokenization boundary changed for {continuation!r}")
    return combined[len(prefix_ids) :]


def run(output: Path) -> dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
    messages = [
        {
            "role": "system",
            "content": "You are a helpful ophthalmological specialist chatbot capable of interpreting retinal OCT images.",
        },
        {
            "role": "user",
            "content": f"Here is an encoding of a retinal OCT image.\n{QUESTION}",
        },
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_prompt = prompt.replace(
        "encoding of a retinal OCT image",
        "encoding of a retinal OCT image <Img><ImageHere></Img>",
        1,
    )
    prompt_parts = image_prompt.split("<ImageHere>")
    if len(prompt_parts) != 2:
        raise ValueError("Author prompt does not have exactly one image marker")
    post_image_prompt = prompt_parts[1]
    variants = {}
    for concept in ("fluid", "present", "absent", "drusen", "cyst"):
        variants[concept] = {}
        for key, surface in (("start", concept), ("with_leading_space", " " + concept)):
            ids = tokenizer.encode(surface, add_special_tokens=False)
            variants[concept][key] = {
                "ids": ids,
                "tokens": tokenizer.convert_ids_to_tokens(ids),
                "single_token": len(ids) == 1,
            }
        try:
            ids = continuation_tokens(tokenizer, post_image_prompt, concept)
            variants[concept]["answer_start"] = {
                "ids": ids,
                "tokens": tokenizer.convert_ids_to_tokens(ids),
                "single_token": len(ids) == 1,
            }
        except ValueError as error:
            variants[concept]["answer_start"] = {"error": str(error), "single_token": False}
    result = {
        "model_id": MODEL_ID,
        "tokenizer_class": type(tokenizer).__name__,
        "question": QUESTION,
        "text_token_count_excluding_image_embeddings": (
            len(tokenizer.encode(prompt_parts[0], add_special_tokens=False))
            + len(tokenizer.encode(post_image_prompt, add_special_tokens=False))
        ),
        "continuation_context": "author_prompt_post_image_segment",
        "variants": variants,
        "status": "tokenization_only_no_model_run",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))
