"""Run one released RetinaVLM Specialist inference on a verified OLIVES image.

This is the first real-model feasibility gate, not a benchmark. Use a validation
image from the official matched-image manifest; the test split is forbidden.
Requires the authors' dependencies, approved Llama 3 access, and CUDA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from retinavlm_released_model import load_released_specialist
from retinavlm_verified_images import load_verified_manifest, source_parts


EXPECTED_ARCHIVE_MD5 = "e08f4198aa919d6900bd30525c137933"
QUESTIONS = {
    "fluid": "Is intraretinal or subretinal fluid present in this OCT scan? Answer present or absent.",
    "IRF": "Is intraretinal fluid present in this OCT scan? Answer present or absent.",
}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            result.update(block)
    return result.hexdigest()


def select_verified_validation_image(manifest: Path, audit_path: Path,
                                     image_root: Path, finding: str) -> tuple[dict[str, str], Path]:
    rows = load_verified_manifest(manifest, audit_path)
    candidates = sorted((row for row in rows if row["split"] == "validation"),
                        key=lambda row: row["source_image_path"])
    if not candidates:
        raise ValueError("No validation image in official manifest")
    row = candidates[0]
    image_path = image_root.joinpath(*source_parts(row["source_image_path"]))
    if not image_path.is_file() or digest(image_path) != row["sha256"]:
        raise ValueError("Selected validation image is missing or has changed")
    if row[finding] not in {"0", "1"}:
        raise ValueError("Selected validation label is not binary")
    return row, image_path


def run(model_dir: Path, authors_code: Path, manifest: Path, audit_path: Path,
        image_root: Path, finding: str,
        out: Path) -> dict:
    row, image_path = select_verified_validation_image(manifest, audit_path, image_root, finding)
    import torch
    from PIL import Image

    with Image.open(image_path) as image:
        image_mode, image_size = image.mode, image.size
        if image.mode != "L":
            raise ValueError(f"Image mode {image.mode!r} needs a documented preprocessing decision")
        image = image.copy()

    from retinavlm_binary_answer import ANSWER_FORMS, answer_token_ids, post_image_segment, predict_answer
    from retinavlm_pre_answer import pre_answer_forward

    model, _ = load_released_specialist(model_dir, authors_code)

    tensor = model.convert_any_image_to_normalized_tensor(image).unsqueeze(0).to(model.model.device)
    samples = {"Image": tensor, "Question": [QUESTIONS[finding]], "Answer": [""]}
    with torch.inference_mode():
        logits, _ = pre_answer_forward(model.model, samples)
        tokenizer = model.model.get_tokenizer()
        # The authors tokenize this suffix separately in prompt_wrap().
        absent_id, present_id = answer_token_ids(
            tokenizer, post_image_segment(samples["Input"][0])
        )
        answer, probability = predict_answer(logits, absent_id, present_id)

    result = {
        "status": "released_model_one_verified_validation_image_inference",
        "model_dir": str(model_dir),
        "source_image_path": row["source_image_path"],
        "local_image_path": str(image_path),
        "image_sha256": row["sha256"],
        "official_archive_md5": EXPECTED_ARCHIVE_MD5,
        "image_mode": image_mode,
        "image_size": image_size,
        "question": QUESTIONS[finding],
        "answer_token_ids": {"absent": absent_id, "present": present_id},
        "answer_token_forms": {"absent": ANSWER_FORMS[0], "present": ANSWER_FORMS[1]},
        "predicted_answer": "present" if answer.item() else "absent",
        "conditional_present_probability": probability.item(),
        "finding": finding,
        "expert_label": int(row[finding]),
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "benchmark_result": False,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--authors-code", type=Path, required=True)
    parser.add_argument("--official-image-manifest", type=Path, required=True)
    parser.add_argument("--image-audit", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--finding", choices=tuple(QUESTIONS), default="fluid")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.model_dir, args.authors_code,
                         args.official_image_manifest, args.image_audit,
                         args.image_root, args.finding, args.out), indent=2))
