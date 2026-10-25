"""Capture the authors' greedy text generation on two fixed validation pairs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from retinavlm_released_model import load_released_specialist


QUESTIONS = {
    "binary": "Is intraretinal fluid present in this OCT scan? Answer present or absent.",
    "report": (
        "Write an extensive report describing the OCT image and listing any present "
        "biomarkers or other observations. Do not provide a disease stage, or referral recommendation yet."
    ),
}


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def run(gate_dir: Path, model_dir: Path, authors_code: Path, out: Path) -> dict:
    import torch
    from PIL import Image

    receipt = json.loads((gate_dir / "selection_receipt.json").read_text(encoding="utf-8"))
    if receipt["status"] != "fixed_irf_validation_gate_13_patient_pairs":
        raise ValueError("Unexpected selection receipt")
    with (gate_dir / "selected_manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    patients = sorted({row["patient_key"] for row in rows})[:2]
    chosen = [row for row in rows if row["patient_key"] in patients]
    if len(chosen) != 4 or {row["IRF"] for row in chosen} != {"0", "1"}:
        raise ValueError("Expected two validation patients with both IRF labels")
    for row in chosen:
        path = gate_dir / row["image_archive_path"]
        if not path.is_file() or sha256(path) != row["sha256"]:
            raise ValueError(f"Changed or missing verified image: {path}")

    model, _ = load_released_specialist(model_dir, authors_code)
    generations = []
    for index, row in enumerate(chosen, start=1):
        with Image.open(gate_dir / row["image_archive_path"]) as image:
            if image.mode != "L":
                raise ValueError("Expected grayscale OCT")
            image = image.copy()
        tensor = model.convert_any_image_to_normalized_tensor(image).unsqueeze(0).to(model.model.device)
        with torch.inference_mode():
            for kind, question in QUESTIONS.items():
                answer = model.model.query(
                    tensor, [question], answer_preamble=[""],
                    max_new_tokens=120 if kind == "report" else 40,
                    output_only=True,
                )[0]
                generations.append({
                    "source_image_path": row["source_image_path"],
                    "patient_key": row["patient_key"],
                    "expert_IRF": int(row["IRF"]),
                    "question_type": kind,
                    "question": question,
                    "generated_text": answer,
                })
        print(f"Generated on {index}/{len(chosen)} validation scans", flush=True)
    result = {
        "status": "retinavlm_irf_authors_generation_diagnostic_complete",
        "benchmark_result": False,
        "selection_receipt": receipt,
        "gpu_name": torch.cuda.get_device_name(0),
        "patients": len(patients),
        "images": len(chosen),
        "generations": generations,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--authors-code", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.gate_dir, args.model_dir, args.authors_code, args.out)
    print(json.dumps(result["generations"], indent=2))
