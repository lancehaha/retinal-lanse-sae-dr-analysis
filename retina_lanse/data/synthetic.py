from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from retina_lanse.data.splits import assert_no_patient_leakage, patient_level_splits


def _make_fundus(size: int, rng: np.random.Generator, image_index: int) -> Image.Image:
    image = Image.new("RGB", (size, size), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    margin = int(size * rng.uniform(0.05, 0.10))
    base = (int(rng.integers(110, 165)), int(rng.integers(35, 65)), int(rng.integers(25, 50)))
    draw.ellipse((margin, margin, size - margin, size - margin), fill=base)
    cx, cy = int(size * rng.uniform(0.46, 0.56)), int(size * rng.uniform(0.46, 0.54))
    disc_x = int(size * rng.uniform(0.65, 0.76))
    disc_y = int(size * rng.uniform(0.43, 0.57))
    draw.ellipse((disc_x - 5, disc_y - 6, disc_x + 5, disc_y + 6), fill=(230, 174, 98))
    for _ in range(10):
        end_x = int(size * rng.uniform(0.12, 0.88))
        end_y = int(size * rng.uniform(0.12, 0.88))
        draw.line((disc_x, disc_y, cx, cy, end_x, end_y), fill=(75, 20, 22), width=int(rng.integers(1, 3)))
    if image_index % 3 == 0:
        for _ in range(5):
            x, y = int(size * rng.uniform(0.25, 0.75)), int(size * rng.uniform(0.25, 0.75))
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(245, 205, 96))
    if image_index % 4 == 0:
        for _ in range(4):
            x, y = int(size * rng.uniform(0.25, 0.75)), int(size * rng.uniform(0.25, 0.75))
            draw.ellipse((x - 2, y - 1, x + 3, y + 2), fill=(90, 10, 18))
    noise = rng.normal(0, 3.0, (size, size, 3))
    array = np.clip(np.asarray(image, dtype=np.float32) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(array).filter(ImageFilter.GaussianBlur(radius=0.35))


def generate_dataset(root: str | Path, num_patients: int, images_per_patient: int, image_size: int,
                     fractions: tuple[float, float, float], seed: int) -> list[dict]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    patient_ids = [f"synthetic-patient-{i:03d}" for i in range(num_patients)]
    assignments = patient_level_splits(patient_ids, fractions, seed)
    records: list[dict] = []
    for patient_index, patient_id in enumerate(patient_ids):
        for eye_index in range(images_per_patient):
            image_id = f"{patient_id}-eye-{eye_index}"
            relative = Path(assignments[patient_id]) / f"{image_id}.png"
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            rng = np.random.default_rng(seed + patient_index * 101 + eye_index)
            _make_fundus(image_size, rng, patient_index + eye_index).save(target)
            records.append({
                "image_id": image_id,
                "patient_id": patient_id,
                "split": assignments[patient_id],
                "dataset": "synthetic_fundus_smoke_only",
                "camera": "synthetic",
                "path": str(target),
            })
    assert_no_patient_leakage(records)
    manifest = root / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records), encoding="utf-8")
    return records

