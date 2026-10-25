"""Prepare and close the OCT input-geometry review before model inference.

The contact sheet is a stratified visual screen, not proof of scanner or
physical-pixel equivalence. A reviewer must inspect it and explicitly record
whether the released full-frame resize is usable for this domain-shift pilot.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from retinavlm_verified_images import ARCHIVE_MD5, EXPERT_IMAGES, sha256, source_parts


DECISION = "reviewed_input_compatible_with_domain_shift"
TILE_WIDTH, TILE_HEIGHT = 360, 240


def load_inputs(manifest: Path, image_audit: Path) -> tuple[list[dict[str, str]], dict]:
    audit = json.loads(image_audit.read_text(encoding="utf-8"))
    if (audit.get("status") != "official_images_matched_and_exact_duplicate_audited"
            or audit.get("source_archive_md5") != ARCHIVE_MD5
            or audit.get("matched_images") != EXPERT_IMAGES
            or audit.get("manifest_sha256") != sha256(manifest)
            or audit.get("near_duplicate_gate") != "reviewed_pass"
            or audit.get("fovea_crop_and_scanner_gate") != "not_run"
            or audit.get("images_with_values_above_255") != 0):
        raise ValueError("Complete official image and duplicate audits are required")
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"source_image_path", "local_image_path", "sha256", "trial",
                "split", "fluid", "width", "height", "mode"}
    if len(rows) != EXPERT_IMAGES or not required.issubset(rows[0] if rows else {}):
        raise ValueError("Official image manifest is incomplete")
    for row in rows:
        source_parts(row["source_image_path"])
        if (row["split"] not in {"train", "validation", "test"}
                or row["fluid"] not in {"0", "1"}
                or int(row["width"]) < 192 or int(row["height"]) < 192
                or row["mode"] not in {"L", "RGB"}):
            raise ValueError(f"Image needs preprocessing review: {row['source_image_path']}")
    return rows, audit


def prepare(manifest: Path, image_audit: Path, contact_sheet: Path,
            packet_path: Path) -> dict:
    if contact_sheet.exists() or packet_path.exists():
        raise FileExistsError("Geometry packet or contact sheet already exists")
    rows, audit = load_inputs(manifest, image_audit)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["trial"], row["split"], row["fluid"])].append(row)
    samples = []
    for group in sorted(groups):
        ordered = sorted(groups[group], key=lambda row: (row["source_image_path"], row["sha256"]))
        for row in (ordered[0], ordered[len(ordered) // 2]):
            if row["source_image_path"] not in {item["source_image_path"] for item in samples}:
                samples.append(row)
    sheet = Image.new("RGB", (4 * TILE_WIDTH, ((len(samples) + 3) // 4) * TILE_HEIGHT), "white")
    draw = ImageDraw.Draw(sheet)
    sample_records = []
    for index, row in enumerate(samples):
        image_path = Path(row["local_image_path"])
        if not image_path.is_file() or sha256(image_path) != row["sha256"]:
            raise ValueError(f"Review image missing or changed: {row['source_image_path']}")
        with Image.open(image_path) as source:
            if (source.mode != row["mode"]
                    or source.size != (int(row["width"]), int(row["height"]))):
                raise ValueError("Review image geometry differs from official manifest")
            model_input = source.convert("L") if source.mode == "RGB" else source
            thumbnail = source.convert("RGB")
            thumbnail.thumbnail((170, 170))
            resized_preview = model_input.resize((192, 192), Image.Resampling.BILINEAR)
            resized_preview = resized_preview.convert("RGB").resize((170, 170))
        x, y = (index % 4) * TILE_WIDTH, (index // 4) * TILE_HEIGHT
        draw.text((x + 4, y + 3), "source", fill="black")
        draw.text((x + 184, y + 3), "192px preview", fill="black")
        sheet.paste(thumbnail, (x + 4 + (170 - thumbnail.width) // 2, y + 20))
        sheet.paste(resized_preview, (x + 184, y + 20))
        draw.text((x + 4, y + 194), f"{row['trial']} {row['split']} fluid={row['fluid']}", fill="black")
        draw.text((x + 4, y + 210), Path(row["source_image_path"]).name, fill="black")
        sample_records.append({key: row[key] for key in ("source_image_path", "sha256", "trial",
                                                         "split", "fluid", "width", "height", "mode")})
    contact_sheet.parent.mkdir(parents=True, exist_ok=True)
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(contact_sheet)
    packet = {
        "status": "geometry_visual_review_required",
        "manifest_sha256": sha256(manifest),
        "pre_review_image_audit_sha256": sha256(image_audit),
        "contact_sheet_sha256": sha256(contact_sheet),
        "source_archive_md5": ARCHIVE_MD5,
        "images": len(rows),
        "dimensions": dict(sorted(Counter(f"{row['width']}x{row['height']}" for row in rows).items())),
        "modes": dict(sorted(Counter(row["mode"] for row in rows).items())),
        "rgb_to_l_policy": "Convert the single RGB OCT to luminance after verifying the unchanged source hash; its channels differ only in the scale marker region",
        "authors_preprocessing": "full-frame scipy zoom to 192x192; integer pixels divided by 255",
        "contact_sheet_preview": "PIL bilinear visual approximation of the authors' SciPy order-1 resize; actual model preprocessing must be checked at runtime",
        "scanner_context": "OLIVES PRIME/TREX Heidelberg Spectralis versus RetinaVLM Topcon training images; physical scale equivalence unverified",
        "samples": sample_records,
    }
    packet_path.write_text(json.dumps(packet, indent=2) + "\n", encoding="utf-8")
    return packet


def finalize(manifest: Path, image_audit: Path, contact_sheet: Path,
             packet_path: Path, review_path: Path) -> dict:
    _, audit = load_inputs(manifest, image_audit)
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    expected_paths = [item["source_image_path"] for item in packet.get("samples", [])]
    if (packet.get("status") != "geometry_visual_review_required"
            or packet.get("manifest_sha256") != sha256(manifest)
            or packet.get("pre_review_image_audit_sha256") != sha256(image_audit)
            or packet.get("contact_sheet_sha256") != sha256(contact_sheet)
            or packet.get("images") != EXPERT_IMAGES
            or not expected_paths
            or review.get("packet_sha256") != sha256(packet_path)
            or review.get("decision") != DECISION
            or review.get("scanner_domain_shift_acknowledged") is not True
            or review.get("representative_images_reviewed") != expected_paths
            or not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip()
            or not isinstance(review.get("notes"), str) or not review["notes"].strip()):
        raise ValueError("Geometry review is incomplete or not bound to the official images")
    audit["fovea_crop_and_scanner_gate"] = DECISION
    audit["geometry_review"] = {
        "packet_sha256": sha256(packet_path),
        "contact_sheet_sha256": sha256(contact_sheet),
        "review_sha256": sha256(review_path),
        "reviewed_images": len(expected_paths),
        "physical_pixel_scale_equivalence": "unverified",
    }
    temporary = image_audit.with_suffix(image_audit.suffix + ".writing")
    temporary.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, image_audit)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "finalize"])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-audit", type=Path, required=True)
    parser.add_argument("--contact-sheet", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--review", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        print(json.dumps(prepare(args.manifest, args.image_audit, args.contact_sheet,
                                 args.packet), indent=2))
    elif args.review is None:
        parser.error("--review is required for finalize")
    else:
        print(json.dumps(finalize(args.manifest, args.image_audit, args.contact_sheet,
                                  args.packet, args.review), indent=2))
