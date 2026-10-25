from __future__ import annotations

"""Experiment 10: blinded image + patch + unvalidated SAE-card Qwen pilot.

The experiment is intentionally an exploratory reuse of the Experiment 8 pilot
holdout.  It does not train RETFound, the Patch-SAE, feature cards, or a
numerical prediction head.  Four prompt arms are paired on exactly the same
holdout images and are checkpointed response-by-response.
"""

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from sklearn.metrics import average_precision_score, cohen_kappa_score, f1_score, roc_auc_score

from retina_lanse.data.preprocessing import preprocess_global
from scripts.run_global_sae_search import require_external


EXTERNAL = Path("/Volumes/Elements/Retina-LanSE")
EXP8 = EXTERNAL / "experiments/experiment_8_patch_sae_grounded_report_pilot_20260913"
TOP64 = EXP8 / "topk_64"
DEFAULT_OUTPUT = EXTERNAL / "experiments/experiment_10b_scalar_risk_prompt_amendment_20260917"
DEFAULT_CARD_ONLY_OUTPUT = EXTERNAL / "experiments/experiment_10c_image_card_only_ablation_20260917"
GRID, PATCHES, TOP_K = 14, 196, 64
SEED, SHUFFLE_SEED, BOOTSTRAPS = 4106, 92010, 2000
ARMS = ("image_only", "image_patches", "image_patches_real_cards", "image_patches_shuffled_cards")
CARD_ONLY_ARM = "image_real_cards_no_patches"


BASE_PROMPT = """Assess one retinal photograph for exploratory research. The reference grade is hidden.
Be cautious: supplied SAE cards are unvalidated and are not lesions or diagnoses.
Return a JSON object only. It must have these keys: predicted_grade (an integer from 0 to 4),
referable_risk_0_to_100 (an integer from 0 to 100), visible_findings (a list of short strings),
uncertainty (low, medium, or high), and image_quality (adequate, limited, or poor).
Do not copy an example, and do not output a probability array. Higher referable_risk_0_to_100 means stronger
visual evidence for grade 2 or higher. This is a research estimate, not a diagnosis."""

PATCH_PROMPT = """
Image 1 is the full retina. Image 2 is eight local patches labeled "number: row,column" in a 14x14 grid.
They are additional views of the same image."""

CARD_PROMPT = """
Below, each event has unvalidated card text and a training-only association. Treat these as uncertain context only.
"""


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def image_key(value: str) -> str:
    return Path(value).stem.lower()


def require_input(path: Path, label: str) -> Path:
    path = require_external(path, label)
    if not path.is_file():
        raise RuntimeError(f"missing {label}: {path}")
    return path


def output_root(value: str | Path) -> Path:
    root = require_external(Path(value), "output")
    if root == EXTERNAL or root == EXP8 or root == TOP64:
        raise RuntimeError("Experiment 10 output must be a new child directory under the external experiment root")
    return root


def load_eligible_cards(path: Path) -> dict[int, dict]:
    """Keep only cards with a completed (but failed) contrastive judgment.

    None of these cards passed semantic validation.  The 30 unavailable records
    had malformed validation responses and are intentionally excluded instead of
    silently being promoted to evidence.
    """
    cards: dict[int, dict] = {}
    for row in read_jsonl(path):
        hypothesis = row.get("hypothesis") if isinstance(row.get("hypothesis"), dict) else {}
        text = str(hypothesis.get("hypothesis", "")).strip()
        if row.get("status") == "failed" and text:
            feature_id = int(row["feature_id"])
            cards[feature_id] = {
                "feature_id": feature_id,
                "hypothesis": text,
                "evidence": str(hypothesis.get("evidence", "")).strip(),
                "alternatives": [str(item) for item in hypothesis.get("alternatives", [])],
                "uncertainty": str(hypothesis.get("uncertainty", "not stated")),
                "confidence": str(hypothesis.get("confidence", "not stated")),
                "prior_status": row.get("status"),
                "prior_high_rate": row.get("high_pattern_rate"),
                "prior_low_rate": row.get("low_pattern_rate"),
                "prior_contrast": row.get("contrast"),
            }
    if len(cards) != 34:
        raise RuntimeError(f"expected exactly 34 failed-but-parsed cards, found {len(cards)}")
    return cards


def load_train_labels_only(path: Path, training_paths: set[str]) -> dict[str, int]:
    """Read labels solely for declared SAE-training images; holdout labels never enter this map."""
    labels: dict[str, int] = {}
    wanted = {image_key(item) for item in training_paths}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            key = image_key(row["image"])
            if key in wanted:
                grade = int(row["level"])
                if grade not in range(5):
                    raise RuntimeError(f"invalid training grade: {key}")
                labels[key] = grade
    # Experiment 8's patient cohort intentionally contains both EyePACS train
    # and unlabeled EyePACS test images.  Only this returned intersection is
    # eligible for the label-informed association table.
    if not labels:
        raise RuntimeError("no selected SAE-training images have an EyePACS label")
    return labels


def holdout_and_training_indices(paths: np.ndarray, patients: np.ndarray, cohorts: dict, holdout_paths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    all_paths = np.asarray(paths).astype(str)
    hold_paths = np.asarray(holdout_paths).astype(str)
    lookup = {path: index for index, path in enumerate(all_paths)}
    if len(lookup) != len(all_paths) or set(hold_paths) - set(lookup):
        raise RuntimeError("holdout paths do not match frozen activation traces")
    train_patients = set(str(item) for item in cohorts["patients"]["sae_train"])
    hold_patients = set(str(item) for item in cohorts["patients"]["pilot_holdout"])
    train = np.asarray([i for i, patient in enumerate(patients.astype(str)) if patient in train_patients], dtype=np.int64)
    hold = np.asarray([lookup[path] for path in hold_paths], dtype=np.int64)
    observed_hold = set(patients[hold].astype(str))
    if observed_hold - hold_patients or set(patients[train].astype(str)) & observed_hold:
        raise RuntimeError("Experiment 8 patient-disjoint cohort contract violated")
    return train, hold


def feature_maxima(ids: np.ndarray, values: np.ndarray, feature_ids: list[int]) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    for feature_id in feature_ids:
        # A feature can occur in any of the 196 patches and any Top-K slot.
        # Chunking avoids a 4000x196x64 temporary float array while Qwen is
        # using shared Apple-GPU memory.
        maximum = np.zeros(len(ids), dtype=np.float32)
        for start in range(0, len(ids), 128):
            end = min(start + 128, len(ids))
            mask = ids[start:end] == feature_id
            maximum[start:end] = np.where(mask, values[start:end], 0.0).max(axis=(1, 2))
        result[feature_id] = maximum
    return result


def build_associations(ids: np.ndarray, values: np.ndarray, paths: np.ndarray, train_indices: np.ndarray, cards: dict[int, dict], labels_path: Path) -> tuple[dict[int, dict], dict]:
    train_paths = set(paths[train_indices].astype(str))
    labels = load_train_labels_only(labels_path, train_paths)
    labelled_train = np.asarray([index for index in train_indices if image_key(str(paths[index])) in labels], dtype=np.int64)
    if not len(labelled_train):
        raise RuntimeError("no labelled Experiment 8 SAE-training images are available")
    grades = np.asarray([labels[image_key(path)] for path in paths[labelled_train].astype(str)], dtype=np.int64)
    association: dict[int, dict] = {}
    maxima = feature_maxima(ids, values, sorted(cards))
    for feature_id, maxima_all in maxima.items():
        active = maxima_all[labelled_train]
        positive = active[active > 0]
        if len(positive) < 8:
            raise RuntimeError(f"eligible card {feature_id} has too few training activations")
        cuts = np.quantile(positive, [0.25, 0.50, 0.75]).astype(float)
        bins = np.zeros(len(active), dtype=np.int8)
        bins[active > 0] = np.searchsorted(cuts, active[active > 0], side="right") + 1
        table = []
        for bin_id in range(5):
            selected = grades[bins == bin_id]
            counts = np.bincount(selected, minlength=5).astype(int)
            smoothed = (counts + 1.0) / (len(selected) + 5.0)
            table.append({"bin": int(bin_id), "n": int(len(selected)), "grade_counts": counts.tolist(), "grade_probability_dirichlet_alpha_1": smoothed.tolist(), "referable_rate_smoothed": float(smoothed[2:].sum())})
        association[feature_id] = {"feature_id": feature_id, "positive_activation_quartiles": cuts.tolist(), "bin_zero_then_positive_quartiles": table,
                                   "training_images": int(len(grades)), "activation_prevalence": float(np.mean(active > 0))}
    prior_counts = np.bincount(grades, minlength=5).astype(int)
    prior = (prior_counts + 1.0) / (len(grades) + 5.0)
    return association, {"grade_counts": prior_counts.tolist(), "grade_probability_dirichlet_alpha_1": prior.tolist(), "referable_probability": float(prior[2:].sum()), "training_images": int(len(grades))}


def event_association(record: dict, association: dict[int, dict]) -> dict:
    item = association[int(record["feature_id"])]
    value = float(record["activation"])
    cuts = np.asarray(item["positive_activation_quartiles"], dtype=float)
    bin_id = int(np.searchsorted(cuts, value, side="right") + 1) if value > 0 else 0
    return {"activation_bin": bin_id, **item["bin_zero_then_positive_quartiles"][bin_id]}


def select_events(ids: np.ndarray, values: np.ndarray, image_index: int, cards: dict[int, dict], maximum: int) -> list[dict]:
    candidates: list[dict] = []
    allowed = set(cards)
    for patch_index in range(PATCHES):
        for rank in range(TOP_K):
            feature_id = int(ids[image_index, patch_index, rank])
            activation = float(values[image_index, patch_index, rank])
            if feature_id in allowed and activation > 0:
                candidates.append({"feature_id": feature_id, "activation": activation, "patch_index": patch_index, "row": patch_index // GRID, "column": patch_index % GRID, "topk_rank": rank + 1})
    candidates.sort(key=lambda row: (-row["activation"], row["feature_id"], row["patch_index"]))
    selected, used = [], set()
    for record in candidates:
        if record["feature_id"] in used:
            continue
        selected.append(record)
        used.add(record["feature_id"])
        if len(selected) == maximum:
            break
    if not selected:
        raise RuntimeError(f"image index {image_index} has no activation for an eligible card")
    for order, record in enumerate(selected, start=1):
        record["event_number"] = order
        record["within_image_rank"] = order
    return selected


def make_crop_sheet(dataset_root: Path, relative_path: str, events: list[dict], destination: Path) -> Path:
    if destination.is_file():
        return destination
    with Image.open(dataset_root / relative_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        tensor = preprocess_global(image, 224)
    frame = Image.fromarray((tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8), "RGB")
    tile, columns, rows = 160, 4, 2
    sheet = Image.new("RGB", (columns * tile, rows * tile), "white")
    draw = ImageDraw.Draw(sheet)
    for item in events:
        index = int(item["event_number"]) - 1
        x, y = (index % columns) * tile, (index // columns) * tile
        patch_index = int(item["patch_index"])
        row, column = divmod(patch_index, GRID)
        crop = frame.crop((column * 16, row * 16, (column + 1) * 16, (row + 1) * 16)).resize((150, 132), Image.Resampling.NEAREST)
        sheet.paste(crop, (x + 5, y + 22))
        draw.text((x + 5, y + 4), f"{item['event_number']}: {row},{column}", fill="black")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, format="PNG")
    return destination


def image_bytes(path: Path, max_side: int) -> bytes:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90, optimize=False)
        return buffer.getvalue()


def call_ollama(base_url: str, model: str, prompt: str, images: list[Path], timeout: int, context_length: int) -> str:
    payload = {"model": model, "stream": False, "format": "json", "options": {"temperature": 0, "num_ctx": context_length, "num_predict": 512},
               "messages": [{"role": "user", "content": prompt, "images": [base64.b64encode(image_bytes(image, 560)).decode("ascii") for image in images]}]}
    request = urllib.request.Request(base_url.rstrip("/") + "/api/chat", data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return str(json.loads(response.read().decode("utf-8"))["message"]["content"])


def call_repair(base_url: str, model: str, malformed: str, timeout: int) -> str:
    prompt = (
        "Return corrected JSON only. Preserve no information beyond this already generated response; do not add observations. "
        "The required keys are predicted_grade, referable_risk_0_to_100, visible_findings, uncertainty, image_quality. "
        "predicted_grade must be exactly one integer in {0, 1, 2, 3, 4}; referable_risk_0_to_100 must be a number from 0 to 100.\n"
        + malformed
    )
    payload = {"model": model, "stream": False, "format": "json", "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 512}, "messages": [{"role": "user", "content": prompt}]}
    request = urllib.request.Request(base_url.rstrip("/") + "/api/chat", data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return str(json.loads(response.read().decode("utf-8"))["message"]["content"])


def parse_json(raw: str) -> dict | None:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            value = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(value, dict):
        return None
    try:
        grade = int(value["predicted_grade"])
        risk = float(value["referable_risk_0_to_100"])
    except (KeyError, TypeError, ValueError):
        return None
    if grade not in range(5) or not np.isfinite(risk) or not 0 <= risk <= 100:
        return None
    findings = value.get("visible_findings")
    if not isinstance(findings, list) or not all(isinstance(item, str) for item in findings):
        return None
    return {"predicted_grade": grade, "referable_risk_0_to_100": risk, "referable_probability": risk / 100.0, "visible_findings": findings,
            "uncertainty": str(value.get("uncertainty", "not stated")), "image_quality": str(value.get("image_quality", "not stated"))}


def shuffled_card_map(cards: dict[int, dict]) -> dict[int, int]:
    ids = np.asarray(sorted(cards), dtype=int)
    permuted = np.random.default_rng(SHUFFLE_SEED).permutation(ids)
    mapping = {int(original): int(replacement) for original, replacement in zip(ids, permuted)}
    if mapping == {int(item): int(item) for item in ids}:
        raise RuntimeError("card shuffle unexpectedly left every description in place")
    return mapping


def card_context(events: list[dict], cards: dict[int, dict], associations: dict[int, dict], shuffle: dict[int, int] | None) -> tuple[str, list[dict]]:
    lines, audit = [], []
    for event in events:
        source_feature = int(event["feature_id"])
        description_feature = int(shuffle[source_feature]) if shuffle is not None else source_feature
        card = cards[description_feature]
        association = event_association(event, associations)
        # Two vision inputs already consume roughly half of the feasible 4096
        # context. Keep every required field while bounding language evidence.
        hypothesis = card["hypothesis"].replace("\n", " ")[:105]
        alternatives = (card["alternatives"][0].replace("\n", " ")[:55] if card["alternatives"] else "not stated")
        grade_profile = [round(float(item), 3) for item in association["grade_probability_dirichlet_alpha_1"]]
        lines.append(
            f"E{event['event_number']} grid({event['row']},{event['column']}) rank{event['within_image_rank']}: "
            f"card={hypothesis}; uncertainty={card['uncertainty']}; nuisance={alternatives}; "
            f"train-bin n={association['n']}, referable={association['referable_rate_smoothed']:.3f}, grades={grade_profile}."
        )
        audit.append({"event_number": event["event_number"], "active_feature_id": source_feature, "description_feature_id": description_feature,
                      "association_feature_id": source_feature, "association": association, "card_prior_validation_status": card["prior_status"]})
    return "\n".join(lines), audit


def prompt_for_arm(arm: str, events: list[dict], cards: dict[int, dict], associations: dict[int, dict], shuffle: dict[int, int]) -> tuple[str, list[dict]]:
    prompt = BASE_PROMPT
    audit: list[dict] = []
    if arm in ("image_patches", "image_patches_real_cards", "image_patches_shuffled_cards"):
        prompt += PATCH_PROMPT
    if arm in ("image_patches_real_cards", "image_patches_shuffled_cards", CARD_ONLY_ARM):
        context, audit = card_context(events, cards, associations, shuffle if arm.endswith("shuffled_cards") else None)
        prompt += CARD_PROMPT + "\n" + context
    if "target" in prompt.lower() or "reference grade is hidden" not in prompt.lower():
        raise RuntimeError("prompt label-leakage static check failed")
    return prompt, audit


def existing_records(path: Path) -> dict[str, dict]:
    return {str(row["relative_path"]): row for row in read_jsonl(path) if row.get("relative_path")}


def run_arm(arm: str, records: list[dict], root: Path, dataset: Path, cards: dict[int, dict], associations: dict[int, dict], shuffle: dict[int, int], base_url: str, model: str, timeout: int, context_length: int, limit: int | None) -> list[dict]:
    arm_root = root / "responses" / arm
    arm_root.mkdir(parents=True, exist_ok=True)
    partial = arm_root / "responses.partial.jsonl"
    prior = existing_records(partial)
    selected = records[:limit] if limit is not None else records
    with partial.open("a", encoding="utf-8") as handle:
        for position, record in enumerate(selected, start=1):
            path = record["relative_path"]
            if path in prior:
                continue
            full = dataset / path
            if not full.is_file():
                result = {"relative_path": path, "patient_id": record["patient_id"], "status": "image_missing"}
            else:
                prompt, card_audit = prompt_for_arm(arm, record["events"], cards, associations, shuffle)
                images = [full]
                crop_path = None
                if arm in ("image_patches", "image_patches_real_cards", "image_patches_shuffled_cards"):
                    crop_path = make_crop_sheet(dataset, path, record["events"], root / "patch_contact_sheets" / (Path(path).stem + ".png"))
                    images.append(crop_path)
                raw, repaired, parsed, error = None, None, None, None
                try:
                    raw = call_ollama(base_url, model, prompt, images, timeout, context_length)
                    parsed = parse_json(raw)
                    if parsed is None:
                        repaired = call_repair(base_url, model, raw, timeout)
                        parsed = parse_json(repaired)
                    status = "complete" if parsed is not None else "unparsed"
                except urllib.error.HTTPError as exc:
                    body = exc.read().decode("utf-8", errors="replace")[:1000]
                    status, error = "request_failed", f"{exc!r}; body={body}"
                except (urllib.error.URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as exc:
                    status, error = "request_failed", repr(exc)
                result = {"relative_path": path, "patient_id": record["patient_id"], "status": status, "events": record["events"], "card_audit": card_audit,
                          "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "full_image_sha256": sha256_path(full), "patch_contact_sheet": str(crop_path) if crop_path else None,
                          "raw_response": raw, "json_repair_response": repaired, "parsed_response": parsed, "error": error}
            handle.write(json.dumps(result, sort_keys=True) + "\n")
            handle.flush()
            print(json.dumps({"arm": arm, "completed": position, "total": len(selected), "status": result["status"], "path": path}, sort_keys=True), flush=True)
    output = [existing_records(partial)[record["relative_path"]] for record in selected]
    final = arm_root / "responses.jsonl"
    final.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in output), encoding="utf-8")
    return output


def metric_set(targets: np.ndarray, grades: np.ndarray, score: np.ndarray, predicted: np.ndarray) -> dict:
    return {"auroc": float(roc_auc_score(targets, score)), "auprc": float(average_precision_score(targets, score)),
            "brier": float(np.mean((score - targets) ** 2)), "qwk": float(cohen_kappa_score(grades, predicted, weights="quadratic")),
            "macro_f1": float(f1_score(grades, predicted, average="macro", labels=[0, 1, 2, 3, 4], zero_division=0)), "ordinal_mae": float(np.mean(np.abs(grades - predicted)))}


def paired_bootstrap(targets: np.ndarray, a: np.ndarray, b: np.ndarray, patients: np.ndarray, seed: int) -> dict:
    estimate = float(roc_auc_score(targets, a) - roc_auc_score(targets, b))
    unique = np.unique(patients)
    groups = {patient: np.where(patients == patient)[0] for patient in unique}
    rng, values = np.random.default_rng(seed), []
    for _ in range(BOOTSTRAPS):
        selected = rng.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([groups[patient] for patient in selected])
        if np.unique(targets[index]).size == 2:
            values.append(float(roc_auc_score(targets[index], a[index]) - roc_auc_score(targets[index], b[index])))
    return {"estimate": estimate, "confidence": 0.95, "interval": np.percentile(values, [2.5, 97.5]).tolist(), "patients": int(len(unique)), "resamples": BOOTSTRAPS, "valid_resamples": int(len(values))}


def finalize(root: Path, prediction_path: Path, arm_rows: dict[str, list[dict]], prior: dict, source_hashes: dict) -> dict:
    frozen = np.load(prediction_path, allow_pickle=False)
    paths, patients = frozen["paths"].astype(str), frozen["patients"].astype(str)
    targets, grades, dense_score = frozen["targets"].astype(int), frozen["grades"].astype(int), frozen["dense_mlp"].astype(float)
    if len(paths) != 162 or len(set(patients)) != 81:
        raise RuntimeError("frozen Experiment 8 pilot-holdout contract changed")
    response_by_arm = {arm: {row["relative_path"]: row for row in rows} for arm, rows in arm_rows.items()}
    metrics, all_scores = {}, {"dense_mlp": dense_score}
    for arm, lookup in response_by_arm.items():
        scores, predicted, completions = [], [], []
        for path in paths:
            row = lookup.get(path)
            parsed = row.get("parsed_response") if row else None
            if row and row.get("status") == "complete" and isinstance(parsed, dict):
                scores.append(float(parsed["referable_probability"])); predicted.append(int(parsed["predicted_grade"])); completions.append(True)
            else:
                scores.append(float(prior["referable_probability"])); predicted.append(int(np.argmax(prior["grade_probability_dirichlet_alpha_1"]))); completions.append(False)
        score, predicted = np.asarray(scores), np.asarray(predicted)
        all_scores[arm] = score
        metrics[arm] = {"conservative_all_images": metric_set(targets, grades, score, predicted), "json_completion_rate": float(np.mean(completions)), "complete_images": int(sum(completions)), "requested_images": int(len(paths))}
    metrics["dense_mlp"] = {"binary_auroc": float(roc_auc_score(targets, dense_score)), "binary_auprc": float(average_precision_score(targets, dense_score))}
    comparisons = {
        "image_only_minus_dense_mlp": paired_bootstrap(targets, all_scores["image_only"], dense_score, patients, 5101),
        "image_patches_minus_image_only": paired_bootstrap(targets, all_scores["image_patches"], all_scores["image_only"], patients, 5102),
        "real_cards_minus_image_patches": paired_bootstrap(targets, all_scores["image_patches_real_cards"], all_scores["image_patches"], patients, 5103),
        "real_cards_minus_shuffled_cards": paired_bootstrap(targets, all_scores["image_patches_real_cards"], all_scores["image_patches_shuffled_cards"], patients, 5104),
        "real_cards_minus_dense_mlp": paired_bootstrap(targets, all_scores["image_patches_real_cards"], dense_score, patients, 5105),
    }
    rows = [{"method": "Dense RETFound MLP (frozen numerical baseline)", "auroc": metrics["dense_mlp"]["binary_auroc"], "auprc": metrics["dense_mlp"]["binary_auprc"], "qwk": "not comparable: binary checkpoint output"}]
    names = {"image_only": "Qwen image-only", "image_patches": "Qwen image + patches", "image_patches_real_cards": "Qwen image + patches + real cards", "image_patches_shuffled_cards": "Qwen image + patches + shuffled cards"}
    for arm in ARMS:
        result = metrics[arm]["conservative_all_images"]
        rows.append({"method": names[arm], "auroc": result["auroc"], "auprc": result["auprc"], "qwk": result["qwk"], "json_completion_rate": metrics[arm]["json_completion_rate"]})
    (root / "experiment_10_results_table.csv").write_text("method,auroc,auprc,qwk,json_completion_rate\n" + "\n".join(
        f"{row['method']},{row['auroc']},{row['auprc']},{row['qwk']},{row.get('json_completion_rate', '')}" for row in rows) + "\n", encoding="utf-8")
    result = {"schema_version": "1.0", "status": "complete_exploratory_pilot", "scope": "Experiment 8 pilot holdout was previously accessed; patch-card text is unvalidated and this is not clinical validation.",
              "holdout": {"images": int(len(paths)), "patients": int(len(set(patients)))}, "prior_for_unparsed": prior, "metrics": metrics, "paired_auroc_differences": comparisons,
              "semantic_card_success": "real_cards_minus_shuffled_cards requires a positive paired 95% interval; no threshold is evaluated until results exist", "qwen_output_contract": "predicted_grade_0_to_4_and_referable_risk_0_to_100; AUROC uses risk divided by 100", "source_hashes": source_hashes,
              "labels_access_audit": {"holdout_targets_loaded_only_in_finalize_after_all_raw_responses": True, "training_only_associations": True}, "result_table": str(root / "experiment_10_results_table.csv")}
    write_json(root / "experiment_10_result.json", result)
    return result


def prepare(args, experiment_name: str = "experiment_10b_scalar_risk_prompt_amendment", specification_arms: tuple[str, ...] = ARMS) -> tuple[dict, list[dict], dict, dict, dict, dict]:
    root = output_root(args.output); root.mkdir(parents=True, exist_ok=True)
    trace_path = require_input(Path(args.traces), "Experiment 8 activation traces")
    cards_path = require_input(Path(args.cards), "Experiment 8 cards")
    cohorts_path = require_input(Path(args.cohorts), "Experiment 8 cohorts")
    predictions_path = require_input(Path(args.predictions), "Experiment 8 frozen predictions")
    labels_path = require_input(Path(args.labels), "EyePACS labels")
    dataset = require_external(Path(args.dataset), "EyePACS dataset")
    if not dataset.is_dir():
        raise RuntimeError(f"missing dataset root: {dataset}")
    trace = np.load(trace_path, allow_pickle=False)
    ids, values, paths, patients = trace["feature_ids"].astype(np.int32), trace["values"].astype(np.float32), trace["paths"].astype(str), trace["patients"].astype(str)
    if ids.shape != (4000, PATCHES, TOP_K) or values.shape != ids.shape or not np.array_equal(paths, trace["paths"].astype(str)):
        raise RuntimeError("Experiment 8 trace shape or metadata contract changed")
    frozen = np.load(predictions_path, allow_pickle=False)
    holdout_paths, holdout_patients = frozen["paths"].astype(str), frozen["patients"].astype(str)
    if len(holdout_paths) != 162 or len(set(holdout_patients)) != 81:
        raise RuntimeError("unexpected Experiment 8 holdout shape")
    cohorts = json.loads(cohorts_path.read_text(encoding="utf-8"))
    train_indices, hold_indices = holdout_and_training_indices(paths, patients, cohorts, holdout_paths)
    cards = load_eligible_cards(cards_path)
    associations, prior = build_associations(ids, values, paths, train_indices, cards, labels_path)
    events = []
    for path, patient, index in zip(holdout_paths, holdout_patients, hold_indices):
        rows = select_events(ids, values, int(index), cards, args.top_events)
        events.append({"relative_path": str(path), "patient_id": str(patient), "trace_index": int(index), "events": rows})
    shuffle = shuffled_card_map(cards)
    source_hashes = {name: {"path": str(path), "sha256": sha256_path(path)} for name, path in {"traces": trace_path, "cards": cards_path, "cohorts": cohorts_path, "predictions": predictions_path, "labels": labels_path}.items()}
    specification = {"schema_version": "1.0", "status": "frozen_before_qwen_inference", "experiment": experiment_name, "model": args.model, "seed": SEED, "shuffle_seed": SHUFFLE_SEED, "arms": list(specification_arms), "top_events": args.top_events, "qwen_output_contract": "predicted_grade_0_to_4_and_referable_risk_0_to_100",
                     "eligible_cards": sorted(cards), "excluded_cards": 30, "card_claim_boundary": "all included cards failed contrastive validation; their wording is unvalidated exploratory context", "holdout": {"images": len(events), "patients": len(set(holdout_patients))},
                     "training_only_feature_associations": True, "holdout_labels_loaded_during_preparation": False, "source_hashes": source_hashes}
    freeze = root / "specification_freeze.json"
    if freeze.is_file() and json.loads(freeze.read_text(encoding="utf-8")) != specification:
        raise RuntimeError("existing specification freeze is incompatible with current inputs")
    write_json(freeze, specification)
    write_json(root / "training_feature_associations.json", {"association": associations, "prior": prior})
    write_json(root / "card_shuffle_map.json", {"seed": SHUFFLE_SEED, "active_feature_to_description_feature": shuffle})
    write_json(root / "holdout_patch_selection.json", {"events": events, "labels_included": False})
    return {"root": root, "dataset": dataset, "predictions": predictions_path, "source_hashes": source_hashes}, events, cards, associations, prior, shuffle


def smoke_checks(prepared: dict, events: list[dict], cards: dict, associations: dict, shuffle: dict, args) -> dict:
    if len(events) < 5 or any(len(item["events"]) > args.top_events or not item["events"] for item in events[:5]):
        raise RuntimeError("smoke event-selection contract failed")
    for item in events[:5]:
        seen = [event["feature_id"] for event in item["events"]]
        if len(seen) != len(set(seen)):
            raise RuntimeError("smoke unique-feature selection failed")
        real, real_audit = prompt_for_arm("image_patches_real_cards", item["events"], cards, associations, shuffle)
        shuffled, shuffled_audit = prompt_for_arm("image_patches_shuffled_cards", item["events"], cards, associations, shuffle)
        if real == shuffled or any(a["active_feature_id"] != b["active_feature_id"] or a["association_feature_id"] != b["association_feature_id"] for a, b in zip(real_audit, shuffled_audit)):
            raise RuntimeError("smoke shuffled-text-only control failed")
    base_url = args.base_url or os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
    # A full 5-image/4-arm smoke proves both visual payload variants and strict JSON parsing.
    # Keep an earlier failed transport attempt as audit evidence.  The actual
    # vision-model smoke uses its own immutable attempt directory.
    smoke_root = prepared["root"] / "smoke_qwen25vl_scalar_risk_range_repair_v2"
    rows = {arm: run_arm(arm, events, smoke_root, prepared["dataset"], cards, associations, shuffle, base_url, args.model, args.timeout, args.context_length, 5) for arm in ARMS}
    parsed = all(row.get("status") == "complete" for values in rows.values() for row in values)
    result = {"status": "complete" if parsed else "failed", "images": 5, "arms": list(ARMS), "strict_json_complete": parsed,
              "checks": {"patch_coordinate_alignment": True, "unique_feature_events": True, "labels_absent_from_prompts": True, "card_mapping": True, "shuffled_text_only": True, "external_output": True},
              "ollama_base_url": base_url, "model": args.model, "response_root": str(smoke_root)}
    write_json(prepared["root"] / "smoke_result.json", result)
    if not parsed:
        raise RuntimeError("Experiment 10 smoke failed: one or more Qwen responses were invalid")
    return result


def finalize_card_only(root: Path, prediction_path: Path, rows: list[dict], prior: dict, source_hashes: dict) -> dict:
    """Evaluate the separately frozen image-plus-card-only ablation after inference ends."""
    frozen = np.load(prediction_path, allow_pickle=False)
    paths, patients = frozen["paths"].astype(str), frozen["patients"].astype(str)
    targets, grades, dense_score = frozen["targets"].astype(int), frozen["grades"].astype(int), frozen["dense_mlp"].astype(float)
    if len(paths) != 162 or len(set(patients)) != 81:
        raise RuntimeError("frozen Experiment 8 pilot-holdout contract changed")
    lookup = {row["relative_path"]: row for row in rows}
    scores, predicted, completed = [], [], []
    for path in paths:
        row = lookup.get(path)
        parsed = row.get("parsed_response") if row else None
        if row and row.get("status") == "complete" and isinstance(parsed, dict):
            scores.append(float(parsed["referable_probability"])); predicted.append(int(parsed["predicted_grade"])); completed.append(True)
        else:
            scores.append(float(prior["referable_probability"])); predicted.append(int(np.argmax(prior["grade_probability_dirichlet_alpha_1"]))); completed.append(False)
    score, predicted = np.asarray(scores), np.asarray(predicted)
    result = {
        "schema_version": "1.0",
        "status": "complete_exploratory_pilot",
        "scope": "Separate Experiment 10C image-plus-real-SAE-card-only ablation. Card wording is unvalidated exploratory context, not lesion confirmation or clinical validation.",
        "holdout": {"images": int(len(paths)), "patients": int(len(set(patients)))},
        "metrics": {
            CARD_ONLY_ARM: {"conservative_all_images": metric_set(targets, grades, score, predicted), "json_completion_rate": float(np.mean(completed)), "complete_images": int(sum(completed)), "requested_images": int(len(paths))},
            "dense_mlp": {"binary_auroc": float(roc_auc_score(targets, dense_score)), "binary_auprc": float(average_precision_score(targets, dense_score))},
        },
        "paired_auroc_differences": {"image_real_cards_no_patches_minus_dense_mlp": paired_bootstrap(targets, score, dense_score, patients, 5110)},
        "qwen_output_contract": "predicted_grade_0_to_4_and_referable_risk_0_to_100; AUROC uses risk divided by 100",
        "source_hashes": source_hashes,
        "labels_access_audit": {"holdout_targets_loaded_only_in_finalize_after_all_raw_responses": True, "training_only_associations": True},
    }
    write_json(root / "experiment_10c_result.json", result)
    (root / "experiment_10c_results_table.csv").write_text(
        "method,auroc,auprc,qwk,json_completion_rate\n"
        f"Dense RETFound MLP (frozen numerical baseline),{result['metrics']['dense_mlp']['binary_auroc']},{result['metrics']['dense_mlp']['binary_auprc']},not_comparable,\n"
        f"Qwen image + real SAE cards (no patches),{result['metrics'][CARD_ONLY_ARM]['conservative_all_images']['auroc']},{result['metrics'][CARD_ONLY_ARM]['conservative_all_images']['auprc']},{result['metrics'][CARD_ONLY_ARM]['conservative_all_images']['qwk']},{result['metrics'][CARD_ONLY_ARM]['json_completion_rate']}\n",
        encoding="utf-8",
    )
    return result


def card_only_smoke_checks(prepared: dict, events: list[dict], cards: dict, associations: dict, shuffle: dict, args) -> dict:
    if len(events) < 5 or any(not item["events"] for item in events[:5]):
        raise RuntimeError("card-only smoke event-selection contract failed")
    for item in events[:5]:
        prompt, audit = prompt_for_arm(CARD_ONLY_ARM, item["events"], cards, associations, shuffle)
        if PATCH_PROMPT.strip() in prompt or not audit or "target" in prompt.lower():
            raise RuntimeError("card-only prompt isolation or label-leakage check failed")
    base_url = args.base_url or os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
    smoke_root = prepared["root"] / "smoke_image_real_cards_no_patches"
    rows = run_arm(CARD_ONLY_ARM, events, smoke_root, prepared["dataset"], cards, associations, shuffle, base_url, args.model, args.timeout, args.context_length, 5)
    parsed = all(row.get("status") == "complete" for row in rows)
    result = {"status": "complete" if parsed else "failed", "images": 5, "arm": CARD_ONLY_ARM, "strict_json_complete": parsed,
              "checks": {"full_image_only": True, "no_patch_contact_sheet": True, "labels_absent_from_prompts": True, "real_card_mapping": True, "external_output": True},
              "ollama_base_url": base_url, "model": args.model, "response_root": str(smoke_root)}
    write_json(prepared["root"] / "smoke_image_real_cards_no_patches_result.json", result)
    if not parsed:
        raise RuntimeError("Experiment 10C smoke failed: one or more Qwen responses were invalid")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment 10 image + patch + SAE-card Qwen pilot")
    parser.add_argument("--stage", choices=("smoke", "run", "finalize", "card_only_smoke", "card_only_run", "card_only_finalize"), required=True)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--traces", default=str(TOP64 / "patch_activation_traces.npz"))
    parser.add_argument("--cards", default=str(TOP64 / "patch_feature_cards.jsonl"))
    parser.add_argument("--cohorts", default=str(EXP8 / "patient_cohorts.json"))
    parser.add_argument("--predictions", default=str(TOP64 / "pilot_holdout_predictions.npz"))
    parser.add_argument("--labels", default=str(EXTERNAL / "datasets/eyepacs/trainLabels.csv"))
    parser.add_argument("--dataset", default=str(EXTERNAL / "datasets/eyepacs"))
    parser.add_argument("--model", default="qwen2.5vl:3b")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--timeout", type=int, default=300)
    # 4096 is the feasible vision context on the current 11.8-GB Apple GPU;
    # eight cropped patches are compacted into one contact sheet.
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--top-events", type=int, default=8, choices=range(1, 17))
    args = parser.parse_args()
    card_only = args.stage.startswith("card_only")
    if card_only and args.output == str(DEFAULT_OUTPUT):
        args.output = str(DEFAULT_CARD_ONLY_OUTPUT)
    prepared, events, cards, associations, prior, shuffle = prepare(
        args,
        experiment_name="experiment_10c_image_card_only_ablation" if card_only else "experiment_10b_scalar_risk_prompt_amendment",
        specification_arms=(CARD_ONLY_ARM,) if card_only else ARMS,
    )
    if args.stage == "card_only_smoke":
        print(json.dumps(card_only_smoke_checks(prepared, events, cards, associations, shuffle, args), sort_keys=True)); return 0
    if args.stage == "card_only_run":
        smoke = prepared["root"] / "smoke_image_real_cards_no_patches_result.json"
        if not smoke.is_file() or json.loads(smoke.read_text()).get("status") != "complete":
            raise RuntimeError("Experiment 10C full run is blocked until the five-image smoke passes")
        base_url = args.base_url or os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
        rows = run_arm(CARD_ONLY_ARM, events, prepared["root"], prepared["dataset"], cards, associations, shuffle, base_url, args.model, args.timeout, args.context_length, None)
        print(json.dumps(finalize_card_only(prepared["root"], prepared["predictions"], rows, prior, prepared["source_hashes"]), sort_keys=True)); return 0
    if args.stage == "card_only_finalize":
        rows = read_jsonl(prepared["root"] / "responses" / CARD_ONLY_ARM / "responses.jsonl")
        if len(rows) != len(events):
            raise RuntimeError("cannot finalize Experiment 10C: not every holdout image has a response")
        print(json.dumps(finalize_card_only(prepared["root"], prepared["predictions"], rows, prior, prepared["source_hashes"]), sort_keys=True)); return 0
    if args.stage == "smoke":
        print(json.dumps(smoke_checks(prepared, events, cards, associations, shuffle, args), sort_keys=True)); return 0
    if args.stage == "run":
        smoke = prepared["root"] / "smoke_result.json"
        if not smoke.is_file() or json.loads(smoke.read_text()).get("status") != "complete":
            raise RuntimeError("full run is blocked until the five-image smoke passes")
        base_url = args.base_url or os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
        arm_rows = {arm: run_arm(arm, events, prepared["root"], prepared["dataset"], cards, associations, shuffle, base_url, args.model, args.timeout, args.context_length, None) for arm in ARMS}
        print(json.dumps(finalize(prepared["root"], prepared["predictions"], arm_rows, prior, prepared["source_hashes"]), sort_keys=True)); return 0
    rows = {arm: read_jsonl(prepared["root"] / "responses" / arm / "responses.jsonl") for arm in ARMS}
    if any(len(rows[arm]) != len(events) for arm in ARMS):
        raise RuntimeError("cannot finalize: not every arm has every holdout image")
    print(json.dumps(finalize(prepared["root"], prepared["predictions"], rows, prior, prepared["source_hashes"]), sort_keys=True))
    return 0


if __name__ == "__main__":
    main()
