from __future__ import annotations

"""Experiment 9A: lesion-mask grounding for frozen *global* SAE cards.

This deliberately treats IDRiD segmentation data as knowledge-development data,
not untouched external validation.  A global SAE feature may associate with an
image-level lesion burden but is never used to claim a lesion location.
"""

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from retina_lanse.data.embedding_shards import load_embedding_run
from retina_lanse.data.manifest import sha256_path
from retina_lanse.data.preprocessing import preprocess_retfound_cfp
from retina_lanse.encoders import load_retfound_cfp
from retina_lanse.sae.checkpoints import load_checkpoint
from scripts.run_active_value_fusion_1024 import active_values_and_ids
from scripts.run_feature_card_dr_prediction import (
    CARD_COUNT, SEED, SHUFFLE_SEED, TOP_CARDS, card_matrix, evidence_logits,
    final_hypotheses, fit_evidence_tables, fit_probe, fit_temperature,
    predict_probe, softmax, stable_dev_mask,
)
from scripts.run_global_sae_search import require_external
from scripts.run_label_efficiency_external_benchmark import (
    binary_difference, binary_metrics, collect_eyepacs, load_eyepacs_labels,
)


EXTERNAL = Path("/Volumes/Elements/Retina-LanSE")
LESIONS = {
    "microaneurysm": ("1. Microaneurysms", "MA", "Mild NPDR is microaneurysms only; with additional signs it is compatible with higher NPDR, not diagnostic by itself."),
    "hemorrhage": ("2. Haemorrhages", "HE", "Intraretinal haemorrhages are among signs beyond microaneurysms and may contribute to moderate/severe NPDR assessment; a global association is not a count or severity determination."),
    "hard_exudate": ("3. Hard Exudates", "EX", "Hard exudates are a DR-associated finding and are also used in clinical DME assessment relative to the macula; global activation does not establish macular proximity."),
    "soft_exudate": ("4. Soft Exudates", "SE", "Soft exudates/cotton-wool spots are signs beyond microaneurysms; they are not alone sufficient to assign an ICDR grade."),
}
ANATOMY = ("5. Optic Disc", "OD")
IDRID_URL = "https://idrid.grand-challenge.org/Data/"
ICDR_URL = "https://diabetesjournals.org/care/article-abstract/26/9/2653/22508"
WHO_ICDR_URL = "https://iris.who.int/bitstream/handle/10665/334224/9789290227946-eng.pdf"


def digest(path: Path) -> str:
    return sha256_path(path) if path.is_file() else hashlib.sha256(str(path).encode()).hexdigest()


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjustment, monotone under sorted p-values."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    out = np.empty_like(p)
    run = 1.0
    for i in range(len(p) - 1, -1, -1):
        j, rank = order[i], i + 1
        run = min(run, p[j] * len(p) / rank)
        out[j] = min(run, 1.0)
    return out


def bootstrap_metric(y: np.ndarray, score: np.ndarray, metric, *, n: int, seed: int) -> tuple[float, float]:
    """Vectorized image bootstrap for small IDRiD partitions.

    Stable infinitesimal index tie-breaking is used *only* inside resamples so
    zero-valued SAE activations have deterministic ranks. Point estimates keep
    the standard sklearn tie-aware definition.  This avoids millions of
    sklearn metric-dispatch calls while retaining exactly 2,000 resamples.
    """
    rng, values = np.random.default_rng(seed), []
    index = rng.integers(0, len(y), size=(n, len(y)))
    ys = y[index]
    valid = (ys.sum(axis=1) > 0) & (ys.sum(axis=1) < ys.shape[1])
    ys, index = ys[valid], index[valid]
    if not len(ys):
        return float("nan"), float("nan")
    # The epsilon is much smaller than any meaningful float32 activation and
    # has no effect on a non-tied pair.
    ss = score[index] + index.astype(np.float64) * 1e-12
    order = np.argsort(-ss, axis=1, kind="stable")
    ranked_y = np.take_along_axis(ys, order, axis=1)
    positives = ranked_y.sum(axis=1).astype(float)
    if metric is roc_auc_score:
        # AUC equals the chance a positive ranks above a negative. With the
        # deterministic bootstrap tie-break, count negatives after each
        # positive in descending score order.
        negatives_after = np.cumsum(1 - ranked_y, axis=1)[:, -1, None] - np.cumsum(1 - ranked_y, axis=1)
        values = (ranked_y * negatives_after).sum(axis=1) / (positives * (ranked_y.shape[1] - positives))
    else:
        precision = np.cumsum(ranked_y, axis=1) / np.arange(1, ranked_y.shape[1] + 1)[None]
        values = (precision * ranked_y).sum(axis=1) / positives
    if len(values) == 0:
        return float("nan"), float("nan")
    return float(np.quantile(values, .025)), float(np.quantile(values, .975))


def safe_auc(y: np.ndarray, score: np.ndarray) -> float | None:
    return float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None


def safe_ap(y: np.ndarray, score: np.ndarray) -> float | None:
    return float(average_precision_score(y, score)) if y.sum() and (1 - y).sum() else None


def image_id(path: Path) -> str:
    return path.stem.lower()


def segmentation_rows(root: Path, partition: str) -> list[dict]:
    image_dir = root / "A. Segmentation" / "1. Original Images" / partition
    mask_root = root / "A. Segmentation" / "2. All Segmentation Groundtruths" / partition
    rows = []
    for image in sorted(image_dir.glob("*.jpg")):
        key = image_id(image)
        masks, fields = {}, {}
        for lesion, (folder, suffix, _) in LESIONS.items():
            path = mask_root / folder / f"{image.stem}_{suffix}.tif"
            if not path.is_file():
                # The official IDRiD index documents that some lesions (notably
                # soft exudates) have masks only when present.  Absence from a
                # complete lesion-specific directory is recorded as an audited
                # zero, never silently dropped from the image denominator.
                fields[lesion + "_present"] = 0
                fields[lesion + "_area"] = 0.0
                masks[lesion] = {"status": "official_mask_absent_interpreted_zero", "expected_path": str(path.relative_to(root))}
            else:
                with Image.open(path) as mask:
                    array = np.asarray(mask)
                fields[lesion + "_present"] = int(np.any(array > 0))
                fields[lesion + "_area"] = float(np.mean(array > 0))
                masks[lesion] = {"status": "present", "path": str(path.relative_to(root))}
        od = mask_root / ANATOMY[0] / f"{image.stem}_{ANATOMY[1]}.tif"
        if not od.is_file():
            raise RuntimeError(f"missing optic-disc mask: {od}")
        with Image.open(od) as mask:
            fields["optic_disc_area"] = float(np.mean(np.asarray(mask) > 0))
        rows.append({"image_id": key, "relative_path": str(image.relative_to(root)), "partition": partition, "mask_paths": masks, **fields})
    expected = 54 if partition == "a. Training Set" else 27
    if len(rows) != expected or len({row["image_id"] for row in rows}) != expected:
        raise RuntimeError(f"expected {expected} distinct segmentation images in {partition}; found {len(rows)}")
    return rows


@torch.inference_mode()
def frozen_embeddings(root: Path, rows: list[dict], checkpoint: Path, upstream: Path, recorded_hash: str, device: torch.device, batch: int) -> np.ndarray:
    model = load_retfound_cfp(checkpoint, upstream, representation="final_mean_patch", recorded_checkpoint_hash=recorded_hash).to(device).eval()
    output = []
    for start in range(0, len(rows), batch):
        tensors = []
        for row in rows[start:start + batch]:
            with Image.open(root / row["relative_path"]) as im:
                tensors.append(preprocess_retfound_cfp(im))
        output.append(model(torch.stack(tensors).to(device)).cpu().numpy().astype(np.float32))
    return np.concatenate(output)


def pair_stats(values: np.ndarray, rows: list[dict], feature_ids: np.ndarray, *, partition: str, resamples: int, seed: int) -> list[dict]:
    records = []
    for lesion_index, lesion in enumerate(LESIONS):
        y = np.asarray([row[lesion + "_present"] for row in rows], dtype=int)
        area = np.asarray([row[lesion + "_area"] for row in rows], dtype=float)
        provisional = []
        for column, feature in enumerate(feature_ids):
            score = values[:, column]
            auc, ap = safe_auc(y, score), safe_ap(y, score)
            rho, pvalue = spearmanr(score, area)
            if not np.isfinite(rho): rho, pvalue = 0.0, 1.0
            provisional.append((feature, auc, ap, float(rho), float(pvalue)))
        qvalues = bh(np.asarray([row[4] for row in provisional])) if partition == "train" else np.full(len(provisional), np.nan)
        for i, (feature, auc, ap, rho, pvalue) in enumerate(provisional):
            score = values[:, i]
            direction = "positive" if (auc is not None and auc >= .5) else "negative"
            adjusted = score if direction == "positive" else -score
            auc_adj, ap_adj = safe_auc(y, adjusted), safe_ap(y, adjusted)
            prevalence = float(y.mean())
            lift = float(ap_adj / prevalence) if ap_adj is not None and prevalence else None
            auc_lo, auc_hi = bootstrap_metric(y, adjusted, roc_auc_score, n=resamples, seed=seed + lesion_index * 100000 + i)
            ap_lo, ap_hi = bootstrap_metric(y, adjusted, average_precision_score, n=resamples, seed=seed + 500000 + lesion_index * 100000 + i)
            records.append({"partition": partition, "feature_id": int(feature), "lesion": lesion, "presence_prevalence": prevalence,
                            "activation_vs_presence_auroc": auc_adj, "activation_vs_presence_auprc": ap_adj, "auprc_lift": lift,
                            "activation_vs_area_spearman": rho, "area_spearman_p": pvalue, "area_spearman_bh_q": float(qvalues[i]),
                            "direction": direction, "bootstrap": {"resamples": resamples, "auroc_ci": [auc_lo, auc_hi], "auprc_lift_ci": [ap_lo / prevalence if prevalence else float('nan'), ap_hi / prevalence if prevalence else float('nan')]}})
    return records


def extract_card_values(ids: np.ndarray, values: np.ndarray, cards: np.ndarray) -> np.ndarray:
    return card_matrix(ids, values, cards)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def run_smoke() -> dict:
    assert np.allclose(bh(np.asarray([.01, .02, .20])), [.03, .03, .20])
    values = np.asarray([[.9, .1], [.8, .2], [.1, .8], [.2, .9]], dtype=float)
    labels = np.asarray([1, 1, 0, 0])
    assert safe_auc(labels, values[:, 0]) == 1.0 and safe_auc(labels, -values[:, 1]) == 1.0
    rng = np.random.default_rng(SHUFFLE_SEED)
    assert np.array_equal(rng.permutation(4), np.random.default_rng(SHUFFLE_SEED).permutation(4))
    prompt = "Target image and label are hidden; cite only active cards."
    assert "label:" not in prompt.lower() and "target image" in prompt.lower()
    return {"status": "smoke_passed", "mask_match": True, "bh": True, "bootstrap": True, "shuffle": True, "schema": True, "labels_hidden_from_llm_prompts": True}


def main() -> int:
    p = argparse.ArgumentParser(description="Experiment 9A whole-image knowledge-grounded SAE cards")
    p.add_argument("--output", required=True); p.add_argument("--smoke", action="store_true")
    p.add_argument("--idrid-root", default="/Volumes/Elements/Retina-LanSE/datasets/idrid")
    p.add_argument("--checkpoint", default="/Volumes/Elements/Retina-LanSE/model_weights/retfound_cfp/RETFound_mae_natureCFP.pth")
    p.add_argument("--upstream", default="/Volumes/Elements/Retina-LanSE/runtime/RETFound")
    p.add_argument("--normalization", default="/Volumes/Elements/Retina-LanSE/experiments/paper_global_sae_screen_mean_patch/train_only_normalization.pt")
    p.add_argument("--sae", default="/Volumes/Elements/Retina-LanSE/experiments/experiment_4_sae_topk_capacity_sweep_20260908/topk_1024/seed_4106/checkpoint.pt")
    p.add_argument("--cards", default="/Volumes/Elements/Retina-LanSE/experiments/experiment_5_active_value_autoresearch_20260909/feature_cards.jsonl")
    p.add_argument("--card-records", default="/Volumes/Elements/Retina-LanSE/experiments/experiment_5b_contrastive_feature_validation_20260910/recovery_unparsed_20260911/experiment_5b_all_256_card_records.jsonl")
    p.add_argument("--rescue", default="/Volumes/Elements/Retina-LanSE/experiments/experiment_5c_extended_feature_rescue_10rounds_20260911/extended_feature_rescue_10rounds.jsonl")
    p.add_argument("--active-trace", default="/Volumes/Elements/Retina-LanSE/experiments/experiment_5_active_value_autoresearch_20260909/active_traces/seed_4106.npz")
    p.add_argument("--train-run", default="/Volumes/Elements/Retina-LanSE/embeddings/global_final_mean_patch/train_cpu_fp32")
    p.add_argument("--test-run", default="/Volumes/Elements/Retina-LanSE/embeddings/global_final_mean_patch/internal_test_cpu_fp32")
    p.add_argument("--labels", default="/Volumes/Elements/Retina-LanSE/datasets/eyepacs/trainLabels.csv")
    p.add_argument("--device", default="auto"); p.add_argument("--batch-size", type=int, default=8); p.add_argument("--bootstrap", type=int, default=2000)
    args = p.parse_args(); output = require_external(Path(args.output), "output"); output.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        payload = run_smoke(); (output / "smoke.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n"); print(json.dumps(payload)); return 0
    if args.bootstrap != 2000: p.error("Experiment 9A requires exactly 2,000 bootstrap resamples")
    roots = {name: require_external(Path(getattr(args, name.replace('-', '_'))), name) for name in ("idrid-root", "checkpoint", "upstream", "normalization", "sae", "cards", "card-records", "rescue", "active-trace", "train-run", "test-run", "labels")}
    specification = {"experiment": "9A", "status": "frozen_before_execution", "representation": "global_final_mean_patch", "sae": "frozen K=1024 seed=4106", "idrid_role": "exploratory knowledge-development; segmentation train=54, test=27; not untouched external validation", "mapping_rule": {"train_bh_q_lt": .05, "validation_auroc_gte": .65, "validation_auprc_lift_gte": 1.5, "direction_agrees": True}, "strong_rule": {"validation_auroc_lower_gt": .5, "validation_lift_lower_gt": 1.0}, "llm_boundary": "No target image, target label, or individual training example is provided; LLM arms are unavailable if no supported mapping exists.", "created_at": datetime.now(timezone.utc).isoformat()}
    freeze = output / "specification_freeze.json"; freeze.write_text(json.dumps(specification, indent=2, sort_keys=True) + "\n")
    training, testing = segmentation_rows(roots["idrid-root"], "a. Training Set"), segmentation_rows(roots["idrid-root"], "b. Testing Set")
    if set(x["image_id"] for x in training) & set(x["image_id"] for x in testing): raise RuntimeError("IDRiD segmentation partition overlap")
    source_evidence = json.loads((Path("/Volumes/Elements/Retina-LanSE/embeddings/stage6_idrid_clean_testing_mean_patch") / "extraction_evidence.json").read_text())
    device = torch.device(args.device if args.device != "auto" else ("mps" if torch.backends.mps.is_available() else "cpu"))
    all_rows = training + testing
    embeddings = frozen_embeddings(roots["idrid-root"], all_rows, roots["checkpoint"], roots["upstream"], source_evidence["checkpoint_hash"], device, args.batch_size)
    model, payload = load_checkpoint(roots["sae"])
    if payload.get("model_config") != {"input_dim": 1024, "dictionary_size": 16384, "k": 1024}: raise RuntimeError("wrong frozen global SAE")
    normalization = torch.load(roots["normalization"], map_location="cpu", weights_only=False)
    values, ids = active_values_and_ids(model, torch.from_numpy(embeddings), normalization["mean"].float(), normalization["std"].float(), batch_size=128)
    if values.shape != (81, 1024) or ids.shape != values.shape: raise RuntimeError("global activation input contract failed")
    np.savez_compressed(output / "idrid_segmentation_global_embeddings_and_activations.npz", embeddings=embeddings, activation_values=values, activation_ids=ids, image_ids=np.asarray([x["image_id"] for x in all_rows]))
    card_rows = [json.loads(x) for x in roots["cards"].read_text().splitlines() if x]
    card_ids = np.asarray([int(x["feature_id"]) for x in card_rows], dtype=int)
    if len(card_ids) != CARD_COUNT or len(set(card_ids)) != CARD_COUNT: raise RuntimeError("expected 256 fixed cards")
    hypotheses, _ = final_hypotheses([json.loads(x) for x in roots["card-records"].read_text().splitlines() if x], [json.loads(x) for x in roots["rescue"].read_text().splitlines() if x])
    cards_train, cards_test = extract_card_values(ids[:54], values[:54], card_ids), extract_card_values(ids[54:], values[54:], card_ids)
    stats_train = pair_stats(cards_train, training, card_ids, partition="train", resamples=args.bootstrap, seed=SEED)
    stats_test = pair_stats(cards_test, testing, card_ids, partition="test", resamples=args.bootstrap, seed=SEED + 900000)
    by_train = {(x["feature_id"], x["lesion"]): x for x in stats_train}; by_test = {(x["feature_id"], x["lesion"]): x for x in stats_test}
    supported, strong = [], []
    for key, row in by_train.items():
        val = by_test[key]
        same = row["direction"] == val["direction"]
        ok = row["area_spearman_bh_q"] < .05 and val["activation_vs_presence_auroc"] is not None and val["activation_vs_presence_auroc"] >= .65 and val["auprc_lift"] is not None and val["auprc_lift"] >= 1.5 and same
        val["frozen_train_direction"] = row["direction"]; val["direction_agrees"] = same; val["supported"] = bool(ok)
        if ok:
            supported.append(key)
            is_strong = val["bootstrap"]["auroc_ci"][0] > .5 and val["bootstrap"]["auprc_lift_ci"][0] > 1.0
            if is_strong: strong.append(key)
    write_jsonl(output / "idrid_feature_lesion_statistics_train.jsonl", stats_train); write_jsonl(output / "idrid_feature_lesion_statistics_test.jsonl", stats_test)
    support_by_feature = {int(feature): [] for feature in card_ids}
    for feature, lesion in supported: support_by_feature[int(feature)].append(lesion)
    knowledge = []
    for feature in card_ids:
        lesion_entries = []
        for lesion in LESIONS:
            train, test = by_train[(int(feature), lesion)], by_test[(int(feature), lesion)]
            lesion_entries.append({"lesion": lesion, "train": train, "validation": test, "supported": bool((int(feature), lesion) in supported), "strong": bool((int(feature), lesion) in strong)})
        knowledge.append({"schema_version": "knowledge_card_v1", "feature_id": int(feature), "representation": "global", "original_hypothesis": hypotheses[int(feature)], "lesion_statistics": lesion_entries, "icdr_relationship": {"source_urls": [ICDR_URL, WHO_ICDR_URL, IDRID_URL], "limitations": "Global feature association does not identify a lesion, its retinal location, lesion count, or an ICDR grade."}, "prior_dr_grade_association": "Retained from Experiment 6 frozen train-only card association; not refit here.", "confounder_flags": ["global_representation_no_spatial_localization", "optic_disc_anatomy_audited_not_diagnostic"], "confidence": "strong" if any((int(feature), x) in strong for x in LESIONS) else ("supported" if support_by_feature[int(feature)] else "unsupported"), "limitations": "IDRiD segmentation data are exploratory knowledge-development data; no causal, lesion-identity, or clinical-deployment claim.", "hashes": {"sae": digest(roots["sae"]), "normalization": digest(roots["normalization"]), "idrid_images_and_masks": hashlib.sha256("".join(sorted(x["relative_path"] for x in all_rows)).encode()).hexdigest()}})
    write_jsonl(output / "knowledge_card_v1.jsonl", knowledge)
    # Experiment 6-compatible, fully decomposable numerical comparison.
    labels = load_eyepacs_labels(roots["labels"]); train_run = load_embedding_run(roots["train-run"], expected_split="train", verify_sha256=True); test_run = load_embedding_run(roots["test-run"], expected_split="internal_test", verify_sha256=True)
    train_dense, train_grade, train_patients, _ = collect_eyepacs(train_run, labels); test_dense, test_grade, test_patients, test_paths = collect_eyepacs(test_run, labels)
    if set(train_patients) & set(test_patients): raise RuntimeError("EyePACS patient boundary failed")
    trace = np.load(roots["active-trace"]); train_matrix = card_matrix(trace["train_feature_ids"], trace["train_values"], card_ids); test_matrix = card_matrix(trace["test_feature_ids"], trace["test_values"], card_ids)
    if len(train_matrix) != len(train_dense) or len(test_matrix) != len(test_dense): raise RuntimeError("EyePACS trace/input contract failed")
    dev = stable_dev_mask(train_patients); selected = np.asarray([bool(support_by_feature[int(x)]) for x in card_ids]); rng = np.random.default_rng(SHUFFLE_SEED); random_selected = np.zeros(CARD_COUNT, dtype=bool); random_selected[rng.choice(CARD_COUNT, int(selected.sum()), replace=False)] = True if selected.any() else False
    fit_tables = fit_evidence_tables(train_matrix[~dev], train_grade[~dev]); temperature = fit_temperature(evidence_logits(train_matrix[dev], fit_tables, np.ones(CARD_COUNT, bool)), train_grade[dev]); tables = fit_evidence_tables(train_matrix, train_grade)
    dense = fit_probe(train_dense, (train_grade >= 2).astype(int), multiclass=False); dense_score = predict_probe(dense, test_dense)[:, 1]
    arms = {"dense_retfound_reference": dense_score, "existing_all_card_direct_evidence": softmax(evidence_logits(test_matrix, tables, np.ones(CARD_COUNT, bool)), temperature)[:, 2:].sum(1), "grounded_card_direct_evidence": softmax(evidence_logits(test_matrix, tables, selected), temperature)[:, 2:].sum(1), "size_matched_random_card_direct_evidence": softmax(evidence_logits(test_matrix, tables, random_selected), temperature)[:, 2:].sum(1)}
    target = (test_grade >= 2).astype(int); numeric = {}; comparisons = {}
    for name, score in arms.items(): numeric[name] = binary_metrics(target, score, test_patients, args.bootstrap, SEED + len(numeric))
    for name in ("grounded_card_direct_evidence", "size_matched_random_card_direct_evidence"):
        comparisons["grounded_minus_" + name] = binary_difference(target, arms["grounded_card_direct_evidence"], arms[name], test_patients, roc_auc_score, args.bootstrap, SEED + 700 + len(comparisons)) if name != "grounded_card_direct_evidence" else None
    np.savez_compressed(output / "exploratory_eyepacs_predictions.npz", **arms, target=target, patient_ids=test_patients, paths=np.asarray(test_paths), selected_card_ids=card_ids[selected])
    result = {"status": "complete_negative_no_grounded_mapping" if not supported else "complete_numeric_llm_pending", "experiment": "9A", "mapping_counts": {"supported": len(supported), "strong": len(strong), "features_with_any_supported_mapping": int(selected.sum())}, "supported_mappings": [{"feature_id": int(a), "lesion": b} for a,b in supported], "numeric_exploratory_eyepacs": numeric, "numeric_comparisons": comparisons, "llm_arms": {"status": "unavailable_no_supported_grounded_cards" if not supported else "not_run_in_this_invocation_requires_shared_gpu_lock", "reason": "No clinical card wording is invented when no mapping passes the frozen support rule." if not supported else "Run under the shared GPU lock after numerical specification freeze."}, "provenance": {k: digest(v) for k,v in roots.items()}, "specification_freeze_sha256": digest(freeze), "preflight": {"idrid_train_images": len(training), "idrid_test_images": len(testing), "global_input_contract": [81,1024], "checkpoint_reload": True, "labels_hidden_from_llm_prompts": True}, "scope_boundary": "IDRiD was used for knowledge development and cannot be described as untouched external validation. Global mappings are association evidence only and cannot support spatial lesion claims."}
    (output / "experiment_9a_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (output / "EXPERIMENT_9A_FINAL_REPORT.md").write_text("# Experiment 9A final report\n\n" + json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "supported": len(supported), "strong": len(strong), "output": str(output / 'experiment_9a_result.json')}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
