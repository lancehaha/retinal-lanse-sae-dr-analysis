"""Single-GPU F/B/C/D runner. Never selects on test data."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .adapter import build_case, load_model
from .lens import auxiliary_loss, concept_score, estimate_direction, fit_calibrator
from .perturb import paired_blurs
from .protocol import PROMPT, SEEDS, audit_manifest, manifest_sha256, read_jsonl, target_report, verified_label, write_jsonl

CHECKPOINT_REVISION = "53bca09462f02c812cf56144105a21c7ae7bfc7b"
READOUT_PREFIX = '{"hard_exudates":"'


def exact_token(tokenizer, term: str) -> int:
    ids = tokenizer.encode(term, add_special_tokens=False)
    prefix_ids = tokenizer.encode(READOUT_PREFIX, add_special_tokens=False)
    joint_ids = tokenizer.encode(READOUT_PREFIX + term, add_special_tokens=False)
    if len(ids) != 1 or joint_ids != prefix_ids + ids:
        raise ValueError(f"J-Lens term {term!r} must be one token after the fixed readout prefix")
    return ids[0]


def prep(model, tokenizer, row: dict, upstream: str, answer: bool = False) -> dict:
    return build_case(model, tokenizer, row["image_path"], target_report(row) if answer else None,
                      upstream_src=upstream)


def with_readout_prefix(case: dict, tokenizer) -> dict:
    """Append only the fixed JSON key; never include the ground-truth value."""
    prefix_ids = tokenizer.encode(READOUT_PREFIX, add_special_tokens=False)
    tokens = torch.tensor([prefix_ids], dtype=case["input_ids"].dtype, device=case["input_ids"].device)
    result = dict(case)
    result["input_ids"] = torch.cat((case["input_ids"], tokens), dim=1)
    result["attention_mask"] = torch.ones_like(result["input_ids"])
    result["readout_position"] = result["input_ids"].shape[1] - 1
    return result


def score_state(model, case: dict, layer: int, direction: torch.Tensor) -> float:
    with torch.no_grad():
        result = model(**{k: case[k] for k in ("pixel_values", "input_ids", "attention_mask", "image_flags")},
                       output_hidden_states=True, return_dict=True, use_cache=False)
    hidden = result.hidden_states[layer][0, case["pre_answer_position"]]
    return float(concept_score(hidden, direction).item())


def auc(scores: list[float], labels: list[int]) -> float:
    positives = [s for s, y in zip(scores, labels) if y == 1]
    negatives = [s for s, y in zip(scores, labels) if y == 0]
    if not positives or not negatives:
        raise ValueError("Both verified classes are required")
    return sum((a > b) + 0.5 * (a == b) for a in positives for b in negatives) / (len(positives) * len(negatives))


def preflight(args, rows: list[dict], model, tokenizer, out: Path) -> dict:
    token_id = exact_token(tokenizer, args.term)
    train = [r for r in rows if r["split"] == "train" and verified_label(r) is not None]
    valid = [r for r in rows if r["split"] == "validation" and verified_label(r) is not None]
    if len(train) < 2 or len(valid) < 2:
        raise ValueError("Preflight requires verified train and validation positives and negatives")
    sample = random.Random(4106).sample(train, min(args.direction_examples, len(train)))
    direction = estimate_direction(model, [with_readout_prefix(prep(model, tokenizer, r, args.upstream), tokenizer)
                                           for r in sample],
                                   args.layer, token_id)
    train_scores = [score_state(model, prep(model, tokenizer, r, args.upstream), args.layer, direction)
                    for r in train]
    labels = [verified_label(r) for r in train]
    slope, intercept = fit_calibrator(torch.tensor(train_scores), torch.tensor(labels))
    val_scores = [score_state(model, prep(model, tokenizer, r, args.upstream), args.layer, direction)
                  for r in valid]
    val_labels = [verified_label(r) for r in valid]
    val_auc = auc(val_scores, val_labels)
    perturb_rows = [r for r in valid if verified_label(r) == 1 and r.get("hard_exudate_mask_path")]
    selected_perturb_rows = random.Random(4106).sample(
        perturb_rows, min(args.perturb_examples, len(perturb_rows)))
    pairs, perturb_errors = [], []
    for row in selected_perturb_rows:
        try:
            lesion_img, background_img = paired_blurs(
                row["image_path"], row["hard_exudate_mask_path"], row["image_id"])
        except ValueError as error:
            perturb_errors.append({"image_id": row["image_id"], "reason": str(error)})
            continue
        baseline = val_scores[valid.index(row)]
        lesion_score = score_state(model, build_case(model, tokenizer, lesion_img, upstream_src=args.upstream),
                                   args.layer, direction)
        background_score = score_state(model, build_case(model, tokenizer, background_img, upstream_src=args.upstream),
                                       args.layer, direction)
        pairs.append({"image_id": row["image_id"], "lesion_change": abs(lesion_score - baseline),
                      "background_change": abs(background_score - baseline)})
    lesion_mean = float(np.mean([r["lesion_change"] for r in pairs])) if pairs else None
    background_mean = float(np.mean([r["background_change"] for r in pairs])) if pairs else None
    perturb_pass = len(pairs) >= args.min_perturb_pairs and lesion_mean > background_mean
    payload = {"layer": args.layer, "term": args.term, "token_id": token_id,
               "readout_prefix": READOUT_PREFIX,
               "readout_position": "future status token after fixed label-free prefix",
               "train_count": len(train), "validation_count": len(valid), "validation_auc": val_auc,
               "slope": slope, "intercept": intercept,
               "discrimination_pass": val_auc >= args.min_auc,
               "perturbation_pass": perturb_pass, "perturbation_pairs": pairs,
               "perturbation_errors": perturb_errors,
               "lesion_mean_abs_change": lesion_mean, "background_mean_abs_change": background_mean,
               "usable_for_supervision": val_auc >= args.min_auc and perturb_pass}
    torch.save(direction, out / "direction.pt")
    (out / "calibration.json").write_text(json.dumps(payload, indent=2))
    write_jsonl(out / "validation_scores.jsonl", [
        {"image_id": r["image_id"], "label": y, "score": s}
        for r, y, s in zip(valid, val_labels, val_scores)])
    return payload


def generate_report(model, tokenizer, row: dict, upstream: str, max_new_tokens: int = 128) -> str:
    case = prep(model, tokenizer, row, upstream)
    from internvl.conversation import get_conv_template
    eos = tokenizer.convert_tokens_to_ids(get_conv_template(model.template).sep.strip())
    kwargs = {"pixel_values": case["pixel_values"], "input_ids": case["input_ids"],
              "attention_mask": case["attention_mask"], "max_new_tokens": max_new_tokens,
              "do_sample": False}
    if eos != tokenizer.unk_token_id:
        kwargs["eos_token_id"] = eos
    with torch.no_grad():
        output = model.generate(**kwargs)
    return tokenizer.decode(output[0], skip_special_tokens=True).split(get_conv_template(model.template).sep.strip())[0].strip()


def train(args, rows: list[dict], model, tokenizer, out: Path) -> dict:
    if args.arm not in {"B", "C", "D"}:
        raise ValueError("Training arm must be B, C or D")
    if args.arm == "D":
        if not args.calibration:
            raise ValueError("D requires --calibration from a passing preflight")
        calibration = json.loads(Path(args.calibration).read_text())
        if not calibration.get("usable_for_supervision"):
            raise ValueError("D requires a signed-off preflight with both discrimination and perturbation gates")
        direction = torch.load(Path(args.calibration).parent / "direction.pt", map_location="cuda", weights_only=True)
        slope, intercept = calibration["slope"], calibration["intercept"]
    else:
        direction = None
    train_rows = [r for r in rows if r["split"] == "train"]
    if not train_rows:
        raise ValueError("No training images")
    probe = None
    if args.arm == "C":
        dim = model.language_model.config.hidden_size
        probe = torch.nn.Linear(dim, 1).to("cuda", dtype=torch.float32)
    params = [p for p in model.parameters() if p.requires_grad]
    if probe is not None:
        params += list(probe.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    losses = []
    drift = []
    for epoch in range(args.epochs):
        if args.arm == "D" and epoch:
            sample = random.Random(4106).sample(train_rows, min(args.direction_examples, len(train_rows)))
            refreshed = estimate_direction(model, [with_readout_prefix(prep(model, tokenizer, r, args.upstream), tokenizer)
                                                   for r in sample],
                                           args.layer, calibration["token_id"]).to("cuda")
            drift.append(float(F.cosine_similarity(direction.float(), refreshed.float(), dim=0)))
            direction = refreshed
        random.Random(args.seed + epoch).shuffle(train_rows)
        model.train()
        for row in train_rows:
            case = prep(model, tokenizer, row, args.upstream, answer=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(**{k: case[k] for k in ("pixel_values", "input_ids", "attention_mask", "image_flags", "labels")},
                           output_hidden_states=args.arm != "B", return_dict=True, use_cache=False)
            loss = output.loss
            label = verified_label(row)
            if args.arm != "B" and label is not None:
                state = output.hidden_states[args.layer][0, case["pre_answer_position"]]
                if args.arm == "C":
                    aux = F.binary_cross_entropy_with_logits(probe(state.float()).reshape(1),
                                                             torch.tensor([float(label)], device="cuda"))
                else:
                    aux = auxiliary_loss(state[None], direction[None],
                                         torch.tensor([label], device="cuda"), slope, intercept)
                loss = loss + args.loss_weight * aux
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
    model.language_model.save_pretrained(out / "adapter")
    if probe is not None:
        torch.save(probe.state_dict(), out / "probe.pt")
    if direction is not None:
        torch.save(direction.cpu(), out / "direction.pt")
    result = {"arm": args.arm, "seed": args.seed, "epochs": args.epochs, "updates": len(losses),
              "mean_loss": float(np.mean(losses)), "direction_cosine_after_refresh": drift,
              "loss_weight": args.loss_weight if args.arm != "B" else 0.0,
              "expected_checkpoint_revision": CHECKPOINT_REVISION,
              "manifest_sha256": manifest_sha256(args.manifest),
              "prompt": PROMPT}
    (out / "run.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["preflight", "train", "predict"])
    p.add_argument("--arm", choices=["F", "B", "C", "D"], default="F")
    p.add_argument("--model", required=True)
    p.add_argument("--upstream", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--adapter")
    p.add_argument("--split", choices=["validation", "test"], default="test")
    p.add_argument("--calibration")
    p.add_argument("--layer", type=int, default=16)
    p.add_argument("--term", default="present")
    p.add_argument("--direction-examples", type=int, default=16)
    p.add_argument("--min-auc", type=float, default=0.60)
    p.add_argument("--perturb-examples", type=int, default=12)
    p.add_argument("--min-perturb-pairs", type=int, default=6)
    p.add_argument("--seed", type=int, choices=SEEDS, default=SEEDS[0])
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--loss-weight", type=float, default=0.1)
    args = p.parse_args()
    rows = read_jsonl(args.manifest)
    audit_manifest(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Seed before LoRA and probe construction so each run is reproducible.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model, tokenizer = load_model(args.model, args.upstream, train=args.command == "train")
    if args.adapter:
        from peft import PeftModel
        model.language_model = PeftModel.from_pretrained(model.language_model, args.adapter)
    if args.command == "preflight":
        result = preflight(args, rows, model, tokenizer, out)
    elif args.command == "train":
        result = train(args, rows, model, tokenizer, out)
    else:
        if args.arm != "F" and not args.adapter:
            raise ValueError("Fine-tuned prediction requires --adapter")
        predictions = [{"image_id": r["image_id"], "split": args.split, "arm": args.arm,
                        "seed": args.seed, "report": generate_report(model, tokenizer, r, args.upstream)}
                       for r in rows if r["split"] == args.split]
        write_jsonl(out / "predictions.jsonl", predictions)
        result = {"predictions": len(predictions)}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
