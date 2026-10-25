"""Resumable matched A/B/C LoRA training with durable per-step map records.

Requires an independently validated map-set receipt. This runner only opens
training images; validation and test inference are separate Slurm stages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F

from profile_retinavlm_olives_lora import add_lora, load_aligned_model
from retinavlm_batch_records import BatchRecorder
from retinavlm_binary_answer import answer_loss, answer_token_ids
from retinavlm_olives_report import REPORT_QUESTION, report_target
from retinavlm_pre_answer import pre_answer_forward, pre_answer_forward_layers
from retinavlm_projected_jacobian import projected_score
from train_retinavlm_adapter_olives import QUESTIONS, SPLIT_SHA, load_image, load_rows, sha256


def load_map_set(path: Path, adapter_sha: str):
    receipt = json.loads(path.read_text())
    if (receipt.get("status") != "validated_retinavlm_olives_map_set"
            or receipt.get("split_sha256") != SPLIT_SHA
            or receipt.get("adapter_sha256") != adapter_sha
            or not isinstance(receipt.get("findings"), dict)
            or not receipt["findings"]):
        raise ValueError("Map set has not passed the aligned-model validation gate")
    maps = {}
    for finding, entry in receipt["findings"].items():
        if finding not in QUESTIONS:
            raise ValueError(f"Unexpected map finding: {finding}")
        directory = Path(entry["directory"])
        map_file = directory / "map_and_probe.pt"
        summary_file = directory / "summary.json"
        gate_file = directory / "validation_gate.json"
        statistical_file = directory / "statistical_gate.json"
        intervention_file = directory / "state_intervention.json"
        if any(not item.is_file() for item in (map_file, summary_file, gate_file,
                                               statistical_file, intervention_file)):
            raise FileNotFoundError(f"Incomplete map bundle for {finding}")
        if (sha256(map_file) != entry["map_sha256"]
                or sha256(summary_file) != entry["summary_sha256"]
                or sha256(gate_file) != entry["gate_sha256"]):
            raise ValueError(f"Map bundle SHA mismatch for {finding}")
        summary = json.loads(summary_file.read_text())
        gate = json.loads(gate_file.read_text())
        statistical = json.loads(statistical_file.read_text())
        intervention = json.loads(intervention_file.read_text())
        layers = tuple(entry["layers"])
        weights = tuple(float(value) for value in entry["weights"])
        if (summary.get("status") != "candidate_map_fitted_validation_pending_gate"
                or summary.get("finding") != finding
                or summary.get("inference_batch_size") != 1
                or summary.get("adapter_sha256") != adapter_sha
                or summary.get("split_sha256") != SPLIT_SHA
                or gate.get("status") != "validated_image_conditioned_map"
                or gate.get("finding") != finding
                or gate.get("map_sha256") != entry["map_sha256"]
                or gate.get("summary_sha256") != entry["summary_sha256"]
                or gate.get("statistical_gate_sha256") != sha256(statistical_file)
                or gate.get("state_intervention_sha256") != sha256(intervention_file)
                or gate.get("split_sha256") != SPLIT_SHA
                or gate.get("adapter_sha256") != adapter_sha
                or gate.get("test_accessed") is not False
                or statistical.get("status") != "candidate_statistical_gate_passed_needs_intervention"
                or intervention.get("status") not in {
                    "candidate_answer_intervention_passed_needs_final_audit",
                    "candidate_answer_intervention_failed"}
                or statistical.get("map_sha256") != entry["map_sha256"]
                or intervention.get("map_sha256") != entry["map_sha256"]
                or intervention.get("reference_batch_size") != 1
                or intervention.get("statistical_gate_sha256") != sha256(statistical_file)
                or intervention.get("adapter_sha256") != adapter_sha
                or intervention.get("selected_layers") != statistical.get("selected_layers")
                or any(layer not in statistical.get("selected_layers", []) for layer in layers)
                or any(not intervention.get("decisions", {}).get(str(layer), {}).get(
                    "answer_influence_gate") for layer in layers)
                or tuple(gate.get("selected_layers", [])) != layers
                or len(layers) < 2 or len(weights) != len(layers)
                or any(layer not in summary["source_layers"] for layer in layers)
                or any(weight < 0 for weight in weights)
                or abs(sum(weights) - 1.0) > 1e-6):
            raise ValueError(f"Map gate/selected layers invalid for {finding}")
        payload = torch.load(map_file, map_location="cpu")
        directions = payload["directions"]
        maps[finding] = {
            "layers": layers, "weights": weights,
            "directions": {layer: directions[layer].float() for layer in layers},
            "calibration": {layer: summary["calibration"][str(layer)] for layer in layers},
            "version": entry["map_sha256"],
        }
    return maps, sha256(path)


def patient_balanced_index(rows):
    index = {finding: {label: defaultdict(list) for label in (0, 1)}
             for finding in QUESTIONS}
    for row in rows:
        for finding in QUESTIONS:
            if row[finding] in {"0", "1"}:
                index[finding][int(row[finding])][row["patient_key"]].append(row)
    for finding in QUESTIONS:
        if any(not index[finding][label] for label in (0, 1)):
            raise ValueError(f"Finding {finding} lacks a training class")
    return index


def scheduled_finding(step, mapped):
    """Half the steps target a mapped finding; half cycle all seven."""
    if step % 2:
        return mapped[((step - 1) // 2) % len(mapped)]
    return tuple(QUESTIONS)[((step // 2) - 1) % len(QUESTIONS)]


def sample_patient_row(index, finding, label, rng):
    patients = sorted(index[finding][label])
    patient = rng.choice(patients)
    return rng.choice(index[finding][label][patient])


def pseudonym(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def atomic_checkpoint(path, payload):
    temporary = path.with_name(path.name + ".writing")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def run(args):
    if (args.arm not in {"A", "B", "C"} or args.total_steps < 1
            or not 1 <= args.stop_step <= args.total_steps or args.save_every < 1
            or args.auxiliary_weight < 0 or args.report_weight < 0):
        raise ValueError("Invalid matched-arm run arguments")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    groups = load_rows(args.split_csv)
    train = groups["train"]
    for row in train:
        path = args.train_images / row["source_image_path"].lstrip("/")
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
    adapter_sha = sha256(args.adapter_checkpoint)
    maps, map_set_sha = load_map_set(args.map_set, adapter_sha)
    mapped = tuple(sorted(maps))
    index = patient_balanced_index(train)
    run_config = {
        "format": "retinavlm_olives_abc_lora_v1",
        "arm": args.arm, "seed": args.seed,
        "total_steps": args.total_steps, "rank": args.rank, "alpha": args.alpha,
        "learning_rate": args.lr, "report_weight": args.report_weight,
        "auxiliary_weight": args.auxiliary_weight,
        "save_every": args.save_every,
        "sampling": "odd steps cycle mapped findings, even steps cycle all seven; alternating labels and uniform patient within finding/label; every batch reads one mapped finding under its exact prompt, separately when needed",
        "split_sha256": SPLIT_SHA, "adapter_sha256": adapter_sha,
        "map_set_sha256": map_set_sha, "mapped_findings": mapped,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.out_dir / "run_config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != json.loads(json.dumps(run_config)):
            raise ValueError("Resume run configuration changed")
    else:
        config_path.write_text(json.dumps(run_config, indent=2) + "\n")

    random.seed(args.seed)
    sampler_rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = load_aligned_model(args)
    mini = model.model
    lora_parameters = add_lora(mini, args.rank, args.alpha)
    torch.cuda.reset_peak_memory_stats()
    mini.eval()
    mini.llama_model.train()
    width = mini.llama_model.config.hidden_size
    probes = torch.nn.ModuleDict()
    if args.arm == "B":
        for finding, entry in maps.items():
            for layer in entry["layers"]:
                probes[f"{finding}_{layer}"] = torch.nn.Linear(width, 1)
        probes.to(mini.llama_model.device)
    parameters = lora_parameters + list(probes.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.lr)
    ids = answer_token_ids(mini.get_tokenizer(), QUESTIONS["B5"])
    latest = args.out_dir / "latest.pt"
    start = 0
    if latest.exists():
        old = torch.load(latest, map_location="cpu")
        if old.get("run_config") != run_config:
            raise ValueError("Resume checkpoint provenance mismatch")
        named = dict(mini.llama_model.named_parameters())
        if set(old["lora_parameters"]) != {name for name, p in named.items() if p.requires_grad}:
            raise ValueError("LoRA parameter names changed on resume")
        for name, value in old["lora_parameters"].items():
            named[name].data.copy_(value.to(named[name].device, dtype=named[name].dtype))
        if args.arm == "B":
            probes.load_state_dict(old["probe_state"], strict=True)
        optimizer.load_state_dict(old["optimizer"])
        sampler_rng.setstate(old["python_rng"])
        torch.set_rng_state(old["torch_rng"])
        torch.cuda.set_rng_state_all(old["cuda_rng"])
        start = old["step"]
    if start >= args.stop_step:
        raise ValueError("Requested segment stop must exceed resume checkpoint")
    record_id = f"{args.arm}_seed{args.seed}_{map_set_sha[:12]}"
    start_time = time.monotonic()
    last_checkpoint_step = start
    auxiliary_gradient_norm_step1 = None
    probe_gradient_norm_step1 = None
    window = {"answer": 0.0, "report": 0.0, "auxiliary": 0.0, "total": 0.0}
    with BatchRecorder(args.out_dir, run_id=record_id, resume_step=start) as recorder:
        for step in range(start + 1, args.stop_step + 1):
            finding = scheduled_finding(step, mapped)
            label = ((step - 1) // 2) % 2
            row = sample_patient_row(index, finding, label, sampler_rng)
            image = load_image(model, args.train_images, row).unsqueeze(0)
            samples = {"Image": image, "Question": [QUESTIONS[finding]], "Answer": [""]}
            monitor_finding = finding if finding in maps else mapped[(step - 1) % len(mapped)]
            selected = maps[monitor_finding]
            optimizer.zero_grad(set_to_none=True)
            if monitor_finding == finding:
                logits, states = pre_answer_forward_layers(mini, samples, selected["layers"])
                monitor_logits = logits
            else:
                logits, _ = pre_answer_forward(mini, samples)
                monitor_samples = {"Image": image,
                                   "Question": [QUESTIONS[monitor_finding]], "Answer": [""]}
                if args.arm == "A":
                    with torch.inference_mode():
                        monitor_logits, states = pre_answer_forward_layers(
                            mini, monitor_samples, selected["layers"])
                else:
                    monitor_logits, states = pre_answer_forward_layers(
                        mini, monitor_samples, selected["layers"])
            target = torch.tensor([label], dtype=torch.long, device=logits.device)
            answer = answer_loss(logits, target, *ids)
            trained_answer_margin = float((logits[0, ids[1]]
                                           - logits[0, ids[0]]).detach())
            answer_margin = float((monitor_logits[0, ids[1]]
                                   - monitor_logits[0, ids[0]]).detach())
            monitor_label = (int(row[monitor_finding])
                             if row[monitor_finding] in {"0", "1"} else None)
            report_text = report_target(row)
            report_ids = mini.get_tokenizer().encode(
                report_text + mini.llm.stop_words[0], add_special_tokens=False)
            if len(report_ids) > mini.max_txt_len:
                raise ValueError(f"Report target too long at step {step}")
            report_samples = {"Image": image, "Question": [REPORT_QUESTION],
                              "Answer": [report_text]}
            inputs, _, mask, targets = mini.form_input(report_samples)
            report = mini.llama_model(inputs_embeds=inputs, attention_mask=mask,
                                      labels=targets, use_cache=False, return_dict=True).loss
            auxiliary_terms = []
            sample_scores = []
            for layer, weight in zip(selected["layers"], selected["weights"], strict=True):
                hidden = states[layer]
                raw = projected_score(hidden, selected["directions"][layer])
                calibration = selected["calibration"][layer]
                calibrated = (float(calibration["scale"]) * raw
                              + float(calibration["intercept"]))
                if args.arm == "B":
                    auxiliary_logit = probes[f"{monitor_finding}_{layer}"](hidden.float()).squeeze(-1)
                else:
                    auxiliary_logit = calibrated
                if args.arm in {"B", "C"} and monitor_label is not None:
                    map_target = torch.tensor([monitor_label], dtype=torch.float32,
                                              device=auxiliary_logit.device)
                    auxiliary_terms.append(
                        weight * F.binary_cross_entropy_with_logits(
                            auxiliary_logit, map_target))
                sample_scores.append({
                    "sample_key": pseudonym(row["source_image_path"]),
                    "patient_key": pseudonym(row["patient_key"]),
                    "trained_finding": finding, "trained_label": label,
                    "trained_answer_margin": trained_answer_margin,
                    "finding": monitor_finding, "expert_label": monitor_label,
                    "layer": layer, "raw_map_score": float(raw.detach()),
                    "calibrated_map_logit": float(calibrated.detach()),
                    "diagnostic_threshold": 0.0,
                    "answer_margin": answer_margin, "final_answer": None,
                    "map_version": selected["version"],
                })
            auxiliary = sum(auxiliary_terms) if auxiliary_terms else torch.zeros_like(answer)
            if step == 1 and args.arm == "C":
                if not auxiliary_terms:
                    raise ValueError("C preflight batch lacks a verified map target")
                gradients = torch.autograd.grad(
                    auxiliary, lora_parameters, retain_graph=True, allow_unused=True)
                auxiliary_gradient_norm_step1 = sum(
                    float(gradient.float().norm()) for gradient in gradients
                    if gradient is not None)
                if (not math.isfinite(auxiliary_gradient_norm_step1)
                        or auxiliary_gradient_norm_step1 <= 0):
                    raise ValueError("J-Lens auxiliary loss did not reach LoRA")
            total = answer + args.report_weight * report + args.auxiliary_weight * auxiliary
            if not all(torch.isfinite(item) for item in (answer, report, auxiliary, total)):
                raise FloatingPointError(f"Nonfinite loss at step {step}")
            total.backward()
            if step == 1 and args.arm == "B":
                probe_gradient_norm_step1 = sum(
                    float(parameter.grad.float().norm()) for parameter in probes.parameters()
                    if parameter.grad is not None)
                if (not math.isfinite(probe_gradient_norm_step1)
                        or probe_gradient_norm_step1 <= 0):
                    raise ValueError("Learned-probe auxiliary loss did not reach probe head")
            gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"Nonfinite gradient at step {step}")
            optimizer.step()
            values = {"answer": float(answer.detach()), "report": float(report.detach()),
                      "auxiliary": float(auxiliary.detach()), "total": float(total.detach())}
            for key in window:
                window[key] += values[key]
            recorder.append({
                "run_id": record_id, "arm": args.arm, "seed": args.seed,
                "global_step": step, "checkpoint_version": last_checkpoint_step,
                "map_version": map_set_sha, "patient_count": 1, "image_count": 1,
                "verified_positive_targets": sum(row[f] == "1" for f in QUESTIONS),
                "verified_negative_targets": sum(row[f] == "0" for f in QUESTIONS),
                "unknown_targets": sum(row[f] not in {"0", "1"} for f in QUESTIONS),
                "learning_rate": args.lr, "total_loss": values["total"],
                "answer_loss": values["answer"], "report_loss": values["report"],
                "auxiliary_loss": values["auxiliary"] if args.arm != "A" else None,
                "elapsed_seconds": time.monotonic() - start_time,
                "sample_layer_scores": sample_scores,
            })
            if step % args.save_every == 0 or step == args.stop_step:
                named = {name: p.detach().cpu()
                         for name, p in mini.llama_model.named_parameters() if p.requires_grad}
                payload = {
                    "run_config": run_config, "step": step, "lora_parameters": named,
                    "probe_state": probes.state_dict() if args.arm == "B" else None,
                    "optimizer": optimizer.state_dict(),
                    "python_rng": sampler_rng.getstate(), "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(),
                }
                snapshot = args.out_dir / f"checkpoint_step_{step:06d}.pt"
                if snapshot.exists():
                    archive = args.out_dir / f"orphan_snapshot_step_{step:06d}.pt"
                    if archive.exists():
                        raise FileExistsError(f"Repeated checkpoint recovery needs review: {archive}")
                    os.replace(snapshot, archive)
                atomic_checkpoint(snapshot, payload)
                # Keep a stable copy for validation curves and selection while
                # the resumable latest pointer advances to later steps.
                latest_link = latest.with_name(latest.name + ".writing")
                if latest_link.exists():
                    latest_link.unlink()
                os.link(snapshot, latest_link)
                os.replace(latest_link, latest)
                last_checkpoint_step = step
                divisor = min(args.save_every, step - start)
                print(json.dumps({
                    "arm": args.arm, "step": step, "segment_stop": args.stop_step,
                    "mean_losses": {key: value / divisor for key, value in window.items()},
                    "elapsed_seconds": time.monotonic() - start_time,
                    "checkpoint": str(latest), "snapshot": str(snapshot),
                }), flush=True)
                window = {key: 0.0 for key in window}
    receipt = {
        "status": "matched_abc_training_complete" if args.stop_step == args.total_steps
                  else "matched_abc_segment_complete",
        "arm": args.arm, "seed": args.seed, "step": args.stop_step,
        "total_steps": args.total_steps, "run_config_sha256": sha256(config_path),
        "checkpoint_sha256": sha256(latest),
        "batch_records_sha256": sha256(args.out_dir / "train_batches.jsonl"),
        "trainable_lora_parameters": sum(p.numel() for p in lora_parameters),
        "extra_probe_parameters": sum(p.numel() for p in probes.parameters()),
        "auxiliary_gradient_norm_step1": auxiliary_gradient_norm_step1,
        "probe_gradient_norm_step1": probe_gradient_norm_step1,
        "validation_images_accessed": False, "test_images_accessed": False,
        "validation_targets_used": False, "test_targets_used": False,
        "max_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "gpu": torch.cuda.get_device_name(0),
    }
    (args.out_dir / f"segment_{args.stop_step:06d}_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("split-csv", "train-images", "model-dir", "authors-code",
                 "adapted-checkpoint", "adapter-checkpoint", "map-set", "out-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--arm", choices=["A", "B", "C"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--report-weight", type=float, default=1.0)
    parser.add_argument("--auxiliary-weight", type=float, default=0.1)
    parser.add_argument("--total-steps", type=int, default=6000)
    parser.add_argument("--stop-step", type=int, required=True)
    parser.add_argument("--save-every", type=int, default=500)
    run(parser.parse_args())
