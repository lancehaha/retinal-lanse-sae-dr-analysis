"""Profile one matched RetinaVLM LoRA answer-plus-report gradient step."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from evaluate_retinavlm_adapter_olives import image_batch
from retinavlm_binary_answer import answer_loss, answer_token_ids
from retinavlm_pre_answer import pre_answer_forward
from retinavlm_released_model import load_released_specialist
from train_retinavlm_adapter_olives import ADAPTED_SHA, QUESTIONS, SPLIT_SHA, install_encoder, load_rows, sha256


FINDING_NAMES = {
    "B1": "intraretinal hyperreflective foci",
    "B2": "partially attached vitreous face",
    "B3": "fully attached vitreous face",
    "B4": "intraretinal fluid",
    "B5": "diffuse retinal thickening or macular edema",
    "B6": "vitreous debris",
    "SRF": "subretinal fluid",
}
REPORT_QUESTION = "Describe the verified retinal OCT findings briefly."


def report_target(row):
    parts = [f"{FINDING_NAMES[finding]}: {'present' if row[finding] == '1' else 'absent'}"
             for finding in QUESTIONS if row[finding] in {"0", "1"}]
    if not parts:
        raise ValueError("No verified finding labels for report target")
    return " " + "; ".join(parts) + "."


def load_aligned_model(args):
    checkpoint = torch.load(args.adapter_checkpoint, map_location="cpu")
    if (checkpoint.get("format") != "retinavlm_olives_adapter_alignment_v1"
            or checkpoint.get("variant") != "adapted_byol"
            or checkpoint.get("step") != 10000
            or checkpoint.get("split_sha256") != SPLIT_SHA
            or checkpoint.get("adapted_encoder_sha256") != ADAPTED_SHA):
        raise ValueError("Expected complete aligned BYOL adapter checkpoint")
    model, _ = load_released_specialist(args.model_dir, args.authors_code)
    install_encoder(model, "adapted_byol", args.adapted_checkpoint)
    model.model.llama_proj.load_state_dict(checkpoint["adapter"], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def add_lora(mini, rank, alpha):
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    if rank <= 0 or alpha <= 0:
        raise ValueError("LoRA rank and alpha must be positive")
    base = prepare_model_for_kbit_training(mini.llama_model, use_gradient_checkpointing=False)
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=rank, lora_alpha=alpha,
        target_modules=["q_proj", "v_proj"], lora_dropout=0.0, bias="none")
    mini.llama_model = get_peft_model(base, config)
    parameters = [p for p in mini.llama_model.parameters() if p.requires_grad]
    if not parameters or any(p.requires_grad for p in mini.visual_encoder.parameters()):
        raise ValueError("LoRA parameter policy failed")
    if any(p.requires_grad for p in mini.llama_proj.parameters()):
        raise ValueError("Aligned adapter must stay frozen")
    return parameters


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    groups = load_rows(args.split_csv)
    row = next(r for r in groups["train"] if r["B5"] in {"0", "1"})
    model = load_aligned_model(args)
    mini = model.model
    parameters = add_lora(mini, args.rank, args.alpha)
    mini.eval()
    mini.llama_model.train()
    optimizer = torch.optim.AdamW(parameters, lr=args.lr)
    ids = answer_token_ids(mini.get_tokenizer(), QUESTIONS["B5"])
    image = image_batch(model, [row], args.train_images)
    label = torch.tensor([int(row["B5"])], dtype=torch.long, device=image.device)
    target_text = report_target(row)
    target_tokens = mini.get_tokenizer().encode(
        target_text + mini.llm.stop_words[0], add_special_tokens=False)
    if len(target_tokens) > mini.max_txt_len:
        raise ValueError(f"Report target has {len(target_tokens)} tokens, limit {mini.max_txt_len}")
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    answer_samples = {"Image": image, "Question": [QUESTIONS["B5"]], "Answer": [""]}
    answer_logits, _ = pre_answer_forward(mini, answer_samples)
    answer = answer_loss(answer_logits, label, *ids)
    report_samples = {"Image": image, "Question": [REPORT_QUESTION],
                      "Answer": [target_text]}
    inputs, _, attention_mask, targets = mini.form_input(report_samples)
    report = mini.llama_model(inputs_embeds=inputs, attention_mask=attention_mask,
                              labels=targets, use_cache=False, return_dict=True).loss
    loss = answer + args.report_weight * report
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite matched LoRA preflight loss")
    loss.backward()
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0))
    if not torch.isfinite(torch.tensor(gradient_norm)) or gradient_norm <= 0:
        raise ValueError("LoRA gradient did not reach the selected parameters")
    optimizer.step()
    torch.cuda.synchronize()
    result = {
        "status": "matched_lora_one_step_preflight_complete",
        "split_sha256": SPLIT_SHA, "adapter_sha256": sha256(args.adapter_checkpoint),
        "adapted_encoder_sha256": ADAPTED_SHA,
        "trainable_lora_parameters": sum(p.numel() for p in parameters),
        "rank": args.rank, "alpha": args.alpha, "lr": args.lr,
        "report_weight": args.report_weight,
        "target_modules": ["q_proj", "v_proj"],
        "report_target_tokens": len(target_tokens),
        "answer_loss": float(answer.detach()), "report_loss": float(report.detach()),
        "total_loss": float(loss.detach()), "gradient_norm": gradient_norm,
        "seconds": time.monotonic() - start,
        "max_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "gpu": torch.cuda.get_device_name(0), "test_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("split-csv", "train-images", "model-dir", "authors-code",
                 "adapted-checkpoint", "adapter-checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--report-weight", type=float, default=1.0)
    run(parser.parse_args())
