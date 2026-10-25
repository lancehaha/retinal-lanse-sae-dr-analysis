"""Minimal single-case input adapter for the upstream InternVL implementation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .protocol import PROMPT

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_image_tiles(path: str | Image.Image, size: int = 448, max_tiles: int = 6) -> torch.Tensor:
    """Fixed dynamic-tiling policy, matching the upstream crop/thumbnail order."""
    image = (path if isinstance(path, Image.Image) else Image.open(path)).convert("RGB")
    width, height = image.size
    ratios = sorted({(i, j) for n in range(1, max_tiles + 1)
                     for i in range(1, n + 1) for j in range(1, n + 1) if i * j == n},
                    key=lambda pair: pair[0] * pair[1])
    cols, rows = (1, 1)
    best_diff = float("inf")
    for pair in ratios:
        diff = abs(pair[0] / pair[1] - width / height)
        if diff < best_diff or (diff == best_diff and width * height > 0.5 * size * size * pair[0] * pair[1]):
            cols, rows = pair
            best_diff = diff
    resized = image.resize((cols * size, rows * size))
    tiles = [resized.crop((x * size, y * size, (x + 1) * size, (y + 1) * size))
             for y in range(rows) for x in range(cols)]
    if len(tiles) > 1:
        tiles.append(image.resize((size, size)))
    arrays = [((np.asarray(tile, dtype=np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)
              for tile in tiles]
    return torch.from_numpy(np.stack(arrays))


def build_case(model, tokenizer, image_path: str | Image.Image, answer: str | None = None,
               upstream_src: str | None = None, device: str = "cuda") -> dict:
    if upstream_src:
        sys.path.insert(0, str(Path(upstream_src) / "internvl_chat"))
    from internvl.conversation import get_conv_template

    pixels = load_image_tiles(image_path)
    image_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    if image_token_id is None or image_token_id == tokenizer.unk_token_id:
        raise ValueError("FundusExpert tokenizer lacks <IMG_CONTEXT>")
    model.img_context_token_id = image_token_id
    template = get_conv_template(model.template)
    template.system_message = model.system_message
    template.append_message(template.roles[0], "<image>\n" + PROMPT)
    template.append_message(template.roles[1], None)
    prompt = template.get_prompt().replace(
        "<image>", "<img>" + "<IMG_CONTEXT>" * model.num_image_token * len(pixels) + "</img>", 1)
    prompt_ids = tokenizer(prompt).input_ids
    if sum(token == image_token_id for token in prompt_ids) != model.num_image_token * len(pixels):
        raise ValueError("Image token count does not match visual tiles")
    ids = list(prompt_ids)
    labels = None
    if answer is not None:
        answer_ids = tokenizer(answer + template.sep.strip(), add_special_tokens=False).input_ids
        ids.extend(answer_ids)
        labels = [-100] * len(prompt_ids) + answer_ids
    result = {
        "pixel_values": pixels.to(device=device, dtype=torch.bfloat16 if device == "cuda" else torch.float32),
        "input_ids": torch.tensor([ids], dtype=torch.long, device=device),
        "attention_mask": torch.ones((1, len(ids)), dtype=torch.long, device=device),
        "image_flags": torch.ones((len(pixels), 1), dtype=torch.long, device=device),
        "pre_answer_position": len(prompt_ids) - 1,
    }
    if labels is not None:
        result["labels"] = torch.tensor([labels], dtype=torch.long, device=device)
    return result


def load_model(model_path: str, upstream_src: str, *, train: bool = False):
    if not Path(model_path).is_dir() or not (Path(model_path) / "config.json").is_file():
        raise FileNotFoundError("Use a local, pinned FundusExpert checkpoint directory")
    if not torch.cuda.is_available():
        raise RuntimeError("The 8B FundusExpert experiment requires a CUDA GPU")
    sys.path.insert(0, str(Path(upstream_src) / "internvl_chat"))
    from transformers import AutoModel, AutoTokenizer

    model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                      trust_remote_code=True, low_cpu_mem_usage=True).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    if train:
        for part in (model.vision_model, model.mlp1, model.language_model):
            part.requires_grad_(False)
        from peft import LoraConfig, get_peft_model

        config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                            target_modules=["attention.wqkv", "attention.wo",
                                            "feed_forward.w1", "feed_forward.w2", "feed_forward.w3"],
                            task_type="CAUSAL_LM")
        model.language_model = get_peft_model(model.language_model, config)
        model.language_model.enable_input_require_grads()
        model.train()
    else:
        model.eval()
    return model, tokenizer
