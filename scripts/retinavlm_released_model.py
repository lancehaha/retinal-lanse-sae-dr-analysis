"""Load the pinned local Specialist checkpoint through the authors' wrapper."""

from __future__ import annotations

import sys
import hashlib
from pathlib import Path


EXPECTED_FILES = (
    "config.json", "model.safetensors.index.json",
    "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
)
LLAMA_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
SPECIALIST_REVISION = "df0de931334869bbd8a59c3a8dab4bce88ef13fe"
EXPECTED_SHA256 = {
    "config.json": "33d120db7cbe3b06e4bf0556555e138b5813bb1e98c9d1640f5cc199cdb58782",
    "model.safetensors.index.json": "65dfbceb982e991eaa66fa381aaeaf042e393dcfa1f16bad14da41ff2c1fe7a2",
    "model-00001-of-00002.safetensors": "68dd42c024bf886e4f99004c4446a9bb1329c476ac7da9e9450a8d11c6827183",
    "model-00002-of-00002.safetensors": "d3945a9df54238d2df1bac905a720f951de99cbb161818b672cb042ae3152f37",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def preflight(model_dir: Path, authors_code: Path) -> None:
    missing = [name for name in EXPECTED_FILES if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete local RetinaVLM-Specialist checkpoint: {missing}")
    mismatched = [name for name, expected in EXPECTED_SHA256.items()
                  if file_sha256(model_dir / name) != expected]
    if mismatched:
        raise ValueError(f"Specialist checkpoint differs from pinned release: {mismatched}")
    if not (authors_code / "models/retinavlm_wrapper.py").is_file():
        raise FileNotFoundError("Authors' SpecialistVLMs code is missing")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: cannot run the authors' 8-bit model path")
    from huggingface_hub import hf_hub_download
    # Check account approval with a small gated file before loading weights.
    hf_hub_download(LLAMA_ID, "tokenizer_config.json")


def load_released_specialist(model_dir: Path, authors_code: Path):
    preflight(model_dir, authors_code)
    from hydra import compose, initialize_config_dir
    sys.path.insert(0, str(authors_code))
    from models.retinavlm_wrapper import RetinaVLM, RetinaVLMConfig

    with initialize_config_dir(config_dir=str((authors_code / "configs").resolve()), version_base=None):
        config = compose(config_name="default", overrides=["paths=template"])
    rv_config = RetinaVLMConfig.from_pretrained(str(model_dir))
    rv_config.update(config)
    rv_config.model.checkpoint_path = None
    rv_config.model.vision_encoder.checkpoint = None  # included in Specialist weights
    model = RetinaVLM.from_pretrained(str(model_dir), config=rv_config).eval()
    return model, rv_config
