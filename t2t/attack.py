# attack.py  — offline-friendly, CLI-overridable, PEFT-aware
import os
import random
import logging
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm  # noqa: F401  (kept for potential verbose runs)

import yaml
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import Dataset

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
    LlamaTokenizer,
)
from peft import PeftModel  # used if checkpoints are LoRA adapters

from attack.attack_model import AttackModel
from attack.utils import Dict
from data.prepare import dataset_prepare


# ---------------------------
# Utils
# ---------------------------
def seed_everything(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def is_peft_adapter_dir(path: str) -> bool:
    """Heuristically check if a directory is a PEFT/LoRA adapter (not a full model)."""
    if not path or not os.path.isdir(path):
        return False
    # typical adapter files
    adapter_markers = [
        "adapter_config.json",
        "adapter_model.bin",
        "adapter_model.safetensors",
    ]
    if any(os.path.exists(os.path.join(path, f)) for f in adapter_markers):
        return True
    # treat checkpoint-* as adapter-like if it has no full weights
    if "checkpoint" in os.path.basename(path):
        # if it contains full weights, it will be handled as full model, so return True here is okay
        return True
    return False


def load_model_either(path_or_adapter: str, base_model_name: str, cfg: Dict) -> torch.nn.Module:
    """
    If `path_or_adapter` is a PEFT adapter directory -> load base model then attach PEFT adapter.
    Else -> load full model from `path_or_adapter`.
    """
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    local_only = True
    cache_dir = cfg.get("cache_path")

    quant_cfg = None
    if cfg.get("int8", False):
        quant_cfg = BitsAndBytesConfig(load_in_8bit=True)

    # Adapter path (no full weights inside)
    if is_peft_adapter_dir(path_or_adapter) and not os.path.exists(os.path.join(path_or_adapter, "pytorch_model.bin")):
        logging.info(f"Detected LoRA/PEFT adapter at {path_or_adapter} — attaching to base {base_model_name}")
        base_kwargs = dict(local_files_only=local_only, cache_dir=cache_dir, torch_dtype=torch_dtype)
        if quant_cfg is not None:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    base_model_name, quantization_config=quant_cfg, **base_kwargs
                )
            except TypeError:
                # older transformers fallback
                model = AutoModelForCausalLM.from_pretrained(
                    base_model_name, load_in_8bit=True, device_map="auto", **base_kwargs
                )
        else:
            model = AutoModelForCausalLM.from_pretrained(base_model_name, **base_kwargs)

        model = PeftModel.from_pretrained(model, path_or_adapter, local_files_only=local_only, cache_dir=cache_dir)
        return model

    # Full model path
    logging.info(f"Loading full model from {path_or_adapter}")
    load_kwargs = dict(local_files_only=local_only, cache_dir=cache_dir, torch_dtype=torch_dtype)
    if quant_cfg is not None:
        try:
            return AutoModelForCausalLM.from_pretrained(
                path_or_adapter, quantization_config=quant_cfg, **load_kwargs
            )
        except TypeError:
            return AutoModelForCausalLM.from_pretrained(
                path_or_adapter, load_in_8bit=True, device_map="auto", **load_kwargs
            )
    return AutoModelForCausalLM.from_pretrained(path_or_adapter, **load_kwargs)


def build_tokenizer(cfg: Dict) -> AutoTokenizer:
    base = cfg["model_name"]
    cache_dir = cfg.get("cache_path")
    local_only = True
    config_obj = AutoConfig.from_pretrained(base, local_files_only=local_only, cache_dir=cache_dir)
    model_type = config_obj.to_dict().get("model_type", "")

    add_eos = bool(cfg.get("add_eos_token", False))
    add_bos = bool(cfg.get("add_bos_token", False))

    if model_type == "llama":
        tok = LlamaTokenizer.from_pretrained(
            base, add_eos_token=add_eos, add_bos_token=add_bos, use_fast=True,
            local_files_only=local_only, cache_dir=cache_dir
        )
    else:
        tok = AutoTokenizer.from_pretrained(
            base, add_eos_token=add_eos, add_bos_token=add_bos, use_fast=True,
            local_files_only=local_only, cache_dir=cache_dir
        )

    pad_id = cfg.get("pad_token_id", None)
    if pad_id is not None:
        logging.info(f"Using pad token id {pad_id}")
        tok.pad_token_id = int(pad_id)
    if tok.pad_token_id is None:
        logging.info("Pad token id is None, setting to eos token id...")
        tok.pad_token_id = tok.eos_token_id
    return tok


def load_mask_model(cfg: Dict, accelerator: Accelerator):
    """Load the mask-filling model if available locally; otherwise return None gracefully."""
    name = cfg.get("mask_filling_model_name")
    if not name:
        return None, None

    int8_kwargs = {}
    half_kwargs = {}
    if cfg.get("int8", False):
        int8_kwargs = dict(load_in_8bit=True, device_map="auto", torch_dtype=torch.bfloat16)
    elif cfg.get("half", False):
        half_kwargs = dict(torch_dtype=torch.bfloat16)

    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            name, local_files_only=True, cache_dir=cfg.get("cache_path"), **int8_kwargs, **half_kwargs
        ).to(accelerator.device)
        try:
            n_positions = model.config.n_positions
        except AttributeError:
            n_positions = 512
        tokenizer = AutoTokenizer.from_pretrained(
            name, model_max_length=n_positions, local_files_only=True, cache_dir=cfg.get("cache_path")
        )
        return model, tokenizer
    except Exception as e:
        logging.warning(f"[Mask model] Could not load '{name}' locally: {e}. Continuing without it.")
        return None, None


# ---------------------------
# Main
# ---------------------------
def main():
    # Load YAML config
    with open("configs/config.yaml", "r") as f:
        cfg = Dict(yaml.safe_load(f))

    # CLI overrides
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m",  "--model_name",        type=str)
    parser.add_argument("-tm", "--target_model",      type=str)
    parser.add_argument("-rm", "--reference_model",   type=str)
    parser.add_argument("-d",  "--dataset_name",      type=str)
    parser.add_argument("--dataset_local_dir",        type=str)
    parser.add_argument("--cache_path",               type=str)
    parser.add_argument("--eval_sta_idx",             type=int)
    parser.add_argument("--eval_end_idx",             type=int)
    parser.add_argument("--output_dir",               type=str)
    args, _ = parser.parse_known_args()

    def _override(k, v):
        if v is not None:
            cfg[k] = v

    _override("model_name",        args.model_name)
    _override("target_model",      args.target_model)
    _override("reference_model",   args.reference_model)
    _override("dataset_name",      args.dataset_name)
    _override("dataset_local_dir", args.dataset_local_dir)
    _override("cache_path",        args.cache_path)
    _override("eval_sta_idx",      args.eval_sta_idx)
    _override("eval_end_idx",      args.eval_end_idx)
    _override("output_dir",        args.output_dir)

    # Logging / accelerator
    accelerator = Accelerator()
    logger = get_logger(__name__, "INFO")
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Reproducibility
    seed_everything(int(cfg.get("seed", 0)))

    # 1) Models (target + reference)
    logging.info(f"Loading target model from {cfg['target_model']}")
    target_model = load_model_either(cfg["target_model"], cfg["model_name"], cfg)

    logging.info(f"Loading reference model from {cfg['reference_model']}")
    reference_model = load_model_either(cfg["reference_model"], cfg["model_name"], cfg)

    logging.info("Successfully loaded models")


    logging.info(f"Loading target model from {cfg['target_model']}")
    target_model = load_model_either(cfg["target_model"], cfg["model_name"], cfg)

    logging.info(f"Loading reference model from {cfg['reference_model']}")
    reference_model = load_model_either(cfg["reference_model"], cfg["model_name"], cfg)

    # === ensure models are on the same device ===
    target_model.to(accelerator.device)
    reference_model.to(accelerator.device)
    target_model.eval()
    reference_model.eval()
    # ============================================

    logging.info("Successfully loaded models")




    # 2) Tokenizer
    tokenizer = build_tokenizer(cfg)

    # 3) Data (offline aware via data.prepare)
    train_dataset, valid_dataset = dataset_prepare(cfg, tokenizer=tokenizer)
    # slice to HF Datasets, then optionally sample
    tr = Dataset.from_dict(train_dataset[cfg.train_sta_idx:cfg.train_end_idx])
    ev = Dataset.from_dict(valid_dataset[cfg.eval_sta_idx:cfg.eval_end_idx])

    max_s = int(cfg.get("maximum_samples", 0) or 0)
    if max_s > 0:
        if max_s < len(tr["text"]):
            tr = Dataset.from_dict(tr[random.sample(range(len(tr["text"])), max_s)])
        if max_s < len(ev["text"]):
            ev = Dataset.from_dict(ev[random.sample(range(len(ev["text"])), max_s)])

    logging.info("Successfully loaded datasets!")

    # 4) Dataloaders
    eval_bs = int(cfg.get("eval_batch_size", 8))
    train_dataloader = DataLoader(tr, batch_size=eval_bs)
    eval_dataloader = DataLoader(ev, batch_size=eval_bs)
    train_dataloader, eval_dataloader = accelerator.prepare(train_dataloader, eval_dataloader)

    # 5) Optional mask model (local-only)
    mask_model, mask_tokenizer = load_mask_model(cfg, accelerator)

    # 6) Pack datasets handle for AttackModel
    datasets_dict = {"target": {"train": train_dataloader, "valid": eval_dataloader}}

    # 7) Run attack
    attack_model = AttackModel(
        target_model=target_model,
        tokenizer=tokenizer,
        datasets=datasets_dict,
        reference_model=reference_model,
        shadow_model=None,
        cfg=cfg,
        mask_model=mask_model,
        mask_tokenizer=mask_tokenizer,
    )
    attack_model.conduct_attack(cfg=cfg)


if __name__ == "__main__":
    main()
