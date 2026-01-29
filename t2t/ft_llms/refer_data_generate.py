# ft_llms/refer_data_generate.py
import os
import glob
import random
import argparse
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from accelerate import Accelerator
from datasets import Dataset, load_from_disk, concatenate_datasets

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
)

from peft import PeftModel

import sys
here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '..'))
from data.prepare import dataset_prepare  # 你已修改以支持 --dataset_local_dir
from attack.utils import Dict

# --------------------------
# CLI
# --------------------------
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_name", type=str, default="gpt2",
                    help="基座模型（本地可用路径或 repo id，本脚本离线加载）")
parser.add_argument("-tm", "--target_model", type=str, default="",
                    help="目标模型目录：可能是 LoRA adapter 输出目录，或完整模型目录")
parser.add_argument("-d", "--dataset_name", type=str, default="wikitext-2-raw-v1")
parser.add_argument("-dc", "--dataset_config_name", type=str, default=None,
                    help="datasets 的 config 名")
parser.add_argument("--cache_path", type=str, default="./cache",
                    help="HF/脚本缓存目录")
parser.add_argument("--dataset_local_dir", type=str, default=None,
                    help="本地已缓存/落盘的数据集目录（如 parquet/json 等），优先从这里读取")
parser.add_argument("--use_dataset_cache", action="store_true", default=True)
parser.add_argument("--packing", action="store_true", default=True)
parser.add_argument("--block_size", type=int, default=128)
parser.add_argument("--preprocessing_num_workers", type=int, default=1)
parser.add_argument("--validation_split_percentage", type=float, default=0.1,
                    help="无 validation 切分时，训练集按该比例切 Validation")
cfg = parser.parse_args()

accelerator = Accelerator()
print(accelerator.device)

# --------------------------
# Utils
# --------------------------
def has_lora_adapter(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    if os.path.isfile(os.path.join(path, "adapter_config.json")):
        return True
    patt = glob.glob(os.path.join(path, "adapter_model.*"))
    return len(patt) > 0

# --------------------------
# Tokenizer / Config (离线)
# --------------------------
config = AutoConfig.from_pretrained(
    cfg.model_name,
    local_files_only=True,
    cache_dir=cfg.cache_path,
)
config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(
    cfg.model_name,
    local_files_only=True,
    cache_dir=cfg.cache_path,
    use_fast=True,
)
if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    print("Pad token id is None, setting to eos token id...")
    tokenizer.pad_token = tokenizer.eos_token
# 让自回归模型更稳：左填充
if not hasattr(tokenizer, "padding_side") or tokenizer.padding_side != "left":
    tokenizer.padding_side = "left"

# --------------------------
# Model (基座 + LoRA / 或完整权重)
# --------------------------
torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

if cfg.target_model and has_lora_adapter(cfg.target_model):
    # 1) 先加载基座
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        local_files_only=True,
        cache_dir=cfg.cache_path,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    # 2) 再加载 LoRA 适配器
    model = PeftModel.from_pretrained(
        base_model,
        cfg.target_model,
        local_files_only=True,
        is_trainable=False,
    )
    # 3) 合并并卸载（可选，推理更稳）
    try:
        model = model.merge_and_unload()
    except Exception:
        pass
else:
    # target 目录若是完整权重就直接加载；否则退回基座
    model_path = cfg.target_model if cfg.target_model else cfg.model_name
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        cache_dir=cfg.cache_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        config=config,
    )

model.eval()

# --------------------------
# Dataset
# --------------------------
# 依赖你已修改的 dataset_prepare：支持从 --dataset_local_dir 读取本地落盘数据
train_dataset, valid_dataset = dataset_prepare(cfg, tokenizer=tokenizer)

# 取一个子集作为提示集（示例：1w~2w）
start, end = 10000, 20000
end = min(end, len(train_dataset))
start = min(start, end)
prompt_dataset = train_dataset.select(range(start, end))

prompt_dataloader = DataLoader(prompt_dataset, batch_size=1, shuffle=False)

# --------------------------
# Accelerator prepare
# --------------------------
model, prompt_dataloader = accelerator.prepare(model, prompt_dataloader)

# --------------------------
# Generate
# --------------------------
generated_dataset = {"text": []}

with torch.inference_mode():
    for batch in tqdm(prompt_dataloader, total=len(prompt_dataloader)):
        prompt_texts = batch["text"] if isinstance(batch["text"], list) else [batch["text"]]
        # 只取前 16 token 作为引子
        enc = tokenizer(prompt_texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
        input_ids = enc.input_ids.to(accelerator.device)
        attn_mask = enc.attention_mask.to(accelerator.device)

        # 截断到前 16 个 token
        clipped_ids = input_ids[:, :16]
        clipped_mask = attn_mask[:, :16]

        gen_kwargs = dict(
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            max_new_tokens=64,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        if hasattr(model, "module"):
            gen_tokens = model.module.generate(
                input_ids=clipped_ids,
                attention_mask=clipped_mask,
                **gen_kwargs,
            )
        else:
            gen_tokens = model.generate(
                input_ids=clipped_ids,
                attention_mask=clipped_mask,
                **gen_kwargs,
            )

        # 计算一下自回归 loss（可选，用于 sanity check）
        try:
            _ = model(gen_tokens, labels=gen_tokens).loss
        except Exception:
            pass

        gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        generated_dataset["text"].extend(gen_text)

# --------------------------
# Save to disk
# --------------------------
generated_dataset = Dataset.from_dict(generated_dataset)

# 处理一下某些硬编码路径的兼容
if cfg.model_name == "/mnt/data0/fuwenjie/MIA-LLMs/cache/models--decapoda-research--llama-7b-hf/snapshots/5f98eefcc80e437ef68d457ad7bf167c2c6a1348":
    cfg.model_name = "decapoda-research/llama-7b-hf"

save_dir = f"{cfg.cache_path}/{cfg.dataset_name}/{cfg.dataset_config_name}/refer@{cfg.model_name}/"
os.makedirs(save_dir, exist_ok=True)

# 每个进程/设备单独落盘
generated_dataset.save_to_disk(save_dir + f"{accelerator.device}")

accelerator.wait_for_everyone()

# 主进程汇总
if accelerator.is_main_process:
    concatenated_dataset = None
    for sub in sorted(os.listdir(save_dir)):
        data_path = os.path.join(save_dir, sub)
        if os.path.isdir(data_path):
            ds = load_from_disk(data_path)
            concatenated_dataset = ds if concatenated_dataset is None else concatenate_datasets([concatenated_dataset, ds])
    if concatenated_dataset is not None:
        concatenated_dataset.save_to_disk(save_dir)
        print(f"[OK] Refer dataset saved to: {save_dir}")
    else:
        print(f"[WARN] No sub-datasets found under: {save_dir}")
