# -*- coding: utf-8 -*-
"""
Step2: 读取 captions（支持 JSON 数组或 JSONL）并用本地 SD 逐条生成合成图。
命名规则：
- 若记录包含 path 且以 noisy_ 开头：noisy_xxxx.jpg -> synth_xxxx.jpg
- 若记录包含 path 但不以 noisy_ 开头：synth_{stem}.jpg
- 若记录无 path：按顺序命名 synth_0000.jpg, synth_0001.jpg, ...
"""

import os
import json
import argparse
from typing import List, Tuple, Optional

from tqdm.auto import tqdm
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.pipelines.stable_diffusion import safety_checker

def _sc(self, clip_input, images):
    return images, [False for _ in images]

safety_checker.StableDiffusionSafetyChecker.forward = _sc


def parse_args():
    ap = argparse.ArgumentParser("Step2: SD synth from captions")
    # 兼容你当前的命令：--captions_json（JSON 数组）
    ap.add_argument("--captions_json", type=str, default=None, help="JSON 数组文件（每个元素含 caption，可选 path）")
    # 同时也允许提供 JSONL（逐行 JSON）
    ap.add_argument("--jsonl", type=str, default=None, help="JSONL 文件，每行 {path, caption} 或 {caption}")
    ap.add_argument("--sd_dir", type=str, required=True, help="本地 Stable Diffusion diffusers 目录")
    ap.add_argument("--lora_dir", type=str, default=None, help="可选：LoRA 目录（save_attn_procs 输出）")
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.5)

    # 为与既有命令兼容：支持 --seed（内部映射到 seed_base）
    ap.add_argument("--seed", type=int, default=None, help="如提供，则作为起始随机种子")
    ap.add_argument("--seed_base", type=int, default=1337, help="基础随机种子（若给了 --seed 将被覆盖）")

    ap.add_argument("--gpu_id", type=int, default=0)
    return ap.parse_args()


def _load_json_array(path: str) -> List[Tuple[Optional[str], str]]:
    """
    支持：
      1) [{'path': '...', 'caption':'...'}, ...]
      2) [{'caption':'...'}, ...] 或 [{'text':'...'}, ...]
      3) ['a caption', 'another caption', ...]
    输出: [(path_or_None, caption_str), ...]
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise ValueError(f"{path} 顶层必须是数组。")

    recs: List[Tuple[Optional[str], str]] = []
    for rec in obj:
        if isinstance(rec, dict):
            cap = (rec.get("caption") or rec.get("text") or "").strip()
            p = rec.get("path")
            if cap:
                recs.append((p, cap))
        elif isinstance(rec, str):
            cap = rec.strip()
            if cap:
                recs.append((None, cap))
    return recs


def _load_jsonl(path: str) -> List[Tuple[Optional[str], str]]:
    """
    一行一个 JSON 对象，期望：
      - {'path': '...', 'caption': '...'}
      - {'caption': '...'} / {'text': '...'}
      - 也允许整行是一个字符串（直接视为 caption）
    输出: [(path_or_None, caption_str), ...]
    """
    recs: List[Tuple[Optional[str], str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            j = json.loads(line)
            if isinstance(j, dict):
                cap = (j.get("caption") or j.get("text") or "").strip()
                pth = j.get("path")
                if cap:
                    recs.append((pth, cap))
            elif isinstance(j, str):
                cap = j.strip()
                if cap:
                    recs.append((None, cap))
    return recs


@torch.no_grad()
def main():
    # 离线环境兼容
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    args = parse_args()
    if args.seed is not None:
        args.seed_base = int(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # Pipeline
    pipe = DiffusionPipeline.from_pretrained(
        args.sd_dir, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    if args.lora_dir and os.path.isdir(args.lora_dir):
        pipe.unet.load_attn_procs(args.lora_dir)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    # 读取 captions：优先 JSON（与你当前命令一致），否则 JSONL
    recs: List[Tuple[Optional[str], str]]
    if args.captions_json and os.path.exists(args.captions_json):
        recs = _load_json_array(args.captions_json)
    elif args.jsonl and os.path.exists(args.jsonl):
        recs = _load_jsonl(args.jsonl)
    else:
        raise FileNotFoundError("请提供 --captions_json（JSON数组）或 --jsonl，并确保文件存在。")

    if len(recs) == 0:
        raise RuntimeError("未解析到任何 (path, caption) 记录。")

    # 合成
    for idx, (src_path, prompt) in enumerate(tqdm(recs, desc="Synthesize", dynamic_ncols=True)):
        if src_path:
            base = os.path.basename(src_path)
            if base.startswith("noisy_"):
                out_name = "synth_" + base[len("noisy_"):]
            else:
                stem, ext = os.path.splitext(base)
                out_name = f"synth_{stem}{ext or '.jpg'}"
        else:
            out_name = f"synth_{idx:04d}.jpg"

        out_path = os.path.join(args.out_dir, out_name)
        g = torch.Generator(device=device).manual_seed(args.seed_base + idx)
        img = pipe(
            prompt,
            num_inference_steps=args.steps,
            width=args.width,
            height=args.height,
            guidance_scale=args.guidance,
            generator=g,
        ).images[0]
        img.save(out_path, format="JPEG", quality=95, optimize=True)

    print(f"[DONE] synthesized {len(recs)} images -> {args.out_dir}")


if __name__ == "__main__":
    main()
