# -*- coding: utf-8 -*-
"""
离线：Kandinsky 2.2 从 caption 批量合成图片
- 支持两种输入：
  (A) --captions_json  指向 JSON 数组：[{ "caption": "...", "id": "...", "path": "..."}, ...]
  (B) --jsonl          指向 JSONL：    {"caption": "...", "path": "..."} 每行一个
- 完全离线：from_pretrained(..., local_files_only=True)
- 生成文件名：优先使用输入记录的 path 文件名基干（若可用），否则为 synth_{idx:04d}.jpg
- 复现性：每张图使用 seed_base + idx 作为随机种子
"""

import os
import json
import argparse
from typing import List, Dict, Any
from tqdm.auto import tqdm

import torch
from PIL import Image

from diffusers import (
    KandinskyV22Pipeline,
    KandinskyV22PriorPipeline,
)

def parse_args():
    ap = argparse.ArgumentParser("Kandinsky 2.2 synth (offline)")
    # 输入（任选其一）
    ap.add_argument("--captions_json", type=str, default=None, help="JSON 数组文件：每个元素至少包含 caption")
    ap.add_argument("--jsonl", type=str, default=None, help="JSONL 文件：每行一个 JSON，至少包含 caption")
    # 模型本地目录
    ap.add_argument("--k22_prior_dir", type=str, required=True, help="Kandinsky 2.2 Prior 本地目录（含 model_index.json）")
    ap.add_argument("--k22_decoder_dir", type=str, required=True, help="Kandinsky 2.2 Decoder 本地目录（含 model_index.json）")
    # 输出
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    # 采样/引导
    ap.add_argument("--prior_steps", type=int, default=40)
    ap.add_argument("--prior_guidance", type=float, default=4.0)
    ap.add_argument("--decoder_steps", type=int, default=50)
    ap.add_argument("--decoder_guidance", type=float, default=4.0)
    # 运行控制
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--max_n", type=int, default=None, help="最多生成多少条（可用于子集快速实验）")
    return ap.parse_args()

def _ensure_offline_env():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("DIFFUSERS_OFFLINE", "1")

def _load_records_from_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        arr = json.load(f)
    if not isinstance(arr, list):
        raise ValueError(f"{path} 不是 JSON 数组")
    recs = []
    for it in arr:
        if not isinstance(it, dict):
            continue
        cap = (it.get("caption") or it.get("overall_caption") or it.get("text") or "").strip()
        if cap:
            recs.append({"caption": cap, "path": it.get("path"), "id": it.get("id")})
    return recs

def _load_records_from_jsonl(path: str) -> List[Dict[str, Any]]:
    recs = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                it = json.loads(ln)
            except Exception:
                continue
            cap = (it.get("caption") or it.get("overall_caption") or it.get("text") or "").strip()
            if cap:
                recs.append({"caption": cap, "path": it.get("path"), "id": it.get("id")})
    return recs

@torch.no_grad()
def main():
    _ensure_offline_env()
    args = parse_args()

    if not args.captions_json and not args.jsonl:
        raise FileNotFoundError("请提供 --captions_json（JSON数组）或 --jsonl（JSONL）。")
    if args.captions_json and not os.path.isfile(args.captions_json):
        raise FileNotFoundError(f"captions_json 不存在: {args.captions_json}")
    if args.jsonl and not os.path.isfile(args.jsonl):
        raise FileNotFoundError(f"jsonl 不存在: {args.jsonl}")

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # 读取 caption 记录
    if args.captions_json:
        records = _load_records_from_json(args.captions_json)
    else:
        records = _load_records_from_jsonl(args.jsonl)

    if not records:
        raise RuntimeError("未解析到任何 (caption) 记录。")

    if args.max_n is not None:
        records = records[: int(args.max_n)]

    # 加载 Kandinsky v2.2 Prior & Decoder （完全离线）
    prior = KandinskyV22PriorPipeline.from_pretrained(
        args.k22_prior_dir, torch_dtype=torch.float16, local_files_only=True
    ).to(device)
    decoder = KandinskyV22Pipeline.from_pretrained(
        args.k22_decoder_dir, torch_dtype=torch.float16, local_files_only=True
    ).to(device)

    # 逐条生成
    for idx, rec in enumerate(tqdm(records, desc="Kandinsky2.2 synth (offline)", dynamic_ncols=True)):
        cap = rec["caption"]
        base_name = None
        if rec.get("path"):
            base = os.path.basename(str(rec["path"]))
            stem, ext = os.path.splitext(base)
            base_name = f"synth_{stem}.jpg"
        if base_name is None:
            base_name = f"synth_{idx:04d}.jpg"

        out_path = os.path.join(args.out_dir, base_name)

        gen = torch.Generator(device=device).manual_seed(args.seed + idx)

        # 1) 文本→图像先验（得到 image_embeds & negative_image_embeds）
        prior_out = prior(
            prompt=cap,
            negative_prompt="",
            generator=gen,
            num_inference_steps=args.prior_steps,
            guidance_scale=args.prior_guidance,
        )
        image_embeds = prior_out.image_embeds
        negative_image_embeds = prior_out.negative_image_embeds

        # 2) 解码器生成图像 —— 注意：**不再传 prompt**（与当前 diffusers 版本对齐）
        dec_out = decoder(
            image_embeds=image_embeds,
            negative_image_embeds=negative_image_embeds,
            height=args.height,
            width=args.width,
            num_inference_steps=args.decoder_steps,
            guidance_scale=args.decoder_guidance,
            generator=gen,
        )
        img: Image.Image = dec_out.images[0]
        img.save(out_path, format="JPEG", quality=95, optimize=True)

    print(f"[DONE] synthesized {len(records)} images -> {args.out_dir}")

if __name__ == "__main__":
    main()
