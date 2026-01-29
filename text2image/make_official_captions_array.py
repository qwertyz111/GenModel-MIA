# -*- coding: utf-8 -*-
"""
make_official_captions_array.py

将官方标注转换为 “字符串数组 JSON”，顺序与给定 images_dir 中的图像文件一一对应：
- 模式 coco: 使用 COCO captions_{train/val}2017.json
- 模式 celeba: 使用 CelebA-Dialog 的 captions_hq.json（或 request_annotated_hq.json / request_hq.json 兜底）

输出 JSON 形如：
[
  "caption for img_0001.jpg",
  "caption for img_0002.jpg",
  ...
]

之后可直接用于：
  step2_sd_synthesize_from_captions.py --captions_json <本脚本输出>
"""

import os
import json
import glob
import argparse
import random
from collections import defaultdict

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def list_images_sorted(root):
    paths = []
    for r, _, fs in os.walk(root):
        for f in fs:
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                paths.append(os.path.join(r, f))
    return sorted(paths)

def build_coco_name2caps(coco_json_path):
    """从 COCO captions_*.json 构建 file_name -> [captions] 映射"""
    with open(coco_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    id2name = {img["id"]: img["file_name"] for img in data["images"]}
    name2caps = defaultdict(list)
    for a in data["annotations"]:
        fname = id2name.get(a["image_id"])
        if fname:
            cap = (a.get("caption") or "").strip()
            if cap:
                name2caps[fname].append(cap)
    return name2caps

def build_celeba_name2caps(captions_dir_or_file):
    """
    CelebA-Dialog:
    - 推荐传入 captions_hq.json 的路径；
    - 若传入目录，则优先找 captions_hq.json，找不到就尝试 request_annotated_hq.json、request_hq.json。
    文件结构（captions_hq.json）为 dict: { "<name>.jpg": { "overall_caption": "...", "attribute_wise_captions": {...} }, ... }
    """
    if os.path.isdir(captions_dir_or_file):
        candidates = [
            os.path.join(captions_dir_or_file, "captions_hq.json"),
            os.path.join(captions_dir_or_file, "request_annotated_hq.json"),
            os.path.join(captions_dir_or_file, "request_hq.json"),
        ]
        cap_path = next((p for p in candidates if os.path.isfile(p)), None)
    else:
        cap_path = captions_dir_or_file if os.path.isfile(captions_dir_or_file) else None

    if not cap_path:
        raise FileNotFoundError(f"未找到 CelebA-Dialog captions 文件：{captions_dir_or_file}")

    with open(cap_path, "r", encoding="utf-8") as f:
        j = json.load(f)

    name2caps = {}
    miss_cnt_attr = 0
    for fname, obj in j.items():
        cap = ""
        # 优先整体描述
        if isinstance(obj, dict):
            if isinstance(obj.get("overall_caption"), str) and obj["overall_caption"].strip():
                cap = obj["overall_caption"].strip()
            else:
                # 退化：拼 attribute_wise_captions
                aw = obj.get("attribute_wise_captions", {})
                if isinstance(aw, dict) and aw:
                    parts = []
                    for k, v in aw.items():
                        if isinstance(v, str) and v.strip():
                            parts.append(v.strip())
                    cap = " ".join(parts).strip()
                else:
                    miss_cnt_attr += 1
        elif isinstance(obj, str):
            cap = obj.strip()

        name2caps[fname] = [cap] if cap else []

    if miss_cnt_attr > 0:
        print(f"[WARN] CelebA: {miss_cnt_attr} entries lack overall/attribute captions.")

    return name2caps

def main():
    ap = argparse.ArgumentParser("Make official captions array JSON aligned to images_dir")
    ap.add_argument("--mode", choices=["coco", "celeba"], required=True,
                    help="选择数据集模式：coco 或 celeba")
    ap.add_argument("--images_dir", required=True, type=str,
                    help="图像目录（将按此目录内排序后的文件名去匹配 caption）")
    # COCO
    ap.add_argument("--coco_ann", type=str, default=None,
                    help="当 --mode=coco 时：captions_train2017.json 或 captions_val2017.json 路径")
    # CelebA
    ap.add_argument("--celeba_caps", type=str, default=None,
                    help="当 --mode=celeba 时：captions_hq.json 路径，或包含其的目录")
    # 通用
    ap.add_argument("--out_json", required=True, type=str,
                    help="输出：字符串数组 JSON")
    ap.add_argument("--n_each", type=int, default=None,
                    help="可选：若指定，则从 images_dir 中按排序后顺序取前 n_each 张（或随机抽样）")
    ap.add_argument("--seed", type=int, default=1337,
                    help="当启用随机抽样时的随机种子")
    ap.add_argument("--pick", choices=["first", "random"], default="first",
                    help="多条 caption 选择策略：first 或 random")
    ap.add_argument("--shuffle", action="store_true",
                    help="对 images 列表进行随机打乱（在抽样之前）。")
    ap.add_argument("--fail_on_missing", action="store_true",
                    help="若有图片找不到 caption，则直接报错；默认用空字符串占位并给出警告。")
    args = ap.parse_args()

    images = list_images_sorted(args.images_dir)
    if not images:
        raise RuntimeError(f"images_dir 中未找到图像: {args.images_dir}")

    # 可选打乱 + 抽样
    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(images)
    if args.n_each is not None:
        if len(images) < args.n_each:
            raise RuntimeError(f"images 数量不足: {len(images)} < n_each={args.n_each}")
        images = images[:args.n_each]

    # 构建 name->captions
    if args.mode == "coco":
        if not args.coco_ann or not os.path.isfile(args.coco_ann):
            raise FileNotFoundError("--coco_ann 未提供或文件不存在")
        name2caps = build_coco_name2caps(args.coco_ann)
        print(f"[INFO] COCO: loaded mapping for {len(name2caps)} file_names from {args.coco_ann}")
    else:
        if not args.celeba_caps or not (os.path.isdir(args.celeba_caps) or os.path.isfile(args.celeba_caps)):
            raise FileNotFoundError("--celeba_caps 未提供或路径非法")
        name2caps = build_celeba_name2caps(args.celeba_caps)
        print(f"[INFO] CelebA: loaded mapping for {len(name2caps)} file_names from {args.celeba_caps}")

    # 逐图像匹配
    miss = []
    out_caps = []
    rng = random.Random(args.seed)
    for p in images:
        bn = os.path.basename(p)
        caps = name2caps.get(bn, [])
        cap = ""
        if caps:
            cap = caps[0] if args.pick == "first" else rng.choice(caps)
        else:
            miss.append(bn)
            if args.fail_on_missing:
                raise KeyError(f"缺少 caption: {bn}")
            # 用空串占位，保持一一对应
            cap = ""
        out_caps.append(cap)

    print(f"[INFO] total images used: {len(images)}, matched: {len(images)-len(miss)}, missing: {len(miss)}")
    if miss:
        print("[WARN] missing examples (show up to 20):", miss[:20])

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out_caps, f, ensure_ascii=False, indent=2)
    print(f"[DONE] write -> {args.out_json}")

if __name__ == "__main__":
    main()
