#!/usr/bin/env python3
import os, argparse, random, shutil
from pathlib import Path

def is_img(p):
    return p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", required=True, help="ImageNet100 根目录（含100个子类文件夹）")
    ap.add_argument("--dst_dir", required=True, help="输出的平铺目录")
    ap.add_argument("--num", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    src = Path(args.src_root)
    dst = Path(args.dst_dir); dst.mkdir(parents=True, exist_ok=True)

    all_imgs = [p for p in src.rglob("*") if p.is_file() and is_img(p)]
    if len(all_imgs) < args.num:
        raise RuntimeError(f"可用图片不足：{len(all_imgs)} < {args.num}")

    random.seed(args.seed)
    chosen = random.sample(all_imgs, args.num)
    for i, p in enumerate(chosen):
        # 扁平化命名：保持原文件名，若重名则加序号前缀
        out = dst / p.name
        if out.exists():
            out = dst / f"{i:05d}_{p.name}"
        shutil.copy2(p, out)

    print(f"[DONE] 拷贝 {args.num} 张到 {dst}")

if __name__ == "__main__":
    main()
