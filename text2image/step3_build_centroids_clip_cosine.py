# /root/Reconstruction-based-Attack-main/step3_build_centroids_clip_cosine.py
# -*- coding: utf-8 -*-
"""
Step3 (CLIP-cosine): Build centroids from synth_dir and noisy_dir using CLIP image embeddings.

Centroids:
- Encode images in synth_dir and noisy_dir -> L2-normalized embeddings
- c_synth = normalize(mean(synth_emb))
- c_noisy = normalize(mean(noisy_emb))

Polarity sanity check (cosine distance):
- d(x,c) = 1 - cos(x,c)
- Probe with k synth embeddings, expect mean_d_to_c_synth < mean_d_to_c_noisy
- If not, swap centroids and record swapped=True

Offline:
- local CLIP via --clip_dir and TRANSFORMERS_OFFLINE/HF_HUB_OFFLINE
"""

import os
import argparse
from typing import List
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from transformers import CLIPProcessor, CLIPModel
from tqdm.auto import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(root: str) -> List[str]:
    out = []
    for r, _, fs in os.walk(root):
        for f in fs:
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                out.append(os.path.join(r, f))
    return out


class CLIPImageEncoder(torch.nn.Module):
    def __init__(self, clip_dir: str, device: torch.device):
        super().__init__()
        self.processor = CLIPProcessor.from_pretrained(clip_dir, local_files_only=True)
        self.model = CLIPModel.from_pretrained(clip_dir, local_files_only=True).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.device = device

    @torch.no_grad()
    def encode(self, pil_imgs: List[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=pil_imgs, return_tensors="pt").to(self.device)
        feat = self.model.get_image_features(**inputs)  # [B, D]
        feat = torch.nn.functional.normalize(feat, p=2, dim=-1)
        return feat


def dist_cosine(X: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    Xn = torch.nn.functional.normalize(X, p=2, dim=-1)
    cn = torch.nn.functional.normalize(c, p=2, dim=-1)
    sim = (Xn * cn).sum(dim=1)
    return 1.0 - sim


def parse_args():
    ap = argparse.ArgumentParser("Step3: Build Centroids (CLIP image encoder + cosine only)")
    ap.add_argument("--synth_dir", type=str, required=True)
    ap.add_argument("--noisy_dir", type=str, required=True)
    ap.add_argument("--n_each", type=int, default=2000)
    ap.add_argument("--clip_dir", type=str, required=True)
    ap.add_argument("--centroid_out", type=str, required=True)
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--probe_k", type=int, default=128,
                    help="how many synth samples used for polarity check; 0 means use all n_each")
    return ap.parse_args()


@torch.no_grad()
def main():
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    args = parse_args()
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    enc = CLIPImageEncoder(args.clip_dir, device=device)
    rng = np.random.default_rng(args.seed)

    def encode_dir(root: str, n: int) -> torch.Tensor:
        paths = list_images(root)
        if len(paths) < n:
            raise RuntimeError(f"Not enough images in {root}: {len(paths)} < {n}")
        sel_idx = rng.choice(len(paths), size=n, replace=False)
        sel = [paths[i] for i in sel_idx]
        feats = []
        for s in tqdm(range(0, len(sel), args.batch_size), desc=f"Encode {os.path.basename(root)}"):
            batch = [Image.open(p).convert("RGB") for p in sel[s:s + args.batch_size]]
            feats.append(enc.encode(batch).cpu())
        return torch.cat(feats, dim=0)  # [N,D] CPU

    synth_emb = encode_dir(args.synth_dir, args.n_each)
    noisy_emb = encode_dir(args.noisy_dir, args.n_each)

    c_synth = torch.nn.functional.normalize(synth_emb.mean(dim=0, keepdim=True), p=2, dim=-1)
    c_noisy = torch.nn.functional.normalize(noisy_emb.mean(dim=0, keepdim=True), p=2, dim=-1)

    # polarity check
    k = min(args.probe_k if args.probe_k > 0 else args.n_each, args.n_each)
    probe_idx = rng.choice(synth_emb.size(0), size=k, replace=False)
    probe = synth_emb[probe_idx]

    d_to_synth = dist_cosine(probe, c_synth)
    d_to_noisy = dist_cosine(probe, c_noisy)
    mean_synth = float(d_to_synth.mean().item())
    mean_noisy = float(d_to_noisy.mean().item())

    swapped = False
    if not (mean_synth < mean_noisy):
        c_synth, c_noisy = c_noisy, c_synth
        swapped = True
        print("[WARN] Polarity check failed (probe closer to noisy) -> swapped centroids.")
        d_to_synth2 = dist_cosine(probe, c_synth)
        d_to_noisy2 = dist_cosine(probe, c_noisy)
        mean_synth = float(d_to_synth2.mean().item())
        mean_noisy = float(d_to_noisy2.mean().item())

    meta = dict(
        encoder=dict(
            type="clip",
            repo_or_dir=os.path.abspath(args.clip_dir),
            pooling="image_features",
            normalize=True,
        ),
        roles=dict(
            synth_dir=os.path.abspath(args.synth_dir),
            noisy_dir=os.path.abspath(args.noisy_dir),
            member_ref="c_synth",
            nonmember_ref="c_noisy",
        ),
        distance="cosine",
        swapped=swapped,
        probe_stats=dict(
            probe_k=int(k),
            mean_d_probe_to_c_synth=mean_synth,
            mean_d_probe_to_c_noisy=mean_noisy,
        ),
        seed=int(args.seed),
        n_each=int(args.n_each),
        batch_size=int(args.batch_size),
    )

    os.makedirs(os.path.dirname(args.centroid_out), exist_ok=True)
    torch.save({"c_synth": c_synth.cpu(), "c_noisy": c_noisy.cpu(), "meta": meta}, args.centroid_out)
    print(f"[SAVE] -> {args.centroid_out}")
    print(f"[META] {meta}")


if __name__ == "__main__":
    main()
