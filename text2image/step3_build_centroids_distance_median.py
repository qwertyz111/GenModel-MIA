# -*- coding: utf-8 -*-
"""
Step3 (extended): Build centroids from synth_dir and noisy_dir using BLIP vision CLS embeddings,
with configurable distance for polarity sanity check.

UPDATE (Geometric median centroid):
- Replace mean-centroid with geometric median centroid:
    c = argmin_z sum_i ||e_i - z||_2   (Weiszfeld algorithm)
- After solving z, we L2-normalize it to keep compatibility with downstream cosine-like usage.

Centroids:
- Encode images in synth_dir and noisy_dir -> normalized embeddings
- c_synth = normalize( geometric_median(synth_emb) )
- c_noisy = normalize( geometric_median(noisy_emb) )

Polarity sanity check:
- Probe with k synth embeddings.
- Compute distances d(probe, c_synth) and d(probe, c_noisy) using chosen distance.
- Expect mean_d_to_c_synth < mean_d_to_c_noisy.
- If not, swap centroids and record swapped=True.

Distances supported for sanity check / meta:
- cosine : d(x,c) = 1 - cos(x,c)
- l2     : d(x,c) = ||x - c||_2
- swd    : Sliced Wasserstein-1 (point-to-centroid):
           d(x,c) = mean_k |u_k^T x - u_k^T c| with random unit projections u_k

Offline:
- local BLIP via --blip_dir and TRANSFORMERS_OFFLINE/HF_HUB_OFFLINE.
"""

import os, argparse
from typing import List
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from transformers import AutoProcessor, BlipForConditionalGeneration
from tqdm.auto import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# -------------------------
# Encoder (BLIP vision CLS)
# -------------------------
class BLIPEncoder(torch.nn.Module):
    def __init__(self, blip_dir: str, device: torch.device):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(blip_dir, local_files_only=True)
        self.model = BlipForConditionalGeneration.from_pretrained(blip_dir, local_files_only=True).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.device = device

    @torch.no_grad()
    def encode(self, pil_imgs: List[Image.Image]) -> torch.Tensor:
        enc = self.processor(images=pil_imgs, return_tensors="pt", padding="max_length").to(self.device)
        out = self.model.vision_model(pixel_values=enc["pixel_values"])
        feat = out.last_hidden_state[:, 0]  # CLS-like
        return torch.nn.functional.normalize(feat, p=2, dim=-1)


def list_images(d: str):
    out = []
    for r, _, fs in os.walk(d):
        for f in fs:
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                out.append(os.path.join(r, f))
    return out


# -------------------------
# Geometric median (Weiszfeld)
# -------------------------
def geometric_median_weiszfeld(
    X: torch.Tensor,
    max_iter: int = 256,
    tol: float = 1e-6,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Solve: argmin_z sum_i ||x_i - z||_2  via Weiszfeld algorithm.

    X: [N, d], float tensor on CPU or GPU.
    Return: z [1, d]
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D [N,d], got {X.shape}")
    N, d = X.shape
    if N < 1:
        raise ValueError("X must have at least 1 point")

    # Initialize with mean (only as a starting point; objective is L1 in Euclidean)
    z = X.mean(dim=0, keepdim=True)

    for _ in range(max_iter):
        # distances: [N]
        diff = X - z  # [N,d]
        dist = torch.norm(diff, dim=1)  # [N]

        # If z coincides with some point, that point is a valid minimizer (subgradient contains 0)
        zero_mask = dist <= eps
        if torch.any(zero_mask):
            idx = int(torch.nonzero(zero_mask, as_tuple=False)[0].item())
            return X[idx:idx+1].clone()

        w = 1.0 / torch.clamp(dist, min=eps)  # [N]
        w_sum = w.sum()
        z_new = (X * w.unsqueeze(1)).sum(dim=0, keepdim=True) / w_sum

        step = torch.norm(z_new - z).item()
        z = z_new
        if step < tol:
            break

    return z


# -------------------------
# Distances (point-to-centroid)
# -------------------------
def dist_cosine(X: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    Xn = torch.nn.functional.normalize(X, p=2, dim=-1)
    cn = torch.nn.functional.normalize(c, p=2, dim=-1)
    sim = (Xn * cn).sum(dim=1)
    return 1.0 - sim


def dist_l2(X: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return torch.norm(X - c, dim=1)


def dist_swd_point_centroid(
    X: torch.Tensor,
    c: torch.Tensor,
    n_proj: int = 128,
    seed: int = 0,
) -> torch.Tensor:
    n, d = X.shape
    device = X.device
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))

    U = torch.randn((n_proj, d), generator=g, device=device)
    U = torch.nn.functional.normalize(U, p=2, dim=1)  # [m,d]

    px = X @ U.T      # [n,m]
    pc = c @ U.T      # [1,m]
    return (px - pc).abs().mean(dim=1)  # [n]


def parse_args():
    ap = argparse.ArgumentParser("Step3: Build Centroids (geometric-median, with distance-aware polarity check)")
    ap.add_argument("--synth_dir", type=str, required=True)
    ap.add_argument("--noisy_dir", type=str, required=True)
    ap.add_argument("--n_each", type=int, default=1000)
    ap.add_argument("--blip_dir", type=str, required=True)
    ap.add_argument("--centroid_out", type=str, required=True)
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=2025)

    # probe for polarity check
    ap.add_argument("--probe_k", type=int, default=128,
                    help="how many synth samples used for polarity check; 0 means use all (n_each)")

    # distance for polarity check + meta
    ap.add_argument("--distance", type=str, default="cosine",
                    choices=["cosine", "l2", "swd"])

    # only used when distance=swd
    ap.add_argument("--swd_proj", type=int, default=128,
                    help="number of random projections for SWD (only when --distance swd)")

    # geometric median solver params (keep defaults; user can override when needed)
    ap.add_argument("--gm_max_iter", type=int, default=256)
    ap.add_argument("--gm_tol", type=float, default=1e-6)

    # keep compatibility with your meta field
    ap.add_argument("--score_formula", type=str, default="d_noisy_minus_d_synth",
                    choices=["d_noisy_minus_d_synth", "d_synth_minus_d_noisy"])
    return ap.parse_args()


@torch.no_grad()
def main():
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    args = parse_args()
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    enc = BLIPEncoder(args.blip_dir, device=device)
    rng = np.random.default_rng(args.seed)

    def encode_dir(root: str, n: int):
        paths = list_images(root)
        if len(paths) < n:
            raise RuntimeError(f"Not enough images in {root}: {len(paths)} < {n}")
        sel = [paths[i] for i in rng.choice(len(paths), size=n, replace=False)]
        feats = []
        for s in tqdm(range(0, len(sel), args.batch_size), desc=f"Encode {os.path.basename(root)}"):
            batch = [Image.open(p).convert("RGB") for p in sel[s:s+args.batch_size]]
            feats.append(enc.encode(batch).cpu())
        return torch.cat(feats, dim=0)  # [N,d] on CPU

    # encode both dirs (normalized embeddings)
    synth_emb = encode_dir(args.synth_dir, args.n_each)  # [N,d] CPU
    noisy_emb = encode_dir(args.noisy_dir, args.n_each)

    # build centroids with geometric median, then normalize to unit length
    z_synth = geometric_median_weiszfeld(
        synth_emb, max_iter=int(args.gm_max_iter), tol=float(args.gm_tol)
    )
    z_noisy = geometric_median_weiszfeld(
        noisy_emb, max_iter=int(args.gm_max_iter), tol=float(args.gm_tol)
    )
    c_synth = torch.nn.functional.normalize(z_synth, p=2, dim=-1)
    c_noisy = torch.nn.functional.normalize(z_noisy, p=2, dim=-1)

    # choose distance for polarity check
    if args.distance == "cosine":
        dist_fn = lambda X, c: dist_cosine(X, c)
    elif args.distance == "l2":
        dist_fn = lambda X, c: dist_l2(X, c)
    elif args.distance == "swd":
        dist_fn = lambda X, c: dist_swd_point_centroid(X, c, n_proj=args.swd_proj, seed=args.seed)
    else:
        raise ValueError(f"Unknown distance: {args.distance}")

    # polarity check using synth probes
    k = min(args.probe_k if args.probe_k > 0 else args.n_each, args.n_each)
    probe_idx = rng.choice(synth_emb.size(0), size=k, replace=False)
    probe = synth_emb[probe_idx]  # [k,d] CPU

    d_to_synth = dist_fn(probe, c_synth)
    d_to_noisy = dist_fn(probe, c_noisy)

    mean_synth = float(d_to_synth.mean().item())
    mean_noisy = float(d_to_noisy.mean().item())

    swapped = False
    if not (mean_synth < mean_noisy):
        c_synth, c_noisy = c_noisy, c_synth
        swapped = True
        print("[WARN] Polarity check failed: synth probes closer to noisy centroid -> swapped centroids.")

        d_to_synth2 = dist_fn(probe, c_synth)
        d_to_noisy2 = dist_fn(probe, c_noisy)
        mean_synth = float(d_to_synth2.mean().item())
        mean_noisy = float(d_to_noisy2.mean().item())

    meta = dict(
        roles=dict(
            synth_dir=os.path.abspath(args.synth_dir),
            noisy_dir=os.path.abspath(args.noisy_dir),
            member_ref="c_synth",
            nonmember_ref="c_noisy",
        ),
        centroid_estimator="geometric_median",
        gm_params=dict(max_iter=int(args.gm_max_iter), tol=float(args.gm_tol)),
        distance=args.distance,
        swd_proj=int(args.swd_proj) if args.distance == "swd" else None,
        score_formula=args.score_formula,
        swapped=swapped,
        probe_stats=dict(
            mean_d_probe_to_c_synth=mean_synth,
            mean_d_probe_to_c_noisy=mean_noisy,
            probe_k=int(k),
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
