# -*- coding: utf-8 -*-
"""
Step4 (extended): Eval MIA by BLIP encoder + two centroids, with configurable distance.

Unified score semantics:
    score(x) = d(x, c_noisy) - d(x, c_synth)
so that score >= 0 means "closer to synth centroid" (member-like), regardless of distance type.

Supported distances:
- cosine : d(x,c) = 1 - cos(x,c)
- l2     : d(x,c) = ||x - c||_2
- swd    : Sliced Wasserstein-1 (point-to-centroid):
           d(x,c) = mean_k |u_k^T x - u_k^T c| with random unit projections u_k

Distance selection:
- --distance auto (default): read centroids meta['distance'] if available, else fallback to cosine
- --distance cosine|l2|swd: override

Also supports optional query-time Gaussian noise and JPEG re-encoding.

Outputs:
- scores_csv: per-sample scores
- roc_csv: ROC curve points
- roc_png: ROC plot
- thr_scan_csv: ACC + (AUC, TPR@FPR=1%) repeated for each scanned threshold (for convenient table viewing)

Offline:
- local BLIP via --blip_dir with TRANSFORMERS_OFFLINE/HF_HUB_OFFLINE set
"""

import os
import io
import csv
import gc
import argparse
from glob import glob
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, BlipForConditionalGeneration
from tqdm.auto import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


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
        feat = out.last_hidden_state[:, 0]  # CLS
        return torch.nn.functional.normalize(feat, p=2, dim=-1)


def list_images(d: str) -> List[str]:
    out = []
    for r, _, fs in os.walk(d):
        for f in fs:
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                out.append(os.path.join(r, f))
    return out


def to_pil_any(x):
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    if isinstance(x, (bytes, bytearray)):
        from io import BytesIO
        return Image.open(BytesIO(x)).convert("RGB")
    arr = torch.as_tensor(x).cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    return Image.fromarray(arr.astype(np.uint8)).convert("RGB")


def add_gaussian_noise_pil(img: Image.Image, sigma: float) -> Image.Image:
    if sigma <= 0:
        return img
    arr = np.asarray(img.convert("RGB")).astype(np.float32)
    noise = np.random.normal(0.0, sigma * 255.0, size=arr.shape).astype(np.float32)
    out = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def maybe_jpeg(img: Image.Image, q: int) -> Image.Image:
    if q and q > 0:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q, optimize=True)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    return img


def build_roc(y_true: np.ndarray, y_score: np.ndarray):
    order = np.argsort(-y_score)
    y_true = y_true[order]
    y_score = y_score[order]
    P = int((y_true == 1).sum())
    N = int((y_true == 0).sum())
    tps = fps = 0
    fpr = [0.0]
    tpr = [0.0]
    i = 0
    while i < len(y_score):
        thr = y_score[i]
        j = i
        while j < len(y_score) and y_score[j] == thr:
            if y_true[j] == 1:
                tps += 1
            else:
                fps += 1
            j += 1
        fpr.append(fps / max(1, N))
        tpr.append(tps / max(1, P))
        i = j
    fpr.append(1.0)
    tpr.append(1.0)
    return np.array(fpr), np.array(tpr)


def auc_trapz(fpr: np.ndarray, tpr: np.ndarray) -> float:
    o = np.argsort(fpr)
    return float(np.trapz(tpr[o], fpr[o]))


def tpr_at_fpr(fpr: np.ndarray, tpr: np.ndarray, target=0.01) -> float:
    o = np.argsort(fpr)
    fpr = fpr[o]
    tpr = tpr[o]
    if target <= fpr[0]:
        return float(tpr[0])
    if target >= fpr[-1]:
        return float(tpr[-1])
    k = np.searchsorted(fpr, target) - 1
    k = np.clip(k, 0, len(fpr) - 2)
    x0, x1 = fpr[k], fpr[k + 1]
    y0, y1 = tpr[k], tpr[k + 1]
    if x1 == x0:
        return float(y0)
    a = (target - x0) / (x1 - x0)
    return float(y0 + a * (y1 - y0))


def acc_at_thr(y_true: np.ndarray, y_score: np.ndarray, thr=0.0) -> float:
    y_pred = (y_score >= thr).astype(np.int64)
    return float((y_pred == y_true).mean())


def load_members_from_pt(shards_dir: str, need: int) -> List[Tuple[Image.Image, str]]:
    outs = []
    shard_files = sorted(glob(os.path.join(shards_dir, "batch_compressed_*.pt")))
    if not shard_files:
        raise RuntimeError(f"No pt shards in {shards_dir}")
    for sp in shard_files:
        data = torch.load(sp, map_location="cpu")
        imgs = data.get("image") or data.get("images")
        if not imgs:
            del data
            continue
        for i, im in enumerate(imgs):
            try:
                img = to_pil_any(im)
                outs.append((img, f"{os.path.basename(sp)}#{i}"))
            except Exception:
                continue
            if len(outs) >= need:
                return outs
        del data
        gc.collect()
    if len(outs) < need:
        raise RuntimeError(f"Only {len(outs)} < {need}")
    return outs


def dist_cosine(X: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    Xn = torch.nn.functional.normalize(X, p=2, dim=-1)
    cn = torch.nn.functional.normalize(c, p=2, dim=-1)
    sim = (Xn * cn).sum(dim=1)
    return 1.0 - sim


def dist_l2(X: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return torch.norm(X - c, dim=1)


def dist_swd_point_centroid(X: torch.Tensor, c: torch.Tensor, n_proj: int = 128, seed: int = 0) -> torch.Tensor:
    n, d = X.shape
    device = X.device
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    U = torch.randn((n_proj, d), generator=g, device=device)
    U = torch.nn.functional.normalize(U, p=2, dim=1)
    px = X @ U.T
    pc = c @ U.T
    return (px - pc).abs().mean(dim=1)


def resolve_distance_name(args_distance: str, meta: Optional[dict]) -> str:
    if args_distance != "auto":
        return args_distance
    if isinstance(meta, dict):
        d = meta.get("distance", None)
        if isinstance(d, str) and d.lower() in ("cosine", "l2", "swd"):
            return d.lower()
    return "cosine"


def parse_args():
    ap = argparse.ArgumentParser("Step4: Eval MIA by encoder-centroid with configurable distance")
    ap.add_argument("--centroids_pt", type=str, required=True)
    ap.add_argument("--blip_dir", type=str, required=True)
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=32)

    ap.add_argument("--query_member_pt_shards_dir", type=str, required=True)
    ap.add_argument("--query_nonmember_dir", type=str, required=True)
    ap.add_argument("--q_each", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2333)
    ap.add_argument("--exclude_list", type=str, default=None)

    ap.add_argument("--query_noise_sigma", type=float, default=0.0)
    ap.add_argument("--query_jpeg_quality", type=int, default=0)

    ap.add_argument("--distance", type=str, default="auto",
                    choices=["auto", "cosine", "l2", "swd"])
    ap.add_argument("--swd_proj", type=int, default=128)

    ap.add_argument("--scores_csv", type=str, required=True)
    ap.add_argument("--roc_csv", type=str, required=True)
    ap.add_argument("--roc_png", type=str, required=True)
    ap.add_argument("--threshold", type=float, default=0.0)

    ap.add_argument("--scan_min", type=float, default=-1.0)
    ap.add_argument("--scan_max", type=float, default=1.0)
    ap.add_argument("--scan_num", type=int, default=41)
    return ap.parse_args()


@torch.no_grad()
def main():
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    args = parse_args()
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    enc = BLIPEncoder(args.blip_dir, device=device)

    C = torch.load(args.centroids_pt, map_location="cpu")
    meta = C.get("meta", None)
    c_synth = torch.nn.functional.normalize(C["c_synth"], p=2, dim=-1)  # [1,d]
    c_noisy = torch.nn.functional.normalize(C["c_noisy"], p=2, dim=-1)  # [1,d]

    dist_name = resolve_distance_name(args.distance, meta)
    if dist_name == "cosine":
        dist_fn = lambda X, c: dist_cosine(X, c)
    elif dist_name == "l2":
        dist_fn = lambda X, c: dist_l2(X, c)
    elif dist_name == "swd":
        dist_fn = lambda X, c: dist_swd_point_centroid(X, c, n_proj=args.swd_proj, seed=args.seed)
    else:
        raise ValueError(f"Unknown distance: {dist_name}")

    print(f"[INFO] distance = {dist_name} (args.distance={args.distance})")
    if dist_name == "swd":
        print(f"[INFO] swd_proj = {args.swd_proj}")

    members = load_members_from_pt(args.query_member_pt_shards_dir, args.q_each)

    class MemDS(Dataset):
        def __len__(self): return len(members)
        def __getitem__(self, i): return members[i]

    mem_dl = DataLoader(
        MemDS(), batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda batch: (list(zip(*batch))[0], list(zip(*batch))[1])
    )

    all_non_paths = list_images(args.query_nonmember_dir)
    if args.exclude_list and os.path.isfile(args.exclude_list):
        with open(args.exclude_list, "r", encoding="utf-8") as f:
            excl = set([ln.strip() for ln in f if ln.strip()])
        base_excl = set(os.path.basename(x) for x in excl)
        all_non_paths = [p for p in all_non_paths if os.path.basename(p) not in base_excl]

    if len(all_non_paths) < args.q_each:
        raise RuntimeError(
            f"Not enough images in {args.query_nonmember_dir} after exclusion: {len(all_non_paths)} < {args.q_each}"
        )

    rng = np.random.default_rng(args.seed)
    non_paths = [all_non_paths[i] for i in rng.choice(len(all_non_paths), size=args.q_each, replace=False)]

    class NonDS(Dataset):
        def __len__(self): return len(non_paths)
        def __getitem__(self, i):
            p = non_paths[i]
            return Image.open(p).convert("RGB"), p

    non_dl = DataLoader(
        NonDS(), batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda batch: (list(zip(*batch))[0], list(zip(*batch))[1])
    )

    def enc_dl(dl):
        feats = []
        tags = []
        for pil_list, tt in tqdm(dl, desc="Encode", dynamic_ncols=True):
            proc = []
            for im in pil_list:
                im2 = add_gaussian_noise_pil(im, args.query_noise_sigma)
                im2 = maybe_jpeg(im2, args.query_jpeg_quality)
                proc.append(im2)
            feats.append(enc.encode(proc).cpu())
            tags.extend(tt)
        return torch.cat(feats, dim=0), tags

    mem_emb, mem_tags = enc_dl(mem_dl)
    non_emb, non_tags = enc_dl(non_dl)

    c_synth_d = c_synth.to(device)
    c_noisy_d = c_noisy.to(device)

    def score_from_emb(E_cpu: torch.Tensor) -> torch.Tensor:
        E = E_cpu.to(device)
        d_noisy = dist_fn(E, c_noisy_d)
        d_synth = dist_fn(E, c_synth_d)
        s = d_noisy - d_synth
        return s.detach().cpu()

    s_mem = score_from_emb(mem_emb)
    s_non = score_from_emb(non_emb)

    y_true = np.concatenate([np.ones(len(s_mem)), np.zeros(len(s_non))]).astype(np.int64)
    y_score = np.concatenate([s_mem.numpy(), s_non.numpy()]).astype(np.float32)

    acc0 = acc_at_thr(y_true, y_score, thr=args.threshold)
    fpr, tpr = build_roc(y_true, y_score)
    auc = auc_trapz(fpr, tpr)
    tpr1 = tpr_at_fpr(fpr, tpr, target=0.01)

    os.makedirs(os.path.dirname(args.scores_csv), exist_ok=True)
    with open(args.scores_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "id_or_path", "score", "pred_member(>=thr)", "distance"])
        for tag, s in zip(mem_tags, s_mem.numpy()):
            w.writerow(["member", tag, float(s), int(s >= args.threshold), dist_name])
        for tag, s in zip(non_tags, s_non.numpy()):
            w.writerow(["nonmember", tag, float(s), int(s >= args.threshold), dist_name])

    os.makedirs(os.path.dirname(args.roc_csv), exist_ok=True)
    with open(args.roc_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["FPR", "TPR"])
        for x, y in zip(fpr, tpr):
            w.writerow([x, y])

    os.makedirs(os.path.dirname(args.roc_png), exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, lw=2, label=f"AUC={auc:.4f}")
    plt.plot([0, 1], [0, 1], "--", lw=1)
    plt.scatter([0.01], [tpr1], s=30)
    plt.text(0.011, tpr1, f"TPR@1%FPR={tpr1:.3f}", fontsize=9)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.grid(alpha=0.3)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC (Encoder-Centroid MIA) - {dist_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(args.roc_png, dpi=150)
    plt.close()

    print("\n=== MIA (Encoder-Centroid) ===")
    print(f"distance: {dist_name}")
    print(f"ACC(thr={args.threshold:.2f}): {acc0:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"TPR@FPR=1%: {tpr1:.4f}")
    print(f"Scores -> {args.scores_csv}")
    print(f"ROC CSV -> {args.roc_csv}")
    print(f"ROC PNG -> {args.roc_png}")

    if args.scan_num > 1:
        scan_thrs = np.linspace(args.scan_min, args.scan_max, args.scan_num, dtype=np.float32)
    else:
        scan_thrs = np.array([args.threshold], dtype=np.float32)

    P = max(1, int((y_true == 1).sum()))
    N = max(1, int((y_true == 0).sum()))

    scan_rows = []  # store [thr, acc]
    best_acc = -1.0
    best_idx = 0

    # 仍然计算 tp/fp 只是为了找到 best thr（按 ACC 最大）
    best_tp = best_fp = best_tn = best_fn = 0

    for idx, thr in enumerate(scan_thrs):
        y_pred = (y_score >= thr).astype(np.int64)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())

        acc_thr = (tp + tn) / (P + N)
        scan_rows.append([float(thr), float(acc_thr)])

        if acc_thr > best_acc:
            best_acc = acc_thr
            best_idx = idx
            best_tp, best_fp, best_tn, best_fn = tp, fp, tn, fn

    best_thr = float(scan_thrs[best_idx])
    best_thr_acc = float(best_acc)

    # 你要求：删掉 TPR/FPR 列，换成 AUC 与 TPR@FPR=1%
    thr_scan_csv = args.scores_csv.replace(".csv", f"_thr_scan_{dist_name}.csv")
    with open(thr_scan_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["threshold", "ACC", "AUC", "TPR@FPR=1%", "distance"])
        for thr, acc_thr in scan_rows:
            w.writerow([thr, acc_thr, float(auc), float(tpr1), dist_name])

    # 仍然在控制台输出“最佳 ACC”以及全局 AUC、TPR@1%FPR
    print("\n=== Threshold Scan (best by ACC) ===")
    print(f"distance: {dist_name}")
    print(f"Scan range: [{args.scan_min:.3f}, {args.scan_max:.3f}] with {len(scan_thrs)} points")
    print("Best threshold by ACC over scanned thresholds:")
    print(f"  thr* = {best_thr:.4f}")
    print(f"  Best ACC(thr*) = {best_thr_acc:.4f}")
    print(f"  AUC = {auc:.4f}")
    print(f"  TPR@FPR=1% = {tpr1:.4f}")
    print(f"Threshold-scan CSV -> {thr_scan_csv}")

    print("\n=== Best Summary ===")
    print(f"Best ACC (thr*): {best_thr_acc:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"TPR@FPR=1%: {tpr1:.4f}")


if __name__ == "__main__":
    main()
