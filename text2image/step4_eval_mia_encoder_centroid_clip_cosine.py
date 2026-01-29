# /root/Reconstruction-based-Attack-main/step4_eval_mia_encoder_centroid_clip_cosine.py
# -*- coding: utf-8 -*-
"""
Step4 (CLIP-cosine): Eval MIA by CLIP image encoder + two centroids (cosine only).

Score:
    score(x) = d(x, c_noisy) - d(x, c_synth)
where d(x,c) = 1 - cos(x,c)

Inputs:
- centroids_pt: contains c_synth, c_noisy (shape [1,D])
- query_member_pt_shards_dir: pt shards (batch_compressed_*.pt)
- query_nonmember_dir: image directory

Outputs:
- scores_csv: per-sample scores
- roc_csv: ROC points
- roc_png: ROC plot
- thr_scan_csv: threshold scan table with ACC + AUC + TPR@1%FPR

Offline:
- local CLIP via --clip_dir and TRANSFORMERS_OFFLINE/HF_HUB_OFFLINE
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
from transformers import CLIPProcessor, CLIPModel
from tqdm.auto import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


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
        feat = self.model.get_image_features(**inputs)
        return torch.nn.functional.normalize(feat, p=2, dim=-1)


def dist_cosine(X: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    # assuming X and c are already L2-normalized; keep normalization for safety
    Xn = torch.nn.functional.normalize(X, p=2, dim=-1)
    cn = torch.nn.functional.normalize(c, p=2, dim=-1)
    sim = (Xn * cn).sum(dim=1)
    return 1.0 - sim


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


def parse_args():
    ap = argparse.ArgumentParser("Step4: Eval MIA (CLIP image encoder + cosine)")
    ap.add_argument("--centroids_pt", type=str, required=True)
    ap.add_argument("--clip_dir", type=str, required=True)
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=64)

    ap.add_argument("--query_member_pt_shards_dir", type=str, required=True)
    ap.add_argument("--query_nonmember_dir", type=str, required=True)
    ap.add_argument("--q_each", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2333)

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
    enc = CLIPImageEncoder(args.clip_dir, device=device)

    C = torch.load(args.centroids_pt, map_location="cpu")
    c_synth = torch.nn.functional.normalize(C["c_synth"], p=2, dim=-1).to(device)  # [1,D]
    c_noisy = torch.nn.functional.normalize(C["c_noisy"], p=2, dim=-1).to(device)  # [1,D]

    # members from pt shards
    members = load_members_from_pt(args.query_member_pt_shards_dir, args.q_each)

    class MemDS(Dataset):
        def __len__(self): return len(members)
        def __getitem__(self, i): return members[i]

    mem_dl = DataLoader(
        MemDS(), batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda batch: (list(zip(*batch))[0], list(zip(*batch))[1])
    )

    # nonmembers from dir
    all_non_paths = list_images(args.query_nonmember_dir)
    if len(all_non_paths) < args.q_each:
        raise RuntimeError(f"Not enough images in {args.query_nonmember_dir}: {len(all_non_paths)} < {args.q_each}")

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
            feats.append(enc.encode([im.convert("RGB") for im in pil_list]).detach().cpu())
            tags.extend(tt)
        return torch.cat(feats, dim=0), tags

    mem_emb, mem_tags = enc_dl(mem_dl)
    non_emb, non_tags = enc_dl(non_dl)

    def score_from_emb(E_cpu: torch.Tensor) -> torch.Tensor:
        E = E_cpu.to(device)
        d_noisy = dist_cosine(E, c_noisy)
        d_synth = dist_cosine(E, c_synth)
        return (d_noisy - d_synth).detach().cpu()

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
        w.writerow(["split", "id_or_path", "score", "pred_member(>=thr)", "distance", "encoder"])
        for tag, s in zip(mem_tags, s_mem.numpy()):
            w.writerow(["member", tag, float(s), int(s >= args.threshold), "cosine", "clip"])
        for tag, s in zip(non_tags, s_non.numpy()):
            w.writerow(["nonmember", tag, float(s), int(s >= args.threshold), "cosine", "clip"])

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
    plt.title("ROC (Encoder-Centroid MIA) - cosine - CLIP")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(args.roc_png, dpi=150)
    plt.close()

    # threshold scan
    if args.scan_num > 1:
        scan_thrs = np.linspace(args.scan_min, args.scan_max, args.scan_num, dtype=np.float32)
    else:
        scan_thrs = np.array([args.threshold], dtype=np.float32)

    thr_scan_csv = args.scores_csv.replace(".csv", "_thr_scan_cosine.csv")
    with open(thr_scan_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["threshold", "ACC", "AUC", "TPR@FPR=1%", "distance", "encoder"])
        for thr in scan_thrs:
            acc_thr = acc_at_thr(y_true, y_score, thr=float(thr))
            w.writerow([float(thr), float(acc_thr), float(auc), float(tpr1), "cosine", "clip"])

    print("\n=== MIA (Encoder-Centroid) ===")
    print("encoder: clip")
    print("distance: cosine")
    print(f"ACC(thr={args.threshold:.2f}): {acc0:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"TPR@FPR=1%: {tpr1:.4f}")
    print(f"Scores -> {args.scores_csv}")
    print(f"ROC CSV -> {args.roc_csv}")
    print(f"ROC PNG -> {args.roc_png}")
    print(f"Threshold-scan CSV -> {thr_scan_csv}")


if __name__ == "__main__":
    main()
