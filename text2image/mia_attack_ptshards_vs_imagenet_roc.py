# -*- coding: utf-8 -*-
"""
MIA with style classifier (member from COCO pt-shards, nonmember from ImageNet100 dir)
Outputs:
- ACC (at --threshold)
- AUC (ROC area via threshold sweep)
- TPR@FPR=1% (linear interpolation)
- ROC curve PNG and CSV

Usage example at bottom.
"""

import os, argparse, csv, gc
from typing import List, Tuple
import numpy as np
from PIL import Image, ImageFile
from glob import glob
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoProcessor, BlipForConditionalGeneration

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}

# ===== classifier & extractor (match your training) =====
class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.1, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )
    def forward(self, x): return self.net(x)

class BLIPFeatureExtractor(nn.Module):
    def __init__(self, blip_dir: str, device: torch.device):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(blip_dir, local_files_only=True)
        self.blip = BlipForConditionalGeneration.from_pretrained(blip_dir, local_files_only=True)
        for p in self.blip.parameters(): p.requires_grad = False
        self.blip.eval().to(device)
        self.device = device
    @torch.no_grad()
    def forward(self, pil_images: List[Image.Image]) -> torch.Tensor:
        enc = self.processor(images=pil_images, return_tensors="pt", padding="max_length")
        pixel_values = enc["pixel_values"].to(self.device)
        out = self.blip.vision_model(pixel_values=pixel_values)
        return out.last_hidden_state[:, 0]  # [B, D]

# ===== data utils =====
def to_pil(x):
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    if isinstance(x, (bytes, bytearray)):
        from io import BytesIO
        return Image.open(BytesIO(x)).convert("RGB")
    arr = torch.as_tensor(x).cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] in (1,3):
        arr = np.transpose(arr, (1,2,0))
    return Image.fromarray(arr.astype("uint8")).convert("RGB")

def load_member_from_pt_shards(shards_dir: str, need: int) -> List[Tuple[Image.Image, str]]:
    shard_files = sorted(glob(os.path.join(shards_dir, "batch_compressed_*.pt")))
    if not shard_files:
        raise RuntimeError(f"No pt shards found under: {shards_dir}")
    out = []
    for sp in shard_files:
        data = torch.load(sp, map_location="cpu")
        imgs = data.get("image") or data.get("images")
        if not imgs:
            del data; continue
        for ii, im in enumerate(imgs):
            try:
                out.append((to_pil(im), f"pt:{os.path.basename(sp)}#{ii}"))
            except Exception:
                continue
            if len(out) >= need:
                return out
        del data, imgs
        gc.collect()
    if len(out) < need:
        raise RuntimeError(f"Not enough images in shards: {len(out)} < {need}")
    return out

def list_images_under(root: str) -> List[str]:
    paths = []
    for r,_,files in os.walk(root):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                paths.append(os.path.join(r,f))
    return paths

def sample_nonmember_from_dir(root: str, need: int, seed: int = 2024) -> List[str]:
    paths = list_images_under(root)
    if len(paths) < need:
        raise RuntimeError(f"Not enough images under {root}: {len(paths)} < {need}")
    rng = np.random.default_rng(seed); rng.shuffle(paths)
    return paths[:need]

class MixedImageDS(Dataset):
    """前半 member(PIL,tag)，后半 nonmember(path)；返回统一 (PIL, label, tag/path)"""
    def __init__(self, members: List[Tuple[Image.Image,str]], non_paths: List[str]):
        self.members = members
        self.non_paths = non_paths
        self.nm = len(members); self.nn = len(non_paths)
    def __len__(self): return self.nm + self.nn
    def __getitem__(self, idx):
        if idx < self.nm:
            img, tag = self.members[idx]
            return img.convert("RGB"), 1, tag
        j = idx - self.nm
        p = self.non_paths[j]
        img = Image.open(p).convert("RGB")
        return img, 0, p

def collate_fn(batch):
    imgs, labs, tags = zip(*batch)
    return list(imgs), torch.tensor(labs, dtype=torch.long), list(tags)

# ===== metrics =====
def acc_at_threshold(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> float:
    y_pred = (y_score >= thr).astype(np.int64)
    return float((y_pred == y_true).mean())

def build_roc_points(y_true: np.ndarray, y_score: np.ndarray):
    # 排序阈值（从高到低）：把每个唯一分数作为阈值
    order = np.argsort(-y_score)
    y_true = y_true[order]
    y_score = y_score[order]

    P = (y_true == 1).sum()
    N = (y_true == 0).sum()
    if P == 0 or N == 0:
        raise ValueError("Both positive(=member) and negative(=nonmember) samples are required.")

    # 逐点扫阈值（典型实现：按得分降序累进）
    tps, fps = 0, 0
    roc_fpr, roc_tpr = [0.0], [0.0]
    i = 0
    while i < len(y_score):
        thr = y_score[i]
        # 处理所有分数==thr 的样本
        j = i
        while j < len(y_score) and y_score[j] == thr:
            if y_true[j] == 1: tps += 1
            else:              fps += 1
            j += 1
        fpr = fps / N
        tpr = tps / P
        roc_fpr.append(fpr)
        roc_tpr.append(tpr)
        i = j

    # 末尾补(1,1)
    roc_fpr.append(1.0)
    roc_tpr.append(1.0)
    return np.array(roc_fpr), np.array(roc_tpr)

def auc_trapz(fpr: np.ndarray, tpr: np.ndarray) -> float:
    # x 需非降序
    order = np.argsort(fpr)
    return float(np.trapz(tpr[order], fpr[order]))

def tpr_at_fpr(fpr: np.ndarray, tpr: np.ndarray, target_fpr: float = 0.01) -> float:
    # 在线性插值域内找 TPR(FPR=target)
    order = np.argsort(fpr)
    fpr = fpr[order]; tpr = tpr[order]
    if target_fpr <= fpr[0]:
        return float(tpr[0])
    if target_fpr >= fpr[-1]:
        return float(tpr[-1])
    # 找到 fpr[i] <= target < fpr[i+1]
    idx = np.searchsorted(fpr, target_fpr) - 1
    idx = np.clip(idx, 0, len(fpr)-2)
    x0, x1 = fpr[idx], fpr[idx+1]
    y0, y1 = tpr[idx], tpr[idx+1]
    if x1 == x0:
        return float(y0)
    alpha = (target_fpr - x0) / (x1 - x0)
    return float(y0 + alpha * (y1 - y0))

# ===== main =====
def parse_args():
    p = argparse.ArgumentParser("MIA from pt-shards (COCO) vs ImageNet100 dir with ROC/AUC/TPR@FPR1%")
    p.add_argument("--member_pt_shards_dir", type=str, required=True,
                   help="COCO member 的 pt 分片目录（如 /root/autodl-tmp/pytorch_datasets_shards/target_member_shards）")
    p.add_argument("--nonmember_dir", type=str, required=True,
                   help="ImageNet100 目录（如 /root/autodl-tmp/ImageNet100/imagenet100）")
    p.add_argument("--blip_dir", type=str, required=True,
                   help="本地微调 BLIP 目录")
    p.add_argument("--clf_ckpt", type=str, required=True,
                   help="风格分类器 ckpt（style_mlp_best.pt）")
    p.add_argument("--save_csv", type=str, required=True, help="逐样本打分 CSV")
    p.add_argument("--roc_csv", type=str, required=True, help="ROC 点 CSV")
    p.add_argument("--roc_png", type=str, required=True, help="ROC 曲线 PNG")
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--n_each", type=int, default=1000, help="两侧各取样本数")
    p.add_argument("--threshold", type=float, default=0.5, help="ACC 计算用阈值")
    return p.parse_args()

@torch.no_grad()
def main():
    args = parse_args()
    os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
    os.environ.setdefault("HF_HUB_OFFLINE","1")

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # 取样（各 n_each）
    members   = load_member_from_pt_shards(args.member_pt_shards_dir, args.n_each)
    non_paths = sample_nonmember_from_dir(args.nonmember_dir, args.n_each, seed=args.seed)

    # 构造数据与 loader
    class _DS(Dataset):
        def __init__(self, m, n): self.m, self.n = m, n
        def __len__(self): return len(self.m) + len(self.n)
        def __getitem__(self, i):
            if i < len(self.m):
                img, tag = self.m[i]; return img, 1, tag
            p = self.n[i - len(self.m)]
            img = Image.open(p).convert("RGB")
            return img, 0, p

    ds = _DS(members, non_paths)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=False,
                    collate_fn=lambda b: ( [x[0] for x in b],
                                           torch.tensor([x[1] for x in b], dtype=torch.long),
                                           [x[2] for x in b]) )

    # 加载特征器 & 分类器
    feat = BLIPFeatureExtractor(args.blip_dir, device=device)
    ck = torch.load(args.clf_ckpt, map_location="cpu")
    in_dim = ck["in_dim"]; hidden = ck.get("hidden", 256); dropout = ck.get("dropout", 0.1)
    clf = MLPClassifier(in_dim, hidden, dropout, 2).to(device)
    clf.load_state_dict(ck["state_dict"]); clf.eval()

    # 推理
    y_true, y_score, recs = [], [], []
    for pil_list, labels, tags in tqdm(dl, desc="Scoring", dynamic_ncols=True):
        feats  = feat(pil_list)
        logits = clf(feats.to(device))
        prob   = torch.softmax(logits, dim=1)[:,1].cpu().numpy()  # synth 概率
        y_true.extend(labels.numpy().tolist())
        y_score.extend(prob.tolist())
        recs.extend(list(zip(tags, labels.numpy().tolist(), prob.tolist())))

    y_true  = np.array(y_true, dtype=np.int64)
    y_score = np.array(y_score, dtype=np.float32)

    # ACC
    acc = acc_at_threshold(y_true, y_score, thr=args.threshold)

    # ROC & AUC
    fpr, tpr = build_roc_points(y_true, y_score)
    auc = auc_trapz(fpr, tpr)

    # TPR@FPR=1%
    tpr_at_1 = tpr_at_fpr(fpr, tpr, target_fpr=0.01)

    # 保存逐样本 CSV
    os.makedirs(os.path.dirname(args.save_csv), exist_ok=True)
    with open(args.save_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id_or_path","gt_member(1/0)","prob_synth","pred_member(1/0;thr={:.2f})".format(args.threshold)])
        for tag, gt, s in recs:
            w.writerow([tag, gt, s, int(s >= args.threshold)])

    # 保存 ROC CSV
    os.makedirs(os.path.dirname(args.roc_csv), exist_ok=True)
    with open(args.roc_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["FPR","TPR"])
        for x, y in zip(fpr, tpr):
            w.writerow([x, y])

    # 画 ROC PNG
    os.makedirs(os.path.dirname(args.roc_png), exist_ok=True)
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
    # 参考线
    plt.plot([0,1],[0,1], linestyle="--", lw=1)
    # 标注 1% 点
    plt.scatter([0.01],[tpr_at_1], s=30)
    plt.text(0.011, tpr_at_1, f"TPR@1%FPR = {tpr_at_1:.3f}", fontsize=9)
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (MIA via Style Classifier)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.roc_png, dpi=150)
    plt.close()

    # 打印汇总
    print("\n=== MIA Summary ===")
    print(f"ACC(thr={args.threshold:.2f}): {acc:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"TPR@FPR=1%: {tpr_at_1:.4f}")
    print(f"ROC CSV -> {args.roc_csv}")
    print(f"ROC PNG -> {args.roc_png}")
    print(f"Detail CSV -> {args.save_csv}")
    print(f"Members = {len(members)}, NonMembers = {len(non_paths)}, Total = {len(recs)}")

if __name__ == "__main__":
    main()
