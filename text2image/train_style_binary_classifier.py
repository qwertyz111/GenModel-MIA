# -*- coding: utf-8 -*-
"""
离线风格二分类器（ImageNet原图 vs SD合成图）
- 冻结本地微调后的 BLIP 视觉编码器，提特征
- 训练一个小 MLP 做 2 类判别
- 与 build_caption.py 产物对齐：IN100_1k.pt + IN100_synth_1k/img_0000_01.jpg ...
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoProcessor, BlipForConditionalGeneration
from typing import List

def parse_args():
    p = argparse.ArgumentParser("Train a binary style classifier (ImageNet vs SD synth)")
    p.add_argument("--in_pt", type=str, required=True,
                   help="原图 .pt（如 /root/autodl-tmp/exp/IN100_1k.pt，内含 {'image': list}）")
    p.add_argument("--synth_dir", type=str, required=True,
                   help="合成图目录（如 /root/autodl-tmp/exp/IN100_synth_1k）")
    p.add_argument("--blip_dir", type=str, required=True,
                   help="本地微调 BLIP 目录（如 /root/autodl-tmp/outputs/blip_finetune_ultra/blip_ft_final）")
    p.add_argument("--save_dir", type=str, default="/root/autodl-tmp/outputs/style_classifier")
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--cls_hidden", type=int, default=256, help="MLP中间维度")
    p.add_argument("--dropout", type=float, default=0.1)
    return p.parse_args()

def set_seed(s):
    import random, numpy as np
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s); np.random.seed(s)

class PairedStyleDataset(Dataset):
    """
    将 ImageNet 原图（来自 .pt）与 SD 合成图（来自目录）配对：
    - 原图记为 label=0
    - 合成图记为 label=1
    合并后再返回一个总样本列表：[(PIL, label), ...]
    """
    def __init__(self, in_images: List[Image.Image], synth_dir: str):
        self.samples = []
        n = len(in_images)
        # 1) 原图
        for i in range(n):
            self.samples.append((in_images[i], 0, f"in_{i}"))
        # 2) 合成图：按 build_caption.py 的命名规则 img_{i:04d}_01.jpg
        miss = 0
        for i in range(n):
            fname = f"img_{i:04d}_01.jpg"
            fpath = os.path.join(synth_dir, fname)
            if os.path.isfile(fpath):
                self.samples.append((fpath, 1, f"synth_{i}"))
            else:
                miss += 1
        if miss > 0:
            print(f"[WARN] Missing synth images: {miss}/{n} will be skipped for synth half.")
        # 最终样本量 ~ 2n - miss

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obj, label, tag = self.samples[idx]
        if isinstance(obj, Image.Image):
            img = obj.convert("RGB")
        else:
            img = Image.open(obj).convert("RGB")
        return img, label

class BLIPFeatureExtractor(nn.Module):
    """
    使用微调好的 BLIP 视觉编码器提特征（冻结参数）。
    返回 CLS token 的特征向量：shape [B, D]
    """
    def __init__(self, blip_dir: str, device: torch.device):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(blip_dir, local_files_only=True)
        self.blip = BlipForConditionalGeneration.from_pretrained(
            blip_dir, local_files_only=True
        )
        # 冻结
        for p in self.blip.parameters():
            p.requires_grad = False
        self.blip.eval().to(device)
        self.device = device

    @torch.no_grad()
    def forward(self, pil_images: List[Image.Image]) -> torch.Tensor:
        enc = self.processor(images=pil_images, return_tensors="pt", padding="max_length")
        pixel_values = enc["pixel_values"].to(self.device)
        # 只走视觉编码器（不走语言/解码器）
        vision_out = self.blip.vision_model(pixel_values=pixel_values)
        last_hidden = vision_out.last_hidden_state  # [B, L, D]
        cls_embed = last_hidden[:, 0]               # [B, D] 取 CLS
        return cls_embed

class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.1, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)

def collate_images(batch):
    # batch: list of (PIL, label)
    imgs, labels = zip(*batch)
    return list(imgs), torch.tensor(labels, dtype=torch.long)

def evaluate(feat_extractor: BLIPFeatureExtractor, classifier: MLPClassifier, loader: DataLoader, device: torch.device):
    classifier.eval()
    correct, total, loss_sum, steps = 0, 0, 0.0, 0
    ce = nn.CrossEntropyLoss()
    with torch.no_grad():
        for pil_list, labels in loader:
            feats = feat_extractor(pil_list)
            logits = classifier(feats.to(device))
            loss = ce(logits, labels.to(device))
            preds = torch.argmax(logits, dim=1)
            correct += (preds.cpu() == labels).sum().item()
            total += labels.size(0)
            loss_sum += loss.item()
            steps += 1
    acc = correct / max(1, total)
    return acc, loss_sum / max(1, steps)

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # 读取 .pt
    data = torch.load(args.in_pt, map_location="cpu")
    assert "image" in data and isinstance(data["image"], list) and len(data["image"]) > 0, \
        f"Bad IN pt: {args.in_pt}"
    in_images = data["image"]

    # 构造数据集
    full_ds = PairedStyleDataset(in_images, args.synth_dir)
    n_total = len(full_ds)
    n_val = max(1, int(n_total * args.val_ratio))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=False, collate_fn=collate_images)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=False, collate_fn=collate_images)

    # BLIP 特征抽取器（冻结）
    feat_extractor = BLIPFeatureExtractor(args.blip_dir, device=device)
    # 通过一次前传确定维度
    with torch.no_grad():
        feats = feat_extractor([in_images[0]])
        in_dim = feats.shape[-1]

    # MLP 分类头
    clf = MLPClassifier(in_dim=in_dim, hidden=args.cls_hidden, dropout=args.dropout, num_classes=2).to(device)
    optimizer = torch.optim.AdamW(clf.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ce = nn.CrossEntropyLoss()

    best_acc, best_path = 0.0, None

    for ep in range(1, args.epochs + 1):
        clf.train()
        pbar = tqdm(train_loader, desc=f"Train ep{ep}/{args.epochs}", dynamic_ncols=True)
        for pil_list, labels in pbar:
            feats = feat_extractor(pil_list)                 # [B, D] on device
            logits = clf(feats.to(device))                   # [B, 2]
            loss = ce(logits, labels.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(clf.parameters(), 1.0)
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # 验证
        val_acc, val_loss = evaluate(feat_extractor, clf, val_loader, device)
        print(f"[VAL] ep{ep}: acc={val_acc:.4f}, loss={val_loss:.4f}")

        # 保存最好
        if val_acc >= best_acc:
            best_acc = val_acc
            best_path = os.path.join(args.save_dir, "style_mlp_best.pt")
            torch.save({"state_dict": clf.state_dict(), "in_dim": in_dim,
                        "hidden": args.cls_hidden, "dropout": args.dropout}, best_path)
            print(f"[SAVE] best model -> {best_path}")

    # 收尾：保存最后一版
    last_path = os.path.join(args.save_dir, "style_mlp_last.pt")
    torch.save({"state_dict": clf.state_dict(), "in_dim": in_dim,
                "hidden": args.cls_hidden, "dropout": args.dropout}, last_path)
    print(f"[DONE] best_acc={best_acc:.4f}, saved last to {last_path}")

if __name__ == "__main__":
    main()
