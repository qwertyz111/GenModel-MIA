#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training-based MIA on AG_NEWS (GPT-2 as target model T)

Ablation 4: Different feature extractors E,
while keeping the original distance:
  - use mean embedding of D_syn and D_real
  - distance = cosine(E(x), center_syn / center_real)
  - score(x) = dist_syn - dist_real  (越大越像 member)

Encoders to compare (可通过 --enc_list 修改):
  - distilbert-base-uncased
  - roberta-base
  - sentence-transformers/all-MiniLM-L6-v2
"""

import argparse
import math
import random
import csv
import os
import re
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    set_seed,
)
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt


# -------------------- helpers --------------------
def batched(it, n):
    for i in range(0, len(it), n):
        yield it[i:i + n]


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(1.0 - np.dot(a, b))


# -------------------- encode with a given extractor --------------------
@torch.inference_mode()
def encode_texts(
    texts: List[str],
    enc_name: str,
    device: str,
    batch_size: int,
    trunc_len: int,
) -> np.ndarray:
    tok = AutoTokenizer.from_pretrained(enc_name)
    enc = AutoModel.from_pretrained(enc_name).to(device)
    enc.eval()

    embs = []
    for chunk in tqdm(list(batched(texts, batch_size)), desc=f"Encoding ({enc_name})"):
        batch = tok(
            chunk,
            padding=True,
            truncation=True,
            max_length=trunc_len,
            return_tensors="pt",
        )
        batch = to_device(batch, device)
        out = enc(**batch).last_hidden_state       # [B, L, H]
        cls = out[:, 0, :]                         # [B, H]
        cls = F.normalize(cls, dim=-1)
        embs.append(cls.cpu().numpy())

    del tok, enc
    torch.cuda.empty_cache()
    return np.vstack(embs)                         # [N, H]


# -------------------- GPT-2 generator -> D_syn --------------------
@torch.inference_mode()
def generate_synthetic_texts(
    n_per_class: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: str,
    gpt_name: str = "gpt2",
) -> List[str]:
    tok = AutoTokenizer.from_pretrained(gpt_name)
    tok.pad_token = tok.eos_token
    gpt = AutoModelForCausalLM.from_pretrained(gpt_name).to(device)
    gpt.eval()

    prompts = {
        0: "Category: World. News headline and brief: ",
        1: "Category: Sports. News headline and brief: ",
        2: "Category: Business. News headline and brief: ",
        3: "Category: Sci/Tech. News headline and brief: ",
    }

    texts = []
    for y in range(4):
        p = prompts[y]
        inputs = tok([p] * n_per_class, return_tensors="pt", padding=True).to(device)
        out = gpt.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.eos_token_id,
        )
        gen = tok.batch_decode(out, skip_special_tokens=True)
        gen = [g[len(p):].strip() if g.startswith(p) else g.strip() for g in gen]
        texts += gen

    del tok, gpt
    torch.cuda.empty_cache()
    return texts


# -------------------- main --------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # 数据量
    p.add_argument("--n_train_per_class", type=int, default=1000)
    p.add_argument("--n_val_per_class", type=int, default=200)
    p.add_argument("--n_syn_per_class", type=int, default=1000)

    # 目标模型 T
    p.add_argument("--gpt_name", type=str, default="gpt2")

    # 不同的 extractor（Ablation 4）
    p.add_argument(
        "--enc_list",
        type=str,
        default="distilbert-base-uncased,roberta-base,sentence-transformers/all-MiniLM-L6-v2",
        help="comma-separated list of encoder names",
    )

    # 生成 / 编码参数
    p.add_argument("--max_new_tokens", type=int, default=40)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--trunc_len", type=int, default=192)

    p.add_argument("--csv_out", type=str, default="agnews_ablation_extractors_cosine.csv")
    args = p.parse_args()
    print(vars(args))

    set_seed(args.seed)
    device = args.device

    # -------------------- 共享真实数据 & 合成数据 --------------------
    print("\nLoading AG_NEWS ...")
    ds = load_dataset("ag_news")
    train_texts, train_labels = ds["train"]["text"], ds["train"]["label"]
    test_texts,  test_labels  = ds["test"]["text"],  ds["test"]["label"]

    def take_per_class(texts, labels, n_per_class):
        buckets = {0: [], 1: [], 2: [], 3: []}
        for t, y in zip(texts, labels):
            if len(buckets[y]) < n_per_class:
                buckets[y].append(t)
            if all(len(v) >= n_per_class for v in buckets.values()):
                break
        flat = [t for c in range(4) for t in buckets[c]]
        return flat

    real_train_texts = take_per_class(train_texts, train_labels, args.n_train_per_class)
    real_val_texts   = take_per_class(test_texts,  test_labels,  args.n_val_per_class)

    print("\nGenerating GPT-2 synthetic texts (shared for all extractors) ...")
    syn_texts = generate_synthetic_texts(
        n_per_class=args.n_syn_per_class,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=device,
        gpt_name=args.gpt_name,
    )

    neg_samples = syn_texts[:len(real_val_texts)]

    # -------------------- 遍历不同 extractor --------------------
    encoders = [s.strip() for s in args.enc_list.split(",") if s.strip()]
    print("\nEncoders (Ablation 4):", encoders)

    if os.path.exists(args.csv_out):
        os.remove(args.csv_out)
    header_written = False

    for enc_name in encoders:
        print("\n" + "=" * 80)
        print(f"[Extractor] {enc_name}")
        print("=" * 80)

        # 1) embedding
        print("Encoding real_train ...")
        emb_real = encode_texts(real_train_texts, enc_name, device,
                                batch_size=args.batch_size, trunc_len=args.trunc_len)
        print("Encoding syn_texts ...")
        emb_syn  = encode_texts(syn_texts, enc_name, device,
                                batch_size=args.batch_size, trunc_len=args.trunc_len)
        print("Encoding real_val (positives) ...")
        emb_val  = encode_texts(real_val_texts, enc_name, device,
                                batch_size=args.batch_size, trunc_len=args.trunc_len)
        print("Encoding synthetic negatives ...")
        emb_neg  = encode_texts(neg_samples, enc_name, device,
                                batch_size=args.batch_size, trunc_len=args.trunc_len)

        # 2) 分布中心：mean（保持 original distance 设计）
        center_real = emb_real.mean(axis=0)
        center_syn  = emb_syn.mean(axis=0)

        # 3) 打分：score = dist_syn - dist_real  （cosine 距离）
        y_true  = [1] * len(emb_val) + [0] * len(emb_neg)
        y_score = []

        print("Scoring positives (real_val) ...")
        for h in emb_val:
            d_real = cosine_dist(h, center_real)
            d_syn  = cosine_dist(h, center_syn)
            y_score.append(d_syn - d_real)

        print("Scoring negatives (synthetic) ...")
        for h in emb_neg:
            d_real = cosine_dist(h, center_real)
            d_syn  = cosine_dist(h, center_syn)
            y_score.append(d_syn - d_real)

        # 4) ROC / AUC
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        if auc < 0.5:
            y_score = [-s for s in y_score]
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc = roc_auc_score(y_true, y_score)

        tpr_at_1fpr = max(tpr[np.array(fpr) <= 0.01]) if np.any(np.array(fpr) <= 0.01) else 0.0
        preds = [1 if s > 0 else 0 for s in y_score]
        acc = np.mean([int(p == y) for p, y in zip(preds, y_true)])

        print(f"\n[Result] enc={enc_name}")
        print(f"Accuracy={acc:.4f}  ROC-AUC={auc:.6f}  TPR@1%FPR={tpr_at_1fpr:.3f}")

        # 5) 保存 ROC 图
        safe_enc = re.sub(r"[^a-zA-Z0-9]+", "_", enc_name)
        fig_name = f"roc_agnews_{safe_enc}_cosine_mean.png"
        plt.figure(figsize=(5, 5))
        plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.scatter([0.01], [tpr_at_1fpr], color="C1",
                    label=f'TPR@1%FPR={tpr_at_1fpr:.3f}')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"AG_NEWS – {enc_name}")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(fig_name, dpi=150)
        plt.close()
        print(f"Saved ROC: {fig_name}")

        # 6) 写入 CSV
        row = {
            "encoder": enc_name,
            "accuracy": round(float(acc), 6),
            "auc": round(float(auc), 6),
            "tpr_at_1fpr": round(float(tpr_at_1fpr), 6),
        }
        with open(args.csv_out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=row.keys())
            if not header_written:
                w.writeheader()
                header_written = True
            w.writerow(row)

    print(f"\nAll done. CSV saved to: {args.csv_out}")


if __name__ == "__main__":
    main()
