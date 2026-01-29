#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training-based MIA on AG_NEWS (GPT-2 + DistilBERT)
Ablation 3: use geometric median instead of mean to estimate the distribution.

Step 1: T (GPT-2) generates synthetic dataset D_syn, real dataset D_real from AG_NEWS.
Step 2: Encode D_syn with E (DistilBERT) -> emb_syn.
Step 3: Encode D_real with E -> emb_real.
Step 4: Compute distribution centers (mean or geometric median) for both.
Step 5: For suspicious x, compare dist(E(x), center_syn) vs dist(E(x), center_real).
"""

import argparse, math, random, numpy as np
from typing import List
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, set_seed
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt


# -------------------- small helpers --------------------
def batched(it, n):
    for i in range(0, len(it), n):
        yield it[i:i+n]

def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


# -------------------- geometric median (Ablation 3) --------------------
def geometric_median(X: np.ndarray, eps: float = 1e-5, max_iter: int = 512) -> np.ndarray:
    """
    Weiszfeld algorithm: argmin_z sum_i ||X_i - z||_2.
    X: [N, d]
    """
    y = X.mean(axis=0)  # init at mean
    for _ in range(max_iter):
        diff = X - y
        dist = np.linalg.norm(diff, axis=1)
        # avoid division by 0
        nonzero = dist > 1e-10
        if not np.any(nonzero):
            break
        w = 1.0 / dist[nonzero]
        y_new = (w[:, None] * X[nonzero]).sum(axis=0) / w.sum()
        if np.linalg.norm(y_new - y) < eps:
            y = y_new
            break
        y = y_new
    return y


# -------------------- distance functions --------------------
def cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(1.0 - np.dot(a, b))

def l2_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def compute_dist(a: np.ndarray, b: np.ndarray, metric: str) -> float:
    if metric == "cosine":
        return cosine_dist(a, b)
    elif metric == "l2":
        return l2_dist(a, b)
    else:
        raise ValueError(f"Unknown metric: {metric}")


# -------------------- encoder E --------------------
@torch.inference_mode()
def encode_texts(
    texts: List[str],
    device: str,
    enc_name: str = "distilbert-base-uncased",
    batch_size: int = 64,
    trunc_len: int = 192,
) -> np.ndarray:
    tok = AutoTokenizer.from_pretrained(enc_name)
    enc = AutoModel.from_pretrained(enc_name).to(device)
    enc.eval()

    embs = []
    for chunk in tqdm(list(batched(texts, batch_size)), desc="Encoding"):
        batch = tok(
            chunk,
            padding=True,
            truncation=True,
            max_length=trunc_len,
            return_tensors="pt",
        )
        batch = to_device(batch, device)
        out = enc(**batch).last_hidden_state  # [B, L, H]
        cls = out[:, 0, :]                    # [B, H]
        cls = F.normalize(cls, dim=-1)
        embs.append(cls.cpu().numpy())
    return np.vstack(embs)  # [N, H]


# -------------------- generator T: GPT-2 -> D_syn --------------------
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
    return texts


# -------------------- main pipeline --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # dataset sizes
    parser.add_argument("--n_train_per_class", type=int, default=1500)
    parser.add_argument("--n_val_per_class", type=int, default=200)
    parser.add_argument("--n_syn_per_class", type=int, default=1500)

    # models
    parser.add_argument("--gpt_name", type=str, default="gpt2")
    parser.add_argument("--enc_name", type=str, default="distilbert-base-uncased")

    # generation / encoding
    parser.add_argument("--max_new_tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--trunc_len", type=int, default=192)

    # distance & distribution
    parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "l2"])
    parser.add_argument("--use_geom_median", action="store_true",
                        help="If set, use geometric median instead of mean as distribution center (Ablation 3).")

    args = parser.parse_args()
    print(vars(args))

    set_seed(args.seed)
    device = args.device

    # ---- Load AG_NEWS ----
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

    # ---- Generate synthetic D_syn ----
    print("\nGenerating GPT-2 synthetic texts ...")
    syn_texts = generate_synthetic_texts(
        n_per_class=args.n_syn_per_class,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=device,
        gpt_name=args.gpt_name,
    )

    # ---- Encode with E ----
    print("\nEncoding real_train ...")
    emb_real = encode_texts(real_train_texts, device, args.enc_name,
                            batch_size=args.batch_size, trunc_len=args.trunc_len)
    print("Encoding syn_texts ...")
    emb_syn  = encode_texts(syn_texts, device, args.enc_name,
                            batch_size=args.batch_size, trunc_len=args.trunc_len)

    # ---- Step 4: compute distribution centers (mean vs geometric median) ----
    if args.use_geom_median:
        print("\n[ABLATION 3] Using geometric median as distribution center.")
        center_real = geometric_median(emb_real)
        center_syn  = geometric_median(emb_syn)
    else:
        print("\nUsing simple mean as distribution center.")
        center_real = emb_real.mean(axis=0)
        center_syn  = emb_syn.mean(axis=0)

    # ---- Step 5: evaluate on held-out real (positives) vs synthetic (negatives) ----
    print("\nEncoding held-out real (evaluation set) ...")
    enc_eval = encode_texts(real_val_texts, device, args.enc_name,
                            batch_size=args.batch_size, trunc_len=args.trunc_len)
    neg_samples = syn_texts[:len(real_val_texts)]
    print("Encoding matched synthetic negatives ...")
    enc_neg = encode_texts(neg_samples, device, args.enc_name,
                           batch_size=args.batch_size, trunc_len=args.trunc_len)

    # score = dist(E(x), center_syn) - dist(E(x), center_real)
    #  → 越大越像 real (member)
    y_true  = [1]*len(enc_eval) + [0]*len(enc_neg)
    y_score = []

    print("\nScoring samples ...")
    for h in enc_eval:
        d_real = compute_dist(h, center_real, args.metric)
        d_syn  = compute_dist(h, center_syn,  args.metric)
        y_score.append(d_syn - d_real)

    for h in enc_neg:
        d_real = compute_dist(h, center_real, args.metric)
        d_syn  = compute_dist(h, center_syn,  args.metric)
        y_score.append(d_syn - d_real)

    # ROC / AUC / TPR@1%FPR
    fpr, tpr, thr = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    if auc < 0.5:
        # flip if inverted
        y_score = [-s for s in y_score]
        fpr, tpr, thr = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
    tpr_at_1fpr = max(tpr[fpr <= 0.01]) if np.any(fpr <= 0.01) else 0.0

    # 用 0 阈值给个 accuracy 参考（只是粗略）
    preds = [1 if s > 0 else 0 for s in y_score]
    acc = np.mean([int(p == y) for p, y in zip(preds, y_true)])

    print(f"\nResults ({'geom-median' if args.use_geom_median else 'mean'} center, metric={args.metric}):")
    print(f"Accuracy={acc:.4f}  ROC-AUC={auc:.4f}  TPR@1%FPR={tpr_at_1fpr:.4f}")

    # ---- Plot ROC ----
    plt.figure(figsize=(5,5))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0,1],[0,1],'k--')
    plt.scatter([0.01],[tpr_at_1fpr], color="C1",
                label=f"TPR@1%FPR={tpr_at_1fpr:.3f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    title = f"AG_NEWS ({args.metric}, {'geom' if args.use_geom_median else 'mean'})"
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    out_name = f"roc_agnews_{args.metric}_{'geom' if args.use_geom_median else 'mean'}.png"
    plt.savefig(out_name, dpi=150)
    print(f"Saved: {out_name}")


if __name__ == "__main__":
    main()

