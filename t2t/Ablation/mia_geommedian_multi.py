#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training-based MIA – Ablation 3: geometric median as distribution center.

- Datasets: agnews / wiki103 / xsum
- Target models T: gpt2, tiiuae/falcon-7b-instruct
- Encoder E: distilbert-base-uncased (fixed)
- Center of D_syn, D_real: geometric median instead of mean
- dist = cosine distance
- score(x) = dist(E(x), center_syn) - dist(E(x), center_real)
"""

import argparse
import math
import random
import csv
from typing import List, Tuple

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


# ----------------- small helpers -----------------
def batched(it, n):
    for i in range(0, len(it), n):
        yield it[i:i + n]


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(1.0 - np.dot(a, b))


def geometric_median(points: np.ndarray, eps: float = 1e-5, max_iter: int = 100) -> np.ndarray:
    """Weiszfeld algorithm."""
    y = points.mean(axis=0)
    for _ in range(max_iter):
        diff = points - y
        dist = np.linalg.norm(diff, axis=1)
        dist = np.where(dist < eps, eps, dist)
        w = 1.0 / dist
        y_new = (points * w[:, None]).sum(axis=0) / w.sum()
        if np.linalg.norm(y_new - y) < eps:
            break
        y = y_new
    return y


# ----------------- dataset loaders -----------------
def load_texts_agnews(n_fit: int, n_eval: int, seed: int) -> Tuple[List[str], List[str]]:
    ds = load_dataset("ag_news")
    train_texts = list(ds["train"]["text"])
    test_texts = list(ds["test"]["text"])
    random.seed(seed)
    random.shuffle(train_texts)
    random.shuffle(test_texts)
    real_fit = train_texts[:n_fit]
    real_eval = test_texts[:n_eval]
    return real_fit, real_eval


def load_texts_wiki103(n_fit: int, n_eval: int, seed: int) -> Tuple[List[str], List[str]]:
    ds = load_dataset("wikitext", "wikitext-103-raw-v1")
    corpus = [t for t in ds["train"]["text"] if t and len(t.split()) >= 20]
    random.seed(seed)
    random.shuffle(corpus)
    real_fit = corpus[:n_fit]
    real_eval = corpus[n_fit:n_fit + n_eval]
    return real_fit, real_eval


def load_texts_xsum(n_fit: int, n_eval: int, seed: int) -> Tuple[List[str], List[str]]:
    # parquet mirror to avoid local script conflict
    ds = load_dataset("sentence-transformers/xsum")
    docs = list(ds["train"]["article"])
    docs = [t for t in docs if t and len(t.split()) >= 20]
    random.seed(seed)
    random.shuffle(docs)
    real_fit = docs[:n_fit]
    real_eval = docs[n_fit:n_fit + n_eval]
    return real_fit, real_eval


def load_real_texts(dataset: str, n_fit: int, n_eval: int, seed: int):
    if dataset == "agnews":
        return load_texts_agnews(n_fit, n_eval, seed)
    if dataset == "wiki103":
        return load_texts_wiki103(n_fit, n_eval, seed)
    if dataset == "xsum":
        return load_texts_xsum(n_fit, n_eval, seed)
    raise ValueError(f"Unknown dataset {dataset}")


# ----------------- generator T -> synthetic texts -----------------
@torch.inference_mode()
def generate_synthetic_texts_for_dataset(
    dataset: str,
    tgt_name: str,
    n_fit: int,
    n_eval: int,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    total = n_fit + n_eval

    tok = AutoTokenizer.from_pretrained(tgt_name)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    # use fp16 for big models like falcon
    dtype = torch.float16 if "falcon" in tgt_name.lower() else None
    gpt = AutoModelForCausalLM.from_pretrained(tgt_name, torch_dtype=dtype).to(device)
    gpt.eval()

    if dataset == "agnews":
        prompts = [
            "News headline and brief: ",
            "World news story: ",
            "Business news: ",
            "Technology news: ",
        ]
    elif dataset == "wiki103":
        prompts = [
            "Wikipedia article: ",
            "Encyclopedia entry: ",
            "Short biography: ",
        ]
    else:  # xsum
        prompts = [
            "BBC news article: ",
            "News report: ",
            "Short news story: ",
        ]

    random.seed(seed)
    syn = []
    for _ in tqdm(range(total), desc=f"Sampling synthetic ({dataset}, {tgt_name})"):
        p = random.choice(prompts)
        inp = tok(p, return_tensors="pt").to(device)
        out = gpt.generate(
            **inp,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.eos_token_id,
        )
        txt = tok.decode(out[0], skip_special_tokens=True)
        if txt.startswith(p):
            txt = txt[len(p):]
        syn.append(txt.strip())
    del gpt, tok
    torch.cuda.empty_cache()

    syn_fit = syn[:n_fit]
    syn_eval = syn[n_fit:n_fit + n_eval]
    return syn_fit, syn_eval


# ----------------- encoder -----------------
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
        out = enc(**batch).last_hidden_state
        cls = out[:, 0, :]
        cls = F.normalize(cls, dim=-1)
        embs.append(cls.cpu().numpy())

    del tok, enc
    torch.cuda.empty_cache()
    return np.vstack(embs)


# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--datasets", type=str, default="agnews,wiki103,xsum")
    ap.add_argument("--targets", type=str, default="gpt2,tiiuae/falcon-7b-instruct")

    ap.add_argument("--n_fit", type=int, default=2000,
                    help="real/syn samples to build centers")
    ap.add_argument("--n_eval", type=int, default=2000,
                    help="real/syn samples for evaluation")

    ap.add_argument("--enc_name", type=str, default="distilbert-base-uncased")
    ap.add_argument("--enc_bs", type=int, default=64)
    ap.add_argument("--enc_trunc_len", type=int, default=192)

    ap.add_argument("--max_new_tokens", type=int, default=40)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=0.95)

    ap.add_argument("--csv_out", type=str, default="geommedian_results.csv")
    args = ap.parse_args()
    print(vars(args))

    set_seed(args.seed)
    device = torch.device(args.device)

    datasets = [s.strip() for s in args.datasets.split(",") if s.strip()]
    targets = [s.strip() for s in args.targets.split(",") if s.strip()]

    with open(args.csv_out, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "target_model", "encoder",
                        "accuracy", "auc", "tpr_at_1fpr"],
        )
        writer.writeheader()

        for ds_name in datasets:
            for tgt in targets:
                print("\n" + "=" * 80)
                print(f"[GeomMedian] dataset={ds_name}  target={tgt}")
                print("=" * 80)

                # ----- real texts -----
                real_fit, real_eval = load_real_texts(
                    ds_name, args.n_fit, args.n_eval, args.seed
                )

                # ----- synthetic texts from target -----
                syn_fit, syn_eval = generate_synthetic_texts_for_dataset(
                    ds_name,
                    tgt,
                    n_fit=args.n_fit,
                    n_eval=args.n_eval,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    seed=args.seed,
                )

                # ----- encode -----
                emb_real_fit = encode_texts(
                    real_fit, args.enc_name, device,
                    batch_size=args.enc_bs, trunc_len=args.enc_trunc_len
                )
                emb_syn_fit = encode_texts(
                    syn_fit, args.enc_name, device,
                    batch_size=args.enc_bs, trunc_len=args.enc_trunc_len
                )
                emb_real_eval = encode_texts(
                    real_eval, args.enc_name, device,
                    batch_size=args.enc_bs, trunc_len=args.enc_trunc_len
                )
                emb_syn_eval = encode_texts(
                    syn_eval, args.enc_name, device,
                    batch_size=args.enc_bs, trunc_len=args.enc_trunc_len
                )

                # ----- geometric median centers -----
                center_real = geometric_median(emb_real_fit)
                center_syn = geometric_median(emb_syn_fit)

                # ----- scoring -----
                y_true = [1] * len(emb_real_eval) + [0] * len(emb_syn_eval)
                y_score = []

                for h in emb_real_eval:
                    d_r = cosine_dist(h, center_real)
                    d_s = cosine_dist(h, center_syn)
                    y_score.append(d_s - d_r)
                for h in emb_syn_eval:
                    d_r = cosine_dist(h, center_real)
                    d_s = cosine_dist(h, center_syn)
                    y_score.append(d_s - d_r)

                fpr, tpr, _ = roc_curve(y_true, y_score)
                auc = roc_auc_score(y_true, y_score)
                if auc < 0.5:
                    y_score = [-s for s in y_score]
                    fpr, tpr, _ = roc_curve(y_true, y_score)
                    auc = roc_auc_score(y_true, y_score)
                tpr_at_1fpr = max(tpr[np.array(fpr) <= 0.01]) \
                    if np.any(np.array(fpr) <= 0.01) else 0.0
                preds = [1 if s > 0 else 0 for s in y_score]
                acc = float(np.mean([int(p == y) for p, y in zip(preds, y_true)]))

                print(f"Accuracy={acc:.4f}  AUC={auc:.6f}  TPR@1%FPR={tpr_at_1fpr:.3f}")

                writer.writerow({
                    "dataset": ds_name,
                    "target_model": tgt,
                    "encoder": args.enc_name,
                    "accuracy": round(acc, 6),
                    "auc": round(float(auc), 6),
                    "tpr_at_1fpr": round(float(tpr_at_1fpr), 6),
                })

    print(f"\nAll done. CSV saved to {args.csv_out}")

if __name__ == "__main__":
    main()


