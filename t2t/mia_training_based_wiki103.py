#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
True MIA-style test on WikiText-103:
Members = train-split samples; Non-members = held-out test-split samples
No GPT-2 generation (avoids trivial real vs synthetic separability)

Pipeline:
- Length filtering & matching
- Encode with transformer
- Single scaler (fit on union of FIT splits)
- Fit Gaussians on disjoint FIT subsets for each class
- Evaluate on disjoint EVAL subsets (balanced)
- Report ROC-AUC and TPR@1%FPR
"""

import argparse, math, random, numpy as np
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

# -------------------- args --------------------
p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, default=42)
p.add_argument("--device", type=str, default="cuda:0")
p.add_argument("--n_train_total", type=int, default=2000, help="total train (member) samples to pull")
p.add_argument("--n_test_total",  type=int, default=2000, help="total test (non-member) samples to pull")
p.add_argument("--n_fit_per_class", type=int, default=1000, help="samples for fitting each Gaussian")
p.add_argument("--n_eval_per_class", type=int, default=500, help="samples for evaluation for each class")
p.add_argument("--enc_name", type=str, default="distilbert-base-uncased")
p.add_argument("--batch_size", type=int, default=64)
p.add_argument("--len_min", type=int, default=30)
p.add_argument("--len_max", type=int, default=160)
p.add_argument("--trunc_len", type=int, default=96, help="hard truncate tokens to this length for both sets")
p.add_argument("--cov_reg", type=float, default=1e-3)
p.add_argument("--auto_flip_auc", action="store_true")
args = p.parse_args()
print(vars(args))

# -------------------- setup --------------------
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
device = torch.device(args.device)

def ok_len(t):
    if not t or not isinstance(t, str): return False
    n = len(t.split())
    return (args.len_min <= n <= args.len_max)

def trim(t):
    toks = t.split()
    if len(toks) > args.trunc_len:
        toks = toks[:args.trunc_len]
    return " ".join(toks)

# -------------------- load & sample --------------------
print("\nLoading WikiText-103 ...")
ds = load_dataset("wikitext", "wikitext-103-raw-v1")

train_pool = [trim(t) for t in ds["train"]["text"] if ok_len(t)]
test_pool  = [trim(t) for t in ds["test"]["text"]  if ok_len(t)]
random.shuffle(train_pool); random.shuffle(test_pool)

# pull requested totals
members_all    = train_pool[:args.n_train_total]  # "member" = from train split
nonmembers_all = test_pool[:args.n_test_total]    # "non-member" = from test split

print(f"Members pool={len(members_all)}  Non-members pool={len(nonmembers_all)}")
print("Avg lengths (tokens): members=%.1f, nonmembers=%.1f" %
      (np.mean([len(t.split()) for t in members_all]),
       np.mean([len(t.split()) for t in nonmembers_all])))

# split into FIT and EVAL subsets (disjoint)
def split_fit_eval(pool, n_fit, n_eval):
    assert n_fit + n_eval <= len(pool), "Pool too small for requested fit/eval sizes."
    fit = pool[:n_fit]
    eval_ = pool[n_fit:n_fit+n_eval]
    return fit, eval_

mem_fit, mem_eval   = split_fit_eval(members_all,    args.n_fit_per_class, args.n_eval_per_class)
non_fit, non_eval   = split_fit_eval(nonmembers_all, args.n_fit_per_class, args.n_eval_per_class)

print(f"Fit sizes: members={len(mem_fit)}, nonmembers={len(non_fit)}")
print(f"Eval sizes: members={len(mem_eval)}, nonmembers={len(non_eval)}")

# -------------------- encoder --------------------
tok = AutoTokenizer.from_pretrained(args.enc_name)
enc = AutoModel.from_pretrained(args.enc_name, torch_dtype=torch.float16).to(device).eval()

def encode_texts(texts, batch_size=None):
    if batch_size is None: batch_size = args.batch_size
    embs = []
    loader = DataLoader(texts, batch_size=batch_size)
    with torch.no_grad():
        for batch in tqdm(loader, desc="Encoding"):
            inputs = tok(list(batch), padding=True, truncation=True,
                         max_length=args.trunc_len+16, return_tensors="pt").to(device)
            h = enc(**inputs).last_hidden_state[:,0,:]
            h = torch.nn.functional.normalize(h, dim=-1)
            embs.append(h.cpu())
    return torch.cat(embs).numpy()

print("\nEncoding FIT subsets ...")
emb_mem_fit  = encode_texts(mem_fit)
emb_non_fit  = encode_texts(non_fit)

# scaler fit on union of FIT sets (neutral)
scaler = StandardScaler().fit(np.vstack([emb_mem_fit, emb_non_fit]))
emb_mem_fit = scaler.transform(emb_mem_fit)
emb_non_fit = scaler.transform(emb_non_fit)

# -------------------- Gaussian modeling --------------------
def fit_gaussian(embs, cov_reg):
    mu = np.mean(embs, axis=0)
    cov = np.cov(embs.T) + cov_reg * np.eye(embs.shape[1])
    inv = np.linalg.pinv(cov)
    logdet = np.log(np.linalg.det(cov) + 1e-12)
    return {"mu": mu, "inv": inv, "logdet": logdet, "dim": embs.shape[1]}

def loglik(x, g):
    diff = (x - g["mu"])
    maha = (diff @ g["inv"] * diff).sum(axis=-1)
    return -0.5 * (maha + g["logdet"] + g["dim"] * math.log(2 * math.pi))

print("Fitting Gaussians ...")
g_mem = fit_gaussian(emb_mem_fit, args.cov_reg)
g_non = fit_gaussian(emb_non_fit, args.cov_reg)

# -------------------- evaluation --------------------
print("\nEncoding EVAL subsets ...")
emb_mem_eval = scaler.transform(encode_texts(mem_eval))
emb_non_eval = scaler.transform(encode_texts(non_eval))

def score_emb(emb):
    # returns llr per row
    llr = loglik(emb, g_mem) - loglik(emb, g_non)
    return llr

y_true  = [1]*len(emb_mem_eval) + [0]*len(emb_non_eval)
y_score = list(score_emb(emb_mem_eval)) + list(score_emb(emb_non_eval))

fpr, tpr, _ = roc_curve(y_true, y_score)
auc = roc_auc_score(y_true, y_score)
if args.auto_flip_auc and auc < 0.5:
    y_score = [-s for s in y_score]
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

tpr_at_1fpr = max(tpr[fpr <= 0.01]) if any(fpr <= 0.01) else 0.0

print(f"\nROC-AUC={auc:.4f}  TPR@1%FPR={tpr_at_1fpr:.4f}")

plt.figure(figsize=(5,5))
plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
plt.plot([0,1],[0,1],'k--')
plt.scatter([0.01],[tpr_at_1fpr], color='r', label=f'TPR@1%FPR={tpr_at_1fpr:.3f}')
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("WikiText-103 True MIA: Train (Members) vs Held-out Test (Non-members)")
plt.legend(loc="lower right"); plt.tight_layout()
plt.savefig("roc_curve_wiki103_true_mia.png", dpi=150)
print("Saved: roc_curve_wiki103_true_mia.png")
