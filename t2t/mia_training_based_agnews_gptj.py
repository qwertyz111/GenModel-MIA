#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Membership-style separability on AG_NEWS (Real vs Synthetic) with GPT-J generator.

What this script does
---------------------
1) Loads AG_NEWS (4 classes).
2) Generates class-conditioned synthetic texts with GPT-J-6B (fallback to GPT-2 if OOM/unavailable).
   - Removes the prompt from decoded outputs to avoid prompt leakage.
3) Encodes texts with a transformer encoder (default: distilbert-base-uncased).
4) Fits Gaussian distributions (mean+cov) on REAL vs SYNTHETIC embeddings.
   - Either globally or per-class (class-conditional Gaussians).
5) Scores held-out REAL (positives) vs SYNTHETIC (negatives), prints metrics and plots ROC.

Suggested environment
---------------------
pip install torch transformers datasets scikit-learn tqdm matplotlib

Example runs
------------
# Global version
python mia_training_based_agnews.py \
  --device cuda:0 \
  --n_train_per_class 1000 --n_val_per_class 200 --n_syn_per_class 1000 \
  --gpt_name EleutherAI/gpt-j-6B --max_new_tokens 40 --batch_size 64

# Class-conditional version
python mia_training_based_agnews.py \
  --device cuda:0 \
  --n_train_per_class 1000 --n_val_per_class 200 --n_syn_per_class 1000 \
  --class_conditional
"""
import os
import math
import random
import argparse
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    set_seed,
)

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# ------------------------------
# Utils
# ------------------------------
LABEL2NAME = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

def batched(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}

def fit_gaussian(embs: np.ndarray, eps: float = 1e-2) -> Dict[str, np.ndarray]:
    mu = embs.mean(axis=0)
    cov = np.cov(embs.T) + eps * np.eye(embs.shape[1])
    inv = np.linalg.pinv(cov)
    logdet = float(np.log(np.linalg.det(cov) + 1e-12))
    return {"mu": mu, "inv": inv, "logdet": logdet, "dim": embs.shape[1]}

def loglik_gaussian(x: np.ndarray, g: Dict[str, np.ndarray]) -> np.ndarray:
    # x: [N, D] or [1, D]
    diff = (x - g["mu"])
    maha = (diff @ g["inv"] * diff).sum(axis=-1)  # (x-mu)^T inv (x-mu)
    return -0.5 * (maha + g["logdet"] + g["dim"] * math.log(2 * math.pi))


# ------------------------------
# Step 1: Generator T -> synthetic AG_NEWS (GPT-J with safe fallback)
# ------------------------------
@torch.inference_mode()
def generate_synthetic_texts(
    n_per_class: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: str,
    gpt_name: str = "EleutherAI/gpt-j-6B",
) -> Tuple[List[str], List[int]]:
    """
    Generate AG_NEWS-style texts conditioned on a simple class prompt.

    - Tries GPT-J-6B first (more realistic), falls back to GPT-2 if OOM/unavailable.
    - Removes the prompt portion from decoded output (slice by token count).
    """
    def _load_generator(name: str):
        print(f"\n[Generator] Loading model: {name}")
        tok = AutoTokenizer.from_pretrained(name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token if tok.eos_token is not None else tok.unk_token
        model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=torch.float16 if "cuda" in device else None,
            low_cpu_mem_usage=True,
        ).to(device)
        model.eval()
        return tok, model

    # Try GPT-J, else fallback to GPT-2
    try:
        tok, gpt = _load_generator(gpt_name)
    except Exception as e:
        print(f"[Generator] Failed to load {gpt_name}: {e}\nFalling back to GPT-2.")
        tok, gpt = _load_generator("gpt2")

    prompts = {
        0: "Category: World. News headline and brief: ",
        1: "Category: Sports. News headline and brief: ",
        2: "Category: Business. News headline and brief: ",
        3: "Category: Sci/Tech. News headline and brief: ",
    }

    texts, labels = [], []
    for y in range(4):
        p = prompts[y]
        inputs = tok([p] * n_per_class, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        # Record prompt token length to slice continuation only
        prompt_len = inputs["input_ids"].shape[1]

        out = gpt.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id if tok.eos_token_id is not None else tok.pad_token_id,
            use_cache=True,
        )
        # Batch-slice: take only continuation tokens beyond the prompt
        cont_tokens = [seq[prompt_len:] for seq in out]
        gen = [tok.decode(ct, skip_special_tokens=True).strip() for ct in cont_tokens]

        texts.extend(gen)
        labels.extend([y] * n_per_class)

    print(f"[Generator] Synthetic samples: {len(texts)}")
    return texts, labels


# ------------------------------
# Step 2: Encoder E -> text embeddings
# ------------------------------
@torch.inference_mode()
def encode_texts(
    texts: List[str],
    device: str,
    enc_name: str = "distilbert-base-uncased",
    batch_size: int = 1,
    normalize: bool = True,
    trunc_len: int = 192,
) -> np.ndarray:
    tok = AutoTokenizer.from_pretrained(enc_name)
    enc = AutoModel.from_pretrained(enc_name).to(device).eval()

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
        # Use first token embedding (DistilBERT has no pooled output)
        h = out[:, 0, :]                      # [B, H]
        if normalize:
            h = F.normalize(h, dim=-1)
        embs.append(h.cpu().numpy())
    return np.vstack(embs)  # [N, H]


# ------------------------------
# Data helpers
# ------------------------------
def take_per_class(texts, labels, n_per_class):
    buckets = {0: [], 1: [], 2: [], 3: []}
    for t, y in zip(texts, labels):
        if len(buckets[y]) < n_per_class:
            buckets[y].append(t)
        if all(len(v) >= n_per_class for v in buckets.values()):
            break
    flat = [t for y in range(4) for t in buckets[y]]
    labs = [y for y in range(4) for _ in range(n_per_class)]
    return flat, labs


# ------------------------------
# Main pipeline
# ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # data sizes
    parser.add_argument("--n_train_per_class", type=int, default=1500,
                        help="real samples used to build D_real distribution (per class)")
    parser.add_argument("--n_val_per_class", type=int, default=200,
                        help="held-out real samples used for eval (per class)")
    parser.add_argument("--n_syn_per_class", type=int, default=1500,
                        help="synthetic samples per class for D_syn")

    # generator settings
    parser.add_argument("--gpt_name", type=str, default="EleutherAI/gpt-j-6B")
    parser.add_argument("--max_new_tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.95)

    # encoder settings
    parser.add_argument("--enc_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--trunc_len", type=int, default=192)
    parser.add_argument("--cov_reg", type=float, default=1e-2)

    # mixture / global vs class-conditional
    parser.add_argument("--class_conditional", action="store_true",
                        help="Fit per-class Gaussians and score with true class label "
                             "(for real_val) and generation class (for syn).")

    args = parser.parse_args()
    print(vars(args))
    set_seed(args.seed)
    device = args.device
    torch.set_grad_enabled(False)

    # ----- Load AG_NEWS real dataset -----
    ds = load_dataset("ag_news")
    train_texts, train_labels = ds["train"]["text"], ds["train"]["label"]
    test_texts,  test_labels  = ds["test"]["text"],  ds["test"]["label"]

    # Subsample per class for distribution fitting & validation
    real_train_texts, real_train_labels = take_per_class(train_texts, train_labels, args.n_train_per_class)
    real_val_texts,   real_val_labels   = take_per_class(test_texts,  test_labels,  args.n_val_per_class)

    # ----- Step 1: Generate synthetic data (class-conditioned) -----
    syn_texts, syn_labels = generate_synthetic_texts(
        n_per_class=args.n_syn_per_class,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=device,
        gpt_name=args.gpt_name,
    )

    # ----- Step 2: Encode real & synthetic -----
    emb_real = encode_texts(real_train_texts, device, args.enc_name, args.batch_size, True, args.trunc_len)
    emb_syn  = encode_texts(syn_texts,        device, args.enc_name, args.batch_size, True, args.trunc_len)

    # Standardize jointly to avoid trivial scale differences
    scaler = StandardScaler().fit(np.vstack([emb_real, emb_syn]))
    emb_real = scaler.transform(emb_real)
    emb_syn  = scaler.transform(emb_syn)

    # ----- Step 3/4: Fit distributions -----
    if args.class_conditional:
        # Build per-class matrices using labels
        real_by_c = {c: [] for c in range(4)}
        syn_by_c  = {c: [] for c in range(4)}

        # Map texts->emb indexing (encode once more aligned to labels)
        # (Re-encode only to align; or simpler: encode again per text in order)
        print("Aligning embeddings per class (real)...")
        emb_real_cc = encode_texts(real_train_texts, device, args.enc_name, args.batch_size, True, args.trunc_len)
        emb_real_cc = scaler.transform(emb_real_cc)
        for e, y in zip(emb_real_cc, real_train_labels):
            real_by_c[y].append(e)

        print("Aligning embeddings per class (synthetic)...")
        emb_syn_cc = encode_texts(syn_texts, device, args.enc_name, args.batch_size, True, args.trunc_len)
        emb_syn_cc = scaler.transform(emb_syn_cc)
        for e, y in zip(emb_syn_cc, syn_labels):
            syn_by_c[y].append(e)

        dist_real = {c: fit_gaussian(np.vstack(real_by_c[c]), eps=args.cov_reg) for c in range(4)}
        dist_syn  = {c: fit_gaussian(np.vstack(syn_by_c[c]),  eps=args.cov_reg) for c in range(4)}
    else:
        dist_real = fit_gaussian(emb_real, eps=args.cov_reg)
        dist_syn  = fit_gaussian(emb_syn,  eps=args.cov_reg)

    # ----- Step 5: Evaluate (counts + ROC/AUC) -----
    enc_tok_eval = AutoTokenizer.from_pretrained(args.enc_name)
    enc_model_eval = AutoModel.from_pretrained(args.enc_name).to(device).eval()

    def embed_one(text: str) -> np.ndarray:
        b = enc_tok_eval([text], padding=True, truncation=True, max_length=args.trunc_len, return_tensors="pt")
        b = to_device(b, device)
        h = enc_model_eval(**b).last_hidden_state[:, 0, :]
        h = F.normalize(h, dim=-1).cpu().numpy()
        return scaler.transform(h)  # [1, D]

    tp = fp = tn = fn = 0
    y_true, y_score = [], []

    # Real held-out (positives)
    for txt, y in tqdm(list(zip(real_val_texts, real_val_labels)), desc="Eval real-heldout"):
        h = embed_one(txt)
        if args.class_conditional:
            ll_r = loglik_gaussian(h, dist_real[y])
            ll_s = loglik_gaussian(h, dist_syn[y])
        else:
            ll_r = loglik_gaussian(h, dist_real)
            ll_s = loglik_gaussian(h, dist_syn)
        llr = float(ll_r - ll_s)
        pred_member = (llr > 0.0)
        tp += int(pred_member)
        fn += int(not pred_member)
        y_true.append(1)
        y_score.append(llr)

    # Synthetic negatives (match count)
    neg_samples = syn_texts[:len(real_val_texts)]
    neg_labels  = syn_labels[:len(real_val_texts)]
    for txt, y in tqdm(list(zip(neg_samples, neg_labels)), desc="Eval synthetic negatives"):
        h = embed_one(txt)
        if args.class_conditional:
            ll_r = loglik_gaussian(h, dist_real[y])
            ll_s = loglik_gaussian(h, dist_syn[y])
        else:
            ll_r = loglik_gaussian(h, dist_real)
            ll_s = loglik_gaussian(h, dist_syn)
        llr = float(ll_r - ll_s)
        pred_member = (llr > 0.0)
        fp += int(pred_member)
        tn += int(not pred_member)
        y_true.append(0)
        y_score.append(llr)

    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    acc       = (tp + tn) / (tp + tn + fp + fn + 1e-9)

    print("\nResults (real-heldout as positives, synthetic as negatives):")
    print(f"TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"Accuracy={acc:.4f}  Precision={precision:.4f}  Recall={recall:.4f}")

    # ROC / AUC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    # auto-flip if inverted
    if auc < 0.5:
        y_score = [-s for s in y_score]
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
    tpr_at_1fpr = max(tpr[fpr <= 0.01]) if np.any(np.array(fpr) <= 0.01) else 0.0
    print(f"\nROC-AUC={auc:.4f}  TPR@1%FPR={tpr_at_1fpr:.4f}")

    # Plot
    plt.figure(figsize=(5,5))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0,1], [0,1], 'k--')
    plt.scatter([0.01], [tpr_at_1fpr], label=f'TPR@1%FPR={tpr_at_1fpr:.3f}')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("AG_NEWS – Real vs Synthetic (GPT-J generator)" + (" [class-cond]" if args.class_conditional else " [global]"))
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("roc_curve_agnews.png", dpi=150)
    print("Saved: roc_curve_agnews.png")

    # Example suspicious sample
    demo_text = real_val_texts[0]
    demo_h = embed_one(demo_text)
    if args.class_conditional:
        y_demo = real_val_labels[0]
        demo_pred = (loglik_gaussian(demo_h, dist_real[y_demo]) > loglik_gaussian(demo_h, dist_syn[y_demo]))
    else:
        demo_pred = (loglik_gaussian(demo_h, dist_real) > loglik_gaussian(demo_h, dist_syn))
    print("\nExample suspicious sample:")
    print(demo_text[:200].replace("\n", " ") + ("..." if len(demo_text) > 200 else ""))
    print("→ Predicted:", "MEMBER" if bool(demo_pred) else "NON-MEMBER")


if __name__ == "__main__":
    main()

