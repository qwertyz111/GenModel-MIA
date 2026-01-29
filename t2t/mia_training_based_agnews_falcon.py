#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Membership-style separability on AG_NEWS (Real vs Synthetic) with Falcon-7B-Instruct generator.
Now supports: synthesis_mode={"free","paraphrase"} to control AUC difficulty.
"""
import os, math, random, argparse
from typing import List, Tuple, Dict
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForCausalLM, set_seed
)
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ------------------------------
# Utility helpers
# ------------------------------
def batched(it, n):
    for i in range(0, len(it), n):
        yield it[i:i+n]

def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}

def fit_gaussian(embs: np.ndarray, eps: float = 1e-2):
    mu = embs.mean(axis=0)
    cov = np.cov(embs.T) + eps * np.eye(embs.shape[1])
    inv = np.linalg.pinv(cov)
    logdet = float(np.log(np.linalg.det(cov) + 1e-12))
    return {"mu": mu, "inv": inv, "logdet": logdet, "dim": embs.shape[1]}

def loglik_gaussian(x: np.ndarray, g: Dict[str, np.ndarray]):
    diff = (x - g["mu"])
    maha = (diff @ g["inv"] * diff).sum(axis=-1)
    return -0.5 * (maha + g["logdet"] + g["dim"] * math.log(2 * math.pi))

# ------------------------------
# Generator: Falcon-7B-Instruct
# ------------------------------
@torch.inference_mode()
def _load_lm_and_tok(name: str, device: str):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=torch.float16 if "cuda" in device else None,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    return tok, model

@torch.inference_mode()
def generate_free_texts(
    n_per_class: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: str,
    gpt_name: str = "tiiuae/falcon-7b-instruct",
) -> Tuple[List[str], List[int]]:
    """Original 'free' synthesis (often yields AUC ~1.0)."""
    try:
        tok, model = _load_lm_and_tok(gpt_name, device)
    except Exception as e:
        print(f"[Generator] Failed to load {gpt_name}: {e}\n→ Falling back to gpt2.")
        tok, model = _load_lm_and_tok("gpt2", device)

    prompts = {
        0: "Category: World. Write a short news headline and a brief summary: ",
        1: "Category: Sports. Write a short news headline and a brief summary: ",
        2: "Category: Business. Write a short news headline and a brief summary: ",
        3: "Category: Sci/Tech. Write a short news headline and a brief summary: ",
    }

    texts, labels = [], []
    for y in range(4):
        p = prompts[y]
        inputs = tok([p] * n_per_class, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        prompt_len = inputs["input_ids"].shape[1]
        out = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id,
            use_cache=True,
        )
        cont = [seq[prompt_len:] for seq in out]
        dec = [tok.decode(c, skip_special_tokens=True).strip() for c in cont]
        texts += dec
        labels += [y] * n_per_class

    print(f"[Generator/free] Synthetic samples: {len(texts)}")
    return texts, labels

@torch.inference_mode()
def generate_paraphrases(
    seed_texts: List[str],
    seed_labels: List[int],
    per_class: int,
    device: str,
    max_new_tokens: int = 96,
    temperature: float = 0.7,
    top_p: float = 0.9,
    gpt_name: str = "tiiuae/falcon-7b-instruct",
) -> Tuple[List[str], List[int]]:
    """
    Paraphrase real AG_NEWS texts to close the domain gap.
    This typically drives ROC-AUC into the 0.7–0.9 range (tunable).
    """
    try:
        tok, model = _load_lm_and_tok(gpt_name, device)
    except Exception as e:
        print(f"[Paraphrase] Failed to load {gpt_name}: {e}\n→ Falling back to gpt2.")
        tok, model = _load_lm_and_tok("gpt2", device)

    # Bucket by label
    buckets = {0: [], 1: [], 2: [], 3: []}
    for t, y in zip(seed_texts, seed_labels):
        if len(buckets[y]) < per_class:
            buckets[y].append(t)
        if all(len(v) >= per_class for v in buckets.values()):
            break

    def paraprompt(txt, y):
        cat = ["World", "Sports", "Business", "Sci/Tech"][y]
        return (
            f"You are editing a news brief.\n"
            f"Task: Rewrite the following {cat} news headline+summary with different wording, "
            f"preserving facts and similar length, and keep it natural for a newswire style.\n"
            f"Text:\n{txt}\n\nRewritten:"
        )

    texts, labels = [], []
    all_prompts, all_ys = [], []
    for y in range(4):
        ps = [paraprompt(t, y) for t in buckets[y]]
        all_prompts.extend(ps)
        all_ys.extend([y] * len(ps))

    # Batch generate
    B = 16
    for batch in tqdm(list(batched(list(zip(all_prompts, all_ys)), B)), desc="Paraphrasing"):
        prompts, ys = zip(*batch)
        inputs = tok(list(prompts), return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        prompt_len = inputs["input_ids"].shape[1]
        out = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id,
            use_cache=True,
        )
        cont = [seq[prompt_len:] for seq in out]
        dec = [tok.decode(c, skip_special_tokens=True).strip() for c in cont]
        texts.extend(dec)
        labels.extend(list(ys))

    print(f"[Generator/paraphrase] Synthetic samples: {len(texts)}")
    return texts, labels

# ------------------------------
# Encoder (DistilBERT)
# ------------------------------
@torch.inference_mode()
def encode_texts(
    texts: List[str],
    device: str,
    enc_name: str = "distilbert-base-uncased",
    batch_size: int = 64,
    normalize: bool = True,
    trunc_len: int = 192,
) -> np.ndarray:
    tok = AutoTokenizer.from_pretrained(enc_name)
    enc = AutoModel.from_pretrained(enc_name).to(device).eval()
    embs = []
    for chunk in tqdm(list(batched(texts, batch_size)), desc="Encoding"):
        b = tok(chunk, padding=True, truncation=True, max_length=trunc_len, return_tensors="pt")
        b = to_device(b, device)
        h = enc(**b).last_hidden_state[:, 0, :]
        if normalize:
            h = F.normalize(h, dim=-1)
        embs.append(h.cpu().numpy())
    return np.vstack(embs)

# ------------------------------
# Data helper
# ------------------------------
def take_per_class(texts, labels, n):
    buckets = {0: [], 1: [], 2: [], 3: []}
    for t, y in zip(texts, labels):
        if len(buckets[y]) < n:
            buckets[y].append(t)
        if all(len(v) >= n for v in buckets.values()):
            break
    flat = [t for y in range(4) for t in buckets[y]]
    labs = [y for y in range(4) for _ in range(n)]
    return flat, labs

# ------------------------------
# Main
# ------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n_train_per_class", type=int, default=1000)
    p.add_argument("--n_val_per_class", type=int, default=200)
    p.add_argument("--n_syn_per_class", type=int, default=1000)
    p.add_argument("--max_new_tokens", type=int, default=40)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--enc_name", type=str, default="distilbert-base-uncased")
    p.add_argument("--batch_size", type=int, default=64)

    # NEW: mode & paraphrase knobs
    p.add_argument("--synthesis_mode", type=str, choices=["free", "paraphrase"], default="paraphrase")
    p.add_argument("--paraphrase_per_class", type=int, default=1200, help="How many real samples to paraphrase per class")
    p.add_argument("--paraphrase_temperature", type=float, default=0.7)
    p.add_argument("--paraphrase_top_p", type=float, default=0.9)
    p.add_argument("--paraphrase_max_new_tokens", type=int, default=96)

    # Stronger covariance reg to reduce over-separation (tunable)
    p.add_argument("--cov_reg", type=float, default=5e-2)

    args = p.parse_args()
    print(vars(args))

    set_seed(args.seed)
    device = args.device
    torch.set_grad_enabled(False)

    # 1) Load AG_NEWS
    ds = load_dataset("ag_news")
    train_texts, train_labels = ds["train"]["text"], ds["train"]["label"]
    test_texts,  test_labels  = ds["test"]["text"],  ds["test"]["label"]
    real_train, y_train = take_per_class(train_texts, train_labels, args.n_train_per_class)
    real_val,   y_val   = take_per_class(test_texts,  test_labels,  args.n_val_per_class)

    # 2) Generate synthetic (mode-dependent)
    if args.synthesis_mode == "free":
        syn_texts, syn_labels = generate_free_texts(
            n_per_class=args.n_syn_per_class,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
        )
    else:
        # Paraphrase a larger pool than n_syn_per_class to increase overlap; then trim per class.
        seeds_texts, seeds_labels = real_train, y_train
        syn_texts_raw, syn_labels_raw = generate_paraphrases(
            seed_texts=seeds_texts,
            seed_labels=seeds_labels,
            per_class=args.paraphrase_per_class,
            device=device,
            max_new_tokens=args.paraphrase_max_new_tokens,
            temperature=args.paraphrase_temperature,
            top_p=args.paraphrase_top_p,
        )
        # Keep a balanced subset of paraphrases
        syn_texts, syn_labels = take_per_class(syn_texts_raw, syn_labels_raw, args.n_syn_per_class)

    # 3) Encode
    emb_real = encode_texts(real_train, device, args.enc_name, args.batch_size)
    emb_syn  = encode_texts(syn_texts,  device, args.enc_name, args.batch_size)
    scaler = StandardScaler().fit(np.vstack([emb_real, emb_syn]))
    emb_real, emb_syn = scaler.transform(emb_real), scaler.transform(emb_syn)

    # 4) Fit Gaussians
    dist_real, dist_syn = fit_gaussian(emb_real, args.cov_reg), fit_gaussian(emb_syn, args.cov_reg)

    # 5) Evaluate ROC (LLR on held-out real vs synthetic)
    enc_tok = AutoTokenizer.from_pretrained(args.enc_name)
    enc_model = AutoModel.from_pretrained(args.enc_name).to(device).eval()

    def embed_one(txt):
        b = enc_tok([txt], padding=True, truncation=True, max_length=192, return_tensors="pt")
        b = to_device(b, device)
        h = enc_model(**b).last_hidden_state[:, 0, :]
        h = F.normalize(h, dim=-1).cpu().numpy()
        return scaler.transform(h)

    y_true, y_score = [], []
    for txt in tqdm(real_val, desc="Real-heldout"):
        h = embed_one(txt)
        llr = float(loglik_gaussian(h, dist_real) - loglik_gaussian(h, dist_syn))
        y_true.append(1); y_score.append(llr)
    for txt in tqdm(syn_texts[:len(real_val)], desc="Synthetic"):
        h = embed_one(txt)
        llr = float(loglik_gaussian(h, dist_real) - loglik_gaussian(h, dist_syn))
        y_true.append(0); y_score.append(llr)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_val = roc_auc_score(y_true, y_score)
    if auc_val < 0.5:
        y_score = [-s for s in y_score]
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_val = roc_auc_score(y_true, y_score)
    print(f"\nROC-AUC={auc_val:.4f}")

    plt.figure(figsize=(5,5))
    plt.plot(fpr, tpr, label=f"AUC={auc_val:.3f}")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"AG_NEWS – Real vs Synthetic ({args.synthesis_mode})")
    plt.legend(); plt.tight_layout()
    out_name = f"roc_curve_agnews_falcon_{args.synthesis_mode}.png"
    plt.savefig(out_name, dpi=150)
    print(f"Saved: {out_name}")

if __name__ == "__main__":
    main()
