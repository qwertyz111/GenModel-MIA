#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified MIA-style separability experiments (36 configs, FIXED):

Datasets:   agnews / wiki103 / xsum
Generators: gpt2 / tiiuae/falcon-7b-instruct
Train sizes: 1000 / 2000 / 3000
Noise modes: clean / noise

This version fixes:
  - GPT2/Falcon sometimes producing too few samples (now loops until enough)
  - tokenizer right-padding warning fixed (padding_side='left' for decoder-only)
  - No negative samples → ROC AUC error (fixed by guaranteeing synthetic count)
"""

import os
import math
import random
import argparse
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
)

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Utils
# ---------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def batched(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


# ---------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------

def load_real_texts_agnews(n):
    ds = load_dataset("ag_news")
    return ds["train"]["text"][:n]


def load_real_texts_wiki103(n, len_min=30, len_max=160):
    ds = load_dataset("wikitext", "wikitext-103-raw-v1")
    out = []
    for t in ds["train"]["text"]:
        if isinstance(t, str):
            t = t.strip()
            L = len(t.split())
            if len_min <= L <= len_max:
                out.append(t)
        if len(out) >= n:
            break
    return out[:n]


def load_real_texts_xsum(n, len_min=10, len_max=80):
    ds = load_dataset("sentence-transformers/xsum")
    out = []
    for t in ds["train"]["summary"]:
        if isinstance(t, str):
            t = t.strip()
            L = len(t.split())
            if len_min <= L <= len_max:
                out.append(t)
        if len(out) >= n:
            break
    return out[:n]


def load_real_texts(dataset, n):
    if dataset == "agnews":
        return load_real_texts_agnews(n)
    elif dataset == "wiki103":
        return load_real_texts_wiki103(n)
    elif dataset == "xsum":
        return load_real_texts_xsum(n)
    else:
        raise ValueError(dataset)


# ---------------------------------------------------------
# Generator prompts
# ---------------------------------------------------------

def dataset_prompts(dataset):
    if dataset == "agnews":
        return [
            "Category: World. Write a news headline and brief: ",
            "Category: Sports. Write a news headline and brief: ",
            "Category: Business. Write a news headline and brief: ",
            "Category: Sci/Tech. Write a news headline and brief: ",
        ]
    elif dataset == "wiki103":
        return [
            "Write a Wikipedia-style paragraph about a historical event: ",
            "Write a Wikipedia-style paragraph about a scientific concept: ",
            "Write a Wikipedia-style paragraph about a famous figure: ",
            "Write a Wikipedia-style paragraph about a geographic location: ",
        ]
    elif dataset == "xsum":
        return [
            "Write a BBC-style concise news summary about politics: ",
            "Write a BBC-style concise news summary about sports: ",
            "Write a BBC-style concise news summary about business: ",
            "Write a BBC-style concise news summary about science & tech: ",
        ]
    else:
        raise ValueError(dataset)


# ---------------------------------------------------------
# Generator loader
# ---------------------------------------------------------

def load_lm_and_tok(model_name, device):
    tok = AutoTokenizer.from_pretrained(model_name)

    # Fix padding side for decoder-only models (GPT2/Falcon)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if "cuda" in device else None,
        low_cpu_mem_usage=True,
    ).to(device)

    model.eval()
    return tok, model


# ---------------------------------------------------------
# FIXED synthetic generator (guarantee n samples)
# ---------------------------------------------------------

@torch.inference_mode()
def generate_synthetic_texts(dataset, model_name, n_needed, device,
                             max_new_tokens=80, temperature=0.9, top_p=0.95, batch_size=16):

    tok, lm = load_lm_and_tok(model_name, device)
    base_prompts = dataset_prompts(dataset)

    texts = []

    # keep generating until enough
    while len(texts) < n_needed:
        to_gen = min(batch_size, n_needed - len(texts))
        prompts = [random.choice(base_prompts) for _ in range(to_gen)]

        inputs = tok(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
            add_special_tokens=False,
        ).to(device)

        out = lm.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id,
        )

        decoded = tok.batch_decode(out, skip_special_tokens=True)

        for dec, p in zip(decoded, prompts):
            dec = dec.strip()
            if dec.startswith(p):
                dec = dec[len(p):].strip()
            texts.append(dec)

    return texts[:n_needed]


# ---------------------------------------------------------
# Encoder (DistilBERT)
# ---------------------------------------------------------

@torch.inference_mode()
def encode_texts(texts, device, enc_name="distilbert-base-uncased", batch_size=64, trunc_len=192):
    tok = AutoTokenizer.from_pretrained(enc_name)
    enc = AutoModel.from_pretrained(enc_name).to(device)
    enc.eval()

    embs = []
    for chunk in tqdm(list(batched(texts, batch_size)), desc="Encoding"):
        b = tok(
            chunk, padding=True, truncation=True,
            max_length=trunc_len, return_tensors="pt"
        )
        b = to_device(b, device)
        out = enc(**b).last_hidden_state
        h = F.normalize(out[:, 0, :], dim=-1)
        embs.append(h.cpu().numpy())
    return np.vstack(embs)


# ---------------------------------------------------------
# Gaussian modeling
# ---------------------------------------------------------

def fit_gaussian(embs, cov_reg=1e-2):
    mu = embs.mean(axis=0)
    cov = np.cov(embs.T) + cov_reg * np.eye(embs.shape[1])
    inv = np.linalg.pinv(cov)
    logdet = float(np.log(np.linalg.det(cov) + 1e-12))
    return {"mu": mu, "inv": inv, "logdet": logdet, "dim": embs.shape[1]}


def loglik(x, g):
    diff = (x - g["mu"])
    maha = (diff @ g["inv"] * diff).sum(axis=-1)
    return -0.5 * (maha + g["logdet"] + g["dim"] * math.log(2 * math.pi))


def add_noise(embs, std):
    if std <= 0:
        return embs
    return embs + np.random.normal(0, std, size=embs.shape)


# ---------------------------------------------------------
# Core experiment
# ---------------------------------------------------------

def run_for_dataset_model(dataset, gen_model, device, enc_name, train_sizes,
                          eval_size, cov_reg, noise_std, results, out_dir):

    max_train = max(train_sizes)
    n_total = max_train + eval_size

    print(f"\n=== [{dataset}] × [{gen_model}] ===")
    print(f"Need {n_total} REAL & SYN samples")

    # ----- Real -----
    real_texts = load_real_texts(dataset, n_total)
    print(f"REAL loaded = {len(real_texts)}")

    # ----- Synthetic -----
    syn_texts = generate_synthetic_texts(dataset, gen_model, n_total, device)
    print(f"SYN generated = {len(syn_texts)}")

    # ----- Encode -----
    emb_real_all = encode_texts(real_texts, device=device, enc_name=enc_name)
    emb_syn_all = encode_texts(syn_texts, device=device, enc_name=enc_name)

    # scale
    scaler = StandardScaler().fit(np.vstack([emb_real_all, emb_syn_all]))
    emb_real_all = scaler.transform(emb_real_all)
    emb_syn_all = scaler.transform(emb_syn_all)

    # EVAL portion
    real_eval_base = emb_real_all[max_train:max_train + eval_size]
    syn_eval_base = emb_syn_all[max_train:max_train + eval_size]

    model_short = "gpt2" if "gpt2" in gen_model else "falcon"

    for N in train_sizes:

        real_train = emb_real_all[:N]
        syn_train = emb_syn_all[:N]

        g_real = fit_gaussian(real_train, cov_reg)
        g_syn = fit_gaussian(syn_train, cov_reg)

        for noise_mode in ["clean", "noise"]:
            if noise_mode == "noise":
                real_eval = add_noise(real_eval_base.copy(), noise_std)
                syn_eval = add_noise(syn_eval_base.copy(), noise_std)
            else:
                real_eval = real_eval_base
                syn_eval = syn_eval_base

            scores_real = loglik(real_eval, g_real) - loglik(real_eval, g_syn)
            scores_syn = loglik(syn_eval, g_real) - loglik(syn_eval, g_syn)

            y_true = np.array([1]*len(scores_real) + [0]*len(scores_syn))
            y_score = np.concatenate([scores_real, scores_syn])

            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_val = roc_auc_score(y_true, y_score)

            # TPR@1%FPR
            m = fpr <= 0.01
            tpr_1 = float(np.max(tpr[m])) if np.any(m) else 0.0

            # Accuracy
            y_pred = (y_score > 0).astype(int)
            acc = float((y_pred == y_true).mean())

            # Save fig
            fig_name = f"{dataset}_{model_short}_{N}_{noise_mode}.png"
            fig_path = os.path.join(out_dir, fig_name)

            plt.figure(figsize=(5, 5))
            plt.plot(fpr, tpr, label=f"AUC={auc_val:.3f}")
            plt.plot([0, 1], [0, 1], "k--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"{dataset} – {model_short}, N={N}, {noise_mode}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(fig_path, dpi=150)
            plt.close()

            print(f"[Saved] {fig_path}")

            results.append({
                "dataset": dataset,
                "model": model_short,
                "train_size": N,
                "noise": noise_mode,
                "accuracy": acc,
                "auc": auc_val,
                "tpr_at_1fpr": tpr_1
            })


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--enc_name", type=str, default="distilbert-base-uncased")
    ap.add_argument("--cov_reg", type=float, default=1e-2)
    ap.add_argument("--noise_std", type=float, default=0.2)
    ap.add_argument("--eval_size", type=int, default=500)
    ap.add_argument("--out_dir", type=str, default="results_figs")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    datasets = ["agnews", "wiki103", "xsum"]
    models = ["gpt2", "tiiuae/falcon-7b-instruct"]
    train_sizes = [1000, 2000, 3000]

    results = []

    for d in datasets:
        for m in models:
            run_for_dataset_model(
                dataset=d,
                gen_model=m,
                device=args.device,
                enc_name=args.enc_name,
                train_sizes=train_sizes,
                eval_size=args.eval_size,
                cov_reg=args.cov_reg,
                noise_std=args.noise_std,
                results=results,
                out_dir=args.out_dir,
            )

    # summary table
    print("\n=== SUMMARY ===")
    print("dataset, model, N, noise, acc, auc, tpr@1%FPR")

    with open("results_summary.csv", "w") as f:
        f.write("dataset,model,train_size,noise,accuracy,auc,tpr_at_1fpr\n")
        for r in results:
            print(r)
            f.write("{dataset},{model},{train_size},{noise},{accuracy:.4f},{auc:.4f},{tpr_at_1fpr:.4f}\n".format(**r))


if __name__ == "__main__":
    main()

