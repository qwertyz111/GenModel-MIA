#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified MIA-style separability experiments (36 configs):

Datasets:   agnews / wiki103 / xsum
Generators: gpt2 / tiiuae/falcon-7b-instruct
Train sizes: 1000 / 2000 / 3000 (number of REAL + SYN samples used to fit Gaussians)
Noise modes: clean / noise (Gaussian noise on EVAL embeddings only)

Per (dataset, generator, train_size, noise_mode):
  - Use REAL texts from the dataset as "members".
  - Use GPT-2 / Falcon generated texts as "non-members".
  - Encode with DistilBERT.
  - Fit 2 Gaussians on train embeddings (REAL / SYN).
  - Score eval REAL vs eval SYN with log-likelihood ratio (LLR).
  - Compute ACC, AUC, TPR@1%FPR.
  - Save a ROC curve figure named:
      {dataset}_{model_short}_{train_size}_{noise_mode}.png

At the end:
  - Print a table of 36 rows.
  - Save the same table to 'results_summary.csv'.
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
# Basic utils
# ---------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def batched(it, n):
    for i in range(0, len(it), n):
        yield it[i:i+n]


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


# ---------------------------------------------------------
# Data loading: REAL texts for 3 datasets
# ---------------------------------------------------------

def load_real_texts_agnews(n_needed: int) -> List[str]:
    ds = load_dataset("ag_news")
    texts = list(ds["train"]["text"])
    # just take first n_needed
    return texts[:n_needed]


def load_real_texts_wiki103(n_needed: int,
                            len_min: int = 30,
                            len_max: int = 160) -> List[str]:
    ds = load_dataset("wikitext", "wikitext-103-raw-v1")
    pool = []
    for t in ds["train"]["text"]:
        if not isinstance(t, str):
            continue
        t = t.strip()
        if not t:
            continue
        n = len(t.split())
        if len_min <= n <= len_max:
            pool.append(t)
        if len(pool) >= n_needed:
            break
    return pool[:n_needed]


def load_real_texts_xsum(n_needed: int,
                         len_min: int = 10,
                         len_max: int = 80) -> List[str]:
    ds = load_dataset("sentence-transformers/xsum")  # fields: article, summary
    pool = []
    for t in ds["train"]["summary"]:
        if not isinstance(t, str):
            continue
        t = t.strip()
        if not t:
            continue
        n = len(t.split())
        if len_min <= n <= len_max:
            pool.append(t)
        if len(pool) >= n_needed:
            break
    return pool[:n_needed]


def load_real_texts(dataset: str, n_needed: int) -> List[str]:
    if dataset == "agnews":
        return load_real_texts_agnews(n_needed)
    elif dataset == "wiki103":
        return load_real_texts_wiki103(n_needed)
    elif dataset == "xsum":
        return load_real_texts_xsum(n_needed)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# ---------------------------------------------------------
# Generator (GPT-2 / Falcon) for synthetic texts
# ---------------------------------------------------------

def load_lm_and_tok(model_name: str, device: str):
    """
    model_name: "gpt2" or "tiiuae/falcon-7b-instruct" (or HF compatible name)
    """
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if "cuda" in device else None,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    return tok, model


def dataset_prompts(dataset: str) -> List[str]:
    """
    Base prompts for each dataset, repeated/cycled when generating many samples.
    """
    if dataset == "agnews":
        return [
            "Category: World. Write a short news headline and brief: ",
            "Category: Sports. Write a short news headline and brief: ",
            "Category: Business. Write a short news headline and brief: ",
            "Category: Sci/Tech. Write a short news headline and brief: ",
        ]
    elif dataset == "wiki103":
        return [
            "Write a Wikipedia-style paragraph about a historical event: ",
            "Write a Wikipedia-style paragraph about a scientific concept: ",
            "Write a Wikipedia-style paragraph about a famous person: ",
            "Write a Wikipedia-style paragraph about a geographic location: ",
        ]
    elif dataset == "xsum":
        return [
            "Write a concise BBC-style news summary about politics: ",
            "Write a concise BBC-style news summary about sports: ",
            "Write a concise BBC-style news summary about business: ",
            "Write a concise BBC-style news summary about science and technology: ",
        ]
    else:
        raise ValueError(f"Unknown dataset for prompts: {dataset}")

@torch.inference_mode()
def generate_synthetic_texts(
    dataset: str,
    model_name: str,
    n_needed: int,
    device: str,
    max_new_tokens: int = 80,
    temperature: float = 0.9,
    top_p: float = 0.95,
    batch_size: int = 16,
) -> List[str]:
    """
    Generic synthetic generator:
      - choose dataset-specific prompts
      - feed into GPT-2 / Falcon
      - strip prompt prefix from decoded text when possible

    修复点：
      * 不再用 zip(prompts, base_prompts) 截断为 4 条
      * 为每个样本直接选择一个 prompt（循环使用）
    """
    tok, lm = load_lm_and_tok(model_name, device)

    # 对 decoder-only 模型（gpt2 / falcon）建议使用 left padding，防止 warning
    if hasattr(tok, "padding_side"):
        tok.padding_side = "left"

    base_prompts = dataset_prompts(dataset)

    # 为 n_needed 个样本构造 prompt 序列（循环使用 base_prompts）
    prompts = [base_prompts[i % len(base_prompts)] for i in range(n_needed)]

    texts: List[str] = []

    for batch_prompts in tqdm(
        list(batched(prompts, batch_size)),
        desc=f"Synth {dataset} with {model_name}"
    ):
        inputs = tok(
            batch_prompts,
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
            use_cache=True,
        )

        decoded = tok.batch_decode(out, skip_special_tokens=True)

        # 把 prompt 前缀切掉，保留 continuation
        for dec, bp in zip(decoded, batch_prompts):
            dec = dec.strip()
            if dec.startswith(bp):
                dec = dec[len(bp):].strip()
            if not dec:
                # 如果极端情况下是空，就保留原文（不强行丢弃）
                dec = dec
            texts.append(dec)

    # 保证长度刚好是 n_needed
    if len(texts) < n_needed:
        # 极端情况下如果比 n_needed 少（几乎不会发生），重复补齐
        times = (n_needed + len(texts) - 1) // len(texts)
        texts = (texts * times)[:n_needed]
    else:
        texts = texts[:n_needed]

    print(f"[Generator] {dataset} with {model_name}: generated {len(texts)} samples (target={n_needed})")
    return texts

@torch.inference_mode()
def encode_texts(
    texts: List[str],
    device: str,
    enc_name: str = "distilbert-base-uncased",
    batch_size: int = 64,
    trunc_len: int = 192,
) -> np.ndarray:
    tok = AutoTokenizer.from_pretrained(enc_name)
    enc = AutoModel.from_pretrained(enc_name).to(device).eval()
    embs = []

    for chunk in tqdm(list(batched(texts, batch_size)), desc="Encoding"):
        b = tok(
            list(chunk),
            padding=True,
            truncation=True,
            max_length=trunc_len,
            return_tensors="pt",
        )
        b = to_device(b, device)
        out = enc(**b).last_hidden_state  # [B, L, H]
        h = out[:, 0, :]                  # CLS / first-token
        h = F.normalize(h, dim=-1)
        embs.append(h.cpu().numpy())
    return np.vstack(embs)


# ---------------------------------------------------------
# Gaussian modeling on embeddings
# ---------------------------------------------------------

def fit_gaussian(embs: np.ndarray, cov_reg: float = 1e-2) -> Dict[str, np.ndarray]:
    mu = embs.mean(axis=0)
    cov = np.cov(embs.T) + cov_reg * np.eye(embs.shape[1])
    inv = np.linalg.pinv(cov)
    logdet = float(np.log(np.linalg.det(cov) + 1e-12))
    return {"mu": mu, "inv": inv, "logdet": logdet, "dim": embs.shape[1]}


def loglik_gaussian(x: np.ndarray, g: Dict[str, np.ndarray]) -> np.ndarray:
    diff = (x - g["mu"])
    maha = (diff @ g["inv"] * diff).sum(axis=-1)
    return -0.5 * (maha + g["logdet"] + g["dim"] * math.log(2 * math.pi))


def add_noise(embs: np.ndarray, std: float) -> np.ndarray:
    if std <= 0.0:
        return embs
    noise = np.random.normal(0.0, std, size=embs.shape)
    return embs + noise


# ---------------------------------------------------------
# Experiment per (dataset, model)
# ---------------------------------------------------------

def run_for_dataset_model(
    dataset: str,
    gen_model: str,
    device: str,
    enc_name: str,
    train_sizes: List[int],
    eval_size: int,
    cov_reg: float,
    noise_std: float,
    results: List[Dict],
    out_dir: str,
):
    """
    For a fixed dataset + generator model, we:
      1) Load REAL texts (max_train + eval_size).
      2) Generate SYN texts (same count).
      3) Encode both once.
      4) Fit scaler on union.
      5) For each train_size + noise_mode, slice embeddings and run MIA scoring.
    """
    max_train = max(train_sizes)
    n_needed = max_train + eval_size

    print(f"\n=== [{dataset}] with [{gen_model}] ===")
    print(f"Need {n_needed} REAL & SYN samples")

    # 1) REAL texts
    real_texts = load_real_texts(dataset, n_needed)
    print(f"Loaded REAL texts: {len(real_texts)}")

    # 2) SYNTHETIC texts
    syn_texts = generate_synthetic_texts(
        dataset=dataset,
        model_name=gen_model,
        n_needed=n_needed,
        device=device,
    )
    print(f"Generated SYN texts: {len(syn_texts)}")

    # 3) Encode
    emb_real_all = encode_texts(real_texts, device=device, enc_name=enc_name)
    emb_syn_all  = encode_texts(syn_texts,  device=device, enc_name=enc_name)
    print(f"Emb REAL shape={emb_real_all.shape}, SYN shape={emb_syn_all.shape}")

    # 4) Scaler on union
    scaler = StandardScaler().fit(np.vstack([emb_real_all, emb_syn_all]))
    emb_real_all = scaler.transform(emb_real_all)
    emb_syn_all  = scaler.transform(emb_syn_all)

    # Pre-split EVAL subset (shared across train_sizes)
    emb_real_eval_base = emb_real_all[max_train:max_train + eval_size]
    emb_syn_eval_base  = emb_syn_all[max_train:max_train + eval_size]

    # For each train size and noise mode
    for train_size in train_sizes:
        emb_real_train = emb_real_all[:train_size]
        emb_syn_train  = emb_syn_all[:train_size]

        # Fit Gaussians on train embeddings
        dist_real = fit_gaussian(emb_real_train, cov_reg)
        dist_syn  = fit_gaussian(emb_syn_train,  cov_reg)

        for noise_label, noise_flag in [("clean", False), ("noise", True)]:
            # Apply noise only on EVAL embeddings (defense at inference time)
            if noise_flag:
                emb_real_eval = add_noise(emb_real_eval_base.copy(), noise_std)
                emb_syn_eval  = add_noise(emb_syn_eval_base.copy(),  noise_std)
            else:
                emb_real_eval = emb_real_eval_base
                emb_syn_eval  = emb_syn_eval_base

            # Scores (LLR): log p_real - log p_syn
            scores_real = loglik_gaussian(emb_real_eval, dist_real) - loglik_gaussian(emb_real_eval, dist_syn)
            scores_syn  = loglik_gaussian(emb_syn_eval,  dist_real) - loglik_gaussian(emb_syn_eval,  dist_syn)

            y_true  = np.array([1]*len(scores_real) + [0]*len(scores_syn))
            y_score = np.concatenate([scores_real, scores_syn])

            # ROC / AUC
            fpr, tpr, thr = roc_curve(y_true, y_score)
            auc_val = roc_auc_score(y_true, y_score)

            # TPR@1%FPR
            mask_1 = fpr <= 0.01
            tpr_at_1fpr = float(np.max(tpr[mask_1])) if np.any(mask_1) else 0.0

            # Accuracy using threshold 0 (LLR>0 => member)
            y_pred = (y_score > 0.0).astype(int)
            acc = float((y_pred == y_true).mean())

            model_short = "gpt2" if "gpt2" in gen_model else "falcon"
            fig_name = f"{dataset}_{model_short}_{train_size}_{noise_label}.png"
            fig_path = os.path.join(out_dir, fig_name)

            plt.figure(figsize=(5, 5))
            plt.plot(fpr, tpr, label=f"AUC={auc_val:.3f}")
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"{dataset} – {model_short}, N={train_size}, {noise_label}")
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(fig_path, dpi=150)
            plt.close()
            print(f"Saved ROC: {fig_path}")

            results.append({
                "dataset": dataset,
                "model": model_short,
                "train_size": train_size,
                "noise": noise_label,
                "accuracy": acc,
                "auc": auc_val,
                "tpr_at_1fpr": tpr_at_1fpr,
            })


# ---------------------------------------------------------
# Main: loop over 3 datasets × 2 models × 3 train_sizes × 2 noise
# ---------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--enc_name", type=str, default="distilbert-base-uncased")
    ap.add_argument("--cov_reg", type=float, default=1e-2)
    ap.add_argument("--noise_std", type=float, default=0.2,
                    help="std of Gaussian noise on eval embeddings for 'noise' mode")
    ap.add_argument("--eval_size", type=int, default=500,
                    help="number of positives / negatives for evaluation per config")
    ap.add_argument("--out_dir", type=str, default="results_figs",
                    help="directory to save the 36 ROC figures")
    args = ap.parse_args()

    print(vars(args))
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = args.device
    enc_name = args.enc_name
    cov_reg = args.cov_reg
    noise_std = args.noise_std
    eval_size = args.eval_size

    datasets = ["agnews", "wiki103", "xsum"]
    gen_models = ["gpt2", "tiiuae/falcon-7b-instruct"]
    train_sizes = [1000, 2000, 3000]

    results: List[Dict] = []

    for dataset in datasets:
        for gen_model in gen_models:
            run_for_dataset_model(
                dataset=dataset,
                gen_model=gen_model,
                device=device,
                enc_name=enc_name,
                train_sizes=train_sizes,
                eval_size=eval_size,
                cov_reg=cov_reg,
                noise_std=noise_std,
                results=results,
                out_dir=args.out_dir,
            )

    # -----------------------------------------------------
    # Print table + save CSV
    # -----------------------------------------------------
    print("\n=== Summary over 36 configs ===")
    header = ["dataset", "model", "train_size", "noise", "accuracy", "auc", "tpr_at_1fpr"]
    print("{:<8} {:<7} {:<10} {:<7} {:<10} {:<8} {:<12}".format(*header))

    lines = []
    for r in results:
        line = "{:<8} {:<7} {:<10} {:<7} {:<10.4f} {:<8.4f} {:<12.4f}".format(
            r["dataset"],
            r["model"],
            r["train_size"],
            r["noise"],
            r["accuracy"],
            r["auc"],
            r["tpr_at_1fpr"],
        )
        print(line)
        lines.append(line)

    # Save CSV file
    csv_path = "results_summary.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in results:
            f.write("{dataset},{model},{train_size},{noise},{accuracy:.6f},{auc:.6f},{tpr_at_1fpr:.6f}\n".format(**r))
    print(f"\nSaved summary table to: {csv_path}")
    print(f"ROC figures saved under: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()

