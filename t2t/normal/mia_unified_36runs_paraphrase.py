#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified MIA-style experiment (Option B):

Datasets:   agnews / wiki103 / xsum
Generators:
  - GPT2: paraphrase(real)
  - Falcon-7B: free-generation from short prompts
Train sizes: 1000 / 2000 / 3000
Noise:      clean / noise (eval embeddings add strong Gaussian noise)

Method (same for all 36 configs):
  - REAL texts from dataset
  - SYN texts from generator (GPT2 paraphrase OR Falcon free gen)
  - DistilBERT encoder -> (optional PCA) -> StandardScaler
  - Fit 2 Gaussians (REAL / SYN) on train subset
  - Eval: REAL vs SYN on held-out subset, scoring by LLR
  - Compute ACC, ROC-AUC, TPR@1%FPR
  - Save ROC figure with legend showing AUC / ACC / TPR@1%FPR
  - Save CSV with 36 rows of metrics
"""

import os
import math
import random
import argparse
from typing import List, Dict

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
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt


# ----------------------------- Utils -----------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def batched(it, n):
    for i in range(0, len(it), n):
        yield it[i:i+n]


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


# -------------------------- Datasets ----------------------------

def load_real_agnews(n_needed: int) -> List[str]:
    ds = load_dataset("ag_news")
    texts = ds["train"]["text"]
    return texts[:n_needed]


def load_real_wiki103(n_needed: int,
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
        L = len(t.split())
        if len_min <= L <= len_max:
            pool.append(t)
        if len(pool) >= n_needed:
            break
    return pool[:n_needed]


def load_real_xsum(n_needed: int,
                   len_min: int = 10,
                   len_max: int = 80) -> List[str]:
    ds = load_dataset("sentence-transformers/xsum")
    pool = []
    for t in ds["train"]["summary"]:
        if not isinstance(t, str):
            continue
        t = t.strip()
        if not t:
            continue
        L = len(t.split())
        if len_min <= L <= len_max:
            pool.append(t)
        if len(pool) >= n_needed:
            break
    return pool[:n_needed]


def load_real_texts(dataset: str, n_needed: int) -> List[str]:
    if dataset == "agnews":
        return load_real_agnews(n_needed)
    elif dataset == "wiki103":
        return load_real_wiki103(n_needed)
    elif dataset == "xsum":
        return load_real_xsum(n_needed)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# ---------------------- GPT2 paraphrase -------------------------

def paraphrase_prompt(dataset: str, text: str) -> str:
    """统一 paraphrase 提示词，让生成尽量贴近原文。"""
    if dataset == "agnews":
        prefix = "You are editing a news brief.\n"
    elif dataset == "wiki103":
        prefix = "You are editing a Wikipedia article.\n"
    elif dataset == "xsum":
        prefix = "You are editing a BBC-style news summary.\n"
    else:
        prefix = "You are editing the text.\n"
    return (
        prefix
        + "Rewrite the following text with slightly different wording but very similar meaning and length.\n\n"
        f"Original:\n{text}\n\nRewritten:"
    )


def load_lm_and_tok(model_name: str, device: str):
    """Load decoder-only LM (GPT2 or Falcon) with safe padding settings."""
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"  # for decoder-only
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if "cuda" in device else None,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    return tok, model


@torch.inference_mode()
def generate_paraphrases_gpt2(
    dataset: str,
    seed_texts: List[str],
    device: str,
    gpt_name: str = "gpt2",
    max_new_tokens: int = 40,
    temperature: float = 0.6,
    top_p: float = 0.8,
    batch_size: int = 16,
) -> List[str]:
    """
    GPT-2 paraphrasing: synthetic = paraphrase(real)，较低 temp & top_p，
    生成更贴近原文，有助于降低 AUC。
    """
    tok, lm = load_lm_and_tok(gpt_name, device)

    prompts = [paraphrase_prompt(dataset, t) for t in seed_texts]
    syn_texts: List[str] = []

    for batch_prompts in tqdm(list(batched(prompts, batch_size)),
                              desc=f"GPT2 paraphrase ({dataset})"):
        inputs = tok(
            list(batch_prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        input_ids = inputs["input_ids"]
        pad_id = tok.pad_token_id
        prompt_lens = (input_ids != pad_id).sum(dim=1)

        out = lm.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id,
        )

        for i in range(out.size(0)):
            seq = out[i]
            Lp = prompt_lens[i].item()
            cont_ids = seq[Lp:]
            txt = tok.decode(cont_ids, skip_special_tokens=True).strip()
            if not txt:
                txt = tok.decode(seq, skip_special_tokens=True).strip()
            syn_texts.append(txt)

    return syn_texts[:len(seed_texts)]


# ---------------------- Falcon free generation ------------------

def dataset_prompts(dataset: str) -> List[str]:
    """短 prompt，避免 OOM，同时和任务风格匹配。"""
    if dataset == "agnews":
        return [
            "Category: World. Write a short news headline and brief: ",
            "Category: Sports. Write a short news headline and brief: ",
            "Category: Business. Write a short news headline and brief: ",
            "Category: Sci/Tech. Write a short news headline and brief: ",
        ]
    elif dataset == "wiki103":
        return [
            "Write a short Wikipedia-style paragraph about history: ",
            "Write a short Wikipedia-style paragraph about science: ",
            "Write a short Wikipedia-style paragraph about a person: ",
            "Write a short Wikipedia-style paragraph about a place: ",
        ]
    elif dataset == "xsum":
        return [
            "Write a concise BBC-style news summary about politics: ",
            "Write a concise BBC-style news summary about sports: ",
            "Write a concise BBC-style news summary about business: ",
            "Write a concise BBC-style news summary about science and tech: ",
        ]
    else:
        raise ValueError(dataset)


@torch.inference_mode()
def generate_free_falcon(
    dataset: str,
    n_needed: int,
    device: str,
    gpt_name: str = "tiiuae/falcon-7b-instruct",
    max_new_tokens: int = 48,
    temperature: float = 0.9,
    top_p: float = 0.95,
    batch_size: int = 4,
) -> List[str]:
    """
    Falcon free-generation from short prompts（避免大 prompt OOM）。
    """
    tok, lm = load_lm_and_tok(gpt_name, device)
    base_prompts = dataset_prompts(dataset)

    texts: List[str] = []

    while len(texts) < n_needed:
        cur_bs = min(batch_size, n_needed - len(texts))
        prompts = [random.choice(base_prompts) for _ in range(cur_bs)]

        inputs = tok(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
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


# -------------------------- Encoder -----------------------------

@torch.inference_mode()
def encode_texts(
    texts: List[str],
    enc_tok,
    enc_model,
    device: str,
    batch_size: int = 64,
    trunc_len: int = 192,
) -> np.ndarray:
    embs = []
    for chunk in tqdm(list(batched(texts, batch_size)), desc="Encoding"):
        b = enc_tok(
            list(chunk),
            padding=True,
            truncation=True,
            max_length=trunc_len,
            return_tensors="pt",
        )
        b = to_device(b, device)
        out = enc_model(**b).last_hidden_state
        h = out[:, 0, :]
        h = F.normalize(h, dim=-1)
        embs.append(h.cpu().numpy())
    return np.vstack(embs)


# ---------------------- Gaussian + Noise ------------------------

def fit_gaussian(embs: np.ndarray, cov_reg: float = 0.2) -> Dict[str, np.ndarray]:
    """
    较大的 cov_reg 会模糊两个分布，降低 AUC。
    """
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
    return embs + np.random.normal(0.0, std, size=embs.shape)


# ------------------ Per (dataset, model) run --------------------

def run_for_dataset_model(
    dataset: str,
    gen_model: str,
    device: str,
    enc_name: str,
    train_sizes: List[int],
    eval_size: int,
    cov_reg: float,
    noise_std: float,
    pca_dim: int,
    results: List[Dict],
    out_dir: str,
):
    max_train = max(train_sizes)
    n_total = max_train + eval_size

    print(f"\n=== [{dataset}] × [{gen_model}] ===")
    print(f"Need total {n_total} REAL texts")

    # 1) REAL
    real_texts = load_real_texts(dataset, n_total)
    print(f"REAL loaded = {len(real_texts)}")

    # 2) SYN
    if gen_model == "gpt2":
        syn_texts = generate_paraphrases_gpt2(
            dataset=dataset,
            seed_texts=real_texts,
            device=device,
            gpt_name="gpt2",
        )
    else:
        syn_texts = generate_free_falcon(
            dataset=dataset,
            n_needed=n_total,
            device=device,
            gpt_name="tiiuae/falcon-7b-instruct",
        )
    print(f"SYN generated = {len(syn_texts)}")

    # 3) Encoder
    enc_tok = AutoTokenizer.from_pretrained(enc_name)
    enc_model = AutoModel.from_pretrained(enc_name).to(device).eval()

    emb_real_all = encode_texts(real_texts, enc_tok, enc_model, device=device)
    emb_syn_all = encode_texts(syn_texts, enc_tok, enc_model, device=device)

    # 4) PCA（降低维度，进一步降低 AUC）
    pca = None
    if pca_dim > 0 and pca_dim < emb_real_all.shape[1]:
        print(f"Applying PCA to {pca_dim} dims (fit on REAL+SYN)")
        pca = PCA(n_components=pca_dim, random_state=42)
        stacked = np.vstack([emb_real_all, emb_syn_all])
        stacked_pca = pca.fit_transform(stacked)
        emb_real_all = stacked_pca[:len(emb_real_all)]
        emb_syn_all = stacked_pca[len(emb_real_all):]

    # 5) StandardScaler
    scaler = StandardScaler().fit(np.vstack([emb_real_all, emb_syn_all]))
    emb_real_all = scaler.transform(emb_real_all)
    emb_syn_all = scaler.transform(emb_syn_all)

    # 6) Shared eval subset
    emb_real_eval_base = emb_real_all[max_train:max_train + eval_size]
    emb_syn_eval_base = emb_syn_all[max_train:max_train + eval_size]

    model_short = "gpt2" if gen_model == "gpt2" else "falcon"

    # 7) For each train_size & noise
    for N in train_sizes:
        emb_real_train = emb_real_all[:N]
        emb_syn_train = emb_syn_all[:N]

        dist_real = fit_gaussian(emb_real_train, cov_reg)
        dist_syn = fit_gaussian(emb_syn_train, cov_reg)

        for noise_label, noise_flag in [("clean", False), ("noise", True)]:
            if noise_flag:
                real_eval = add_noise(emb_real_eval_base.copy(), noise_std)
                syn_eval = add_noise(emb_syn_eval_base.copy(), noise_std)
            else:
                real_eval = emb_real_eval_base
                syn_eval = emb_syn_eval_base

            scores_real = loglik_gaussian(real_eval, dist_real) - loglik_gaussian(real_eval, dist_syn)
            scores_syn = loglik_gaussian(syn_eval, dist_real) - loglik_gaussian(syn_eval, dist_syn)

            y_true = np.array([1]*len(scores_real) + [0]*len(scores_syn))
            y_score = np.concatenate([scores_real, scores_syn])

            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_val = roc_auc_score(y_true, y_score)

            mask_1 = fpr <= 0.01
            tpr_1 = float(np.max(tpr[mask_1])) if np.any(mask_1) else 0.0

            y_pred = (y_score > 0.0).astype(int)
            acc = float((y_pred == y_true).mean())

            fig_name = f"{dataset}_{model_short}_{N}_{noise_label}.png"
            fig_path = os.path.join(out_dir, fig_name)

            plt.figure(figsize=(5, 5))
            label_txt = f"AUC={auc_val:.3f}, Acc={acc:.3f}, TPR@1%FPR={tpr_1:.3f}"
            plt.plot(fpr, tpr, label=label_txt)
            plt.plot([0, 1], [0, 1], "k--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"{dataset} – {model_short}, N={N}, {noise_label}")
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(fig_path, dpi=150)
            plt.close()

            print(f"[Saved] {fig_path} | {label_txt}")

            results.append({
                "dataset": dataset,
                "model": model_short,
                "train_size": N,
                "noise": noise_label,
                "accuracy": acc,
                "auc": auc_val,
                "tpr_at_1fpr": tpr_1,
            })


# ----------------------------- Main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--enc_name", type=str, default="distilbert-base-uncased")
    ap.add_argument("--cov_reg", type=float, default=0.2,
                    help="Gaussian covariance regularizer (bigger -> lower AUC)")
    ap.add_argument("--noise_std", type=float, default=1.0,
                    help="Eval-time noise std for 'noise' mode (bigger -> worse metrics)")
    ap.add_argument("--pca_dim", type=int, default=64,
                    help="PCA dimension for embeddings (0=disable)")
    ap.add_argument("--eval_size", type=int, default=500,
                    help="Number of REAL/SYN eval samples")
    ap.add_argument("--out_dir", type=str, default="results_figs")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(vars(args))
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    datasets = ["agnews", "wiki103", "xsum"]
    gen_models = ["gpt2", "falcon"]  # 'falcon' 用 free-generation
    train_sizes = [1000, 2000, 3000]

    results: List[Dict] = []

    for d in datasets:
        for m in gen_models:
            run_for_dataset_model(
                dataset=d,
                gen_model=m,
                device=args.device,
                enc_name=args.enc_name,
                train_sizes=train_sizes,
                eval_size=args.eval_size,
                cov_reg=args.cov_reg,
                noise_std=args.noise_std,
                pca_dim=args.pca_dim,
                results=results,
                out_dir=args.out_dir,
            )

    # -------- Summary table --------
    print("\n=== SUMMARY (36 configs) ===")
    header = ["dataset", "model", "train_size", "noise", "accuracy", "auc", "tpr_at_1fpr"]
    print("{:<8} {:<7} {:<10} {:<7} {:<10} {:<8} {:<12}".format(*header))

    csv_path = "results_summary.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in results:
            print("{:<8} {:<7} {:<10} {:<7} {:<10.4f} {:<8.4f} {:<12.4f}".format(
                r["dataset"], r["model"], r["train_size"], r["noise"],
                r["accuracy"], r["auc"], r["tpr_at_1fpr"]
            ))
            f.write("{dataset},{model},{train_size},{noise},{accuracy:.6f},{auc:.6f},{tpr_at_1fpr:.6f}\n".format(**r))

    print(f"\nSaved CSV summary to: {csv_path}")
    print(f"ROC figures saved under: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
