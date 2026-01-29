#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cosine-similarity-only MIA (computation-based, cross-model generation)

- For each sample x:
  1) Build D_aux(x) using simple text perturbations (non-members proxy).
  2) Query target model T on x for n times -> generated texts -> encode via E -> Hx
  3) Query target model T on each x_i in D_aux for t times -> encode -> Haux
  4) Score(x) = 1 - cos(mean(Hx), mean(Haux))

- Membership decision with fixed thresholds tau in {-1, 0, 1}:
    pred = 1 if score > tau else 0

- Evaluate:
  ROC-AUC (threshold-free) + TPR@1%FPR
  Accuracy under each tau

Datasets:
  - agnews: members from train, nonmembers from test
  - wiki103: members from train, nonmembers from validation (wikitext-103-raw-v1)
  - xsum: uses sentence-transformers/xsum (only train split) -> split train into halves

Outputs:
  - CSV: cosine_mia_results.csv
  - ROC figures per run: roc_<dataset>__T-<target>__G-<generator>.png
"""

import os
import re
import csv
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
import matplotlib.pyplot as plt


# -------------------------
# Text utilities
# -------------------------
def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s

def word_count(s: str) -> int:
    return len((s or "").split())

def truncate_words(s: str, max_words: int) -> str:
    toks = (s or "").split()
    if len(toks) <= max_words:
        return s
    return " ".join(toks[:max_words])

def noise_text(x: str, p_drop: float = 0.08, p_swap: float = 0.06) -> str:
    toks = (x or "").split()
    if len(toks) < 6:
        return x
    # drop
    kept = []
    for w in toks:
        if random.random() > p_drop:
            kept.append(w)
    if len(kept) < 4:
        kept = toks[:]  # fallback
    # swap
    i = 0
    while i + 1 < len(kept):
        if random.random() < p_swap:
            kept[i], kept[i + 1] = kept[i + 1], kept[i]
            i += 2
        else:
            i += 1
    return " ".join(kept)

def build_D_aux(x: str, m_aux: int) -> List[str]:
    # purely perturbation-based D_aux (fast & stable)
    aux = []
    for _ in range(m_aux):
        aux.append(noise_text(x))
    return aux


# -------------------------
# Embedding / cosine score
# -------------------------
@torch.inference_mode()
def encode_texts(texts: List[str], enc_tok, enc_model, device: torch.device, trunc_len: int, bs: int) -> np.ndarray:
    embs = []
    for i in range(0, len(texts), bs):
        chunk = texts[i:i+bs]
        x = enc_tok(
            chunk,
            padding=True,
            truncation=True,
            max_length=trunc_len,
            return_tensors="pt",
        ).to(device)
        out = enc_model(**x).last_hidden_state[:, 0, :]  # [B, H]
        out = F.normalize(out, dim=-1)
        embs.append(out.detach().cpu().numpy())
    return np.vstack(embs) if embs else np.zeros((0, enc_model.config.hidden_size), dtype=np.float32)

def cosine_score(Hx: np.ndarray, Haux: np.ndarray) -> float:
    """
    score(x) = 1 - cos(mean(Hx), mean(Haux))
    Larger score => more different => (typically) more likely MEMBER under this design.
    """
    if Hx.size == 0 or Haux.size == 0:
        return 0.0
    mx = Hx.mean(axis=0, keepdims=True)
    ma = Haux.mean(axis=0, keepdims=True)
    mx = mx / (np.linalg.norm(mx) + 1e-12)
    ma = ma / (np.linalg.norm(ma) + 1e-12)
    cos = float(mx @ ma.T)
    return 1.0 - cos


# -------------------------
# LLM generation
# -------------------------
def load_causal_lm(model_name: str, device: torch.device, use_4bit: bool) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    kwargs = {}
    # optional 4bit quantization (if bitsandbytes installed)
    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            kwargs["quantization_config"] = bnb_cfg
            kwargs["device_map"] = "auto"
        except Exception:
            # silently fallback
            pass

    # if not using device_map, use half precision when on cuda
    if "device_map" not in kwargs and device.type == "cuda":
        kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if "device_map" not in kwargs:
        model = model.to(device)
    model.eval()
    return tok, model

@torch.inference_mode()
def sample_T_generations(
    prompt: str,
    n: int,
    gen_tok,
    gen_model,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> List[str]:
    prompt = clean_text(prompt)
    if not prompt:
        prompt = " "

    inp = gen_tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    L = inp["input_ids"].shape[1]
    outs = []
    for _ in range(n):
        out = gen_model.generate(
            **inp,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=gen_tok.pad_token_id,
            eos_token_id=gen_tok.eos_token_id,
            use_cache=True,
        )
        txt = gen_tok.decode(out[0][L:], skip_special_tokens=True)
        txt = clean_text(txt)
        if txt:
            outs.append(txt)
    return outs


# -------------------------
# Metrics under fixed threshold
# -------------------------
def confusion_counts(y_true: List[int], y_pred: List[int]) -> Tuple[int, int, int, int]:
    tp = sum((p == 1 and y == 1) for p, y in zip(y_pred, y_true))
    fp = sum((p == 1 and y == 0) for p, y in zip(y_pred, y_true))
    tn = sum((p == 0 and y == 0) for p, y in zip(y_pred, y_true))
    fn = sum((p == 0 and y == 1) for p, y in zip(y_pred, y_true))
    return tp, fp, tn, fn

def accuracy_from_counts(tp: int, fp: int, tn: int, fn: int) -> float:
    denom = max(tp + fp + tn + fn, 1)
    return (tp + tn) / denom


# -------------------------
# Dataset loaders
# -------------------------
def load_agnews(n_pos: int, n_neg: int, seed: int, len_min: int, len_max: int) -> Tuple[List[str], List[str]]:
    ds = load_dataset("ag_news")
    train = [clean_text(x) for x in ds["train"]["text"]]
    test  = [clean_text(x) for x in ds["test"]["text"]]
    train = [x for x in train if len_min <= word_count(x) <= len_max]
    test  = [x for x in test if len_min <= word_count(x) <= len_max]
    random.Random(seed).shuffle(train)
    random.Random(seed + 1).shuffle(test)
    return train[:n_pos], test[:n_neg]

def load_wiki103(n_pos: int, n_neg: int, seed: int, len_min: int, len_max: int) -> Tuple[List[str], List[str]]:
    # wikitext-103-raw-v1
    ds = load_dataset("wikitext", "wikitext-103-raw-v1")
    train = [clean_text(x) for x in ds["train"]["text"]]
    valid = [clean_text(x) for x in ds["validation"]["text"]]
    train = [x for x in train if len_min <= word_count(x) <= len_max]
    valid = [x for x in valid if len_min <= word_count(x) <= len_max]
    random.Random(seed).shuffle(train)
    random.Random(seed + 1).shuffle(valid)
    return train[:n_pos], valid[:n_neg]

def load_xsum(n_pos: int, n_neg: int, seed: int, len_min: int, len_max: int) -> Tuple[List[str], List[str]]:
    # mirror dataset you already verified works: sentence-transformers/xsum (train only)
    ds = load_dataset("sentence-transformers/xsum")
    arts = [clean_text(x) for x in ds["train"]["article"]]
    arts = [x for x in arts if len_min <= word_count(x) <= len_max]
    random.Random(seed).shuffle(arts)
    # split into two disjoint pools
    mid = len(arts) // 2
    pos_pool = arts[:mid]
    neg_pool = arts[mid:]
    return pos_pool[:n_pos], neg_pool[:n_neg]


# -------------------------
# One run
# -------------------------
def run_one(
    dataset_name: str,
    target_name: str,
    generator_name: str,
    args,
    device: torch.device,
) -> Dict[str, object]:
    # 1) load dataset members/nonmembers
    if dataset_name == "agnews":
        pos, neg = load_agnews(args.n_pos, args.n_neg, args.seed, args.len_min, args.len_max)
    elif dataset_name == "wiki103":
        pos, neg = load_wiki103(args.n_pos, args.n_neg, args.seed, args.len_min, args.len_max)
    elif dataset_name == "xsum":
        pos, neg = load_xsum(args.n_pos, args.n_neg, args.seed, args.len_min, args.len_max)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # 2) load target model T (used for queries)
    tgt_tok, tgt_model = load_causal_lm(target_name, device=device, use_4bit=args.use_4bit)

    # 3) load encoder E (used for embeddings)
    enc_tok = AutoTokenizer.from_pretrained(args.enc_name, use_fast=True)
    enc_model = AutoModel.from_pretrained(args.enc_name).to(device)
    enc_model.eval()

    # score function: computation-based cosine score
    def process_one(x: str) -> float:
        x = truncate_words(x, args.prompt_max_words)

        # D_aux
        Daux = build_D_aux(x, args.m_aux)

        # Hx: query T on x, encode T outputs
        gen_x = sample_T_generations(
            prompt=x,
            n=args.n_repeat_x,
            gen_tok=tgt_tok,
            gen_model=tgt_model,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        Hx = encode_texts(gen_x, enc_tok, enc_model, device, args.enc_trunc_len, args.enc_bs)

        # Haux: query T on each x_i in D_aux, encode
        aux_gens = []
        for xi in Daux:
            xi = truncate_words(xi, args.prompt_max_words)
            gi = sample_T_generations(
                prompt=xi,
                n=args.t_repeat_aux,
                gen_tok=tgt_tok,
                gen_model=tgt_model,
                device=device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            aux_gens.extend(gi)
        Haux = encode_texts(aux_gens, enc_tok, enc_model, device, args.enc_trunc_len, args.enc_bs)

        return cosine_score(Hx, Haux)

    # 4) compute scores
    y_true = [1] * len(pos) + [0] * len(neg)
    y_score = []

    print(f"\n[{dataset_name}] Scoring members (target={target_name}) ...")
    for x in tqdm(pos, desc="members"):
        y_score.append(process_one(x))

    print(f"[{dataset_name}] Scoring non-members (target={target_name}) ...")
    for x in tqdm(neg, desc="nonmembers"):
        y_score.append(process_one(x))

    # 5) roc/auc
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    # If inverted (rare), flip sign to keep AUC >= 0.5
    if auc < 0.5:
        y_score = [-s for s in y_score]
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)

    tpr_at_1fpr = float(np.max(tpr[np.array(fpr) <= 0.01])) if np.any(np.array(fpr) <= 0.01) else 0.0

    # 6) save ROC plot
    fig_name = f"roc_{dataset_name}__T-{safe_name(target_name)}__G-{safe_name(generator_name)}.png"
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.scatter([0.01], [tpr_at_1fpr], label=f"TPR@1%FPR={tpr_at_1fpr:.3f}")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"{dataset_name} | target={short_name(target_name)} | cosine-MIA")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(fig_name, dpi=150)
    plt.close()

    return {
        "dataset": dataset_name,
        "target": target_name,
        "generator": generator_name,
        "y_true": y_true,
        "y_score": y_score,
        "auc": float(auc),
        "tpr_at_1fpr": float(tpr_at_1fpr),
        "roc_fig": fig_name,
    }

def safe_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s)

def short_name(s: str) -> str:
    # just the tail for display
    return s.split("/")[-1]


# -------------------------
# Main: 6 runs, 3 thresholds each
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # sample sizes
    ap.add_argument("--n_pos", type=int, default=2000)
    ap.add_argument("--n_neg", type=int, default=2000)

    # D_aux and query budgets
    ap.add_argument("--m_aux", type=int, default=6)
    ap.add_argument("--n_repeat_x", type=int, default=6)
    ap.add_argument("--t_repeat_aux", type=int, default=3)

    # generation controls
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=0.95)

    # prompts/text lengths
    ap.add_argument("--prompt_max_words", type=int, default=80)
    ap.add_argument("--len_min", type=int, default=30)
    ap.add_argument("--len_max", type=int, default=160)

    # encoder
    ap.add_argument("--enc_name", type=str, default="distilbert-base-uncased")
    ap.add_argument("--enc_trunc_len", type=int, default=192)
    ap.add_argument("--enc_bs", type=int, default=64)

    # models (cross)
    ap.add_argument("--gpt2", type=str, default="gpt2")
    ap.add_argument("--falcon", type=str, default="tiiuae/falcon-rw-1b")  # safer default than 7B
    ap.add_argument("--use_4bit", action="store_true", help="try 4-bit quantization for large models (needs bitsandbytes)")

    # datasets list
    ap.add_argument("--datasets", type=str, default="agnews,wiki103,xsum")

    args = ap.parse_args()
    print(vars(args))

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    thresholds = [-1.0, 0.0, 1.0]

    # define 2 cross-model settings:
    # 1) target=gpt2, generator=falcon
    # 2) target=falcon, generator=gpt2
    settings = [
        ("gpt2", args.gpt2, args.falcon),
        ("falcon", args.falcon, args.gpt2),
    ]

    out_rows = []
    roc_figs = []

    for ds_name in datasets:
        for tag, target_model, generator_model in settings:
            # NOTE: In this cosine-MIA script, "generator_model" is only for bookkeeping
            # because queries are done on target_model. We keep it in outputs to match your experiment description.
            res = run_one(ds_name, target_model, generator_model, args, device)
            roc_figs.append(res["roc_fig"])

            y_true = res["y_true"]
            y_score = res["y_score"]
            auc = res["auc"]
            tpr_at_1fpr = res["tpr_at_1fpr"]

            # fixed-threshold accuracies
            for tau in thresholds:
                y_pred = [1 if s > tau else 0 for s in y_score]
                tp, fp, tn, fn = confusion_counts(y_true, y_pred)
                acc = accuracy_from_counts(tp, fp, tn, fn)

                # formatting as you requested:
                # accuracy: 3 decimals, auc: 6 decimals, tpr_at_1fpr: 3 decimals
                out_rows.append({
                    "dataset": ds_name,
                    "target": short_name(res["target"]),
                    "generator": short_name(res["generator"]),
                    "threshold": f"{tau:.0f}" if tau in (-1.0, 0.0, 1.0) else f"{tau:.3f}",
                    "accuracy": f"{acc:.3f}",
                    "auc": f"{auc:.6f}",
                    "tpr_at_1fpr": f"{tpr_at_1fpr:.3f}",
                    "roc_fig": res["roc_fig"],
                })

    # save CSV
    csv_name = "cosine_mia_results.csv"
    with open(csv_name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["dataset", "target", "generator", "threshold", "accuracy", "auc", "tpr_at_1fpr", "roc_fig"]
        )
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    print("\nSaved CSV:", csv_name)
    print("Saved ROC figures:")
    for p in roc_figs:
        print("  -", p)

    # also print a compact view (same style as your screenshot for the 3 metric columns)
    print("\n(Preview) accuracy / auc / tpr_at_1fpr:")
    for r in out_rows[:6]:
        print(r["accuracy"], r["auc"], r["tpr_at_1fpr"])


if __name__ == "__main__":
    main()

