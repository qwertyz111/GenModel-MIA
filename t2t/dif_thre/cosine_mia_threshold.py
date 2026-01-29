#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cosine-similarity-only MIA (computation-based, cross-model generation)

Changes vs your original:
- NO plots
- thresholds: from -1 to 1, step 0.06
- save all threshold accuracies to CSV

Keeps the original logic:
- D_aux(x) from perturbations
- Query target model T on x for n_repeat_x times -> generated texts -> encode -> Hx
- Query target model T on each xi in D_aux for t_repeat_aux times -> encode -> Haux
- score(x) = 1 - cos(mean(Hx), mean(Haux))
- Evaluate ROC-AUC, TPR@1%FPR (threshold-free), and accuracy for each threshold
"""

import re
import csv
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


# -------------------------
# Text utilities
# -------------------------
def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

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
    kept = []
    for w in toks:
        if random.random() > p_drop:
            kept.append(w)
    if len(kept) < 4:
        kept = toks[:]  # fallback
    i = 0
    while i + 1 < len(kept):
        if random.random() < p_swap:
            kept[i], kept[i + 1] = kept[i + 1], kept[i]
            i += 2
        else:
            i += 1
    return " ".join(kept)

def build_D_aux(x: str, m_aux: int) -> List[str]:
    return [noise_text(x) for _ in range(m_aux)]

def safe_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s)

def short_name(s: str) -> str:
    return s.split("/")[-1]


# -------------------------
# Embedding / cosine score
# -------------------------
@torch.inference_mode()
def encode_texts(
    texts: List[str],
    enc_tok,
    enc_model,
    device: torch.device,
    trunc_len: int,
    bs: int
) -> np.ndarray:
    if not texts:
        return np.zeros((0, enc_model.config.hidden_size), dtype=np.float32)

    embs = []
    for i in range(0, len(texts), bs):
        chunk = texts[i:i + bs]
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
    return np.vstack(embs)

def cosine_score(Hx: np.ndarray, Haux: np.ndarray) -> float:
    """
    score(x) = 1 - cos(mean(Hx), mean(Haux))
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
# LLM load & generation
# -------------------------
def load_causal_lm(model_name: str, device: torch.device) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    kwargs = {}
    if device.type == "cuda":
        kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs).to(device)
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
# Dataset loaders
# -------------------------
def load_agnews(n_pos: int, n_neg: int, seed: int, len_min: int, len_max: int) -> Tuple[List[str], List[str]]:
    ds = load_dataset("ag_news")
    train = [clean_text(x) for x in ds["train"]["text"]]
    test  = [clean_text(x) for x in ds["test"]["text"]]
    train = [x for x in train if len_min <= word_count(x) <= len_max]
    test  = [x for x in test  if len_min <= word_count(x) <= len_max]
    random.Random(seed).shuffle(train)
    random.Random(seed + 1).shuffle(test)
    return train[:n_pos], test[:n_neg]

def load_wiki103(n_pos: int, n_neg: int, seed: int, len_min: int, len_max: int) -> Tuple[List[str], List[str]]:
    ds = load_dataset("wikitext", "wikitext-103-raw-v1")
    train = [clean_text(x) for x in ds["train"]["text"]]
    valid = [clean_text(x) for x in ds["validation"]["text"]]
    train = [x for x in train if len_min <= word_count(x) <= len_max]
    valid = [x for x in valid if len_min <= word_count(x) <= len_max]
    random.Random(seed).shuffle(train)
    random.Random(seed + 1).shuffle(valid)
    return train[:n_pos], valid[:n_neg]

def load_xsum(n_pos: int, n_neg: int, seed: int, len_min: int, len_max: int) -> Tuple[List[str], List[str]]:
    ds = load_dataset("sentence-transformers/xsum")
    arts = [clean_text(x) for x in ds["train"]["article"]]
    arts = [x for x in arts if len_min <= word_count(x) <= len_max]
    random.Random(seed).shuffle(arts)
    mid = len(arts) // 2
    pos_pool = arts[:mid]
    neg_pool = arts[mid:]
    return pos_pool[:n_pos], neg_pool[:n_neg]


# -------------------------
# One run (one dataset + one target model)
# -------------------------
def run_one(dataset_name: str, target_name: str, generator_name: str, args, device: torch.device) -> Dict[str, object]:
    # load dataset
    if dataset_name == "agnews":
        pos, neg = load_agnews(args.n_pos, args.n_neg, args.seed, args.len_min, args.len_max)
    elif dataset_name == "wiki103":
        pos, neg = load_wiki103(args.n_pos, args.n_neg, args.seed, args.len_min, args.len_max)
    elif dataset_name == "xsum":
        pos, neg = load_xsum(args.n_pos, args.n_neg, args.seed, args.len_min, args.len_max)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # load target model (used for querying)
    tgt_tok, tgt_model = load_causal_lm(target_name, device=device)

    # load encoder
    enc_tok = AutoTokenizer.from_pretrained(args.enc_name, use_fast=True)
    enc_model = AutoModel.from_pretrained(args.enc_name).to(device).eval()

    def process_one(x: str) -> float:
        x = truncate_words(x, args.prompt_max_words)
        Daux = build_D_aux(x, args.m_aux)

        # Query T on x -> generated texts
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

        # Query T on each xi in D_aux -> generated texts
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

    y_true = [1] * len(pos) + [0] * len(neg)
    y_score = []

    print(f"\n[{dataset_name}] target={target_name} scoring members ...")
    for x in tqdm(pos, desc="members"):
        y_score.append(process_one(x))

    print(f"[{dataset_name}] target={target_name} scoring nonmembers ...")
    for x in tqdm(neg, desc="nonmembers"):
        y_score.append(process_one(x))

    # ROC/AUC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = float(roc_auc_score(y_true, y_score))

    # Flip if inverted
    if auc < 0.5:
        y_score = [-s for s in y_score]
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = float(roc_auc_score(y_true, y_score))

    tpr_at_1fpr = float(np.max(tpr[np.array(fpr) <= 0.01])) if np.any(np.array(fpr) <= 0.01) else 0.0

    # cleanup
    del tgt_tok, tgt_model, enc_tok, enc_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "dataset": dataset_name,
        "target": target_name,
        "generator": generator_name,  # bookkeeping only, as your original script
        "y_true": y_true,
        "y_score": y_score,
        "auc": auc,
        "tpr_at_1fpr": tpr_at_1fpr,
    }


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--n_pos", type=int, default=2000)
    ap.add_argument("--n_neg", type=int, default=2000)

    ap.add_argument("--m_aux", type=int, default=6)
    ap.add_argument("--n_repeat_x", type=int, default=6)
    ap.add_argument("--t_repeat_aux", type=int, default=3)

    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=0.95)

    ap.add_argument("--prompt_max_words", type=int, default=80)
    ap.add_argument("--len_min", type=int, default=30)
    ap.add_argument("--len_max", type=int, default=160)

    ap.add_argument("--enc_name", type=str, default="distilbert-base-uncased")
    ap.add_argument("--enc_trunc_len", type=int, default=192)
    ap.add_argument("--enc_bs", type=int, default=64)

    ap.add_argument("--gpt2", type=str, default="gpt2")
    ap.add_argument("--falcon", type=str, default="tiiuae/falcon-rw-1b")

    ap.add_argument("--datasets", type=str, default="agnews,wiki103,xsum")
    ap.add_argument("--csv_out", type=str, default="cosine_mia_threshold_sweep.csv")
    args = ap.parse_args()

    print(vars(args), flush=True)

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    # thresholds: -1 to 1, step 0.06 (include 1.00)
    thresholds = np.round(np.arange(-1.0, 1.0001, 0.06), 2)

    # cross-model settings (same structure as your original)
    settings = [
        ("gpt2", args.gpt2, args.falcon),      # tag, target, generator(bookkeeping)
        ("falcon", args.falcon, args.gpt2),
    ]

    out_rows: List[Dict[str, str]] = []

    for ds_name in datasets:
        for tag, target_model, generator_model in settings:
            res = run_one(ds_name, target_model, generator_model, args, device)

            y_true = res["y_true"]
            y_score = res["y_score"]
            auc = res["auc"]
            tpr_at_1fpr = res["tpr_at_1fpr"]

            y_true_np = np.asarray(y_true, dtype=np.int32)
            y_score_np = np.asarray(y_score, dtype=np.float32)

            # accuracy per threshold
            for tau in thresholds:
                y_pred = (y_score_np > tau).astype(np.int32)
                acc = float((y_pred == y_true_np).mean())

                out_rows.append({
                    "dataset": ds_name,
                    "target": short_name(res["target"]),
                    "generator": short_name(res["generator"]),
                    "threshold": f"{tau:.2f}",
                    "accuracy": f"{acc:.6f}",
                    "auc": f"{auc:.6f}",
                    "tpr_at_1fpr": f"{tpr_at_1fpr:.6f}",
                })

    # save CSV
    with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["dataset", "target", "generator", "threshold", "accuracy", "auc", "tpr_at_1fpr"]
        )
        w.writeheader()
        w.writerows(out_rows)

    print("\nSaved CSV:", args.csv_out, flush=True)


if __name__ == "__main__":
    main()
