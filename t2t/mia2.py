#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Computation-based MIA (text) per your slide:
  Step 1: Build D_aux(x) from x via paraphrasing/perturbation (non-members).
  Step 2: Query target model T n times on x → Hx; encode with E.
  Step 3: For each x_i in D_aux, query T t times → Haux; encode with E.
  Step 4: Compare distributions P_T(x) vs P_T(D_aux); score = E_h[log p_x(h) - log p_aux(h)].
Evaluate ROC-AUC / TPR@1%FPR using AG_NEWS (train as members, test as non-members).
"""

import argparse, math, random, numpy as np
from typing import List, Tuple, Dict
import torch, torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM, set_seed
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# -------------------- utils --------------------
def to_device(batch, device): return {k: v.to(device) for k, v in batch.items()}
def batched(it, n): 
    for i in range(0, len(it), n): 
        yield it[i:i+n]

def fit_gaussian(embs: np.ndarray, cov_reg: float = 1e-2) -> Dict[str, np.ndarray]:
    mu = embs.mean(axis=0)
    cov = np.cov(embs.T) + cov_reg * np.eye(embs.shape[1])
    inv = np.linalg.pinv(cov)
    logdet = float(np.log(np.linalg.det(cov) + 1e-12))
    return {"mu": mu, "inv": inv, "logdet": logdet, "dim": embs.shape[1]}

def loglik(x: np.ndarray, g: Dict[str, np.ndarray]) -> np.ndarray:
    diff = (x - g["mu"])
    maha = (diff @ g["inv"] * diff).sum(axis=-1)
    return -0.5 * (maha + g["logdet"] + g["dim"] * math.log(2 * math.pi))

# -------------------- Step 1: build D_aux(x) --------------------
def noise_text(x: str, p_drop: float = 0.08, p_swap: float = 0.05) -> str:
    toks = x.split()
    toks = [w for w in toks if np.random.rand() > p_drop or len(toks) < 5]
    i = 0
    while i + 1 < len(toks):
        if np.random.rand() < p_swap:
            toks[i], toks[i+1] = toks[i+1], toks[i]
            i += 2
        else:
            i += 1
    return " ".join(toks) if toks else x

@torch.inference_mode()
def build_D_aux(x: str, m: int, device: str, paraphraser: str = "Vamsi/T5_Paraphrase_Paws") -> List[str]:
    """Paraphrase x into m variants (fallback to simple noise)."""
    aux = []
    try:
        p_tok = AutoTokenizer.from_pretrained(paraphraser)
        p_mod = AutoModelForSeq2SeqLM.from_pretrained(paraphraser).to(device).eval()
        prompts = [f"paraphrase: {x} </s>"] * m
        enc = p_tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        out = p_mod.generate(**enc, do_sample=True, top_p=0.95, temperature=0.9,
                             max_new_tokens=64, num_return_sequences=1, use_cache=True)
        cand = p_tok.batch_decode(out, skip_special_tokens=True)
        seen = set()
        for t in cand:
            t = t.strip()
            if t and t.lower() != x.lower() and t not in seen:
                aux.append(t); seen.add(t)
        # top-up with noise if needed
        while len(aux) < m:
            aux.append(noise_text(x))
    except Exception as e:
        print(f"[D_aux] paraphraser failed: {e} -> using noisy variants.")
        aux = [noise_text(x) for _ in range(m)]
    return aux[:m]

# -------------------- Step 2/3: repeatedly query T and encode with E --------------------
@torch.inference_mode()
def sample_T_and_encode(q: str, n: int, gen_tok, gen_model, enc_tok, enc_model,
                        device: str, max_new_tokens: int, temperature: float, top_p: float, enc_trunc_len: int) -> np.ndarray:
    embs = []
    for _ in range(n):
        inp = gen_tok(q if q.strip() else " ", return_tensors="pt", add_special_tokens=False).to(device)
        out = gen_model.generate(**inp, do_sample=True, temperature=temperature, top_p=top_p,
                                 max_new_tokens=max_new_tokens, pad_token_id=gen_tok.eos_token_id, use_cache=True)
        # strip prompt
        L = inp["input_ids"].shape[1]
        gen_txt = gen_tok.decode(out[0][L:], skip_special_tokens=True).strip()
        if not gen_txt:
            continue
        x = enc_tok(gen_txt, return_tensors="pt", padding=True, truncation=True, max_length=enc_trunc_len).to(device)
        h = enc_model(**x).last_hidden_state[:, 0, :]
        h = F.normalize(h, dim=-1).cpu().numpy()
        embs.append(h[0])
    if not embs:
        H = np.zeros((1, enc_model.config.hidden_size), dtype=np.float32)
    else:
        H = np.vstack(embs)
    return H

def comp_based_score(Hx: np.ndarray, Haux: np.ndarray, cov_reg: float) -> float:
    Gx, Gaux = fit_gaussian(Hx, cov_reg), fit_gaussian(Haux, cov_reg)
    return float(loglik(Hx, Gx).mean() - loglik(Hx, Gaux).mean())

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # dataset (AG_NEWS: train→members, test→non-members)
    ap.add_argument("--n_pos", type=int, default=400, help="number of member eval samples (from train)")
    ap.add_argument("--n_neg", type=int, default=400, help="number of non-member eval samples (from test)")
    # computation-based params
    ap.add_argument("--m_aux", type=int, default=8, help="size of D_aux per x")
    ap.add_argument("--n_repeat_x", type=int, default=8, help="n times to query T on x")
    ap.add_argument("--t_repeat_aux", type=int, default=4, help="t times to query T on each x_i in D_aux")
    # models
    ap.add_argument("--gen_name", type=str, default="gpt2", help="target model T (causal LM)")
    ap.add_argument("--enc_name", type=str, default="distilbert-base-uncased", help="feature extractor E")
    # generation / encoding
    ap.add_argument("--max_new_tokens", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--enc_trunc_len", type=int, default=192)
    # gaussian
    ap.add_argument("--cov_reg", type=float, default=1e-2)
    args = ap.parse_args()
    print(vars(args))

    set_seed(args.seed)
    device = args.device
    torch.set_grad_enabled(False)

    # Load dataset (AG_NEWS)
    ds = load_dataset("ag_news")
    train_texts, train_labels = ds["train"]["text"], ds["train"]["label"]
    test_texts,  test_labels  = ds["test"]["text"],  ds["test"]["label"]
    random.seed(args.seed)
    pos_samples = random.sample(list(train_texts), min(args.n_pos, len(train_texts)))
    neg_samples = random.sample(list(test_texts),  min(args.n_neg, len(test_texts)))

    # Prepare T and E
    gen_tok = AutoTokenizer.from_pretrained(args.gen_name)
    if gen_tok.pad_token_id is None:
        gen_tok.pad_token_id = gen_tok.eos_token_id
    gen_model = AutoModelForCausalLM.from_pretrained(args.gen_name).to(device).eval()

    enc_tok = AutoTokenizer.from_pretrained(args.enc_name)
    enc_model = AutoModel.from_pretrained(args.enc_name).to(device).eval()

    # Evaluate
    y_true, y_score = [], []

    def process_one(x: str) -> float:
        # Step1: D_aux
        D_aux = build_D_aux(x, m=args.m_aux, device=device)
        # Step2: Hx
        Hx = sample_T_and_encode(x, args.n_repeat_x, gen_tok, gen_model, enc_tok, enc_model,
                                 device, args.max_new_tokens, args.temperature, args.top_p, args.enc_trunc_len)
        # Step3: Haux
        Haux_list = []
        for xi in D_aux:
            Hi = sample_T_and_encode(xi, args.t_repeat_aux, gen_tok, gen_model, enc_tok, enc_model,
                                     device, args.max_new_tokens, args.temperature, args.top_p, args.enc_trunc_len)
            Haux_list.append(Hi)
        Haux = np.vstack(Haux_list)
        # Step4: score
        return comp_based_score(Hx, Haux, cov_reg=args.cov_reg)

    print("\nScoring members (train) ...")
    for x in tqdm(pos_samples):
        y_true.append(1)
        y_score.append(process_one(x))

    print("Scoring non-members (test) ...")
    for x in tqdm(neg_samples):
        y_true.append(0)
        y_score.append(process_one(x))

    # ROC / AUC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    # flip if inverted
    if auc < 0.5:
        y_score = [-s for s in y_score]
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
    tpr_at_1fpr = max(tpr[np.array(fpr) <= 0.01]) if np.any(np.array(fpr) <= 0.01) else 0.0

    print(f"\nROC-AUC={auc:.4f}  TPR@1%FPR={tpr_at_1fpr:.4f}")

    plt.figure(figsize=(5,5))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0,1],[0,1],'k--')
    plt.scatter([0.01],[tpr_at_1fpr], label=f'TPR@1%FPR={tpr_at_1fpr:.3f}')
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("Noise-gpt2-agnews")
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig("roc_curve_comp_based.png", dpi=150)
    print("Saved: roc_curve_comp_based.png")


if __name__ == "__main__":
    main()

