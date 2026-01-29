#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved Computation-based MIA on AG_NEWS

Score = w1 * (LLR on embeddings) 
      + w2 * (mean cosine stability gap) 
      - w3 * (NLL variance gap)

Expected: AUC typically 0.7–0.85 with defaults (depends on compute).
"""

import argparse, math, random, numpy as np
from typing import List, Tuple, Dict
import torch, torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM, set_seed
)
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# -------------------- small utils --------------------
def to_device(batch, device): return {k: v.to(device) for k, v in batch.items()}
def batched(it, n):
    for i in range(0, len(it), n):
        yield it[i:i+n]

def pairwise_cos_mean(embs: np.ndarray) -> float:
    if len(embs) < 2: return 0.0
    X = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
    M = X @ X.T
    iu = np.triu_indices_from(M, 1)
    return float(M[iu].mean())

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

# -------------------- D_aux(x) --------------------
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
        while len(aux) < m:
            aux.append(noise_text(x))
    except Exception as e:
        print(f"[D_aux] paraphraser failed: {e} -> using noisy variants.")
        aux = [noise_text(x) for _ in range(m)]
    return aux[:m]

# -------------------- Target T: generate repeats --------------------
@torch.inference_mode()
def generate_texts(q: str, n: int, gen_tok, gen_model, device: str,
                   max_new_tokens: int, temperature: float, top_p: float,
                   temp_jitter: float = 0.05) -> List[str]:
    outs = []
    for _ in range(n):
        t_cur = max(0.1, random.uniform(temperature - temp_jitter, temperature + temp_jitter))
        inp = gen_tok(q if q.strip() else " ", return_tensors="pt", add_special_tokens=False).to(device)
        L = inp["input_ids"].shape[1]
        out = gen_model.generate(
            **inp, do_sample=True, temperature=t_cur, top_p=top_p,
            max_new_tokens=max_new_tokens, pad_token_id=gen_tok.eos_token_id, use_cache=True
        )
        txt = gen_tok.decode(out[0][L:], skip_special_tokens=True).strip()
        if txt: outs.append(txt)
    return outs if outs else [""]

# -------------------- Encoder E --------------------
@torch.inference_mode()
def encode_texts(texts: List[str], enc_tok, enc_model, device: str, trunc: int, bs: int) -> np.ndarray:
    embs = []
    for chunk in batched(texts, bs):
        x = enc_tok(list(chunk), return_tensors="pt", padding=True, truncation=True, max_length=trunc).to(device)
        out = enc_model(**x)
        if hasattr(out, "last_hidden_state"):
            h = out.last_hidden_state
            if "attention_mask" in x:
                m = x["attention_mask"].unsqueeze(-1).expand(h.size()).float()
                h = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)  # mean pooling
            else:
                h = h[:, 0, :]
        else:
            h = out.pooler_output
        h = F.normalize(h, dim=-1)
        embs.append(h.cpu())
    return torch.cat(embs).numpy()

# -------------------- NLL (perplexity proxy) --------------------
@torch.inference_mode()
def nll_list(texts: List[str], gen_tok, gen_model, device: str, bs: int = 8, max_len: int = 256) -> List[float]:
    vals = []
    for chunk in batched(texts, bs):
        enc = gen_tok(list(chunk), return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
        input_ids = enc["input_ids"]
        attn = enc["attention_mask"]
        shift_labels = input_ids.clone()
        out = gen_model(input_ids, attention_mask=attn, labels=shift_labels)
        # transformers会自动shift labels并返回loss（平均token NLL）
        loss = out.loss.detach().cpu().numpy().tolist()
        if isinstance(loss, float):  # 可能是标量
            vals.extend([loss] * input_ids.size(0))
        else:
            vals.extend(loss)
    return vals

# -------------------- scoring per x --------------------
def score_one(
    x: str, args, gen_tok, gen_model, enc_tok, enc_model, device, scaler
) -> float:
    # Step1: D_aux
    D_aux = build_D_aux(x, m=args.m_aux, device=device)

    # Step2: repeats on x
    g_texts_x = generate_texts(x, args.n_repeat_x, gen_tok, gen_model, device,
                               args.max_new_tokens, args.temperature, args.top_p, temp_jitter=0.05)
    Hx = encode_texts(g_texts_x, enc_tok, enc_model, device, args.enc_trunc_len, args.enc_bs)
    Hx = scaler.transform(Hx)
    cos_x = pairwise_cos_mean(Hx)
    nll_x = np.var(nll_list(g_texts_x, gen_tok, gen_model, device, bs=8))

    # Step3: repeats on each xi in D_aux
    g_texts_aux = []
    for xi in D_aux:
        g_texts_aux.extend(generate_texts(xi, args.t_repeat_aux, gen_tok, gen_model, device,
                                          args.max_new_tokens, args.temperature, args.top_p, temp_jitter=0.05))
    Haux = encode_texts(g_texts_aux, enc_tok, enc_model, device, args.enc_trunc_len, args.enc_bs)
    Haux = scaler.transform(Haux)
    cos_aux = pairwise_cos_mean(Haux)
    nll_aux = np.var(nll_list(g_texts_aux, gen_tok, gen_model, device, bs=8))

    # Step4: LLR on embeddings
    Gx, Gaux = fit_gaussian(Hx, args.cov_reg), fit_gaussian(Haux, args.cov_reg)
    llr = float(loglik(Hx, Gx).mean() - loglik(Hx, Gaux).mean())

    # Combined score
    score = args.w1 * llr + args.w2 * (cos_x - cos_aux) - args.w3 * (nll_x - nll_aux)
    return float(score)

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # eval sizes
    ap.add_argument("--n_pos", type=int, default=300)
    ap.add_argument("--n_neg", type=int, default=300)
    # repeats / aux sizes
    ap.add_argument("--m_aux", type=int, default=8)
    ap.add_argument("--n_repeat_x", type=int, default=12)
    ap.add_argument("--t_repeat_aux", type=int, default=6)
    # models
    ap.add_argument("--gen_name", type=str, default="gpt2")
    ap.add_argument("--enc_name", type=str, default="sentence-transformers/all-mpnet-base-v2")
    # generation / encoding
    ap.add_argument("--max_new_tokens", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--enc_trunc_len", type=int, default=256)
    ap.add_argument("--enc_bs", type=int, default=256)
    # gaussian & weights
    ap.add_argument("--cov_reg", type=float, default=1e-2)
    ap.add_argument("--w1", type=float, default=1.0)
    ap.add_argument("--w2", type=float, default=0.7)
    ap.add_argument("--w3", type=float, default=0.5)
    args = ap.parse_args()
    print(vars(args))

    set_seed(args.seed)
    device = args.device
    torch.set_grad_enabled(False)

    # Data
    ds = load_dataset("ag_news")
    train_texts = list(ds["train"]["text"])
    test_texts  = list(ds["test"]["text"])
    random.seed(args.seed)
    pos_samples = random.sample(train_texts, min(args.n_pos, len(train_texts)))
    neg_samples = random.sample(test_texts,  min(args.n_neg, len(test_texts)))

    # Models
    gen_tok = AutoTokenizer.from_pretrained(args.gen_name)
    if gen_tok.pad_token_id is None:
        gen_tok.pad_token_id = gen_tok.eos_token_id
    gen_model = AutoModelForCausalLM.from_pretrained(args.gen_name).to(device).eval()

    enc_tok = AutoTokenizer.from_pretrained(args.enc_name)
    enc_model = AutoModel.from_pretrained(args.enc_name).to(device).eval()

    # A global scaler for embedding space (fit with a small warmup set)
    warm = pos_samples[:100] + neg_samples[:100]
    warm_texts = []
    for w in warm:
        warm_texts.extend(generate_texts(w, 2, gen_tok, gen_model, device,
                                         args.max_new_tokens, args.temperature, args.top_p))
    warm_emb = encode_texts(warm_texts, enc_tok, enc_model, device, args.enc_trunc_len, args.enc_bs)
    scaler = StandardScaler().fit(warm_emb)

    # Evaluate
    y_true, y_score = [], []

    print("\nScoring members (train) ...")
    for x in tqdm(pos_samples):
        y_true.append(1)
        y_score.append(score_one(x, args, gen_tok, gen_model, enc_tok, enc_model, device, scaler))

    print("Scoring non-members (test) ...")
    for x in tqdm(neg_samples):
        y_true.append(0)
        y_score.append(score_one(x, args, gen_tok, gen_model, enc_tok, enc_model, device, scaler))

    # ROC / AUC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    if auc < 0.5:  # flip if inverted
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
    plt.title("noise_gpt2_agnews")
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig("roc_curve_comp_based.png", dpi=150)
    print("Saved: roc_curve_comp_based.png")

    # also persist scores if你想用绘图脚本再处理
    np.savez("mia_scores_improved.npz", y_true=np.array(y_true), y_score=np.array(y_score))
    print("Saved: mia_scores_improved.npz")


if __name__ == "__main__":
    main()

