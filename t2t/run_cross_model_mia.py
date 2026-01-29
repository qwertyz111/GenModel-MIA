#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-model Computation-based MIA, 6 runs:
  Datasets: ag_news, wikitext-103, xsum
  Pairs:    (T=gpt2,   G=falcon-7b-instruct), (T=falcon-7b-instruct, G=gpt2)
Total eval size per run = 2000 (n_pos=1000, n_neg=1000)

Output:
  - Per-run ROC curve PNG -> roc_<dataset>__T-<tgt>__G-<gen>.png
  - Summary CSV          -> results_cross_mia.csv  (model,accuracy,auc,tpr_at_1fpr)
"""

import os, math, random, json, csv, numpy as np
from typing import List, Dict, Tuple
import argparse
from tqdm import tqdm

import torch, torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM, set_seed
)

from datasets import load_dataset
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# -------------------- tiny utils --------------------
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
        while len(aux) < m:  # top up
            aux.append(noise_text(x))
    except Exception as e:
        print(f"[D_aux] paraphraser failed: {e} -> using noisy variants.")
        aux = [noise_text(x) for _ in range(m)]
    return aux[:m]


# -------------------- loaders --------------------
def load_causal_lm(name: str, device: str):
    """
    Try BF16/FP16; if OOM and bitsandbytes/accelerate are available, fallback to 8-bit/device_map='auto'.
    """
    kw = {}
    # prefer bfloat16 if available
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        kw["torch_dtype"] = torch.bfloat16
    else:
        kw["torch_dtype"] = torch.float16

    try:
        model = AutoModelForCausalLM.from_pretrained(name, **kw).to(device).eval()
        tok = AutoTokenizer.from_pretrained(name)
        if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
        return tok, model, "fp16/bf16"
    except Exception as e:
        print(f"[load_causal_lm] FP16/BF16 load failed for {name}: {e}")
        # try 8-bit w/ bitsandbytes
        try:
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(name, quantization_config=bnb, device_map="auto")
            tok = AutoTokenizer.from_pretrained(name)
            if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
            return tok, model, "int8"
        except Exception as e2:
            print(f"[load_causal_lm] 8-bit load also failed for {name}: {e2}")
            raise

def load_encoder(name: str, device: str):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name).to(device).eval()
    return tok, model


# -------------------- generation / encoding / NLL --------------------
@torch.inference_mode()
def generate_texts(q: str, n: int, tok, model, device: str,
                   max_new_tokens: int, temperature: float, top_p: float,
                   temp_jitter: float = 0.05) -> List[str]:
    outs = []
    for _ in range(n):
        t_cur = max(0.1, random.uniform(temperature - temp_jitter, temperature + temp_jitter))
        inp = tok(q if q.strip() else " ", return_tensors="pt", add_special_tokens=False).to(device)
        L = inp["input_ids"].shape[1]
        out = model.generate(
            **inp, do_sample=True, temperature=t_cur, top_p=top_p,
            max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id, use_cache=True
        )
        txt = tok.decode(out[0][L:], skip_special_tokens=True).strip()
        if txt: outs.append(txt)
    return outs if outs else [""]

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

@torch.inference_mode()
def nll_list(texts: List[str], tok, model, device: str, bs: int = 8, max_len: int = 256) -> List[float]:
    vals = []
    for chunk in batched(texts, bs):
        enc = tok(list(chunk), return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
        ids, attn = enc["input_ids"], enc["attention_mask"]
        out = model(ids, attention_mask=attn, labels=ids)
        loss = out.loss.detach().cpu().numpy().tolist()
        if isinstance(loss, float):
            vals.extend([loss] * ids.size(0))
        else:
            vals.extend(loss)
    return vals


# -------------------- per-sample score --------------------
def score_one(x: str, args,
              # target model T (also used for NLL)
              tok_T, model_T,
              # generator (could be T or alternate G)
              tok_G, model_G,
              # encoder
              enc_tok, enc_model, device, scaler) -> float:

    D_aux = build_D_aux(x, m=args.m_aux, device=device)

    # repeats on x
    g_x = generate_texts(x, args.n_repeat_x, tok_G, model_G, device,
                         args.max_new_tokens, args.temperature, args.top_p)
    Hx = encode_texts(g_x, enc_tok, enc_model, device, args.enc_trunc_len, args.enc_bs)
    Hx = scaler.transform(Hx)
    cos_x = pairwise_cos_mean(Hx)
    var_nll_x = np.var(nll_list(g_x, tok_T, model_T, device, bs=8))  # under TARGET

    # repeats on D_aux
    g_aux = []
    for xi in D_aux:
        g_aux.extend(generate_texts(xi, args.t_repeat_aux, tok_G, model_G, device,
                                    args.max_new_tokens, args.temperature, args.top_p))
    Haux = encode_texts(g_aux, enc_tok, enc_model, device, args.enc_trunc_len, args.enc_bs)
    Haux = scaler.transform(Haux)
    cos_aux = pairwise_cos_mean(Haux)
    var_nll_aux = np.var(nll_list(g_aux, tok_T, model_T, device, bs=8))

    # LLR
    Gx, Gaux = fit_gaussian(Hx, args.cov_reg), fit_gaussian(Haux, args.cov_reg)
    llr = float(loglik(Hx, Gx).mean() - loglik(Hx, Gaux).mean())

    return float(args.w1*llr + args.w2*(cos_x - cos_aux) - args.w3*(var_nll_x - var_nll_aux))


# -------------------- datasets --------------------
def load_members_nonmembers(dataset: str, total: int, seed: int = 42) -> Tuple[List[str], List[str]]:
    """
    Return (members, nonmembers), each length = total//2.
    """
    random.seed(seed)
    half = total // 2

    if dataset.lower() in ["ag_news", "agnews", "ag-news"]:
        ds = load_dataset("ag_news")
        train_texts = list(ds["train"]["text"])
        test_texts  = list(ds["test"]["text"])
        members = random.sample(train_texts, half)
        nonmembers = random.sample(test_texts, half)
        return members, nonmembers

    if dataset.lower() in ["wikitext-103", "wiki103", "wikitext103"]:
        ds = load_dataset("wikitext", "wikitext-103-v1")
        train_txt = [t for t in ds["train"]["text"] if t and len(t.split()) >= 20]
        test_txt  = [t for t in ds["test"]["text"]  if t and len(t.split()) >= 20]
        members = random.sample(train_txt, half)
        nonmembers = random.sample(test_txt, half)
        return members, nonmembers

    if dataset.lower() in ["xsum"]:
        # sentence-transformers/xsum has only 'train'; split it ourselves
        ds = load_dataset("sentence-transformers/xsum")
        all_articles = [t for t in ds["train"]["article"] if t and 30 <= len(t.split()) <= 150]
        random.shuffle(all_articles)
        assert len(all_articles) >= total, "XSum mirror not large enough for requested total."
        members = all_articles[:half]
        nonmembers = all_articles[half:total]
        return members, nonmembers

    raise ValueError(f"Unknown dataset: {dataset}")


# -------------------- one run --------------------
def run_one(dataset: str, tgt_name: str, gen_name: str, args, device: str) -> Dict[str, float]:
    print(f"\n=== RUN: dataset={dataset} | TARGET={tgt_name} | GENERATOR={gen_name} ===")
    # data
    members, nonmembers = load_members_nonmembers(dataset, args.total_eval, args.seed)

    # models
    tok_T, model_T, mode_T = load_causal_lm(tgt_name, device)
    tok_G, model_G, mode_G = load_causal_lm(gen_name, device)
    print(f"[Target] {tgt_name} loaded in {mode_T}; [Generator] {gen_name} loaded in {mode_G}")

    enc_tok, enc_model = load_encoder(args.enc_name, device)

    # scaler warmup
    warm = members[:100] + nonmembers[:100]
    warm_gen = []
    for w in warm:
        warm_gen.extend(generate_texts(w, 2, tok_G, model_G, device,
                                       args.max_new_tokens, args.temperature, args.top_p))
    warm_emb = encode_texts(warm_gen, enc_tok, enc_model, device, args.enc_trunc_len, args.enc_bs)
    scaler = StandardScaler().fit(warm_emb)

    # score
    y_true, y_score = [], []
    print("Scoring members ...")
    for x in tqdm(members):
        y_true.append(1)
        y_score.append(score_one(x, args, tok_T, model_T, tok_G, model_G, enc_tok, enc_model, device, scaler))

    print("Scoring non-members ...")
    for x in tqdm(nonmembers):
        y_true.append(0)
        y_score.append(score_one(x, args, tok_T, model_T, tok_G, model_G, enc_tok, enc_model, device, scaler))

    # metrics
    fpr, tpr, thr = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    if auc < 0.5:  # flip if inverted
        y_score = [-s for s in y_score]
        fpr, tpr, thr = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)

    # choose threshold by Youden's J for accuracy
    J = tpr - fpr
    j_idx = int(np.argmax(J))
    thr_star = thr[j_idx] if j_idx < len(thr) else 0.0
    y_pred = [1 if s >= thr_star else 0 for s in y_score]
    acc = float((np.array(y_pred) == np.array(y_true)).mean())

    tpr_at_1 = max(tpr[np.array(fpr) <= 0.01]) if np.any(np.array(fpr) <= 0.01) else 0.0

    # plot
    tag = f"{dataset}__T-{tgt_name.split('/')[-1]}__G-{gen_name.split('/')[-1]}"
    png = f"roc_{tag}.png"
    plt.figure(figsize=(5,5))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0,1],[0,1],'k--')
    plt.scatter([0.01],[tpr_at_1], label=f'TPR@1%FPR={tpr_at_1:.3f}')
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(tag)
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(png, dpi=150)
    print(f"Saved: {png}")

    return {
        "model": tag,
        "accuracy": acc,
        "auc": float(auc),
        "tpr_at_1fpr": float(tpr_at_1),
    }


# -------------------- main: 6 runs --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # total eval size per run
    ap.add_argument("--total_eval", type=int, default=2000)  # 1000 members + 1000 non-members

    # repeats / aux
    ap.add_argument("--m_aux", type=int, default=8)
    ap.add_argument("--n_repeat_x", type=int, default=10)   # 可按显存/时间增减
    ap.add_argument("--t_repeat_aux", type=int, default=5)

    # encoder
    ap.add_argument("--enc_name", type=str, default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--enc_trunc_len", type=int, default=256)
    ap.add_argument("--enc_bs", type=int, default=256)

    # generation / NLL
    ap.add_argument("--max_new_tokens", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=0.95)

    # gaussian + weights
    ap.add_argument("--cov_reg", type=float, default=1e-2)
    ap.add_argument("--w1", type=float, default=1.0)
    ap.add_argument("--w2", type=float, default=0.7)
    ap.add_argument("--w3", type=float, default=0.5)

    # model names
    ap.add_argument("--gpt2_name", type=str, default="gpt2")
    ap.add_argument("--falcon_name", type=str, default="tiiuae/falcon-7b-instruct")

    args = ap.parse_args()
    print(json.dumps(vars(args), indent=2))
    set_seed(args.seed)
    device = args.device
    torch.set_grad_enabled(False)

    # 6 runs
    datasets = ["ag_news", "wikitext-103", "xsum"]
    pairs = [
        ("gpt2", "falcon"),   # T=gpt2,   G=falcon
        ("falcon", "gpt2"),   # T=falcon, G=gpt2
    ]

    name_map = {"gpt2": args.gpt2_name, "falcon": args.falcon_name}

    results = []
    for ds_name in datasets:
        for (tgt_key, gen_key) in pairs:
            tgt = name_map[tgt_key]
            gen = name_map[gen_key]
            res = run_one(ds_name, tgt, gen, args, device)
            results.append(res)

    # write CSV
    csv_path = "results_cross_mia.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model","accuracy","auc","tpr_at_1fpr"])
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"Saved: {csv_path}")
    print("\nDone.")
    

if __name__ == "__main__":
    main()

