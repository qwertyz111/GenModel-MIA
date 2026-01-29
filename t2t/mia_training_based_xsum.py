#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XSum membership-style separability (aim AUC ~0.7–0.9)

Key choices to avoid trivial separation:
  • Dataset: sentence-transformers/xsum  (single 'train' split; fields: article, summary)
  • Negatives: conditioned synthetic summaries via facebook/bart-large-xsum
  • Length matching (±1 word) & de-dup between train/val
  • Encoder: sentence-transformers/all-mpnet-base-v2 (default)
  • Scaler fit on REAL only (no synthetic leakage)
  • Optional PCA to 64–256 dims (default off)
"""

import argparse, math, random, numpy as np
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# -------------------- args --------------------
p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, default=42)
p.add_argument("--device", type=str, default="cuda:0")
p.add_argument("--n_train", type=int, default=2000)
p.add_argument("--n_val", type=int, default=500)
p.add_argument("--n_syn", type=int, default=2000)
p.add_argument("--max_new_tokens", type=int, default=80)
p.add_argument("--batch_size", type=int, default=32)
p.add_argument("--len_min", type=int, default=10)
p.add_argument("--len_max", type=int, default=80)
p.add_argument("--trunc_len", type=int, default=256)
p.add_argument("--enc_name", type=str, default="sentence-transformers/all-mpnet-base-v2")
p.add_argument("--gen_name", type=str, default="facebook/bart-large-xsum")
p.add_argument("--cov_reg", type=float, default=1e-2)
p.add_argument("--pca_dim", type=int, default=0, help="0=disable; else reduce to this dim")
p.add_argument("--scale_on", type=str, choices=["real","both"], default="real",
               help="fit StandardScaler on 'real' only (recommended) or 'both'")
args = p.parse_args()
print(vars(args))

# -------------------- setup --------------------
torch.set_grad_enabled(False)
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
device = torch.device(args.device)

def ok_len(t: str) -> bool:
    n = len(t.split()) if t else 0
    return (args.len_min <= n <= args.len_max)

# -------------------- data --------------------
print("\nLoading XSum (parquet mirror: sentence-transformers/xsum) ...")
ds = load_dataset("sentence-transformers/xsum")  # train split only, fields: article, summary

# use summaries as REAL texts
all_summs = [t for t in ds["train"]["summary"] if isinstance(t, str) and t.strip() and ok_len(t)]
random.shuffle(all_summs)
real_train = all_summs[:args.n_train]
real_val   = all_summs[args.n_train: args.n_train + args.n_val]
# de-dup leakage
train_set = set(real_train)
real_val  = [t for t in real_val if t not in train_set]
print(f"Real train={len(real_train)}  Real val={len(real_val)}")

# -------------------- conditioned synthetic (BART-xsum) --------------------
print("\nGenerating conditioned synthetic summaries with", args.gen_name)
bart_tok = AutoTokenizer.from_pretrained(args.gen_name)
bart = AutoModelForSeq2SeqLM.from_pretrained(args.gen_name).to(device).eval()

all_articles = [t for t in ds["train"]["article"] if isinstance(t, str) and t.strip()]
random.shuffle(all_articles)
src_articles = all_articles[:max(args.n_syn, args.n_train + args.n_val)]

val_lens = np.array([len(t.split()) for t in real_val]) if len(real_val) else np.array([args.len_min, args.len_max])
def accept_like_val_len(txt: str, window:int=1)->bool:
    if not txt: return False
    L = len(txt.split())
    if L < args.len_min or L > args.len_max: return False
    if len(val_lens):
        mask = (val_lens >= L-window) & (val_lens <= L+window)
        p = 0.15 + 0.85*float(mask.mean())
    else:
        p = 0.5
    return (np.random.rand() < p)

syn_texts = []
pbar = tqdm(total=args.n_syn, desc="Synth accepted (BART-xsum)")
for art in src_articles:
    if len(syn_texts) >= args.n_syn: break
    enc = bart_tok(art, return_tensors="pt", truncation=True, max_length=512).to(device)
    out = bart.generate(
        **enc,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        top_p=0.95,
        temperature=0.9,
        num_return_sequences=1,
        length_penalty=1.0,
        early_stopping=True,
        use_cache=True,
    )
    txt = bart_tok.decode(out[0], skip_special_tokens=True).strip()
    if accept_like_val_len(txt, window=1):
        syn_texts.append(txt); pbar.update(1)
pbar.close()
print(f"Synthetic kept: {len(syn_texts)}")

# -------------------- encoder --------------------
print("\nEncoding with encoder:", args.enc_name)
enc_tok = AutoTokenizer.from_pretrained(args.enc_name)
enc = AutoModel.from_pretrained(args.enc_name).to(device).eval()

def encode_texts(texts):
    embs=[]; loader=DataLoader(texts, batch_size=args.batch_size)
    for batch in tqdm(loader, desc="Encoding"):
        x = enc_tok(list(batch), padding=True, truncation=True,
                    max_length=args.trunc_len, return_tensors="pt").to(device)
        out = enc(**x)
        if hasattr(out, "last_hidden_state"):
            h = out.last_hidden_state
            # mean pool
            if "attention_mask" in x:
                mask = x["attention_mask"].unsqueeze(-1).expand(h.size()).float()
                h = (h*mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            else:
                h = h[:,0,:]
        else:
            h = out.pooler_output
        h = torch.nn.functional.normalize(h, dim=-1)
        embs.append(h.cpu())
    return torch.cat(embs).numpy()

emb_real_train = encode_texts(real_train)
emb_syn_train  = encode_texts(syn_texts)

# -------------------- PCA (optional) --------------------
pca = None
if args.pca_dim and args.pca_dim > 0:
    print(f"\nPCA → {args.pca_dim} dims (fit on REAL only)")
    pca = PCA(n_components=args.pca_dim, random_state=args.seed)
    emb_real_train = pca.fit_transform(emb_real_train)
    emb_syn_train  = pca.transform(emb_syn_train)

# -------------------- scaling --------------------
if args.scale_on == "real":
    print("Scaler: fit on REAL only")
    scaler = StandardScaler().fit(emb_real_train)
else:
    print("Scaler: fit on BOTH real+synthetic")
    scaler = StandardScaler().fit(np.vstack([emb_real_train, emb_syn_train]))
emb_real_train = scaler.transform(emb_real_train)
emb_syn_train  = scaler.transform(emb_syn_train)

# -------------------- Gaussians --------------------
def fit_gaussian(embs: np.ndarray, cov_reg: float):
    mu = np.mean(embs, axis=0)
    cov = np.cov(embs.T) + cov_reg*np.eye(embs.shape[1])
    inv = np.linalg.pinv(cov)
    logdet = np.log(np.linalg.det(cov) + 1e-12)
    return {"mu": mu, "inv": inv, "logdet": logdet, "dim": embs.shape[1]}

def loglik(x: np.ndarray, g):
    diff = (x - g["mu"])
    maha = (diff @ g["inv"] * diff).sum(axis=-1)
    return -0.5*(maha + g["logdet"] + g["dim"]*math.log(2*math.pi))

dist_real = fit_gaussian(emb_real_train, args.cov_reg)
dist_syn  = fit_gaussian(emb_syn_train,  args.cov_reg)

# -------------------- scoring --------------------
def embed_score(texts):
    scores=[]
    loader=DataLoader(texts, batch_size=1)  # per-text scoring
    for [t] in tqdm(loader, desc="Scoring"):
        # encode
        x = enc_tok([t], padding=True, truncation=True, max_length=args.trunc_len, return_tensors="pt").to(device)
        out = enc(**x)
        if hasattr(out,"last_hidden_state"):
            h = out.last_hidden_state
            if "attention_mask" in x:
                mask = x["attention_mask"].unsqueeze(-1).expand(h.size()).float()
                h = (h*mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            else:
                h = h[:,0,:]
        else:
            h = out.pooler_output
        h = torch.nn.functional.normalize(h, dim=-1).cpu().numpy()
        # PCA/scaler
        if pca is not None:
            h = pca.transform(h)
        h = scaler.transform(h)
        # log-likelihood ratio
        s = float(loglik(h, dist_real) - loglik(h, dist_syn))
        scores.append(s)
    return scores

neg_samples = random.sample(syn_texts, min(len(syn_texts), len(real_val)))
y_true  = [1]*len(real_val) + [0]*len(neg_samples)
y_score = embed_score(real_val) + embed_score(neg_samples)

fpr, tpr, _ = roc_curve(y_true, y_score)
auc = roc_auc_score(y_true, y_score)
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
plt.title("XSum – Real vs BART-xsum Synthetic (hard negatives)")
plt.legend(loc="lower right"); plt.tight_layout()
plt.savefig("roc_curve_xsum.png", dpi=150)
print("Saved: roc_curve_xsum.png")
