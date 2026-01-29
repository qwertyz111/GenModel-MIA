import os
import math
import random
import argparse
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    AutoConfig,
    set_seed,
)


# ------------------------------
# Utils
# ------------------------------
LABEL2NAME = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

def batched(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}

def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return 1 - a_norm @ b_norm.T  # cosine distance matrix


# ------------------------------
# Step 1: Generator T -> synthetic AG_NEWS
# ------------------------------
@torch.inference_mode()
def generate_synthetic_texts(
    n_per_class: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: str,
    gpt_name: str = "gpt2",
) -> Tuple[List[str], List[int]]:
    """
    Use GPT-2 to sample synthetic headlines/articles conditioned by a simple prompt.
    """
    tok = AutoTokenizer.from_pretrained(gpt_name)
    # gpt2 has no pad -> set pad to eos
    tok.pad_token = tok.eos_token
    gpt = AutoModelForCausalLM.from_pretrained(gpt_name).to(device)
    gpt.eval()

    prompts = {
        0: "Category: World. News headline and brief: ",
        1: "Category: Sports. News headline and brief: ",
        2: "Category: Business. News headline and brief: ",
        3: "Category: Sci/Tech. News headline and brief: ",
    }

    texts, labels = [], []
    for y in range(4):
        p = prompts[y]
        inputs = tok([p] * n_per_class, return_tensors="pt", padding=True).to(device)
        out = gpt.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.eos_token_id,
        )
        gen = tok.batch_decode(out, skip_special_tokens=True)
        # strip prefix prompt to keep only generated continuation
        gen = [g[len(p):].strip() if g.startswith(p) else g.strip() for g in gen]
        texts += gen
        labels += [y] * n_per_class

    return texts, labels


# ------------------------------
# Step 2/3: Encoder E -> text embeddings
# ------------------------------
@torch.inference_mode()
def encode_texts(
    texts: List[str],
    device: str,
    enc_name: str = "distilbert-base-uncased",
    batch_size: int = 64,
    normalize: bool = True,
) -> np.ndarray:
    tok = AutoTokenizer.from_pretrained(enc_name)
    enc = AutoModel.from_pretrained(enc_name).to(device)
    enc.eval()

    embs = []
    for chunk in tqdm(list(batched(texts, batch_size)), desc="Encoding"):
        batch = tok(
            chunk,
            padding=True,
            truncation=True,
            max_length=192,
            return_tensors="pt",
        )
        batch = to_device(batch, device)
        outputs = enc(**batch).last_hidden_state  # [B, L, H]
        # Use CLS-equivalent: DistilBERT has no [CLS], take first token embedding
        cls = outputs[:, 0, :]                     # [B, H]
        if normalize:
            cls = F.normalize(cls, dim=-1)
        embs.append(cls.cpu().numpy())
    return np.vstack(embs)  # [N, H]


# ------------------------------
# Step 4: Fit Gaussian (mean + cov) as distribution summary
# ------------------------------
def fit_gaussian(embs: np.ndarray, eps: float = 1e-5):
    mu = embs.mean(axis=0)
    # shrinked covariance for stability
    cov = np.cov(embs.T)
    cov = cov + eps * np.eye(cov.shape[0])
    inv = np.linalg.inv(cov)
    logdet = float(np.linalg.slogdet(cov)[1])
    return {"mu": mu, "inv": inv, "logdet": logdet, "dim": embs.shape[1]}

def loglik_gaussian(x: np.ndarray, g):
    # log N(x | mu, cov)
    diff = (x - g["mu"])
    maha = (diff @ g["inv"] * diff).sum(axis=-1)  # (x-mu)^T inv (x-mu)
    return -0.5 * (maha + g["logdet"] + g["dim"] * math.log(2 * math.pi))


# ------------------------------
# Step 5: Membership decision for a suspicious sample
# ------------------------------
@torch.inference_mode()
def membership_decision(
    text: str,
    enc_tok,
    enc_model,
    dist_real,
    dist_syn,
    device: str,
) -> Tuple[bool, float, float]:
    batch = enc_tok(
        [text],
        padding=True,
        truncation=True,
        max_length=192,
        return_tensors="pt",
    )
    batch = to_device(batch, device)
    h = enc_model(**batch).last_hidden_state[:, 0, :]  # [1, H]
    h = F.normalize(h, dim=-1).cpu().numpy()           # [1, H]

    ll_real = float(loglik_gaussian(h, dist_real))
    ll_syn = float(loglik_gaussian(h, dist_syn))
    # If closer (higher log-likelihood) to REAL than SYN -> member
    return (ll_real > ll_syn), ll_real, ll_syn


# ------------------------------
# Pipeline
# ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # data sizes
    parser.add_argument("--n_train_per_class", type=int, default=1500,
                        help="real samples used to build D_real distribution")
    parser.add_argument("--n_val_per_class", type=int, default=200,
                        help="held-out real samples used for eval of decision rule")
    parser.add_argument("--n_syn_per_class", type=int, default=1500,
                        help="synthetic samples per class for D_syn")

    # generator settings
    parser.add_argument("--gpt_name", type=str, default="gpt2")
    parser.add_argument("--max_new_tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.95)

    # encoder settings
    parser.add_argument("--enc_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--batch_size", type=int, default=64)

    # mixture / global distribution or class-conditional
    parser.add_argument("--class_conditional", action="store_true",
                        help="if set, fit per-class Gaussians then average loglik; "
                             "otherwise fit global Gaussians (default).")

    args = parser.parse_args()
    print(vars(args))
    set_seed(args.seed)

    device = args.device

    # ----- Load AG_NEWS real dataset -----
    ds = load_dataset("ag_news")
    # Split: use a subset to speed up if needed
    train_texts, train_labels = ds["train"]["text"], ds["train"]["label"]
    test_texts, test_labels = ds["test"]["text"], ds["test"]["label"]

    # Subsample per class for distribution fitting & validation
    def take_per_class(texts, labels, n_per_class):
        buckets = {0: [], 1: [], 2: [], 3: []}
        for t, y in zip(texts, labels):
            if len(buckets[y]) < n_per_class:
                buckets[y].append(t)
            if all(len(v) >= n_per_class for v in buckets.values()):
                break
        flat = [t for y in range(4) for t in buckets[y]]
        labs = [y for y in range(4) for _ in range(n_per_class)]
        return flat, labs

    real_train_texts, real_train_labels = take_per_class(train_texts, train_labels, args.n_train_per_class)
    real_val_texts, real_val_labels = take_per_class(test_texts, test_labels, args.n_val_per_class)

    # ----- Step 1: Generate synthetic data with GPT-2 -----
    syn_texts, syn_labels = generate_synthetic_texts(
        n_per_class=args.n_syn_per_class,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=device,
        gpt_name=args.gpt_name,
    )

    # ----- Step 2/3: Encode real & synthetic with DistilBERT -----
    enc_tok = AutoTokenizer.from_pretrained(args.enc_name)
    enc_model = AutoModel.from_pretrained(args.enc_name).to(device).eval()

    emb_real = encode_texts(real_train_texts, device, args.enc_name, args.batch_size, normalize=True)
    emb_syn = encode_texts(syn_texts, device, args.enc_name, args.batch_size, normalize=True)

    # (optional) class-conditional embeddings
    if args.class_conditional:
        emb_real_cls = {c: [] for c in range(4)}
        for e, y in zip(encode_texts(real_train_texts, device, args.enc_name, args.batch_size, True), real_train_labels[:len(real_train_texts)]):
            emb_real_cls[y].append(e)
        # but above line re-encodes and misaligns; simpler: recompute aligned
        emb_real_cls = {c: [] for c in range(4)}
        idx = {c: 0 for c in range(4)}
        for t, y in zip(real_train_texts, real_train_labels):
            emb = encode_texts([t], device, args.enc_name, 1, True)
            emb_real_cls[y].append(emb[0])
            if all(len(v) >= args.n_train_per_class for v in emb_real_cls.values()):
                break
        emb_real_cls = {c: np.vstack(v) for c, v in emb_real_cls.items()}

        emb_syn_cls = {c: [] for c in range(4)}
        for t, y in zip(syn_texts, syn_labels):
            emb = encode_texts([t], device, args.enc_name, 1, True)
            emb_syn_cls[y].append(emb[0])
        emb_syn_cls = {c: np.vstack(v) for c, v in emb_syn_cls.items()}

    # ----- Step 4: Fit distributions (Gaussian with shrinkage) -----
    if args.class_conditional:
        dist_real = {c: fit_gaussian(emb_real_cls[c]) for c in range(4)}
        dist_syn  = {c: fit_gaussian(emb_syn_cls[c]) for c in range(4)}
    else:
        dist_real = fit_gaussian(emb_real)
        dist_syn  = fit_gaussian(emb_syn)

    # ----- Step 5: Evaluate on held-out real (members) vs synthetic (non-members proxy) -----
    # Here we simulate suspicious samples: (A) true real held-out -> expected "member"
    # and (B) synthetic samples -> expected "non-member".
    print("\nEvaluating membership decisions...")
    enc_tok_eval = enc_tok
    enc_model_eval = enc_model

    tp = fp = tn = fn = 0
    # Real held-out
    for txt in tqdm(real_val_texts, desc="Eval real held-out"):
        if args.class_conditional:
            # without label info, average class loglik as a simple mixture prior
            with torch.inference_mode():
                batch = enc_tok_eval([txt], padding=True, truncation=True, max_length=192, return_tensors="pt")
                batch = to_device(batch, device)
                h = enc_model_eval(**batch).last_hidden_state[:,0,:]
                h = F.normalize(h, dim=-1).cpu().numpy()
            llr = np.mean([loglik_gaussian(h, dist_real[c]) for c in range(4)], axis=0)
            lls = np.mean([loglik_gaussian(h, dist_syn[c]) for c in range(4)], axis=0)
            pred_member = llr > lls
        else:
            pred_member = loglik_gaussian(x=encode_texts([txt], device, args.enc_name, 1, True), g=dist_real) > \
                          loglik_gaussian(x=encode_texts([txt], device, args.enc_name, 1, True), g=dist_syn)

        if bool(pred_member):
            tp += 1
        else:
            fn += 1

    # Synthetic as negatives
    neg_samples = syn_texts[:len(real_val_texts)]
    for txt in tqdm(neg_samples, desc="Eval synthetic negatives"):
        if args.class_conditional:
            with torch.inference_mode():
                batch = enc_tok_eval([txt], padding=True, truncation=True, max_length=192, return_tensors="pt")
                batch = to_device(batch, device)
                h = enc_model_eval(**batch).last_hidden_state[:,0,:]
                h = F.normalize(h, dim=-1).cpu().numpy()
            llr = np.mean([loglik_gaussian(h, dist_real[c]) for c in range(4)], axis=0)
            lls = np.mean([loglik_gaussian(h, dist_syn[c]) for c in range(4)], axis=0)
            pred_member = llr > lls
        else:
            pred_member = loglik_gaussian(h=encode_texts([txt], device, args.enc_name, 1, True), g=dist_real) > \
                          loglik_gaussian(h=encode_texts([txt], device, args.enc_name, 1, True), g=dist_syn)

        if bool(pred_member):
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    acc       = (tp + tn) / (tp + tn + fp + fn + 1e-9)
    print(f"\nResults (real-heldout as positives, synthetic as negatives):")
    print(f"TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"Accuracy={acc:.4f}  Precision={precision:.4f}  Recall={recall:.4f}")

    # Demo: single suspicious sample
    demo_text = real_val_texts[0]
    with torch.inference_mode():
        batch = enc_tok_eval([demo_text], padding=True, truncation=True, max_length=192, return_tensors="pt")
        batch = to_device(batch, device)
        h = enc_model_eval(**batch).last_hidden_state[:,0,:]
        h = F.normalize(h, dim=-1).cpu().numpy()
    if args.class_conditional:
        llr = np.mean([loglik_gaussian(h, dist_real[c]) for c in range(4)], axis=0)
        lls = np.mean([loglik_gaussian(h, dist_syn[c]) for c in range(4)], axis=0)
        pred = bool(llr > lls)
    else:
        pred = bool(loglik_gaussian(h, dist_real) > loglik_gaussian(h, dist_syn))
    print("\nExample suspicious sample:")
    print(demo_text[:200].replace("\n", " ") + ("..." if len(demo_text) > 200 else ""))
    print("→ Predicted:", "MEMBER" if pred else "NON-MEMBER")


if __name__ == "__main__":
    main()
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# -------------------- 新增：收集打分 --------------------
y_true, y_score = [], []

# real-heldout 正样本
for txt in tqdm(real_val_texts, desc="Eval real-heldout (ROC)"):
    with torch.inference_mode():
        batch = enc_tok_eval([txt], padding=True, truncation=True, max_length=192, return_tensors="pt")
        batch = to_device(batch, device)
        h = enc_model_eval(**batch).last_hidden_state[:, 0, :]
        h = F.normalize(h, dim=-1).cpu().numpy()
    ll_real = loglik_gaussian(h, dist_real)
    ll_syn  = loglik_gaussian(h, dist_syn)
    y_true.append(1)                            # real → 正样本
    y_score.append(float(ll_real - ll_syn))     # 打分：越大越像真实

# synthetic 负样本
for txt in tqdm(neg_samples, desc="Eval synthetic (ROC)"):
    with torch.inference_mode():
        batch = enc_tok_eval([txt], padding=True, truncation=True, max_length=192, return_tensors="pt")
        batch = to_device(batch, device)
        h = enc_model_eval(**batch).last_hidden_state[:, 0, :]
        h = F.normalize(h, dim=-1).cpu().numpy()
    ll_real = loglik_gaussian(h, dist_real)
    ll_syn  = loglik_gaussian(h, dist_syn)
    y_true.append(0)                            # synthetic → 负样本
    y_score.append(float(ll_real - ll_syn))

# -------------------- 计算并绘制 ROC/AUC --------------------
fpr, tpr, _ = roc_curve(y_true, y_score)
auc_val = roc_auc_score(y_true, y_score)
print(f"\nROC-AUC = {auc_val:.4f}")

plt.figure(figsize=(5, 5))
plt.plot(fpr, tpr, label=f"AUC = {auc_val:.4f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Membership Inference ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)
plt.show()


