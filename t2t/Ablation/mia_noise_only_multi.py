#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Noise-only cosine MIA (no clean baseline)
- Datasets: agnews, wiki103, xsum
- Extractors: user-provided list (same as your ablation list)
- Noise: Gaussian noise added to extracted embedding h, then L2-normalize
- Score(x) = cos(h_noisy, mu_real) - cos(h_noisy, mu_aux)

Outputs:
- one CSV: mia_noise_only_results.csv
- ROC plots per setting: roc_noise_{dataset}_{extractor_sanitized}.png
"""

import argparse, random, re
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, set_seed
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import csv


# ---------------- utils ----------------
def sanitize_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_")

@torch.inference_mode()
def encode_texts_noisy(
    texts,
    tok,
    enc_model,
    device,
    batch_size: int,
    trunc_len: int,
    sigma: float,
):
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        x = tok(
            batch,
            padding=True,
            truncation=True,
            max_length=trunc_len,
            return_tensors="pt",
        ).to(device)
        h = enc_model(**x).last_hidden_state[:, 0, :]         # [B, H]
        h = F.normalize(h, dim=-1)

        # ----- noise injected HERE (on embedding) -----
        if sigma > 0:
            h = h + torch.randn_like(h) * sigma
            h = F.normalize(h, dim=-1)

        embs.append(h.cpu())
    return torch.cat(embs, dim=0).numpy()  # [N, H]

def cosine_with_center(X: np.ndarray, center: np.ndarray) -> np.ndarray:
    # both assumed normalized
    # X: [N,H], center: [H]
    return (X * center[None, :]).sum(axis=1)

def load_real_member_nonmember(dataset: str, n_fit: int, n_eval: int, seed: int):
    rng = random.Random(seed)

    if dataset == "agnews":
        ds = load_dataset("ag_news")
        members = list(ds["train"]["text"])
        nonmembers = list(ds["test"]["text"])
        rng.shuffle(members)
        rng.shuffle(nonmembers)
        mem_fit = members[:n_fit]
        mem_eval = members[n_fit:n_fit + n_eval]
        non_eval = nonmembers[:n_eval]
        return mem_fit, mem_eval, non_eval

    if dataset == "wiki103":
        ds = load_dataset("wikitext", "wikitext-103-raw-v1")
        texts = [t for t in ds["train"]["text"] if t and len(t.split()) >= 20]
        rng.shuffle(texts)
        mem_fit = texts[:n_fit]
        mem_eval = texts[n_fit:n_fit + n_eval]
        # non-member proxy: later part of same corpus (still “non-member” w.r.t. fit subset)
        non_eval = texts[n_fit + n_eval:n_fit + 2 * n_eval]
        if len(non_eval) < n_eval:
            # fallback: take more from end
            non_eval = texts[-n_eval:]
        return mem_fit, mem_eval, non_eval

    if dataset == "xsum":
        ds = load_dataset("sentence-transformers/xsum")  # only 'train'
        docs = [t for t in ds["train"]["article"] if t and len(t.split()) >= 20]
        rng.shuffle(docs)
        mem_fit = docs[:n_fit]
        mem_eval = docs[n_fit:n_fit + n_eval]
        non_eval = docs[n_fit + n_eval:n_fit + 2 * n_eval]
        if len(non_eval) < n_eval:
            non_eval = docs[-n_eval:]
        return mem_fit, mem_eval, non_eval

    raise ValueError(f"Unknown dataset: {dataset}")

def compute_tpr_at_fpr(fpr, tpr, target_fpr=0.01) -> float:
    fpr = np.asarray(fpr)
    tpr = np.asarray(tpr)
    mask = fpr <= target_fpr
    if not np.any(mask):
        return 0.0
    return float(np.max(tpr[mask]))


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--datasets", type=str, default="agnews,wiki103,xsum",
                    help="comma-separated: agnews,wiki103,xsum")
    ap.add_argument("--extractors", type=str, required=True,
                    help="comma-separated extractor names (keep same list as before)")
    ap.add_argument("--n_fit", type=int, default=2000, help="members used to fit center")
    ap.add_argument("--n_eval", type=int, default=2000, help="members/nonmembers for eval")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--trunc_len", type=int, default=192)

    ap.add_argument("--sigma", type=float, default=0.05,
                    help="Gaussian noise std added to embedding (noise-only version)")
    ap.add_argument("--csv_out", type=str, default="mia_noise_only_results.csv")
    args = ap.parse_args()

    print(vars(args), flush=True)
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)

    datasets = [s.strip() for s in args.datasets.split(",") if s.strip()]
    extractors = [s.strip() for s in args.extractors.split(",") if s.strip()]

    # CSV header
    with open(args.csv_out, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "extractor", "n_fit", "n_eval", "sigma",
                        "accuracy", "auc", "tpr_at_1fpr", "roc_png"]
        )
        writer.writeheader()

        for ds_name in datasets:
            # load real member/nonmember pools once per dataset
            mem_fit, mem_eval, non_eval = load_real_member_nonmember(
                ds_name, args.n_fit, args.n_eval, args.seed
            )
            print(f"\n[DATA] {ds_name}: mem_fit={len(mem_fit)} mem_eval={len(mem_eval)} non_eval={len(non_eval)}", flush=True)

            for ext in extractors:
                print(f"\n[RUN] dataset={ds_name} extractor={ext} (noise-only, sigma={args.sigma})", flush=True)

                tok = AutoTokenizer.from_pretrained(ext)
                enc = AutoModel.from_pretrained(ext).to(device).eval()

                # encode noisy
                X_fit = encode_texts_noisy(mem_fit, tok, enc, device, args.batch_size, args.trunc_len, args.sigma)
                X_mem = encode_texts_noisy(mem_eval, tok, enc, device, args.batch_size, args.trunc_len, args.sigma)
                X_non = encode_texts_noisy(non_eval, tok, enc, device, args.batch_size, args.trunc_len, args.sigma)

                # centers (mean of normalized vectors -> re-normalize)
                mu_real = X_fit.mean(axis=0)
                mu_real = mu_real / (np.linalg.norm(mu_real) + 1e-9)

                mu_aux = X_non.mean(axis=0)
                mu_aux = mu_aux / (np.linalg.norm(mu_aux) + 1e-9)

                # score = cos(h, mu_real) - cos(h, mu_aux)
                s_mem = cosine_with_center(X_mem, mu_real) - cosine_with_center(X_mem, mu_aux)
                s_non = cosine_with_center(X_non, mu_real) - cosine_with_center(X_non, mu_aux)

                y_true = np.array([1] * len(s_mem) + [0] * len(s_non))
                y_score = np.concatenate([s_mem, s_non])

                auc = float(roc_auc_score(y_true, y_score))
                fpr, tpr, _ = roc_curve(y_true, y_score)

                # if inverted, flip scores (only affects labeling direction)
                if auc < 0.5:
                    y_score = -y_score
                    auc = float(roc_auc_score(y_true, y_score))
                    fpr, tpr, _ = roc_curve(y_true, y_score)

                tpr1 = compute_tpr_at_fpr(fpr, tpr, 0.01)

                # accuracy with threshold 0
                y_pred = (y_score > 0).astype(int)
                acc = float((y_pred == y_true).mean())

                roc_png = f"roc_noise_{ds_name}_{sanitize_name(ext)}.png"
                plt.figure(figsize=(5, 5))
                plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
                plt.plot([0, 1], [0, 1], "k--")
                plt.scatter([0.01], [tpr1], label=f"TPR@1%FPR={tpr1:.3f}")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title(f"Noise-only Cosine MIA\n{ds_name} | {ext} | sigma={args.sigma}")
                plt.legend(loc="lower right")
                plt.tight_layout()
                plt.savefig(roc_png, dpi=150)
                plt.close()

                writer.writerow({
                    "dataset": ds_name,
                    "extractor": ext,
                    "n_fit": args.n_fit,
                    "n_eval": args.n_eval,
                    "sigma": round(float(args.sigma), 6),
                    "accuracy": round(acc, 6),
                    "auc": round(auc, 6),
                    "tpr_at_1fpr": round(float(tpr1), 6),
                    "roc_png": roc_png,
                })

                # free GPU mem
                del tok, enc
                torch.cuda.empty_cache()

                print(f"[RESULT] acc={acc:.4f} auc={auc:.4f} tpr@1%fpr={tpr1:.4f} saved {roc_png}", flush=True)

    print(f"\nAll done. Results saved to: {args.csv_out}", flush=True)


if __name__ == "__main__":
    main()

