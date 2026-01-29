#!/usr/bin/env python3
import argparse, os, re, glob
import numpy as np
import matplotlib.pyplot as plt

def try_load_roc_npz(run_dir):
    """
    在 run_dir 下递归找含有 fpr/tpr 的 npz 文件，返回 (fpr, tpr) 或 None
    常见命名：*roc*.npz 或 attack_data_*.npz
    """
    cand = []
    cand += glob.glob(os.path.join(run_dir, "*.npz"))
    cand += glob.glob(os.path.join(run_dir, "**", "*.npz"), recursive=True)
    for p in cand:
        try:
            dat = np.load(p)
            keys = set(dat.files)
            # 常见字段名兼容
            for fpr_key in ["fpr", "roc_fpr", "FPR"]:
                for tpr_key in ["tpr", "roc_tpr", "TPR"]:
                    if fpr_key in keys and tpr_key in keys:
                        fpr = np.asarray(dat[fpr_key])
                        tpr = np.asarray(dat[tpr_key])
                        if fpr.ndim == 1 and tpr.ndim == 1 and len(fpr) == len(tpr):
                            return fpr, tpr, p
        except Exception:
            pass
    return None, None, None

def read_auc_from_log(log_path):
    """
    从 attack.py 日志抽取 AUC 数值
    例：
      "AUC on the target model: 0.504525"
    """
    auc = None
    if not os.path.exists(log_path):
        return auc
    pat = re.compile(r"AUC on the target model:\s*([0-9]*\.?[0-9]+)")
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pat.search(line)
            if m:
                try:
                    auc = float(m.group(1))
                except Exception:
                    pass
    return auc

def save_roc_curve(fpr, tpr, out_path, title=None):
    plt.figure(figsize=(5,5), dpi=150)
    plt.plot(fpr, tpr, linewidth=2, label="ROC")
    plt.plot([0,1],[0,1], linestyle="--", linewidth=1, label="Chance")
    plt.xlim([0,1]); plt.ylim([0,1])
    plt.xlabel("FPR"); plt.ylabel("TPR")
    if title:
        plt.title(title)
    plt.legend(loc="lower right")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def save_auc_placeholder(auc, out_path):
    plt.figure(figsize=(5,5), dpi=150)
    # 画一条对角线 + 标注 AUC
    plt.plot([0,1],[0,1], linestyle="--", linewidth=1, label="Chance")
    plt.xlim([0,1]); plt.ylim([0,1])
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"AUC = {auc:.4f}" if auc is not None else "AUC = N/A")
    plt.text(0.5, 0.1, "No ROC points found.\nShowing placeholder.", ha="center")
    plt.legend(loc="lower right")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="单次运行输出目录（含 run.log 或 npz）")
    args = ap.parse_args()

    # 1) 优先用 npz 的 fpr/tpr 画真 ROC
    fpr, tpr, npz_path = try_load_roc_npz(args.run_dir)
    if fpr is not None:
        # 计算 AUC 并存图
        try:
            from sklearn.metrics import auc as calc_auc
            auc_val = calc_auc(fpr, tpr)
        except Exception:
            # 手动积分
            auc_val = np.trapz(tpr, fpr)

        out_path = os.path.join(args.run_dir, "roc.png")
        save_roc_curve(fpr, tpr, out_path, title=f"ROC (AUC={auc_val:.4f})")
        print(f"[OK] ROC saved: {out_path} (src: {npz_path}, AUC={auc_val:.4f})")
        return

    # 2) 回退：从日志抓 AUC，画占位图
    log_path = os.path.join(args.run_dir, "run.log")
    auc = read_auc_from_log(log_path)
    out_path = os.path.join(args.run_dir, "auc_placeholder.png")
    save_auc_placeholder(auc, out_path)
    print(f"[WARN] No ROC npz found. Placeholder saved: {out_path} (AUC={auc})")

if __name__ == "__main__":
    main()

