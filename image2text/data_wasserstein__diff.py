import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from transformers import AutoTokenizer, AutoModel
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
import torch
import matplotlib.pyplot as plt
import ot  # POT (Python Optimal Transport) library


# Global model initialization (load only once to avoid duplicate memory occupation)
class TextEncoder:
    def __init__(self, model_name='/root/autodl-tmp/all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.encoder_type = None
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.init_model()

    def init_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.model.to(self.device)
            self.encoder_type = 'sentence'
        except ImportError:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.encoder_type = 'transformers'

    def encode(self, texts, batch_size=16):
        if not texts:
            return np.array([])

        torch.cuda.empty_cache()

        if self.encoder_type == 'sentence':
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_tensor=True,
                show_progress_bar=False
            )
            return embeddings.cpu().numpy()
        else:
            embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    batch_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    embeddings.extend(batch_emb)

                torch.cuda.empty_cache()

            return np.array(embeddings)


# Load data
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


# Extract vlm_1 and ground truth answers
def extract_answers(data):
    vlm_1_answers = []
    ground_truth_answers = []

    for item in data:
        conversations = item.get('conversations_0.1', [])
        for i in range(len(conversations)):
            if conversations[i].get('from') == 'vlm_1':
                vlm_1_answers.append(conversations[i].get('value', ''))
            elif conversations[i].get('from') == 'ground truth':
                ground_truth_answers.append(conversations[i].get('value', ''))

    return vlm_1_answers, ground_truth_answers


# Calculate Wasserstein distance
def wasserstein_distance_matrix(a, b):
    if a.shape[0] == 0 or b.shape[0] == 0:
        return float('inf')

    if a.shape[0] < 2 or b.shape[0] < 2:
        mean_a = np.mean(a, axis=0) if a.shape[0] > 0 else np.zeros(b.shape[1])
        mean_b = np.mean(b, axis=0) if b.shape[0] > 0 else np.zeros(a.shape[1])
        return np.linalg.norm(mean_a - mean_b)

    pca = PCA(n_components=1)
    all_embeddings = np.vstack([a, b])
    projected = pca.fit_transform(all_embeddings)

    proj_a = projected[:a.shape[0]].flatten()
    proj_b = projected[a.shape[0]:].flatten()

    return wasserstein_distance(proj_a, proj_b)


def wasserstein_distance_single(vlm_emb, vlm_mean_emb, gt_mean_emb):
    vlm_single_dist = np.tile(vlm_emb.reshape(1, -1), (5, 1))
    vlm_center_dist = np.tile(vlm_mean_emb.reshape(1, -1), (5, 1))
    gt_center_dist = np.tile(gt_mean_emb.reshape(1, -1), (5, 1))

    dist_to_vlm = wasserstein_distance_matrix(vlm_single_dist, vlm_center_dist)
    dist_to_gt = wasserstein_distance_matrix(vlm_single_dist, gt_center_dist)

    return dist_to_vlm, dist_to_gt


# Main analysis function (with multi-threshold processing)
def analyze_data(vlm_1_answers, ground_truth_answers, encoder):
    vlm_train, vlm_test = train_test_split(vlm_1_answers, test_size=0.1, random_state=42)
    gt_train, gt_test = train_test_split(ground_truth_answers, test_size=0.1, random_state=42)

    print(f"VLM_1 training set size: {len(vlm_train)}, test set size: {len(vlm_test)}")
    print(f"Ground Truth training set size: {len(gt_train)}, test set size: {len(gt_test)}")

    print("Encoding VLM_1 training data...")
    vlm_train_embeddings = encoder.encode(vlm_train, batch_size=16)

    print("Encoding Ground Truth training data...")
    gt_train_embeddings = encoder.encode(gt_train, batch_size=16)

    print("Encoding VLM_1 test data...")
    vlm_test_embeddings = encoder.encode(vlm_test, batch_size=16)

    print("Encoding Ground Truth test data...")
    gt_test_embeddings = encoder.encode(gt_test, batch_size=16)

    # Calculate distribution centers (means)
    vlm_mean = np.mean(vlm_train_embeddings, axis=0)
    gt_mean = np.mean(gt_train_embeddings, axis=0)

    print("Calculating Wasserstein distance...")
    # Calculate distances from test samples to both centers
    vlm_to_vlm = []
    vlm_to_gt = []
    for emb in vlm_test_embeddings:
        dist_vlm, dist_gt = wasserstein_distance_single(emb, vlm_mean, gt_mean)
        vlm_to_vlm.append(dist_vlm)
        vlm_to_gt.append(dist_gt)

    gt_to_vlm = []
    gt_to_gt = []
    for emb in gt_test_embeddings:
        dist_vlm, dist_gt = wasserstein_distance_single(emb, vlm_mean, gt_mean)
        gt_to_vlm.append(dist_vlm)
        gt_to_gt.append(dist_gt)

    # True labels (consistent with L2 version)
    true_labels = [1] * len(vlm_test) + [0] * len(gt_test)

    # Multi-threshold settings (consistent with L2 version)
    thresholds = [-0.01, -0.03, -0.05, -0.07, -0.1, 0, 0.01, 0.03, 0.05, 0.07, 0.1]
    final_auc = None
    final_tpr = None
    final_fpr = None
    final_tpr_curve = None

    for t in thresholds:
        print(f"\n===== Results using threshold {t} =====")

        # Threshold-based prediction logic (consistent with L2 version)
        vlm_preds = [1 if dist_vlm < (dist_gt + t) else 0
                     for dist_vlm, dist_gt in zip(vlm_to_vlm, vlm_to_gt)]
        gt_preds = [1 if dist_vlm < (dist_gt + t) else 0
                    for dist_vlm, dist_gt in zip(gt_to_vlm, gt_to_gt)]

        all_preds = vlm_preds + gt_preds

        # Calculate metrics
        accuracy = accuracy_score(true_labels, all_preds)
        print(f"Accuracy: {accuracy:.4f}")

        # Calculate scores (consistent with L2 version)
        scores = [(-d_gt) + d_vlm for d_vlm, d_gt in zip(vlm_to_vlm + gt_to_vlm,
                                                         vlm_to_gt + gt_to_gt)]
        auc = roc_auc_score(true_labels, scores)
        print(f"AUC: {auc:.4f}")

        # 计算TPR@FPR=1%
        fpr, tpr, _ = roc_curve(true_labels, scores)
        idx = np.argmin(np.abs(fpr - 0.01))
        tpr_at_fpr_1 = tpr[idx]
        print(f"TPR@FPR=1%: {tpr_at_fpr_1:.4f}")

        # Save results of the last threshold for plotting
        if t == thresholds[-1]:
            final_accuracy = accuracy
            final_auc = auc
            final_tpr = tpr_at_fpr_1
            final_fpr = fpr
            final_tpr_curve = tpr

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(final_fpr, final_tpr_curve, label=f'ROC Curve (AUC = {final_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('roc_curve.png')
    print("\nROC curve saved as roc_curve.png")

    return final_accuracy, final_auc, final_tpr


if __name__ == "__main__":
    try:
        import ot

        print("Using POT library to calculate Wasserstein distance")
    except ImportError:
        print("Error: Need to install Python Optimal Transport library")
        print("Please run: pip install POT")
        exit(1)

    encoder = TextEncoder()
    print(f"Using encoder type: {encoder.encoder_type}, device: {encoder.device}")

    data = load_data('/root/autodl-tmp/Apple_1100_items.json')

    vlm_1_answers, ground_truth_answers = extract_answers(data)
    vlm_1_answers = [ans for ans in vlm_1_answers if ans]
    ground_truth_answers = [ans for ans in ground_truth_answers if ans]

    print(f"VLM_1 answer count: {len(vlm_1_answers)}")
    print(f"Ground Truth answer count: {len(ground_truth_answers)}")

    if len(vlm_1_answers) < 2 or len(ground_truth_answers) < 2:
        print("Error: Insufficient data for analysis")
        exit(1)

    accuracy, auc, tpr_at_fpr_1 = analyze_data(vlm_1_answers, ground_truth_answers, encoder)

    print("\n===== Final Results =====")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"TPR@FPR=1%: {tpr_at_fpr_1:.4f}")