import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from transformers import AutoTokenizer, AutoModel
import torch
from scipy.spatial.distance import cosine  # cosine returns distance, 1-distance is similarity
import matplotlib.pyplot as plt


# Global model initialization (load only once to avoid repeated memory occupation)
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

    def encode(self, texts, batch_size=16):  # Reduce batch size, adjust according to GPU memory
        if not texts:
            return np.array([])

        # Manually clear memory
        torch.cuda.empty_cache()

        if self.encoder_type == 'sentence':
            # Control batch size for sentence-transformers
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_tensor=True,
                show_progress_bar=False
            )
            return embeddings.cpu().numpy()
        else:
            # Batch processing of text (instead of processing one by one in loop)
            embeddings = []
            # Process in batches to avoid oversized single input
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)

                with torch.no_grad():  # Disable gradient calculation to save memory
                    outputs = self.model(**inputs)
                    # Get [CLS] vector
                    batch_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    embeddings.extend(batch_emb)

                # Clear memory after each batch processing
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


# Main analysis function (receives encoder instance to avoid duplicate initialization)
def analyze_data(vlm_1_answers, ground_truth_answers, encoder):
    vlm_train, vlm_test = train_test_split(vlm_1_answers, test_size=0.1, random_state=42)
    gt_train, gt_test = train_test_split(ground_truth_answers, test_size=0.1, random_state=42)

    print(f"VLM_1 training set size: {len(vlm_train)}, test set size: {len(vlm_test)}")
    print(f"Ground Truth training set size: {len(gt_train)}, test set size: {len(gt_test)}")

    # Specify batch size during encoding (adjust according to GPU memory, such as 8/16/32)
    print("Encoding VLM_1 training data...")
    vlm_train_embeddings = encoder.encode(vlm_train, batch_size=16)

    print("Encoding Ground Truth training data...")
    gt_train_embeddings = encoder.encode(gt_train, batch_size=16)

    # Calculate mean vectors of training set (for subsequent similarity calculation)
    vlm_mean = np.mean(vlm_train_embeddings, axis=0)
    gt_mean = np.mean(gt_train_embeddings, axis=0)

    print("Encoding VLM_1 test data...")
    vlm_test_embeddings = encoder.encode(vlm_test, batch_size=16)

    print("Encoding Ground Truth test data...")
    gt_test_embeddings = encoder.encode(gt_test, batch_size=16)

    # Calculate cosine similarity (1 - cosine distance)
    # Sample to vlm center similarity: vlm_test samples -> vlm_mean; gt_test samples -> vlm_mean
    vlm_to_vlm_sim = [1 - cosine(emb, vlm_mean) for emb in vlm_test_embeddings]
    gt_to_vlm_sim = [1 - cosine(emb, vlm_mean) for emb in gt_test_embeddings]

    # Sample to gt center similarity: vlm_test samples -> gt_mean; gt_test samples -> gt_mean
    vlm_to_gt_sim = [1 - cosine(emb, gt_mean) for emb in vlm_test_embeddings]
    gt_to_gt_sim = [1 - cosine(emb, gt_mean) for emb in gt_test_embeddings]

    # Unified true labels (1: vlm samples, 0: gt samples)
    true_labels = [1] * len(vlm_test) + [0] * len(gt_test)

    # Define all thresholds to be tested (similarity thresholds, corresponding to original distance threshold logic)
    thresholds = [0.01, 0.03, 0.05, 0.07, 0.1, 0, -0.01, -0.03, -0.05, -0.07, -0.1]
    final_auc = None
    final_tpr = None

    for t in thresholds:
        print(f"\n===== Results using threshold {t} =====")

        # Calculate prediction results: similarity determination logic
        # For vlm test samples: if similarity to vlm > similarity to gt + t → classified as 1 (member)
        vlm_preds = [1 if sim_vlm > (sim_gt + t) else 0
                     for sim_vlm, sim_gt in zip(vlm_to_vlm_sim, vlm_to_gt_sim)]
        # For gt test samples: same logic (should theoretically be classified as 0)
        gt_preds = [1 if sim_vlm > (sim_gt + t) else 0
                    for sim_vlm, sim_gt in zip(gt_to_vlm_sim, gt_to_gt_sim)]

        all_preds = vlm_preds + gt_preds

        # Calculate accuracy
        accuracy = accuracy_score(true_labels, all_preds)
        print(f"Accuracy: {accuracy:.4f}")

        # Calculate AUC (based on similarity designed score: higher score means more likely positive sample)
        # Positive samples (vlm) should be closer to vlm center and farther from gt center → score = similarity to vlm - similarity to gt
        scores = [sim_vlm - sim_gt for sim_vlm, sim_gt in zip(vlm_to_vlm_sim + gt_to_vlm_sim,
                                                             vlm_to_gt_sim + gt_to_gt_sim)]
        auc = roc_auc_score(true_labels, scores)
        print(f"AUC: {auc:.4f}")

        # Calculate TPR@FPR=1%
        fpr, tpr, _ = roc_curve(true_labels, scores)
        idx = np.argmin(np.abs(fpr - 0.01))  # Find the point closest to 1% FPR
        tpr_at_fpr_1 = tpr[idx]
        print(f"TPR@FPR=1%: {tpr_at_fpr_1:.4f}")

        # Record metrics for the last threshold for plotting
        if t == thresholds[-1]:
            final_auc = auc
            final_tpr = tpr_at_fpr_1
            final_fpr = fpr
            final_tpr_curve = tpr

    # Plot ROC curve (using results from the last threshold)
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

    return accuracy, final_auc, final_tpr


if __name__ == "__main__":
    # Initialize encoder (globally unique instance)
    encoder = TextEncoder()
    print(f"Using encoder type: {encoder.encoder_type}, device: {encoder.device}")

    # Load data
    data = load_data('/root/autodl-tmp/aPPLE-2100.json')

    # Extract answers
    vlm_1_answers, ground_truth_answers = extract_answers(data)
    vlm_1_answers = [ans for ans in vlm_1_answers if ans]  # Filter empty answers
    ground_truth_answers = [ans for ans in ground_truth_answers if ans]

    print(f"VLM_1 answer count: {len(vlm_1_answers)}")
    print(f"Ground Truth answer count: {len(ground_truth_answers)}")

    # Analyze data
    accuracy, auc, tpr_at_fpr_1 = analyze_data(vlm_1_answers, ground_truth_answers, encoder)

    # Output final results
    print("\n===== Final Results =====")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"TPR@FPR=1%: {tpr_at_fpr_1:.4f}")