import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from transformers import AutoTokenizer, AutoModel
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
import torch
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import ot  # POT (Python Optimal Transport) library


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

        # Manually clean up memory
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
            # Batch process texts (instead of processing one by one)
            embeddings = []
            # Process in batches to avoid oversized single inputs
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

                # Clean up memory after each batch
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


# Text preprocessing function
def preprocess_text(text):
    """
    Clean text, remove meaningless content
    :param text: Original text
    :return: Cleaned text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Remove </s> tags
    text = text.replace("</s>", "")
    
    # Remove <s> tags
    text = text.replace("<s>", "")
    
    # Remove special tokens like [PAD], [CLS], [SEP], etc.
    special_tokens = ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]", "<pad>", "<unk>", "<bos>", "<eos>"]
    for token in special_tokens:
        text = text.replace(token, "")
    
    # Remove extra whitespace characters
    text = " ".join(text.split())
    
    # If text is too short, it may be meaningless, return empty string
    if len(text.strip()) < 3:
        return ""
    
    return text.strip()


def preprocess_texts(texts):
    """
    Preprocess a list of texts in batch
    :param texts: List of texts
    :return: List of cleaned texts
    """
    return [preprocess_text(text) for text in texts]


# Calculate Wasserstein distance
def wasserstein_distance_matrix(a, b):
    """Calculate Wasserstein distance between two embedding matrices"""
    # To use wasserstein_distance, we need to project high-dimensional embeddings to 1D
    # Use PCA to project high-dimensional vectors to 1D
    if a.shape[0] == 0 or b.shape[0] == 0:
        return float('inf')

    # If there are too few samples, direct calculation will cause errors, at least 2 samples are needed
    if a.shape[0] < 2 or b.shape[0] < 2:
        # Use mean vector to calculate Euclidean distance as approximation
        mean_a = np.mean(a, axis=0) if a.shape[0] > 0 else np.zeros(b.shape[1])
        mean_b = np.mean(b, axis=0) if b.shape[0] > 0 else np.zeros(a.shape[1])
        return np.linalg.norm(mean_a - mean_b)

    # Project embedding vectors to 1D
    pca = PCA(n_components=1)
    all_embeddings = np.vstack([a, b])
    projected = pca.fit_transform(all_embeddings)

    # Separate projected vectors
    proj_a = projected[:a.shape[0]].flatten()
    proj_b = projected[a.shape[0]:].flatten()

    # Calculate Wasserstein distance
    return wasserstein_distance(proj_a, proj_b)

def wasserstein_distance_single(vlm_emb, vlm_mean_emb, gt_mean_emb):
    """Calculate Wasserstein distance between a single vector and two distribution centers"""
    # Create distribution for single vector (duplicate multiple times to form distribution)
    vlm_single_dist = np.tile(vlm_emb.reshape(1, -1), (5, 1))  # Duplicate 5 times
    vlm_center_dist = np.tile(vlm_mean_emb.reshape(1, -1), (5, 1))
    gt_center_dist = np.tile(gt_mean_emb.reshape(1, -1), (5, 1))

    # Calculate distances
    dist_to_vlm = wasserstein_distance_matrix(vlm_single_dist, vlm_center_dist)
    dist_to_gt = wasserstein_distance_matrix(vlm_single_dist, gt_center_dist)

    return dist_to_vlm, dist_to_gt


# Main analysis function (accepts encoder instance to avoid duplicate initialization)
def analyze_data(vlm_1_answers, ground_truth_answers, encoder):
    vlm_train, vlm_test = train_test_split(vlm_1_answers, test_size=0.1, random_state=42)
    gt_train, gt_test = train_test_split(ground_truth_answers, test_size=0.1, random_state=42)

    print(f"VLM_1 training set size: {len(vlm_train)}, test set size: {len(vlm_test)}")
    print(f"Ground Truth training set size: {len(gt_train)}, test set size: {len(gt_test)}")

    # Specify batch size during encoding (adjust according to GPU memory, e.g., 8/16/32)
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
    # Distance between vlm_test and vlm_train distributions
    print("Calculating Wasserstein distances...")
    # Distance from VLM test samples to two centers
    vlm_to_vlm = []
    vlm_to_gt = []
    for emb in vlm_test_embeddings:
        dist_vlm, dist_gt = wasserstein_distance_single(emb, vlm_mean, gt_mean)
        vlm_to_vlm.append(dist_vlm)
        vlm_to_gt.append(dist_gt)

    # Distance from GT test samples to two centers
    gt_to_vlm = []
    gt_to_gt = []
    for emb in gt_test_embeddings:
        dist_vlm, dist_gt = wasserstein_distance_single(emb, vlm_mean, gt_mean)
        gt_to_vlm.append(dist_vlm)
        gt_to_gt.append(dist_gt)

    # Make predictions based on Wasserstein distances
    # Smaller distance indicates higher similarity, so predict the class with smaller distance
    vlm_predictions = [0 if dist_vlm < dist_gt else 1 for dist_vlm, dist_gt in zip(vlm_to_vlm, vlm_to_gt)]
    gt_predictions = [0 if dist_vlm < dist_gt else 1 for dist_vlm, dist_gt in zip(gt_to_vlm, gt_to_gt)]

    vlm_true = [0] * len(vlm_test)  # VLM_1 belongs to class 0
    gt_true = [1] * len(gt_test)  # Ground Truth belongs to class 1

    all_predictions = vlm_predictions + gt_predictions
    all_true = vlm_true + gt_true

    accuracy = accuracy_score(all_true, all_predictions)
    print(f"Accuracy: {accuracy:.4f}")

    # For AUC calculation, use negative Wasserstein distance as scores (smaller distance is better)
    scores_1 = [-dist for dist in vlm_to_vlm] + [-dist for dist in gt_to_vlm]  # Distance to VLM
    auc1 = roc_auc_score(all_true, scores_1)
    print(f"AUC_1 (based on distance to VLM): {auc1:.4f}")

    scores_2 = [-dist for dist in vlm_to_gt] + [-dist for dist in gt_to_gt]  # Distance to GT
    auc2 = roc_auc_score(all_true, scores_2)
    print(f"AUC_2 (based on distance to GT): {auc2:.4f}")

    # Combined scores: negative VLM distance - negative GT distance = GT distance - VLM distance
    # In this way, when GT distance < VLM distance, the score is positive, predict GT
    scores = [s2 - s1 for s1, s2 in zip(scores_1, scores_2)]
    auc = roc_auc_score(all_true, scores)
    print(f"AUC_3 (combined scores): {auc:.4f}")

    # Calculate TPR@FPR=1%
    fpr, tpr, _ = roc_curve(all_true, scores_1)
    idx = np.argmin(np.abs(fpr - 0.01))
    tpr_at_fpr_1_1 = tpr[idx]
    print(f"TPR@FPR=1%(1): {tpr_at_fpr_1_1:.4f}")

    fpr, tpr, _ = roc_curve(all_true, scores_2)
    idx = np.argmin(np.abs(fpr - 0.01))
    tpr_at_fpr_1_2 = tpr[idx]
    print(f"TPR@FPR=1%(2): {tpr_at_fpr_1_2:.4f}")

    fpr, tpr, _ = roc_curve(all_true, scores)
    idx = np.argmin(np.abs(fpr - 0.01))
    tpr_at_fpr_1 = tpr[idx]
    print(f"TPR@FPR=1%: {tpr_at_fpr_1:.4f}")

    # Draw ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('roc_curve.png')
    print("ROC curve saved as roc_curve.png")

    return accuracy, auc, tpr_at_fpr_1


if __name__ == "__main__":
    # Check if ot library is installed
    try:
        import ot

        print("Using POT library to calculate Wasserstein distance")
    except ImportError:
        print("Error: Need to install Python Optimal Transport library")
        print("Please run: pip install POT")
        exit(1)

    # Initialize encoder (global unique instance)
    encoder = TextEncoder()
    print(f"Using encoder type: {encoder.encoder_type}, device: {encoder.device}")

    # Load data
    data = load_data('/root/autodl-tmp/gpt_2000.json')

    # Extract answers
    vlm_1_answers, ground_truth_answers = extract_answers(data)
    
    # Preprocess text
    print("Preprocessing VLM_1 answers...")
    vlm_1_answers = preprocess_texts(vlm_1_answers)
    print("Preprocessing Ground Truth answers...")
    ground_truth_answers = preprocess_texts(ground_truth_answers)
    
    # Filter out empty values
    vlm_1_answers = [ans for ans in vlm_1_answers if ans]
    ground_truth_answers = [ans for ans in ground_truth_answers if ans]

    print(f"VLM_1 answer count: {len(vlm_1_answers)}")
    print(f"Ground Truth answer count: {len(ground_truth_answers)}")

    # Check if there is enough data
    if len(vlm_1_answers) < 2 or len(ground_truth_answers) < 2:
        print("Error: Insufficient data for analysis")
        exit(1)

    # Analyze data
    accuracy, auc, tpr_at_fpr_1 = analyze_data(vlm_1_answers, ground_truth_answers, encoder)

    # Output final results
    print("===== Final Results =====")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"TPR@FPR=1%: {tpr_at_fpr_1:.4f}")

