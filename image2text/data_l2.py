import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from transformers import AutoTokenizer, AutoModel
import torch
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


# Text preprocessing function
def preprocess_text(text):
    """
    Clean text, remove meaningless content
    :param text: Original text
    :return: Cleaned text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Remove </s> tokens
    text = text.replace("</s>", "")
    
    # Remove <s> tokens
    text = text.replace("<s>", "")
    
    # Remove special tokens like [PAD], [CLS], [SEP]
    special_tokens = ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]", "<pad>", "<unk>", "<bos>", "<eos>"]
    for token in special_tokens:
        text = text.replace(token, "")
    
    # Remove extra whitespace characters
    text = " ".join(text.split())
    
    # If text is too short, it might be meaningless, return empty string
    if len(text.strip()) < 3:
        return ""
    
    return text.strip()


def preprocess_texts(texts):
    """
    Batch preprocess text list
    :param texts: List of texts
    :return: Cleaned text list
    """
    return [preprocess_text(text) for text in texts]


# Calculate L_2 distance (Euclidean distance)
def l2_distance(a, b):
    return np.linalg.norm(a - b)


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

    vlm_mean = np.mean(vlm_train_embeddings, axis=0)
    gt_mean = np.mean(gt_train_embeddings, axis=0)

    print("Encoding VLM_1 test data...")
    vlm_test_embeddings = encoder.encode(vlm_test, batch_size=16)

    print("Encoding Ground Truth test data...")
    gt_test_embeddings = encoder.encode(gt_test, batch_size=16)

    # Subsequent logic remains unchanged...
    vlm_to_vlm = [l2_distance(emb, vlm_mean) for emb in vlm_test_embeddings]
    vlm_to_gt = [l2_distance(emb, gt_mean) for emb in vlm_test_embeddings]

    gt_to_vlm = [l2_distance(emb, vlm_mean) for emb in gt_test_embeddings]
    gt_to_gt = [l2_distance(emb, gt_mean) for emb in gt_test_embeddings]

    # Note: Smaller L_2 distance indicates higher similarity, so comparison logic is opposite to cosine similarity
    vlm_predictions = [1 if dist_vlm < dist_gt else 0 for dist_vlm, dist_gt in zip(vlm_to_vlm, vlm_to_gt)]
    gt_predictions = [1 if dist_vlm < dist_gt else 0 for dist_vlm, dist_gt in zip(gt_to_vlm, gt_to_gt)]

    vlm_true = [1] * len(vlm_test)
    gt_true = [0] * len(gt_test)

    all_predictions = vlm_predictions + gt_predictions
    all_true = vlm_true + gt_true

    accuracy = accuracy_score(all_true, all_predictions)
    print(f"Accuracy: {accuracy:.4f}")

    scores_1 = [dist for dist in vlm_to_gt] + [dist for dist in gt_to_gt]
    auc1 = roc_auc_score(all_true, scores_1)
    print(f"AUC_1: {auc1:.4f}")

    scores_2 = [-dist for dist in vlm_to_vlm] + [-dist for dist in gt_to_vlm]
    auc2 = roc_auc_score(all_true, scores_2)
    print(f"AUC_2: {auc2:.4f}")

    scores = [s1 + s2 for s1, s2 in zip(scores_1, scores_2)]
    auc = roc_auc_score(all_true, scores)
    print(f"AUC_3: {auc:.4f}")

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

    # Plot ROC curve
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
    # Initialize encoder (globally unique instance)
    encoder = TextEncoder()
    print(f"Using encoder type: {encoder.encoder_type}, device: {encoder.device}")

    # Load data
    data = load_data('/root/autodl-tmp/gpt_3000.json')

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

    # Analyze data
    accuracy, auc, tpr_at_fpr_1 = analyze_data(vlm_1_answers, ground_truth_answers, encoder)

    # Output final results
    print("===== Final Results =====")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"TPR@FPR=1%: {tpr_at_fpr_1:.4f}")
