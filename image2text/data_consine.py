import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from transformers import AutoTokenizer, AutoModel
import torch
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import random
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords  # New: for stopwords filtering

# Load stopwords
stop_words = set(stopwords.words('english'))


# Global model initialization (load only once to avoid repeated memory usage)
class TextEncoder:
    def __init__(self, model_name='/root/autodl-tmp/bert-base-uncased'):
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

    def encode(self, texts, batch_size=16):  # Reduce batch size according to GPU memory
        if not texts:
            return np.array([])

        # Manually clean up memory
        torch.cuda.empty_cache()

        if self.encoder_type == 'sentence':
            # Control sentence-transformers batch size
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_tensor=True,
                show_progress_bar=False
            )
            return embeddings.cpu().numpy()
        else:
            # Batch process texts (replace loop-by-loop processing)
            embeddings = []
            # Process in batches to avoid single input being too large
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
                    # Take [CLS] vector
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
    Clean text and remove meaningless content
    :param text: Raw text
    :return: Cleaned text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Remove </s> tag
    text = text.replace("</s>", "")
    
    # Remove <s> tag
    text = text.replace("<s>", "")
    
    # Remove [PAD], [CLS], [SEP] and other special tokens
    special_tokens = ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]", "<pad>", "<unk>", "<bos>", "<eos>"]
    for token in special_tokens:
        text = text.replace(token, "")
    
    # Remove extra whitespace
    text = " ".join(text.split())
    
    # If text is too short, may be meaningless, return empty string
    if len(text.strip()) < 3:
        return ""
    
    # Filter stopwords (keep non-stopwords)
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    
    # If no valid words after filtering, return empty string
    if not filtered_words:
        return ""
        
    return " ".join(filtered_words)


def preprocess_texts(texts):
    """
    Batch preprocess text list
    :param texts: Text list
    :return: Cleaned text list
    """
    return [preprocess_text(text) for text in texts]


# Calculate cosine similarity
def cosine_similarity(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0
    return np.dot(a, b) / (norm_a * norm_b)


# Main analysis function (receive encoder instance to avoid repeated initialization)
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

    vlm_mean = np.mean(vlm_train_embeddings, axis=0)
    gt_mean = np.mean(gt_train_embeddings, axis=0)

    print("Encoding VLM_1 test data...")
    vlm_test_embeddings = encoder.encode(vlm_test, batch_size=16)

    print("Encoding Ground Truth test data...")
    gt_test_embeddings = encoder.encode(gt_test, batch_size=16)

    # Following logic remains unchanged...
    vlm_to_vlm = [cosine_similarity(emb, vlm_mean) for emb in vlm_test_embeddings]
    vlm_to_gt = [cosine_similarity(emb, gt_mean) for emb in vlm_test_embeddings]

    gt_to_vlm = [cosine_similarity(emb, vlm_mean) for emb in gt_test_embeddings]
    gt_to_gt = [cosine_similarity(emb, gt_mean) for emb in gt_test_embeddings]

    vlm_predictions = [1 if sim_vlm > sim_gt else 0 for sim_vlm, sim_gt in zip(vlm_to_vlm, vlm_to_gt)]
    gt_predictions = [1 if sim_vlm > sim_gt else 0 for sim_vlm, sim_gt in zip(gt_to_vlm, gt_to_gt)]

    vlm_true = [1] * len(vlm_test)
    gt_true = [0] * len(gt_test)

    all_predictions = vlm_predictions + gt_predictions
    all_true = vlm_true + gt_true

    accuracy = accuracy_score(all_true, all_predictions)
    print(f"Accuracy: {accuracy:.4f}")

    scores_1 = [-sim for sim in vlm_to_gt] + [-sim for sim in gt_to_gt]
    auc1 = roc_auc_score(all_true, scores_1)
    print(f"AUC_1: {auc1:.4f}")

    scores_2 = [sim for sim in vlm_to_vlm] + [sim for sim in gt_to_vlm]
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
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
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
    # Initialize encoder (global unique instance)
    encoder = TextEncoder()
    print(f"Encoder type: {encoder.encoder_type}, Device: {encoder.device}")

    # Load data
    data = load_data('/root/autodl-tmp/gpt_3000.json')

    # Extract answers
    vlm_1_answers, ground_truth_answers = extract_answers(data)
    
    # Preprocess text
    print("Preprocessing VLM_1 answers...")
    vlm_1_answers = preprocess_texts(vlm_1_answers)
    print("Preprocessing Ground Truth answers...")
    ground_truth_answers = preprocess_texts(ground_truth_answers)
    
    # Filter empty values
    vlm_1_answers = [ans for ans in vlm_1_answers if ans]
    ground_truth_answers = [ans for ans in ground_truth_answers if ans]

    print(f"Number of VLM_1 answers: {len(vlm_1_answers)}")
    print(f"Number of Ground Truth answers: {len(ground_truth_answers)}")

    # Analyze data
    accuracy, auc, tpr_at_fpr_1 = analyze_data(vlm_1_answers, ground_truth_answers, encoder)

    # Output final results
    print("===== Final Results =====")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"TPR@FPR=1%: {tpr_at_fpr_1:.4f}")