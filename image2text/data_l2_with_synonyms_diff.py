
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from transformers import AutoTokenizer, AutoModel
import torch
# Replace cosine distance with L2 distance (Euclidean distance)
from scipy.spatial.distance import euclidean  # L2 distance
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import wordnet
import random

# Download required NLTK data
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')


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
                    outputs = self.model(** inputs)
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


# Function to get synonyms
def get_synonyms(word):
    """Get synonyms for a word"""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if lemma.name() != word:  # Exclude original word
                # Replace underscores with spaces (handle multi-word synonyms)
                synonym = lemma.name().replace('_', ' ')
                synonyms.add(synonym)
    return list(synonyms)


# Perform synonym replacement on text
def replace_with_synonyms(text, replace_ratio=0.2):
    """
    Perform synonym replacement on words in the text
    :param text: Input text
    :param replace_ratio: Replacement ratio (0-1)
    :return: Text after replacement
    """
    if not text or not isinstance(text, str):
        return text

    # Tokenize
    words = text.split()
    if not words:
        return text

    # Calculate the number of words to replace
    num_replace = max(1, int(len(words) * replace_ratio))

    # Randomly select word indices to replace
    replace_indices = random.sample(range(len(words)), min(num_replace, len(words)))

    # Attempt replacement for each selected word
    for idx in replace_indices:
        word = words[idx]
        # Get synonyms
        synonyms = get_synonyms(word)
        if synonyms:
            # Randomly select a synonym for replacement
            words[idx] = random.choice(synonyms)

    return ' '.join(words)


# Perform synonym replacement on vlm_1 test set
def replace_synonyms_in_test_set(vlm_test, replace_ratio=0.2):
    """
    Perform synonym replacement on vlm_1 test set
    :param vlm_test: List of vlm_1 test set
    :param replace_ratio: Replacement ratio
    :return: Replaced vlm_1 list
    """
    replaced_vlm_test = []
    for text in vlm_test:
        replaced_text = replace_with_synonyms(text, replace_ratio)
        replaced_vlm_test.append(replaced_text)
    return replaced_vlm_test


# Main analysis function (accepts encoder instance to avoid duplicate initialization)
def analyze_data(vlm_1_answers, ground_truth_answers, encoder):
    vlm_train, vlm_test = train_test_split(vlm_1_answers, test_size=0.1, random_state=42)
    gt_train, gt_test = train_test_split(ground_truth_answers, test_size=0.1, random_state=42)

    print(f"VLM_1 training set size: {len(vlm_train)}, test set size: {len(vlm_test)}")
    print(f"Ground Truth training set size: {len(gt_train)}, test set size: {len(gt_test)}")

    # Perform synonym replacement on vlm_1 test set (20% replacement ratio)
    print("Performing synonym replacement on vlm_1 test set...")
    vlm_test = replace_synonyms_in_test_set(vlm_test, replace_ratio=0.2)
    print("Synonym replacement completed")

    # Specify batch size during encoding (adjust according to GPU memory, e.g., 8/16/32)
    print("Encoding VLM_1 training data...")
    vlm_train_embeddings = encoder.encode(vlm_train, batch_size=16)

    print("Encoding Ground Truth training data...")
    gt_train_embeddings = encoder.encode(gt_train, batch_size=16)

    # Calculate mean vectors of training set (for subsequent distance calculations)
    vlm_mean = np.mean(vlm_train_embeddings, axis=0)
    gt_mean = np.mean(gt_train_embeddings, axis=0)

    print("Encoding VLM_1 test data...")
    vlm_test_embeddings = encoder.encode(vlm_test, batch_size=16)

    print("Encoding Ground Truth test data...")
    gt_test_embeddings = encoder.encode(gt_test, batch_size=16)

    # Calculate L2 distance (Euclidean distance)
    # Distance from samples to vlm center: vlm_test samples -> vlm_mean; gt_test samples -> vlm_mean
    vlm_to_vlm_dist = [euclidean(emb, vlm_mean) for emb in vlm_test_embeddings]
    gt_to_vlm_dist = [euclidean(emb, vlm_mean) for emb in gt_test_embeddings]

    # Distance from samples to gt center: vlm_test samples -> gt_mean; gt_test samples -> gt_mean
    vlm_to_gt_dist = [euclidean(emb, gt_mean) for emb in vlm_test_embeddings]
    gt_to_gt_dist = [euclidean(emb, gt_mean) for emb in gt_test_embeddings]

    # Unified true labels (1: vlm samples, 0: gt samples)
    true_labels = [1] * len(vlm_test) + [0] * len(gt_test)

    # Define all thresholds to be tested (distance thresholds, opposite logic to original similarity thresholds)
    # L2 distance: smaller distance means higher similarity, so threshold logic needs adjustment
    thresholds = [0.01, 0.03, 0.05, 0.07, 0.1, 0, -0.01, -0.03, -0.05, -0.07, -0.1]
    final_auc = None
    final_tpr = None

    for t in thresholds:
        print(f"\n===== Results using threshold {t} =====")

        # Calculate prediction results: distance judgment logic
        # For vlm test samples: if distance to vlm < distance to gt + t → classified as 1 (member)
        vlm_preds = [1 if dist_vlm < (dist_gt + t) else 0
                     for dist_vlm, dist_gt in zip(vlm_to_vlm_dist, vlm_to_gt_dist)]
        # For gt test samples: same logic (theoretically should be classified as 0)
        gt_preds = [1 if dist_vlm < (dist_gt + t) else 0
                    for dist_vlm, dist_gt in zip(gt_to_vlm_dist, gt_to_gt_dist)]

        all_preds = vlm_preds + gt_preds

        # Calculate accuracy
        accuracy = accuracy_score(true_labels, all_preds)
        print(f"Accuracy: {accuracy:.4f}")

        # Calculate AUC (based on distance-designed scores: higher scores indicate more likely positive samples)
        # Positive samples (vlm) should be closer to vlm center and farther from gt center → score = distance to gt - distance to vlm
        scores = [dist_gt - dist_vlm for dist_vlm, dist_gt in zip(vlm_to_vlm_dist + gt_to_vlm_dist,
                                                                vlm_to_gt_dist + gt_to_gt_dist)]
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
    plt.title('ROC Curve (L2 Distance, vlm_1 with synonym replacement)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('roc_curve_l2_with_synonyms.png')
    print("\nROC curve saved as roc_curve_l2_with_synonyms.png")

    return accuracy, auc, tpr_at_fpr_1


if __name__ == "__main__":
    # Initialize encoder (global unique instance)
    encoder = TextEncoder()
    print(f"Using encoder type: {encoder.encoder_type}, device: {encoder.device}")

    # Load data
    data = load_data('/root/autodl-tmp/llava_3000.json')

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
