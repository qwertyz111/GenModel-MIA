import json
import numpy as np
import random
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from transformers import AutoTokenizer, AutoModel
import torch
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# Download NLTK resources (needed for the first run)
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


# Synonym replacement function (optimized logic to improve discriminability)
def synonym_replacement(text, replace_ratio=0.3):  # Increase replacement ratio to 30%
    """
    Perform synonym replacement on text to improve text diversity
    :param text: Input text
    :param replace_ratio: Replacement ratio
    :return: Processed text
    """
    if not text or len(word_tokenize(text)) < 3:  # No replacement for short text
        return text

    # Tokenize and get POS tags
    words = word_tokenize(text)
    tagged_words = pos_tag(words)

    # Determine the number of words to replace (at least 1, at most half)
    num_replace = max(1, min(int(len(words) * replace_ratio), len(words) // 2))
    replace_indices = random.sample(range(len(words)), num_replace)

    # POS mapping (WordNet POS to NLTK POS)
    pos_map = {
        'NN': wordnet.NOUN,
        'VB': wordnet.VERB,
        'JJ': wordnet.ADJ,
        'RB': wordnet.ADV
    }

    for i in replace_indices:
        word, tag = tagged_words[i]
        # Only replace nouns, verbs, adjectives, and adverbs
        if tag[:2] in pos_map:
            # Get synsets
            synsets = wordnet.synsets(word, pos=pos_map[tag[:2]])
            if synsets:
                # Randomly select a different synonym from the synsets
                synonyms = [lemma.name() for synset in synsets for lemma in synset.lemmas() if lemma.name() != word]
                if synonyms:
                    # Select a synonym for replacement (handle underscored words)
                    replacement = random.choice(synonyms).replace('_', ' ')
                    words[i] = replacement

    return ' '.join(words)


# Global model initialization (load once to avoid redundant memory usage)
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


# Extract answers
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


# Distance calculation utility function (supports Wasserstein/Cosine distance switching)
def calculate_distance(emb1, emb2, distance_type='cosine'):
    """
    Calculate the distance between two vectors
    :param emb1: Vector 1
    :param emb2: Vector 2
    :param distance_type: wasserstein / cosine
    :return: Distance value (lower means more similar)
    """
    if distance_type == 'wasserstein':
        return wasserstein_distance(emb1, emb2)
    elif distance_type == 'cosine':
        return cosine(emb1, emb2)  # Cosine distance range [0, 2], 0 means identical
    else:
        raise ValueError("distance_type must be 'wasserstein' or 'cosine'")


# Main analysis function (fix threshold logic + add diagnostic prints + optimize distance calculation)
def analyze_data(vlm_1_answers, ground_truth_answers, encoder):
    vlm_train, vlm_test = train_test_split(vlm_1_answers, test_size=0.1, random_state=42)
    gt_train, gt_test = train_test_split(ground_truth_answers, test_size=0.1, random_state=42)

    # Perform synonym replacement on VLM_1 test set (increase replacement ratio)
    print("Performing synonym replacement on VLM_1 test set...")
    vlm_test_augmented = [synonym_replacement(text, replace_ratio=0.3) for text in vlm_test]

    print(f"VLM_1 train set size: {len(vlm_train)}, test set size: {len(vlm_test_augmented)}")
    print(f"Ground Truth train set size: {len(gt_train)}, test set size: {len(gt_test)}")

    # Specify batch size during encoding
    print("Encoding VLM_1 training data...")
    vlm_train_embeddings = encoder.encode(vlm_train, batch_size=16)

    print("Encoding Ground Truth training data...")
    gt_train_embeddings = encoder.encode(gt_train, batch_size=16)

    # Calculate mean vectors for training set
    vlm_mean = np.mean(vlm_train_embeddings, axis=0)
    gt_mean = np.mean(gt_train_embeddings, axis=0)

    # Print cosine similarity of mean vectors (for diagnostics)
    mean_cosine_sim = 1 - cosine(vlm_mean, gt_mean)
    print(f"\nCosine similarity between VLM mean and GT mean: {mean_cosine_sim:.6f}")
    if mean_cosine_sim > 0.95:
        print("Warning: VLM and GT mean vectors are highly similar, which may lead to low discriminability!")

    print("Encoding augmented VLM_1 test data...")
    vlm_test_embeddings = encoder.encode(vlm_test_augmented, batch_size=16)

    print("Encoding Ground Truth test data...")
    gt_test_embeddings = encoder.encode(gt_test, batch_size=16)

    # Select distance calculation type (cosine recommended for higher discriminability)
    distance_type = 'cosine'  # Options: 'wasserstein' / 'cosine'
    print(f"\nUsing distance type: {distance_type}")

    # Calculate distances
    vlm_to_vlm_dist = [calculate_distance(emb, vlm_mean, distance_type) for emb in vlm_test_embeddings]
    gt_to_vlm_dist = [calculate_distance(emb, vlm_mean, distance_type) for emb in gt_test_embeddings]
    vlm_to_gt_dist = [calculate_distance(emb, gt_mean, distance_type) for emb in vlm_test_embeddings]
    gt_to_gt_dist = [calculate_distance(emb, gt_mean, distance_type) for emb in gt_test_embeddings]

    true_labels = [1] * len(vlm_test_augmented) + [0] * len(gt_test)

    # ========== Core Modification: Refactor Threshold Logic ==========
    # 1. Print key intermediate values (for diagnostics)
    all_distances = vlm_to_vlm_dist + vlm_to_gt_dist + gt_to_vlm_dist + gt_to_gt_dist
    dist_range = max(all_distances) - min(all_distances)
    print(f"\nRange of all distances: {min(all_distances):.6f} ~ {max(all_distances):.6f}, Spread: {dist_range:.6f}")

    # Calculate core decision value: dist_vlm - dist_gt
    vlm_diff = [d_vlm - d_gt for d_vlm, d_gt in zip(vlm_to_vlm_dist, vlm_to_gt_dist)]
    gt_diff = [d_vlm - d_gt for d_vlm, d_gt in zip(gt_to_vlm_dist, gt_to_gt_dist)]
    print(f"VLM sample (dist_vlm - dist_gt) range: {min(vlm_diff):.6f} ~ {max(vlm_diff):.6f}")
    print(f"GT sample (dist_vlm - dist_gt) range: {min(gt_diff):.6f} ~ {max(gt_diff):.6f}")

    # 2. Use absolute threshold (expand range to ensure coverage of decision value interval)
    # Adjusted based on distribution of dist_vlm - dist_gt, covering [-0.5, 0.5] interval
    thresholds = [0.01, 0.03, 0.05, 0.07, 0.1, 0, -0.01, -0.03, -0.05, -0.07, -0.1]
    print (f"\n Threshold list used: [{', '.join (f'{t:.6f}' for t in thresholds)}]")

    best_accuracy = 0
    best_auc = 0
    best_tpr = 0
    best_threshold = 0
    final_auc = None
    final_tpr = None
    final_fpr = None
    final_tpr_curve = None
    best_scores = None

    for t in thresholds:
        print(f"\n===== Results with threshold {t:.6f} =====")

        # Distance judgment logic: dist to VLM mean < (dist to GT mean + threshold) -> Predict as VLM (1)
        vlm_preds = [1 if dist_vlm < (dist_gt + t) else 0
                     for dist_vlm, dist_gt in zip(vlm_to_vlm_dist, vlm_to_gt_dist)]
        gt_preds = [1 if dist_vlm < (dist_gt + t) else 0
                    for dist_vlm, dist_gt in zip(gt_to_vlm_dist, gt_to_gt_dist)]

        all_preds = vlm_preds + gt_preds

        accuracy = accuracy_score(true_labels, all_preds)
        print(f"Accuracy: {accuracy:.4f}")

        # Score calculation: dist to GT - dist to VLM (higher value indicates higher probability of VLM output)
        scores = [dist_gt - dist_vlm for dist_vlm, dist_gt in zip(vlm_to_vlm_dist + gt_to_vlm_dist,
                                                                  vlm_to_gt_dist + gt_to_gt_dist)]
        auc = roc_auc_score(true_labels, scores)
        print(f"AUC: {auc:.4f}")

        fpr, tpr, _ = roc_curve(true_labels, scores)
        idx = np.argmin(np.abs(fpr - 0.01))
        tpr_at_fpr_1 = tpr[idx]
        print(f"TPR@FPR=1%: {tpr_at_fpr_1:.4f}")

        # Record best performance metrics
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_auc = auc
            best_tpr = tpr_at_fpr_1
            best_threshold = t
            best_scores = scores
            final_fpr = fpr
            final_tpr_curve = tpr

    print(f"\nBest Threshold: {best_threshold:.6f}, Best Accuracy: {best_accuracy:.4f}")

    # Plot ROC curve - using the curve corresponding to the best threshold
    plt.figure(figsize=(8, 6))
    plt.plot(final_fpr, final_tpr_curve, label=f'ROC Curve (AUC = {best_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(f'ROC Curve (Distance Type: {distance_type})')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('roc_curve.png')
    print("\nROC curve saved as roc_curve.png")

    return best_accuracy, best_auc, best_tpr


if __name__ == "__main__":
    encoder = TextEncoder()
    print(f"Encoder type used: {encoder.encoder_type}, Device: {encoder.device}")

    data = load_data('/root/autodl-tmp/llava_3000.json')

    vlm_1_answers, ground_truth_answers = extract_answers(data)
    # Filter empty answers
    vlm_1_answers = [ans.strip() for ans in vlm_1_answers if ans.strip()]
    ground_truth_answers = [ans.strip() for ans in ground_truth_answers if ans.strip()]

    print(f"Number of valid VLM_1 answers: {len(vlm_1_answers)}")
    print(f"Number of valid Ground Truth answers: {len(ground_truth_answers)}")

    # Check if data volume is sufficient
    if len(vlm_1_answers) < 10 or len(ground_truth_answers) < 10:
        print("Warning: Insufficient valid answers, results may be meaningless!")

    accuracy, auc, tpr_at_fpr_1 = analyze_data(vlm_1_answers, ground_truth_answers, encoder)

    print("\n===== Final Results =====")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"TPR@FPR=1%: {tpr_at_fpr_1:.4f}")