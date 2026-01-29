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

# Download nltk required resources (needed for first run)
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


# Synonym replacement function
def synonym_replacement(text, replace_ratio=0.2):
    """
    Perform synonym replacement on text
    :param text: Input text
    :param replace_ratio: Replacement ratio
    :return: Processed text
    """
    if not text:
        return text

    # Tokenize and get part-of-speech tags
    words = word_tokenize(text)
    tagged_words = pos_tag(words)

    # Determine the number of words to replace
    num_replace = max(1, int(len(words) * replace_ratio))
    replace_indices = random.sample(range(len(words)), min(num_replace, len(words)))

    # Part-of-speech mapping (WordNet POS to nltk POS)
    pos_map = {
        'NN': wordnet.NOUN,
        'VB': wordnet.VERB,
        'JJ': wordnet.ADJ,
        'RB': wordnet.ADV
    }

    for i in replace_indices:
        word, tag = tagged_words[i]
        # Only replace nouns, verbs, adjectives, adverbs
        if tag[:2] in pos_map:
            # Get synonym sets
            synsets = wordnet.synsets(word, pos=pos_map[tag[:2]])
            if synsets:
                # Randomly select a different synonym from the synonym set
                synonyms = [lemma.name() for synset in synsets for lemma in synset.lemmas() if lemma.name() != word]
                if synonyms:
                    # Select a synonym for replacement (handle underscore-connected words)
                    replacement = random.choice(synonyms).replace('_', ' ')
                    words[i] = replacement

    return ' '.join(words)


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


# Main analysis function (modified to use cosine similarity)
def analyze_data(vlm_1_answers, ground_truth_answers, encoder):
    vlm_train, vlm_test = train_test_split(vlm_1_answers, test_size=0.1, random_state=42)
    gt_train, gt_test = train_test_split(ground_truth_answers, test_size=0.1, random_state=42)

    # Perform synonym replacement on vlm_1 test set (20% ratio)
    print("Performing synonym replacement on VLM_1 test set...")
    vlm_test_augmented = [synonym_replacement(text, replace_ratio=0.2) for text in vlm_test]

    print(f"VLM_1 training set size: {len(vlm_train)}, test set size: {len(vlm_test_augmented)}")
    print(f"Ground Truth training set size: {len(gt_train)}, test set size: {len(gt_test)}")

    # Specify batch size during encoding
    print("Encoding VLM_1 training data...")
    vlm_train_embeddings = encoder.encode(vlm_train, batch_size=16)

    print("Encoding Ground Truth training data...")
    gt_train_embeddings = encoder.encode(gt_train, batch_size=16)

    # Calculate mean vectors of training set
    vlm_mean = np.mean(vlm_train_embeddings, axis=0)
    gt_mean = np.mean(gt_train_embeddings, axis=0)

    print("Encoding enhanced VLM_1 test data...")
    vlm_test_embeddings = encoder.encode(vlm_test_augmented, batch_size=16)

    print("Encoding Ground Truth test data...")
    gt_test_embeddings = encoder.encode(gt_test, batch_size=16)

    # Calculate Wasserstein distance
    vlm_to_vlm_dist = [wasserstein_distance(emb, vlm_mean) for emb in vlm_test_embeddings]
    gt_to_vlm_dist = [wasserstein_distance(emb, vlm_mean) for emb in gt_test_embeddings]

    vlm_to_gt_dist = [wasserstein_distance(emb, gt_mean) for emb in vlm_test_embeddings]
    gt_to_gt_dist = [wasserstein_distance(emb, gt_mean) for emb in gt_test_embeddings]

    true_labels = [1] * len(vlm_test_augmented) + [0] * len(gt_test)

    # Threshold adjustment (Wasserstein distance range is [0,∞), using appropriate threshold range for distance)
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    final_auc = None
    final_tpr = None

    for t in thresholds:
        print(f"\n===== Results using threshold {t} =====")

        # Similarity judgment logic: similarity to VLM mean > similarity to GT mean - threshold → predict as VLM (1)
        vlm_preds = [1 if dist_vlm < (dist_gt + t) else 0
                     for dist_vlm, dist_gt in zip(vlm_to_vlm_dist, vlm_to_gt_dist)]
        gt_preds = [1 if dist_vlm < (dist_gt + t) else 0
                    for dist_vlm, dist_gt in zip(gt_to_vlm_dist, gt_to_gt_dist)]

        all_preds = vlm_preds + gt_preds

        accuracy = accuracy_score(true_labels, all_preds)
        print(f"Accuracy: {accuracy:.4f}")

        # Score calculation: distance to GT - distance to VLM (larger values more likely to be VLM output)
        scores = [dist_gt - dist_vlm for dist_vlm, dist_gt in zip(vlm_to_vlm_dist + gt_to_vlm_dist,
                                                                 vlm_to_gt_dist + gt_to_gt_dist)]
        auc = roc_auc_score(true_labels, scores)
        print(f"AUC: {auc:.4f}")
        fpr, tpr, _ = roc_curve(true_labels, scores)
        idx = np.argmin(np.abs(fpr - 0.01))
        tpr_at_fpr_1 = tpr[idx]
        print(f"TPR@FPR=1%: {tpr_at_fpr_1:.4f}")
        if t == thresholds[-1]:
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

    return accuracy, final_auc, final_tpr


if __name__ == "__main__":
    encoder = TextEncoder()
    print(f"Using encoder type: {encoder.encoder_type}, device: {encoder.device}")

    data = load_data('/root/autodl-tmp/llava_3000.json')

    vlm_1_answers, ground_truth_answers = extract_answers(data)
    vlm_1_answers = [ans for ans in vlm_1_answers if ans]
    ground_truth_answers = [ans for ans in ground_truth_answers if ans]

    print(f"VLM_1 answer count: {len(vlm_1_answers)}")
    print(f"Ground Truth answer count: {len(ground_truth_answers)}")

    accuracy, auc, tpr_at_fpr_1 = analyze_data(vlm_1_answers, ground_truth_answers, encoder)

    print("\n===== Final Results =====")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"TPR@FPR=1%: {tpr_at_fpr_1:.4f}")