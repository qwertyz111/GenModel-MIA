import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from transformers import AutoTokenizer, AutoModel
import torch
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os
import random
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords

# Download required NLTK resources
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# Load stopwords list
stop_words = set(stopwords.words('english'))

# Global model initialization (load only once to avoid repeated memory usage)
class TextEncoder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        # Try to use local model; if not exists, use remote model
        local_model_path = f'/root/autodl-tmp/{model_name}'
        if os.path.exists(local_model_path):
            model_path = local_model_path
        else:
            model_path = model_name

        self.model_name = model_path
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
                    outputs = self.model(** inputs)
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


# Get synonyms for a specific part-of-speech
def get_synonym(word, pos):
    """Get synonyms for a word with specific part-of-speech"""
    synsets = wordnet.synsets(word, pos=pos)
    synonyms = set()
    for ss in synsets:
        for lemma in ss.lemmas():
            synonym = lemma.name().replace('_', ' ')
            if '-' not in synonym and synonym != word:  # Filter words with hyphens and original word
                synonyms.add(synonym)
    return list(synonyms)


# Replace synonyms in sentences (by percentage)
def replace_synonyms(sentence, replace_percent=20):
    """
    Replace synonyms in a sentence by percentage of replaceable words
    :param sentence: Input sentence
    :param replace_percent: Replacement percentage (0-100)
    :return: Sentence after replacement
    """
    words = nltk.word_tokenize(sentence)
    tagged_words = nltk.pos_tag(words)  # Get part-of-speech tags

    # Filter replaceable candidate words
    candidates = []
    for idx, (word, tag) in enumerate(tagged_words):
        # Filter conditions: non-stopwords, alphabetic words, length > 2, specific POS
        if (word.lower() not in stop_words and
                word.isalpha() and
                len(word) > 2):

            # Map NLTK POS to WordNet POS
            wn_pos = None
            if tag.startswith('NN'):  # Noun
                wn_pos = wordnet.NOUN
            elif tag.startswith('VB'):  # Verb
                wn_pos = wordnet.VERB
            elif tag.startswith('JJ'):  # Adjective
                wn_pos = wordnet.ADJ
            elif tag.startswith('RB'):  # Adverb
                wn_pos = wordnet.ADV

            if wn_pos:
                synonyms = get_synonym(word.lower(), wn_pos)
                if synonyms:  # Ensure synonyms exist
                    candidates.append((idx, word, synonyms))

    # If no replaceable words, return original sentence
    if not candidates:
        return sentence

    # Calculate number of words to replace by percentage (at least 1, unless percentage is 0)
    num_to_replace = max(1, round(len(candidates) * replace_percent / 100)) if replace_percent > 0 else 0
    # Ensure replacement count does not exceed candidate count
    num_to_replace = min(num_to_replace, len(candidates))

    # Randomly select specified number of words to replace
    selected = random.sample(candidates, num_to_replace)

    # Perform replacement
    words_copy = words.copy()
    for idx, original_word, synonyms in selected:
        # Randomly select a synonym
        replacement = random.choice(synonyms)
        # Preserve the original capitalization
        if original_word[0].isupper():
            replacement = replacement.capitalize()
        words_copy[idx] = replacement

    return ' '.join(words_copy)


# Calculate cosine similarity
def cosine_similarity(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0
    return np.dot(a, b) / (norm_a * norm_b)


# Euclidean distance
def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


# Calculate optimal z value (minimize sum of expected distances from all samples to z)
def find_optimal_z(embeddings):
    """
    Find z such that the sum of expected distances from all samples E(x(i)) to z is minimized.
    Mathematically, this is equivalent to finding the geometric median of all sample points.
    """
    # Initial guess point: use mean as starting point
    initial_z = np.mean(embeddings, axis=0)

    # Define objective function: sum of distances from all points to z
    def objective(z):
        z = z.reshape(initial_z.shape)
        total_distance = 0
        for emb in embeddings:
            total_distance += euclidean_distance(emb, z)
        return total_distance

    # Use optimization algorithm to find minimum
    result = minimize(
        objective,
        initial_z.flatten(),
        method='L-BFGS-B',
        options={'maxiter': 100}
    )

    # Return optimized z value
    return result.x.reshape(initial_z.shape)


# Main analysis function (receive encoder instance to avoid repeated initialization)
def analyze_data_with_optimal_z(vlm_1_answers, ground_truth_answers, encoder):
    vlm_train, vlm_test = train_test_split(vlm_1_answers, test_size=0.1, random_state=42)
    gt_train, gt_test = train_test_split(ground_truth_answers, test_size=0.1, random_state=42)

    # Perform synonym replacement on ground truth test set (20% replacement ratio)
    print("Performing synonym replacement on Ground Truth test set...")
    gt_test_synonyms = [replace_synonyms(sentence, replace_percent=20) for sentence in gt_test]

    # Print some replacement examples
    print("Synonym replacement examples:")
    for i in range(min(5, len(gt_test))):
        print(f"Original sentence: {gt_test[i]}")
        print(f"After replacement: {gt_test_synonyms[i]}")
        print()

    print(f"VLM_1 training set size: {len(vlm_train)}, test set size: {len(vlm_test)}")
    print(f"Ground Truth training set size: {len(gt_train)}, test set size: {len(gt_test)}")

    # Specify batch size during encoding (adjust according to GPU memory, e.g., 8/16/32)
    print("Encoding VLM_1 training data...")
    vlm_train_embeddings = encoder.encode(vlm_train, batch_size=16)

    print("Encoding Ground Truth training data...")
    gt_train_embeddings = encoder.encode(gt_train, batch_size=16)

    # Combine embeddings of all training samples
    all_train_embeddings = np.vstack([vlm_train_embeddings, gt_train_embeddings])

    # Find optimal z value
    print("Computing optimal z value (minimize sum of expected distances from all samples to z)...")
    optimal_z = find_optimal_z(all_train_embeddings)
    print(f"Optimal z value computed, dimension: {optimal_z.shape}")

    # Calculate original means for comparison
    vlm_mean = np.mean(vlm_train_embeddings, axis=0)
    gt_mean = np.mean(gt_train_embeddings, axis=0)

    print("Encoding VLM_1 test data...")
    vlm_test_embeddings = encoder.encode(vlm_test, batch_size=16)

    print("Encoding Ground Truth test data (after synonym replacement)...")
    gt_test_embeddings = encoder.encode(gt_test_synonyms, batch_size=16)  # Use test set after synonym replacement

    # Calculate distances to optimal z
    vlm_to_z = [euclidean_distance(emb, optimal_z) for emb in vlm_test_embeddings]
    gt_to_z = [euclidean_distance(emb, optimal_z) for emb in gt_test_embeddings]

    # Calculate distances to original means
    vlm_to_vlm = [cosine_similarity(emb, vlm_mean) for emb in vlm_test_embeddings]
    vlm_to_gt = [cosine_similarity(emb, gt_mean) for emb in vlm_test_embeddings]
    gt_to_vlm = [cosine_similarity(emb, vlm_mean) for emb in gt_test_embeddings]
    gt_to_gt = [cosine_similarity(emb, gt_mean) for emb in gt_test_embeddings]

    # Make predictions using optimal z
    z_predictions = [1 if dist_vlm < dist_gt else 0 for dist_vlm, dist_gt in zip(vlm_to_z, gt_to_z)]

    # Make predictions using original method
    original_predictions = [1 if sim_vlm > sim_gt else 0 for sim_vlm, sim_gt in zip(vlm_to_vlm, vlm_to_gt)]
    gt_original_predictions = [1 if sim_vlm > sim_gt else 0 for sim_vlm, sim_gt in zip(gt_to_vlm, gt_to_gt)]

    vlm_true = [1] * len(vlm_test)
    gt_true = [0] * len(gt_test)

    all_true = vlm_true + gt_true

    # Evaluate optimal z method
    all_z_predictions = z_predictions + [0] * len(gt_test)  # Predict GT samples as 0
    z_accuracy = accuracy_score(all_true, all_z_predictions)
    print(f"Accuracy using optimal z: {z_accuracy:.4f}")

    # Evaluate original method
    all_original_predictions = original_predictions + gt_original_predictions
    original_accuracy = accuracy_score(all_true, all_original_predictions)
    print(f"Original method accuracy: {original_accuracy:.4f}")

    # Calculate AUC (use negative distance as score, smaller distance means higher score)
    z_scores_1 = [-dist for dist in vlm_to_z] + [-dist for dist in gt_to_z]
    z_auc_1 = roc_auc_score(all_true, z_scores_1)
    print(f"AUC_1 using optimal z: {z_auc_1:.4f}")

    # Calculate distances to original means
    vlm_to_vlm = [cosine_similarity(emb, vlm_mean) for emb in vlm_test_embeddings]
    vlm_to_gt = [cosine_similarity(emb, gt_mean) for emb in vlm_test_embeddings]
    gt_to_vlm = [cosine_similarity(emb, vlm_mean) for emb in gt_test_embeddings]
    gt_to_gt = [cosine_similarity(emb, gt_mean) for emb in gt_test_embeddings]

    z_scores_2 = [sim for sim in vlm_to_vlm] + [sim for sim in gt_to_vlm]
    z_auc_2 = roc_auc_score(all_true, z_scores_2)
    print(f"AUC_2 using optimal z: {z_auc_2:.4f}")

    z_scores = [s1 + s2 for s1, s2 in zip(z_scores_1, z_scores_2)]
    z_auc = roc_auc_score(all_true, z_scores)
    print(f"AUC_3 using optimal z: {z_auc:.4f}")

    # Original method AUC
    scores_1 = [-sim for sim in vlm_to_gt] + [-sim for sim in gt_to_gt]
    scores_2 = [sim for sim in vlm_to_vlm] + [sim for sim in gt_to_vlm]
    scores = [s1 + s2 for s1, s2 in zip(scores_1, scores_2)]
    original_auc = roc_auc_score(all_true, scores)
    print(f"Original method AUC: {original_auc:.4f}")

    # Calculate TPR@FPR=1%
    fpr, tpr, _ = roc_curve(all_true, z_scores_1)
    idx = np.argmin(np.abs(fpr - 0.01))
    z_tpr_at_fpr_1_1 = tpr[idx]
    print(f"TPR@FPR=1%(1) using optimal z: {z_tpr_at_fpr_1_1:.4f}")

    fpr, tpr, _ = roc_curve(all_true, z_scores_2)
    idx = np.argmin(np.abs(fpr - 0.01))
    z_tpr_at_fpr_1_2 = tpr[idx]
    print(f"TPR@FPR=1%(2) using optimal z: {z_tpr_at_fpr_1_2:.4f}")

    fpr, tpr, _ = roc_curve(all_true, z_scores)
    idx = np.argmin(np.abs(fpr - 0.01))
    z_tpr_at_fpr_1 = tpr[idx]
    print(f"TPR@FPR=1% using optimal z: {z_tpr_at_fpr_1:.4f}")

    fpr, tpr, _ = roc_curve(all_true, scores)
    idx = np.argmin(np.abs(fpr - 0.01))
    original_tpr_at_fpr_1 = tpr[idx]
    print(f"TPR@FPR=1% for original method: {original_tpr_at_fpr_1:.4f}")

    # Plot ROC curves for comparison
    plt.figure(figsize=(10, 8))

    # ROC curve for optimal z method
    fpr_z, tpr_z, _ = roc_curve(all_true, z_scores)
    plt.plot(fpr_z, tpr_z, label=f'Optimal z method (AUC = {z_auc:.2f})')

    # ROC curve for original method
    fpr_orig, tpr_orig, _ = roc_curve(all_true, scores)
    plt.plot(fpr_orig, tpr_orig, label=f'Original method (AUC = {original_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curves Comparison (Optimal z vs Original method, GT test set with synonym replacement)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('roc_curve_comparison_with_synonyms.png')
    print("ROC curves comparison saved as roc_curve_comparison_with_synonyms.png")

    # Calculate and print distance statistics
    print("===== Distance Statistics =====")
    print(f"Average distance from VLM_1 test samples to optimal z: {np.mean(vlm_to_z):.4f}")
    print(f"Average distance from Ground Truth test samples (after replacement) to optimal z: {np.mean(gt_to_z):.4f}")
    print(f"Average distance from VLM_1 training samples to optimal z: {np.mean([euclidean_distance(emb, optimal_z) for emb in vlm_train_embeddings]):.4f}")
    print(f"Average distance from Ground Truth training samples to optimal z: {np.mean([euclidean_distance(emb, optimal_z) for emb in gt_train_embeddings]):.4f}")

    # Compare distances between optimal z and original means
    print(f"Distance between optimal z and VLM_1 mean: {euclidean_distance(optimal_z, vlm_mean):.4f}")
    print(f"Distance between optimal z and Ground Truth mean: {euclidean_distance(optimal_z, gt_mean):.4f}")
    print(f"Distance between VLM_1 mean and Ground Truth mean: {euclidean_distance(vlm_mean, gt_mean):.4f}")

    return {
        'optimal_z': optimal_z,
        'z_accuracy': z_accuracy,
        'original_accuracy': original_accuracy,
        'z_auc_1': z_auc_1,
        'z_auc_2': z_auc_2,
        'z_auc': z_auc,
        'original_auc': original_auc,
        'z_tpr_at_fpr_1_1': z_tpr_at_fpr_1_1,
        'z_tpr_at_fpr_1_2': z_tpr_at_fpr_1_2,
        'z_tpr_at_fpr_1': z_tpr_at_fpr_1,
        'original_tpr_at_fpr_1': original_tpr_at_fpr_1
    }


if __name__ == "__main__":
    # Initialize encoder (global unique instance)
    encoder = TextEncoder()
    print(f"Encoder type: {encoder.encoder_type}, Device: {encoder.device}")

    # Load data
    data = load_data('/root/autodl-tmp/Apple_1100_items.json')

    # Extract answers
    vlm_1_answers, ground_truth_answers = extract_answers(data)
    vlm_1_answers = [ans for ans in vlm_1_answers if ans]
    ground_truth_answers = [ans for ans in ground_truth_answers if ans]

    print(f"Number of VLM_1 answers: {len(vlm_1_answers)}")
    print(f"Number of Ground Truth answers: {len(ground_truth_answers)}")

    # Analyze data
    results = analyze_data_with_optimal_z(vlm_1_answers, ground_truth_answers, encoder)

    # Output final results
    print("===== Final Results Comparison (GT test set with synonym replacement) =====")
    print(f"Optimal z method accuracy: {results['z_accuracy']:.4f}")
    print(f"Original method accuracy: {results['original_accuracy']:.4f}")
    print(f"Optimal z method AUC_1: {results['z_auc_1']:.4f}")
    print(f"Optimal z method AUC_2: {results['z_auc_2']:.4f}")
    print(f"Optimal z method AUC_3: {results['z_auc']:.4f}")
    print(f"Original method AUC: {results['original_auc']:.4f}")
    print(f"Optimal z method TPR@FPR=1%(1): {results['z_tpr_at_fpr_1_1']:.4f}")
    print(f"Optimal z method TPR@FPR=1%(2): {results['z_tpr_at_fpr_1_2']:.4f}")
    print(f"Optimal z method TPR@FPR=1%: {results['z_tpr_at_fpr_1']:.4f}")
    print(f"Original method TPR@FPR=1%: {results['original_tpr_at_fpr_1']:.4f}")
