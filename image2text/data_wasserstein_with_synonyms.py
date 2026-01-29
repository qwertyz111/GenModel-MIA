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
from nltk.corpus import stopwords  # Added: for stop words filtering
from scipy.stats import wasserstein_distance

# Download required NLTK resources
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Modify averaged_perceptron_tagger download logic, use resource name specified in error message
try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng')

# Keep previous punkt_tab resource download (if already added)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# Load stop words list
stop_words = set(stopwords.words('english'))


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


# Get synonyms for specific POS (improved version)
def get_synonym(word, pos):
    """Get synonyms for a word in a specific POS"""
    synsets = wordnet.synsets(word, pos=pos)
    synonyms = set()
    for ss in synsets:
        for lemma in ss.lemmas():
            synonym = lemma.name().replace('_', ' ')
            if '-' not in synonym and synonym != word:  # Filter words containing hyphens and the original word
                synonyms.add(synonym)
    return list(synonyms)


# Perform synonym replacement on sentences (changed to percentage replacement)
def replace_synonyms(sentence, replace_percent=20):
    """
    Perform synonym replacement on sentences, replacing by a percentage of replaceable words
    :param sentence: Input sentence
    :param replace_percent: Replacement percentage (0-100)
    :return: Replaced sentence
    """
    words = nltk.word_tokenize(sentence)
    tagged_words = nltk.pos_tag(words)  # Get POS tags

    # Filter candidate words for replacement
    candidates = []
    for idx, (word, tag) in enumerate(tagged_words):
        # Filtering conditions: non-stop words, alphabetic, length > 2
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
                if synonyms:  # Ensure there are synonyms
                    candidates.append((idx, word, synonyms))

    # If no replaceable words, return original sentence
    if not candidates:
        return sentence

    # Calculate number of words to replace by percentage (at least 1, unless percentage is 0)
    num_to_replace = max(1, round(len(candidates) * replace_percent / 100)) if replace_percent > 0 else 0
    # Ensure replacement count doesn't exceed total candidate words
    num_to_replace = min(num_to_replace, len(candidates))

    # Randomly select specified number of words for replacement
    selected = random.sample(candidates, num_to_replace)

    # Execute replacement
    words_copy = words.copy()
    for idx, original_word, synonyms in selected:
        # Randomly select a synonym
        replacement = random.choice(synonyms)
        # Maintain original word's case format
        if original_word[0].isupper():
            replacement = replacement.capitalize()
        words_copy[idx] = replacement

    return ' '.join(words_copy)


# Text encoding function
def encode_texts(texts, model_name='/root/autodl-tmp/all-MiniLM-L6-v2'):
    """
    Use pretrained model to encode text, return embedding vectors
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, convert_to_tensor=True)
        return embeddings.cpu().numpy()
    except ImportError:
        # If sentence-transformers is not installed, use transformers
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        embeddings = []
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
                # Use [CLS] token embedding or average pooling
                embedding = outputs.last_hidden_state[:, 0, :].numpy()
                embeddings.append(embedding[0])

        return np.array(embeddings)


# Calculate Wasserstein distance (Earth Mover's Distance)
def wasserstein_distance_wrapper(a, b):
    """
    Calculate Wasserstein distance between two distributions.
    Since embedding vectors are high-dimensional, we use distances on each dimension.
    """
    # Method 1: Calculate Wasserstein distance for each dimension then average
    if a.ndim == 1 and b.ndim == 1:
        return wasserstein_distance(a, b)
    elif a.ndim == 2 and b.ndim == 2:
        # For 2D arrays, calculate Wasserstein distance for each feature dimension
        distances = []
        for i in range(a.shape[1]):
            dist = wasserstein_distance(a[:, i], b[:, i])
            distances.append(dist)
        return np.mean(distances)
    else:
        # If it's a single vector, treat it as a single-point distribution
        # To calculate distance, we can repeat single vector to form a distribution
        if a.ndim == 1:
            a = a.reshape(1, -1)
        if b.ndim == 1:
            b = b.reshape(1, -1)
        # For single-point distribution, return Euclidean distance as approximation
        return np.linalg.norm(a - b)


# Use Wasserstein distance for distribution comparison
def compute_wasserstein_distances(vlm_embeddings, gt_embeddings, test_embeddings):
    """
    Calculate Wasserstein distance from test samples to VLM and GT distributions
    """
    # Use training distribution as reference distribution
    # For Wasserstein distance, we need to compare two distributions
    # Here we compare each test sample with the entire training distribution

    vlm_to_vlm_distances = []
    vlm_to_gt_distances = []
    gt_to_vlm_distances = []
    gt_to_gt_distances = []

    # Calculate distance from each test sample to training distribution
    for test_emb in test_embeddings:
        # Compare test sample with VLM training distribution
        # To use Wasserstein distance, we need to compare single test vector with distribution
        # Here we create a distribution containing test vector (repeated multiple times) to compare with training distribution

        # Create a distribution for each test sample (repeat test vector multiple times)
        test_dist = np.tile(test_emb.reshape(1, -1), (vlm_embeddings.shape[0], 1))

        # Calculate Wasserstein distance
        dist_vlm = wasserstein_distance_wrapper(test_dist, vlm_embeddings)
        dist_gt = wasserstein_distance_wrapper(test_dist, gt_embeddings)

        vlm_to_vlm_distances.append(dist_vlm)
        vlm_to_gt_distances.append(dist_gt)

    return vlm_to_vlm_distances, vlm_to_gt_distances, gt_to_vlm_distances, gt_to_gt_distances


# Main analysis function
def analyze_data(vlm_1_answers, ground_truth_answers):
    # Split training set (90%) and test set (10%)
    vlm_train, vlm_test = train_test_split(vlm_1_answers, test_size=0.1, random_state=42)
    gt_train, gt_test = train_test_split(ground_truth_answers, test_size=0.1, random_state=42)

    print(f"VLM_1 training set size: {len(vlm_train)}, test set size: {len(vlm_test)}")
    print(f"Ground Truth training set size: {len(gt_train)}, test set size: {len(gt_test)}")

    # Perform synonym replacement on test set ground truth (using 20% replacement ratio)
    print("Performing synonym replacement on test set Ground Truth...")
    gt_test_synonyms = [replace_synonyms(sentence, replace_percent=20) for sentence in gt_test]

    # Print some replacement examples
    print("Synonym replacement examples:")
    for i in range(min(5, len(gt_test))):
        print(f"Original: {gt_test[i]}")
        print(f"After replacement: {gt_test_synonyms[i]}")
        print()

    # Step 1 and 3: Encode training set text
    print("Encoding VLM_1 training data...")
    vlm_train_embeddings = encode_texts(vlm_train)

    print("Encoding Ground Truth training data...")
    gt_train_embeddings = encode_texts(gt_train)

    # Step 5: Encode test set and compare with distribution
    print("Encoding VLM_1 test data...")
    vlm_test_embeddings = encode_texts(vlm_test)

    print("Encoding Ground Truth test data (after synonym replacement)...")
    gt_test_embeddings = encode_texts(gt_test_synonyms)  # Use test set after synonym replacement

    # Calculate using Wasserstein distance
    print("Calculating Wasserstein distance...")

    # Calculate distances of VLM test samples to two distributions
    vlm_to_vlm_distances = []
    vlm_to_gt_distances = []

    for test_emb in vlm_test_embeddings:
        # Create distribution for test sample (repeat multiple times)
        test_dist = np.tile(test_emb.reshape(1, -1), (vlm_train_embeddings.shape[0], 1))

        # Calculate Wasserstein distance to VLM training distribution
        dist_vlm = wasserstein_distance_wrapper(test_dist, vlm_train_embeddings)
        # Calculate Wasserstein distance to GT training distribution
        dist_gt = wasserstein_distance_wrapper(test_dist, gt_train_embeddings)

        vlm_to_vlm_distances.append(dist_vlm)
        vlm_to_gt_distances.append(dist_gt)

    # Calculate distances of GT test samples to two distributions
    gt_to_vlm_distances = []
    gt_to_gt_distances = []

    for test_emb in gt_test_embeddings:
        # Create distribution for test sample (repeat multiple times)
        test_dist = np.tile(test_emb.reshape(1, -1), (vlm_train_embeddings.shape[0], 1))

        # Calculate Wasserstein distance to VLM training distribution
        dist_vlm = wasserstein_distance_wrapper(test_dist, vlm_train_embeddings)
        # Calculate Wasserstein distance to GT training distribution
        dist_gt = wasserstein_distance_wrapper(test_dist, gt_train_embeddings)

        gt_to_vlm_distances.append(dist_vlm)
        gt_to_gt_distances.append(dist_gt)

    # Prediction: if sample distance to VLM distribution is smaller, predict VLM, else predict GT
    # Note: smaller distance means higher similarity
    vlm_predictions = [1 if dist_vlm < dist_gt else 0 for dist_vlm, dist_gt in
                       zip(vlm_to_vlm_distances, vlm_to_gt_distances)]
    gt_predictions = [1 if dist_vlm < dist_gt else 0 for dist_vlm, dist_gt in
                      zip(gt_to_vlm_distances, gt_to_gt_distances)]

    # True labels: 1 for VLM, 0 for GT
    vlm_true = [1] * len(vlm_test)
    gt_true = [0] * len(gt_test)

    # Combine predictions and true labels
    all_predictions = vlm_predictions + gt_predictions
    all_true = vlm_true + gt_true

    # Calculate accuracy
    accuracy = accuracy_score(all_true, all_predictions)
    print(f"Accuracy: {accuracy:.4f}")

    # Calculate AUC
    # Use distance difference as score (closer to VLM means more likely VLM, so score = dist_gt - dist_vlm)
    scores_1 = [dist_gt - dist_vlm for dist_vlm, dist_gt in zip(vlm_to_vlm_distances, vlm_to_gt_distances)] + \
               [dist_gt - dist_vlm for dist_vlm, dist_gt in zip(gt_to_vlm_distances, gt_to_gt_distances)]

    auc = roc_auc_score(all_true, scores_1)
    print(f"AUC: {auc:.4f}")

    # Calculate TPR@FPR=1%
    fpr, tpr, thresholds = roc_curve(all_true, scores_1)
    # Find threshold closest to 1% FPR
    idx = np.argmin(np.abs(fpr - 0.01))
    tpr_at_fpr_1 = tpr[idx]
    print(f"TPR@FPR=1%: {tpr_at_fpr_1:.4f}")

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve (Wasserstein distance, Ground Truth with synonym replacement)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('roc_curve_wasserstein.png')
    print("ROC curve saved as roc_curve_wasserstein.png")

    return accuracy, auc, tpr_at_fpr_1


if __name__ == "__main__":
    # Load data
    data = load_data('/root/autodl-tmp/llava_3000.json')

    # Extract answers
    vlm_1_answers, ground_truth_answers = extract_answers(data)

    # Filter out empty answers
    vlm_1_answers = [ans for ans in vlm_1_answers if ans]
    ground_truth_answers = [ans for ans in ground_truth_answers if ans]

    print(f"VLM_1 answer count: {len(vlm_1_answers)}")
    print(f"Ground Truth answer count: {len(ground_truth_answers)}")

    # Analyze data
    accuracy, auc, tpr_at_fpr_1 = analyze_data(vlm_1_answers, ground_truth_answers)

    # Output final results
    print("===== Final Results =====")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"TPR@FPR=1%: {tpr_at_fpr_1:.4f}")
