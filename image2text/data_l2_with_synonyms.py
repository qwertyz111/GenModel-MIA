import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from transformers import AutoTokenizer, AutoModel
import torch
import matplotlib.pyplot as plt
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


# Get synonyms for specific parts of speech
def get_synonym(word, pos):
    synsets = wordnet.synsets(word, pos=pos)
    synonyms = set()
    for ss in synsets:
        for lemma in ss.lemmas():
            synonym = lemma.name().replace('_', ' ')
            if '-' not in synonym and synonym != word:
                synonyms.add(synonym)
    return list(synonyms)


# Synonym replacement for sentences (improved version)
def replace_synonyms(sentence, replace_percent=20, min_replace=1, max_replace=None):
    """
    Synonym replacement for sentences with dynamic adjustment of replacement ratio
    :param sentence: Input sentence
    :param replace_percent: Base replacement percentage (0-100)
    :param min_replace: Minimum number of words to replace
    :param max_replace: Maximum number of words to replace (None means no limit)
    :return: Sentence after replacement
    """

    words = nltk.word_tokenize(sentence)
    sentence_length = len(words)
    tagged_words = nltk.pos_tag(words)  # Get part-of-speech tags

    # Filter candidate words for replacement
    candidates = []
    for idx, (word, tag) in enumerate(tagged_words):
        # Filtering conditions: non-stop words, alphabetic words, length greater than 2, specific POS
        # Protection: numbers, proper nouns (starting with uppercase but not at sentence start), single character words
        if (word.lower() not in stop_words and
                word.isalpha() and
                len(word) > 2 and
                not word.isdigit() and
                not (idx > 0 and word[0].isupper() and word[1:].islower())):  # Protect proper nouns

            # Map NLTK POS to WordNet POS
            wn_pos = None
            if tag.startswith('NN') and not tag.startswith('NNP'):  # Noun but not proper noun
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
                    # Sort by synonym length, prioritize synonyms with similar length (more likely to have similar semantics)
                    synonyms_sorted = sorted(synonyms, key=lambda x: abs(len(x) - len(word)))
                    candidates.append((idx, word, synonyms_sorted))

    # If there are no replaceable words, return the original sentence directly
    if not candidates:
        return sentence

    # Dynamically adjust replacement ratio based on sentence length
    # Short sentences (<10 words) maintain original ratio, medium sentences (10-20 words) slightly decrease, long sentences (>20 words) further decrease
    if sentence_length < 10:
        adjusted_percent = replace_percent
    elif sentence_length < 20:
        adjusted_percent = replace_percent * 0.8
    else:
        adjusted_percent = replace_percent * 0.6

    # Calculate the number of words to replace based on percentage
    num_to_replace = round(len(candidates) * adjusted_percent / 100)

    # Apply minimum and maximum word replacement limits
    num_to_replace = max(min_replace, num_to_replace)
    if max_replace is not None:
        num_to_replace = min(num_to_replace, max_replace)

    # Ensure replacement count doesn't exceed total candidate words
    num_to_replace = min(num_to_replace, len(candidates))

    # Randomly select specified number of words for replacement
    selected = random.sample(candidates, num_to_replace)

    # Execute replacement
    words_copy = words.copy()
    for idx, original_word, synonyms in selected:
        # Prioritize the first 3 most similar synonyms (similar length)
        num_synonyms_to_consider = min(3, len(synonyms))
        replacement = random.choice(synonyms[:num_synonyms_to_consider])
        # Maintain the original case format of the word
        if original_word[0].isupper():
            replacement = replacement.capitalize()
        words_copy[idx] = replacement

    return ' '.join(words_copy)


# Text encoding function
def encode_texts(texts, model_name='/root/autodl-tmp/all-MiniLM-L6-v2'):
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, convert_to_tensor=True)
        return embeddings.cpu().numpy()
    except ImportError:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        embeddings = []
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :].numpy()
                embeddings.append(embedding[0])

        return np.array(embeddings)


# Calculate L2 distance (Euclidean distance)
def l2_distance(a, b):
    """Calculate L2 distance (Euclidean distance) between two vectors"""
    return np.linalg.norm(a - b)


# Main analysis function
def analyze_data(vlm_1_answers, ground_truth_answers):
    # Split training and test sets
    vlm_train, vlm_test = train_test_split(vlm_1_answers, test_size=0.1, random_state=42)
    gt_train, gt_test = train_test_split(ground_truth_answers, test_size=0.1, random_state=42)

    print(f"VLM_1 training set size: {len(vlm_train)}, test set size: {len(vlm_test)}")
    print(f"Ground Truth training set size: {len(gt_train)}, test set size: {len(gt_test)}")

    # Perform synonym replacement on vlm_1 in the test set (using 20% replacement ratio)
    print("Performing synonym replacement on test set VLM_1...")
    vlm_test_synonyms = [replace_synonyms(sentence, replace_percent=20) for sentence in vlm_test]

    # Print replacement examples
    print("Synonym replacement examples:")
    for i in range(min(5, len(vlm_test))):
        print(f"Original: {vlm_test[i]}")
        print(f"Replaced: {vlm_test_synonyms[i]}")
        print()

    # Encode training set text
    print("Encoding VLM_1 training data...")
    vlm_train_embeddings = encode_texts(vlm_train)

    print("Encoding Ground Truth training data...")
    gt_train_embeddings = encode_texts(gt_train)

    # Calculate distribution centers (means)
    vlm_mean = np.mean(vlm_train_embeddings, axis=0)
    gt_mean = np.mean(gt_train_embeddings, axis=0)

    # Encode test set
    print("Encoding VLM_1 test data (after synonym replacement)...")
    vlm_test_embeddings = encode_texts(vlm_test_synonyms)

    print("Encoding Ground Truth test data...")
    gt_test_embeddings = encode_texts(gt_test)

    # Calculate L2 distance between each test sample and two distribution centers (smaller distance means more similarity)
    vlm_to_vlm = [l2_distance(emb, vlm_mean) for emb in vlm_test_embeddings]  # Distance from VLM test samples to VLM center
    vlm_to_gt = [l2_distance(emb, gt_mean) for emb in vlm_test_embeddings]    # Distance from VLM test samples to GT center
    gt_to_vlm = [l2_distance(emb, vlm_mean) for emb in gt_test_embeddings]    # Distance from GT test samples to VLM center
    gt_to_gt = [l2_distance(emb, gt_mean) for emb in gt_test_embeddings]      # Distance from GT test samples to GT center

    # Prediction logic: smaller distance means more similarity, so if distance to VLM center is less than distance to GT center, predict as VLM (1)
    vlm_predictions = [1 if dist_vlm < dist_gt else 0 for dist_vlm, dist_gt in zip(vlm_to_vlm, vlm_to_gt)]
    gt_predictions = [1 if dist_vlm < dist_gt else 0 for dist_vlm, dist_gt in zip(gt_to_vlm, gt_to_gt)]

    # True labels: 1 represents VLM, 0 represents GT
    vlm_true = [1] * len(vlm_test)
    gt_true = [0] * len(gt_test)

    # Combine predictions and true labels
    all_predictions = vlm_predictions + gt_predictions
    all_true = vlm_true + gt_true

    # Calculate accuracy
    accuracy = accuracy_score(all_true, all_predictions)
    print(f"Accuracy: {accuracy:.4f}")

    # Calculate AUC: smaller L2 distance indicates higher similarity, so use negative distance as score (higher score means more likely to belong to corresponding class)
    # For determining if it's GT (label 0): smaller distance to GT center means more likely to be GT, so score is -gt_to_gt
    # For determining if it's VLM (label 1): smaller distance to VLM center means more likely to be VLM, so score is -gt_to_vlm
    scores_1 = [-dist for dist in vlm_to_gt] + [-dist for dist in gt_to_gt]  # Based on distance to GT center
    auc1 = roc_auc_score(all_true, scores_1)
    print(f"AUC_1: {auc1:.4f}")

    scores_2 = [-dist for dist in vlm_to_vlm] + [-dist for dist in gt_to_vlm]  # Based on distance to VLM center
    auc2 = roc_auc_score(all_true, scores_2)
    print(f"AUC_2: {auc2:.4f}")

    scores = [s1 + s2 for s1, s2 in zip(scores_1, scores_2)]  # Combine scores
    auc = roc_auc_score(all_true, scores)
    print(f"AUC_3: {auc:.4f}")

    # Calculate TPR@FPR=1%
    fpr, tpr, thresholds = roc_curve(all_true, scores_1)
    idx = np.argmin(np.abs(fpr - 0.01))
    tpr_at_fpr_1_1 = tpr[idx]
    print(f"TPR@FPR=1%(1): {tpr_at_fpr_1_1:.4f}")

    fpr, tpr, thresholds = roc_curve(all_true, scores_2)
    idx = np.argmin(np.abs(fpr - 0.01))
    tpr_at_fpr_1_2 = tpr[idx]
    print(f"TPR@FPR=1%(2): {tpr_at_fpr_1_2:.4f}")

    fpr, tpr, thresholds = roc_curve(all_true, scores)
    idx = np.argmin(np.abs(fpr - 0.01))
    tpr_at_fpr_1 = tpr[idx]
    print(f"TPR@FPR=1%: {tpr_at_fpr_1:.4f}")

    # Draw ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve (Using L2 Distance, VLM_1 with Synonym Replacement)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('roc_curve_l2.png')
    print("ROC curve saved as roc_curve_l2.png")

    return accuracy, auc, tpr_at_fpr_1


if __name__ == "__main__":
    # Load data
    data = load_data('/root/autodl-tmp/gpt_1000.json')

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