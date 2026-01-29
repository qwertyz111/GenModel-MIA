import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import numpy as np
from transformers import AutoTokenizer, AutoModel
import os
import nltk
from nltk.corpus import wordnet
import random
import re

# Download necessary NLTK data
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


# ==================== Text Preprocessing ====================
def preprocess_text(text):
    """
    Preprocess text to remove special tokens that interfere with experimental analysis.
    
    Processing includes:
    1. Remove </s> end marker
    2. Remove <s> start marker
    3. Remove [SEP], [CLS], [PAD] and other BERT special tokens
    4. Remove <pad>, <unk>, <eos>, <bos> and other common special tokens
    5. Remove redundant whitespace
    6. Remove leading and trailing whitespace
    """
    if not text:
        return text
    
    # Define patterns for special tokens to be removed
    special_tokens = [
        r'</s>',           # End marker
        r'<s>',            # Start marker
        r'\[SEP\]',        # BERT SEP marker
        r'\[CLS\]',        # BERT CLS marker
        r'\[PAD\]',        # BERT PAD marker
        r'\[UNK\]',        # BERT UNK marker
        r'<pad>',          # pad marker
        r'<unk>',          # unknown marker
        r'<eos>',          # end of sequence
        r'<bos>',          # begin of sequence
        r'<\|endoftext\|>',  # GPT end marker
        r'<\|startoftext\|>', # GPT start marker
    ]
    
    # Remove special tokens one by one (case-insensitive)
    for token in special_tokens:
        text = re.sub(token, '', text, flags=re.IGNORECASE)
    
    # Remove redundant whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading and trailing whitespace
    text = text.strip()
    
    return text


# Load JSON data
def load_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


# Extract VLM and Ground Truth texts (with preprocessing)
def extract_texts(data):
    text_1 = []  # VLM text
    text_2 = []  # Ground Truth text (D_aux)

    for item in data:
        conversations = item.get("conversations_0.1", [])
        for conv in conversations:
            if conv["from"] == "vlm_1":
                # Preprocess VLM text
                processed_text = preprocess_text(conv["value"])
                text_1.append(processed_text)
            elif conv["from"] == "ground truth":
                # Preprocess Ground Truth text
                processed_text = preprocess_text(conv["value"])
                text_2.append(processed_text)

    return text_1, text_2


# Get synonyms of a word
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ')
            if synonym.lower() != word.lower():
                synonyms.add(synonym)
    return list(synonyms)


# Perform synonym replacement on text (20% replacement ratio)
def replace_with_synonyms(text, replace_ratio=0.2):
    words = text.split()
    num_to_replace = int(len(words) * replace_ratio)
    
    # Randomly select positions of words to replace
    positions_to_replace = random.sample(range(len(words)), min(num_to_replace, len(words)))
    
    modified_words = words.copy()
    for pos in positions_to_replace:
        word = words[pos]
        # Only replace alphabetic words
        if word.isalpha():
            synonyms = get_synonyms(word)
            if synonyms:
                # Select a random synonym for replacement
                synonym = random.choice(synonyms)
                modified_words[pos] = synonym
    
    return ' '.join(modified_words)


# Extract text features using pre-trained model (single pass)
def extract_features(texts, tokenizer, model, device, batch_size=32):
    features = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]

        # Tokenize text
        inputs = tokenizer(batch_texts, padding=True, truncation=True, 
                          max_length=512, return_tensors="pt")

        # Move input to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get model output
        with torch.no_grad():
            outputs = model(**inputs)

        # Use [CLS] token representation as features
        batch_features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        features.append(batch_features)

    # Concatenate all batches
    features = np.vstack(features)
    return features


# ===================== Step 2: Repeat input x for n times, calculate distribution P_T(x) =====================
def compute_distribution_for_x(texts, tokenizer, model, device, n_times=10, batch_size=32):
    """
    Step 2: For each input text x, repeat input to model T for n times,
    obtain n feature vectors E(T^1(x)), ..., E(T^n(x)),
    and calculate distribution P_T(x), represented here by mean and variance.
    
    Obtain different feature representations during inference by enabling dropout.
    """
    model.train()  # Enable dropout for diverse outputs
    
    all_means = []
    all_vars = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Store features from n outputs
        batch_features_list = []
        
        for _ in range(n_times):
            # Tokenize text
            inputs = tokenizer(batch_texts, padding=True, truncation=True, 
                              max_length=512, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get model output (results differ each time with dropout enabled)
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Use [CLS] token representation as features
            batch_features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            batch_features_list.append(batch_features)
        
        # Stack features from n times: shape = (n_times, batch_size, feature_dim)
        stacked_features = np.stack(batch_features_list, axis=0)
        
        # Calculate mean and variance for each sample as distribution representation
        # mean shape: (batch_size, feature_dim)
        batch_means = np.mean(stacked_features, axis=0)
        batch_vars = np.var(stacked_features, axis=0)
        
        all_means.append(batch_means)
        all_vars.append(batch_vars)
    
    model.eval()  # Restore eval mode
    
    means = np.vstack(all_means)
    variances = np.vstack(all_vars)
    
    return means, variances


# ===================== Step 3: Repeat input D_aux for t times, calculate distribution P_T(D_aux) =====================
def compute_distribution_for_d_aux(d_aux_texts, tokenizer, model, device, t_times=10, batch_size=32):
    """
    Step 3: For each xi in D_aux (ground truth), repeat input to model T for t times,
    obtain t feature vectors E(T^1(xi)), ..., E(T^t(xi)),
    and calculate distribution P_T(D_aux).
    
    D_aux = ground truth (as per user requirement, this definition is fixed).
    """
    model.train()  # Enable dropout for diverse outputs
    
    all_means = []
    all_vars = []
    
    for i in range(0, len(d_aux_texts), batch_size):
        batch_texts = d_aux_texts[i:i+batch_size]
        
        # Store features from t outputs
        batch_features_list = []
        
        for _ in range(t_times):
            # Tokenize text
            inputs = tokenizer(batch_texts, padding=True, truncation=True, 
                              max_length=512, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get model output
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Use [CLS] token representation as features
            batch_features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            batch_features_list.append(batch_features)
        
        # Stack features from t times
        stacked_features = np.stack(batch_features_list, axis=0)
        
        # Calculate mean and variance for each sample as distribution representation
        batch_means = np.mean(stacked_features, axis=0)
        batch_vars = np.var(stacked_features, axis=0)
        
        all_means.append(batch_means)
        all_vars.append(batch_vars)
    
    model.eval()
    
    means = np.vstack(all_means)
    variances = np.vstack(all_vars)
    
    return means, variances


# Define classifier based on distribution features
class DistributionClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_classes=2):
        super(DistributionClassifier, self).__init__()
        # Input is concatenated mean and variance, hence input_size * 2
        self.fc1 = nn.Linear(input_size * 2, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


# Calculate TPR@FPR=1%
def calculate_tpr_at_fpr(y_true, y_scores, fpr_threshold=0.01):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Find the point closest to the FPR threshold
    idx = np.argmin(np.abs(fpr - fpr_threshold))
    tpr_at_fpr = tpr[idx]

    return tpr_at_fpr


# Main function
def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Hyperparameters
    n_times = 10  # Step 2: Number of times to repeat input x
    t_times = 10  # Step 3: Number of times to repeat input D_aux

    # Load data
    json_path = "/root/autodl-tmp/llava_MINI_output_1000.json"
    data = load_data(json_path)

    # Extract and preprocess text (removing </s> and other special tokens)
    print("Extracting and preprocessing text (removing </s> and other special tokens)...")
    text_1, text_2 = extract_texts(data)
    print(f"Extracted {len(text_1)} VLM texts (x) and {len(text_2)} Ground Truth texts (D_aux)")

    # Print preprocessing example
    if len(text_1) > 0:
        print(f"\nPreprocessed VLM text example: '{text_1[0][:100]}...'")

    # Loading pre-trained model
    print("\nLoading pre-trained model...")
    model_name = "/root/autodl-tmp/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name)
    encoder.to(device)

    # Splitting dataset
    print("Splitting dataset...")
    # text_1 (member/vlm) split 8:2
    text_1_train, text_1_test = train_test_split(text_1, test_size=0.2, random_state=42)
    
    # text_2 (D_aux/ground truth) split 8:2
    text_2_train, text_2_test = train_test_split(text_2, test_size=0.2, random_state=42)

    # Perform 20% synonym replacement on member_test text
    print("Performing 20% synonym replacement on member_test data...")
    modified_text_1_test = [replace_with_synonyms(text, replace_ratio=0.2) for text in text_1_test]

    # ===================== Step 2: Calculate P_T(x) =====================
    print(f"\n[Step 2] Repeating VLM text input {n_times} times to calculate distribution P_T(x)...")
    
    print("Calculating distribution for training VLM text...")
    member_train_means, member_train_vars = compute_distribution_for_x(
        text_1_train, tokenizer, encoder, device, n_times=n_times
    )
    
    print("Calculating distribution for test VLM text (with synonym replacement)...")
    member_test_means, member_test_vars = compute_distribution_for_x(
        modified_text_1_test, tokenizer, encoder, device, n_times=n_times
    )

    # ===================== Step 3: Calculate P_T(D_aux) =====================
    # D_aux = ground truth (this definition is fixed)
    print(f"\n[Step 3] Repeating D_aux (ground truth) input {t_times} times to calculate distribution P_T(D_aux)...")
    
    print("Calculating distribution for training D_aux...")
    non_member_train_means, non_member_train_vars = compute_distribution_for_d_aux(
        text_2_train, tokenizer, encoder, device, t_times=t_times
    )
    
    print("Calculating distribution for test D_aux...")
    non_member_test_means, non_member_test_vars = compute_distribution_for_d_aux(
        text_2_test, tokenizer, encoder, device, t_times=t_times
    )

    # ===================== Step 4: Membership Inference by Comparing Distributions =====================
    print("\n[Step 4] Training classifier to compare distributions P_T(x) and P_T(D_aux)...")
    
    # Concatenate mean and variance as distribution features
    member_train_features = np.concatenate([member_train_means, member_train_vars], axis=1)
    member_test_features = np.concatenate([member_test_means, member_test_vars], axis=1)
    non_member_train_features = np.concatenate([non_member_train_means, non_member_train_vars], axis=1)
    non_member_test_features = np.concatenate([non_member_test_means, non_member_test_vars], axis=1)

    # Prepare training data
    X_train = np.vstack([member_train_features, non_member_train_features])
    y_train = np.concatenate([np.ones(len(member_train_features)), np.zeros(len(non_member_train_features))])

    # Prepare test data
    X_test = np.vstack([member_test_features, non_member_test_features])
    y_test = np.concatenate([np.ones(len(member_test_features)), np.zeros(len(non_member_test_features))])

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize model - input dimension is feature dimension (concatenated mean and variance)
    input_size = member_train_means.shape[1]  # Original feature dimension
    model = DistributionClassifier(input_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train model
    print("Training classification model...")
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # Evaluate model
    print("\nEvaluating model...")
    model.eval()
    with torch.no_grad():
        # Get prediction results
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)

        # Get prediction probabilities
        probabilities = torch.softmax(outputs, dim=1)
        member_probs = probabilities[:, 1].cpu().numpy()
        predicted = predicted.cpu().numpy()
        y_test_np = y_test_tensor.cpu().numpy()

    # Calculate accuracy
    accuracy = accuracy_score(y_test_np, predicted)

    # Calculate AUC
    auc = roc_auc_score(y_test_np, member_probs)

    # Calculate TPR@FPR=1%
    tpr_at_fpr = calculate_tpr_at_fpr(y_test_np, member_probs)

    # Print results
    print("\n" + "="*50)
    print("Evaluation Results:")
    print("="*50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"TPR@FPR=1%: {tpr_at_fpr:.4f}")
    print("="*50)


if __name__ == "__main__":
    main()
