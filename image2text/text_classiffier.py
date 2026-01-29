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
import re

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


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
    text_2 = []  # Ground Truth text

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


# ==================== Step 2 & Step 3 Improved Implementation ====================
class FeatureExtractorWithDropout(nn.Module):
    """
    Wraps pre-trained model with controllable dropout layer for Monte Carlo sampling
    """
    def __init__(self, base_model, dropout_rate=0.2):
        super(FeatureExtractorWithDropout, self).__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(p=dropout_rate)
    
    def forward(self, **inputs):
        outputs = self.base_model(**inputs)
        # Apply dropout to CLS token output
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        return cls_output


def extract_features_with_distribution(texts, tokenizer, model, device, n_times=20, batch_size=32, dropout_rate=0.2):
    """
    Step 2 & 3 Improved Implementation:
    Perform n inferences for each text using Monte Carlo Dropout to simulate multiple query distributions.
    
    Improvements:
    1. Use independent dropout layer with controllable dropout rate.
    2. Increase sampling times (n_times) for more stable distribution estimation.
    3. Calculate richer distribution statistics.
    
    Parameters:
        texts: List of input texts
        tokenizer: Tokenizer
        model: Pre-trained model
        device: Calculation device
        n_times: Sampling times (simulating multiple queries)
        batch_size: Batch size
        dropout_rate: Monte Carlo Dropout rate
    
    Returns:
        Distribution features for each sample [mean, std, min, max, median, q25, q75]
    """
    all_distribution_features = []
    
    # Create feature extractor with dropout
    feature_extractor = FeatureExtractorWithDropout(model, dropout_rate=dropout_rate).to(device)
    feature_extractor.train()  # Keep dropout enabled
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize text
        inputs = tokenizer(batch_texts, padding=True, truncation=True,
                           max_length=512, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        batch_multi_features = []
        
        # Perform n inferences per batch (simulating multiple VLM queries)
        for _ in range(n_times):
            with torch.no_grad():
                batch_features = feature_extractor(**inputs).cpu().numpy()
            batch_multi_features.append(batch_features)
        
        # Stack results from n inferences: shape = (n_times, batch_size, hidden_size)
        batch_multi_features = np.stack(batch_multi_features, axis=0)
        
        # Calculate distribution features for each sample
        for j in range(len(batch_texts)):
            sample_features = batch_multi_features[:, j, :]  # shape = (n_times, hidden_size)
            
            # Calculate distribution statistics (richer features)
            mean_feat = np.mean(sample_features, axis=0)      # Mean
            std_feat = np.std(sample_features, axis=0)        # Std dev
            min_feat = np.min(sample_features, axis=0)        # Min
            max_feat = np.max(sample_features, axis=0)        # Max
            median_feat = np.median(sample_features, axis=0)  # Median
            q25_feat = np.percentile(sample_features, 25, axis=0)  # 25th percentile
            q75_feat = np.percentile(sample_features, 75, axis=0)  # 75th percentile
            
            # Merge distribution features
            distribution_feat = np.concatenate([
                mean_feat, std_feat, min_feat, max_feat, 
                median_feat, q25_feat, q75_feat
            ])
            all_distribution_features.append(distribution_feat)
    
    return np.array(all_distribution_features)


# ==================== Classifier Definition ====================
class DistributionClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=512, num_classes=2):
        super(DistributionClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# Calculate TPR@FPR=1%
def calculate_tpr_at_fpr(y_true, y_scores, fpr_threshold=0.01):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    idx = np.argmin(np.abs(fpr - fpr_threshold))
    tpr_at_fpr = tpr[idx]
    return tpr_at_fpr


# Main function
def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    json_path = "/root/autodl-tmp/llava_MINI_output_1000.json"
    data = load_data(json_path)

    # Extract and preprocess text (removing </s> and other special tokens)
    print("Extracting and preprocessing text (removing </s> and other special tokens)...")
    text_1, text_2 = extract_texts(data)
    print(f"Extracted {len(text_1)} VLM texts (member) and {len(text_2)} Ground Truth texts (D_aux/non-member)")
    
    # Print preprocessing example
    if len(text_1) > 0:
        print(f"\nPreprocessed VLM text example: '{text_1[0][:100]}...'")

    # Loading pre-trained model
    print("\nLoading pre-trained model...")
    model_name = "/root/autodl-tmp/bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name)
    encoder.to(device)

    # Set inference parameters
    n_times = 20  # Perform n inferences per sample (increased sampling times)
    dropout_rate = 0.2  # Monte Carlo Dropout rate
    print(f"Performing {n_times} inferences per sample to calculate distribution features (dropout_rate={dropout_rate})...")

    # Step 2: Extract distribution features P_T(x) for member (VLM) text
    print("\nStep 2: Extracting VLM text distribution features P_T(x)...")
    features_1 = extract_features_with_distribution(
        text_1, tokenizer, encoder, device, 
        n_times=n_times, dropout_rate=dropout_rate
    )
    print(f"Member distribution features dimension: {features_1.shape}")

    # Step 3: Extract distribution features P_T(D_aux) for Ground Truth (D_aux) text
    print("\nStep 3: Extracting Ground Truth (D_aux) distribution features P_T(D_aux)...")
    features_2 = extract_features_with_distribution(
        text_2, tokenizer, encoder, device, 
        n_times=n_times, dropout_rate=dropout_rate
    )
    print(f"Non-member (D_aux) distribution features dimension: {features_2.shape}")

    # Step 4: Train classifier to distinguish between the two distributions
    print("\nSplitting dataset...")
    member_train, member_test = train_test_split(features_1, test_size=0.2, random_state=42)
    non_member_train, non_member_test = train_test_split(features_2, test_size=0.2, random_state=42)

    # Prepare training data
    X_train = np.vstack([member_train, non_member_train])
    y_train = np.concatenate([np.ones(len(member_train)), np.zeros(len(non_member_train))])

    # Prepare test data
    X_test = np.vstack([member_test, non_member_test])
    y_test = np.concatenate([np.ones(len(member_test)), np.zeros(len(non_member_test))])

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize model (input dimension is 7x original: mean, std, min, max, median, q25, q75)
    input_size = X_train.shape[1]
    print(f"Classifier input dimension: {input_size}")
    model = DistributionClassifier(input_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train model
    print("\nStep 4: Training classification model (based on distribution features)...")
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
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

    # Evaluate model
    print("\nEvaluating model...")
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.softmax(outputs, dim=1)
        member_probs = probabilities[:, 1].cpu().numpy()
        predicted = predicted.cpu().numpy()
        y_test_np = y_test_tensor.cpu().numpy()

    # Calculate metrics
    accuracy = accuracy_score(y_test_np, predicted)
    auc = roc_auc_score(y_test_np, member_probs)
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


if __name__ == "__main__":
    main()
