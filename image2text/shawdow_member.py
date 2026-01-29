import os
import json
import numpy as np
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import copy
from sentence_transformers import SentenceTransformer
import warnings

# ---------------------- 配置与初始化 ----------------------
warnings.filterwarnings('ignore')

def clean_text(text):
    """
    清洗文本：移除</s>标签、特殊控制字符及多余空格
    """
    if not text:
        return ""
    # 1. 移除模型结束符 </s>
    text = text.replace('</s>', '')
    # 2. 移除控制字符和不可见字符
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    # 3. 规范化空白字符
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# ---------------------- 特征提取核心逻辑 ----------------------
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0
    return dot_product / (norm_vec1 * norm_vec2)

def compute_all_similarities(vectors_source, vectors_reference):
    """
    计算源向量集中每个向量到参考集所有向量的平均相似度
    """
    avg_similarities = []
    for vs in vectors_source:
        sims = [cosine_similarity(vs, vr) for vr in vectors_reference]
        avg_similarities.append(np.mean(sims))
    return np.array(avg_similarities)

def sample_mean_features(data, sample_times, sample_size):
    """
    对相似度数组进行多次采样，每次采样sample_size个点计算均值，作为训练特征
    """
    features = []
    for _ in range(sample_times):
        sample = np.random.choice(data, size=sample_size, replace=True)
        features.append(np.mean(sample))
    return features

# ---------------------- 二分类器模型 ----------------------
class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def train_model(model, train_loader, val_loader, epochs=25):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    best_loss = float('inf')
    best_wts = None
    
    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.squeeze(), target)
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output.squeeze(), target).item()
        val_loss /= len(val_loader)
        
        scheduler.step(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best_wts = copy.deepcopy(model.state_dict())
            
        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1}/{epochs} | Val Loss: {val_loss:.4f}')
            
    if best_wts:
        model.load_state_dict(best_wts)
    return model

def evaluate_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    probs, labels = [], []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            probs.extend(output.squeeze().cpu().tolist())
            labels.extend(target.cpu().tolist())
            
    probs = np.array(probs)
    labels = np.array(labels)
    preds = (probs > 0.5).astype(int)
    
    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, probs)
    
    # 精确计算 TPR@FPR=1%
    fpr, tpr, _ = roc_curve(labels, probs)
    tpr_at_fpr_1 = np.interp(0.01, fpr, tpr)
    
    return {
        'accuracy': acc,
        'auc': auc,
        'tpr_at_fpr_1': tpr_at_fpr_1
    }

# ---------------------- 主程序 ----------------------
def main():
    # 路径配置
    DATA_PATH = r'/root/autodl-tmp/gpt_1000.json'
    MODEL_NAME = '/root/autodl-tmp/all-MiniLM-L6-v2'  # 可根据实际环境修改
    
    # 1. 加载并清洗数据
    print(f"正在加载数据: {DATA_PATH}")
    raw_data = load_json(DATA_PATH)
    
    vlm_texts = []
    gt_texts = []
    
    for item in raw_data:
        convs = item.get('conversations_0.1', [])
        # 提取当前条目中所有的vlm_1和ground truth回答并拼接
        v_parts = [clean_text(c['value']) for c in convs if c.get('from') == 'vlm_1']
        g_parts = [clean_text(c['value']) for c in convs if c.get('from') == 'ground truth']
        
        v_full = " ".join([p for p in v_parts if p])
        g_full = " ".join([p for p in g_parts if p])
        
        if v_full: vlm_texts.append(v_full)
        if g_full: gt_texts.append(g_full)
        
    print(f"提取完成: VLM_1(Member)样本={len(vlm_texts)}, Ground Truth(Non-member)样本={len(gt_texts)}")

    # 2. 向量化
    print(f"正在使用模型 {MODEL_NAME} 进行向量化...")
    model_st = SentenceTransformer(MODEL_NAME)
    vlm_embeddings = model_st.encode(vlm_texts, show_progress_bar=True)
    gt_embeddings = model_st.encode(gt_texts, show_progress_bar=True)

    # 3. 构造特征 (参考 Shadow Member 思路)
    # 将 Non-member (GT) 划分为两部分：ground_1 (用于测试/训练) 和 ground_ref (作为计算相似度的参考)
    # Member (VLM) 同样计算到 ground_ref 的相似度
    gt_eval, gt_ref = train_test_split(gt_embeddings, test_size=0.3, random_state=42)
    
    print("计算相似度特征...")
    # 计算 Member (VLM) 到参考集的相似度
    member_sims = compute_all_similarities(vlm_embeddings, gt_ref)
    # 计算 Non-member (GT_eval) 到参考集的相似度
    non_member_sims = compute_all_similarities(gt_eval, gt_ref)

    # 4. 采样生成最终训练集
    # 采样参数
    sample_size = 10
    train_times, val_times, test_times = 800, 200, 100
    
    # 划分相似度原始数据
    m_train_raw, m_temp = train_test_split(member_sims, test_size=0.3, random_state=42)
    m_val_raw, m_test_raw = train_test_split(m_temp, test_size=0.33, random_state=42)
    
    nm_train_raw, nm_temp = train_test_split(non_member_sims, test_size=0.3, random_state=42)
    nm_val_raw, nm_test_raw = train_test_split(nm_temp, test_size=0.33, random_state=42)
    
    def build_sampled_dataset(m_data, nm_data, times):
        m_feats = sample_mean_features(m_data, times, sample_size)
        nm_feats = sample_mean_features(nm_data, times, sample_size)
        X = torch.tensor([[f] for f in m_feats + nm_feats], dtype=torch.float32)
        y = torch.tensor([1.0]*times + [0.0]*times, dtype=torch.float32)
        return TensorDataset(X, y)

    train_ds = build_sampled_dataset(m_train_raw, nm_train_raw, train_times)
    val_ds = build_sampled_dataset(m_val_raw, nm_val_raw, val_times)
    test_ds = build_sampled_dataset(m_test_raw, nm_test_raw, test_times)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    test_loader = DataLoader(test_ds, batch_size=32)

    # 5. 模型训练与评估
    print("\n开始分类器训练...")
    classifier = BinaryClassifier(input_dim=1)
    classifier = train_model(classifier, train_loader, val_loader)
    
    print("\n执行最终评估...")
    metrics = evaluate_model(classifier, test_loader)
    
    print("\n" + "="*30)
    print(f"Shadow Member 分类结果:")
    print(f"准确率: {metrics['accuracy']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"TPR@FPR=1%: {metrics['tpr_at_fpr_1']:.4f}")
    print("="*30)

if __name__ == "__main__":
    main()
