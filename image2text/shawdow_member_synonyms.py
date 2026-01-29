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
import random
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords

# ---------------------- 配置与初始化 ----------------------
warnings.filterwarnings('ignore')

# 下载NLTK所需资源
def download_nltk_resources():
    """下载NLTK所需的词库、词性标注器等资源"""
    resources = {
        'corpora/wordnet': 'wordnet',
        'taggers/averaged_perceptron_tagger_eng': 'averaged_perceptron_tagger_eng',
        'tokenizers/punkt_tab': 'punkt_tab'
    }
    for resource_path, resource_name in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            print(f"正在下载NLTK资源: {resource_name}")
            nltk.download(resource_name)

# 初始化停用词表
def init_stopwords():
    """初始化英文停用词表"""
    try:
        stop_words = set(stopwords.words('english'))
        print(f"成功加载停用词表，共{len(stop_words)}个停用词")
        return stop_words
    except LookupError:
        print("未找到停用词资源，正在下载...")
        nltk.download('stopwords')
        return set(stopwords.words('english'))

download_nltk_resources()
STOP_WORDS = init_stopwords()

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

# ---------------------- 同义词替换核心逻辑 ----------------------
def map_nltk_to_wordnet_pos(nltk_tag):
    """将NLTK词性标注映射为WordNet兼容的词性"""
    if nltk_tag.startswith('NN'):
        return wordnet.NOUN
    elif nltk_tag.startswith('VB'):
        return wordnet.VERB
    elif nltk_tag.startswith('JJ'):
        return wordnet.ADJ
    elif nltk_tag.startswith('RB'):
        return wordnet.ADV
    else:
        return None

def get_valid_synonyms(word, wordnet_pos):
    """获取单词在指定词性下的有效同义词"""
    synsets = wordnet.synsets(word, pos=wordnet_pos)
    synonyms = set()
    for ss in synsets:
        for lemma in ss.lemmas():
            synonym = lemma.name().replace('_', ' ')
            if '-' not in synonym and synonym != word and synonym.isalpha():
                synonyms.add(synonym)
    return list(synonyms)

def preserve_case(original_word, replacement_word):
    """保持原始单词的大小写格式"""
    if original_word.isupper():
        return replacement_word.upper()
    elif original_word[0].isupper():
        return replacement_word.capitalize()
    else:
        return replacement_word.lower()

def replace_synonyms(sentence, replace_percent=20):
    """
    对句子进行指定百分比的同义词替换
    :param sentence: 输入句子
    :param replace_percent: 替换百分比（0-100）
    :return: 替换后的句子
    """
    if not sentence.strip():
        return sentence

    try:
        words = nltk.word_tokenize(sentence)
        tagged_words = nltk.pos_tag(words)
    except Exception as e:
        print(f"分词/词性标注失败: {str(e)}，返回原句")
        return sentence

    # 筛选可替换的候选词
    replace_candidates = []
    for idx, (word, tag) in enumerate(tagged_words):
        if (word.lower() not in STOP_WORDS and
                word.isalpha() and
                len(word) > 2):
            wn_pos = map_nltk_to_wordnet_pos(tag)
            if wn_pos:
                synonyms = get_valid_synonyms(word.lower(), wn_pos)
                if synonyms:
                    replace_candidates.append({
                        'idx': idx,
                        'original': word,
                        'synonyms': synonyms
                    })

    if not replace_candidates:
        return sentence

    # 计算替换数量
    replace_percent = max(0, min(100, replace_percent))
    num_to_replace = max(1, round(len(replace_candidates) * replace_percent / 100)) if replace_percent > 0 else 0
    num_to_replace = min(num_to_replace, len(replace_candidates))

    # 随机选择候选词进行替换
    selected_candidates = random.sample(replace_candidates, num_to_replace)
    words_copy = words.copy()

    for candidate in selected_candidates:
        replacement = random.choice(candidate['synonyms'])
        words_copy[candidate['idx']] = preserve_case(candidate['original'], replacement)

    return ' '.join(words_copy)

# ---------------------- 特征提取核心逻辑 ----------------------
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0
    return dot_product / (norm_vec1 * norm_vec2)

def compute_all_similarities(vectors_source, vectors_reference):
    """计算源向量集中每个向量到参考集所有向量的平均相似度"""
    avg_similarities = []
    for vs in vectors_source:
        sims = [cosine_similarity(vs, vr) for vr in vectors_reference]
        avg_similarities.append(np.mean(sims))
    return np.array(avg_similarities)

def sample_mean_features(data, sample_times, sample_size):
    """对相似度数组进行多次采样，每次采样sample_size个点计算均值"""
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
    
    # 精确计算 TPR@FPR=1% 使用线性插值
    fpr, tpr, _ = roc_curve(labels, probs)
    target_fpr = 0.01
    if target_fpr <= fpr[0]:
        tpr_at_fpr_1 = tpr[0]
    elif target_fpr >= fpr[-1]:
        tpr_at_fpr_1 = tpr[-1]
    else:
        tpr_at_fpr_1 = np.interp(target_fpr, fpr, tpr)
    
    # AUC方向性检查
    if auc < 0.5:
        print(f"警告: AUC={auc:.4f}<0.5，可能需要反转分数方向")
    
    return {
        'accuracy': acc,
        'auc': auc,
        'tpr_at_fpr_1': tpr_at_fpr_1
    }

# ---------------------- 主程序 ----------------------
def main():
    # 路径配置
    DATA_PATH = r'/root/autodl-tmp/aPPLE-2100.json'
    MODEL_NAME = '/root/autodl-tmp/all-MiniLM-L6-v2'  # 可根据实际环境修改
    
    # 1. 加载并清洗数据
    print(f"正在加载数据: {DATA_PATH}")
    raw_data = load_json(DATA_PATH)
    
    vlm_texts = []
    gt_texts = []
    
    for item in raw_data:
        convs = item.get('conversations_0.1', [])
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
    gt_eval, gt_ref = train_test_split(gt_embeddings, test_size=0.3, random_state=42)
    
    print("计算相似度特征...")
    member_sims = compute_all_similarities(vlm_embeddings, gt_ref)
    non_member_sims = compute_all_similarities(gt_eval, gt_ref)

    # 4. 采样生成最终训练集
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
    
    print("\n" + "="*50)
    print(f"Shadow Member (原始测试集VLM_1) 分类结果:")
    print(f"准确率: {metrics['accuracy']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"TPR@FPR=1%: {metrics['tpr_at_fpr_1']:.4f}")
    print("="*50)

    # ============ 新增：同义词替换实验 ============
    print("\n\n" + "="*50)
    print("开始进行同义词替换实验...")
    print("="*50)
    
    # 从相似度数据中获取对应的测试集VLM_1原始文本
    # 需要重新划分vlm_texts以获得对应的测试集索引
    vlm_train_idx, vlm_temp_idx = train_test_split(
        range(len(vlm_texts)), test_size=0.3, random_state=42
    )
    vlm_val_idx, vlm_test_idx = train_test_split(
        vlm_temp_idx, test_size=0.33, random_state=42
    )
    
    vlm_test_texts = [vlm_texts[i] for i in vlm_test_idx]
    
    # Step 2: 对测试集VLM_1进行20%同义词替换
    print(f"\n对测试集VLM_1进行20%同义词替换...")
    print(f"原始测试集大小: {len(vlm_test_texts)}")
    
    vlm_test_synonyms = []
    for idx, text in enumerate(vlm_test_texts):
        if idx % 100 == 0 and idx > 0:
            print(f"已完成 {idx}/{len(vlm_test_texts)} 条文本的替换")
        replaced_text = replace_synonyms(text, replace_percent=20)
        vlm_test_synonyms.append(replaced_text)
    
    # 打印替换示例
    print("\n同义词替换示例 (前5条):")
    for i in range(min(5, len(vlm_test_texts))):
        print(f"\n示例 {i+1}:")
        print(f"原文: {vlm_test_texts[i][:150]}{'...' if len(vlm_test_texts[i]) > 150 else ''}")
        print(f"替换后: {vlm_test_synonyms[i][:150]}{'...' if len(vlm_test_synonyms[i]) > 150 else ''}")
    
    # Step 3: 对替换后的测试集进行向量化和评估
    print("\n\n正在对替换后的测试集进行向量化...")
    vlm_test_synonyms_embeddings = model_st.encode(vlm_test_synonyms, show_progress_bar=True)
    
    print("计算替换后测试集的相似度特征...")
    vlm_test_synonyms_sims = compute_all_similarities(vlm_test_synonyms_embeddings, gt_ref)
    
    # 生成替换后测试集的特征
    m_test_synonyms_feats = sample_mean_features(vlm_test_synonyms_sims, test_times, sample_size)
    nm_test_feats = sample_mean_features(nm_test_raw, test_times, sample_size)
    
    X_test_synonyms = torch.tensor(
        [[f] for f in m_test_synonyms_feats + nm_test_feats], 
        dtype=torch.float32
    )
    y_test_synonyms = torch.tensor([1.0]*test_times + [0.0]*test_times, dtype=torch.float32)
    test_synonyms_ds = TensorDataset(X_test_synonyms, y_test_synonyms)
    test_synonyms_loader = DataLoader(test_synonyms_ds, batch_size=32)
    
    print("评估替换后测试集的性能...")
    metrics_synonyms = evaluate_model(classifier, test_synonyms_loader)
    
    print("\n" + "="*50)
    print(f"Shadow Member (同义词替换后VLM_1) 分类结果:")
    print(f"准确率: {metrics_synonyms['accuracy']:.4f}")
    print(f"AUC: {metrics_synonyms['auc']:.4f}")
    print(f"TPR@FPR=1%: {metrics_synonyms['tpr_at_fpr_1']:.4f}")
    print("="*50)
    
    # 对比结果
    print("\n" + "="*50)
    print("对比分析 (原始 vs 同义词替换):")
    print(f"准确率变化: {metrics['accuracy']:.4f} → {metrics_synonyms['accuracy']:.4f} "
          f"({metrics_synonyms['accuracy']-metrics['accuracy']:+.4f})")
    print(f"AUC变化: {metrics['auc']:.4f} → {metrics_synonyms['auc']:.4f} "
          f"({metrics_synonyms['auc']-metrics['auc']:+.4f})")
    print(f"TPR@FPR=1%变化: {metrics['tpr_at_fpr_1']:.4f} → {metrics_synonyms['tpr_at_fpr_1']:.4f} "
          f"({metrics_synonyms['tpr_at_fpr_1']-metrics['tpr_at_fpr_1']:+.4f})")
    print("="*50)

if __name__ == "__main__":
    main()
