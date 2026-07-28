"""
使用句子嵌入对 KIT-ML 动作描述进行语义分类，并保存置信度
使用方法: python scripts/analyzers/classify_by_embedding.py
"""

import os
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
TEXTS_DIR = PROJECT_ROOT / 'KIT-ML' / 'texts'
VECS_DIR = PROJECT_ROOT / 'KIT-ML' / 'new_joint_vecs'
OUTPUT_DIR = PROJECT_ROOT / 'metrics'
OUTPUT_FILE = OUTPUT_DIR / 'action_classification_with_conf.json'

# 动作标签候选集
ACTION_LABELS = {
    'walk': 'walking or stepping forward',
    'run': 'running or jogging fast',
    'jump': 'jumping or leaping up',
    'kick': 'kicking with leg or foot',
    'dance': 'dancing or swaying rhythmically',
    'sit': 'sitting down or crouching',
    'stand': 'standing up or rising',
    'turn': 'turning or rotating body',
    'reach': 'reaching for something with hand',
    'throw': 'throwing an object',
}

# 加载模型
model = SentenceTransformer('all-MiniLM-L6-v2')

# 预编码标签描述
label_descs = list(ACTION_LABELS.values())
label_emb = model.encode(label_descs, convert_to_tensor=True)

def classify_text(text):
    """返回 (标签, 置信度)"""
    text_emb = model.encode(text, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(text_emb, label_emb)[0]
    best_idx = torch.argmax(cos_scores).item()
    best_label = list(ACTION_LABELS.keys())[best_idx]
    confidence = cos_scores[best_idx].item()
    return best_label, confidence

# --- 主流程 ---
npy_files = [f.replace('.npy', '') for f in os.listdir(VECS_DIR) if f.endswith('.npy')]
print(f"找到 {len(npy_files)} 个运动文件")

results = {}
label_counts = {}

for file in tqdm(npy_files, desc="分类中"):
    txt_path = TEXTS_DIR / f"{file}.txt"
    if not txt_path.exists():
        results[file] = {'label': 'unknown', 'confidence': 0.0}
        continue
    with open(txt_path, 'r', encoding='utf-8') as f:
        desc = f.readline().strip()
    label, conf = classify_text(desc)
    results[file] = {'label': label, 'confidence': conf}
    label_counts[label] = label_counts.get(label, 0) + 1

# 保存结果（包含置信度）
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"分类结果已保存至 {OUTPUT_FILE}")
print("\n动作类型分布:")
for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
    print(f"  {label}: {count} ({count/len(npy_files)*100:.1f}%)")

avg_conf = np.mean([v['confidence'] for v in results.values() if v['label'] != 'unknown'])
print(f"\n平均置信度: {avg_conf:.3f}")