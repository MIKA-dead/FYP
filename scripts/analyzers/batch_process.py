"""
批量评估 KIT-ML 数据集，生成知识库草稿
使用方法: python scripts/analyzers/batch_process.py
"""
import numpy as np
import os
import json
import sys
from pathlib import Path
from tqdm import tqdm

# 添加 evaluator 模块到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'evaluator'))
from evaluate_motion import MotionEvaluator

# --- 配置 ---
PROJECT_ROOT = Path(__file__).parent.parent.parent  # 回到项目根目录
KIT_ML_DIR = PROJECT_ROOT / 'KIT-ML'
VECS_DIR = KIT_ML_DIR / 'new_joint_vecs'
TEXTS_DIR = KIT_ML_DIR / 'texts'
OUTPUT_DIR = PROJECT_ROOT / 'metrics'
OUTPUT_JSON = OUTPUT_DIR / 'batch_eval_results.json'
KNOWLEDGE_MD = PROJECT_ROOT / '.agents' / 'skills' / 'motion-patterns' / 'auto_learned_patterns.md'

# 确保输出目录存在
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
KNOWLEDGE_MD.parent.mkdir(parents=True, exist_ok=True)

def convert_to_serializable(obj):
    """递归将 numpy 类型和 Python 原生类型转换为 JSON 可序列化格式"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, bool):           # 新增：处理布尔值
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (str, int, float)):
        return obj
    elif obj is None:
        return None
    else:
        # 对于未知类型，尝试转换为字符串
        return str(obj)

# --- 初始化评估器 ---
evaluator = MotionEvaluator(kit_ml_dir=str(KIT_ML_DIR))

# --- 获取所有 .npy 文件 ---
npy_files = [f for f in os.listdir(VECS_DIR) if f.endswith('.npy')]
print(f"找到 {len(npy_files)} 个运动文件")

# --- 批量评估 ---
results = []
high_quality_samples = []  # 评分 >= 85

for file in tqdm(npy_files, desc="评估中"):
    file_path = VECS_DIR / file
    try:
        result = evaluator.evaluate(str(file_path))
        result['file'] = file
        results.append(result)
        
        if result['score'] >= 85:
            motion = np.load(file_path)
            high_quality_samples.append({
                'file': file,
                'score': result['score'],
                'mean': np.mean(motion, axis=0).tolist(),
                'std': np.std(motion, axis=0).tolist(),
                'shape': (int(motion.shape[0]), int(motion.shape[1]))  # 转为 Python int
            })
    except Exception as e:
        print(f"处理 {file} 时出错: {e}")

# --- 转换 results 为可序列化格式 ---
results_serializable = convert_to_serializable(results)

# --- 保存完整结果 ---
with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(results_serializable, f, indent=2, ensure_ascii=False)
print(f"完整评估结果已保存至 {OUTPUT_JSON}")

# --- 生成知识库 Markdown ---
lines = []
lines.append("# KIT-ML 自动学习的运动模式\n")
lines.append(f"生成时间: {np.datetime64('now').astype(str)}\n")
lines.append(f"样本总数: {len(npy_files)}, 高质量样本(评分≥85): {len(high_quality_samples)}\n\n")

if high_quality_samples:
    # 按评分排序
    sorted_samples = sorted(high_quality_samples, key=lambda x: x['score'], reverse=True)
    
    # Top 20 列表
    lines.append("## 最优质样本 (Top 20)\n")
    lines.append("| 文件 | 评分 | 帧数 | 特征维度 |")
    lines.append("|------|------|------|---------|")
    for s in sorted_samples[:20]:
        lines.append(f"| {s['file']} | {s['score']:.1f} | {s['shape'][0]} | {s['shape'][1]} |")
    
    # 典型原型（所有高质量样本的均值）
    all_means = np.array([s['mean'] for s in sorted_samples])
    prototype_mean = np.mean(all_means, axis=0)
    prototype_std = np.std(all_means, axis=0)
    
    lines.append("\n## 典型原型特征（基于所有高质量样本）\n")
    lines.append(f"- 根节点高度（索引2）: 均值 {prototype_mean[2]:.3f}, 标准差 {prototype_std[2]:.3f}")
    
    # 足部接触（最后4维）
    foot_idx = -4
    foot_mean = np.mean(prototype_mean[foot_idx:])
    foot_std = np.std(prototype_std[foot_idx:])
    lines.append(f"- 足部接触概率（最后4维平均）: 均值 {foot_mean:.3f}, 标准差 {foot_std:.3f}")
    
    lines.append("\n## 使用建议")
    lines.append("- 生成行走动作时，根节点高度应保持在约 {:.2f} ~ {:.2f} 区间".format(
        prototype_mean[2] - prototype_std[2], 
        prototype_mean[2] + prototype_std[2]
    ))
    lines.append("- 足部接触比例建议控制在 0.2 ~ 0.5 之间")
else:
    lines.append("未发现高质量样本（评分 ≥ 85）")

# 保存知识库文件
with open(KNOWLEDGE_MD, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print(f"知识库草稿已保存至 {KNOWLEDGE_MD}")