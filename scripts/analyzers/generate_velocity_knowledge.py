"""
生成包含速度特征的分类知识库（使用所有样本，忽略置信度）
使用方法: python scripts/analyzers/generate_velocity_knowledge.py
"""
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent.parent
VECS_DIR = PROJECT_ROOT / 'KIT-ML' / 'new_joint_vecs'
CLASS_FILE = PROJECT_ROOT / 'metrics' / 'action_classification_with_conf.json'
OUTPUT_DIR = PROJECT_ROOT / '.agents' / 'skills' / 'motion-patterns'
OUTPUT_FILE = OUTPUT_DIR / 'classified_patterns_velocity.md'

def load_classification():
    with open(CLASS_FILE, 'r') as f:
        return json.load(f)

def compute_statistics(file_list, vecs_dir):
    """计算一组文件的位置和速度统计"""
    if not file_list:
        return None
    all_motions = []
    for f in file_list:
        try:
            motion = np.load(vecs_dir / f"{f}.npy")
            all_motions.append(motion)
        except Exception as e:
            pass
    if not all_motions:
        return None
    
    # 堆叠所有帧
    stacked = np.vstack(all_motions)
    
    # 计算位置统计
    mean_pos = np.mean(stacked, axis=0)
    std_pos = np.std(stacked, axis=0)
    
    # 提取关节位置特征（假设索引 3~65 是 21 个关节的局部位置，每个 3 维）
    pos_features = stacked[:, 3:66]  # 如果数据维度是 251，则 3:66 正好是 63 维（21*3）
    # 计算速度（帧间差分，用前向差分并保留长度）
    vel = np.diff(pos_features, axis=0, prepend=pos_features[:1, :])  # (T, 63)
    mean_vel = np.mean(vel, axis=0)
    std_vel = np.std(vel, axis=0)
    # 计算速度的模长（每个关节的速度大小）
    vel_mag = np.linalg.norm(vel.reshape(vel.shape[0], -1, 3), axis=2)  # (T, 21)
    mean_vel_mag = np.mean(vel_mag)
    std_vel_mag = np.std(vel_mag)
    
    frames = [m.shape[0] for m in all_motions]
    return {
        'mean_pos': mean_pos,
        'std_pos': std_pos,
        'mean_vel': mean_vel,
        'std_vel': std_vel,
        'mean_vel_mag': mean_vel_mag,
        'std_vel_mag': std_vel_mag,
        'frames': frames,
        'count': len(all_motions)
    }

def main():
    classification = load_classification()
    
    # 按标签分组（不筛选置信度，使用所有样本）
    groups = defaultdict(list)
    for file, info in classification.items():
        groups[info['label']].append(file)
    
    lines = []
    lines.append("# KIT-ML 分类运动模式（含速度特征）\n")
    lines.append(f"生成时间: {np.datetime64('now').astype(str)}\n")
    lines.append("说明：本知识库同时提供位置和速度统计。速度特征对动作类型更敏感，建议在评估时优先使用速度模长均值。\n")
    
    # 先计算所有样本的总体统计（作为默认参考）
    all_files = [f for v in groups.values() for f in v]
    all_stats = compute_statistics(all_files, VECS_DIR)
    if all_stats:
        lines.append("## 总体（全部样本）\n")
        lines.append("| 特征 | 均值 | 标准差 |")
        lines.append("|------|------|--------|")
        lines.append(f"| 根节点高度 (索引2) | {all_stats['mean_pos'][2]:.3f} | {all_stats['std_pos'][2]:.3f} |")
        lines.append(f"| 关节速度模长 | {all_stats['mean_vel_mag']:.3f} | {all_stats['std_vel_mag']:.3f} |")
        lines.append(f"| 帧数范围 | {min(all_stats['frames'])} ~ {max(all_stats['frames'])} | 平均 {np.mean(all_stats['frames']):.1f} |")
        lines.append("\n")
    
    # 按动作类型输出
    for label, files in sorted(groups.items(), key=lambda x: -len(x[1])):
        if len(files) < 5:
            continue
        stats = compute_statistics(files, VECS_DIR)
        if stats is None:
            continue
        
        lines.append(f"## {label.upper()} ({len(files)} 个样本)\n")
        lines.append("| 特征 | 均值 | 标准差 | 说明 |")
        lines.append("|------|------|--------|------|")
        lines.append(f"| 根节点高度 (索引2) | {stats['mean_pos'][2]:.3f} | {stats['std_pos'][2]:.3f} | 位置特征，受数据采集影响大 |")
        lines.append(f"| 关节速度模长 | {stats['mean_vel_mag']:.3f} | {stats['std_vel_mag']:.3f} | **推荐使用**，反映运动强度 |")
        lines.append(f"| 帧数范围 | {min(stats['frames'])} ~ {max(stats['frames'])} | 平均 {np.mean(stats['frames']):.1f} | |")
        lines.append("\n")
    
    lines.append("## 使用建议\n")
    lines.append("1. 对于动作类型识别，优先使用**关节速度模长**的均值和标准差作为评估基准。")
    lines.append("2. 根节点高度仅作为辅助参考，因为其值易受坐标缩放影响。")
    lines.append("3. 如果生成的动作速度模长明显偏离对应动作类型的均值范围，应视为不符合预期。")
    
    # 写入文件
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"知识库已生成：{OUTPUT_FILE}")
    print("包含各动作类型的位置和速度统计。")

if __name__ == "__main__":
    main()