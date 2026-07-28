"""
HumanML3D 运动数据集的速度评估脚本
计算运动文件的速度模长统计
使用方法: python scripts/evaluator/evaluate_velocity_humanml3d.py <file.npy>
"""
import numpy as np
import sys
import json
from pathlib import Path

def compute_velocity(motion):
    """
    计算关节速度模长
    motion: (帧数, 251) 的 numpy 数组
    返回: 速度模长均值, 标准差
    """
    # 提取关节位置特征（索引 3~65 是 21 个关节的局部位置，每个 3 维）
    pos_features = motion[:, 3:66]  # (T, 63)
    
    # 计算速度（帧间差分）
    vel = np.diff(pos_features, axis=0, prepend=pos_features[:1, :])  # (T, 63)
    
    # 计算速度模长（每个关节的速度大小）
    vel_mag = np.linalg.norm(vel.reshape(vel.shape[0], -1, 3), axis=2)  # (T, 21)
    
    # 求所有关节、所有帧的平均速度模长
    mean_vel = np.mean(vel_mag)
    std_vel = np.std(vel_mag)
    
    return mean_vel, std_vel

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python evaluate_velocity_humanml3d.py <file.npy>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    motion = np.load(file_path)
    
    mean_vel, std_vel = compute_velocity(motion)
    
    result = {
        "file": str(Path(file_path).name),
        "shape": motion.shape,
        "mean_velocity": float(mean_vel),
        "std_velocity": float(std_vel)
    }
    
    print(json.dumps(result, indent=2))