import numpy as np
import os

# 加载数据
mean = np.load('KIT-ML/Mean.npy')
std = np.load('KIT-ML/Std.npy')

# 定义输出文件
output_file = 'motion_rules.txt'

# 特征名称（仅做注释用，实际按索引即可）
feature_names = {
    0: '根节点X速度', 1: '根节点Z速度', 2: '根节点高度',
    # 3-65: 关节位置，66-128: 关节速度，129-254: 关节旋转，255-258: 足部接触
}

with open(output_file, 'w') as f:
    f.write("# KIT-ML 数据集特征合理性规则\n")
    f.write("# 格式：特征索引 | 均值 | 标准差 | 合理区间下限 | 合理区间上限\n")
    f.write("# 合理区间定义为 均值 ± 2 * 标准差\n\n")
    for i in range(len(mean)):
        lower = mean[i] - 2 * std[i]
        upper = mean[i] + 2 * std[i]
        f.write(f"{i:3d} | {mean[i]:8.4f} | {std[i]:8.4f} | {lower:8.4f} | {upper:8.4f}\n")

print(f"规则已生成到 {output_file}")