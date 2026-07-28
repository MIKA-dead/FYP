"""
生成行走动作的KIT-ML格式.npy文件
"""
import numpy as np
import os

def generate_walking_motion(n_frames=47, fps=30):
    """
    生成行走动作
    
    参数:
        n_frames: 帧数
        fps: 帧率
    
    返回:
        motion: (n_frames, 251) 的numpy数组
    """
    # 加载统计量
    mean = np.load('KIT-ML/Mean.npy')
    std = np.load('KIT-ML/Std.npy')
    
    # 初始化运动数据（使用均值作为基础）
    motion = np.tile(mean, (n_frames, 1))
    
    # 时间序列
    t = np.linspace(0, 2 * np.pi, n_frames)
    
    # 行走周期参数
    walk_freq = 1.5  # 行走频率 (Hz)
    phase = t * walk_freq
    
    # 1. 根节点速度 (root_vx, root_vz) - 索引 0, 1
    # 行走时前进速度
    motion[:, 0] = mean[0] + 0.8 * std[0] * np.ones(n_frames)  # root_vx (前进)
    motion[:, 1] = mean[1] + 0.1 * std[1] * np.sin(phase * 2)  # root_vz (左右摇摆)
    
    # 2. 根节点高度 (root_height) - 索引 2
    # 行走时上下起伏
    motion[:, 2] = mean[2] + 0.3 * std[2] * np.abs(np.sin(phase))
    
    # 3. 腿部关节运动 (关节 8-13, 大腿和小腿)
    # 左腿 (joint 8, 10, 12)
    leg_swing = 0.6 * np.sin(phase)
    motion[:, 56:59] = mean[56:59] + 0.4 * std[56:59] * leg_swing[:, np.newaxis]  # joint_8 位置
    motion[:, 62:65] = mean[62:65] + 0.3 * std[62:65] * leg_swing[:, np.newaxis]  # joint_10 位置
    motion[:, 68:71] = mean[68:71] + 0.5 * std[68:71] * leg_swing[:, np.newaxis]  # joint_12 位置
    
    # 右腿 (joint 9, 11, 13) - 与左腿相反
    motion[:, 59:62] = mean[59:62] + 0.4 * std[59:62] * (-leg_swing)[:, np.newaxis]  # joint_9 位置
    motion[:, 65:68] = mean[65:68] + 0.3 * std[65:68] * (-leg_swing)[:, np.newaxis]  # joint_11 位置
    motion[:, 71:74] = mean[71:74] + 0.5 * std[71:74] * (-leg_swing)[:, np.newaxis]  # joint_13 位置
    
    # 4. 腿部速度
    leg_vel = 0.6 * np.cos(phase) * walk_freq
    motion[:, 119:122] = mean[119:122] + 0.4 * std[119:122] * leg_vel[:, np.newaxis]
    motion[:, 125:128] = mean[125:128] + 0.3 * std[125:128] * leg_vel[:, np.newaxis]
    motion[:, 131:134] = mean[131:134] + 0.5 * std[131:134] * leg_vel[:, np.newaxis]
    
    # 右腿速度
    motion[:, 122:125] = mean[122:125] + 0.4 * std[122:125] * (-leg_vel)[:, np.newaxis]
    motion[:, 128:131] = mean[128:131] + 0.3 * std[128:131] * (-leg_vel)[:, np.newaxis]
    motion[:, 134:137] = mean[134:137] + 0.5 * std[134:137] * (-leg_vel)[:, np.newaxis]
    
    # 5. 手臂摆动 (关节 14-17)
    arm_swing = 0.3 * np.sin(phase)
    # 左臂
    motion[:, 74:77] = mean[74:77] + 0.2 * std[74:77] * (-arm_swing)[:, np.newaxis]
    motion[:, 80:83] = mean[80:83] + 0.15 * std[80:83] * (-arm_swing)[:, np.newaxis]
    # 右臂
    motion[:, 77:80] = mean[77:80] + 0.2 * std[77:80] * arm_swing[:, np.newaxis]
    motion[:, 83:86] = mean[83:86] + 0.15 * std[83:86] * arm_swing[:, np.newaxis]
    
    # 6. 足部接触 (索引 247-250)
    # 行走时交替接触地面
    foot_contact = np.zeros((n_frames, 4))
    foot_contact[:, 0] = np.where(np.sin(phase) > 0, 0.8, 0.2)  # 左脚
    foot_contact[:, 2] = np.where(np.sin(phase) > 0, 0.2, 0.8)  # 右脚
    motion[:, 247:251] = foot_contact
    
    # 7. 添加轻微的随机扰动以增加自然感
    noise = np.random.normal(0, 0.02, motion.shape) * std
    motion += noise
    
    return motion

def main():
    print("生成行走动作...")
    
    # 生成运动
    motion = generate_walking_motion(n_frames=47)
    
    # 确保输出目录存在
    output_dir = 'KIT-ML/new_joint_vecs'
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存为.npy文件
    output_path = os.path.join(output_dir, 'walking_001.npy')
    np.save(output_path, motion)
    
    print(f"已保存到: {output_path}")
    print(f"数据形状: {motion.shape}")
    
    # 使用评估脚本验证
    print("\n进行质量评估...")
    from evaluate_motion import MotionEvaluator
    evaluator = MotionEvaluator()
    result = evaluator.evaluate(output_path)
    
    print(f"评分: {result['score']:.1f}/100")
    print(f"异常比例: {result['bounds_check']['outlier_ratio']:.2%}")
    print(f"高度异常比例: {result['height_outlier_ratio']:.2%}")
    print(f"建议: {result['recommendations']}")

if __name__ == "__main__":
    main()
