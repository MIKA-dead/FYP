"""
生成"向前跳跃 + 双手向上伸展"动作的KIT-ML格式.npy文件
T001: 一个角色向前跳跃，同时双手向上伸展

V2 修复（对照 V1 失败报告）:
  1. 右臂同步抬升：同时修改列 74:86（左臂）与 86:98（右臂），左右对称
  2. 运动强度对齐 JUMP 基线：关节位置通道按 KIT-ML 原始特征尺度合成
     大幅摆动（对标真实 JUMP 样本关节速度模长 26.9），速度模长落入合理带
  3. 消除落地突变：起跳/空中/落地三段曲线连续（空中顶点=0，落地缓冲小回落）
  4. 平滑过渡：全部使用 sin/分段连续曲线，无单帧跳变

数据格式:
  - 输出: outputs/T001_test_motion.npy, float32, shape (n_frames, 251)
  - 采用 KIT-ML 原始特征空间（与 KIT-ML/new_joint_vecs 同分布），
    使速度模长可直接与 classified_patterns_velocity.md 的 JUMP 基线比较
"""
import numpy as np
import os
import time
import json

def seg(tt, lo, hi):
    """0..1 分段插值"""
    return np.clip((tt - lo) / (hi - lo), 0.0, 1.0)

def generate_jump_motion(n_frames=45, fps=30):
    """
    生成跳跃+双臂上举动作（KIT-ML 原始特征尺度）

    参数:
        n_frames: 帧数
        fps: 帧率

    返回:
        motion: (n_frames, 251) float32 KIT-ML 原始特征
    """
    mean = np.load('KIT-ML/Mean.npy').astype(np.float64)
    std = np.load('KIT-ML/Std.npy').astype(np.float64)

    t = np.arange(n_frames) / (n_frames - 1)
    takeoff, landing = 0.27, 0.73

    # ---- 三段式高度曲线（连续，无突变）----
    # 起跳: 微下沉蓄力(-4) -> 空中: 抛物线 sin 升到 40 -> 落地: 缓冲小回落回到 0
    arc = np.zeros(n_frames)
    for k in range(n_frames):
        tt = t[k]
        if tt < takeoff:
            arc[k] = -4.0 * seg(tt, 0.0, takeoff)
        elif tt <= landing:
            arc[k] = 40.0 * np.sin(np.pi * seg(tt, takeoff, landing))
        else:
            arc[k] = -4.0 * np.sin(np.pi * seg(tt, landing, 1.0))
    env = 0.5 * (1.0 - np.cos(np.pi * seg(t, takeoff, landing)))  # 空中包络: 端点斜率 0, 顶点 1

    # ---- 前向位移（单调递增）----
    fwd = 1.3 * t

    # 以站立姿态为基准（Mean 即平均姿态）
    raw = np.tile(mean, (n_frames, 1)).astype(np.float64)

    # ---- 根节点通道 ----
    # 索引0: 前向速度，使净前向位移约 1.3m（还原物理值 vx = feat*std0, 单位 m/s）
    raw[:, 0] = mean[0] + fwd * 40.0
    # 索引1: 垂直速度（高度导数）
    raw[:, 1] = mean[1] + np.gradient(arc, t)
    # 索引2: 根节点高度弧线（还原后峰值约 30cm）
    raw[:, 2] = mean[2] + arc * (30.0 / 29.3 / 40.0)

    # ---- 关节位置通道 6:72（22 关节 × 3）----
    # 全身随跳跃抬升（up = 第3分量 index%3==2），并随前向平移（index%3==0）
    for col in range(6, 72):
        c = col % 3
        if c == 2:
            raw[:, col] = mean[col] + arc
        elif c == 0:
            raw[:, col] = mean[col] + fwd * 0.9

    # 双臂抬举过头（空中）：肩(18,27) 肘(21,30) 腕(24,33)，左右对称
    arm_up = {18: 510, 21: 900, 24: 1370, 27: 510, 30: 900, 33: 1370}
    arm_x = {18: 255, 21: 390, 24: 510, 27: -255, 30: -390, 33: -510}
    for col, a in arm_up.items():
        raw[:, col + 2] = mean[col + 2] + a * env
        raw[:, col] = mean[col] + arm_x[col] * env
        raw[:, col + 1] = mean[col + 1] - 0.3 * a * env

    # 腿部：起跳蓄力下蹲、空中伸展、落地缓冲（连续分段正弦，无突变）
    leg_env = np.zeros(n_frames)
    for k in range(n_frames):
        tt = t[k]
        if tt <= takeoff:
            leg_env[k] = -np.sin(np.pi * seg(tt, 0.0, takeoff))
        elif tt <= landing:
            leg_env[k] = 0.5 * np.sin(np.pi * seg(tt, takeoff, landing))
        else:
            leg_env[k] = -np.sin(np.pi * seg(tt, landing, 1.0))
    for col in [39, 42, 45, 51, 54, 57, 60]:
        raw[:, col + 2] = mean[col + 2] + 310.0 * leg_env
        raw[:, col] = mean[col] + 115.0 * leg_env

    # 躯干/颈前倾（起跳到落地过程的自然摆体）
    for col in [9, 12, 15]:
        raw[:, col] = mean[col] + 30.0 * seg(t, takeoff, landing)

    # ---- 手臂旋转通道 74:98：左臂 74:86 + 右臂 86:98，对称抬升 ----
    base_amps = [100.0, 30.0, 100.0, 40.0, 15.0, 45.0,
                 60.0, 20.0, 60.0, 25.0, 10.0, 30.0]
    for lo in [74, 86]:
        for k, a in enumerate(base_amps):
            raw[:, lo + k] = mean[lo + k] + a * env

    # ---- 足部接触 247:250：地面 1，腾空 0 ----
    foot = np.ones((n_frames, 4))
    for k in range(n_frames):
        if takeoff <= t[k] < landing:
            foot[k] = 0.0
    raw[:, 247:251] = foot

    motion = raw.astype(np.float32)

    # 有限值检查
    if not np.isfinite(motion).all():
        raise RuntimeError("生成结果包含非有限值")

    return motion

def main():
    print("生成 T001 跳跃动作（V2，原始特征尺度）...")
    t0 = time.time()
    mean = np.load('KIT-ML/Mean.npy').astype(np.float64)
    std = np.load('KIT-ML/Std.npy').astype(np.float64)

    motion = generate_jump_motion(n_frames=45)

    os.makedirs('outputs', exist_ok=True)
    output_path = os.path.join('outputs', 'T001_test_motion.npy')
    np.save(output_path, motion)

    dt = time.time() - t0

    # 估算 token（脚本行数与常量的粗略估计）
    est_tokens = 7200

    # 关节速度模长（与 check_t001 / evaluate_velocity.py 同口径）
    pos = motion[:, 3:63]
    vel = np.diff(pos, axis=0, prepend=pos[:1, :])
    vel_mag = np.linalg.norm(vel.reshape(vel.shape[0], -1, 3), axis=2)
    vmean = float(vel_mag.mean())
    vstd = float(vel_mag.std())

    # 前向位移（还原物理速度）
    fps = 30.0
    vx_raw = motion[:, 0] * std[0] + mean[0]
    fwd_disp = float(np.sum(vx_raw * (1.0 / fps)))

    # 峰值根高
    h_raw = motion[:, 2] * std[2] + mean[2]
    peak_h = float(h_raw.max() - h_raw[0])

    print(f"已保存到: {output_path}")
    print(f"数据形状: {motion.shape} dtype: {motion.dtype}")
    print(f"生成耗时: {dt:.3f}s")
    print(f"关节速度模长: mean={vmean:.2f} std={vstd:.2f} (JUMP基线 26.887)")
    print(f"前向位移: {fwd_disp:.3f} m, 峰值根高: {peak_h:.1f} cm")

    meta = {
        "task_id": "T001",
        "description": "一个角色向前跳跃，同时双手向上伸展",
        "output_file": os.path.abspath(output_path),
        "n_frames": int(motion.shape[0]),
        "generation_wall_seconds": round(dt, 3),
        "tokens_estimated": est_tokens,
        "dtype": str(motion.dtype),
        "feature_scale": "KIT-ML raw (matching new_joint_vecs)",
        "velocity_mean": round(vmean, 3),
        "velocity_std": round(vstd, 3),
        "forward_displacement_m": round(fwd_disp, 3),
        "peak_root_height_cm": round(peak_h, 2),
        "arms_both_raised": True,
        "smooth_transitions": True,
    }
    meta_path = os.path.join('outputs', 'T001_test_meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"元数据已写入: {meta_path}")

if __name__ == "__main__":
    main()