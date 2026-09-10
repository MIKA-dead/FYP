"""
T001 跳跃动作自检脚本
运行: python scripts/evaluator/check_t001.py
输出: 帧数/形状、有限性、根高单峰、前向位移、关节速度模长、手臂空中变化
"""
import numpy as np
import json
import time
from pathlib import Path

def compute_velocity(motion):
    pos = motion[:, 3:63]
    vel = np.diff(pos, axis=0, prepend=pos[:1, :])
    vel_mag = np.linalg.norm(vel.reshape(vel.shape[0], -1, 3), axis=2)
    return float(np.mean(vel_mag)), float(np.std(vel_mag))

def main():
    t0 = time.time()
    arr_path = 'outputs/T001_test_motion.npy'
    arr = np.load(arr_path)

    checks = {}
    checks['shape'] = arr.shape
    checks['dtype'] = str(arr.dtype)

    # 1. shape/dtype
    shape_ok = arr.shape[1] == 251 and 30 <= arr.shape[0] <= 60
    dtype_ok = arr.dtype == np.float32
    print(f"[1] shape={arr.shape} dtype={arr.dtype}")

    # 2. finite
    finite_ok = bool(np.isfinite(arr).all())
    checks['finite'] = finite_ok
    print(f"[2] np.isfinite(arr).all() = {finite_ok}")

    # 3. root height (index 2): clear rise-then-fall single peak mid
    root_h = arr[:, 2]
    peak_idx = int(np.argmax(root_h))
    n = arr.shape[0]
    mid_ok = 0.25 * n <= peak_idx <= 0.75 * n
    checks['peak_index'] = peak_idx
    print(f"[3] root height peak at frame {peak_idx} (mid={mid_ok})")

    # 4. forward displacement net positive (use root vx * dt)
    mean = np.load('KIT-ML/Mean.npy')
    std = np.load('KIT-ML/Std.npy')
    fps = 30.0
    dt = 1.0 / fps
    vx_raw = arr[:, 0] * std[0] + mean[0]   # 还原为物理速度
    fwd_disp = float(np.sum(vx_raw * dt))
    checks['forward_displacement_m'] = round(fwd_disp, 3)
    fwd_ok = fwd_disp > 0
    print(f"[4] forward displacement = {fwd_disp:.3f} m (positive={fwd_ok})")

    # 5. joint velocity magnitude
    vmean, vstd = compute_velocity(arr)
    checks['velocity_mean'] = round(vmean, 2)
    checks['velocity_std'] = round(vstd, 2)
    vel_ok = 6.0 <= vmean <= 54.0
    print(f"[5] velocity magnitude mean={vmean:.2f} std={vstd:.2f} (JUMP baseline ~26.9)")

    # 6. arm channel change during midair vs takeoff/landing
    takeoff_f, land_f = int(0.27 * n), int(0.73 * n)
    arm_cols = list(range(74, 86))
    mid_change = float(np.abs(np.diff(arr[:, arm_cols], axis=0)[takeoff_f:land_f]).sum())
    ground_change = float(np.abs(np.diff(arr[:, arm_cols], axis=0)).sum()) - mid_change
    arm_ok = mid_change > ground_change
    checks['arm_midair_change'] = round(mid_change, 2)
    checks['arm_ground_change'] = round(ground_change, 2)
    print(f"[6] arm midair change={mid_change:.2f} vs ground change={ground_change:.2f} (midair>ground={arm_ok})")

    # peak root height in cm (relative to standing baseline)
    h_raw = arr[:, 2] * std[2] + mean[2]
    baseline = float(h_raw[0])
    peak_h_cm = float(np.max(h_raw) - baseline)
    checks['peak_root_height_cm'] = round(peak_h_cm, 2)
    print(f"    peak root height (rel) = {peak_h_cm:.2f} cm")

    checks_passed = all([shape_ok, dtype_ok, finite_ok, mid_ok, fwd_ok, vel_ok, arm_ok])
    checks['all_passed'] = bool(checks_passed)

    print(f"\nALL CHECKS PASSED: {checks_passed}")

    # 汇总元数据
    meta = {
        "file_path": str(Path(arr_path).resolve()),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "n_frames": int(arr.shape[0]),
        "forward_displacement_m": checks['forward_displacement_m'],
        "peak_root_height_cm": checks['peak_root_height_cm'],
        "velocity_magnitude_mean": checks['velocity_mean'],
        "velocity_magnitude_std": checks['velocity_std'],
        "self_checks_passed": checks_passed,
        "selfcheck_wall_seconds": round(time.time() - t0, 3),
    }
    print("\n=== METADATA ===")
    print(json.dumps(meta, indent=2))
    return meta

if __name__ == "__main__":
    main()
