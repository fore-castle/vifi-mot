"""Phase 6 补充分析：FTM 系统性偏移 vs 随机噪声分解 + MOT 增益上限估算

核心发现：ACF lag-1=0.93 + 92.6% 低频能量 + 滤波仅降 2.7% MAE
→ 说明 FTM 噪声的主体是慢变化的系统偏移（非对称 bias），不是随机抖动

需要回答：
1. FTM 误差中有多少是可通过 temporal model 消除的？
2. 不同去噪水平对应多少 MOT IDF1 增益？（通过 oracle 实验评估）
"""

import pickle
import numpy as np
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_ROOT = Path("/Users/zstar/auto_search/vifi-mot/RAN4model_dfv4p4/seqs/outdoor")
SCENES = ["scene1", "scene2", "scene3", "scene4"]


def load_all_ftm_depth_pairs():
    """加载所有 outdoor 序列的 (FTM, depth) 时间序列对"""
    all_series = []
    for scene in SCENES:
        scene_dir = DATA_ROOT / scene
        if not scene_dir.exists():
            continue
        for seq_dir in sorted(scene_dir.iterdir()):
            sync_dir = seq_dir / "sync_ts16_dfv4p4"
            ftm_path = sync_dir / "FTM_li_sync_dfv4p4.pkl"
            bbx_path = sync_dir / "BBX5_sync_dfv4p4.pkl"
            if not ftm_path.exists() or not bbx_path.exists():
                continue
            with open(ftm_path, "rb") as f:
                ftm = pickle.load(f)
            with open(bbx_path, "rb") as f:
                bbx5 = pickle.load(f)
            T, N = ftm.shape[0], ftm.shape[1]
            for s in range(N):
                ftm_range = ftm[:, s, 0, 0]  # mm
                ftm_std = ftm[:, s, 0, 1]    # mm
                depth = bbx5[:, s, 0, 2]      # meters
                valid = (depth > 0.1) & (np.abs(ftm_range) > 10)
                if valid.sum() < 50:
                    continue
                all_series.append({
                    "scene": scene,
                    "seq": seq_dir.name,
                    "phone_idx": s,
                    "ftm_mm": ftm_range,
                    "ftm_std_mm": ftm_std,
                    "depth_m": depth,
                    "valid": valid,
                    "T": T,
                })
    return all_series


def main():
    print("=" * 70)
    print("Phase 6 补充分析：偏移分解 + MOT 增益上限")
    print("=" * 70)
    
    all_series = load_all_ftm_depth_pairs()
    print(f"\n总序列数: {len(all_series)}, 总有效帧: {sum(s['valid'].sum() for s in all_series):,}")
    
    # =========================================================================
    # Part 1: 噪声分解 — 系统性偏移 vs 随机噪声
    # =========================================================================
    print("\n" + "=" * 70)
    print("Part 1: 噪声分解 — per-sequence bias vs residual noise")
    print("=" * 70)
    
    # 假设 FTM 噪声 = per-sequence constant bias + time-varying residual
    # noise(t) = bias + residual(t)
    # bias = mean(noise) over the sequence
    # residual(t) = noise(t) - bias
    
    biases = []
    residual_stds = []
    residual_maes = []
    raw_maes = []
    
    for s in all_series:
        ftm_m = s["ftm_mm"][s["valid"]] / 1000.0
        depth_m = s["depth_m"][s["valid"]]
        noise = ftm_m - depth_m
        
        bias = noise.mean()
        residual = noise - bias
        
        biases.append(bias)
        residual_stds.append(residual.std())
        residual_maes.append(np.abs(residual).mean())
        raw_maes.append(np.abs(noise).mean())
    
    biases = np.array(biases)
    residual_stds = np.array(residual_stds)
    residual_maes = np.array(residual_maes)
    raw_maes = np.array(raw_maes)
    
    print(f"\n  Per-sequence bias:")
    print(f"    mean(|bias|): {np.abs(biases).mean():.4f} m")
    print(f"    std(bias): {biases.std():.4f} m")
    print(f"    range: [{biases.min():.3f}, {biases.max():.3f}] m")
    print(f"\n  Residual (去除 per-seq bias 后):")
    print(f"    mean(residual_std): {residual_stds.mean():.4f} m")
    print(f"    mean(residual_MAE): {residual_maes.mean():.4f} m")
    print(f"\n  对比:")
    print(f"    Raw MAE: {raw_maes.mean():.4f} m")
    print(f"    去除 bias 后 MAE: {residual_maes.mean():.4f} m")
    print(f"    bias 贡献比例: {(1 - residual_maes.mean() / raw_maes.mean()) * 100:.1f}%")
    
    # 更精细：用 sliding window bias (local mean) 而非 global mean
    print("\n  Sliding window bias removal (更好的去偏估计):")
    windows = [10, 20, 30, 50, 100]
    for w in windows:
        local_maes = []
        for s in all_series:
            ftm_m = s["ftm_mm"] / 1000.0
            depth_m = s["depth_m"]
            valid = s["valid"]
            noise_full = ftm_m - depth_m
            
            # 用 sliding window 均值作为 local bias estimate
            n = len(noise_full)
            debiased = noise_full.copy()
            half_w = w // 2
            for i in range(n):
                if not valid[i]:
                    continue
                start = max(0, i - half_w)
                end = min(n, i + half_w + 1)
                local_valid = valid[start:end]
                if local_valid.sum() > 0:
                    local_bias = noise_full[start:end][local_valid].mean()
                    debiased[i] = noise_full[i] - local_bias
            
            local_maes.extend(np.abs(debiased[valid]).tolist())
        
        local_mae = np.mean(local_maes)
        reduction = (1 - local_mae / raw_maes.mean()) * 100
        print(f"    window={w:3d} ({w*0.1:.1f}s): residual MAE={local_mae:.4f}m, reduction={reduction:.1f}%")
    
    # =========================================================================
    # Part 2: 不同噪声水平下的理论 compat 区分力
    # =========================================================================
    print("\n" + "=" * 70)
    print("Part 2: 不同噪声水平的理论 compat 区分力")
    print("=" * 70)
    
    # 模拟不同去噪水平：加上不同大小的高斯噪声到 depth（作为 denoised FTM）
    # 然后计算 compat 分布
    sigma_compat = 1.5  # Gaussian compat 的 σ
    
    noise_levels = [0.0, 0.3, 0.5, 0.8, 1.0, 1.5, 1.83]  # 最后一个是当前 raw FTM std
    
    # 首先收集所有正确和错误配对的 depth 差
    correct_depths = []  # depth values for correct pairs
    wrong_depth_diffs = []  # |depth_i - depth_j| for wrong pairs (same seq, diff person)
    
    for s in all_series:
        correct_depths.extend(s["depth_m"][s["valid"]].tolist())
    
    for scene in SCENES:
        scene_series = [s for s in all_series if s["scene"] == scene]
        for i, s1 in enumerate(scene_series):
            for j, s2 in enumerate(scene_series):
                if i >= j or s1["seq"] != s2["seq"]:
                    continue
                valid = s1["valid"] & s2["valid"]
                if valid.sum() < 10:
                    continue
                diff = np.abs(s1["depth_m"][valid] - s2["depth_m"][valid])
                wrong_depth_diffs.extend(diff[:300].tolist())
    
    wrong_depth_diffs = np.array(wrong_depth_diffs)
    print(f"\n  Wrong pair depth difference stats:")
    print(f"    mean: {wrong_depth_diffs.mean():.3f} m")
    print(f"    median: {np.median(wrong_depth_diffs):.3f} m")
    print(f"    P25/P75: {np.percentile(wrong_depth_diffs, 25):.3f} / {np.percentile(wrong_depth_diffs, 75):.3f} m")
    print(f"    <1m: {(wrong_depth_diffs < 1.0).mean()*100:.1f}%  (hard cases)")
    print(f"    <2m: {(wrong_depth_diffs < 2.0).mean()*100:.1f}%  (medium cases)")
    
    print(f"\n  理论 compat 区分力 vs FTM 噪声水平 (σ_compat={sigma_compat}m):")
    print(f"  {'Noise σ':>8} | {'Correct compat':>14} | {'Wrong compat':>12} | {'Gap':>6} | {'Correct>0.5':>11} | {'Wrong<0.5':>9}")
    print(f"  {'-'*8} | {'-'*14} | {'-'*12} | {'-'*6} | {'-'*11} | {'-'*9}")
    
    np.random.seed(42)
    n_sim = 50000
    
    for noise_std in noise_levels:
        # Correct pair: denoised_FTM = depth + N(0, noise_std)
        depths_sample = np.random.choice(correct_depths, size=n_sim)
        ftm_denoised = depths_sample + np.random.randn(n_sim) * noise_std
        correct_compat = np.exp(-(depths_sample - ftm_denoised)**2 / (2 * sigma_compat**2))
        
        # Wrong pair: person_i_depth vs person_j_denoised_FTM
        # depth_diff drawn from wrong_depth_diffs distribution
        diffs_sample = np.random.choice(wrong_depth_diffs, size=n_sim)
        signs = np.random.choice([-1, 1], size=n_sim)
        ftm_wrong = depths_sample + signs * diffs_sample + np.random.randn(n_sim) * noise_std
        wrong_compat = np.exp(-(depths_sample - ftm_wrong)**2 / (2 * sigma_compat**2))
        
        gap = correct_compat.mean() - wrong_compat.mean()
        print(f"  {noise_std:>8.2f} | {correct_compat.mean():>14.4f} | {wrong_compat.mean():>12.4f} | {gap:>6.4f} | {(correct_compat>0.5).mean()*100:>10.1f}% | {(wrong_compat<0.5).mean()*100:>8.1f}%")
    
    # =========================================================================
    # Part 3: Oracle MOT 实验 — 用不同程度的 "去噪 FTM" 跑 tracker
    # =========================================================================
    print("\n" + "=" * 70)
    print("Part 3: Oracle MOT 实验设计")
    print("=" * 70)
    print("""
  为评估去噪对 MOT 的实际增益，我们设计 Oracle 实验：
  
  修改 FTM 信号: denoised_ftm = α * depth + (1-α) * raw_ftm
  
  - α=0: 原始 FTM (当前状态)
  - α=0.5: 50% 去噪 (误差减半)
  - α=0.8: 80% 去噪 (大部分噪声去除)
  - α=1: 完美去噪 (FTM=depth, Oracle 上限)
  
  对每个 α，跑 WiFiJointTracker 得到 IDF1，绘制 denoising quality → MOT gain 曲线。
  这给出了 Phase 6 的理论增益上限。
""")
    
    # 实际跑 Oracle 实验
    print("  正在运行 Oracle MOT 实验...")
    
    import subprocess
    
    # 首先需要修改 run_baseline.py 支持 oracle denoising
    # 或者我们直接修改数据来运行
    # 更简单的方法：直接修改 vifi_mot.py 的加载逻辑？
    # 最简单：给 run_baseline.py 加一个 --oracle-denoise-alpha 参数
    
    # 先跑 alpha=1 (完美去噪) 看上限
    # 方法：创建一个临时脚本，修改 FTM 数据后跑 tracker
    
    oracle_script = '''
import sys, pickle, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.vifi_mot import load_sequence, get_all_outdoor_sequences
from scripts.run_baseline import build_args

# 直接 hack: 在加载后修改 FTM
DATA_ROOT = Path("/Users/zstar/auto_search/vifi-mot/RAN4model_dfv4p4/seqs/outdoor")
ALPHA = float(sys.argv[1])  # denoising level
TEST_SCENE = sys.argv[2] if len(sys.argv) > 2 else "scene4"

print(f"Oracle α={ALPHA}, test={TEST_SCENE}")

# 加载并修改 FTM 数据 → 跑 WiFiJointTracker
from data.vifi_mot import get_all_outdoor_sequences, load_sequence
from trackers.wifi_joint import WiFiJointTracker
import motmetrics as mm

seqs = get_all_outdoor_sequences()
test_seqs = [s for s in seqs if s.scene == TEST_SCENE]

acc = mm.MOTAccumulator(auto_id=True)
total_frames = 0

for meta in test_seqs:
    frames = list(load_sequence(meta))
    total_frames += len(frames)
    
    tracker = WiFiJointTracker(
        max_age=30, min_hits=3, iou_threshold=0.3,
        delta_t=3, inertia=0.2,
        wifi_weight=0.10, depth_sigma=1.5,
        ema_alpha=0.85, bind_init_threshold=0.7,
        unbind_threshold=0.15,
        switch_penalty=0.30, keep_bonus=0.20,
        assign_threshold=0.40,
        no_ghost_pool=True,
    )
    
    for frame in frames:
        # Oracle denoise: FTM = α * depth + (1-α) * FTM
        # 对每个有 phone 的 detection，用其 depth 修正 FTM
        n_phones = frame.ftm.shape[0] if frame.ftm is not None else 0
        ftm_m = np.full(n_phones, np.nan)
        ftm_valid = np.zeros(n_phones, dtype=bool)
        
        if frame.ftm is not None:
            for p in range(n_phones):
                raw_ftm_mm = frame.ftm[p, 0]
                if abs(raw_ftm_mm) > 10:
                    raw_ftm_m = raw_ftm_mm / 1000.0
                    # 找到对应 phone p 的 detection depth
                    phone_depth = None
                    for det in frame.detections:
                        if det.subj_idx == p:
                            phone_depth = det.depth
                            break
                    
                    if phone_depth is not None and phone_depth > 0.1:
                        # Oracle denoise
                        denoised = ALPHA * phone_depth + (1 - ALPHA) * raw_ftm_m
                        ftm_m[p] = denoised
                        ftm_valid[p] = True
                    else:
                        ftm_m[p] = raw_ftm_m
                        ftm_valid[p] = True
        
        # 准备 detections
        dets = []
        for det in frame.detections:
            dets.append([det.x, det.y, det.x + det.w, det.y + det.h, 1.0, det.depth])
        
        if len(dets) == 0:
            dets_arr = np.empty((0, 6))
        else:
            dets_arr = np.array(dets)
        
        tracks = tracker.update(dets_arr, ftm_m, ftm_valid)
        
        # Accumulate for motmetrics
        gt_ids = []
        gt_boxes = []
        hyp_ids = []
        hyp_boxes = []
        
        for det in frame.detections:
            gt_ids.append(det.track_id)
            gt_boxes.append([det.x, det.y, det.w, det.h])
        
        for trk in tracks:
            x1, y1, x2, y2, tid = trk
            hyp_ids.append(int(tid))
            hyp_boxes.append([x1, y1, x2-x1, y2-y1])
        
        # IoU distance
        if len(gt_boxes) > 0 and len(hyp_boxes) > 0:
            gt_arr = np.array(gt_boxes)
            hyp_arr = np.array(hyp_boxes)
            dist = mm.distances.iou_matrix(gt_arr, hyp_arr, max_iou=0.5)
        elif len(gt_boxes) > 0:
            dist = np.full((len(gt_boxes), 0), np.nan)
        else:
            dist = np.full((0, len(hyp_boxes)), np.nan)
        
        acc.update(gt_ids, hyp_ids, dist)

# Compute metrics
mh = mm.metrics.create()
summary = mh.compute(acc, metrics=["mota", "idf1", "num_switches"], name="overall")
idf1 = summary["idf1"].values[0] * 100
mota = summary["mota"].values[0] * 100
idsw = summary["num_switches"].values[0]
print(f"  α={ALPHA:.2f} | {TEST_SCENE} | IDF1={idf1:.2f}% | MOTA={mota:.2f}% | IDsw={idsw}")
'''
    
    oracle_path = Path("/Users/zstar/test/vifi/mot_baseline/scripts/phase6_oracle_mot.py")
    with open(oracle_path, "w") as f:
        f.write(oracle_script)
    
    print(f"  Oracle script saved: {oracle_path}")
    print(f"  Running Oracle experiments on scene4 (hardest scene)...")
    
    alphas = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
    results = []
    
    for alpha in alphas:
        cmd = ["/Users/zstar/miniforge3/bin/python3", str(oracle_path), str(alpha), "scene4"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        output = result.stdout.strip()
        if result.returncode == 0:
            # Parse the last line
            for line in output.split("\n"):
                if f"α={alpha:.2f}" in line:
                    print(f"    {line.strip()}")
                    results.append({"alpha": alpha, "output": line.strip()})
        else:
            print(f"    α={alpha}: ERROR - {result.stderr[-200:]}")
            results.append({"alpha": alpha, "error": result.stderr[-200:]})
    
    # 也跑 scene1/scene3
    print(f"\n  Running on scene1 and scene3...")
    for scene in ["scene1", "scene3"]:
        for alpha in [0.0, 0.5, 1.0]:
            cmd = ["/Users/zstar/miniforge3/bin/python3", str(oracle_path), str(alpha), scene]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if f"α={alpha:.2f}" in line:
                        print(f"    {line.strip()}")


if __name__ == "__main__":
    main()
