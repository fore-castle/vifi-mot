"""Phase 6 数据探索：FTM 时间序列噪声结构分析

分析内容：
1. FTM 噪声的自相关函数 (ACF) — 判断噪声是否有时间结构
2. 噪声功率谱密度 (PSD) — 判断低频/高频噪声分布
3. 噪声分布特征 — 偏度、峰度、是否正态
4. 滑动窗口滤波 baseline — 中值/均值滤波能降低多少误差
5. Oracle 去噪上限 — 如果完美去噪，compat 精度能提升多少
"""

import pickle
import numpy as np
from pathlib import Path
import json

DATA_ROOT = Path("/Users/zstar/auto_search/vifi-mot/RAN4model_dfv4p4/seqs/outdoor")
SCENES = ["scene1", "scene2", "scene3", "scene4"]


def load_all_ftm_depth_pairs():
    """加载所有 outdoor 序列的 (FTM, depth) 时间序列对"""
    all_series = []  # List of dict: {scene, seq, phone_idx, ftm_mm, ftm_std_mm, depth_m, T}
    
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
                ftm = pickle.load(f)  # (T, N, 1, 2)
            with open(bbx_path, "rb") as f:
                bbx5 = pickle.load(f)  # (T, N, 1, 5)
            
            T, N = ftm.shape[0], ftm.shape[1]
            for s in range(N):
                ftm_range = ftm[:, s, 0, 0]  # mm
                ftm_std = ftm[:, s, 0, 1]    # mm
                depth = bbx5[:, s, 0, 2]      # meters
                
                # 过滤无效帧 (depth=0 或 ftm=0 表示该人不在视野)
                valid = (depth > 0.1) & (np.abs(ftm_range) > 10)  # depth>0.1m, ftm>10mm
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


def compute_noise_series(series):
    """计算噪声序列 = FTM(m) - depth(m)，只在 valid 帧"""
    ftm_m = series["ftm_mm"] / 1000.0
    depth_m = series["depth_m"]
    valid = series["valid"]
    noise = ftm_m[valid] - depth_m[valid]
    return noise


def compute_acf(noise, max_lag=50):
    """计算自相关函数"""
    n = len(noise)
    if n < max_lag + 10:
        max_lag = n // 2
    noise_centered = noise - noise.mean()
    var = np.var(noise_centered)
    if var < 1e-10:
        return np.zeros(max_lag + 1)
    acf = np.correlate(noise_centered, noise_centered, mode='full')
    acf = acf[n-1:n+max_lag]  # lag 0 to max_lag
    acf = acf / (var * n)
    return acf


def compute_psd(noise, fs=10.0):
    """计算功率谱密度 (Welch 方法的简化版本)"""
    n = len(noise)
    # 直接用 FFT
    noise_centered = noise - noise.mean()
    fft = np.fft.rfft(noise_centered)
    psd = np.abs(fft) ** 2 / n
    freqs = np.fft.rfftfreq(n, d=1.0/fs)
    return freqs, psd


def sliding_filter(signal, valid, window, mode="median"):
    """滑动窗口滤波（中值或均值）"""
    n = len(signal)
    filtered = signal.copy()
    half_w = window // 2
    for i in range(n):
        if not valid[i]:
            continue
        start = max(0, i - half_w)
        end = min(n, i + half_w + 1)
        local_valid = valid[start:end]
        local_vals = signal[start:end][local_valid]
        if len(local_vals) == 0:
            continue
        if mode == "median":
            filtered[i] = np.median(local_vals)
        else:
            filtered[i] = np.mean(local_vals)
    return filtered


def gaussian_compat(depth, ftm_m, sigma=1.5):
    """计算 depth-FTM 高斯兼容性"""
    return np.exp(-(depth - ftm_m)**2 / (2 * sigma**2))


def main():
    print("=" * 70)
    print("Phase 6 数据探索：FTM 噪声结构分析")
    print("=" * 70)
    
    # 1. 加载数据
    print("\n[1] 加载所有 outdoor FTM-depth 序列对...")
    all_series = load_all_ftm_depth_pairs()
    print(f"    总序列数: {len(all_series)}")
    scene_counts = ', '.join(f'{sc}: {sum(1 for x in all_series if x["scene"]==sc)}' for sc in SCENES)
    print(f"    按场景: {scene_counts}")
    
    total_valid_frames = sum(s["valid"].sum() for s in all_series)
    print(f"    总有效帧: {total_valid_frames:,}")
    
    # 2. 全局噪声统计
    print("\n[2] 全局噪声统计 (noise = FTM_m - depth_m)...")
    all_noise = np.concatenate([compute_noise_series(s) for s in all_series])
    print(f"    样本数: {len(all_noise):,}")
    print(f"    均值: {all_noise.mean():.4f} m")
    print(f"    中位数: {np.median(all_noise):.4f} m")
    print(f"    标准差: {all_noise.std():.4f} m")
    print(f"    MAE: {np.abs(all_noise).mean():.4f} m")
    
    from scipy.stats import skew, kurtosis
    print(f"    偏度 (scipy): {skew(all_noise):.4f}")
    print(f"    峰度 (excess): {kurtosis(all_noise):.4f}")
    
    # 分位数
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    vals = np.percentile(all_noise, percentiles)
    print(f"    分位数:")
    for p, v in zip(percentiles, vals):
        print(f"      P{p:02d}: {v:+.3f} m")
    
    # 3. 按场景分析噪声
    print("\n[3] 按场景噪声统计...")
    for scene in SCENES:
        scene_noise = np.concatenate([compute_noise_series(s) for s in all_series if s["scene"] == scene])
        print(f"    {scene}: mean={scene_noise.mean():+.3f}m, std={scene_noise.std():.3f}m, "
              f"MAE={np.abs(scene_noise).mean():.3f}m, N={len(scene_noise):,}")
    
    # 4. 自相关分析
    print("\n[4] 自相关分析 (ACF)...")
    print("    计算每个序列的 ACF，然后平均...")
    max_lag = 50  # 50 frames = 5 seconds
    acf_sum = np.zeros(max_lag + 1)
    acf_count = 0
    
    per_scene_acf = {s: np.zeros(max_lag + 1) for s in SCENES}
    per_scene_acf_count = {s: 0 for s in SCENES}
    
    for s in all_series:
        noise = compute_noise_series(s)
        if len(noise) < max_lag + 10:
            continue
        acf = compute_acf(noise, max_lag)
        if len(acf) == max_lag + 1:
            acf_sum += acf
            acf_count += 1
            per_scene_acf[s["scene"]] += acf
            per_scene_acf_count[s["scene"]] += 1
    
    avg_acf = acf_sum / max(acf_count, 1)
    print(f"    有效序列数: {acf_count}")
    print(f"    平均 ACF (lag 0-10):")
    for lag in range(11):
        print(f"      lag {lag:2d} ({lag*0.1:.1f}s): {avg_acf[lag]:.4f}")
    print(f"    平均 ACF (lag 15-50, 每5帧):")
    for lag in range(15, 51, 5):
        if lag < len(avg_acf):
            print(f"      lag {lag:2d} ({lag*0.1:.1f}s): {avg_acf[lag]:.4f}")
    
    # ACF 半衰期 (下降到 0.5 的 lag)
    half_life = None
    for lag in range(1, len(avg_acf)):
        if avg_acf[lag] < 0.5:
            half_life = lag
            break
    print(f"    ACF 半衰期: lag={half_life} ({half_life*0.1:.1f}s)" if half_life else "    ACF 半衰期: >5s")
    
    # 1/e 衰减时间
    e_time = None
    for lag in range(1, len(avg_acf)):
        if avg_acf[lag] < 1.0/np.e:
            e_time = lag
            break
    print(f"    ACF 1/e 衰减: lag={e_time} ({e_time*0.1:.1f}s)" if e_time else "    ACF 1/e 衰减: >5s")
    
    print("\n    按场景 ACF 半衰期:")
    for scene in SCENES:
        if per_scene_acf_count[scene] > 0:
            scene_acf = per_scene_acf[scene] / per_scene_acf_count[scene]
            hl = None
            for lag in range(1, len(scene_acf)):
                if scene_acf[lag] < 0.5:
                    hl = lag
                    break
            print(f"      {scene}: half_life={hl} ({hl*0.1:.1f}s)" if hl else f"      {scene}: half_life>5s")
    
    # 5. 功率谱密度
    print("\n[5] 功率谱密度分析...")
    # 对几个长序列做 PSD
    long_series = [s for s in all_series if s["valid"].sum() > 500]
    print(f"    长序列 (>500有效帧): {len(long_series)}")
    
    # 计算平均 PSD（分段平均）
    psd_bins = 50
    avg_psd = np.zeros(psd_bins)
    avg_freqs = np.linspace(0, 5.0, psd_bins)  # 0-5Hz (Nyquist at 5Hz for 10Hz sampling)
    psd_count = 0
    
    for s in long_series[:20]:  # 取前 20 个长序列
        noise = compute_noise_series(s)
        freqs, psd = compute_psd(noise, fs=10.0)
        # 将 PSD 插值到统一频率 bin
        if len(freqs) > 2:
            psd_interp = np.interp(avg_freqs, freqs, psd)
            avg_psd += psd_interp
            psd_count += 1
    
    if psd_count > 0:
        avg_psd /= psd_count
        # 分析低频 vs 高频能量
        low_mask = avg_freqs <= 1.0   # 0-1 Hz
        mid_mask = (avg_freqs > 1.0) & (avg_freqs <= 3.0)  # 1-3 Hz
        high_mask = avg_freqs > 3.0   # 3-5 Hz
        
        total_power = avg_psd.sum()
        low_power = avg_psd[low_mask].sum() / total_power * 100
        mid_power = avg_psd[mid_mask].sum() / total_power * 100
        high_power = avg_psd[high_mask].sum() / total_power * 100
        
        print(f"    频率能量分布:")
        print(f"      低频 (0-1 Hz): {low_power:.1f}%")
        print(f"      中频 (1-3 Hz): {mid_power:.1f}%")
        print(f"      高频 (3-5 Hz): {high_power:.1f}%")
        print(f"    结论: {'低频主导 → 有去噪潜力（慢变化偏移）' if low_power > 60 else '宽带噪声 → temporal model 挑战大'}")
    
    # 6. 噪声时间结构类型分析
    print("\n[6] 噪声时间结构分类...")
    # 对每个序列分类：平稳 / 有偏移 / 有突刺
    n_stationary = 0
    n_drifting = 0
    n_spiky = 0
    drift_amounts = []
    spike_counts = []
    
    for s in all_series:
        noise = compute_noise_series(s)
        if len(noise) < 100:
            continue
        
        # 检测漂移：前半 vs 后半的均值差
        half = len(noise) // 2
        drift = abs(noise[:half].mean() - noise[half:].mean())
        drift_amounts.append(drift)
        
        # 检测突刺：超过 3σ 的样本比例
        std = noise.std()
        if std > 0:
            spikes = np.abs(noise - noise.mean()) > 3 * std
            spike_rate = spikes.mean()
            spike_counts.append(spike_rate)
        else:
            spike_rate = 0
            spike_counts.append(0)
        
        if drift > 1.0:  # >1m 的漂移
            n_drifting += 1
        elif spike_rate > 0.05:  # >5% 突刺
            n_spiky += 1
        else:
            n_stationary += 1
    
    total = n_stationary + n_drifting + n_spiky
    print(f"    平稳噪声: {n_stationary}/{total} ({n_stationary/total*100:.1f}%)")
    print(f"    有漂移 (>1m shift): {n_drifting}/{total} ({n_drifting/total*100:.1f}%)")
    print(f"    有突刺 (>5% outliers): {n_spiky}/{total} ({n_spiky/total*100:.1f}%)")
    print(f"    平均漂移量: {np.mean(drift_amounts):.3f}m")
    print(f"    平均突刺率: {np.mean(spike_counts)*100:.2f}%")
    
    # 7. 滑动窗口滤波 baseline
    print("\n[7] 滑动窗口滤波 baseline...")
    windows = [3, 5, 7, 11, 15, 21]
    
    for mode in ["median", "mean"]:
        print(f"\n    --- {mode.upper()} 滤波 ---")
        print(f"    {'Window':>8} | {'MAE_raw':>8} | {'MAE_filt':>8} | {'Δ MAE':>8} | {'Δ%':>6}")
        print(f"    {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*6}")
        
        for w in windows:
            raw_errors = []
            filt_errors = []
            
            for s in all_series:
                ftm_m = s["ftm_mm"] / 1000.0
                depth_m = s["depth_m"]
                valid = s["valid"]
                
                # Raw error
                raw_err = np.abs(ftm_m[valid] - depth_m[valid])
                raw_errors.extend(raw_err.tolist())
                
                # Filtered FTM
                ftm_filtered = sliding_filter(ftm_m, valid, w, mode=mode)
                filt_err = np.abs(ftm_filtered[valid] - depth_m[valid])
                filt_errors.extend(filt_err.tolist())
            
            raw_mae = np.mean(raw_errors)
            filt_mae = np.mean(filt_errors)
            delta = filt_mae - raw_mae
            pct = delta / raw_mae * 100
            print(f"    {w:>8} | {raw_mae:>8.4f} | {filt_mae:>8.4f} | {delta:>+8.4f} | {pct:>+5.1f}%")
    
    # 8. Gaussian compat 精度分析
    print("\n[8] Gaussian compat 精度分析...")
    print("    比较：raw FTM vs 中值滤波(w=7) vs Oracle (perfect=depth)")
    
    # 计算不同情况下的 compat 分布
    compat_raw_correct = []     # 正确配对 (same phone-person)
    compat_raw_wrong = []       # 错误配对 (different phone-person)
    compat_filt_correct = []
    compat_filt_wrong = []
    compat_oracle_correct = []  # Oracle: FTM=depth (完美去噪)
    
    sigma = 1.5
    for s in all_series:
        ftm_m = s["ftm_mm"] / 1000.0
        depth_m = s["depth_m"]
        valid = s["valid"]
        
        # Correct pair: same phone-person
        raw_compat = gaussian_compat(depth_m[valid], ftm_m[valid], sigma)
        compat_raw_correct.extend(raw_compat.tolist())
        
        # Filtered
        ftm_filtered = sliding_filter(ftm_m, valid, 7, "median")
        filt_compat = gaussian_compat(depth_m[valid], ftm_filtered[valid], sigma)
        compat_filt_correct.extend(filt_compat.tolist())
        
        # Oracle (FTM = depth → compat = 1.0)
        compat_oracle_correct.extend(np.ones(valid.sum()).tolist())
    
    # Wrong pairs: 同场景不同 phone 的 cross-pair
    for scene in SCENES:
        scene_series = [s for s in all_series if s["scene"] == scene]
        for i, s1 in enumerate(scene_series):
            for j, s2 in enumerate(scene_series):
                if i == j or s1["seq"] != s2["seq"]:
                    continue
                # person i's depth vs person j's FTM
                valid = s1["valid"] & s2["valid"]
                if valid.sum() < 10:
                    continue
                depth_i = s1["depth_m"][valid]
                ftm_j = s2["ftm_mm"][valid] / 1000.0
                raw_compat = gaussian_compat(depth_i, ftm_j, sigma)
                compat_raw_wrong.extend(raw_compat[:200].tolist())  # 限制数量
                
                ftm_j_filt = sliding_filter(s2["ftm_mm"] / 1000.0, s2["valid"], 7, "median")
                filt_compat = gaussian_compat(depth_i, ftm_j_filt[valid], sigma)
                compat_filt_wrong.extend(filt_compat[:200].tolist())
    
    compat_raw_correct = np.array(compat_raw_correct)
    compat_filt_correct = np.array(compat_filt_correct)
    compat_raw_wrong = np.array(compat_raw_wrong) if compat_raw_wrong else np.array([0.0])
    compat_filt_wrong = np.array(compat_filt_wrong) if compat_filt_wrong else np.array([0.0])
    
    print(f"\n    正确配对 compat (σ={sigma}m):")
    print(f"      Raw FTM:   mean={compat_raw_correct.mean():.4f}, >0.5: {(compat_raw_correct>0.5).mean()*100:.1f}%, >0.8: {(compat_raw_correct>0.8).mean()*100:.1f}%")
    print(f"      Median(7): mean={compat_filt_correct.mean():.4f}, >0.5: {(compat_filt_correct>0.5).mean()*100:.1f}%, >0.8: {(compat_filt_correct>0.8).mean()*100:.1f}%")
    print(f"      Oracle:    mean=1.0000, >0.5: 100.0%, >0.8: 100.0%")
    
    print(f"\n    错误配对 compat (σ={sigma}m):")
    print(f"      Raw FTM:   mean={compat_raw_wrong.mean():.4f}, >0.5: {(compat_raw_wrong>0.5).mean()*100:.1f}%")
    print(f"      Median(7): mean={compat_filt_wrong.mean():.4f}, >0.5: {(compat_filt_wrong>0.5).mean()*100:.1f}%")
    
    # 区分力: correct_mean - wrong_mean
    print(f"\n    区分力 (correct_mean - wrong_mean):")
    print(f"      Raw:       {compat_raw_correct.mean() - compat_raw_wrong.mean():.4f}")
    print(f"      Median(7): {compat_filt_correct.mean() - compat_filt_wrong.mean():.4f}")
    print(f"      Oracle:    {1.0 - compat_raw_wrong.mean():.4f}")  # wrong pair 不受去噪影响
    
    # 9. 用更小的 sigma 评估
    print("\n[9] 不同 σ 下的去噪价值...")
    sigmas = [0.5, 0.8, 1.0, 1.5, 2.0]
    print(f"    {'σ':>5} | {'Raw correct':>12} | {'Filt correct':>12} | {'Raw wrong':>10} | {'Filt wrong':>10} | {'Gap Raw':>8} | {'Gap Filt':>8}")
    print(f"    {'-'*5} | {'-'*12} | {'-'*12} | {'-'*10} | {'-'*10} | {'-'*8} | {'-'*8}")
    
    for sigma in sigmas:
        # Recompute for correct pairs
        rc = []
        fc = []
        for s in all_series:
            ftm_m = s["ftm_mm"] / 1000.0
            depth_m = s["depth_m"]
            valid = s["valid"]
            rc.extend(gaussian_compat(depth_m[valid], ftm_m[valid], sigma).tolist())
            ftm_f = sliding_filter(ftm_m, valid, 7, "median")
            fc.extend(gaussian_compat(depth_m[valid], ftm_f[valid], sigma).tolist())
        
        rc = np.array(rc)
        fc = np.array(fc)
        
        # Wrong pairs (reuse cross-pairs from same sequences)
        rw_vals = []
        fw_vals = []
        for scene in SCENES:
            scene_series = [s for s in all_series if s["scene"] == scene]
            for i, s1 in enumerate(scene_series):
                for j, s2 in enumerate(scene_series):
                    if i == j or s1["seq"] != s2["seq"]:
                        continue
                    valid = s1["valid"] & s2["valid"]
                    if valid.sum() < 10:
                        continue
                    depth_i = s1["depth_m"][valid]
                    ftm_j = s2["ftm_mm"][valid] / 1000.0
                    rw_vals.extend(gaussian_compat(depth_i, ftm_j, sigma)[:200].tolist())
                    ftm_j_f = sliding_filter(s2["ftm_mm"]/1000.0, s2["valid"], 7, "median")
                    fw_vals.extend(gaussian_compat(depth_i, ftm_j_f[valid], sigma)[:200].tolist())
        
        rw = np.array(rw_vals) if rw_vals else np.array([0.0])
        fw = np.array(fw_vals) if fw_vals else np.array([0.0])
        
        gap_raw = rc.mean() - rw.mean()
        gap_filt = fc.mean() - fw.mean()
        print(f"    {sigma:>5.1f} | {rc.mean():>12.4f} | {fc.mean():>12.4f} | {rw.mean():>10.4f} | {fw.mean():>10.4f} | {gap_raw:>8.4f} | {gap_filt:>8.4f}")
    
    # 10. 结论
    print("\n" + "=" * 70)
    print("结论与 Phase 6 可行性判断")
    print("=" * 70)
    
    acf_lag1 = avg_acf[1] if len(avg_acf) > 1 else 0
    print(f"\n  1. ACF lag-1 = {acf_lag1:.4f}")
    if acf_lag1 > 0.7:
        print(f"     → 强时间相关性！temporal model 有很大去噪空间")
    elif acf_lag1 > 0.4:
        print(f"     → 中等时间相关性，temporal model 可以利用")
    else:
        print(f"     → 弱时间相关性，temporal model 获益有限")
    
    print(f"\n  2. 噪声类型分布: {n_stationary}平稳 / {n_drifting}漂移 / {n_spiky}突刺")
    if n_drifting > total * 0.3:
        print(f"     → 大量漂移序列，temporal model 可以学习 slow drift correction")
    
    if psd_count > 0 and low_power > 50:
        print(f"\n  3. 低频能量 {low_power:.1f}% → 噪声有慢变化结构，滤波/去噪有潜力")
    
    print(f"\n  4. 中值滤波(w=7) 已能提升 compat：")
    print(f"     正确配对 compat: {compat_raw_correct.mean():.4f} → {compat_filt_correct.mean():.4f}")
    print(f"     如果去噪模型能比中值滤波更好（学习非对称噪声模式），")
    print(f"     有望进一步提升 compat 区分力。")
    
    print(f"\n  最终判断: ", end="")
    if acf_lag1 > 0.5 or n_drifting > total * 0.2:
        print("✅ GO — FTM 噪声有显著时间结构，temporal denoising 有潜力")
    else:
        print("⚠️ MARGINAL — 需要非 temporal 的去噪策略（如 context-aware）")
    
    # 保存详细结果
    results = {
        "noise_stats": {
            "mean": float(all_noise.mean()),
            "median": float(np.median(all_noise)),
            "std": float(all_noise.std()),
            "mae": float(np.abs(all_noise).mean()),
            "skewness": float(skew(all_noise)),
            "kurtosis": float(kurtosis(all_noise)),
        },
        "acf": {
            "lag_1": float(avg_acf[1]),
            "lag_5": float(avg_acf[5]),
            "lag_10": float(avg_acf[10]),
            "half_life_frames": half_life,
        },
        "psd_energy": {
            "low_0_1Hz_pct": float(low_power) if psd_count > 0 else None,
            "mid_1_3Hz_pct": float(mid_power) if psd_count > 0 else None,
            "high_3_5Hz_pct": float(high_power) if psd_count > 0 else None,
        },
        "noise_types": {
            "stationary": n_stationary,
            "drifting": n_drifting,
            "spiky": n_spiky,
        },
        "compat_improvement": {
            "raw_correct_mean": float(compat_raw_correct.mean()),
            "filt_correct_mean": float(compat_filt_correct.mean()),
            "raw_wrong_mean": float(compat_raw_wrong.mean()),
            "filt_wrong_mean": float(compat_filt_wrong.mean()),
        },
    }
    
    out_path = Path("/Users/zstar/test/vifi/mot_baseline/exps/phase6_noise_analysis.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  结果已保存: {out_path}")


if __name__ == "__main__":
    main()
