"""Analyze IMU features vs FTM noise for denoising input extension."""
import sys, os, pickle
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_ROOT = Path("/Users/zstar/auto_search/vifi-mot/RAN4model_dfv4p4/seqs/outdoor")
SCENES = ["scene1", "scene2", "scene3", "scene4"]

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# Collect per-frame data
all_noise_abs = []
all_ftm_m = []
all_depth_m = []
# IMU19 per-axis: accel(3), gyro(3), mag(3), lin_accel(3), gravity(3), quat(4)
all_imu = []

for scene in SCENES:
    scene_dir = DATA_ROOT / scene
    if not scene_dir.exists():
        continue
    for seq_dir in sorted(scene_dir.iterdir()):
        if not seq_dir.is_dir():
            continue
        sync_dir = seq_dir / "sync_ts16_dfv4p4"
        ftm_path = sync_dir / "FTM_li_sync_dfv4p4.pkl"
        bbx_path = sync_dir / "BBX5_sync_dfv4p4.pkl"
        imu_path = sync_dir / "IMU19_sync_dfv4p4.pkl"
        
        if not all(p.exists() for p in [ftm_path, bbx_path, imu_path]):
            continue
        
        ftm = load_pkl(str(ftm_path))     # (T, N, 1, 2)
        bbx5 = load_pkl(str(bbx_path))    # (T, N, 1, 5)
        imu = load_pkl(str(imu_path))     # (T, N, 1, 19)
        
        T, N = ftm.shape[0], min(ftm.shape[1], imu.shape[1])
        
        for p in range(N):
            for t in range(T):
                ftm_mm = ftm[t, p, 0, 0]
                if np.isnan(ftm_mm):
                    continue
                depth = bbx5[t, p, 0, 2]
                if depth < 0.1:
                    continue
                
                ftm_m_val = ftm_mm / 1000.0
                noise = ftm_m_val - depth
                
                all_noise_abs.append(abs(noise))
                all_ftm_m.append(ftm_m_val)
                all_depth_m.append(depth)
                all_imu.append(imu[t, p, 0, :12])  # first 12 dims

all_noise_abs = np.array(all_noise_abs)
all_imu = np.array(all_imu)

print(f"Total valid frames: {len(all_noise_abs):,}")
print(f"FTM |noise|: mean={all_noise_abs.mean():.3f}m, median={np.median(all_noise_abs):.3f}m")

# Derived IMU features
accel_raw = all_imu[:, 0:3]       # includes gravity
gyro = all_imu[:, 3:6]
mag = all_imu[:, 6:9]
lin_accel = all_imu[:, 9:12]

accel_raw_mag = np.linalg.norm(accel_raw, axis=1)
gyro_mag = np.linalg.norm(gyro, axis=1)
mag_mag = np.linalg.norm(mag, axis=1)
lin_accel_mag = np.linalg.norm(lin_accel, axis=1)

# Also compute gravity magnitude as a proxy for device tilt
gravity = accel_raw - lin_accel  # approx gravity vector
gravity_mag = np.linalg.norm(gravity, axis=1)

print("\n" + "=" * 70)
print("1. Feature Statistics")
print("=" * 70)
features = {
    "LinAccel Mag": lin_accel_mag,
    "Gyro Mag": gyro_mag,
    "Mag Mag": mag_mag,
    "Accel Raw Mag": accel_raw_mag,
    "Gravity Mag": gravity_mag,
}
for name, arr in features.items():
    print(f"  {name:>18s}: mean={arr.mean():.3f}, std={arr.std():.3f}, "
          f"range=[{arr.min():.2f}, {arr.max():.2f}]")

print("\n" + "=" * 70)
print("2. Correlation with |FTM noise|")
print("=" * 70)
# Magnitude features
for name, arr in features.items():
    r_p, p_p = pearsonr(arr, all_noise_abs)
    r_s, p_s = spearmanr(arr, all_noise_abs)
    sig = "***" if p_p < 0.001 else "**" if p_p < 0.01 else "*" if p_p < 0.05 else ""
    print(f"  {name:>18s}: Pearson r={r_p:+.4f} {sig:>3s}, Spearman ρ={r_s:+.4f}")

# Per-axis features
axis_names = ["Accel_X", "Accel_Y", "Accel_Z",
              "Gyro_X", "Gyro_Y", "Gyro_Z",
              "Mag_X", "Mag_Y", "Mag_Z",
              "LinAccel_X", "LinAccel_Y", "LinAccel_Z"]
print("\n  Per-axis (absolute value correlation):")
for i, name in enumerate(axis_names):
    col = np.abs(all_imu[:, i])  # use absolute value for signed features
    r_p, _ = pearsonr(col, all_noise_abs)
    r_s, _ = spearmanr(col, all_noise_abs)
    print(f"    |{name:>12s}|: Pearson r={r_p:+.4f}, Spearman ρ={r_s:+.4f}")

print("\n" + "=" * 70)
print("3. FTM Noise by Motion Regime (LinAccel terciles)")
print("=" * 70)
p33, p67 = np.percentile(lin_accel_mag, [33, 67])
for label, mask in [
    ("Low motion (accel<p33)", lin_accel_mag < p33),
    ("Med motion", (lin_accel_mag >= p33) & (lin_accel_mag < p67)),
    ("High motion (accel>p67)", lin_accel_mag >= p67),
]:
    n_sub = all_noise_abs[mask]
    print(f"  {label:>35s}: N={mask.sum():>6d}, MAE={n_sub.mean():.3f}m, "
          f"median={np.median(n_sub):.3f}m")

print("\n" + "=" * 70)
print("4. FTM Noise by Gyro Regime (terciles)")
print("=" * 70)
g33, g67 = np.percentile(gyro_mag, [33, 67])
for label, mask in [
    ("Low rotation (gyro<g33)", gyro_mag < g33),
    ("Med rotation", (gyro_mag >= g33) & (gyro_mag < g67)),
    ("High rotation (gyro>g67)", gyro_mag >= g67),
]:
    n_sub = all_noise_abs[mask]
    print(f"  {label:>35s}: N={mask.sum():>6d}, MAE={n_sub.mean():.3f}m, "
          f"median={np.median(n_sub):.3f}m")

print("\n" + "=" * 70)
print("5. Per-Scene Breakdown")
print("=" * 70)
idx = 0
for scene in SCENES:
    scene_dir = DATA_ROOT / scene
    if not scene_dir.exists():
        continue
    n_scene = 0
    for seq_dir in sorted(scene_dir.iterdir()):
        if not seq_dir.is_dir():
            continue
        sync_dir = seq_dir / "sync_ts16_dfv4p4"
        ftm_path = sync_dir / "FTM_li_sync_dfv4p4.pkl"
        if ftm_path.exists():
            ftm = load_pkl(str(ftm_path))
            bbx5 = load_pkl(str(sync_dir / "BBX5_sync_dfv4p4.pkl"))
            for p in range(ftm.shape[1]):
                for t in range(ftm.shape[0]):
                    if not np.isnan(ftm[t, p, 0, 0]) and bbx5[t, p, 0, 2] > 0.1:
                        n_scene += 1
    
    scene_noises = all_noise_abs[idx:idx+n_scene]
    scene_lin_accel = lin_accel_mag[idx:idx+n_scene]
    scene_gyro = gyro_mag[idx:idx+n_scene]
    idx += n_scene
    
    r_la, _ = pearsonr(scene_lin_accel, scene_noises)
    r_gy, _ = pearsonr(scene_gyro, scene_noises)
    print(f"  {scene}: N={n_scene:>6d}, MAE={scene_noises.mean():.3f}m, "
          f"r(lin_accel)={r_la:+.4f}, r(gyro)={r_gy:+.4f}")

print("\n" + "=" * 70)
print("6. Feature Recommendation Summary")
print("=" * 70)
print("  Based on correlation with |FTM noise|:")
print("  - LinAccel (3 axes + mag): motion context for multipath changes")
print("  - Gyro (3 axes + mag): orientation changes affect signal")
print("  - Mag (3 axes + mag): device orientation in environment")
print("  Total candidate features: 2 (FTM) + 3 (LinAccel) + 3 (Gyro) = 8")
print("  or full: 2 (FTM) + 9 (IMUagm9) = 11")
print("=" * 70)
