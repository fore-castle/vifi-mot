"""Phase 6 Feature Extension Experiment

Tests whether adding IMU and/or RSSI features improves FTM denoising.

Feature configs:
  A: FTM + FTM_std (2 features, baseline)
  B: FTM + FTM_std + IMU_accel(3) + IMU_gyro(3) (8 features)
  C: FTM + FTM_std + RSSI (3 features)
  D: FTM + FTM_std + IMU_accel(3) + IMU_gyro(3) + RSSI (9 features)

Model: Causal CNN (same as baseline), hidden=32→48 for larger configs.
Protocol: LOSO 4-fold, same hyperparams as baseline.
"""
import sys, os, pickle, time, json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ===========================================================================
# Data
# ===========================================================================
DATA_ROOT = Path("/Users/zstar/auto_search/vifi-mot/RAN4model_dfv4p4/seqs/outdoor")
SCENES = ["scene1", "scene2", "scene3", "scene4"]

def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f)

@dataclass
class PhoneSeriesExt:
    scene: str
    seq_name: str
    phone_idx: int
    ftm_mm: np.ndarray       # (T,)
    ftm_std_mm: np.ndarray   # (T,)
    depth_m: np.ndarray      # (T,)
    valid: np.ndarray        # (T,) bool
    T: int
    # Optional extra features
    accel: np.ndarray = None   # (T, 3)
    gyro: np.ndarray = None    # (T, 3)
    rssi: np.ndarray = None    # (T,)


def load_all_phone_series_ext(feature_set: str = "ftm_only") -> List[PhoneSeriesExt]:
    """Load with optional IMU/RSSI features."""
    load_imu = feature_set in ("imu", "full")
    load_rssi = feature_set in ("rssi", "full")
    
    series_list = []
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
            if not ftm_path.exists() or not bbx_path.exists():
                continue
            
            ftm = load_pkl(str(ftm_path))
            bbx5 = load_pkl(str(bbx_path))
            
            imu = None
            rssi = None
            if load_imu:
                imu_path = sync_dir / "IMUagm9_sync_dfv4p4.pkl"
                if imu_path.exists():
                    imu = load_pkl(str(imu_path))
            if load_rssi:
                rssi_path = sync_dir / "RSSI_li_sync_dfv4p4.pkl"
                if rssi_path.exists():
                    rssi = load_pkl(str(rssi_path))
            
            T, N = ftm.shape[0], ftm.shape[1]
            for s in range(N):
                ftm_range = ftm[:, s, 0, 0].astype(np.float64)
                ftm_std = ftm[:, s, 0, 1].astype(np.float64)
                depth = bbx5[:, s, 0, 2].astype(np.float64)
                valid = (depth > 0.1) & (np.abs(ftm_range) > 10) & (~np.isnan(ftm_range))
                if valid.sum() < 50:
                    continue
                
                accel = gyro = rssi_arr = None
                if imu is not None and s < imu.shape[1]:
                    imu_s = imu[:, s, 0, :]  # (T, 9)
                    accel = imu_s[:, 0:3].astype(np.float64)
                    gyro = imu_s[:, 3:6].astype(np.float64)
                if rssi is not None and s < rssi.shape[1]:
                    rssi_arr = rssi[:, s, 0, 0].astype(np.float64)
                    # Replace NaN with -100 (weak signal proxy)
                    rssi_arr = np.where(np.isnan(rssi_arr), -100.0, rssi_arr)
                
                series_list.append(PhoneSeriesExt(
                    scene=scene, seq_name=seq_dir.name, phone_idx=s,
                    ftm_mm=ftm_range, ftm_std_mm=ftm_std, depth_m=depth,
                    valid=valid, T=T,
                    accel=accel, gyro=gyro, rssi=rssi_arr,
                ))
    return series_list


# ===========================================================================
# Dataset
# ===========================================================================
class FTMDenoiseDatasetExt(Dataset):
    def __init__(self, series_list: List[PhoneSeriesExt],
                 window_size: int = 15, causal: bool = True, stride: int = 1,
                 feature_set: str = "ftm_only"):
        self.window_size = window_size
        self.causal = causal
        self.samples = []
        
        for ps in series_list:
            ftm_m = ps.ftm_mm / 1000.0
            ftm_std_m = ps.ftm_std_mm / 1000.0
            depth_m = ps.depth_m
            valid = ps.valid
            
            # Build feature arrays (T, n_feat)
            feat_cols = [ftm_m.reshape(-1, 1), ftm_std_m.reshape(-1, 1)]
            if feature_set in ("imu", "full") and ps.accel is not None:
                feat_cols.append(ps.accel)  # (T, 3)
                feat_cols.append(ps.gyro)   # (T, 3)
            if feature_set in ("rssi", "full") and ps.rssi is not None:
                feat_cols.append(ps.rssi.reshape(-1, 1))  # (T, 1)
            
            features_all = np.concatenate(feat_cols, axis=1)  # (T, n_feat)
            
            # Find contiguous valid segments
            segments = self._find_contiguous_segments(valid, min_len=window_size)
            
            for start, end in segments:
                for i in range(start, end - window_size + 1, stride):
                    window = features_all[i:i + window_size]
                    target_idx = i + window_size - 1 if causal else i + window_size // 2
                    target = depth_m[target_idx]
                    self.samples.append((window.astype(np.float32), np.float32(target)))
    
    @staticmethod
    def _find_contiguous_segments(valid, min_len):
        segments = []
        start = None
        for i, v in enumerate(valid):
            if v and start is None:
                start = i
            elif not v and start is not None:
                if i - start >= min_len:
                    segments.append((start, i))
                start = None
        if start is not None and len(valid) - start >= min_len:
            segments.append((start, len(valid)))
        return segments
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        features, target = self.samples[idx]
        return torch.from_numpy(features), torch.tensor(target)


# ===========================================================================
# Model (same CausalDilatedCNN as baseline)
# ===========================================================================
class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=0)
    
    def forward(self, x):
        # x: (B, C, T)
        x = nn.functional.pad(x, (self.padding, 0))
        return self.conv(x)


class CausalDilatedCNN(nn.Module):
    def __init__(self, in_features=2, hidden_dim=32, n_layers=4):
        super().__init__()
        self.input_proj = nn.Conv1d(in_features, hidden_dim, 1)
        self.layers = nn.ModuleList()
        dilations = [2**i for i in range(n_layers)]
        for d in dilations:
            self.layers.append(nn.ModuleDict({
                "conv": CausalConv1d(hidden_dim, hidden_dim, 3, d),
                "norm": nn.LayerNorm(hidden_dim),
                "act": nn.GELU(),
            }))
        self.head = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        # x: (B, T, F) -> (B, F, T)
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        for layer in self.layers:
            residual = x
            x = layer["conv"](x)
            x = layer["norm"](x.transpose(1, 2)).transpose(1, 2)
            x = layer["act"](x)
            x = x + residual
        # Take last timestep (causal)
        x = x[:, :, -1]
        return self.head(x)


# ===========================================================================
# Training
# ===========================================================================
def train_one_fold(train_series, test_series, feature_set, window_size=15,
                   hidden_dim=32, epochs=20, train_stride=3, lr=1e-3):
    """Train one LOSO fold, return test MAE and raw MAE."""
    train_ds = FTMDenoiseDatasetExt(train_series, window_size, True, train_stride, feature_set)
    test_ds = FTMDenoiseDatasetExt(test_series, window_size, True, 1, feature_set)
    
    n_feat = train_ds.samples[0][0].shape[1]
    
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=0)
    
    model = CausalDilatedCNN(in_features=n_feat, hidden_dim=hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.HuberLoss(delta=1.0)
    
    best_mae = float("inf")
    t0 = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for feats, targets in train_loader:
            optimizer.zero_grad()
            preds = model(feats).squeeze(-1)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * feats.size(0)
        scheduler.step()
        
        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            all_preds, all_targets = [], []
            with torch.no_grad():
                for feats, targets in test_loader:
                    preds = model(feats).squeeze(-1)
                    all_preds.append(preds.numpy())
                    all_targets.append(targets.numpy())
            all_preds = np.concatenate(all_preds)
            all_targets = np.concatenate(all_targets)
            mae = np.abs(all_preds - all_targets).mean()
            if mae < best_mae:
                best_mae = mae
    
    # Raw MAE (no denoising)
    raw_maes = []
    for ps in test_series:
        ftm_m = ps.ftm_mm[ps.valid] / 1000.0
        depth_m = ps.depth_m[ps.valid]
        raw_maes.append(np.abs(ftm_m - depth_m).mean())
    raw_mae = np.mean(raw_maes) if raw_maes else float("nan")
    
    elapsed = time.time() - t0
    return {
        "best_mae": best_mae,
        "raw_mae": raw_mae,
        "reduction_pct": (1 - best_mae / raw_mae) * 100 if raw_mae > 0 else 0,
        "equiv_alpha": 1 - best_mae / raw_mae if raw_mae > 0 else 0,
        "n_features": n_feat,
        "elapsed_sec": elapsed,
    }


# ===========================================================================
# Main
# ===========================================================================
def main():
    feature_configs = {
        "ftm_only": {"desc": "FTM+std (baseline)", "hidden": 32},
        "imu": {"desc": "FTM+std+IMU(accel+gyro)", "hidden": 32},
        "rssi": {"desc": "FTM+std+RSSI", "hidden": 32},
        "full": {"desc": "FTM+std+IMU+RSSI", "hidden": 48},
    }
    
    print("=" * 75)
    print("Phase 6 Feature Extension: Multi-feature FTM Denoising")
    print("=" * 75)
    
    all_results = {}
    
    for feat_name, config in feature_configs.items():
        print(f"\n{'─'*75}")
        print(f"Feature set: {feat_name} — {config['desc']}")
        print(f"{'─'*75}")
        
        all_series = load_all_phone_series_ext(feat_name)
        print(f"  Loaded {len(all_series)} phone series")
        
        fold_results = []
        for test_scene in SCENES:
            train = [s for s in all_series if s.scene != test_scene]
            test = [s for s in all_series if s.scene == test_scene]
            
            res = train_one_fold(train, test, feat_name,
                                hidden_dim=config["hidden"])
            res["test_scene"] = test_scene
            fold_results.append(res)
            
            print(f"  {test_scene}: raw_mae={res['raw_mae']:.3f}m → "
                  f"denoise_mae={res['best_mae']:.3f}m "
                  f"(red={res['reduction_pct']:.1f}%, eq_α={res['equiv_alpha']:.3f}) "
                  f"[{res['elapsed_sec']:.0f}s, {res['n_features']}feat]")
        
        avg_red = np.mean([r["reduction_pct"] for r in fold_results])
        avg_eq = np.mean([r["equiv_alpha"] for r in fold_results])
        avg_mae = np.mean([r["best_mae"] for r in fold_results])
        
        all_results[feat_name] = {
            "desc": config["desc"],
            "avg_reduction_pct": avg_red,
            "avg_equiv_alpha": avg_eq,
            "avg_mae": avg_mae,
            "per_scene": fold_results,
        }
        
        print(f"  AVG: reduction={avg_red:.1f}%, equiv_α={avg_eq:.3f}, mae={avg_mae:.3f}m")
    
    # Summary
    print(f"\n{'='*75}")
    print("SUMMARY: Feature Extension Comparison")
    print(f"{'='*75}")
    print(f"{'Config':>12} | {'Features':>30} | {'equiv_α':>8} | {'reduction':>10} | {'MAE':>7}")
    print(f"{'-'*12} | {'-'*30} | {'-'*8} | {'-'*10} | {'-'*7}")
    
    for feat_name, res in all_results.items():
        print(f"{feat_name:>12} | {res['desc']:>30} | {res['avg_equiv_alpha']:>7.3f} | "
              f"{res['avg_reduction_pct']:>9.1f}% | {res['avg_mae']:>6.3f}m")
    
    # Baseline comparison
    baseline_eq = all_results["ftm_only"]["avg_equiv_alpha"]
    print(f"\nΔ equiv_α vs baseline:")
    for feat_name, res in all_results.items():
        if feat_name != "ftm_only":
            delta = res["avg_equiv_alpha"] - baseline_eq
            print(f"  {feat_name:>12}: {delta:+.3f} ({'+' if delta > 0 else ''}{delta/baseline_eq*100:.1f}%)")
    
    # Save
    out_path = Path(__file__).resolve().parent.parent / "exps" / "phase6_feature_extension.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
