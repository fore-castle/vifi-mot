"""Phase 6A: FTM Denoising — Data preparation + Model + Training

Directory structure:
    denoising/
    ├── dataset.py       — LOSO dataset + DataLoader
    ├── models.py        — CausalDilatedCNN, BiLSTM denoiser
    ├── train.py         — LOSO 4-fold training loop
    ├── evaluate.py      — MAE / equiv-alpha / per-scene metrics
    └── inference.py     — Load model + denoise FTM for MOT pipeline
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ===========================================================================
# Constants
# ===========================================================================
DATA_ROOT = Path("/Users/zstar/auto_search/vifi-mot/RAN4model_dfv4p4/seqs/outdoor")
SCENES = ["scene1", "scene2", "scene3", "scene4"]
FS = 10.0  # sampling rate (Hz)


# ===========================================================================
# Raw data loading
# ===========================================================================
@dataclass
class PhoneSeries:
    """One phone's time series across a full sequence."""
    scene: str
    seq_name: str
    phone_idx: int
    ftm_mm: np.ndarray       # (T,) FTM range in mm
    ftm_std_mm: np.ndarray   # (T,) FTM reported std in mm
    depth_m: np.ndarray      # (T,) camera depth in meters (GT)
    valid: np.ndarray        # (T,) bool — both FTM and depth available
    T: int


def load_all_phone_series() -> List[PhoneSeries]:
    """Load all outdoor phone time series."""
    series_list: List[PhoneSeries] = []
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
            with open(ftm_path, "rb") as f:
                ftm = pickle.load(f)  # (T, N, 1, 2)
            with open(bbx_path, "rb") as f:
                bbx5 = pickle.load(f)  # (T, N, 1, 5)
            T, N = ftm.shape[0], ftm.shape[1]
            for s in range(N):
                ftm_range = ftm[:, s, 0, 0].astype(np.float64)
                ftm_std = ftm[:, s, 0, 1].astype(np.float64)
                depth = bbx5[:, s, 0, 2].astype(np.float64)
                # Valid: depth > 0.1m AND FTM not NaN/zero
                valid = (depth > 0.1) & (np.abs(ftm_range) > 10) & (~np.isnan(ftm_range))
                if valid.sum() < 50:
                    continue
                series_list.append(PhoneSeries(
                    scene=scene,
                    seq_name=seq_dir.name,
                    phone_idx=s,
                    ftm_mm=ftm_range,
                    ftm_std_mm=ftm_std,
                    depth_m=depth,
                    valid=valid,
                    T=T,
                ))
    return series_list


# ===========================================================================
# LOSO splits
# ===========================================================================
def loso_split(all_series: List[PhoneSeries], test_scene: str
               ) -> Tuple[List[PhoneSeries], List[PhoneSeries]]:
    """Split into train / test by scene (Leave-One-Scene-Out)."""
    train = [s for s in all_series if s.scene != test_scene]
    test = [s for s in all_series if s.scene == test_scene]
    return train, test


# ===========================================================================
# Windowed Dataset for training
# ===========================================================================
class FTMDenoiseDataset(Dataset):
    """Sliding-window dataset for FTM denoising.
    
    Each sample: 
        input:  (window_size, n_features)  — FTM_m and FTM_std_m over the window
        target: scalar — depth_m at the LAST frame of the window (causal target)
    
    For non-causal mode, target is depth_m at the CENTER frame.
    """
    
    def __init__(self, series_list: List[PhoneSeries], 
                 window_size: int = 15,
                 causal: bool = True,
                 stride: int = 1):
        """
        Args:
            series_list: list of PhoneSeries to draw windows from
            window_size: number of frames in each window
            causal: if True, predict last frame; if False, predict center frame
            stride: step between consecutive windows
        """
        self.window_size = window_size
        self.causal = causal
        
        # Pre-extract all valid windows
        self.samples: List[Tuple[np.ndarray, float]] = []
        
        for ps in series_list:
            ftm_m = ps.ftm_mm / 1000.0
            ftm_std_m = ps.ftm_std_mm / 1000.0
            depth_m = ps.depth_m
            valid = ps.valid
            
            # Find contiguous valid segments
            segments = self._find_contiguous_segments(valid, min_len=window_size)
            
            for start, end in segments:
                # Slide window over this segment
                for i in range(start, end - window_size + 1, stride):
                    window_ftm = ftm_m[i:i + window_size]
                    window_std = ftm_std_m[i:i + window_size]
                    
                    if causal:
                        target_idx = i + window_size - 1
                    else:
                        target_idx = i + window_size // 2
                    
                    target = depth_m[target_idx]
                    
                    # Stack features: (window_size, 2)
                    features = np.stack([window_ftm, window_std], axis=-1)
                    self.samples.append((features.astype(np.float32), 
                                       np.float32(target)))
    
    @staticmethod
    def _find_contiguous_segments(valid: np.ndarray, min_len: int
                                  ) -> List[Tuple[int, int]]:
        """Find contiguous True segments of at least min_len."""
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
# Full-sequence Dataset for evaluation
# ===========================================================================
class FTMDenoiseFullSeqDataset:
    """For evaluation: yields full sequences (not windowed).
    
    Used to run the denoiser over complete sequences and compute per-sequence metrics.
    """
    
    def __init__(self, series_list: List[PhoneSeries]):
        self.series_list = series_list
    
    def __len__(self):
        return len(self.series_list)
    
    def __getitem__(self, idx) -> PhoneSeries:
        return self.series_list[idx]


# ===========================================================================
# Quick stats
# ===========================================================================
def dataset_stats(ds: FTMDenoiseDataset) -> dict:
    """Compute basic statistics of the dataset."""
    n = len(ds)
    if n == 0:
        return {"n_samples": 0}
    # Sample a few to get feature stats
    sample_features = np.array([ds.samples[i][0] for i in range(min(1000, n))])
    sample_targets = np.array([ds.samples[i][1] for i in range(min(1000, n))])
    return {
        "n_samples": n,
        "ftm_mean": float(sample_features[:, :, 0].mean()),
        "ftm_std": float(sample_features[:, :, 0].std()),
        "target_mean": float(sample_targets.mean()),
        "target_std": float(sample_targets.std()),
    }


if __name__ == "__main__":
    print("Loading all phone series...")
    all_series = load_all_phone_series()
    print(f"Total series: {len(all_series)}")
    for scene in SCENES:
        n = sum(1 for s in all_series if s.scene == scene)
        print(f"  {scene}: {n} series")
    
    print("\nLOSO splits + dataset sizes (window=15, causal=True, stride=1):")
    for test_scene in SCENES:
        train, test = loso_split(all_series, test_scene)
        train_ds = FTMDenoiseDataset(train, window_size=15, causal=True, stride=1)
        test_ds = FTMDenoiseDataset(test, window_size=15, causal=True, stride=1)
        print(f"  Test={test_scene}: train={len(train_ds):,} samples, test={len(test_ds):,} samples")
    
    print("\nDataset stats (fold=scene4 test):")
    train, test = loso_split(all_series, "scene4")
    train_ds = FTMDenoiseDataset(train, window_size=15, causal=True, stride=1)
    test_ds = FTMDenoiseDataset(test, window_size=15, causal=True, stride=1)
    print(f"  Train: {dataset_stats(train_ds)}")
    print(f"  Test:  {dataset_stats(test_ds)}")
    
    # Also non-causal
    print("\nNon-causal dataset sizes (window=15, stride=1):")
    for test_scene in SCENES:
        train, test = loso_split(all_series, test_scene)
        train_ds = FTMDenoiseDataset(train, window_size=15, causal=False, stride=1)
        test_ds = FTMDenoiseDataset(test, window_size=15, causal=False, stride=1)
        print(f"  Test={test_scene}: train={len(train_ds):,} samples, test={len(test_ds):,} samples")
