"""Phase 6C: FTM Denoising Inference for MOT Integration

Provides a stateful denoiser that can be called frame-by-frame in the MOT loop.
Maintains a sliding window buffer per phone for causal inference.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from denoising.models import build_model

CKPT_DIR = Path(__file__).resolve().parent.parent / "checkpoints_denoising"


class FTMDenoiser:
    """Stateful FTM denoiser for online MOT integration.
    
    Usage:
        denoiser = FTMDenoiser("causal_cnn", "scene4", window_size=15)
        
        # Each frame:
        denoised_ftm = denoiser.denoise(ftm_m, ftm_valid)
    """
    
    def __init__(self, model_name: str = "causal_cnn", 
                 test_scene: str = "scene4",
                 window_size: int = 15,
                 device: str = "cpu",
                 ckpt_path: Optional[str] = None):
        """
        Args:
            model_name: "causal_cnn", "causal_lstm", or "bilstm"
            test_scene: which fold checkpoint to load (e.g. "scene4" loads
                       the model trained on scene1+2+3, tested on scene4)
            window_size: must match training window size
            device: "cpu" or "cuda"
            ckpt_path: override checkpoint path (if None, uses default naming)
        """
        self.model_name = model_name
        self.window_size = window_size
        self.device = device
        self.causal = (model_name != "bilstm")
        
        # Load model
        if ckpt_path is None:
            ckpt_path = CKPT_DIR / f"{model_name}_{test_scene}_best.pth"
        else:
            ckpt_path = Path(ckpt_path)
        
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.model = build_model(model_name, in_features=2, hidden_dim=32)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.model.to(device)
        
        # Per-phone sliding window buffers
        # Key: phone_index, Value: list of (ftm_m, ftm_std_m)
        self._buffers: dict[int, list] = {}
    
    def reset(self):
        """Reset all buffers (call at start of new sequence)."""
        self._buffers.clear()
    
    def denoise(self, ftm_m: np.ndarray, ftm_valid: np.ndarray,
                ftm_std_m: Optional[np.ndarray] = None) -> np.ndarray:
        """Denoise FTM for current frame.
        
        Args:
            ftm_m: (N_phones,) raw FTM range in meters
            ftm_valid: (N_phones,) bool mask
            ftm_std_m: (N_phones,) FTM std in meters (optional, zeros if None)
        
        Returns:
            denoised_ftm: (N_phones,) denoised range in meters
                         (invalid phones remain as raw value)
        """
        n_phones = len(ftm_m)
        if ftm_std_m is None:
            ftm_std_m = np.zeros(n_phones)
        
        denoised = ftm_m.copy()
        
        for p in range(n_phones):
            if not ftm_valid[p] or np.isnan(ftm_m[p]):
                continue
            
            # Update buffer
            if p not in self._buffers:
                self._buffers[p] = []
            self._buffers[p].append((ftm_m[p], ftm_std_m[p]))
            
            # Keep buffer at window_size
            if len(self._buffers[p]) > self.window_size:
                self._buffers[p] = self._buffers[p][-self.window_size:]
            
            # Need at least window_size frames for prediction
            buf = self._buffers[p]
            if len(buf) < self.window_size:
                # Not enough history — fall back to raw value
                continue
            
            # Build input tensor: (1, window_size, 2)
            features = np.array(buf, dtype=np.float32)  # (window_size, 2)
            x = torch.from_numpy(features).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                pred = self.model(x)  # (1, 1)
            
            denoised[p] = pred.item()
        
        return denoised


class FTMDenoiserBatch:
    """Batch denoiser for offline processing of full sequences.
    
    More efficient than frame-by-frame: processes all windows at once.
    """
    
    def __init__(self, model_name: str = "causal_cnn",
                 test_scene: str = "scene4",
                 window_size: int = 15,
                 device: str = "cpu",
                 ckpt_path: Optional[str] = None):
        self.model_name = model_name
        self.window_size = window_size
        self.device = device
        self.causal = (model_name != "bilstm")
        
        if ckpt_path is None:
            ckpt_path = CKPT_DIR / f"{model_name}_{test_scene}_best.pth"
        else:
            ckpt_path = Path(ckpt_path)
        
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.model = build_model(model_name, in_features=2, hidden_dim=32)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.model.to(device)
    
    def denoise_sequence(self, ftm_m: np.ndarray, ftm_std_m: np.ndarray,
                         valid: np.ndarray) -> np.ndarray:
        """Denoise a full sequence for one phone.
        
        Args:
            ftm_m: (T,) FTM range in meters
            ftm_std_m: (T,) FTM std in meters
            valid: (T,) bool mask
        
        Returns:
            denoised: (T,) denoised range (invalid frames keep raw value)
        """
        T = len(ftm_m)
        denoised = ftm_m.copy()
        w = self.window_size
        
        # Find contiguous valid segments
        segments = self._find_segments(valid, min_len=w)
        
        for start, end in segments:
            # Extract segment
            seg_ftm = ftm_m[start:end]
            seg_std = ftm_std_m[start:end]
            seg_len = end - start
            
            # Build all windows for this segment
            if self.causal:
                # Causal: window [t-w+1, t], predict t
                windows = []
                for t in range(w - 1, seg_len):
                    feat = np.stack([seg_ftm[t-w+1:t+1], seg_std[t-w+1:t+1]], axis=-1)
                    windows.append(feat)
                
                if not windows:
                    continue
                
                # Batch inference
                batch = torch.from_numpy(np.array(windows, dtype=np.float32)).to(self.device)
                with torch.no_grad():
                    preds = self.model(batch).squeeze(1).cpu().numpy()
                
                # Write back
                for i, t in enumerate(range(w - 1, seg_len)):
                    denoised[start + t] = preds[i]
            else:
                # Non-causal: centered window, predict center
                half_w = w // 2
                windows = []
                target_indices = []
                for t in range(half_w, seg_len - half_w):
                    feat = np.stack([seg_ftm[t-half_w:t+half_w+1][:w], 
                                   seg_std[t-half_w:t+half_w+1][:w]], axis=-1)
                    if feat.shape[0] == w:
                        windows.append(feat)
                        target_indices.append(t)
                
                if not windows:
                    continue
                
                batch = torch.from_numpy(np.array(windows, dtype=np.float32)).to(self.device)
                with torch.no_grad():
                    preds = self.model(batch).squeeze(1).cpu().numpy()
                
                for i, t in enumerate(target_indices):
                    denoised[start + t] = preds[i]
        
        return denoised
    
    @staticmethod
    def _find_segments(valid: np.ndarray, min_len: int):
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


if __name__ == "__main__":
    # Quick test
    print("Testing FTMDenoiser (online, causal_cnn, scene4)...")
    denoiser = FTMDenoiser("causal_cnn", "scene4")
    
    # Simulate 20 frames
    np.random.seed(42)
    for t in range(20):
        ftm_m = np.array([3.5 + np.random.randn() * 1.5])
        ftm_valid = np.array([True])
        ftm_std_m = np.array([0.5])
        denoised = denoiser.denoise(ftm_m, ftm_valid, ftm_std_m)
        if t >= 14:  # after buffer fills
            print(f"  t={t:2d}: raw={ftm_m[0]:.3f}m → denoised={denoised[0]:.3f}m")
    
    print("\nTesting FTMDenoiserBatch (offline, causal_cnn, scene4)...")
    batch_denoiser = FTMDenoiserBatch("causal_cnn", "scene4")
    ftm_seq = 4.0 + np.random.randn(100) * 1.5
    std_seq = np.full(100, 0.5)
    valid_seq = np.ones(100, dtype=bool)
    denoised_seq = batch_denoiser.denoise_sequence(ftm_seq, std_seq, valid_seq)
    
    raw_mae = np.abs(ftm_seq - 4.0).mean()
    den_mae = np.abs(denoised_seq - 4.0).mean()
    print(f"  Raw MAE to truth: {raw_mae:.3f}m")
    print(f"  Denoised MAE:     {den_mae:.3f}m")
    print(f"  Reduction: {(1 - den_mae/raw_mae)*100:.1f}%")
