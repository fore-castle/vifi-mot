"""T4: Heteroscedastic FTM denoiser — predicts (mu, sigma) per frame.

Inspired by RLoc (IMWUT'23, Gaussian NLL uncertainty quantification) and
Bayesian KalmanNet (TSP'25, beta-annealed second-moment loss + ANEES
calibration).

Loss:  L = (1-beta)*(y-mu)^2 + beta*[ log(sigma^2)/2 + (y-mu)^2/(2 sigma^2) ]
       beta annealed 0 -> 1 (learn mean first, then variance).
Calibration: temperature s* on held-out TRAIN split s.t. mean z^2 == 1
       (ANEES), applied at inference: sigma_cal = s* . sigma.

Train:  python denoising/hetero.py --train            # all 4 folds
Usage:  den = HeteroFTMDenoiser("scene1"); mu, sig = den.denoise(...)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from denoising.models import CausalConv1d
from denoising.dataset import (load_all_phone_series, loso_split,
                               FTMDenoiseDataset, SCENES)

CKPT_DIR = Path(__file__).resolve().parent.parent / "checkpoints_denoising"
SIGMA_MIN = 0.10  # meters, variance floor


class HeteroCausalCNN(nn.Module):
    """CausalDilatedCNN trunk with a 2-channel (mu, s) head."""

    def __init__(self, in_features: int = 2, hidden_dim: int = 32,
                 n_layers: int = 4, kernel_size: int = 3):
        super().__init__()
        self.input_proj = nn.Conv1d(in_features, hidden_dim, 1)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(CausalConv1d(hidden_dim, hidden_dim,
                                            kernel_size, 2 ** i))
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        h = self.input_proj(x)
        for conv, norm in zip(self.layers, self.norms):
            r = h
            h = conv(h)
            h = norm(h.transpose(1, 2)).transpose(1, 2)
            h = F.gelu(h) + r
        out = self.head(h[:, :, -1])          # (B, 2)
        mu = out[:, 0]
        var = F.softplus(out[:, 1]) + SIGMA_MIN ** 2
        return mu, var


def train_fold(test_scene: str, window=15, epochs=24, batch=256,
               lr=1e-3, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    all_series = load_all_phone_series()
    train_series, _ = loso_split(all_series, test_scene)
    ds = FTMDenoiseDataset(train_series, window_size=window, causal=True,
                           stride=2)
    n_val = max(1, int(0.1 * len(ds)))
    tr, va = random_split(ds, [len(ds) - n_val, n_val],
                          generator=torch.Generator().manual_seed(seed))
    dl_tr = DataLoader(tr, batch_size=batch, shuffle=True)
    dl_va = DataLoader(va, batch_size=1024)

    model = HeteroCausalCNN()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_va, best_state = float("inf"), None

    for ep in range(epochs):
        beta = min(1.0, ep / (epochs / 2))
        model.train()
        for xb, yb in dl_tr:
            mu, var = model(xb)
            nll = 0.5 * torch.log(var) + (yb - mu) ** 2 / (2 * var)
            loss = ((1 - beta) * (yb - mu) ** 2 + beta * nll).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        # validate with full NLL
        model.eval()
        va_nll, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in dl_va:
                mu, var = model(xb)
                va_nll += float((0.5 * torch.log(var)
                                 + (yb - mu) ** 2 / (2 * var)).sum())
                n += len(yb)
        va_nll /= n
        if beta >= 1.0 and va_nll < best_va:
            best_va, best_state = va_nll, {
                k: v.clone() for k, v in model.state_dict().items()}
        print(f"  [{test_scene}] ep{ep:02d} beta={beta:.2f} valNLL={va_nll:.4f}",
              flush=True)

    model.load_state_dict(best_state)
    # --- ANEES temperature calibration on the val split ---
    model.eval()
    z2s = []
    with torch.no_grad():
        for xb, yb in dl_va:
            mu, var = model(xb)
            z2s.append(((yb - mu) ** 2 / var).numpy())
    anees = float(np.concatenate(z2s).mean())
    temp = float(np.sqrt(max(anees, 1e-6)))   # sigma_cal = temp * sigma
    print(f"  [{test_scene}] valNLL={best_va:.4f} ANEES={anees:.3f} temp={temp:.3f}")

    ckpt = {"model_state": model.state_dict(), "temp": temp,
            "val_nll": best_va, "anees": anees, "window": window}
    path = CKPT_DIR / f"hetero_cnn_{test_scene}_best.pth"
    torch.save(ckpt, path)
    print(f"  saved -> {path}")


class HeteroFTMDenoiser:
    """Online frame-by-frame (mu, sigma) denoiser, mirrors FTMDenoiser API."""

    def __init__(self, test_scene: str, window_size: int = 15):
        ckpt = torch.load(CKPT_DIR / f"hetero_cnn_{test_scene}_best.pth",
                          map_location="cpu", weights_only=False)
        self.model = HeteroCausalCNN()
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.temp = float(ckpt["temp"])
        self.window_size = window_size
        self._buffers: dict[int, list] = {}

    def reset(self):
        self._buffers.clear()

    def denoise(self, ftm_m, ftm_valid, ftm_std_m=None):
        """Returns (denoised, sigma). Fallback: raw value, sigma=NaN."""
        n = len(ftm_m)
        if ftm_std_m is None:
            ftm_std_m = np.zeros(n)
        out_mu = ftm_m.copy()
        out_sig = np.full(n, np.nan)
        for p in range(n):
            if not ftm_valid[p] or np.isnan(ftm_m[p]):
                continue
            buf = self._buffers.setdefault(p, [])
            buf.append((ftm_m[p], ftm_std_m[p]))
            if len(buf) > self.window_size:
                del buf[:len(buf) - self.window_size]
            if len(buf) < self.window_size:
                continue
            x = torch.from_numpy(np.array(buf, dtype=np.float32)).unsqueeze(0)
            with torch.no_grad():
                mu, var = self.model(x)
            out_mu[p] = float(mu.item())
            out_sig[p] = self.temp * float(np.sqrt(var.item()))
        return out_mu, out_sig


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--scenes", nargs="+", default=SCENES)
    args = ap.parse_args()
    if args.train:
        for sc in args.scenes:
            print(f"=== fold test={sc} ===")
            train_fold(sc)
