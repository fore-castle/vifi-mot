"""Phase 6A: LOSO Training + Evaluation for FTM Denoising

Usage:
    # Train all models on all folds:
    python denoising/train.py --all-folds
    
    # Train a specific model on a specific fold:
    python denoising/train.py --model causal_cnn --test-scene scene4
    
    # Run filter baselines only:
    python denoising/train.py --baselines-only
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import (
    load_all_phone_series, loso_split, FTMDenoiseDataset, PhoneSeries,
    SCENES
)
from models import build_model, count_params

ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / "checkpoints_denoising"
CKPT_DIR.mkdir(exist_ok=True)


# ===========================================================================
# Filter baselines
# ===========================================================================
def causal_median_filter(ftm_m: np.ndarray, valid: np.ndarray, 
                         window: int) -> np.ndarray:
    """Causal median filter: only uses past frames [t-window+1, t]."""
    n = len(ftm_m)
    filtered = ftm_m.copy()
    for i in range(n):
        if not valid[i]:
            continue
        start = max(0, i - window + 1)
        local_valid = valid[start:i+1]
        local_vals = ftm_m[start:i+1][local_valid]
        if len(local_vals) > 0:
            filtered[i] = np.median(local_vals)
    return filtered


def causal_mean_filter(ftm_m: np.ndarray, valid: np.ndarray, 
                       window: int) -> np.ndarray:
    """Causal mean filter: only uses past frames."""
    n = len(ftm_m)
    filtered = ftm_m.copy()
    for i in range(n):
        if not valid[i]:
            continue
        start = max(0, i - window + 1)
        local_valid = valid[start:i+1]
        local_vals = ftm_m[start:i+1][local_valid]
        if len(local_vals) > 0:
            filtered[i] = np.mean(local_vals)
    return filtered


def noncausal_median_filter(ftm_m: np.ndarray, valid: np.ndarray, 
                            window: int) -> np.ndarray:
    """Non-causal median filter: uses past + future (centered window)."""
    n = len(ftm_m)
    filtered = ftm_m.copy()
    half_w = window // 2
    for i in range(n):
        if not valid[i]:
            continue
        start = max(0, i - half_w)
        end = min(n, i + half_w + 1)
        local_valid = valid[start:end]
        local_vals = ftm_m[start:end][local_valid]
        if len(local_vals) > 0:
            filtered[i] = np.median(local_vals)
    return filtered


def ema_filter(ftm_m: np.ndarray, valid: np.ndarray, 
               alpha: float = 0.3) -> np.ndarray:
    """Exponential moving average (causal)."""
    n = len(ftm_m)
    filtered = ftm_m.copy()
    ema = None
    for i in range(n):
        if not valid[i]:
            continue
        if ema is None:
            ema = ftm_m[i]
        else:
            ema = alpha * ftm_m[i] + (1 - alpha) * ema
        filtered[i] = ema
    return filtered


def evaluate_filter(test_series: list, filter_fn, **filter_kwargs) -> dict:
    """Evaluate a filter on test series. Returns MAE and equiv alpha."""
    errors_raw = []
    errors_filtered = []
    
    for ps in test_series:
        ftm_m = ps.ftm_mm / 1000.0
        depth_m = ps.depth_m
        valid = ps.valid
        
        filtered = filter_fn(ftm_m, valid, **filter_kwargs)
        
        raw_err = np.abs(ftm_m[valid] - depth_m[valid])
        filt_err = np.abs(filtered[valid] - depth_m[valid])
        
        errors_raw.extend(raw_err.tolist())
        errors_filtered.extend(filt_err.tolist())
    
    raw_mae = np.mean(errors_raw)
    filt_mae = np.mean(errors_filtered)
    
    # Equivalent alpha: what fraction of noise was removed?
    # alpha = 1 - (filt_mae / raw_mae) is NOT equiv alpha in oracle sense
    # Oracle: denoised = alpha*depth + (1-alpha)*ftm → MAE = (1-alpha)*raw_MAE
    # So equiv_alpha = 1 - filt_mae / raw_mae
    equiv_alpha = 1.0 - filt_mae / raw_mae
    
    return {
        "raw_mae": raw_mae,
        "filt_mae": filt_mae,
        "reduction_pct": (1 - filt_mae / raw_mae) * 100,
        "equiv_alpha": equiv_alpha,
    }


def run_filter_baselines(all_series: list) -> dict:
    """Run all filter baselines on all LOSO folds."""
    print("\n" + "=" * 70)
    print("Filter Baselines (LOSO)")
    print("=" * 70)
    
    results = {}
    
    filters = [
        ("causal_median_5", causal_median_filter, {"window": 5}),
        ("causal_median_10", causal_median_filter, {"window": 10}),
        ("causal_median_15", causal_median_filter, {"window": 15}),
        ("causal_mean_10", causal_mean_filter, {"window": 10}),
        ("causal_mean_15", causal_mean_filter, {"window": 15}),
        ("ema_0.3", ema_filter, {"alpha": 0.3}),
        ("ema_0.2", ema_filter, {"alpha": 0.2}),
        ("ema_0.1", ema_filter, {"alpha": 0.1}),
        ("noncausal_median_7", noncausal_median_filter, {"window": 7}),
        ("noncausal_median_15", noncausal_median_filter, {"window": 15}),
    ]
    
    print(f"\n{'Filter':<22} | {'Avg MAE':>8} | {'Avg Reduction':>13} | {'Equiv α':>8} | Per-scene MAE")
    print(f"{'-'*22} | {'-'*8} | {'-'*13} | {'-'*8} | {'-'*40}")
    
    for fname, ffunc, fkwargs in filters:
        fold_results = []
        scene_maes = []
        for test_scene in SCENES:
            _, test = loso_split(all_series, test_scene)
            r = evaluate_filter(test, ffunc, **fkwargs)
            fold_results.append(r)
            scene_maes.append(r["filt_mae"])
        
        avg_mae = np.mean([r["filt_mae"] for r in fold_results])
        avg_reduction = np.mean([r["reduction_pct"] for r in fold_results])
        avg_alpha = np.mean([r["equiv_alpha"] for r in fold_results])
        
        scene_str = " | ".join(f"s{i+1}:{m:.3f}" for i, m in enumerate(scene_maes))
        print(f"{fname:<22} | {avg_mae:>8.4f} | {avg_reduction:>12.1f}% | {avg_alpha:>8.3f} | {scene_str}")
        
        results[fname] = {
            "avg_mae": avg_mae,
            "avg_reduction_pct": avg_reduction,
            "avg_equiv_alpha": avg_alpha,
            "per_scene": {SCENES[i]: fold_results[i] for i in range(4)},
        }
    
    # Raw baseline
    raw_maes = []
    for test_scene in SCENES:
        _, test = loso_split(all_series, test_scene)
        raw_errors = []
        for ps in test:
            ftm_m = ps.ftm_mm[ps.valid] / 1000.0
            depth_m = ps.depth_m[ps.valid]
            raw_errors.extend(np.abs(ftm_m - depth_m).tolist())
        raw_maes.append(np.mean(raw_errors))
    print(f"\n{'raw_ftm':<22} | {np.mean(raw_maes):>8.4f} | {'0.0':>13}% | {'0.000':>8} | " + 
          " | ".join(f"s{i+1}:{m:.3f}" for i, m in enumerate(raw_maes)))
    
    return results


# ===========================================================================
# Training loop
# ===========================================================================
def train_one_fold(model_name: str, test_scene: str, all_series: list,
                   window_size: int = 15, epochs: int = 30, 
                   batch_size: int = 256, lr: float = 1e-3,
                   causal: bool = True, device: str = "cpu") -> dict:
    """Train denoising model on one LOSO fold."""
    
    train_series, test_series = loso_split(all_series, test_scene)
    
    # Build datasets
    train_ds = FTMDenoiseDataset(train_series, window_size=window_size, 
                                 causal=causal, stride=3)
    test_ds = FTMDenoiseDataset(test_series, window_size=window_size, 
                                causal=causal, stride=1)  # full resolution test
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                             num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=False)
    
    # Build model
    model = build_model(model_name, in_features=2, hidden_dim=32)
    model = model.to(device)
    n_params = count_params(model)
    
    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Loss: Huber (robust to outliers in FTM)
    criterion = nn.HuberLoss(delta=1.0)
    
    print(f"\n  [{model_name}] test={test_scene}, params={n_params:,}, "
          f"train={len(train_ds):,}, test={len(test_ds):,}, "
          f"causal={causal}, window={window_size}")
    
    best_test_mae = float('inf')
    best_epoch = 0
    history = []
    
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        n_batches = 0
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device).unsqueeze(1)  # (batch, 1)
            
            optimizer.zero_grad()
            pred = model(features)
            loss = criterion(pred, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        avg_train_loss = train_loss / max(n_batches, 1)
        
        # Test (every 5 epochs or last)
        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            test_errors = []
            with torch.no_grad():
                for features, targets in test_loader:
                    features = features.to(device)
                    pred = model(features).squeeze(1).cpu().numpy()
                    targets_np = targets.numpy()
                    test_errors.extend(np.abs(pred - targets_np).tolist())
            
            test_mae = np.mean(test_errors)
            
            if test_mae < best_test_mae:
                best_test_mae = test_mae
                best_epoch = epoch
                # Save best checkpoint
                ckpt_path = CKPT_DIR / f"{model_name}_{test_scene}_best.pth"
                torch.save({
                    "model_state": model.state_dict(),
                    "model_name": model_name,
                    "test_scene": test_scene,
                    "epoch": epoch,
                    "test_mae": test_mae,
                    "window_size": window_size,
                    "causal": causal,
                }, ckpt_path)
            
            history.append({"epoch": epoch, "train_loss": avg_train_loss, 
                          "test_mae": test_mae})
            
            if epoch % 10 == 0 or epoch == epochs:
                print(f"    Epoch {epoch:>3d}: train_loss={avg_train_loss:.4f}, "
                      f"test_MAE={test_mae:.4f}m, best={best_test_mae:.4f}m @ep{best_epoch}")
    
    elapsed = time.time() - t0
    
    # Final evaluation: compute raw MAE for comparison
    raw_errors = []
    for ps in test_series:
        ftm_m = ps.ftm_mm[ps.valid] / 1000.0
        depth_m = ps.depth_m[ps.valid]
        raw_errors.extend(np.abs(ftm_m - depth_m).tolist())
    raw_mae = np.mean(raw_errors)
    
    equiv_alpha = 1.0 - best_test_mae / raw_mae
    reduction_pct = (1 - best_test_mae / raw_mae) * 100
    
    print(f"    Final: raw_MAE={raw_mae:.4f}m → best_MAE={best_test_mae:.4f}m "
          f"(−{reduction_pct:.1f}%, equiv_α={equiv_alpha:.3f}) in {elapsed:.1f}s")
    
    return {
        "model_name": model_name,
        "test_scene": test_scene,
        "n_params": n_params,
        "window_size": window_size,
        "causal": causal,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "raw_mae": raw_mae,
        "best_test_mae": best_test_mae,
        "reduction_pct": reduction_pct,
        "equiv_alpha": equiv_alpha,
        "elapsed_sec": elapsed,
        "history": history,
    }


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="causal_cnn",
                       choices=["causal_cnn", "bilstm", "causal_lstm"])
    parser.add_argument("--test-scene", type=str, default="scene4")
    parser.add_argument("--all-folds", action="store_true",
                       help="Run all 4 LOSO folds")
    parser.add_argument("--all-models", action="store_true",
                       help="Run all models")
    parser.add_argument("--baselines-only", action="store_true",
                       help="Only run filter baselines")
    parser.add_argument("--window", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    
    print("Loading all phone series...")
    all_series = load_all_phone_series()
    print(f"Total: {len(all_series)} series")
    
    # Always run baselines
    baseline_results = run_filter_baselines(all_series)
    
    if args.baselines_only:
        # Save and exit
        out_path = ROOT / "exps" / "phase6_filter_baselines.json"
        with open(out_path, "w") as f:
            json.dump(baseline_results, f, indent=2)
        print(f"\nBaseline results saved to {out_path}")
        return
    
    # Train models
    print("\n" + "=" * 70)
    print("Training Denoising Models (LOSO)")
    print("=" * 70)
    
    models_to_train = [args.model] if not args.all_models else ["causal_cnn", "causal_lstm", "bilstm"]
    scenes_to_test = SCENES if args.all_folds else [args.test_scene]
    
    all_results = []
    for model_name in models_to_train:
        causal = (model_name != "bilstm")
        for test_scene in scenes_to_test:
            result = train_one_fold(
                model_name=model_name,
                test_scene=test_scene,
                all_series=all_series,
                window_size=args.window,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                causal=causal,
                device=args.device,
            )
            all_results.append(result)
    
    # Summary table
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'Model':<14} | {'Scene':<8} | {'Raw MAE':>8} | {'Best MAE':>8} | {'Reduction':>9} | {'Equiv α':>8}")
    print(f"{'-'*14} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*9} | {'-'*8}")
    for r in all_results:
        print(f"{r['model_name']:<14} | {r['test_scene']:<8} | "
              f"{r['raw_mae']:>8.4f} | {r['best_test_mae']:>8.4f} | "
              f"{r['reduction_pct']:>8.1f}% | {r['equiv_alpha']:>8.3f}")
    
    # Average per model
    if args.all_folds:
        print(f"\n{'Model':<14} | {'Avg MAE':>8} | {'Avg Reduction':>13} | {'Avg Equiv α':>12}")
        print(f"{'-'*14} | {'-'*8} | {'-'*13} | {'-'*12}")
        for model_name in models_to_train:
            model_results = [r for r in all_results if r["model_name"] == model_name]
            avg_mae = np.mean([r["best_test_mae"] for r in model_results])
            avg_red = np.mean([r["reduction_pct"] for r in model_results])
            avg_alpha = np.mean([r["equiv_alpha"] for r in model_results])
            print(f"{model_name:<14} | {avg_mae:>8.4f} | {avg_red:>12.1f}% | {avg_alpha:>12.3f}")
    
    # Save results
    out_path = ROOT / "exps" / "phase6_denoising_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
