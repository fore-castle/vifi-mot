"""Strict leave-one-scene-out training driver for the my_vifi affinity model.

Re-uses my_vifi's Dataset / Model / Loss but overrides the fold split to be
**by RAN4model scene** (each fold leaves one of scene1..scene4 entirely out
of training), then trains 4 checkpoints + 4 norm_stats files into

    mot_baseline/checkpoints_loso/full/fold{1..4}_best.pth
    mot_baseline/data/norm_stats_loso_fold{1..4}.npz

Indoor scene0 stays in the train pool every fold (it is treated as a held-out
domain for the OOD evaluation step, never used as a 4-fold test scene).

CPU-only on macOS. Lightweight: ~90k-param model, fp32, no autocast.
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

MOT_ROOT = Path(__file__).resolve().parent.parent
MY_VIFI = "/Users/zstar/test/vifi/my_vifi"
DATA_ROOT = Path("/Users/zstar/auto_search/vifi-mot/RAN4model_dfv4p4")
sys.path.insert(0, MY_VIFI)

import config as cfg_lib                        # noqa: E402
import model as model_lib                       # noqa: E402
import loss as loss_lib                         # noqa: E402
from dataset import ViFiDataset, compute_norm_stats   # noqa: E402


# ---- Per-scene sequence enumeration -----------------------------------------

def list_scene_sequences(scene_name: str):
    base = DATA_ROOT / "seqs" / "outdoor" / scene_name
    return sorted(p.name for p in base.iterdir() if p.is_dir())


def build_loso_test_sequences():
    """Return [[scene1 seqs], [scene2 seqs], [scene3 seqs], [scene4 seqs]]."""
    return [
        list_scene_sequences("scene1"),
        list_scene_sequences("scene2"),
        list_scene_sequences("scene3"),
        list_scene_sequences("scene4"),
    ]


# ---- Train one fold ---------------------------------------------------------

def train_one_fold(fold: int,
                   test_sequences,
                   ckpt_dir: Path,
                   norm_stats_path: Path,
                   epochs: int = 30,
                   batch_size: int = 32,
                   lr: float = 1e-3,
                   num_workers: int = 0,
                   device: str = "cpu") -> Path:
    """Train one LOSO fold and return best-checkpoint path."""
    print(f"\n{'='*70}\nFOLD {fold}: test = {len(test_sequences[fold-1])} sequences\n{'='*70}",
          flush=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- assemble Config ----------------------------------------------------
    config = cfg_lib.Config()
    config.data_root = DATA_ROOT
    config.checkpoint_dir = ckpt_dir
    config.fold = fold
    config.test_sequences = test_sequences
    config.epochs = epochs
    config.batch_size = batch_size
    config.lr = lr
    config.num_workers = num_workers
    config.device = device
    config.val_ratio = 0.1
    config.val_seed = 2026

    dev = torch.device(device)

    # --- norm_stats: compute on this fold's train split, save to npz --------
    norm_stats = compute_norm_stats(config.data_root, config=config)
    np.savez(norm_stats_path,
             cam_mean=norm_stats["cam_mean"], cam_std=norm_stats["cam_std"],
             ph_mean=norm_stats["ph_mean"], ph_std=norm_stats["ph_std"])
    print(f"  saved norm_stats to {norm_stats_path}", flush=True)

    # --- datasets -----------------------------------------------------------
    train_ds = ViFiDataset(config.data_root, split="train", fold=fold,
                           config=config, norm_stats=norm_stats)
    val_ds = ViFiDataset(config.data_root, split="val", fold=fold,
                         config=config, norm_stats=norm_stats)
    if len(val_ds) == 0:
        val_ds = train_ds
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, drop_last=False)
    print(f"  train_samples={len(train_ds)}  val_samples={len(val_ds)}",
          flush=True)

    # --- model + loss --------------------------------------------------------
    model = model_lib.MultimodalNetwork(config).to(dev)
    criterion = loss_lib.AffinityLoss(config.Nm_phone, config.Nm_camera).to(dev)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=config.momentum)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[epochs // 2], gamma=0.1)

    best_acc = 0.0
    best_path = None

    for epoch in range(1, epochs + 1):
        t_ep = time.time()

        model.train()
        tr_loss = 0.0
        tr_acc = 0.0
        n_batches = 0
        for batch in train_loader:
            cam = batch["camera"].to(dev)
            ph = batch["phone"].to(dev)
            cm = batch["camera_mask"].to(dev)
            pm = batch["phone_mask"].to(dev)
            am = batch["aff_mat"].to(dev)
            optimizer.zero_grad()
            out = model(cam, ph, cm, pm)
            _, _, _, loss, _, _, acc, _ = criterion(out, am, pm, cm)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
            tr_acc += acc.item()
            n_batches += 1
        tr_loss /= max(n_batches, 1)
        tr_acc /= max(n_batches, 1)

        # ---- val ----
        model.eval()
        v_loss = 0.0
        v_acc = 0.0
        n_v = 0
        with torch.no_grad():
            for batch in val_loader:
                cam = batch["camera"].to(dev)
                ph = batch["phone"].to(dev)
                cm = batch["camera_mask"].to(dev)
                pm = batch["phone_mask"].to(dev)
                am = batch["aff_mat"].to(dev)
                out = model(cam, ph, cm, pm)
                _, _, _, loss, _, _, acc, _ = criterion(out, am, pm, cm)
                v_loss += loss.item()
                v_acc += acc.item()
                n_v += 1
        v_loss /= max(n_v, 1)
        v_acc /= max(n_v, 1)

        scheduler.step()
        dt = time.time() - t_ep
        print(f"  ep{epoch:>3d}/{epochs} train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
              f"val_loss={v_loss:.4f} val_acc={v_acc:.4f}  [{dt:.1f}s]",
              flush=True)

        if v_acc > best_acc and v_acc < 1.0:
            if best_path and best_path.exists():
                best_path.unlink()
            best_acc = v_acc
            best_path = ckpt_dir / f"fold{fold}_best_epoch{epoch}_acc{v_acc:.4f}.pth"
            torch.save(model, best_path)
            print(f"    new best → {best_path.name}", flush=True)

    print(f"FOLD {fold} done. best val acc={best_acc:.4f}  ckpt={best_path}", flush=True)
    return best_path


# ---- Main entry -------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=str, default="1,2,3,4")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    test_sequences = build_loso_test_sequences()
    for fi, seqs in enumerate(test_sequences, 1):
        print(f"  fold {fi}: test scene{fi} → {len(seqs)} sequences")

    ckpt_dir = MOT_ROOT / "checkpoints_loso" / "full"
    norm_dir = MOT_ROOT / "data"
    norm_dir.mkdir(parents=True, exist_ok=True)

    folds = [int(x) for x in args.folds.split(",")]
    for fold in folds:
        norm_path = norm_dir / f"affinity_norm_stats_loso_fold{fold}.npz"
        train_one_fold(
            fold=fold,
            test_sequences=test_sequences,
            ckpt_dir=ckpt_dir,
            norm_stats_path=norm_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
        )


if __name__ == "__main__":
    main()
