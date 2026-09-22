"""S11: quantify the LOSO leakage in the learned-matching pipeline.

Two leaks found in the baseline (verified by reading the code):

L1  scripts/phase6b_learned_matching.py:59  `for scene in SCENES:` with NO
    `test_scene` exclusion -> the matching MLP for fold k is trained on data
    from scene k as well. The 80/20 "val" split at :196-201 is drawn from that
    same contaminated pool, so model selection is contaminated too.
    (The same line was copied into t9 / t12 / t15, i.e. my own scripts.)

L2  denoising/train.py:269-295  selects the denoiser checkpoint by MAE on the
    HELD-OUT TEST scene (`if test_mae < best_test_mae: torch.save(...)`), with
    no validation split at all.

This script retrains the 6-dim matching MLP with a clean protocol:
  - training pool  = scenes != test_scene            (fixes L1)
  - val split      = 10% of the training pool only   (fixes L1 selection)
  - denoiser       = unchanged causal_cnn ckpt        (L2 still present, so the
                     measured drop isolates L1 alone)
Everything else (features, architecture, epochs, seeds, class balance) is
identical to phase6b_learned_matching.py.

Output: checkpoints_matching/matching_clean_{scene}_best.pth
Then evaluate with scripts/v2_experiments.py config `base_clean`.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

MOT = Path("/Users/zstar/auto_search/vifi-mot/mot_baseline")
sys.path.insert(0, str(MOT))

from denoising.inference import FTMDenoiserBatch                  # noqa: E402
from scripts.phase6b_learned_matching import (                    # noqa: E402
    MatchingMLP, _build_features, _compute_auc)

DATA_ROOT = Path("/Users/zstar/auto_search/vifi-mot/RAN4model_dfv4p4/seqs/outdoor")
SCENES = ["scene1", "scene2", "scene3", "scene4"]
OUT = Path(__file__).resolve().parent


def build_dataset(test_scene, exclude_test: bool):
    """exclude_test=True -> clean LOSO; False -> reproduce the leaky baseline."""
    denoiser = FTMDenoiserBatch("causal_cnn", test_scene, window_size=15)
    pos, neg = [], []
    for scene in SCENES:
        if exclude_test and scene == test_scene:
            continue                      # <-- the missing line
        for seq_dir in sorted((DATA_ROOT / scene).iterdir()):
            if not seq_dir.is_dir():
                continue
            sync = seq_dir / "sync_ts16_dfv4p4"
            fp, bp = sync / "FTM_li_sync_dfv4p4.pkl", sync / "BBX5_sync_dfv4p4.pkl"
            if not fp.exists() or not bp.exists():
                continue
            ftm = pickle.load(open(fp, "rb"))
            bbx5 = pickle.load(open(bp, "rb"))
            op = sync / "BBX5_Others_sync_dfv4p4.pkl"
            others = pickle.load(open(op, "rb")) if op.exists() else None
            T, N = min(ftm.shape[0], bbx5.shape[0]), ftm.shape[1]
            den = np.full((T, N), np.nan)
            std = np.zeros((T, N))
            ok = np.zeros((T, N), dtype=bool)
            for p in range(N):
                r = ftm[:, p, 0, 0].astype(np.float64)
                s = ftm[:, p, 0, 1].astype(np.float64)
                v = (~np.isnan(r)) & (np.abs(r) > 10)
                if v.sum() >= 15:
                    den[:, p] = denoiser.denoise_sequence(r / 1000.0, s / 1000.0, v)
                    std[:, p] = s / 1000.0
                    ok[:, p] = v
            for t in range(T):
                legit = [(k, float(bbx5[t, k, 0, 2])) for k in range(N)
                         if float(bbx5[t, k, 0, 2]) > 0.1]
                pss = []
                if others is not None and t < others.shape[0]:
                    pss = [float(others[t, o, 0, 2]) for o in range(others.shape[1])
                           if float(others[t, o, 0, 2]) > 0.1]
                for p in range(N):
                    if not ok[t, p]:
                        continue
                    for k, depth in legit:
                        f = _build_features(depth, den[t, p], std[t, p])
                        (pos if k == p else neg).append(f)
                    for depth in pss:
                        neg.append(_build_features(depth, den[t, p], std[t, p]))
    n_pos = len(pos)
    tgt = min(len(neg), n_pos * 3)
    rng = np.random.RandomState(42)
    if len(neg) > tgt:
        neg = [neg[i] for i in rng.choice(len(neg), tgt, replace=False)]
    X = np.array(pos + neg, dtype=np.float32)
    y = np.array([1.0] * n_pos + [0.0] * len(neg), dtype=np.float32)
    return X, y, n_pos, len(neg)


def train_fold(test_scene, epochs=30, lr=1e-3):
    torch.manual_seed(0)
    X, y, n_pos, n_neg = build_dataset(test_scene, exclude_test=True)
    print(f"  [{test_scene}] clean pool: {n_pos} pos / {n_neg} neg")
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(X))
    n_tr = int(0.8 * len(X))
    tr, va = idx[:n_tr], idx[n_tr:]
    Xt, yt = torch.from_numpy(X[tr]), torch.from_numpy(y[tr])
    Xv, yv = torch.from_numpy(X[va]), torch.from_numpy(y[va])
    pw = n_neg / max(n_pos, 1)

    model = MatchingMLP(in_features=6, hidden=16)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best_auc, best_state = 0.0, None
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_tr)
        for i in range(0, n_tr, 1024):
            b = perm[i:i + 1024]
            opt.zero_grad()
            pr = model(Xt[b])
            w = torch.where(yt[b] > 0.5, pw, 1.0)
            nn.functional.binary_cross_entropy(pr, yt[b], weight=w).backward()
            opt.step()
        sched.step()
        if ep % 5 == 0 or ep == 1:
            model.eval()
            with torch.no_grad():
                auc = _compute_auc(model(Xv).numpy(), yv.numpy())
            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    torch.save({"model_state": model.state_dict(), "in_features": 6,
                "protocol": "clean_loso"},
               MOT / "checkpoints_matching" / f"matching_clean_{test_scene}_best.pth")
    print(f"  [{test_scene}] val AUC={best_auc:.4f} (train-pool val) saved")
    return best_auc


if __name__ == "__main__":
    res = {sc: train_fold(sc) for sc in SCENES}
    (OUT / "clean_train_auc.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
