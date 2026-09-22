"""T12: Retrain the 6-dim MatchingMLP on hetero-mu denoised features.

Same features/architecture as Phase 6B (6-dim, reported std), only the
denoised range comes from the heteroscedastic CNN's mu (which beats the
original causal_cnn on all 4 folds: MAE -3.5% .. -11.7%). Eliminates the
train/test feature-distribution shift that hurt t10 on scene3.
"""
import sys, pickle, json
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.phase6b_learned_matching import MatchingMLP, _compute_auc, _build_features
from scripts.t9_hetero_matching import HeteroBatch

DATA_ROOT = Path("/Users/zstar/auto_search/vifi-mot/RAN4model_dfv4p4/seqs/outdoor")
SCENES = ["scene1", "scene2", "scene3", "scene4"]
ROOT = Path(__file__).resolve().parent.parent
W = 15


def build_dataset(test_scene):
    hb = HeteroBatch(test_scene)
    pos, neg = [], []
    for scene in SCENES:
        for seq_dir in sorted((DATA_ROOT / scene).iterdir()):
            sync = seq_dir / "sync_ts16_dfv4p4"
            fp, bp = sync / "FTM_li_sync_dfv4p4.pkl", sync / "BBX5_sync_dfv4p4.pkl"
            if not fp.exists() or not bp.exists():
                continue
            ftm = pickle.load(open(fp, "rb"))
            bbx = pickle.load(open(bp, "rb"))
            op = sync / "BBX5_Others_sync_dfv4p4.pkl"
            others = pickle.load(open(op, "rb")) if op.exists() else None
            T, N = min(ftm.shape[0], bbx.shape[0]), ftm.shape[1]
            mus = np.full((T, N), np.nan)
            stds = np.zeros((T, N))
            ok = np.zeros((T, N), dtype=bool)
            for p in range(N):
                r = ftm[:, p, 0, 0].astype(np.float64) / 1000.0
                s = ftm[:, p, 0, 1].astype(np.float64) / 1000.0
                v = (~np.isnan(ftm[:, p, 0, 0])) & (np.abs(ftm[:, p, 0, 0]) > 10)
                if v.sum() >= W:
                    mu, sg = hb.run(r, s, v)
                    mus[:, p] = mu
                    stds[:, p] = s
                    ok[:, p] = v & ~np.isnan(sg)   # only frames with real mu
            for t in range(T):
                legit = [(s, float(bbx[t, s, 0, 2])) for s in range(N)
                         if float(bbx[t, s, 0, 2]) > 0.1]
                pss = []
                if others is not None and t < others.shape[0]:
                    pss = [float(others[t, o, 0, 2])
                           for o in range(others.shape[1])
                           if float(others[t, o, 0, 2]) > 0.1]
                for p in range(N):
                    if not ok[t, p]:
                        continue
                    for s, depth in legit:
                        f = _build_features(depth, mus[t, p], stds[t, p])
                        (pos if s == p else neg).append(f)
                    for depth in pss:
                        neg.append(_build_features(depth, mus[t, p], stds[t, p]))
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
    X, y, n_pos, n_neg = build_dataset(test_scene)
    print(f"  [{test_scene}] {n_pos} pos / {n_neg} neg")
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
    best_auc, best_state = 0, None
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_tr)
        for i in range(0, n_tr, 1024):
            b = perm[i:i + 1024]
            opt.zero_grad()
            pr = model(Xt[b])
            w = torch.where(yt[b] > 0.5, pw, 1.0)
            loss = nn.functional.binary_cross_entropy(pr, yt[b], weight=w)
            loss.backward()
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
    torch.save({"model_state": model.state_dict(), "in_features": 6},
               ROOT / "checkpoints_matching" / f"matching_hmu_{test_scene}_best.pth")
    print(f"  [{test_scene}] val AUC={best_auc:.4f} saved")
    return best_auc


if __name__ == "__main__":
    res = {sc: train_fold(sc) for sc in SCENES}
    print(json.dumps(res, indent=2))
