"""T9: Uncertainty-aware learned matching.

Retrains the tiny MatchingMLP with features derived from the heteroscedastic
denoiser (mu, sigma-hat) instead of the raw reported FTM std:

  feat = [Delta, |Delta|, depth, mu, sigma_hat,
          Delta^2/(2 sigma_hat^2),  log sigma_hat]      (7-dim)

The Mahalanobis-style exponent now uses the *calibrated, per-frame* sigma
(RLoc / Bayesian-KalmanNet idea), so "large error" is judged relative to the
current measurement quality, not a global sigma=1.3.

Train: python scripts/t9_hetero_matching.py
"""
import sys, pickle, json
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from denoising.hetero import HeteroCausalCNN, SIGMA_MIN
from scripts.phase6b_learned_matching import MatchingMLP, _compute_auc

DATA_ROOT = Path("/Users/zstar/auto_search/vifi-mot/RAN4model_dfv4p4/seqs/outdoor")
SCENES = ["scene1", "scene2", "scene3", "scene4"]
ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / "checkpoints_matching"
W = 15


class HeteroBatch:
    """Whole-sequence (mu, sigma_cal) inference."""

    def __init__(self, test_scene):
        ck = torch.load(ROOT / "checkpoints_denoising" /
                        f"hetero_cnn_{test_scene}_best.pth",
                        map_location="cpu", weights_only=False)
        self.m = HeteroCausalCNN()
        self.m.load_state_dict(ck["model_state"])
        self.m.eval()
        self.temp = float(ck["temp"])

    def run(self, ftm_m, std_m, valid):
        T = len(ftm_m)
        mu = ftm_m.copy()
        sig = np.full(T, np.nan)
        segs, st = [], None
        for i, v in enumerate(valid):
            if v and st is None:
                st = i
            elif not v and st is not None:
                if i - st >= W:
                    segs.append((st, i))
                st = None
        if st is not None and T - st >= W:
            segs.append((st, T))
        for a, b in segs:
            X = [np.stack([ftm_m[t - W + 1:t + 1], std_m[t - W + 1:t + 1]],
                          axis=-1) for t in range(a + W - 1, b)]
            if not X:
                continue
            X = torch.from_numpy(np.array(X, dtype=np.float32))
            with torch.no_grad():
                m_, v_ = self.m(X)
            idx = np.arange(a + W - 1, b)
            mu[idx] = m_.numpy()
            sig[idx] = self.temp * np.sqrt(v_.numpy())
        return mu, sig


def build_features(depth, mu, sig):
    delta = depth - mu
    return [delta, abs(delta), depth, mu, sig,
            (delta ** 2) / (2 * sig ** 2), float(np.log(sig))]


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
            sigs = np.full((T, N), np.nan)
            for p in range(N):
                r = ftm[:, p, 0, 0].astype(np.float64) / 1000.0
                s = ftm[:, p, 0, 1].astype(np.float64) / 1000.0
                v = (~np.isnan(ftm[:, p, 0, 0])) & (np.abs(ftm[:, p, 0, 0]) > 10)
                if v.sum() >= W:
                    mu, sg = hb.run(r, s, v)
                    mus[:, p], sigs[:, p] = mu, sg
            for t in range(T):
                legit = [(s, float(bbx[t, s, 0, 2])) for s in range(N)
                         if float(bbx[t, s, 0, 2]) > 0.1]
                pss = []
                if others is not None and t < others.shape[0]:
                    pss = [float(others[t, o, 0, 2])
                           for o in range(others.shape[1])
                           if float(others[t, o, 0, 2]) > 0.1]
                for p in range(N):
                    if np.isnan(sigs[t, p]):
                        continue
                    for s, depth in legit:
                        f = build_features(depth, mus[t, p], sigs[t, p])
                        (pos if s == p else neg).append(f)
                    for depth in pss:
                        neg.append(build_features(depth, mus[t, p], sigs[t, p]))
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

    model = MatchingMLP(in_features=7, hidden=16)
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
    torch.save({"model_state": model.state_dict(), "in_features": 7},
               CKPT_DIR / f"matching_hetero_{test_scene}_best.pth")
    print(f"  [{test_scene}] val AUC={best_auc:.4f} saved")
    return best_auc


if __name__ == "__main__":
    res = {}
    for sc in SCENES:
        res[sc] = train_fold(sc)
    print(json.dumps(res, indent=2))
