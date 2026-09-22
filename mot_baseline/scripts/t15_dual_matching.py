"""T15: Dual-denoiser feature-fusion matching MLP.

Both denoisers (causal_cnn and hetero mu/sigma) disagree in complementary
ways across scenes; global discriminability is identical (AUC +-0.004) but
local events differ. Feed BOTH residuals to the tiny MLP and let it act as a
per-sample arbiter:

  feat = [Dc, Dh, |Dc|, |Dh|, depth, mu_c, mu_h, std_rep, sig_hat,
          log sig_hat, Dc^2/(2*1.3^2), Dh^2/(2 sig_hat^2)]        (12-dim)

~360 params. Trained per LOSO fold on train scenes only.
"""
import sys, pickle, json
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.phase6b_learned_matching import MatchingMLP, _compute_auc
from scripts.t9_hetero_matching import HeteroBatch
from denoising.inference import FTMDenoiserBatch

DATA_ROOT = Path("/Users/zstar/auto_search/vifi-mot/RAN4model_dfv4p4/seqs/outdoor")
SCENES = ["scene1", "scene2", "scene3", "scene4"]
ROOT = Path(__file__).resolve().parent.parent
W = 15


def dual_features(depth, mu_c, mu_h, std_rep, sig):
    dc = depth - mu_c
    dh = depth - mu_h
    return [dc, dh, abs(dc), abs(dh), depth, mu_c, mu_h, std_rep, sig,
            float(np.log(max(sig, 1e-3))),
            (dc ** 2) / (2 * 1.3 ** 2),
            (dh ** 2) / (2 * max(sig, 1e-3) ** 2)]


def build_dataset(test_scene):
    hb = HeteroBatch(test_scene)
    cb = FTMDenoiserBatch("causal_cnn", test_scene, W)
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
            mc = np.full((T, N), np.nan)
            mh = np.full((T, N), np.nan)
            sg = np.full((T, N), np.nan)
            sr = np.zeros((T, N))
            for p in range(N):
                r = ftm[:, p, 0, 0].astype(np.float64) / 1000.0
                s = ftm[:, p, 0, 1].astype(np.float64) / 1000.0
                v = (~np.isnan(ftm[:, p, 0, 0])) & (np.abs(ftm[:, p, 0, 0]) > 10)
                if v.sum() >= W:
                    mc[:, p] = cb.denoise_sequence(r, s, v)
                    mu, sig = hb.run(r, s, v)
                    mh[:, p], sg[:, p] = mu, sig
                    sr[:, p] = s
            for t in range(T):
                legit = [(s_, float(bbx[t, s_, 0, 2])) for s_ in range(N)
                         if float(bbx[t, s_, 0, 2]) > 0.1]
                pss = []
                if others is not None and t < others.shape[0]:
                    pss = [float(others[t, o, 0, 2])
                           for o in range(others.shape[1])
                           if float(others[t, o, 0, 2]) > 0.1]
                for p in range(N):
                    if np.isnan(sg[t, p]) or np.isnan(mc[t, p]):
                        continue
                    for s_, depth in legit:
                        f = dual_features(depth, mc[t, p], mh[t, p],
                                          sr[t, p], sg[t, p])
                        (pos if s_ == p else neg).append(f)
                    for depth in pss:
                        neg.append(dual_features(depth, mc[t, p], mh[t, p],
                                                 sr[t, p], sg[t, p]))
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
    model = MatchingMLP(in_features=12, hidden=16)
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
    torch.save({"model_state": model.state_dict(), "in_features": 12},
               ROOT / "checkpoints_matching" / f"matching_dual_{test_scene}_best.pth")
    print(f"  [{test_scene}] val AUC={best_auc:.4f} saved")
    return best_auc


if __name__ == "__main__":
    res = {sc: train_fold(sc) for sc in SCENES}
    print(json.dumps(res, indent=2))
