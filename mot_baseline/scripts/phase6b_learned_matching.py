"""Phase 6B: Learned Matching — replace Gaussian compat with learned function

Design principles (addressing Method D failure modes):
1. Use DENOISED FTM (not raw) — 50% less noise
2. Tiny MLP (~200 params, not 90k) — prevent overfitting
3. Instantaneous matching only (no temporal modeling) — EMA handles accumulation
4. Binary classification: (depth, denoised_ftm) → same person?

Input features (6-dim):
  0: Δ = depth - denoised_ftm
  1: |Δ|
  2: depth
  3: denoised_ftm  
  4: ftm_std_m
  5: Δ²/σ² (current Gaussian exponent)

Model: 6 → 16 (ReLU) → 8 (ReLU) → 1 (sigmoid)
Loss: BCE with class-balanced weighting
Protocol: LOSO 4-fold
"""
import sys, os, pickle, time, json
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from denoising.inference import FTMDenoiserBatch

# ===========================================================================
# Data loading
# ===========================================================================
DATA_ROOT = Path("/Users/zstar/auto_search/vifi-mot/RAN4model_dfv4p4/seqs/outdoor")
SCENES = ["scene1", "scene2", "scene3", "scene4"]

def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def build_matching_dataset(test_scene: str, model_name: str = "causal_cnn"):
    """Build (features, label) pairs for one LOSO fold.
    
    For each frame, for each valid phone:
      - Positive: (detection_depth, denoised_ftm) where detection belongs to the phone holder
      - Negative: (detection_depth, denoised_ftm) where detection belongs to a DIFFERENT person
    
    Also includes passersby (BBX5_Others) as negatives — they have no phone.
    """
    denoiser = FTMDenoiserBatch(model_name, test_scene, window_size=15)
    
    pos_features = []
    neg_features = []
    
    for scene in SCENES:
        scene_dir = DATA_ROOT / scene
        for seq_dir in sorted(scene_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            sync = seq_dir / "sync_ts16_dfv4p4"
            ftm_p = sync / "FTM_li_sync_dfv4p4.pkl"
            bbx_p = sync / "BBX5_sync_dfv4p4.pkl"
            if not ftm_p.exists() or not bbx_p.exists():
                continue
            
            ftm = load_pkl(str(ftm_p))       # (T, N_legit, 1, 2)
            bbx5 = load_pkl(str(bbx_p))       # (T, N_legit, 1, 5)
            
            # Load others (passersby) if available
            others_p = sync / "BBX5_Others_sync_dfv4p4.pkl"
            others = load_pkl(str(others_p)) if others_p.exists() else None
            
            T = min(ftm.shape[0], bbx5.shape[0])
            N_legit = ftm.shape[1]
            
            # Denoise each phone's FTM
            denoised_ftm = np.full((T, N_legit), np.nan)
            ftm_std_m = np.zeros((T, N_legit))
            valid = np.zeros((T, N_legit), dtype=bool)
            
            for p in range(N_legit):
                ftm_range = ftm[:, p, 0, 0].astype(np.float64)
                ftm_std = ftm[:, p, 0, 1].astype(np.float64)
                ftm_v = ~(np.isnan(ftm_range)) & (np.abs(ftm_range) > 10)
                
                if ftm_v.sum() >= 15:
                    denoised_ftm[:, p] = denoiser.denoise_sequence(
                        ftm_range / 1000.0, ftm_std / 1000.0, ftm_v)
                    ftm_std_m[:, p] = ftm_std / 1000.0
                    valid[:, p] = ftm_v
            
            # Build pairs per frame
            for t in range(T):
                # Collect legitimate user depths
                legit_depths = []
                for s in range(N_legit):
                    d = float(bbx5[t, s, 0, 2])
                    if d > 0.1:
                        legit_depths.append((s, d))
                
                # Collect passerby depths (negatives only)
                passerby_depths = []
                if others is not None and t < others.shape[0]:
                    for o in range(others.shape[1]):
                        d = float(others[t, o, 0, 2])
                        if d > 0.1:
                            passerby_depths.append(d)
                
                # For each valid phone
                for p in range(N_legit):
                    if not valid[t, p]:
                        continue
                    d_ftm = denoised_ftm[t, p]
                    d_std = ftm_std_m[t, p]
                    
                    # Positive pair: phone p with its holder's detection
                    for s, depth in legit_depths:
                        if s == p:
                            feat = _build_features(depth, d_ftm, d_std)
                            pos_features.append(feat)
                        else:
                            # Negative: phone p with another person's detection
                            feat = _build_features(depth, d_ftm, d_std)
                            neg_features.append(feat)
                    
                    # Negative: phone p with passerby detections
                    for depth in passerby_depths:
                        feat = _build_features(depth, d_ftm, d_std)
                        neg_features.append(feat)
    
    # Subsample negatives to balance (target 3:1 ratio)
    n_pos = len(pos_features)
    n_neg_target = min(len(neg_features), n_pos * 3)
    if len(neg_features) > n_neg_target:
        rng = np.random.RandomState(42)
        neg_idx = rng.choice(len(neg_features), n_neg_target, replace=False)
        neg_features = [neg_features[i] for i in neg_idx]
    
    features = np.array(pos_features + neg_features, dtype=np.float32)
    labels = np.array([1.0] * n_pos + [0.0] * len(neg_features), dtype=np.float32)
    
    return features, labels, n_pos, len(neg_features)


def _build_features(depth: float, denoised_ftm: float, ftm_std: float, sigma: float = 1.3):
    """Build 6-dim feature vector."""
    delta = depth - denoised_ftm
    return [
        delta,
        abs(delta),
        depth,
        denoised_ftm,
        ftm_std,
        (delta ** 2) / (2 * sigma ** 2),  # Gaussian exponent
    ]


# ===========================================================================
# Model
# ===========================================================================
class MatchingMLP(nn.Module):
    """Tiny MLP for depth-FTM matching. ~200 params."""
    
    def __init__(self, in_features=6, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)
    
    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ===========================================================================
# Training
# ===========================================================================
def train_fold(test_scene: str, model_name: str = "causal_cnn",
               epochs: int = 30, lr: float = 1e-3):
    """Train one LOSO fold of the matching model."""
    features, labels, n_pos, n_neg = build_matching_dataset(test_scene, model_name)
    
    print(f"  Dataset: {n_pos} pos, {n_neg} neg ({n_pos+n_neg} total)")
    
    # Train/test split: use 80/20 random split (within the fold's data)
    rng = np.random.RandomState(42)
    n = len(features)
    idx = rng.permutation(n)
    n_train = int(0.8 * n)
    train_idx, val_idx = idx[:n_train], idx[n_train:]
    
    train_X = torch.from_numpy(features[train_idx])
    train_y = torch.from_numpy(labels[train_idx])
    val_X = torch.from_numpy(features[val_idx])
    val_y = torch.from_numpy(labels[val_idx])
    
    # Class weights for BCE
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)])
    
    model = MatchingMLP(in_features=6, hidden=16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.BCELoss(weight=torch.where(train_y > 0.5, pos_weight, torch.ones(1)))
    
    # Batch training
    batch_size = 1024
    best_auc = 0
    best_state = None
    
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train)
        total_loss = 0
        for i in range(0, n_train, batch_size):
            batch_idx = perm[i:i+batch_size]
            optimizer.zero_grad()
            preds = model(train_X[batch_idx])
            batch_y = train_y[batch_idx]
            w = torch.where(batch_y > 0.5, pos_weight.item(), 1.0)
            loss = nn.functional.binary_cross_entropy(preds, batch_y, weight=w)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        
        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_preds = model(val_X).numpy()
                val_labels = val_y.numpy()
                # Simple AUC
                auc = _compute_auc(val_preds, val_labels)
                if auc > best_auc:
                    best_auc = auc
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    model.load_state_dict(best_state)
    return model, best_auc


def _compute_auc(preds, labels):
    """Simple AUC computation."""
    try:
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(labels, preds)
    except Exception:
        # Fallback: fast rank-based AUC (Mann-Whitney U, O(N log N))
        pos_preds = preds[labels > 0.5]
        neg_preds = preds[labels < 0.5]
        if len(pos_preds) == 0 or len(neg_preds) == 0:
            return 0.5
        all_preds = np.concatenate([pos_preds, neg_preds])
        all_labels = np.concatenate([np.ones(len(pos_preds)), np.zeros(len(neg_preds))])
        order = np.argsort(all_preds)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(all_preds) + 1, dtype=np.float64)
        pos_rank_sum = ranks[all_labels > 0.5].sum()
        n_pos = len(pos_preds)
        n_neg = len(neg_preds)
        u = pos_rank_sum - n_pos * (n_pos + 1) / 2
        return float(u / (n_pos * n_neg))


def evaluate_model(model, features, labels, gaussian_sigma=1.3):
    """Evaluate learned matching vs Gaussian compat."""
    model.eval()
    with torch.no_grad():
        learned_scores = model(torch.from_numpy(features)).numpy()
    
    # Gaussian compat for comparison
    deltas = features[:, 0]  # Δ = depth - denoised_ftm
    gaussian_scores = np.exp(-(deltas ** 2) / (2 * gaussian_sigma ** 2))
    
    pos_mask = labels > 0.5
    neg_mask = ~pos_mask
    
    result = {
        "learned": {
            "pos_mean": float(learned_scores[pos_mask].mean()),
            "pos_std": float(learned_scores[pos_mask].std()),
            "neg_mean": float(learned_scores[neg_mask].mean()),
            "neg_std": float(learned_scores[neg_mask].std()),
            "gap": float(learned_scores[pos_mask].mean() - learned_scores[neg_mask].mean()),
        },
        "gaussian": {
            "pos_mean": float(gaussian_scores[pos_mask].mean()),
            "pos_std": float(gaussian_scores[pos_mask].std()),
            "neg_mean": float(gaussian_scores[neg_mask].mean()),
            "neg_std": float(gaussian_scores[neg_mask].std()),
            "gap": float(gaussian_scores[pos_mask].mean() - gaussian_scores[neg_mask].mean()),
        },
    }
    
    # AUC comparison
    try:
        from sklearn.metrics import roc_auc_score
        result["learned_auc"] = float(roc_auc_score(labels, learned_scores))
        result["gaussian_auc"] = float(roc_auc_score(labels, gaussian_scores))
    except:
        result["learned_auc"] = float(_compute_auc(learned_scores, labels))
        result["gaussian_auc"] = float(_compute_auc(gaussian_scores, labels))
    
    return result


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("=" * 75)
    print("Phase 6B: Learned Matching (denoised FTM → compat score)")
    print("=" * 75)
    
    all_results = {}
    
    for test_scene in SCENES:
        print(f"\n{'─'*75}")
        print(f"LOSO fold: test = {test_scene}")
        print(f"{'─'*75}")
        
        # Train
        t0 = time.time()
        model, best_auc = train_fold(test_scene)
        elapsed = time.time() - t0
        
        n_params = model.n_params()
        print(f"  Model: {n_params} params, best train AUC = {best_auc:.4f}, [{elapsed:.0f}s]")
        
        # Evaluate on held-out test set (rebuild full dataset and evaluate)
        features, labels, n_pos, n_neg = build_matching_dataset(test_scene)
        result = evaluate_model(model, features, labels)
        result["n_params"] = n_params
        result["test_scene"] = test_scene
        
        all_results[test_scene] = result
        
        print(f"  Learned:  pos={result['learned']['pos_mean']:.3f}±{result['learned']['pos_std']:.3f}, "
              f"neg={result['learned']['neg_mean']:.3f}±{result['learned']['neg_std']:.3f}, "
              f"gap={result['learned']['gap']:.3f}, AUC={result['learned_auc']:.4f}")
        print(f"  Gaussian: pos={result['gaussian']['pos_mean']:.3f}±{result['gaussian']['pos_std']:.3f}, "
              f"neg={result['gaussian']['neg_mean']:.3f}±{result['gaussian']['neg_std']:.3f}, "
              f"gap={result['gaussian']['gap']:.3f}, AUC={result['gaussian_auc']:.4f}")
        
        delta_gap = result['learned']['gap'] - result['gaussian']['gap']
        delta_auc = result['learned_auc'] - result['gaussian_auc']
        print(f"  Δ gap: {delta_gap:+.3f}, Δ AUC: {delta_auc:+.4f}")
        
        # Save checkpoint
        ckpt_dir = Path(__file__).resolve().parent.parent / "checkpoints_matching"
        ckpt_dir.mkdir(exist_ok=True)
        torch.save({
            "model_state": model.state_dict(),
            "test_scene": test_scene,
            "n_params": n_params,
        }, str(ckpt_dir / f"matching_{test_scene}_best.pth"))
    
    # Summary
    print(f"\n{'='*75}")
    print("SUMMARY: Learned vs Gaussian Matching")
    print(f"{'='*75}")
    
    avg_learned_gap = np.mean([r['learned']['gap'] for r in all_results.values()])
    avg_gaussian_gap = np.mean([r['gaussian']['gap'] for r in all_results.values()])
    avg_learned_auc = np.mean([r['learned_auc'] for r in all_results.values()])
    avg_gaussian_auc = np.mean([r['gaussian_auc'] for r in all_results.values()])
    
    print(f"{'Method':>10} | {'avg gap':>8} | {'avg AUC':>8}")
    print(f"{'-'*10} | {'-'*8} | {'-'*8}")
    print(f"{'Gaussian':>10} | {avg_gaussian_gap:>7.3f} | {avg_gaussian_auc:>7.4f}")
    print(f"{'Learned':>10} | {avg_learned_gap:>7.3f} | {avg_learned_auc:>7.4f}")
    print(f"{'Δ':>10} | {avg_learned_gap - avg_gaussian_gap:>+7.3f} | "
          f"{avg_learned_auc - avg_gaussian_auc:>+7.4f}")
    
    # Save
    out_path = Path(__file__).resolve().parent.parent / "exps" / "phase6b_matching.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")
    print("Checkpoints saved to checkpoints_matching/")


if __name__ == "__main__":
    main()
