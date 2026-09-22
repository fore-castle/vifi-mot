"""Pre-compute Z-score normalization stats for the learned affinity model.

Re-uses my_vifi's `compute_norm_stats` over fold-1 training sequences and saves
to a single .npz file consumed by trackers/affinity_helper.py.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

MY_VIFI = "/Users/zstar/test/vifi/my_vifi"
DATA_ROOT = "/Users/zstar/auto_search/vifi-mot/RAN4model_dfv4p4"
sys.path.insert(0, MY_VIFI)

from config import Config           # noqa: E402
from dataset import compute_norm_stats   # noqa: E402


def main():
    out_path = Path(__file__).resolve().parent.parent / "data" / "affinity_norm_stats.npz"
    if out_path.exists():
        print(f"already exists: {out_path}")
        return

    config = Config()
    config.data_root = Path(DATA_ROOT)
    config.fold = 1
    stats = compute_norm_stats(config.data_root, config=config)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path,
             cam_mean=stats["cam_mean"],
             cam_std=stats["cam_std"],
             ph_mean=stats["ph_mean"],
             ph_std=stats["ph_std"])
    print(f"saved norm stats to {out_path}")


if __name__ == "__main__":
    main()
