"""Hyperparameter ablation for WiFi-OC-SORT on a single scene.

Sweeps wifi_weight, depth_sigma, bind_threshold and prints a comparison table.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data.vifi_mot import list_sequences, load_sequence
from eval.mot_eval import Evaluator
from trackers import OCSortTracker, WiFiOCSortTracker

from scripts.run_baseline import run_sequence


def evaluate_config(scene: str, tracker_factory) -> dict:
    metas = list_sequences(scenes=[scene])
    evaluator = Evaluator()
    n_frames = 0
    for m in metas:
        frames = load_sequence(m)
        if not frames:
            continue
        tracker = tracker_factory()
        gt, hyp = run_sequence(tracker, frames)
        evaluator.add_sequence(m.name, gt, hyp)
        n_frames += len(frames)
    summary = evaluator.summarize()
    if "OVERALL" not in summary.index:
        return {}
    o = summary.loc["OVERALL"].to_dict()
    return {
        "MOTA": o.get("mota", 0) * 100,
        "IDF1": o.get("idf1", 0) * 100,
        "IDsw": int(o.get("num_switches", 0)),
        "FP":   int(o.get("num_false_positives", 0)),
        "FN":   int(o.get("num_misses", 0)),
        "MT":   int(o.get("mostly_tracked", 0)),
        "ML":   int(o.get("mostly_lost", 0)),
        "n_frames": n_frames,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene4")
    args = ap.parse_args()
    scene = args.scene

    rows: List[Tuple[str, dict]] = []

    print(f"\nRunning OC-SORT baseline on {scene}...")
    rows.append(("OC-SORT (no wifi)",
                 evaluate_config(scene,
                                 lambda: OCSortTracker(max_age=30, min_hits=3, iou_threshold=0.3))))

    grid = [
        # wifi_weight, depth_sigma, bind_threshold
        (0.05, 2.5, 0.5),
        (0.10, 2.5, 0.5),
        (0.15, 2.5, 0.5),
        (0.20, 2.5, 0.5),
        (0.30, 2.5, 0.5),
        (0.10, 1.5, 0.5),
        (0.10, 3.5, 0.5),
        (0.10, 2.5, 0.4),
        (0.10, 2.5, 0.6),
        (0.15, 3.5, 0.5),
        (0.20, 3.5, 0.5),
    ]
    for w, sigma, bt in grid:
        tag = f"WiFi w={w:.2f} σ={sigma:.1f} τ={bt:.2f}"
        print(f"\nRunning {tag} on {scene}...")
        def factory(w=w, sigma=sigma, bt=bt):
            return WiFiOCSortTracker(
                max_age=30, min_hits=3, iou_threshold=0.3,
                wifi_weight=w, depth_sigma=sigma,
                bind_threshold=bt, unbind_threshold=max(0.05, bt - 0.45),
                bind_init_threshold=min(0.95, bt + 0.1),
                ema_alpha=0.85,
            )
        rows.append((tag, evaluate_config(scene, factory)))

    # Pretty print
    print("\n" + "=" * 96)
    print(f"{'Config':<32} {'MOTA':>7} {'IDF1':>7} {'IDsw':>6} {'FP':>6} {'FN':>6} {'MT':>4} {'ML':>4}")
    print("-" * 96)
    base_idf1 = rows[0][1].get("IDF1", 0)
    for tag, r in rows:
        if not r:
            print(f"{tag:<32} (failed)")
            continue
        d = r["IDF1"] - base_idf1
        print(f"{tag:<32} {r['MOTA']:>6.2f}% {r['IDF1']:>6.2f}% "
              f"{r['IDsw']:>6d} {r['FP']:>6d} {r['FN']:>6d} {r['MT']:>4d} {r['ML']:>4d}"
              f"  Δ{d:+.2f}")
    print("=" * 96)


if __name__ == "__main__":
    main()
