"""Evaluate LOSO checkpoints with the WiFi-Affinity tracker.

Picks fold{N}_best_*.pth (highest val acc) from checkpoints_loso/full/ and the
matching norm_stats_loso_fold{N}.npz, then runs run_baseline.py on test
scene{N}. Results aggregated into a single JSON for the LOSO-strict report.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = "/Users/zstar/miniforge3/bin/python3"


def find_best(ckpt_dir: Path, fold: int) -> Path | None:
    pattern = re.compile(rf"fold{fold}_best_epoch(\d+)_acc([0-9.]+)\.pth$")
    best = None
    for p in ckpt_dir.glob(f"fold{fold}_best_epoch*_acc*.pth"):
        m = pattern.search(p.name)
        if m:
            acc = float(m.group(2))
            if best is None or acc > best[0]:
                best = (acc, p)
    return best[1] if best else None


def run_one(fold: int, args) -> dict:
    ckpt_dir = ROOT / "checkpoints_loso" / "full"
    ckpt = find_best(ckpt_dir, fold)
    if ckpt is None:
        print(f"[fold{fold}] no checkpoint found at {ckpt_dir}; skipping",
              flush=True)
        return {"fold": fold, "missing": True}
    norm = ROOT / "data" / f"affinity_norm_stats_loso_fold{fold}.npz"
    if not norm.exists():
        print(f"[fold{fold}] missing norm_stats {norm}", flush=True)
        return {"fold": fold, "missing_norm": True}

    test_scene = f"scene{fold}"
    out_path = ROOT / "exps" / f"wifi_affinity_loso_fold{fold}_{test_scene}.json"
    cmd = [
        PY, "scripts/run_baseline.py",
        "--tracker", "wifi_affinity",
        "--test-scene", test_scene,
        "--wifi-weight", str(args.wifi_weight),
        "--bind-threshold", str(args.bind_threshold),
        "--bind-init-threshold", str(args.bind_init_threshold),
        "--unbind-threshold", str(args.unbind_threshold),
        "--ema-alpha", str(args.ema_alpha),
        "--affinity-ckpt", str(ckpt),
        "--affinity-norm-stats", str(norm),
        "--out", str(out_path),
    ]
    print(f"\n[fold{fold}] running on {test_scene} with ckpt={ckpt.name}",
          flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)

    with open(out_path) as f:
        results = json.load(f)
    overall = results[0]["overall"]
    return {"fold": fold, "test_scene": test_scene,
            "ckpt": str(ckpt), "norm_stats": str(norm),
            "overall": overall}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=str, default="1,2,3,4")
    ap.add_argument("--wifi-weight", type=float, default=0.10)
    ap.add_argument("--bind-threshold", type=float, default=0.50)
    ap.add_argument("--bind-init-threshold", type=float, default=0.60)
    ap.add_argument("--unbind-threshold", type=float, default=0.10)
    ap.add_argument("--ema-alpha", type=float, default=0.85)
    args = ap.parse_args()

    folds = [int(x) for x in args.folds.split(",")]
    rows = [run_one(f, args) for f in folds]

    out_dir = ROOT / "exps"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "wifi_affinity_loso_summary.json"
    with open(summary_path, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\nSaved summary → {summary_path}")

    # Pretty-print
    print("\n========= STRICT LOSO 4-fold WiFi-Affinity =========")
    print(f"{'fold':<6}{'scene':<8}{'MOTA':>8}{'IDF1':>8}{'IDsw':>7}{'FP':>7}{'FN':>7}")
    motas, idf1s = [], []
    for r in rows:
        if r.get("missing") or r.get("missing_norm"):
            continue
        o = r["overall"]
        m = o.get("mota", 0)
        i = o.get("idf1", 0)
        if isinstance(m, (int, float)):
            motas.append(m * 100)
        if isinstance(i, (int, float)):
            idf1s.append(i * 100)
        print(f"{r['fold']:<6}{r['test_scene']:<8}"
              f"{m*100:>7.2f}%{i*100:>7.2f}%"
              f"{int(o.get('num_switches',0)):>7d}"
              f"{int(o.get('num_false_positives',0)):>7d}"
              f"{int(o.get('num_misses',0)):>7d}")
    if motas and idf1s:
        print(f"\nAVG  MOTA={sum(motas)/len(motas):.2f}%   "
              f"IDF1={sum(idf1s)/len(idf1s):.2f}%   "
              f"(across {len(motas)} folds)")


if __name__ == "__main__":
    main()
