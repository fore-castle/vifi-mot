"""Phase 6 Oracle MOT: 直接在 run_baseline 加 oracle-denoise-alpha 参数
用法: python scripts/phase6_oracle_mot.py --alpha 0.5 --test-scene scene4
"""
import sys, os, time
import numpy as np

# NumPy 2.0 compat for motmetrics
if not hasattr(np, 'asfarray'):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.vifi_mot import list_sequences, load_sequence, Frame
from trackers.wifi_joint import WiFiJointTracker
from typing import List, Tuple
import argparse


def _frame_dets_xyxy_scores(frame: Frame):
    dets = []
    scores = []
    for d in frame.detections:
        dets.append(list(d.bbox))
        scores.append(d.score)
    if not dets:
        return np.empty((0, 4)), np.empty((0,))
    return np.array(dets), np.array(scores)


def _frame_det_depths(frame: Frame):
    return np.array([d.depth for d in frame.detections], dtype=np.float64)


def _frame_wireless_context_oracle(frame: Frame, alpha: float):
    """Oracle denoised FTM: ftm_denoised = alpha * depth + (1-alpha) * raw_ftm
    
    alpha=0: raw FTM (no denoising)
    alpha=1: perfect denoising (FTM = depth of phone holder)
    """
    if frame.ftm is None:
        return np.zeros((0,)), np.zeros((0,), dtype=bool)
    
    n_phones = frame.ftm.shape[0]
    ftm_m = np.full(n_phones, np.nan)
    ftm_valid = np.zeros(n_phones, dtype=bool)
    
    for p in range(n_phones):
        raw_ftm_mm = frame.ftm[p, 0]
        if np.isnan(raw_ftm_mm):
            continue
        raw_ftm_m = raw_ftm_mm / 1000.0
        
        if alpha == 0.0:
            ftm_m[p] = raw_ftm_m
            ftm_valid[p] = True
        else:
            # Find the depth of the person carrying phone p
            phone_depth = None
            for det in frame.detections:
                if det.subj_idx == p and det.depth > 0.1:
                    phone_depth = det.depth
                    break
            
            if phone_depth is not None:
                ftm_m[p] = alpha * phone_depth + (1 - alpha) * raw_ftm_m
                ftm_valid[p] = True
            else:
                # Person not visible, use raw FTM
                ftm_m[p] = raw_ftm_m
                ftm_valid[p] = True
    
    return ftm_m, ftm_valid


def run_oracle_experiment(alpha: float, test_scene: str, wifi_weight: float = 0.10):
    """Run WiFiJointTracker with oracle-denoised FTM."""
    from eval.mot_eval import Evaluator
    
    seqs = list_sequences(scenes=[test_scene])
    # Only outdoor
    seqs = [s for s in seqs if s.kind == "outdoor"]
    
    evaluator = Evaluator()
    total_frames = 0
    
    for meta in seqs:
        frames = load_sequence(meta)
        if not frames:
            continue
        total_frames += len(frames)
        
        tracker = WiFiJointTracker(
            max_age=30, min_hits=3, iou_threshold=0.3,
            delta_t=3, inertia=0.2,
            wifi_weight=wifi_weight, depth_sigma=1.5,
            ema_alpha=0.85, bind_init_threshold=0.7,
            unbind_threshold=0.15,
            switch_penalty=0.30, keep_bonus=0.20,
            assign_threshold=0.40,
            no_ghost_pool=True,
        )
        
        seq_id_map = {}
        gt_seq = []
        hyp_seq = []
        
        for frame in frames:
            # GT
            gt_items = []
            for d in frame.detections:
                tid = seq_id_map.setdefault(d.gt_track_id, len(seq_id_map) + 1)
                x1, y1, x2, y2 = d.bbox
                gt_items.append((tid, (x1, y1, x2-x1, y2-y1)))
            gt_seq.append((frame.frame_id, gt_items))
            
            # Detections
            dets, scores = _frame_dets_xyxy_scores(frame)
            depths = _frame_det_depths(frame)
            
            # Oracle-denoised FTM
            ftm_m, ftm_valid = _frame_wireless_context_oracle(frame, alpha)
            
            # Tracker update
            outputs = tracker.update(dets, depths, scores, ftm_m, ftm_valid)
            
            # Hypothesis
            hyp_items = []
            for tid, bbox in outputs:
                x1, y1, x2, y2 = bbox
                hyp_items.append((int(tid), (float(x1), float(y1),
                                            float(x2-x1), float(y2-y1))))
            hyp_seq.append((frame.frame_id, hyp_items))
        
        evaluator.add_sequence(meta.name, gt_seq, hyp_seq)
    
    summary = evaluator.summarize()
    overall = summary.loc["OVERALL"].to_dict() if "OVERALL" in summary.index else {}
    
    idf1 = overall.get("idf1", 0) * 100
    mota = overall.get("mota", 0) * 100
    idsw = int(overall.get("num_switches", 0))
    fp = int(overall.get("num_false_positives", 0))
    fn = int(overall.get("num_misses", 0))
    
    return {
        "alpha": alpha,
        "test_scene": test_scene,
        "idf1": idf1,
        "mota": mota,
        "idsw": idsw,
        "fp": fp,
        "fn": fn,
        "n_frames": total_frames,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, nargs="+", 
                       default=[0.0, 0.3, 0.5, 0.7, 0.9, 1.0])
    parser.add_argument("--test-scene", type=str, nargs="+",
                       default=["scene1", "scene2", "scene3", "scene4"])
    parser.add_argument("--wifi-weight", type=float, default=0.10)
    args = parser.parse_args()
    
    print("=" * 70)
    print("Phase 6 Oracle MOT: Denoising Level -> IDF1 Gain")
    print(f"WiFi weight: {args.wifi_weight}, Tracker: WiFiJointTracker (global-only)")
    print("=" * 70)
    
    results = []
    
    print(f"\n{'a':>5} | {'Scene':>8} | {'IDF1':>7} | {'MOTA':>7} | {'IDsw':>6} | {'FP':>6} | {'FN':>6}")
    print(f"{'-'*5} | {'-'*8} | {'-'*7} | {'-'*7} | {'-'*6} | {'-'*6} | {'-'*6}")
    
    for scene in args.test_scene:
        for alpha in args.alpha:
            t0 = time.time()
            res = run_oracle_experiment(alpha, scene, args.wifi_weight)
            elapsed = time.time() - t0
            results.append(res)
            print(f"{alpha:>5.2f} | {scene:>8} | {res['idf1']:>6.2f}% | {res['mota']:>6.2f}% | "
                  f"{res['idsw']:>6d} | {res['fp']:>6d} | {res['fn']:>6d}  ({elapsed:.1f}s)")
    
    # Summary: average per alpha
    print(f"\n{'='*70}")
    print("Average IDF1 per alpha:")
    print(f"{'a':>5} | {'avg IDF1':>9} | {'D vs a=0':>9}")
    print(f"{'-'*5} | {'-'*9} | {'-'*9}")
    
    baseline_idf1 = None
    for alpha in args.alpha:
        alpha_results = [r for r in results if r["alpha"] == alpha]
        avg_idf1 = np.mean([r["idf1"] for r in alpha_results])
        if alpha == 0.0:
            baseline_idf1 = avg_idf1
        delta = avg_idf1 - baseline_idf1 if baseline_idf1 is not None else 0
        print(f"{alpha:>5.2f} | {avg_idf1:>8.2f}% | {delta:>+8.2f}pt")
    
    # Save results
    import json
    out_path = Path(__file__).resolve().parent.parent / "exps" / "phase6_oracle_mot.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
