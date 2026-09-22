"""Phase 0: Vi-Fi Depth Quality & Data Feasibility Analysis.

Three experiments:
  0.1  depth vs FTM range error distribution (per scene, per sequence)
  0.2  AP position estimation via least squares
  0.3  OC-SORT IDsw distribution + occlusion event statistics

Usage:
    /Users/zstar/miniforge3/bin/python3 scripts/phase0_data_validation.py
"""

from __future__ import annotations

import os
import sys
import json
import pickle
import collections
from typing import Dict, List, Tuple, Optional

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.vifi_mot import list_sequences, load_sequence, SequenceMeta

# ---------------------------------------------------------------------------
# Camera intrinsics (ZED2 HD720, from depth_to_dist.py)
# ---------------------------------------------------------------------------
FX = 528.365
FY = 527.925
CX = 638.925
CY = 359.2805


def det_to_3d(cx: float, cy: float, depth: float) -> Tuple[float, float, float]:
    """Project detection (pixel cx, cy, depth_m) to 3D camera frame."""
    X = (cx - CX) / FX * depth
    Y = -(cy - CY) / FY * depth
    Z = depth
    return X, Y, Z


# ---------------------------------------------------------------------------
# Experiment 0.1: Depth vs FTM error distribution
# ---------------------------------------------------------------------------

def experiment_01_depth_vs_ftm(scenes: List[str]) -> dict:
    """Compare detection depth with FTM range for legitimate users."""

    results = {}
    all_errors = []
    all_ftm_stds = []
    all_abs_errors = []

    for scene in scenes:
        seqs = list_sequences(scenes=[scene])
        scene_errors = []
        scene_ftm_stds = []
        scene_abs_errors = []
        seq_stats = {}

        for meta in seqs:
            frames = load_sequence(meta)
            seq_errors = []

            for fr in frames:
                if fr.ftm is None:
                    continue
                for det in fr.detections:
                    if not det.is_legitimate:
                        continue
                    s = det.subj_idx
                    if s >= fr.ftm.shape[0]:
                        continue
                    ftm_range_mm = fr.ftm[s, 0]
                    ftm_std_mm = fr.ftm[s, 1]
                    if np.isnan(ftm_range_mm) or np.isnan(ftm_std_mm):
                        continue
                    if np.isnan(det.depth) or det.depth <= 0:
                        continue

                    ftm_m = ftm_range_mm / 1000.0
                    ftm_std_m = ftm_std_mm / 1000.0
                    error = det.depth - ftm_m  # positive = depth > FTM

                    seq_errors.append(error)
                    scene_errors.append(error)
                    scene_ftm_stds.append(ftm_std_m)
                    scene_abs_errors.append(abs(error))

            if seq_errors:
                arr = np.array(seq_errors)
                seq_stats[meta.seq_id] = {
                    "n_frames": len(arr),
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "median_abs": float(np.median(np.abs(arr))),
                    "max_abs": float(np.max(np.abs(arr))),
                }

        if scene_errors:
            err_arr = np.array(scene_errors)
            std_arr = np.array(scene_ftm_stds)
            abs_arr = np.array(scene_abs_errors)

            # Correlation: FTM std vs |error|
            valid = (std_arr > 0) & (abs_arr > 0)
            if valid.sum() > 10:
                corr = float(np.corrcoef(std_arr[valid], abs_arr[valid])[0, 1])
            else:
                corr = float("nan")

            results[scene] = {
                "n_pairs": len(err_arr),
                "mean_error_m": float(np.mean(err_arr)),
                "std_error_m": float(np.std(err_arr)),
                "median_abs_error_m": float(np.median(abs_arr)),
                "p25_abs": float(np.percentile(abs_arr, 25)),
                "p75_abs": float(np.percentile(abs_arr, 75)),
                "p95_abs": float(np.percentile(abs_arr, 95)),
                "ftm_std_mean_m": float(np.mean(std_arr)),
                "ftm_std_median_m": float(np.median(std_arr)),
                "corr_ftm_std_vs_abs_error": corr,
                "per_sequence": seq_stats,
            }
            all_errors.extend(scene_errors)
            all_ftm_stds.extend(scene_ftm_stds)
            all_abs_errors.extend(scene_abs_errors)

    # Overall stats
    if all_errors:
        err_all = np.array(all_errors)
        abs_all = np.array(all_abs_errors)
        results["OVERALL"] = {
            "n_pairs": len(err_all),
            "mean_error_m": float(np.mean(err_all)),
            "std_error_m": float(np.std(err_all)),
            "median_abs_error_m": float(np.median(abs_all)),
            "p25_abs": float(np.percentile(abs_all, 25)),
            "p75_abs": float(np.percentile(abs_all, 75)),
            "p95_abs": float(np.percentile(abs_all, 95)),
        }

    return results


# ---------------------------------------------------------------------------
# Experiment 0.2: AP position estimation via least squares
# ---------------------------------------------------------------------------

def experiment_02_ap_estimation(scenes: List[str]) -> dict:
    """Estimate AP 3D position using (detection_3d, FTM_range) pairs.

    For each sequence, we know:
    - detection 3D position in camera frame: (X, Y, Z) from depth projection
    - FTM range: distance from AP to phone

    The AP is fixed in the scene, so across all frames:
        ||p_det(t) - p_AP|| ≈ r_ftm(t)

    We solve: minimize Σ (||p_det - p_AP|| - r_ftm)²
    """
    from scipy.optimize import least_squares

    results = {}

    for scene in scenes:
        seqs = list_sequences(scenes=[scene])
        all_points = []   # (X, Y, Z) in camera frame
        all_ranges = []   # FTM range in meters

        for meta in seqs:
            frames = load_sequence(meta)
            for fr in frames:
                if fr.ftm is None:
                    continue
                for det in fr.detections:
                    if not det.is_legitimate:
                        continue
                    s = det.subj_idx
                    if s >= fr.ftm.shape[0]:
                        continue
                    ftm_mm = fr.ftm[s, 0]
                    if np.isnan(ftm_mm) or np.isnan(det.depth) or det.depth <= 0:
                        continue
                    cx, cy = det.cxcywh[0], det.cxcywh[1]
                    X, Y, Z = det_to_3d(cx, cy, det.depth)
                    all_points.append((X, Y, Z))
                    all_ranges.append(ftm_mm / 1000.0)

        if len(all_points) < 20:
            results[scene] = {"error": "too few data points", "n": len(all_points)}
            continue

        pts = np.array(all_points)   # (N, 3)
        rngs = np.array(all_ranges)  # (N,)

        # Subsample for efficiency (max 5000 points)
        if len(pts) > 5000:
            idx = np.random.RandomState(42).choice(len(pts), 5000, replace=False)
            pts_sub = pts[idx]
            rngs_sub = rngs[idx]
        else:
            pts_sub = pts
            rngs_sub = rngs

        def residuals(ap_pos):
            dists = np.sqrt(np.sum((pts_sub - ap_pos) ** 2, axis=1))
            return dists - rngs_sub

        # Initial guess: centroid of detections (rough AP location)
        x0 = np.mean(pts_sub, axis=0)

        try:
            res = least_squares(residuals, x0, method='lm', max_nfev=500)
            ap_est = res.x
            final_residuals = residuals(ap_est)
            rms_residual = float(np.sqrt(np.mean(final_residuals ** 2)))
            median_residual = float(np.median(np.abs(final_residuals)))

            results[scene] = {
                "n_points": len(pts),
                "ap_estimated_camera_frame": {
                    "X": float(ap_est[0]),
                    "Y": float(ap_est[1]),
                    "Z": float(ap_est[2]),
                },
                "rms_residual_m": rms_residual,
                "median_abs_residual_m": median_residual,
                "p25_residual": float(np.percentile(np.abs(final_residuals), 25)),
                "p75_residual": float(np.percentile(np.abs(final_residuals), 75)),
                "optimization_success": bool(res.success),
            }
        except Exception as e:
            results[scene] = {"error": str(e), "n": len(pts)}

    return results


# ---------------------------------------------------------------------------
# Experiment 0.3: IDsw distribution + occlusion analysis
# ---------------------------------------------------------------------------

def experiment_03_idsw_occlusion(scenes: List[str]) -> dict:
    """Run OC-SORT and analyze IDsw distribution + occlusion events."""
    # Import OC-SORT tracker
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from trackers.ocsort import OCSortTracker

    results = {}
    all_idsw_per_seq = {}
    all_occlusion_stats = {}

    for scene in scenes:
        seqs = list_sequences(scenes=[scene])
        scene_idsw = []
        scene_occlusion_events = []
        scene_gap_durations = []
        seq_details = {}

        for meta in seqs:
            frames = load_sequence(meta)
            tracker = OCSortTracker(max_age=30, min_hits=3, iou_threshold=0.3)

            # Run tracker
            import motmetrics as mm
            acc = mm.MOTAccumulator(auto_id=True)

            # Also track detection gaps per GT track
            gt_last_seen: Dict[str, int] = {}
            gt_gaps: Dict[str, List[int]] = collections.defaultdict(list)

            total_idsw = 0

            for fr in frames:
                dets_xyxy = np.array([d.bbox for d in fr.detections], dtype=np.float32)
                scores = np.array([d.score for d in fr.detections], dtype=np.float32)
                gt_ids = [d.gt_track_id for d in fr.detections]

                if len(dets_xyxy) == 0:
                    dets_xyxy = np.empty((0, 4), dtype=np.float32)
                    scores = np.empty((0,), dtype=np.float32)

                tracks = tracker.update(dets_xyxy, scores)

                # Record for motmetrics
                trk_ids = [t.track_id for t in tracks]
                trk_bboxes = np.array([t.last_obs for t in tracks], dtype=np.float32) if tracks else np.empty((0, 4))
                if len(trk_bboxes) == 0:
                    trk_bboxes = np.empty((0, 4), dtype=np.float32)

                acc.update(gt_ids, trk_ids, None)  # distances computed by motmetrics

                # Track occlusion gaps
                for gt_id in gt_ids:
                    if gt_id in gt_last_seen:
                        gap = fr.frame_id - gt_last_seen[gt_id] - 1
                        if gap > 0:
                            gt_gaps[gt_id].append(gap)
                            if gap >= 5:
                                scene_occlusion_events.append({
                                    "seq": meta.seq_id,
                                    "gt_id": gt_id,
                                    "gap": gap,
                                })
                    gt_last_seen[gt_id] = fr.frame_id

            # Get IDsw from motmetrics
            events = acc.events
            if len(events) > 0:
                sw_mask = events.Type == "SWITCH"
                total_idsw = int(sw_mask.sum())
            scene_idsw.append(total_idsw)

            # Occlusion stats for this sequence
            gaps_all = []
            for g_list in gt_gaps.values():
                gaps_all.extend(g_list)
            long_gaps = [g for g in gaps_all if g >= 5]
            scene_gap_durations.extend(long_gaps)

            seq_details[meta.seq_id] = {
                "idsw": total_idsw,
                "n_frames": len(frames),
                "n_gt_tracks": len(gt_last_seen),
                "total_gaps": len(gaps_all),
                "gaps_ge_5": len(long_gaps),
                "max_gap": max(gaps_all) if gaps_all else 0,
            }

        results[scene] = {
            "n_sequences": len(seqs),
            "total_idsw": sum(scene_idsw),
            "mean_idsw_per_seq": float(np.mean(scene_idsw)) if scene_idsw else 0,
            "total_occlusion_events_ge5": len(scene_occlusion_events),
            "gap_duration_stats": {
                "count": len(scene_gap_durations),
                "mean": float(np.mean(scene_gap_durations)) if scene_gap_durations else 0,
                "max": max(scene_gap_durations) if scene_gap_durations else 0,
                "median": float(np.median(scene_gap_durations)) if scene_gap_durations else 0,
            },
            "per_sequence": seq_details,
        }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    scenes = ["scene1", "scene2", "scene3", "scene4"]
    out_dir = os.path.join(os.path.dirname(__file__), "..", "exps")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("Phase 0: Vi-Fi Data Feasibility Analysis")
    print("=" * 70)

    # --- Experiment 0.1 ---
    print("\n>>> Experiment 0.1: Depth vs FTM Error Distribution")
    print("-" * 50)
    r01 = experiment_01_depth_vs_ftm(scenes)

    for scene, stats in r01.items():
        if scene == "OVERALL":
            continue
        print(f"\n  [{scene}]  n_pairs={stats['n_pairs']}")
        print(f"    mean error:  {stats['mean_error_m']:+.3f} m")
        print(f"    std error:   {stats['std_error_m']:.3f} m")
        print(f"    median |err|:{stats['median_abs_error_m']:.3f} m")
        print(f"    p25/p75:     {stats['p25_abs']:.3f} / {stats['p75_abs']:.3f} m")
        print(f"    p95 |err|:   {stats['p95_abs']:.3f} m")
        print(f"    FTM std mean:{stats['ftm_std_mean_m']:.3f} m")
        print(f"    corr(FTM_std, |err|): {stats['corr_ftm_std_vs_abs_error']:.3f}")

    if "OVERALL" in r01:
        ov = r01["OVERALL"]
        print(f"\n  [OVERALL]  n_pairs={ov['n_pairs']}")
        print(f"    mean error:  {ov['mean_error_m']:+.3f} m")
        print(f"    std error:   {ov['std_error_m']:.3f} m")
        print(f"    median |err|:{ov['median_abs_error_m']:.3f} m")
        print(f"    p95 |err|:   {ov['p95_abs']:.3f} m")

    # Verdict
    ov = r01.get("OVERALL", {})
    mean_ok = abs(ov.get("mean_error_m", 99)) < 1.5
    std_ok = ov.get("std_error_m", 99) < 2.5
    print(f"\n  >>> Verdict 0.1: mean={'PASS' if mean_ok else 'FAIL'} (|{ov.get('mean_error_m',0):.2f}| < 1.5), "
          f"std={'PASS' if std_ok else 'FAIL'} ({ov.get('std_error_m',0):.2f} < 2.5)")

    with open(os.path.join(out_dir, "phase0_01_depth_ftm.json"), "w") as f:
        json.dump(r01, f, indent=2, default=str)
    print(f"  Saved to exps/phase0_01_depth_ftm.json")

    # --- Experiment 0.2 ---
    print("\n\n>>> Experiment 0.2: AP Position Estimation")
    print("-" * 50)
    r02 = experiment_02_ap_estimation(scenes)

    for scene, stats in r02.items():
        if "error" in stats:
            print(f"\n  [{scene}] ERROR: {stats['error']}")
            continue
        ap = stats["ap_estimated_camera_frame"]
        print(f"\n  [{scene}]  n_points={stats['n_points']}")
        print(f"    AP est (camera frame): X={ap['X']:.2f}, Y={ap['Y']:.2f}, Z={ap['Z']:.2f} m")
        print(f"    RMS residual:  {stats['rms_residual_m']:.3f} m")
        print(f"    median |resid|:{stats['median_abs_residual_m']:.3f} m")
        print(f"    p25/p75 resid: {stats['p25_residual']:.3f} / {stats['p75_residual']:.3f} m")

    # Verdict
    residuals = [s.get("rms_residual_m", 99) for s in r02.values() if "rms_residual_m" in s]
    ap_ok = all(r < 2.0 for r in residuals) if residuals else False
    print(f"\n  >>> Verdict 0.2: {'PASS' if ap_ok else 'FAIL'} "
          f"(all RMS residuals < 2.0m: {[f'{r:.2f}' for r in residuals]})")

    with open(os.path.join(out_dir, "phase0_02_ap_estimation.json"), "w") as f:
        json.dump(r02, f, indent=2, default=str)
    print(f"  Saved to exps/phase0_02_ap_estimation.json")

    # --- Experiment 0.3 ---
    print("\n\n>>> Experiment 0.3: IDsw Distribution + Occlusion Analysis")
    print("-" * 50)
    r03 = experiment_03_idsw_occlusion(scenes)

    for scene, stats in r03.items():
        print(f"\n  [{scene}]  {stats['n_sequences']} sequences")
        print(f"    total IDsw:         {stats['total_idsw']}")
        print(f"    mean IDsw/seq:      {stats['mean_idsw_per_seq']:.1f}")
        print(f"    occlusion events(≥5):{stats['total_occlusion_events_ge5']}")
        gap_s = stats["gap_durations_stats"]
        print(f"    gap≥5 stats: count={gap_s['count']}, mean={gap_s['mean']:.1f}, "
              f"max={gap_s['max']}, median={gap_s['median']:.1f}")

        # Top 5 worst sequences by IDsw
        per_seq = stats["per_sequence"]
        sorted_seqs = sorted(per_seq.items(), key=lambda x: x[1]["idsw"], reverse=True)
        print(f"    Top 5 IDsw sequences:")
        for seq_id, d in sorted_seqs[:5]:
            print(f"      {seq_id}: IDsw={d['idsw']}, frames={d['n_frames']}, "
                  f"gaps≥5={d['gaps_ge_5']}, max_gap={d['max_gap']}")

    with open(os.path.join(out_dir, "phase0_03_idsw_occlusion.json"), "w") as f:
        json.dump(r03, f, indent=2, default=str)
    print(f"  Saved to exps/phase0_03_idsw_occlusion.json")

    # --- Final Summary ---
    print("\n" + "=" * 70)
    print("Phase 0 Summary")
    print("=" * 70)
    print(f"  0.1 Depth vs FTM:  {'PASS ✅' if (mean_ok and std_ok) else 'FAIL ❌'}")
    print(f"  0.2 AP Estimation: {'PASS ✅' if ap_ok else 'FAIL ❌'}")
    print(f"  0.3 IDsw/Occlusion: (informational, no pass/fail)")
    gate_pass = mean_ok and std_ok and ap_ok
    print(f"\n  Decision Gate 0: {'PASS → proceed to Phase 1' if gate_pass else 'FAIL → need strategy adjustment'}")


if __name__ == "__main__":
    main()
