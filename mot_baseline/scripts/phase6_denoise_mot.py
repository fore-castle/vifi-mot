"""Phase 6C: MOT Integration with Learned FTM Denoiser

Runs WiFiJointTracker with online FTM denoising (FTMDenoiser) for each LOSO fold.
Evaluates all 3 model types: causal_cnn, causal_lstm, bilstm.

Usage:
    python scripts/phase6_denoise_mot.py
    python scripts/phase6_denoise_mot.py --model causal_cnn --test-scene scene4
"""
import sys, os, time, argparse, json
import numpy as np

# NumPy 2.0 compat for motmetrics
if not hasattr(np, 'asfarray'):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.vifi_mot import list_sequences, load_sequence, Frame
from trackers.wifi_joint import WiFiJointTracker
from denoising.inference import FTMDenoiser, FTMDenoiserBatch
from eval.mot_eval import Evaluator
from typing import List, Tuple


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


def _frame_raw_ftm(frame: Frame):
    """Extract raw FTM from frame, return (ftm_m, ftm_std_m, ftm_valid)."""
    if frame.ftm is None:
        return np.zeros((0,)), np.zeros((0,)), np.zeros((0,), dtype=bool)
    
    n_phones = frame.ftm.shape[0]
    ftm_m = np.full(n_phones, np.nan)
    ftm_std_m = np.zeros(n_phones)
    ftm_valid = np.zeros(n_phones, dtype=bool)
    
    for p in range(n_phones):
        raw_ftm_mm = frame.ftm[p, 0]
        if np.isnan(raw_ftm_mm):
            continue
        ftm_m[p] = raw_ftm_mm / 1000.0
        # FTM std is in column 1
        if frame.ftm.shape[1] > 1:
            ftm_std_m[p] = frame.ftm[p, 1] / 1000.0
        ftm_valid[p] = True
    
    return ftm_m, ftm_std_m, ftm_valid


def run_denoised_mot(model_name: str, test_scene: str, 
                     wifi_weight: float = 0.10,
                     window_size: int = 15,
                     use_batch: bool = False):
    """Run WiFiJointTracker with learned FTM denoiser.
    
    Args:
        model_name: "causal_cnn", "causal_lstm", or "bilstm"
        test_scene: LOSO test fold (e.g. "scene4")
        wifi_weight: WiFi weight in tracker cost
        window_size: denoiser window size (must match training)
        use_batch: if True, use FTMDenoiserBatch (offline, full-sequence)
    """
    seqs = list_sequences(scenes=[test_scene])
    seqs = [s for s in seqs if s.kind == "outdoor"]
    
    evaluator = Evaluator()
    total_frames = 0
    total_denoised_frames = 0
    total_fallback_frames = 0
    
    for meta in seqs:
        frames = load_sequence(meta)
        if not frames:
            continue
        total_frames += len(frames)
        
        if use_batch:
            # Offline: denoise entire sequence first, then run tracker
            results = _run_batch_denoise_mot(
                model_name, test_scene, meta, frames, 
                wifi_weight, window_size, evaluator)
            total_denoised_frames += results["denoised"]
            total_fallback_frames += results["fallback"]
        else:
            # Online: frame-by-frame denoising within tracker loop
            results = _run_online_denoise_mot(
                model_name, test_scene, meta, frames,
                wifi_weight, window_size, evaluator)
            total_denoised_frames += results["denoised"]
            total_fallback_frames += results["fallback"]
    
    summary = evaluator.summarize()
    overall = summary.loc["OVERALL"].to_dict() if "OVERALL" in summary.index else {}
    
    idf1 = overall.get("idf1", 0) * 100
    mota = overall.get("mota", 0) * 100
    idsw = int(overall.get("num_switches", 0))
    fp = int(overall.get("num_false_positives", 0))
    fn = int(overall.get("num_misses", 0))
    
    return {
        "model_name": model_name,
        "test_scene": test_scene,
        "mode": "batch" if use_batch else "online",
        "idf1": idf1,
        "mota": mota,
        "idsw": idsw,
        "fp": fp,
        "fn": fn,
        "n_frames": total_frames,
        "denoised_phone_frames": total_denoised_frames,
        "fallback_phone_frames": total_fallback_frames,
    }


def _run_online_denoise_mot(model_name, test_scene, meta, frames,
                            wifi_weight, window_size, evaluator):
    """Online mode: FTMDenoiser runs frame-by-frame in the tracker loop."""
    denoiser = FTMDenoiser(model_name, test_scene, window_size)
    
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
    denoised_count = 0
    fallback_count = 0
    
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
        
        # Raw FTM
        ftm_m, ftm_std_m, ftm_valid = _frame_raw_ftm(frame)
        
        if len(ftm_m) > 0 and ftm_valid.any():
            # Denoise
            denoised_ftm = denoiser.denoise(ftm_m, ftm_valid, ftm_std_m)
            # Count how many phones were actually denoised vs fallback
            for p in range(len(ftm_m)):
                if ftm_valid[p]:
                    if p in denoiser._buffers and len(denoiser._buffers[p]) >= window_size:
                        denoised_count += 1
                    else:
                        fallback_count += 1
        else:
            denoised_ftm = ftm_m
        
        # Tracker update with denoised FTM
        outputs = tracker.update(dets, depths, scores, denoised_ftm, ftm_valid)
        
        # Hypothesis
        hyp_items = []
        for tid, bbox in outputs:
            x1, y1, x2, y2 = bbox
            hyp_items.append((int(tid), (float(x1), float(y1),
                                        float(x2-x1), float(y2-y1))))
        hyp_seq.append((frame.frame_id, hyp_items))
    
    evaluator.add_sequence(meta.name, gt_seq, hyp_seq)
    return {"denoised": denoised_count, "fallback": fallback_count}


def _run_batch_denoise_mot(model_name, test_scene, meta, frames,
                           wifi_weight, window_size, evaluator):
    """Batch mode: denoise full sequence offline, then run tracker."""
    batch_denoiser = FTMDenoiserBatch(model_name, test_scene, window_size)
    
    # First pass: collect all FTM per phone
    n_frames_seq = len(frames)
    # Determine n_phones from first frame with FTM
    n_phones = 0
    for f in frames:
        if f.ftm is not None:
            n_phones = f.ftm.shape[0]
            break
    
    if n_phones == 0:
        # No FTM data, just run with raw
        return _run_raw_mot(meta, frames, wifi_weight, evaluator)
    
    # Collect per-phone time series
    ftm_series = np.full((n_frames_seq, n_phones), np.nan)
    std_series = np.zeros((n_frames_seq, n_phones))
    valid_series = np.zeros((n_frames_seq, n_phones), dtype=bool)
    
    for t, frame in enumerate(frames):
        if frame.ftm is None:
            continue
        for p in range(min(n_phones, frame.ftm.shape[0])):
            raw_mm = frame.ftm[p, 0]
            if not np.isnan(raw_mm):
                ftm_series[t, p] = raw_mm / 1000.0
                if frame.ftm.shape[1] > 1:
                    std_series[t, p] = frame.ftm[p, 1] / 1000.0
                valid_series[t, p] = True
    
    # Denoise each phone series
    denoised_series = ftm_series.copy()
    denoised_count = 0
    fallback_count = 0
    
    for p in range(n_phones):
        if valid_series[:, p].sum() < window_size:
            fallback_count += int(valid_series[:, p].sum())
            continue
        denoised_series[:, p] = batch_denoiser.denoise_sequence(
            ftm_series[:, p], std_series[:, p], valid_series[:, p])
        # Count denoised vs fallback frames for this phone
        for t in range(n_frames_seq):
            if valid_series[t, p]:
                if denoised_series[t, p] != ftm_series[t, p]:
                    denoised_count += 1
                else:
                    fallback_count += 1
    
    # Second pass: run tracker with denoised FTM
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
    
    for t, frame in enumerate(frames):
        gt_items = []
        for d in frame.detections:
            tid = seq_id_map.setdefault(d.gt_track_id, len(seq_id_map) + 1)
            x1, y1, x2, y2 = d.bbox
            gt_items.append((tid, (x1, y1, x2-x1, y2-y1)))
        gt_seq.append((frame.frame_id, gt_items))
        
        dets, scores = _frame_dets_xyxy_scores(frame)
        depths = _frame_det_depths(frame)
        
        # Use pre-denoised FTM
        ftm_m = denoised_series[t, :]
        ftm_valid = valid_series[t, :]
        
        outputs = tracker.update(dets, depths, scores, ftm_m, ftm_valid)
        
        hyp_items = []
        for tid, bbox in outputs:
            x1, y1, x2, y2 = bbox
            hyp_items.append((int(tid), (float(x1), float(y1),
                                        float(x2-x1), float(y2-y1))))
        hyp_seq.append((frame.frame_id, hyp_items))
    
    evaluator.add_sequence(meta.name, gt_seq, hyp_seq)
    return {"denoised": denoised_count, "fallback": fallback_count}


def _run_raw_mot(meta, frames, wifi_weight, evaluator):
    """Fallback: run tracker with raw FTM (no denoising)."""
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
        gt_items = []
        for d in frame.detections:
            tid = seq_id_map.setdefault(d.gt_track_id, len(seq_id_map) + 1)
            x1, y1, x2, y2 = d.bbox
            gt_items.append((tid, (x1, y1, x2-x1, y2-y1)))
        gt_seq.append((frame.frame_id, gt_items))
        
        dets, scores = _frame_dets_xyxy_scores(frame)
        depths = _frame_det_depths(frame)
        
        ftm_m, _, ftm_valid = _frame_raw_ftm(frame)
        outputs = tracker.update(dets, depths, scores, ftm_m, ftm_valid)
        
        hyp_items = []
        for tid, bbox in outputs:
            x1, y1, x2, y2 = bbox
            hyp_items.append((int(tid), (float(x1), float(y1),
                                        float(x2-x1), float(y2-y1))))
        hyp_seq.append((frame.frame_id, hyp_items))
    
    evaluator.add_sequence(meta.name, gt_seq, hyp_seq)
    return {"denoised": 0, "fallback": 0}


def main():
    parser = argparse.ArgumentParser(description="Phase 6C: MOT with learned FTM denoiser")
    parser.add_argument("--model", type=str, nargs="+",
                       default=["causal_cnn", "causal_lstm", "bilstm"])
    parser.add_argument("--test-scene", type=str, nargs="+",
                       default=["scene1", "scene2", "scene3", "scene4"])
    parser.add_argument("--wifi-weight", type=float, default=0.10)
    parser.add_argument("--window-size", type=int, default=15)
    parser.add_argument("--batch", action="store_true",
                       help="Use batch (offline) denoising instead of online")
    args = parser.parse_args()
    
    mode = "batch" if args.batch else "online"
    
    print("=" * 75)
    print(f"Phase 6C: MOT with Learned FTM Denoiser ({mode} mode)")
    print(f"WiFi weight: {args.wifi_weight}, Window: {args.window_size}")
    print("=" * 75)
    
    all_results = []
    
    # Also run raw baseline for comparison
    print(f"\n{'Model':>12} | {'Scene':>8} | {'IDF1':>7} | {'MOTA':>7} | {'IDsw':>6} | "
          f"{'Denoised':>9} | {'Fallback':>9} | {'Time':>6}")
    print("-" * 85)
    
    # Raw baseline
    for scene in args.test_scene:
        t0 = time.time()
        seqs = list_sequences(scenes=[scene])
        seqs = [s for s in seqs if s.kind == "outdoor"]
        evaluator = Evaluator()
        for meta in seqs:
            frames = load_sequence(meta)
            if frames:
                _run_raw_mot(meta, frames, args.wifi_weight, evaluator)
        summary = evaluator.summarize()
        overall = summary.loc["OVERALL"].to_dict() if "OVERALL" in summary.index else {}
        elapsed = time.time() - t0
        idf1 = overall.get("idf1", 0) * 100
        mota = overall.get("mota", 0) * 100
        idsw = int(overall.get("num_switches", 0))
        print(f"{'raw':>12} | {scene:>8} | {idf1:>6.2f}% | {mota:>6.2f}% | "
              f"{idsw:>6d} | {'—':>9} | {'—':>9} | {elapsed:>5.1f}s")
        all_results.append({
            "model_name": "raw",
            "test_scene": scene,
            "mode": "none",
            "idf1": idf1,
            "mota": mota,
            "idsw": idsw,
        })
    
    print("-" * 85)
    
    # Denoised models
    for model_name in args.model:
        for scene in args.test_scene:
            t0 = time.time()
            res = run_denoised_mot(model_name, scene, args.wifi_weight,
                                  args.window_size, use_batch=args.batch)
            elapsed = time.time() - t0
            all_results.append(res)
            print(f"{model_name:>12} | {scene:>8} | {res['idf1']:>6.2f}% | "
                  f"{res['mota']:>6.2f}% | {res['idsw']:>6d} | "
                  f"{res['denoised_phone_frames']:>9d} | "
                  f"{res['fallback_phone_frames']:>9d} | {elapsed:>5.1f}s")
    
    # Summary
    print(f"\n{'='*75}")
    print("Average IDF1 per model (4-fold LOSO):")
    print(f"{'Model':>12} | {'avg IDF1':>9} | {'Δ vs raw':>9} | {'avg IDsw':>9}")
    print("-" * 50)
    
    raw_results = [r for r in all_results if r["model_name"] == "raw"]
    raw_avg_idf1 = np.mean([r["idf1"] for r in raw_results])
    print(f"{'raw':>12} | {raw_avg_idf1:>8.2f}% | {'—':>9} | "
          f"{np.mean([r['idsw'] for r in raw_results]):>8.0f}")
    
    for model_name in args.model:
        model_results = [r for r in all_results if r["model_name"] == model_name]
        if model_results:
            avg_idf1 = np.mean([r["idf1"] for r in model_results])
            avg_idsw = np.mean([r["idsw"] for r in model_results])
            delta = avg_idf1 - raw_avg_idf1
            print(f"{model_name:>12} | {avg_idf1:>8.2f}% | {delta:>+8.2f}pt | {avg_idsw:>8.0f}")
    
    # Save
    out_path = Path(__file__).resolve().parent.parent / "exps" / "phase6_denoise_mot.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
