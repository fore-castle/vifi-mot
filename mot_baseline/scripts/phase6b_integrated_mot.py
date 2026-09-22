"""Phase 6B Integration: MOT with Learned Matching MLP + Denoised FTM

End-to-end evaluation: WiFiJointTracker with:
  1. Online FTM denoising (CausalDilatedCNN, Phase 6A)
  2. Learned matching MLP replacing Gaussian _compat (Phase 6B)

Runs 4-fold LOSO: for each fold, loads fold-specific denoiser + matching MLP.

Usage:
    python -u scripts/phase6b_integrated_mot.py
    python -u scripts/phase6b_integrated_mot.py --test-scene scene1
"""
import sys, os, time, argparse, json
import numpy as np
from pathlib import Path

# NumPy 2.0 compat for motmetrics
if not hasattr(np, 'asfarray'):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from data.vifi_mot import list_sequences, load_sequence, Frame
from trackers.wifi_joint import WiFiJointTracker
from denoising.inference import FTMDenoiser
from eval.mot_eval import Evaluator

# Import MatchingMLP from phase6b script
from scripts.phase6b_learned_matching import MatchingMLP

SCENES = ["scene1", "scene2", "scene3", "scene4"]
ROOT = Path(__file__).resolve().parent.parent
CKPT_DENOISE = ROOT / "checkpoints_denoising"
CKPT_MATCHING = ROOT / "checkpoints_matching"


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
    """Extract raw FTM: (ftm_m, ftm_std_m, ftm_valid)."""
    if frame.ftm is None:
        return np.zeros((0,)), np.zeros((0,)), np.zeros((0,), dtype=bool)
    n_phones = frame.ftm.shape[0]
    ftm_m = np.full(n_phones, np.nan)
    ftm_std_m = np.zeros(n_phones)
    ftm_valid = np.zeros(n_phones, dtype=bool)
    for p in range(n_phones):
        raw_mm = frame.ftm[p, 0]
        if np.isnan(raw_mm):
            continue
        ftm_m[p] = raw_mm / 1000.0
        if frame.ftm.shape[1] > 1:
            ftm_std_m[p] = frame.ftm[p, 1] / 1000.0
        ftm_valid[p] = True
    return ftm_m, ftm_std_m, ftm_valid


def load_matching_model(test_scene: str) -> MatchingMLP:
    """Load trained matching MLP for the given LOSO fold."""
    ckpt_path = CKPT_MATCHING / f"matching_{test_scene}_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Matching checkpoint not found: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model = MatchingMLP(in_features=6, hidden=16)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def run_fold(test_scene: str, model_name: str, wifi_weight: float,
             window_size: int, depth_sigma: float,
             use_learned_matching: bool):
    """Run MOT for one LOSO fold with denoised FTM + optional learned matching."""
    seqs = list_sequences(scenes=[test_scene])
    seqs = [s for s in seqs if s.kind == "outdoor"]

    # Load denoiser
    denoiser = FTMDenoiser(model_name, test_scene, window_size)

    # Load matching model (if requested)
    matching_model = None
    if use_learned_matching:
        matching_model = load_matching_model(test_scene)

    evaluator = Evaluator()
    total_frames = 0
    total_denoised = 0
    total_fallback = 0

    for meta in seqs:
        frames = load_sequence(meta)
        if not frames:
            continue
        total_frames += len(frames)

        # Fresh tracker per sequence
        tracker = WiFiJointTracker(
            max_age=30, min_hits=3, iou_threshold=0.3,
            delta_t=3, inertia=0.2,
            wifi_weight=wifi_weight, depth_sigma=depth_sigma,
            ema_alpha=0.85, bind_init_threshold=0.7,
            unbind_threshold=0.15,
            switch_penalty=0.30, keep_bonus=0.20,
            assign_threshold=0.40,
            no_ghost_pool=True,
            matching_model=matching_model,
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
                gt_items.append((tid, (x1, y1, x2 - x1, y2 - y1)))
            gt_seq.append((frame.frame_id, gt_items))

            # Detections
            dets, scores = _frame_dets_xyxy_scores(frame)
            depths = _frame_det_depths(frame)

            # Raw FTM
            ftm_m, ftm_std_m, ftm_valid = _frame_raw_ftm(frame)

            # Denoise
            if len(ftm_m) > 0 and ftm_valid.any():
                denoised_ftm = denoiser.denoise(ftm_m, ftm_valid, ftm_std_m)
                for p in range(len(ftm_m)):
                    if ftm_valid[p]:
                        if p in denoiser._buffers and len(denoiser._buffers[p]) >= window_size:
                            total_denoised += 1
                        else:
                            total_fallback += 1
            else:
                denoised_ftm = ftm_m

            # Tracker update with denoised FTM + ftm_std_m (for learned matching)
            outputs = tracker.update(dets, depths, scores,
                                     denoised_ftm, ftm_valid, ftm_std_m)

            hyp_items = []
            for tid, bbox in outputs:
                x1, y1, x2, y2 = bbox
                hyp_items.append((int(tid), (float(x1), float(y1),
                                             float(x2 - x1), float(y2 - y1))))
            hyp_seq.append((frame.frame_id, hyp_items))

        evaluator.add_sequence(meta.name, gt_seq, hyp_seq)

    summary = evaluator.summarize()
    overall = summary.loc["OVERALL"].to_dict() if "OVERALL" in summary.index else {}

    return {
        "test_scene": test_scene,
        "model_name": model_name,
        "matching": "learned" if use_learned_matching else "gaussian",
        "depth_sigma": depth_sigma,
        "wifi_weight": wifi_weight,
        "idf1": float(overall.get("idf1", 0)) * 100,
        "mota": float(overall.get("mota", 0)) * 100,
        "idsw": int(overall.get("num_switches", 0)),
        "fp": int(overall.get("num_false_positives", 0)),
        "fn": int(overall.get("num_misses", 0)),
        "n_frames": total_frames,
        "denoised_phone_frames": total_denoised,
        "fallback_phone_frames": total_fallback,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 6B: MOT with learned matching")
    parser.add_argument("--model", type=str, default="causal_cnn")
    parser.add_argument("--test-scene", type=str, nargs="+", default=SCENES)
    parser.add_argument("--wifi-weight", type=float, default=0.10)
    parser.add_argument("--window-size", type=int, default=15)
    parser.add_argument("--depth-sigma", type=float, default=1.3,
                        help="Sigma for Gaussian (best for denoised FTM)")
    args = parser.parse_args()

    print("=" * 80)
    print("Phase 6B: MOT — Gaussian vs Learned Matching (both with denoised FTM)")
    print(f"Model: {args.model}, σ={args.depth_sigma}, wifi_weight={args.wifi_weight}")
    print("=" * 80)

    all_results = []

    header = f"{'Match':>10} | {'Scene':>8} | {'IDF1':>7} | {'MOTA':>7} | {'IDsw':>6}"
    print(f"\n{header}")
    print("-" * 55)

    for scene in args.test_scene:
        # Run with Gaussian matching (baseline for denoised FTM)
        t0 = time.time()
        res_g = run_fold(scene, args.model, args.wifi_weight,
                         args.window_size, args.depth_sigma,
                         use_learned_matching=False)
        t_g = time.time() - t0
        all_results.append(res_g)
        print(f"{'Gaussian':>10} | {scene:>8} | {res_g['idf1']:>6.2f}% | "
              f"{res_g['mota']:>6.2f}% | {res_g['idsw']:>6d}  [{t_g:.1f}s]")

        # Run with learned matching MLP
        t0 = time.time()
        res_l = run_fold(scene, args.model, args.wifi_weight,
                         args.window_size, args.depth_sigma,
                         use_learned_matching=True)
        t_l = time.time() - t0
        all_results.append(res_l)
        print(f"{'Learned':>10} | {scene:>8} | {res_l['idf1']:>6.2f}% | "
              f"{res_l['mota']:>6.2f}% | {res_l['idsw']:>6d}  [{t_l:.1f}s]")

        delta = res_l['idf1'] - res_g['idf1']
        print(f"{'Δ':>10} | {'':>8} | {delta:>+6.2f}pt |")
        print()

    # Summary
    print("=" * 80)
    print("SUMMARY (4-fold LOSO average)")
    print("=" * 80)

    for match_type in ["gaussian", "learned"]:
        rs = [r for r in all_results if r["matching"] == match_type]
        avg_idf1 = np.mean([r["idf1"] for r in rs])
        avg_mota = np.mean([r["mota"] for r in rs])
        avg_idsw = np.mean([r["idsw"] for r in rs])
        label = "Gaussian" if match_type == "gaussian" else "Learned"
        print(f"  {label:>10}: IDF1={avg_idf1:.2f}%, MOTA={avg_mota:.2f}%, IDsw={avg_idsw:.1f}")

    g_avg = np.mean([r["idf1"] for r in all_results if r["matching"] == "gaussian"])
    l_avg = np.mean([r["idf1"] for r in all_results if r["matching"] == "learned"])
    print(f"\n  Δ Learned vs Gaussian: {l_avg - g_avg:+.2f}pt IDF1")

    # Save
    out_path = ROOT / "exps" / "phase6b_integrated_mot.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
