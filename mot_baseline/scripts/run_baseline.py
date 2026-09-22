"""Run a tracker over a set of sequences and report MOT metrics.

Usage:
    python scripts/run_baseline.py --tracker sort --test-scene scene4
    python scripts/run_baseline.py --tracker ocsort --all-folds
    python scripts/run_baseline.py --tracker bytetrack --test-scene scene1 --score-mode depth
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data.vifi_mot import Frame, list_sequences, load_sequence, SequenceMeta
from eval.mot_eval import Evaluator
from trackers import (ByteTracker, OCSortTracker, SortTracker,
                      WiFiOCSortTracker, WiFiAffinityTracker,
                      WiFiSpatialOCSortTracker, WiFiOCSortTrackletMerger,
                      ReliabilityGateTracker, WiFiJointTracker)


TRACKER_REGISTRY = {
    "sort": SortTracker,
    "ocsort": OCSortTracker,
    "bytetrack": ByteTracker,
    "wifi_ocsort": WiFiOCSortTracker,
    "wifi_affinity": WiFiAffinityTracker,
    "wifi_spatial": WiFiSpatialOCSortTracker,
    "wifi_merger": WiFiOCSortTrackletMerger,
    "wifi_reliability": ReliabilityGateTracker,
    "wifi_joint": WiFiJointTracker,
}


def _frame_dets_xyxy_scores(frame: Frame) -> Tuple[np.ndarray, np.ndarray]:
    if not frame.detections:
        return np.zeros((0, 4)), np.zeros((0,))
    boxes = np.array([d.bbox for d in frame.detections], dtype=np.float64)
    scores = np.array([d.score for d in frame.detections], dtype=np.float64)
    return boxes, scores


def _frame_det_depths(frame: Frame) -> np.ndarray:
    if not frame.detections:
        return np.zeros((0,))
    return np.array([d.depth for d in frame.detections], dtype=np.float64)


def _frame_wireless_context(frame: Frame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (ftm_m, ftm_valid, ftm_std_m).

    ftm_m: (N_phone,) FTM range converted to meters, NaN if missing.
    ftm_valid: bool mask combining wireless availability AND legit_valid (i.e.
    only count phones whose owner is currently visible — this matches a more
    realistic deployment where AP only trusts phones near visible users; if you
    want pure wireless validity, drop the legit_valid AND).
    ftm_std_m: (N_phone,) FTM standard deviation converted to meters.
    """
    if frame.ftm is None:
        return np.zeros((0,)), np.zeros((0,), dtype=bool), np.zeros((0,))
    ftm_mm = frame.ftm[:, 0]
    ftm_m = ftm_mm / 1000.0
    ftm_std_m = frame.ftm[:, 1] / 1000.0
    valid = ~np.isnan(ftm_m)
    # We do NOT gate by legit_valid here: a phone can broadcast wireless even
    # if its bbox is currently occluded. Whether the wireless reading itself
    # is NaN is the only filter.
    return ftm_m, valid, ftm_std_m


def _frame_gt_items(frame: Frame, id_map) -> List[Tuple[int, Tuple[float, float, float, float]]]:
    out: List[Tuple[int, Tuple[float, float, float, float]]] = []
    for d in frame.detections:
        tid = id_map.setdefault(d.gt_track_id, len(id_map) + 1)
        x1, y1, x2, y2 = d.bbox
        w = x2 - x1
        h = y2 - y1
        out.append((tid, (x1, y1, w, h)))
    return out


# ---------------------------------------------------------------------------
# Detection noise injection (Phase 4)
# ---------------------------------------------------------------------------

_noise_rng = None   # module-level RNG, initialised by run_sequence


def _init_noise_rng(seed: int) -> None:
    global _noise_rng
    _noise_rng = np.random.default_rng(seed)


def _inject_det_noise(
    dets: np.ndarray,
    depths: np.ndarray,
    scores: np.ndarray,
    *,
    bbox_std: float = 0.0,
    depth_std: float = 0.0,
    drop_rate: float = 0.0,
    fp_count: int = 0,
    img_w: int = 1280,
    img_h: int = 720,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply synthetic detection noise.

    Parameters
    ----------
    dets      : (D, 4) xyxy boxes
    depths    : (D,)   depth in metres
    scores    : (D,)   detection scores
    bbox_std  : std of Gaussian jitter (pixels) applied to each bbox coord
    depth_std : std of Gaussian jitter (metres) added to depth
    drop_rate : probability of dropping each detection
    fp_count  : number of random false-positive boxes to inject per frame

    Returns (dets_noisy, depths_noisy, scores_noisy).
    Ground-truth items used by Evaluator are NOT affected.
    """
    if len(dets) == 0:
        return dets, depths, scores

    rng = _noise_rng if _noise_rng is not None else np.random.default_rng(0)

    # 1. Drop detections
    keep = np.ones(len(dets), dtype=bool)
    if drop_rate > 0:
        keep = rng.random(len(dets)) >= drop_rate
        # Ensure at least one detection survives
        if not keep.any() and len(dets) > 0:
            keep[rng.integers(len(dets))] = True
        dets = dets[keep]
        depths = depths[keep]
        scores = scores[keep]

    if len(dets) == 0:
        return dets, depths, scores

    # 2. Bbox jitter
    if bbox_std > 0:
        jitter = rng.normal(0, bbox_std, size=dets.shape)
        dets = dets + jitter
        # Clamp to image bounds
        dets[:, [0, 2]] = np.clip(dets[:, [0, 2]], 0, img_w)
        dets[:, [1, 3]] = np.clip(dets[:, [1, 3]], 0, img_h)
        # Ensure x2 > x1 and y2 > y1
        dets[:, 2] = np.maximum(dets[:, 2], dets[:, 0] + 1)
        dets[:, 3] = np.maximum(dets[:, 3], dets[:, 1] + 1)

    # 3. Depth noise
    if depth_std > 0:
        depths = depths + rng.normal(0, depth_std, size=depths.shape)
        depths = np.maximum(depths, 0.1)  # depth must be positive

    # 4. Inject false positives
    if fp_count > 0:
        fp_boxes = np.zeros((fp_count, 4), dtype=np.float64)
        for i in range(fp_count):
            cx = rng.uniform(0, img_w)
            cy = rng.uniform(0, img_h)
            w = rng.uniform(30, 150)
            h = rng.uniform(60, 250)
            fp_boxes[i] = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
        fp_boxes[:, [0, 2]] = np.clip(fp_boxes[:, [0, 2]], 0, img_w)
        fp_boxes[:, [1, 3]] = np.clip(fp_boxes[:, [1, 3]], 0, img_h)
        fp_boxes[:, 2] = np.maximum(fp_boxes[:, 2], fp_boxes[:, 0] + 1)
        fp_boxes[:, 3] = np.maximum(fp_boxes[:, 3], fp_boxes[:, 1] + 1)
        fp_depths = rng.uniform(1.0, 15.0, size=fp_count)
        fp_scores = np.full(fp_count, 0.3)  # low-confidence FPs
        dets = np.vstack([dets, fp_boxes])
        depths = np.concatenate([depths, fp_depths])
        scores = np.concatenate([scores, fp_scores])

    return dets, depths, scores


def run_sequence(tracker, frames: List[Frame],
                 noise_bbox_std: float = 0.0,
                 noise_depth_std: float = 0.0,
                 noise_drop_rate: float = 0.0,
                 noise_fp_count: int = 0):
    """Step the tracker over a sequence. Returns lists used by Evaluator.

    gt:  [(frame_id, [(gt_id, (x,y,w,h))])]
    hyp: [(frame_id, [(track_id, (x,y,w,h))])]

    Noise params (Phase 4): applied to detections AFTER GT items extraction.
    """
    id_map = {}
    gt_seq, hyp_seq = [], []
    has_noise = (noise_bbox_std > 0 or noise_depth_std > 0
                 or noise_drop_rate > 0 or noise_fp_count > 0)
    for frame in frames:
        gt_items = _frame_gt_items(frame, id_map)
        gt_seq.append((frame.frame_id, gt_items))

        dets, scores = _frame_dets_xyxy_scores(frame)
        depths = _frame_det_depths(frame)

        # Apply detection noise (Phase 4)
        if has_noise:
            dets, depths, scores = _inject_det_noise(
                dets, depths, scores,
                bbox_std=noise_bbox_std,
                depth_std=noise_depth_std,
                drop_rate=noise_drop_rate,
                fp_count=noise_fp_count,
            )

        if isinstance(tracker, WiFiAffinityTracker):
            outputs = tracker.update(frame)
        elif isinstance(tracker, WiFiSpatialOCSortTracker):
            ftm_m, ftm_valid, ftm_std = _frame_wireless_context(frame)
            outputs = tracker.update(dets, depths, scores, ftm_m, ftm_valid,
                                     ftm_std_m=ftm_std)
        elif isinstance(tracker, (WiFiOCSortTracker, WiFiOCSortTrackletMerger,
                                   ReliabilityGateTracker, WiFiJointTracker)):
            ftm_m, ftm_valid, _ = _frame_wireless_context(frame)
            outputs = tracker.update(dets, depths, scores, ftm_m, ftm_valid)
        elif isinstance(tracker, ByteTracker):
            outputs = tracker.update(dets, scores)
        elif isinstance(tracker, OCSortTracker):
            outputs = tracker.update(dets, scores)
        else:
            outputs = tracker.update(dets)
        hyp_items: List[Tuple[int, Tuple[float, float, float, float]]] = []
        for tid, box in outputs:
            x1, y1, x2, y2 = box
            hyp_items.append((int(tid), (float(x1), float(y1),
                                         float(x2 - x1), float(y2 - y1))))
        hyp_seq.append((frame.frame_id, hyp_items))
    return gt_seq, hyp_seq


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tracker", choices=list(TRACKER_REGISTRY.keys()),
                   default="sort")
    p.add_argument("--test-scene", default="scene4",
                   help="scene used as test set; the rest are unused for tracker training "
                        "(SORT-family is unsupervised, so this only defines the report split).")
    p.add_argument("--all-folds", action="store_true",
                   help="Run scene1, scene2, scene3, scene4 each as test split and "
                        "aggregate. Outputs a per-fold summary.")
    p.add_argument("--score-mode", default="constant", choices=["constant", "depth"])
    p.add_argument("--out", default=None,
                   help="output JSON path; default exps/<tracker>_<scene>.json")
    p.add_argument("--max-age", type=int, default=30)
    p.add_argument("--min-hits", type=int, default=3)
    p.add_argument("--iou-threshold", type=float, default=0.3)
    # WiFi-OC-SORT params
    p.add_argument("--wifi-weight", type=float, default=0.4)
    p.add_argument("--depth-sigma", type=float, default=1.5)
    p.add_argument("--ema-alpha", type=float, default=0.85)
    p.add_argument("--bind-threshold", type=float, default=0.6)
    p.add_argument("--unbind-threshold", type=float, default=0.15)
    p.add_argument("--bind-init-threshold", type=float, default=0.7)
    # WiFi-Spatial extra params
    p.add_argument("--spatial-weight", type=float, default=0.4)
    p.add_argument("--sigma-perp", type=float, default=2.5)
    p.add_argument("--sigma-radial", type=float, default=1.5)
    p.add_argument("--sigma-ap", type=float, default=0.5)
    p.add_argument("--sigma-ftm-floor", type=float, default=0.4)
    p.add_argument("--gate-chi2", type=float, default=9.21)
    p.add_argument("--ap-warmup-frames", type=int, default=50)
    p.add_argument("--ap-warmup-min-pairs", type=int, default=20)
    p.add_argument("--compat-mode", default="scalar", choices=["scalar", "maha"],
                   help="association cost compatibility function: scalar (default, "
                        "Method-A-style 1D gaussian) or maha (Phase 1 Mahalanobis 3D).")
    p.add_argument("--no-ekf", action="store_true",
                   help="disable EKF wireless update (Phase 2). Phase 1 only.")
    p.add_argument("--bridging-max-age", type=int, default=200,
                   help="Direction β: how many frames a bound track survives "
                        "without visual match (FTM keeps it alive). Default 200. "
                        "Set to max_age to disable bridging (Method A behaviour).")
    p.add_argument("--bbox-proximity-px", type=float, default=300.0,
                   help="Direction β v2 (merger): max pixel center distance "
                        "between gap boundary bboxes to allow a merge. Guards "
                        "against super-tracks when the phone binding changes.")
    # Phase 3: Reliability-gated WiFi matching
    p.add_argument("--reliability-gate", type=float, default=0.40,
                   help="Phase 3: WiFi cost skipped when R < this threshold.")
    p.add_argument("--w-consistency", type=float, default=0.40,
                   help="Phase 3: weight for consistency component of R.")
    p.add_argument("--w-stability", type=float, default=0.30,
                   help="Phase 3: weight for stability component of R.")
    p.add_argument("--w-exclusivity", type=float, default=0.30,
                   help="Phase 3: weight for exclusivity component of R.")
    p.add_argument("--ftm-jump-threshold", type=float, default=999.0,
                   help="Phase 3: FTM jump (metres) that triggers force-unbind. "
                        "Default 999 (disabled) — FTM naturally fluctuates 2-5m.")
    p.add_argument("--ftm-jump-window", type=int, default=8,
                   help="Phase 3: rolling window (frames) for FTM jump detection.")
    # Phase 4: Detection noise injection
    p.add_argument("--noise-bbox-std", type=float, default=0.0,
                   help="Phase 4: Gaussian bbox jitter std (pixels). "
                        "Simulates detection localisation error.")
    p.add_argument("--noise-depth-std", type=float, default=0.0,
                   help="Phase 4: Gaussian depth noise std (metres). "
                        "Simulates depth estimation error.")
    p.add_argument("--noise-drop-rate", type=float, default=0.0,
                   help="Phase 4: probability of dropping each detection. "
                        "Simulates occlusion / missed detections.")
    p.add_argument("--noise-fp-count", type=int, default=0,
                   help="Phase 4: number of random false-positive boxes "
                        "injected per frame.")
    p.add_argument("--noise-seed", type=int, default=42,
                   help="Phase 4: random seed for noise RNG.")
    # Phase 5: WiFi Joint (global assignment + ghost pool + FTM depth anchor)
    p.add_argument("--switch-penalty", type=float, default=0.30,
                   help="Phase 5: cost penalty for switching phone assignment.")
    p.add_argument("--keep-bonus", type=float, default=0.20,
                   help="Phase 5: cost bonus for maintaining existing assignment.")
    p.add_argument("--assign-threshold", type=float, default=0.40,
                   help="Phase 5: min score to accept global phone assignment.")
    p.add_argument("--ghost-max-age", type=int, default=60,
                   help="Phase 5: max frames a ghost track survives in pool.")
    p.add_argument("--reconnect-compat-gate", type=float, default=0.30,
                   help="Phase 5: min depth-FTM compat for ghost reconnection.")
    p.add_argument("--reconnect-pixel-gate", type=float, default=200.0,
                   help="Phase 5: max pixel distance for ghost reconnection.")
    p.add_argument("--reconnect-max-gap", type=int, default=60,
                   help="Phase 5: max occlusion gap (frames) for reconnection.")
    p.add_argument("--depth-anchor-sigma", type=float, default=1.5,
                   help="Phase 5: FTM uncertainty (m) for ghost depth anchor.")
    p.add_argument("--no-ghost-pool", action="store_true",
                   help="Phase 5 ablation: disable ghost pool, use global assign only.")
    # WiFi-Affinity model paths
    p.add_argument("--affinity-ckpt", type=str,
                   default="/Users/zstar/test/vifi/my_vifi/checkpoints_strict/full/"
                           "fold1_best_epoch29_acc0.8552.pth")
    p.add_argument("--affinity-norm-stats", type=str,
                   default=os.path.join(ROOT, "data", "affinity_norm_stats.npz"))
    p.add_argument("--affinity-device", type=str, default="cpu")
    return p.parse_args()


def make_tracker(name: str, args):
    common = dict(max_age=args.max_age, min_hits=args.min_hits,
                  iou_threshold=args.iou_threshold)
    if name == "sort":
        return SortTracker(**common)
    if name == "ocsort":
        return OCSortTracker(**common)
    if name == "bytetrack":
        return ByteTracker(track_thresh=0.5, new_track_thresh=0.6,
                           max_age=args.max_age, min_hits=args.min_hits)
    if name == "wifi_ocsort":
        return WiFiOCSortTracker(
            **common,
            wifi_weight=args.wifi_weight,
            depth_sigma=args.depth_sigma,
            ema_alpha=args.ema_alpha,
            bind_threshold=args.bind_threshold,
            unbind_threshold=args.unbind_threshold,
            bind_init_threshold=args.bind_init_threshold,
        )
    if name == "wifi_spatial":
        return WiFiSpatialOCSortTracker(
            **common,
            spatial_weight=args.spatial_weight,
            fallback_sigma=args.depth_sigma,
            ema_alpha=args.ema_alpha,
            bind_threshold=args.bind_threshold,
            unbind_threshold=args.unbind_threshold,
            bind_init_threshold=args.bind_init_threshold,
            sigma_perp=args.sigma_perp,
            sigma_radial=args.sigma_radial,
            sigma_ap=args.sigma_ap,
            sigma_ftm_floor=args.sigma_ftm_floor,
            gate_chi2=args.gate_chi2,
            ap_warmup_frames=args.ap_warmup_frames,
            ap_warmup_min_pairs=args.ap_warmup_min_pairs,
            use_ekf_update=not args.no_ekf,
            bridging_max_age=args.bridging_max_age,
            compat_mode=args.compat_mode,
        )
    if name == "wifi_merger":
        return WiFiOCSortTrackletMerger(
            bridging_max_age=args.bridging_max_age,
            bbox_proximity_px=args.bbox_proximity_px,
            **common,
            wifi_weight=args.wifi_weight,
            depth_sigma=args.depth_sigma,
            ema_alpha=args.ema_alpha,
            bind_threshold=args.bind_threshold,
            unbind_threshold=args.unbind_threshold,
            bind_init_threshold=args.bind_init_threshold,
        )
    if name == "wifi_reliability":
        return ReliabilityGateTracker(
            **common,
            wifi_weight=args.wifi_weight,
            depth_sigma=args.depth_sigma,
            ema_alpha=args.ema_alpha,
            reliability_gate=args.reliability_gate,
            w_consistency=args.w_consistency,
            w_stability=args.w_stability,
            w_exclusivity=args.w_exclusivity,
            ftm_jump_threshold=args.ftm_jump_threshold,
            ftm_jump_window=args.ftm_jump_window,
            bind_init_threshold=args.bind_init_threshold,
        )
    if name == "wifi_affinity":
        return WiFiAffinityTracker(
            **common,
            wifi_weight=args.wifi_weight,
            ema_alpha=args.ema_alpha,
            bind_threshold=args.bind_threshold,
            unbind_threshold=args.unbind_threshold,
            bind_init_threshold=args.bind_init_threshold,
            ckpt_path=args.affinity_ckpt,
            norm_stats_path=args.affinity_norm_stats,
            affinity_device=args.affinity_device,
        )
    if name == "wifi_joint":
        return WiFiJointTracker(
            **common,
            wifi_weight=args.wifi_weight,
            depth_sigma=args.depth_sigma,
            ema_alpha=args.ema_alpha,
            bind_init_threshold=args.bind_init_threshold,
            unbind_threshold=args.unbind_threshold,
            switch_penalty=args.switch_penalty,
            keep_bonus=args.keep_bonus,
            assign_threshold=args.assign_threshold,
            ghost_max_age=args.ghost_max_age,
            reconnect_compat_gate=args.reconnect_compat_gate,
            reconnect_pixel_gate=args.reconnect_pixel_gate,
            reconnect_max_gap=args.reconnect_max_gap,
            depth_anchor_sigma=args.depth_anchor_sigma,
            no_ghost_pool=args.no_ghost_pool,
        )
    raise KeyError(name)


def evaluate_scene(tracker_name: str, test_scene: str,
                   args) -> dict:
    metas = list_sequences(scenes=[test_scene])
    if not metas:
        raise SystemExit(f"No sequences found for scene {test_scene}")
    evaluator = Evaluator()
    t0 = time.time()
    n_frames = 0
    # Init noise RNG (Phase 4) — one seed per evaluation for reproducibility
    _init_noise_rng(getattr(args, 'noise_seed', 42))
    noise_kw = dict(
        noise_bbox_std=getattr(args, 'noise_bbox_std', 0.0),
        noise_depth_std=getattr(args, 'noise_depth_std', 0.0),
        noise_drop_rate=getattr(args, 'noise_drop_rate', 0.0),
        noise_fp_count=getattr(args, 'noise_fp_count', 0),
    )
    for m in metas:
        frames = load_sequence(m, score_mode=args.score_mode)
        if not frames:
            continue
        tracker = make_tracker(tracker_name, args)
        gt_seq, hyp_seq = run_sequence(tracker, frames, **noise_kw)
        evaluator.add_sequence(m.name, gt_seq, hyp_seq)
        n_frames += len(frames)
    summary = evaluator.summarize()
    elapsed = time.time() - t0
    overall = summary.loc["OVERALL"].to_dict() if "OVERALL" in summary.index else {}
    overall["elapsed_sec"] = elapsed
    overall["n_frames"] = n_frames
    overall["n_sequences"] = len(metas)
    print(f"\n=== {tracker_name} on {test_scene} ===")
    print(evaluator.format(summary))
    print(f"\nelapsed {elapsed:.1f}s   sequences={len(metas)}   frames={n_frames}")
    return {"tracker": tracker_name, "test_scene": test_scene,
            "summary": summary.to_dict(),
            "overall": overall}


def main():
    args = build_args()
    if args.all_folds:
        scenes = ["scene1", "scene2", "scene3", "scene4"]
    else:
        scenes = [args.test_scene]
    all_results = []
    for sc in scenes:
        res = evaluate_scene(args.tracker, sc, args)
        all_results.append(res)
    out_path = args.out or os.path.join(
        ROOT, "exps", f"{args.tracker}_{'all' if args.all_folds else args.test_scene}_"
                     f"{args.score_mode}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Concise table
    if args.all_folds:
        print("\n========= Cross-scene summary =========")
        print(f"{'scene':<10} {'MOTA':>7} {'IDF1':>7} {'IDsw':>6} {'FP':>7} {'FN':>7}")
        for r in all_results:
            o = r["overall"]
            mota = o.get("mota", 0) * 100 if isinstance(o.get("mota"), (int, float)) else 0
            idf1 = o.get("idf1", 0) * 100 if isinstance(o.get("idf1"), (int, float)) else 0
            idsw = o.get("num_switches", 0)
            fp = o.get("num_false_positives", 0)
            fn = o.get("num_misses", 0)
            print(f"{r['test_scene']:<10} {mota:>6.2f}% {idf1:>6.2f}% {int(idsw):>6d} {int(fp):>7d} {int(fn):>7d}")


if __name__ == "__main__":
    main()
