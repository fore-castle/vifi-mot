"""S5 pilot: MOT with REAL YOLOv8m detections (one sequence, scene4/20211007_134254).

Data alignment (verified):
  - yolov8m_20211007_134254.pkl : dict "YYYY-MM-DD HH_MM_SS.ffffff.png" -> (N,5) [x1,y1,x2,y2,score]
    2014 frames, timestamps = RGB_ts16 list shifted by a constant -43200s (12h timezone), residual 0.
  - sync frame t (BBX5 index, T=1796) -> RGBg_ts16[t] -> detection frame.
    All 1796 sync frames found in the detection dict.

Depth for a detection: taken from the best-IoU GT box (IoU >= --iou-depth).
Rationale: a real ZED pipeline samples the depth map inside the detected box; for a
well-localized detection that value matches the GT box depth. Unmatched (false-positive)
detections get depth = NaN, which the tracker already handles as "no wireless evidence".
This proxy is documented as a limitation.

GT for evaluation is unchanged (BBX5 legit + Others), so FN / FP / localization jitter
from the real detector all show up in the metrics.

Usage:
  python -u s5_real_detection.py                     # all configs
  python -u s5_real_detection.py --configs gt_base yolo_base
"""
from __future__ import annotations

import argparse
import datetime
import json
import pickle
import sys
from pathlib import Path

import numpy as np

if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)

MOT_ROOT = Path("/Users/zstar/auto_search/vifi-mot/mot_baseline")
sys.path.insert(0, str(MOT_ROOT))

from data.vifi_mot import list_sequences, load_sequence          # noqa: E402
from trackers.wifi_joint_v2 import WiFiJointTrackerV2            # noqa: E402
from denoising.inference import FTMDenoiser                      # noqa: E402
from eval.mot_eval import Evaluator                              # noqa: E402
from scripts.phase6b_integrated_mot import (                     # noqa: E402
    _frame_raw_ftm, load_matching_model)
from scripts.v2_experiments import BASE_KW                       # noqa: E402

DATA_ROOT = Path("/Users/zstar/auto_search/vifi-mot/RAN4model_dfv4p4")
SEQ_DIR = DATA_ROOT / "seqs/outdoor/scene4/20211007_134254"
DET_PKL = DATA_ROOT / "yolov8m_20211007_134254.pkl"
TEST_SCENE = "scene4"          # models for this fold never saw scene4
OUT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# detection loading / alignment
# ---------------------------------------------------------------------------

def load_yolo_dets():
    """Return list of (N,5) arrays indexed by sync frame id."""
    raw = pickle.load(open(DET_PKL, "rb"))

    def key_ts(k):
        s = k.replace(".png", "").replace("_", ":")
        return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f").timestamp()

    # map by sub-second fraction (timezone-independent, verified unique & complete)
    by_frac = {round(key_ts(k) % 1, 6): k for k in raw}
    rgbg = json.load(open(SEQ_DIR / "RGBg_ts16_dfv4p4_ls.json"))
    out = []
    for t in rgbg:
        k = by_frac[round(float(t) % 1, 6)]
        out.append(np.asarray(raw[k], dtype=np.float64).reshape(-1, 5))
    return out


def _iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ix = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    iy = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = ix * iy
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / np.maximum(ua, 1e-9)


def det_depths_from_gt(dets, gt_boxes, gt_depths, iou_thr):
    """Assign each detection the depth of its best-IoU GT box (NaN if none)."""
    d = np.full(len(dets), np.nan)
    if len(dets) == 0 or len(gt_boxes) == 0:
        return d
    M = _iou_matrix(dets[:, :4], gt_boxes)
    for i in range(len(dets)):
        j = int(np.argmax(M[i]))
        if M[i, j] >= iou_thr:
            d[i] = gt_depths[j]
    return d


# ---------------------------------------------------------------------------
# configs
# ---------------------------------------------------------------------------

CONFIGS = {
    # visual-only reference (no wireless at all)
    "ocsort":      dict(det="gt",   wifi=False),
    "ocsort_yolo": dict(det="yolo", wifi=False),
    # Phase 6B baseline (causal denoiser + 6-dim learned MLP)
    "gt_base":     dict(det="gt",   learned=True),
    "yolo_base":   dict(det="yolo", learned=True),
    # ours: T15 arbiter + T11 sigma gate + T3b occlusion depth gate
    "gt_ours":     dict(det="gt",   dual=True, wifi_gate=True, occl_depth_gate=True),
    "yolo_ours":   dict(det="yolo", dual=True, wifi_gate=True, occl_depth_gate=True),
}


def run(cfg, yolo_dets, iou_depth=0.3, window_size=15, score_thr=0.0,
        seq_suffix="20211007_134254"):
    meta = [s for s in list_sequences(scenes=[TEST_SCENE])
            if s.name.endswith(seq_suffix)][0]
    frames = load_sequence(meta)

    use_dual = cfg.get("dual", False)
    use_wifi = cfg.get("wifi", True)
    denoiser = denoiser_h = None
    matching_model = None
    if use_wifi:
        denoiser = FTMDenoiser("causal_cnn", TEST_SCENE, window_size)
        if use_dual:
            from denoising.hetero import HeteroFTMDenoiser
            from scripts.phase6b_learned_matching import MatchingMLP
            import torch
            denoiser_h = HeteroFTMDenoiser(TEST_SCENE, window_size)
            ck = torch.load(MOT_ROOT / "checkpoints_matching" /
                            f"matching_dual_{TEST_SCENE}_best.pth",
                            map_location="cpu", weights_only=False)
            matching_model = MatchingMLP(in_features=12, hidden=16)
            matching_model.load_state_dict(ck["model_state"])
            matching_model.eval()
        elif cfg.get("learned"):
            matching_model = load_matching_model(TEST_SCENE)

    kw = dict(BASE_KW)
    if not use_wifi:
        kw["wifi_weight"] = 0.0
    for k in ("wifi_gate", "occl_depth_gate"):
        if k in cfg:
            kw[k] = cfg[k]

    tracker = WiFiJointTrackerV2(matching_model=matching_model, **kw)
    if use_dual:
        tracker.dual_matching = True

    seq_id_map, gt_seq, hyp_seq = {}, [], []
    n_det = n_gt = 0
    for t, frame in enumerate(frames):
        gt_boxes, gt_depths, gt_items = [], [], []
        for det in frame.detections:
            tid = seq_id_map.setdefault(det.gt_track_id, len(seq_id_map) + 1)
            x1, y1, x2, y2 = det.bbox
            gt_items.append((tid, (x1, y1, x2 - x1, y2 - y1)))
            gt_boxes.append([x1, y1, x2, y2])
            gt_depths.append(det.depth)
        gt_seq.append((frame.frame_id, gt_items))
        gt_boxes = np.asarray(gt_boxes).reshape(-1, 4)
        gt_depths = np.asarray(gt_depths)
        n_gt += len(gt_boxes)

        if cfg["det"] == "gt":
            dets = gt_boxes
            scores = np.ones(len(dets))
            depths = gt_depths
        else:
            raw = yolo_dets[t]
            if score_thr > 0 and len(raw):
                raw = raw[raw[:, 4] >= score_thr]
            dets, scores = raw[:, :4], raw[:, 4]
            depths = det_depths_from_gt(raw, gt_boxes, gt_depths, iou_depth)
        n_det += len(dets)

        ftm_m, ftm_std_m, ftm_valid = _frame_raw_ftm(frame)
        std_for_tracker = ftm_std_m
        if use_wifi and len(ftm_m) > 0 and ftm_valid.any():
            denoised = denoiser.denoise(ftm_m, ftm_valid, ftm_std_m)
            if use_dual:
                mu_h, sig = denoiser_h.denoise(ftm_m, ftm_valid, ftm_std_m)
                tracker._mu_h = mu_h
                tracker._sigma_hat = sig
        else:
            denoised = ftm_m

        outputs = tracker.update(dets, depths, scores, denoised,
                                 ftm_valid, std_for_tracker)
        hyp_seq.append((frame.frame_id,
                        [(int(i), (float(b[0]), float(b[1]),
                                   float(b[2] - b[0]), float(b[3] - b[1])))
                         for i, b in outputs]))

    ev = Evaluator()
    ev.add_sequence(meta.name, gt_seq, hyp_seq)
    s = ev.summarize()
    o = s.loc["OVERALL"].to_dict() if "OVERALL" in s.index else {}
    return {"idf1": float(o.get("idf1", 0)) * 100,
            "mota": float(o.get("mota", 0)) * 100,
            "idsw": int(o.get("num_switches", 0)),
            "fp": int(o.get("num_false_positives", 0)),
            "fn": int(o.get("num_misses", 0)),
            "n_det": n_det, "n_gt": n_gt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--iou-depth", type=float, default=0.3)
    ap.add_argument("--score-thr", type=float, default=0.0)
    ap.add_argument("--sweep", action="store_true",
                    help="sweep detection score thresholds on the yolo configs")
    ap.add_argument("--variance", action="store_true",
                    help="per-sequence base-vs-ours spread on GT detections")
    args = ap.parse_args()

    yolo_dets = load_yolo_dets()
    print(f"YOLOv8m dets loaded: {len(yolo_dets)} sync frames, "
          f"{sum(len(d) for d in yolo_dets)} boxes total\n")

    if args.variance:
        # How noisy is a single-sequence comparison? Run base vs ours with GT
        # detections on every scene4 sequence and look at the delta spread.
        seqs = [s.name.split("-")[-1]
                for s in list_sequences(scenes=[TEST_SCENE]) if s.kind == "outdoor"]
        rows, deltas = {}, []
        print(f"{'sequence':>16} | {'base':>7} | {'ours':>7} | {'delta':>7}")
        print("-" * 46)
        for sq in seqs:
            b = run(CONFIGS["gt_base"], yolo_dets, seq_suffix=sq)
            o = run(CONFIGS["gt_ours"], yolo_dets, seq_suffix=sq)
            dl = o["idf1"] - b["idf1"]
            deltas.append(dl)
            rows[sq] = {"base": b["idf1"], "ours": o["idf1"], "delta": dl}
            print(f"{sq:>16} | {b['idf1']:>6.2f}% | {o['idf1']:>6.2f}% | "
                  f"{dl:>+6.2f}", flush=True)
        d = np.array(deltas)
        print("-" * 46)
        print(f"delta: mean={d.mean():+.2f}  std={d.std():.2f}  "
              f"min={d.min():+.2f}  max={d.max():+.2f}  "
              f"positive={int((d > 0).sum())}/{len(d)}")
        (OUT_DIR / "variance.json").write_text(json.dumps(
            {"per_sequence": rows,
             "mean": float(d.mean()), "std": float(d.std()),
             "min": float(d.min()), "max": float(d.max()),
             "n_positive": int((d > 0).sum()), "n": len(d)}, indent=2))
        print(f"saved -> {OUT_DIR/'variance.json'}")
        return

    if args.sweep:
        rows = {}
        print(f"{'thr':>5} | {'config':>12} | {'IDF1':>7} | {'MOTA':>7} | "
              f"{'IDsw':>5} | {'FP':>6} | {'FN':>6}")
        print("-" * 64)
        for thr in [0.0, 0.3, 0.5, 0.7]:
            for name in ["ocsort_yolo", "yolo_base", "yolo_ours"]:
                r = run(CONFIGS[name], yolo_dets, args.iou_depth,
                        score_thr=thr)
                rows[f"{name}@{thr}"] = r
                print(f"{thr:>5.1f} | {name:>12} | {r['idf1']:>6.2f}% | "
                      f"{r['mota']:>6.2f}% | {r['idsw']:>5d} | {r['fp']:>6d} | "
                      f"{r['fn']:>6d}", flush=True)
            print()
        (OUT_DIR / "sweep.json").write_text(json.dumps(rows, indent=2))
        print(f"saved -> {OUT_DIR/'sweep.json'}")
        return

    res = {}
    print(f"{'config':>12} | {'det':>5} | {'IDF1':>7} | {'MOTA':>7} | "
          f"{'IDsw':>5} | {'FP':>6} | {'FN':>6}")
    print("-" * 68)
    for name in args.configs:
        r = run(CONFIGS[name], yolo_dets, args.iou_depth,
                score_thr=args.score_thr)
        res[name] = r
        print(f"{name:>12} | {CONFIGS[name]['det']:>5} | {r['idf1']:>6.2f}% | "
              f"{r['mota']:>6.2f}% | {r['idsw']:>5d} | {r['fp']:>6d} | "
              f"{r['fn']:>6d}", flush=True)

    (OUT_DIR / "results.json").write_text(json.dumps(res, indent=2))
    print(f"\nsaved -> {OUT_DIR/'results.json'}")


if __name__ == "__main__":
    main()
