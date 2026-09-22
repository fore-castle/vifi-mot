"""Tracklet bridging via wireless FTM continuity (Direction β, post-hoc).

Pipeline:
    1. Run Method A (WiFiOCSortTracker) -> list of (frame, track_id, bbox) per sequence.
    2. Group tracklets by bound_phone.
    3. For each pair of consecutive tracklets sharing the same phone:
       a. compute the gap in frames
       b. during the gap, sample FTM from that phone
       c. if FTM remains valid AND the FTM value is consistent with the depths
          at the ends of the two tracklets, merge them (rename second -> first).
    4. Optionally, look at "swap" events (two tracks crossing) and use the
       wireless binding history to prefer keeping the assignment that keeps
       each track on its bound phone.

Output: bridged tracklets -> evaluate IDF1.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.vifi_mot import list_sequences, load_sequence
from trackers.wifi_ocsort import WiFiOCSortTracker


METHOD_A_PARAMS = dict(
    wifi_weight=0.10, depth_sigma=1.5, bind_threshold=0.50,
    unbind_threshold=0.15, bind_init_threshold=0.70, ema_alpha=0.85,
)
GAP_MAX = 20
DEPTH_TOLERANCE = 3.0    # meters


def _tracklet_depths_near(seq_frames, end_frame: int, window: int = 3) -> List[float]:
    """Depth values (meters) from detections around `end_frame`."""
    ds = []
    for f in range(max(0, end_frame - window), min(len(seq_frames), end_frame + window + 1)):
        for d in seq_frames[f].detections:
            if np.isfinite(d.depth):
                ds.append(float(d.depth))
    return ds


def _ftm_values_in(seq_frames, start: int, end: int, bp: int) -> List[float]:
    """FTM values in meters for phone bp during [start, end]."""
    out: List[float] = []
    for f in range(max(0, start), min(len(seq_frames), end + 1)):
        fr = seq_frames[f]
        if fr.ftm is None:
            continue
        if bp < 0 or bp >= len(fr.ftm):
            continue
        v = fr.ftm[bp, 0] / 1000.0
        if np.isfinite(v):
            out.append(float(v))
    return out


def _merge_tracklets(tracklets: Dict[int, List[Tuple[int, np.ndarray, int]]],
                     seq_frames) -> Dict[int, List[Tuple[int, np.ndarray, int]]]:
    """In-place merge of consecutive tracklets sharing a phone across a short gap.

    Returns the modified mapping {renamed_tid -> [(frame, bbox, bound_phone), ...]}.
    """
    # Build per-phone ordered tracklet spans
    spans_by_phone: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
    for tid, records in tracklets.items():
        records.sort(key=lambda x: x[0])
        bp = records[0][2]
        if bp < 0:
            continue
        spans_by_phone[bp].append((records[0][0], records[-1][0], tid))

    # For each phone, sort spans by start frame and try to merge consecutive pairs
    rename: Dict[int, int] = {}
    for bp, spans in spans_by_phone.items():
        spans.sort(key=lambda x: x[0])
        for i in range(len(spans) - 1):
            s1, e1, tid1 = spans[i]
            s2, e2, tid2 = spans[i + 1]
            # Apply any previous renames
            while tid1 in rename:
                tid1 = rename[tid1]
            gap = s2 - e1 - 1
            if gap < 0 or gap > GAP_MAX:
                continue
            # Collect FTM values during the gap
            ftms = _ftm_values_in(seq_frames, e1, s2, bp)
            if not ftms:
                continue
            ftm_mean = float(np.mean(ftms))
            ftm_std = float(np.std(ftms))
            # Depths at ends
            end_depths = _tracklet_depths_near(seq_frames, e1, window=3)
            start_depths = _tracklet_depths_near(seq_frames, s2, window=3)
            if not end_depths or not start_depths:
                continue
            end_depth = float(np.median(end_depths))
            start_depth = float(np.median(start_depths))
            # Bridge if FTM mean is within tolerance of BOTH end depths
            # and FTM is not super-noisy
            if ftm_std > 1.5:
                continue
            if abs(ftm_mean - end_depth) > DEPTH_TOLERANCE:
                continue
            if abs(ftm_mean - start_depth) > DEPTH_TOLERANCE:
                continue
            # Merge: append tid2's records to tid1 and mark tid2 as renamed
            tracklets[tid1].extend(tracklets[tid2])
            rename[tid2] = tid1
            del tracklets[tid2]
            spans[i + 1] = (s1, e2, tid1)

    return tracklets


def analyse_scene(scene_name: str) -> Dict:
    metas = list_sequences(scenes=[scene_name])
    per_seq = []
    total_merges = 0
    for m in metas:
        frames = load_sequence(m)
        tk = WiFiOCSortTracker(**METHOD_A_PARAMS)
        records = []   # (frame, track_id, bbox, bound_phone)
        for i, fr in enumerate(frames):
            if not fr.detections:
                continue
            boxes = np.array([d.bbox for d in fr.detections], float)
            depths = np.array([d.depth for d in fr.detections], float)
            scores = np.array([d.score for d in fr.detections], float)
            if fr.ftm is None:
                ftm_m = np.zeros(0, float)
                ftm_valid = np.zeros(0, bool)
            else:
                ftm_m = fr.ftm[:, 0] / 1000.0
                ftm_valid = ~np.isnan(ftm_m)
            outs = tk.update(boxes, depths, scores, ftm_m, ftm_valid)
            id_bound = {t.track_id: t.bound_phone for t in tk.tracks}
            for tid, bbox in outs:
                bp = id_bound.get(tid, None)
                records.append((i, int(tid), bbox.copy(),
                                int(bp) if bp is not None else -1))
        # Group by track
        per_track: Dict[int, List[Tuple[int, np.ndarray, int]]] = defaultdict(list)
        for f, tid, bb, bp in records:
            per_track[tid].append((f, bb, bp))
        n_tracks_before = len(per_track)
        per_track = _merge_tracklets(per_track, frames)
        n_tracks_after = len(per_track)
        merges = n_tracks_before - n_tracks_after
        total_merges += merges
        per_seq.append({
            "sequence": m.name,
            "n_tracks_before": n_tracks_before,
            "n_tracks_after": n_tracks_after,
            "merges": merges,
        })
    return {
        "scene": scene_name,
        "sequences": len(per_seq),
        "total_merges": total_merges,
        "per_seq": per_seq,
    }


def main():
    for s in ["scene1", "scene2", "scene3", "scene4"]:
        print(f"\n=== {s} ===")
        r = analyse_scene(s)
        print(f"  sequences : {r['sequences']}")
        print(f"  merges    : {r['total_merges']}")


if __name__ == "__main__":
    main()
