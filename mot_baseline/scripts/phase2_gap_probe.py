"""Probe tracklet gap structure of Method A.

For each sequence, run Method A, group tracklets, and for consecutive spans
(ordered by start_frame) report:
    - gap distribution
    - fraction of pairs that share the same bound phone

This tells us the empirical structure Direction β has to work with.
"""

import os, sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.vifi_mot import list_sequences, load_sequence
from trackers.wifi_ocsort import WiFiOCSortTracker

METHOD_A_PARAMS = dict(
    wifi_weight=0.10, depth_sigma=1.5, bind_threshold=0.50,
    unbind_threshold=0.15, bind_init_threshold=0.70, ema_alpha=0.85,
)


def run_scene(scene):
    metas = list_sequences(scenes=[scene])
    all_gaps_same = []
    all_gaps_diff = []
    for m in metas:
        frames = load_sequence(m)
        tk = WiFiOCSortTracker(**METHOD_A_PARAMS)
        per_track = defaultdict(list)
        for i, fr in enumerate(frames):
            if not fr.detections:
                continue
            boxes = np.array([d.bbox for d in fr.detections], float)
            depths = np.array([d.depth for d in fr.detections], float)
            scores = np.array([d.score for d in fr.detections], float)
            if fr.ftm is None:
                ftm_m = np.zeros(0, float); ftm_valid = np.zeros(0, bool)
            else:
                ftm_m = fr.ftm[:, 0] / 1000.0
                ftm_valid = ~np.isnan(ftm_m)
            outs = tk.update(boxes, depths, scores, ftm_m, ftm_valid)
            id_bound = {t.track_id: t.bound_phone for t in tk.tracks}
            for tid, _ in outs:
                bp = id_bound.get(tid, None)
                per_track[tid].append((i, int(bp) if bp is not None else -1))
        # Build per-phone spans
        per_phone = defaultdict(list)
        for tid, rec in per_track.items():
            rec.sort(key=lambda x: x[0])
            bp = rec[0][1]
            per_phone[bp].append((rec[0][0], rec[-1][0], tid))
        for bp, spans in per_phone.items():
            if bp < 0:
                continue
            spans.sort(key=lambda x: x[0])
            for i in range(len(spans) - 1):
                s1, e1, _ = spans[i]
                s2, e2, _ = spans[i + 1]
                gap = s2 - e1 - 1
                all_gaps_same.append(gap)
        # Across different phones, also compute gap of consecutive spans
        # (but this is less relevant for tracklet bridging)
    return np.array(all_gaps_same, dtype=int) if all_gaps_same else np.array([])


for s in ["scene1", "scene2", "scene3", "scene4"]:
    g = run_scene(s)
    print(f"\n=== {s}  (same-phone consecutive gaps: {len(g)}) ===")
    if len(g) == 0:
        print("  No consecutive same-phone spans")
        continue
    for thresh in (5, 10, 15, 20, 30, 50, 100):
        pct = (g <= thresh).mean() * 100.0
        print(f"  gap <= {thresh:3d} frames : {pct:5.1f}%  ({int((g<=thresh).sum())}/{len(g)})")
    print(f"  median gap : {int(np.median(g))} frames")
    print(f"  mean   gap : {g.mean():.1f} frames")
