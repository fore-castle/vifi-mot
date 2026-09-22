"""Proper IDsw attribution: per GT, count distinct tracklets assigned.

For each GT identity, collect the list of unique track_ids that covered it
during the sequence. Each unique track_id is one "tracklet segment" — the
number of such segments minus 1 is the number of IDsw events for that GT.
"""

import os, sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.vifi_mot import list_sequences, load_sequence
from trackers.wifi_ocsort import WiFiOCSortTracker
from trackers.utils import iou_xyxy

METHOD_A_PARAMS = dict(
    wifi_weight=0.10, depth_sigma=1.5, bind_threshold=0.50,
    unbind_threshold=0.15, bind_init_threshold=0.70, ema_alpha=0.85,
)


def analyse_scene(scene):
    metas = list_sequences(scenes=[scene])
    agg = {"phone_sw": 0, "other_sw": 0, "phone_gt": 0, "other_gt": 0,
           "phone_tracklets": 0, "other_tracklets": 0}
    for m in metas:
        frames = load_sequence(m)
        tk = WiFiOCSortTracker(**METHOD_A_PARAMS)
        gt_to_tracks = defaultdict(set)        # gt_id -> set of tids
        gt_is_legit = defaultdict(bool)
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
            # Mark legitimate
            for d in fr.detections:
                if d.is_legitimate:
                    gt_is_legit[d.gt_track_id] = True
            # For each output, find which GT it overlaps most with
            out_boxes = np.array([bb for _, bb in outs], float)
            if len(out_boxes) == 0:
                continue
            gt_boxes = np.array([np.array(d.bbox, float) for d in fr.detections])
            if len(gt_boxes) == 0:
                continue
            ious = iou_xyxy(out_boxes, gt_boxes)  # (nout, ngt)
            gt_ids = [d.gt_track_id for d in fr.detections]
            for k in range(len(outs)):
                j = int(ious[k].argmax())
                if ious[k, j] > 0.3:
                    tid = int(outs[k][0])
                    # normalise: phone-bound IDs have a huge offset
                    if tid >= 100000:
                        # find the underlying track to recover its stable ID
                        tid_norm = None
                        for t in tk.tracks:
                            if t.bound_phone is not None and 100000 + t.bound_phone == tid:
                                tid_norm = t.track_id
                                break
                        if tid_norm is None:
                            tid_norm = tid - 100000
                        tid = tid_norm
                    gt_to_tracks[gt_ids[j]].add(tid)
        for gt, tids in gt_to_tracks.items():
            n_segments = len(tids)
            switches = max(0, n_segments - 1)
            if gt_is_legit[gt]:
                agg["phone_sw"] += switches
                agg["phone_gt"] += 1
                agg["phone_tracklets"] += n_segments
            else:
                agg["other_sw"] += switches
                agg["other_gt"] += 1
                agg["other_tracklets"] += n_segments
    return agg


print(f"{'scene':10s} {'ph_gt':>6} {'ph_tklt':>8} {'ph_sw':>6} {'ot_gt':>6} {'ot_tklt':>8} {'ot_sw':>6} {'ot_pct':>7}")
for s in ["scene1", "scene2", "scene3", "scene4"]:
    a = analyse_scene(s)
    tot = a["phone_sw"] + a["other_sw"]
    pct = a["other_sw"] / tot * 100 if tot else 0
    print(f"{s:10s} {a['phone_gt']:>6} {a['phone_tracklets']:>8} {a['phone_sw']:>6} "
          f"{a['other_gt']:>6} {a['other_tracklets']:>8} {a['other_sw']:>6} {pct:>6.1f}%")
