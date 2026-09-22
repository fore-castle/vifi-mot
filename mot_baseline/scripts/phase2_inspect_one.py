"""Inspect one specific phone-holder GT's tracker assignments frame-by-frame.
Reveal whether phone_sw counts are real tracklet breaks or bind/unbind churn.
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


def inspect_sequence(seq_meta):
    frames = load_sequence(seq_meta)
    tk = WiFiOCSortTracker(**METHOD_A_PARAMS)
    gt_to_frames = defaultdict(list)   # gt_id -> [(frame, tid_output, underlying_tid)]
    gt_is_legit = set()
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
        for d in fr.detections:
            if d.is_legitimate:
                gt_is_legit.add(d.gt_track_id)
        out_boxes = np.array([bb for _, bb in outs], float)
        gt_boxes = np.array([np.array(d.bbox, float) for d in fr.detections])
        if len(out_boxes) == 0 or len(gt_boxes) == 0:
            continue
        ious = iou_xyxy(out_boxes, gt_boxes)
        gt_ids = [d.gt_track_id for d in fr.detections]
        for k in range(len(outs)):
            j = int(ious[k].argmax())
            if ious[k, j] > 0.3:
                tid_out = int(outs[k][0])
                # Resolve underlying track id
                underlying = None
                for t in tk.tracks:
                    if t.bound_phone is not None and 100000 + t.bound_phone == tid_out:
                        underlying = t.track_id
                        break
                if underlying is None:
                    underlying = tid_out
                gt_to_frames[gt_ids[j]].append((i, tid_out, underlying))

    for gt in sorted(gt_to_frames):
        if gt not in gt_is_legit:
            continue
        seq = sorted(gt_to_frames[gt], key=lambda x: x[0])
        # Find runs of the same underlying track
        runs = []
        cur_tid = seq[0][2]
        cur_start = seq[0][0]
        cur_end = seq[0][0]
        for f, t_out, t_und in seq[1:]:
            if t_und == cur_tid:
                cur_end = f
            else:
                runs.append((cur_start, cur_end, cur_tid))
                cur_tid = t_und
                cur_start = f
                cur_end = f
        runs.append((cur_start, cur_end, cur_tid))
        switches = max(0, len(runs) - 1)
        if switches >= 3:
            print(f"\n  gt={gt}: {len(seq)} frames, {len(runs)} runs, {switches} switches")
            for r in runs[:20]:
                print(f"    frames {r[0]:4d}..{r[1]:4d}  tid={r[2]}")
            if len(runs) > 20:
                print(f"    ... ({len(runs) - 20} more runs)")
            break  # Just print the first bad case


metas = list_sequences(scenes=["scene1"])
print(f"Inspecting first sequence: {metas[0].name}")
inspect_sequence(metas[0])
