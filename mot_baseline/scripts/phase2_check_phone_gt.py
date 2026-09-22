"""Check: within a single outdoor sequence, does the same phone index get
carried by different physical people (different GT IDs)?

If yes, then merger can create super-tracks across people, which hurts IDF1.
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


def check_scene(scene):
    metas = list_sequences(scenes=[scene])
    violations = 0
    for m in metas:
        frames = load_sequence(m)
        tk = WiFiOCSortTracker(**METHOD_A_PARAMS)
        phone_to_gts = defaultdict(set)   # phone_idx -> set of gt_ids
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
            # For each bound track, find which GT it's covering
            out_boxes = np.array([bb for _, bb in outs], float)
            if len(out_boxes) == 0:
                continue
            gt_boxes = np.array([np.array(d.bbox, float) for d in fr.detections])
            if len(gt_boxes) == 0:
                continue
            ious = iou_xyxy(out_boxes, gt_boxes)
            gt_ids = [d.gt_track_id for d in fr.detections]
            for k in range(len(outs)):
                tid_out = int(outs[k][0])
                if tid_out < 100000:
                    continue  # unbound
                bp = tid_out - 100000
                j = int(ious[k].argmax())
                if ious[k, j] > 0.3:
                    phone_to_gts[bp].add(gt_ids[j])
        # Check if any phone covers >1 GT
        for bp, gts in phone_to_gts.items():
            if len(gts) > 1:
                violations += 1
                print(f"  seq={m.name} phone={bp} covers GTs={sorted(gts)}")
    return violations


for s in ["scene1", "scene2", "scene3", "scene4"]:
    print(f"\n=== {s} ===")
    v = check_scene(s)
    print(f"  Total multi-GT phones: {v}")
