"""Proper IDsw attribution: which GT identities are switching?

We match each tracker tracklet to the GT identity that it overlaps with most,
then for each GT identity count how many times the assigned track_id changes.
We report:
    - per GT identity: is_legitimate (phone holder?) / n_tracks_assigned / n_switches
    - aggregate: how many switches happen on phone-holder GTs vs non-phone GTs
"""

import os, sys
from collections import defaultdict
from typing import List, Tuple
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.vifi_mot import list_sequences, load_sequence
from trackers.wifi_ocsort import WiFiOCSortTracker
from trackers.utils import iou_xyxy

METHOD_A_PARAMS = dict(
    wifi_weight=0.10, depth_sigma=1.5, bind_threshold=0.50,
    unbind_threshold=0.15, bind_init_threshold=0.70, ema_alpha=0.85,
)


def _assign_gt_to_track(track_records, gt_by_frame):
    """For each track_id, pick the GT identity with max IoU overlap across frames."""
    per_track_gt = {}
    for tid, records in track_records.items():
        gt_count = defaultdict(int)
        for f, bbox in records:
            if f not in gt_by_frame:
                continue
            gts = gt_by_frame[f]   # list of (gt_id, gt_bbox)
            if not gts:
                continue
            gids = [g[0] for g in gts]
            gbs = np.stack([g[1] for g in gts])
            ious = iou_xyxy(np.array(bbox, float)[None, :], gbs)[0]
            j = int(ious.argmax())
            if ious[j] > 0.3:
                gt_count[gids[j]] += 1
        if gt_count:
            best = max(gt_count, key=lambda k: gt_count[k])
            per_track_gt[tid] = best
        else:
            per_track_gt[tid] = None
    return per_track_gt


def analyse_scene(scene):
    metas = list_sequences(scenes=[scene])
    agg_phone_sw = 0
    agg_other_sw = 0
    agg_phone_ids = 0
    agg_other_ids = 0
    for m in metas:
        frames = load_sequence(m)
        tk = WiFiOCSortTracker(**METHOD_A_PARAMS)
        per_track = defaultdict(list)   # tid -> [(f, bbox)]
        gt_by_frame = {}
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
            for tid, bb in outs:
                per_track[tid].append((i, bb.copy()))
            gt_by_frame[i] = []
            for d in fr.detections:
                gt_by_frame[i].append((d.gt_track_id, np.array(d.bbox, float)))
            # Mark legitimate: which gt_track_ids are phone holders this frame?
        # Per GT identity, mark whether it's legitimate (phone holder) in any frame
        gt_is_legit = defaultdict(bool)
        for i, fr in enumerate(frames):
            for d in fr.detections:
                if d.is_legitimate:
                    gt_is_legit[d.gt_track_id] = True

        per_track_gt = _assign_gt_to_track(per_track, gt_by_frame)
        # Per GT, count switches
        gt_tracks = defaultdict(list)   # gt_id -> [(frame, tid)]
        for tid, records in per_track.items():
            gt = per_track_gt[tid]
            if gt is None:
                continue
            for f, _ in records:
                gt_tracks[gt].append((f, tid))
        for gt, seq in gt_tracks.items():
            seq.sort(key=lambda x: x[0])
            # Group consecutive frames by tid into segments.
            segments: List[Tuple[int, int, int]] = []  # (start, end, tid)
            seg_start = seq[0][0]
            seg_tid = seq[0][1]
            prev_f = seq[0][0]
            for f, tid in seq[1:]:
                if tid == seg_tid and f - prev_f <= 1:
                    prev_f = f
                    continue
                segments.append((seg_start, prev_f, seg_tid))
                seg_start = f
                seg_tid = tid
                prev_f = f
            segments.append((seg_start, prev_f, seg_tid))
            # Count transitions between consecutive segments of the same GT.
            switches = max(0, len(segments) - 1)
            if gt_is_legit[gt]:
                agg_phone_sw += switches
                agg_phone_ids += 1
            else:
                agg_other_sw += switches
                agg_other_ids += 1
    return agg_phone_ids, agg_phone_sw, agg_other_ids, agg_other_sw


print(f"{'scene':10s} {'phone_ids':>10} {'phone_sw':>10} {'other_ids':>10} {'other_sw':>10} {'other_pct':>10}")
for s in ["scene1", "scene2", "scene3", "scene4"]:
    a, b, c, d = analyse_scene(s)
    tot = b + d
    pct_other = d / tot * 100 if tot else 0
    print(f"{s:10s} {a:>10} {b:>10} {c:>10} {d:>10} {pct_other:>9.1f}%")
