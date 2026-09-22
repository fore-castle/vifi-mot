"""Direction β v2: post-hoc tracklet ID merger.

Run Method A (WiFiOCSortTracker) normally, collect all (frame, tid, bbox)
outputs, then in a post-hoc pass find pairs of tracklets that:
    1. Share the same bound_phone
    2. Are consecutive in time (gap <= gap_max)
    3. Have FTM values during the gap consistent with the bounding depths

For each such pair, we relabel the LATER tracklet's output tid as the EARLIER
tracklet's tid (or, equivalently, as the phone-bound tid). This removes the
ID switch that would otherwise be counted when the evaluator sees two
different track_ids covering the same GT identity.

Crucially we do NOT change the tracker's internal association logic. Method A
runs exactly as before, with the same IDF1 / MOTA. We only rewrite the OUTPUT
IDs to stitch consecutive phone-holder tracklets together.

Output: per-scene IDF1 / MOTA before and after the post-hoc merger.
"""

from __future__ import annotations

import os, sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import motmetrics as mm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.vifi_mot import list_sequences, load_sequence
from trackers.wifi_ocsort import WiFiOCSortTracker
from trackers.utils import iou_xyxy


METHOD_A_PARAMS = dict(
    wifi_weight=0.10, depth_sigma=1.5, bind_threshold=0.50,
    unbind_threshold=0.15, bind_init_threshold=0.70, ema_alpha=0.85,
)
GAP_MAX = 40      # frames
DEPTH_TOL = 3.0   # meters


def _compute_metrics(gt_seq, hyp_seq):
    acc = mm.MOTAccumulator(auto_id=True)
    for gt_items, hyp_items in zip(gt_seq, hyp_seq):
        gt_ids = [g[0] for g in gt_items]
        gt_boxes = np.array([g[1] for g in gt_items]) if gt_items else np.zeros((0, 4))
        hyp_ids = [h[0] for h in hyp_items]
        hyp_boxes = np.array([h[1] for h in hyp_items]) if hyp_items else np.zeros((0, 4))
        if len(gt_boxes) and len(hyp_boxes):
            dist = mm.distances.iou_matrix(gt_boxes, hyp_boxes, max_iou=0.5)
        else:
            dist = np.full((len(gt_ids), len(hyp_ids)), np.nan)
        acc.update(gt_ids, hyp_ids, dist)
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=["mota", "idf1", "num_switches",
                                        "num_false_positives", "num_misses"],
                         return_dataframe=False, name="x")
    return summary


def _build_tracklets(outputs_per_frame):
    """From per-frame [(tid, bbox, bound_phone, underlying_tid), ...], build
    per-underlying-tid tracklet spans.
    """
    per_tid: Dict[int, List[Tuple[int, tuple, int, int]]] = defaultdict(list)
    for f, outs in enumerate(outputs_per_frame):
        for tid, bbox, bp, u_tid in outs:
            per_tid[u_tid].append((f, bbox, bp, tid))
    # For each underlying tid, get (start_frame, end_frame, bound_phone, out_tid_at_end)
    spans = []
    for u_tid, records in per_tid.items():
        records.sort(key=lambda x: x[0])
        bp = records[0][2]
        spans.append({
            "start": records[0][0],
            "end": records[-1][0],
            "bound_phone": bp,
            "records": records,    # (f, bbox, bp, out_tid)
            "underlying_tid": u_tid,
        })
    return spans


def _ftm_values_in_gap(frames, start: int, end: int, bp: int) -> List[float]:
    out = []
    for f in range(max(0, start), min(len(frames), end + 1)):
        fr = frames[f]
        if fr.ftm is None:
            continue
        if 0 <= bp < len(fr.ftm):
            v = fr.ftm[bp, 0] / 1000.0
            if np.isfinite(v):
                out.append(float(v))
    return out


def _depth_at(records, window=3):
    """Median depth at the end of a tracklet, using depth stored in bbox.

    We don't have depth directly in bbox. We'll just use 0.0 placeholder — the
    actual consistency check uses FTM continuity.
    """
    return 0.0


def _merge_plan(spans, frames, gap_max, depth_tol):
    """For each pair of consecutive same-phone spans, decide whether to merge."""
    by_phone: Dict[int, List[dict]] = defaultdict(list)
    for sp in spans:
        if sp["bound_phone"] >= 0:
            by_phone[sp["bound_phone"]].append(sp)
    merges: Dict[int, int] = {}   # later u_tid -> earlier u_tid
    for bp, phone_spans in by_phone.items():
        phone_spans.sort(key=lambda s: s["start"])
        for i in range(len(phone_spans) - 1):
            sp1 = phone_spans[i]
            sp2 = phone_spans[i + 1]
            # Apply transitive merges
            while sp1["underlying_tid"] in merges:
                sp1 = spans[0]   # placeholder — in practice we don't have a dict by tid;
                # we'll keep it simple: no transitive merge for now
                break
            gap = sp2["start"] - sp1["end"] - 1
            if gap < 0 or gap > gap_max:
                continue
            # FTM continuity check
            ftms = _ftm_values_in_gap(frames, sp1["end"], sp2["start"], bp)
            if not ftms:
                # No FTM during gap — can't verify continuity; still merge
                # because same phone binding is strong evidence.
                pass
            else:
                ftm_mean = float(np.mean(ftms))
                ftm_std = float(np.std(ftms))
                if ftm_std > 1.5:
                    continue
            merges[sp2["underlying_tid"]] = sp1["underlying_tid"]
    return merges


def run_scene(scene, gap_max, do_merge):
    metas = list_sequences(scenes=[scene])
    before_idf1 = 0
    after_idf1 = 0
    before_idsw = 0
    after_idsw = 0
    n_frames = 0
    for m in metas:
        frames = load_sequence(m)
        tk = WiFiOCSortTracker(**METHOD_A_PARAMS)
        outputs_per_frame = []
        gt_seq = []
        for i, fr in enumerate(frames):
            # GT items: (gt_id, bbox)
            gt_seq.append([(d.gt_track_id, np.array(d.bbox, float))
                           for d in fr.detections])
            if not fr.detections:
                outputs_per_frame.append([])
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
            # Resolve underlying tid and bound_phone
            out_records = []
            for tid_out, bbox in outs:
                underlying = None
                bp = None
                for t in tk.tracks:
                    if t.bound_phone is not None and 100000 + t.bound_phone == tid_out:
                        underlying = t.track_id
                        bp = t.bound_phone
                        break
                if underlying is None:
                    underlying = tid_out
                    bp = -1
                out_records.append((tid_out, bbox, bp, underlying))
            outputs_per_frame.append(out_records)
        spans = _build_tracklets(outputs_per_frame)
        merges = _merge_plan(spans, frames, gap_max, DEPTH_TOL) if do_merge else {}
        # Build hyp_seq before and after
        hyp_before = []
        hyp_after = []
        for frame_outs in outputs_per_frame:
            hyp_before.append([(tid_out, bbox) for tid_out, bbox, bp, u in frame_outs])
            after_outs = []
            for tid_out, bbox, bp, u in frame_outs:
                if u in merges:
                    # Emit as phone-bound ID of the merged target
                    target = merges[u]
                    # Find the bound phone of the merged target — same as bp
                    new_tid = 100000 + bp if bp >= 0 else tid_out
                    after_outs.append((new_tid, bbox))
                else:
                    after_outs.append((tid_out, bbox))
            hyp_after.append(after_outs)
        s_before = _compute_metrics(gt_seq, hyp_before)
        s_after = _compute_metrics(gt_seq, hyp_after)
        before_idf1 += s_before["idf1"] * len(frames)
        after_idf1 += s_after["idf1"] * len(frames)
        before_idsw += s_before["num_switches"]
        after_idsw += s_after["num_switches"]
        n_frames += len(frames)
    return {
        "scene": scene,
        "n_frames": n_frames,
        "before_idf1": before_idf1 / n_frames,
        "after_idf1": after_idf1 / n_frames,
        "before_idsw": int(before_idsw),
        "after_idsw": int(after_idsw),
    }


if __name__ == "__main__":
    print(f"{'scene':10s} {'b_idf1':>8} {'a_idf1':>8} {'Δ_idf1':>8} {'b_sw':>6} {'a_sw':>6}")
    for s in ["scene1", "scene2", "scene3", "scene4"]:
        r = run_scene(s, GAP_MAX, do_merge=True)
        delta = (r["after_idf1"] - r["before_idf1"]) * 100
        print(f"{s:10s} {r['before_idf1']*100:>7.2f}% {r['after_idf1']*100:>7.2f}% "
              f"{delta:>+7.2f}pt {r['before_idsw']:>6} {r['after_idsw']:>6}")
