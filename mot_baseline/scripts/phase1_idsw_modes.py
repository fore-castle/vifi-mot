"""Method A IDsw failure-mode analysis.

For each outdoor scene, run WiFi-OC-SORT (Method A) with its best tuning, then
inspect the output tracklets. For every IDsw event we classify it as:

1. **occlusion-gap**: a tracklet terminated, and a new tracklet for the same GT
   identity appeared within `gap_window` frames.
2. **close-proximity swap**: two different GT identities are tracked, and their
   assigned IDs flip between frames t and t+1 while their bboxes are within
   `close_px` pixels of each other.
3. **other**: anything else (rare, usually false positive / track churn).

For occlusion-gap IDsws we additionally check: could an FTM measurement from
the same phone have bridged the gap? We compare the FTM range to the detection
depths of both the terminating tracklet and the newly appearing one.

Output: ``exps/method_a_idsw_modes.json`` (per-scene breakdown) and a printed
summary table.
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
from trackers.utils import xyxy_to_usr


GAP_WINDOW = 15          # frames
CLOSE_PX = 120.0         # pixel center distance threshold
IDSW_GAP_BIAS = 80.0     # pixel bbox-center proximity at gap boundaries
METHOD_A_PARAMS = dict(
    wifi_weight=0.10, depth_sigma=1.5, bind_threshold=0.50,
    unbind_threshold=0.15, bind_init_threshold=0.70,
    ema_alpha=0.85,
)


def _bbox_center(xyxy):
    return np.array([(xyxy[0] + xyxy[2]) / 2.0,
                     (xyxy[1] + xyxy[3]) / 2.0], dtype=float)


def analyse_scene(scene_name: str):
    metas = list_sequences(scenes=[scene_name])
    per_seq: List[Dict] = []
    total_idsw = 0
    total_gap = 0
    total_swap = 0
    total_other = 0
    bridgeable = 0
    for m in metas:
        frames = load_sequence(m)
        tk = WiFiOCSortTracker(**METHOD_A_PARAMS)
        # We only care about tracklet-level output; don't need GT assignment
        # at this stage. We'll post-process via the output JSON for IDsw
        # classification — but to be self-contained, we collect tracks here.
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
            # Map track_id -> bound phone for each output
            id_bound = {}
            for t in tk.tracks:
                id_bound[t.track_id] = t.bound_phone
            for tid, bbox in outs:
                records.append((i, int(tid), bbox.copy(),
                                int(id_bound.get(tid, -1))
                                if id_bound.get(tid, -1) is not None else -1))

        # Build per-track time series
        per_track: Dict[int, List[Tuple[int, np.ndarray, int]]] = defaultdict(list)
        for f, tid, bb, bp in records:
            per_track[tid].append((f, bb, bp))

        # Detect tracklet gaps: tracklet ends and a same-bound-phone tracklet
        # starts within GAP_WINDOW frames.
        gap_events = []
        swap_events = []
        # Build list of (start_frame, end_frame, track_id, bound_phone)
        spans = []
        for tid, seq in per_track.items():
            seq = sorted(seq, key=lambda x: x[0])
            spans.append((seq[0][0], seq[-1][0], tid, seq[0][2]))
        spans.sort(key=lambda x: (x[0], x[2]))
        # Gap: for each span ending at e, look for another span starting at
        # s <= e + GAP_WINDOW and s > e, same bound phone.
        for i, (e_s, e_e, e_tid, e_bp) in enumerate(spans):
            if e_bp < 0:
                continue
            for j in range(len(spans)):
                s_s, s_e, s_tid, s_bp = spans[j]
                if s_tid == e_tid:
                    continue
                if s_s <= e_e:
                    continue
                if s_s - e_e > GAP_WINDOW:
                    continue
                if s_bp != e_bp:
                    continue
                # Occlusion gap bridged by wireless
                gap_events.append({
                    "gap": int(s_s - e_e),
                    "end_frame": int(e_e), "end_track": int(e_tid),
                    "start_frame": int(s_s), "start_track": int(s_tid),
                    "bound_phone": int(e_bp),
                })
                break
        total_gap += len(gap_events)
        # Swap: for consecutive frames t and t+1, two tracks swap bbox.
        # Approximate by checking, within each frame, whether two tracklets
        # with very close centers have a sudden ID flip.
        # We use records grouped by frame.
        by_frame: Dict[int, List[Tuple[int, np.ndarray]]] = defaultdict(list)
        for f, tid, bb, bp in records:
            by_frame[f].append((tid, bb))
        frame_ids = sorted(by_frame)
        for k in range(1, len(frame_ids)):
            f_prev, f_curr = frame_ids[k - 1], frame_ids[k]
            if f_curr - f_prev != 1:
                continue
            prev = by_frame[f_prev]
            curr = by_frame[f_curr]
            # Pair closest across frames
            if len(prev) == 0 or len(curr) == 0:
                continue
            P = np.stack([_bbox_center(b) for _, b in prev])
            C = np.stack([_bbox_center(b) for _, b in curr])
            diff = P[:, None, :] - C[None, :, :]   # (nP, nC, 2)
            dist = np.linalg.norm(diff, axis=2)
            # Hungarian-ish: for each prev, pick min; for each curr, pick min
            best_prev = dist.argmin(axis=0)
            best_curr = dist.argmin(axis=1)
            for c in range(len(curr)):
                p = best_prev[c]
                if best_curr[p] != c:
                    continue
                d = dist[p, c]
                if d > CLOSE_PX:
                    continue
                tid_prev = prev[p][0]
                tid_curr = curr[c][0]
                if tid_prev == tid_curr:
                    continue
                # Potential ID swap. Look at next frame too to see if it
                # flipped.
                if k + 1 < len(frame_ids) and frame_ids[k + 1] == f_curr + 1:
                    nxt = by_frame[frame_ids[k + 1]]
                    n_centers = np.stack([_bbox_center(b) for _, b in nxt])
                    nxt_center_curr = _bbox_center(curr[c][1])
                    d_next = np.linalg.norm(
                        n_centers - nxt_center_curr[None, :], axis=1)
                    j = int(d_next.argmin())
                    if d_next[j] <= CLOSE_PX and nxt[j][0] == tid_prev:
                        swap_events.append({
                            "frame": int(f_curr),
                            "track_a": int(tid_prev),
                            "track_b": int(tid_curr),
                        })
        total_swap += len(swap_events)
        # Bridgeability: for each gap, could FTM at (end+1) .. (start-1) have
        # helped? We check if the FTM signal was valid in the gap.
        bridgeable_scene = 0
        for ev in gap_events:
            # Look at raw frames during the gap
            n_ftm = 0
            for fi in range(ev["end_frame"] + 1, ev["start_frame"]):
                if fi >= len(frames):
                    break
                fr = frames[fi]
                if fr.ftm is None:
                    continue
                bp = ev["bound_phone"]
                if bp < 0 or bp >= len(fr.ftm):
                    continue
                if np.isnan(fr.ftm[bp, 0]):
                    continue
                n_ftm += 1
            if n_ftm > 0:
                bridgeable_scene += 1
        bridgeable += bridgeable_scene

        per_seq.append({
            "sequence": m.name,
            "n_tracks": len(per_track),
            "gap_idsw": len(gap_events),
            "swap_idsw": len(swap_events),
            "bridgeable_gap_idsw": bridgeable_scene,
            "gaps": gap_events,
        })
        total_idsw += len(gap_events) + len(swap_events)
    summary = {
        "scene": scene_name,
        "sequences": len(per_seq),
        "total_idsw_estimated": total_idsw,
        "occlusion_gap": total_gap,
        "close_swap": total_swap,
        "bridgeable_gap": bridgeable,
        "per_seq": per_seq,
    }
    return summary


def main():
    scenes = ["scene1", "scene2", "scene3", "scene4"]
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "exps")
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for s in scenes:
        print(f"\n=== {s} ===")
        r = analyse_scene(s)
        results.append(r)
        print(f"  sequences         : {r['sequences']}")
        print(f"  occlusion-gap IDsw: {r['occlusion_gap']}")
        print(f"    bridgeable      : {r['bridgeable_gap']}")
        print(f"  close-swap IDsw   : {r['close_swap']}")
        print(f"  total IDsw est.   : {r['total_idsw_estimated']}")
    with open(os.path.join(out_dir, "method_a_idsw_modes.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_dir}/method_a_idsw_modes.json")


if __name__ == "__main__":
    main()
