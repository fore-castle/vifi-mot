"""Direction β v2: WiFiOCSortTracker wrapper that post-hoc merges
consecutive same-phone tracklets into a single output identity.

Pipeline:
    1. Wrap ``WiFiOCSortTracker``. Each .update() call runs the underlying
       tracker and collects per-frame outputs.
    2. Track per-underlying-tid tracklets (start, end, bound_phone).
    3. When a tracklet terminates, record it.
    4. When a new tracklet starts and binds to the same phone as a recently
       terminated one (gap <= bridging_max_age, FTM continuous), mark a merge:
       the new tracklet's output tid gets relabelled to the old tracklet's
       phone-bound id (100000 + phone).
    5. The output emitted to the evaluator is the *merged* id, so the evaluator
       sees continuous identity across the occlusion gap — no IDsw counted.

Crucially, the tracker's internal association / bind / unbind logic is NOT
modified. Method A runs exactly as before. Only the output tid is rewritten.
"""

from __future__ import annotations

from collections import defaultdict
from typing import List, Optional, Tuple

import numpy as np

from .utils import iou_xyxy
from .wifi_ocsort import WiFiOCSortTracker


class WiFiOCSortTrackletMerger:
    """Wrapper around WiFiOCSortTracker that does post-hoc tracklet merging."""

    def __init__(self,
                 bridging_max_age: int = 40,
                 require_ftm_continuity: bool = True,
                 ftm_std_max: float = 1.5,
                 bbox_proximity_px: float = 300.0,
                 boundary_iou_min: float = 0.10,
                 **method_a_kwargs):
        self._inner = WiFiOCSortTracker(**method_a_kwargs)
        self._bridging_max_age = bridging_max_age
        self._require_ftm_continuity = require_ftm_continuity
        self._ftm_std_max = ftm_std_max
        self._bbox_proximity_px = bbox_proximity_px
        self._boundary_iou_min = boundary_iou_min

        # Per-underlying-tid tracking of active tracklet spans
        # u_tid -> {"start": int, "end": int, "bound_phone": int,
        #           "last_output_tid": int, "last_frame": int}
        self._active: dict = {}
        # Terminated tracklets: list of dicts
        self._terminated: list = []
        # Merge map: u_tid -> target_output_tid (used to rewrite output tids)
        self._merge_map: dict = {}
        # FTM history per phone (ring buffer of recent values)
        self._ftm_history: dict = defaultdict(list)
        # Current frame counter
        self._frame = 0

    # Proxy attributes so run_baseline.py dispatch logic works
    @property
    def tracks(self):
        return self._inner.tracks

    @property
    def phone_id_offset(self):
        return self._inner.phone_id_offset

    def update(self, dets_xyxy, det_depths, scores, ftm_m, ftm_valid):
        self._frame += 1

        # Update FTM history (per phone)
        for p in range(len(ftm_m)):
            if ftm_valid[p]:
                self._ftm_history[p].append((self._frame, float(ftm_m[p])))
                if len(self._ftm_history[p]) > 200:
                    self._ftm_history[p] = self._ftm_history[p][-200:]

        # Run underlying tracker
        outs = self._inner.update(dets_xyxy, det_depths, scores, ftm_m, ftm_valid)

        # Resolve underlying_tid for each output
        resolved: List[Tuple[int, np.ndarray, int, int]] = []  # (tid_out, bbox, bp, u_tid)
        for tid_out, bbox in outs:
            underlying = None
            bp = -1
            for t in self._inner.tracks:
                if t.bound_phone is not None and self._inner.phone_id_offset + t.bound_phone == tid_out:
                    underlying = t.track_id
                    bp = t.bound_phone
                    break
            if underlying is None:
                underlying = tid_out
            resolved.append((tid_out, bbox, bp, underlying))

        # Update per-u_tid active spans
        seen_u = set()
        for tid_out, bbox, bp, u in resolved:
            seen_u.add(u)
            if u not in self._active:
                self._active[u] = {
                    "start": self._frame,
                    "end": self._frame,
                    "bound_phone": bp,
                    "last_output_tid": tid_out,
                    "first_bbox": bbox.copy(),
                    "last_bbox": bbox.copy(),
                    "records": [],
                }
            self._active[u]["end"] = self._frame
            self._active[u]["bound_phone"] = max(self._active[u]["bound_phone"], bp)
            self._active[u]["last_output_tid"] = tid_out
            self._active[u]["last_bbox"] = bbox.copy()
            self._active[u]["records"].append((self._frame, bbox, tid_out))

        # Terminate inactive tracklets
        to_terminate = [u for u in self._active if u not in seen_u]
        for u in to_terminate:
            self._terminated.append(self._active.pop(u))
            self._terminated[-1]["underlying_tid"] = u

        # Detect merges: when a NEW active u_tid binds to a phone that was
        # recently used by a terminated tracklet, and gap <= bridging_max_age,
        # merge: rewrite the new tracklet's output tid to phone_id_offset + phone.
        for u, info in list(self._active.items()):
            bp = info["bound_phone"]
            if bp < 0:
                continue
            if u in self._merge_map:
                continue
            # Find a recently terminated tracklet with same phone
            best_term = None
            best_gap = self._bridging_max_age + 1
            for t in reversed(self._terminated):
                if t.get("bound_phone", -1) != bp:
                    continue
                gap = info["start"] - t["end"] - 1
                if gap < 0:
                    continue
                if gap > self._bridging_max_age:
                    break  # terminated is sorted by end; older ones have even larger gap
                if gap < best_gap:
                    best_gap = gap
                    best_term = t
            if best_term is None:
                continue
            # Spatial gate at the gap boundary: IoU between the terminated
            # tracklet's last bbox and the new tracklet's first bbox. IoU > 0
            # means they occupy similar image regions at the boundary — strong
            # evidence that the same person reappeared. IoU == 0 usually means
            # the phone binding has shifted to a different pedestrian.
            if "last_bbox" in best_term and "first_bbox" in info:
                last = np.array(best_term["last_bbox"], dtype=float).reshape(1, 4)
                first = np.array(info["first_bbox"], dtype=float).reshape(1, 4)
                # Extrapolate the terminated tracklet's bbox forward by gap frames
                # using a rough motion estimate: keep the last bbox.
                iou_val = float(iou_xyxy(last, first)[0, 0])
                if iou_val < self._boundary_iou_min:
                    # Also check pixel center proximity as a looser fallback
                    c1 = np.array([(last[0, 0] + last[0, 2]) / 2,
                                   (last[0, 1] + last[0, 3]) / 2])
                    c2 = np.array([(first[0, 0] + first[0, 2]) / 2,
                                   (first[0, 1] + first[0, 3]) / 2])
                    if float(np.linalg.norm(c1 - c2)) > self._bbox_proximity_px:
                        continue
            # Optional: FTM continuity check
            if self._require_ftm_continuity:
                hist = self._ftm_history.get(bp, [])
                gap_ftms = [v for (f, v) in hist
                            if best_term["end"] < f < info["start"]]
                if gap_ftms and float(np.std(gap_ftms)) > self._ftm_std_max:
                    continue
            # Merge: rewrite this u_tid's output to phone-bound ID
            merged_tid = self._inner.phone_id_offset + bp
            self._merge_map[u] = merged_tid

        # Emit outputs, applying merge rewrites
        final = []
        for tid_out, bbox, bp, u in resolved:
            if u in self._merge_map:
                final.append((self._merge_map[u], bbox))
            else:
                final.append((tid_out, bbox))
        return final
