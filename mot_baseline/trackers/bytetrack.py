"""ByteTrack (Zhang et al., ECCV 2022).

The hallmark is the two-stage association:
1. High-score detections are associated against existing tracks first.
2. Then low-score detections are used to recover lost tracks (the "second
   association"), which is the core idea that gives ByteTrack its name.

ViFi BBX5 GT does not include detection confidence; we derive a pseudo
score from depth: nearer subjects = more confident detections. This is
purely heuristic and exists so that ByteTrack's two-stage logic is exercised.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .utils import KalmanBox, hungarian, iou_xyxy


@dataclass
class _BTTrack:
    kf: KalmanBox
    track_id: int
    state: str = "tracked"       # "tracked" / "lost"
    hits: int = 1
    age: int = 0
    time_since_update: int = 0


class ByteTracker:
    def __init__(self,
                 track_thresh: float = 0.5,
                 match_thresh_high: float = 0.2,    # 1 - IoU threshold
                 match_thresh_low: float = 0.5,
                 new_track_thresh: float = 0.6,
                 max_age: int = 30,
                 min_hits: int = 3,
                 ) -> None:
        self.track_thresh = track_thresh
        self.match_thresh_high = match_thresh_high
        self.match_thresh_low = match_thresh_low
        self.new_track_thresh = new_track_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: List[_BTTrack] = []
        self.frame_count = 0
        self._next_id = 1

    def _predict_all(self) -> List[np.ndarray]:
        return [t.kf.predict() for t in self.tracks]

    def _match(self, dets: np.ndarray, track_idxs: List[int],
               preds: np.ndarray, max_cost: float
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(track_idxs) == 0 or len(dets) == 0:
            return (np.empty((0, 2), dtype=int),
                    np.arange(len(dets), dtype=int),
                    np.arange(len(track_idxs), dtype=int))
        iou = iou_xyxy(dets, preds[track_idxs])
        cost = 1.0 - iou
        return hungarian(cost, max_cost=max_cost)

    def update(self,
               dets_xyxy: np.ndarray,
               scores: np.ndarray,
               ) -> List[Tuple[int, np.ndarray]]:
        self.frame_count += 1
        if len(dets_xyxy) == 0:
            dets_xyxy = np.zeros((0, 4))
            scores = np.zeros((0,))

        # Split detections by score
        high_mask = scores >= self.track_thresh
        low_mask = (scores < self.track_thresh) & (scores > 0.1)
        high_dets = dets_xyxy[high_mask]
        low_dets = dets_xyxy[low_mask]
        high_scores = scores[high_mask]

        # Predict all
        if self.tracks:
            preds = np.stack(self._predict_all(), axis=0)
        else:
            preds = np.zeros((0, 4))

        # First association: high score dets vs all tracks (tracked + lost)
        active_idxs = list(range(len(self.tracks)))
        m1, unm_d1, unm_t1 = self._match(high_dets, active_idxs, preds,
                                         max_cost=self.match_thresh_high)
        matched_pairs: List[Tuple[int, int]] = []
        for d_idx, t_idx in m1:
            matched_pairs.append((d_idx, active_idxs[t_idx]))
        for d_idx, t_idx in matched_pairs:
            self.tracks[t_idx].kf.update(high_dets[d_idx])
            self.tracks[t_idx].hits += 1
            self.tracks[t_idx].time_since_update = 0
            self.tracks[t_idx].state = "tracked"

        # Second association: low score dets vs unmatched tracks (only "tracked"
        # ones - we keep lost tracks for high-score reactivation only)
        remaining_track_idxs = [active_idxs[i] for i in unm_t1
                                if self.tracks[active_idxs[i]].state == "tracked"]
        m2, unm_d2, unm_t2 = self._match(low_dets, remaining_track_idxs, preds,
                                         max_cost=self.match_thresh_low)
        for d_idx, sub_t in m2:
            t_idx = remaining_track_idxs[sub_t]
            self.tracks[t_idx].kf.update(low_dets[d_idx])
            self.tracks[t_idx].hits += 1
            self.tracks[t_idx].time_since_update = 0

        # Unmatched (still tracked) tracks -> lost; unmatched lost -> stay lost
        still_unmatched_track_idxs = set(remaining_track_idxs)
        for sub_t in unm_t2:
            t_idx = remaining_track_idxs[sub_t]
            still_unmatched_track_idxs.add(t_idx)
        for sub_t in unm_t1:
            t_idx = active_idxs[sub_t]
            if self.tracks[t_idx].state == "lost":
                still_unmatched_track_idxs.add(t_idx)

        for t_idx in still_unmatched_track_idxs:
            self.tracks[t_idx].time_since_update += 1
            self.tracks[t_idx].state = "lost"

        # New tracks: only from very high score unmatched detections
        for d_idx in unm_d1:
            if high_scores[d_idx] >= self.new_track_thresh:
                new = _BTTrack(KalmanBox(high_dets[d_idx]), self._next_id)
                self._next_id += 1
                self.tracks.append(new)

        # Aging
        survivors: List[_BTTrack] = []
        for t in self.tracks:
            t.age += 1
            if t.time_since_update <= self.max_age:
                survivors.append(t)
        self.tracks = survivors

        # Emit
        out: List[Tuple[int, np.ndarray]] = []
        for t in self.tracks:
            if t.state != "tracked":
                continue
            if t.hits >= self.min_hits or self.frame_count <= self.min_hits:
                out.append((t.track_id, t.kf.bbox.copy()))
        return out
