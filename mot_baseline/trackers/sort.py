"""SORT tracker (Bewley et al., 2016).

Reference implementation distilled to ~150 LoC. No external ML libs needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from .utils import KalmanBox, hungarian, iou_xyxy


@dataclass
class _Track:
    kf: KalmanBox
    track_id: int
    age: int = 0
    hits: int = 1
    time_since_update: int = 0
    history: List[np.ndarray] = field(default_factory=list)


class SortTracker:
    def __init__(self,
                 max_age: int = 30,
                 min_hits: int = 3,
                 iou_threshold: float = 0.3,
                 ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: List[_Track] = []
        self.frame_count = 0
        self._next_id = 1

    def update(self, dets_xyxy: np.ndarray) -> List[Tuple[int, np.ndarray]]:
        """Step one frame. dets_xyxy: (N, 4). Returns list of (id, xyxy)."""
        self.frame_count += 1

        # Predict for all existing tracks
        predicted: List[np.ndarray] = []
        to_remove: List[int] = []
        for i, t in enumerate(self.tracks):
            pred = t.kf.predict()
            if np.any(np.isnan(pred)):
                to_remove.append(i)
                predicted.append(np.zeros(4))
            else:
                predicted.append(pred)
        if to_remove:
            self.tracks = [t for i, t in enumerate(self.tracks) if i not in set(to_remove)]
            predicted = [p for i, p in enumerate(predicted) if i not in set(to_remove)]

        # Association
        if len(self.tracks) == 0 or len(dets_xyxy) == 0:
            matched = np.empty((0, 2), dtype=int)
            unm_t = np.arange(len(self.tracks), dtype=int)
            unm_d = np.arange(len(dets_xyxy), dtype=int)
        else:
            pred_arr = np.stack(predicted, axis=0)
            iou = iou_xyxy(dets_xyxy, pred_arr)   # (D, T)
            cost = 1.0 - iou
            matched, unm_d, unm_t = hungarian(cost,
                                              max_cost=1.0 - self.iou_threshold)

        # Apply matches (matched[:,0]=det idx, matched[:,1]=track idx)
        for d_idx, t_idx in matched:
            self.tracks[t_idx].kf.update(dets_xyxy[d_idx])
            self.tracks[t_idx].hits += 1
            self.tracks[t_idx].time_since_update = 0

        # Unmatched detections -> new tracks
        for d_idx in unm_d:
            new = _Track(KalmanBox(dets_xyxy[d_idx]), self._next_id)
            self._next_id += 1
            self.tracks.append(new)

        # Unmatched tracks -> increment time_since_update
        for t_idx in unm_t:
            self.tracks[t_idx].time_since_update += 1

        # Age, remove stale
        survivors: List[_Track] = []
        for t in self.tracks:
            t.age += 1
            if t.time_since_update <= self.max_age:
                survivors.append(t)
        self.tracks = survivors

        # Emit confirmed tracks
        out: List[Tuple[int, np.ndarray]] = []
        for t in self.tracks:
            if t.time_since_update > 0:
                continue
            if t.hits >= self.min_hits or self.frame_count <= self.min_hits:
                out.append((t.track_id, t.kf.bbox.copy()))
        return out
