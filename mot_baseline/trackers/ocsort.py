"""OC-SORT (Cao et al., CVPR 2023).

A pragmatic, self-contained reimplementation focused on the three core ideas:
1. Observation-Centric Re-Update (ORU): when a track is re-associated after
   being lost, the Kalman state is virtually re-stepped using the last
   observation -> current observation linear trajectory.
2. Observation-Centric Momentum (OCM): adds a directional consistency cost
   between the previous-observation->current-observation direction and each
   candidate detection direction.
3. Observation-Centric Recovery (OCR): a second association round between
   unmatched detections and unmatched tracks using their last observations
   (instead of predictions).

This is not the official optimized implementation; it follows the official
spec closely enough to give a strong-baseline equivalent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .utils import KalmanBox, hungarian, iou_xyxy


def _box_center(b: np.ndarray) -> np.ndarray:
    return np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0])


def _cosine(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    return float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))


@dataclass
class _OCTrack:
    kf: KalmanBox
    track_id: int
    last_obs: np.ndarray             # last observed xyxy
    last_obs_frame: int              # frame at which last_obs was recorded
    age: int = 0
    hits: int = 1
    time_since_update: int = 0
    obs_history: List[Tuple[int, np.ndarray]] = field(default_factory=list)


class OCSortTracker:
    def __init__(self,
                 max_age: int = 30,
                 min_hits: int = 3,
                 iou_threshold: float = 0.3,
                 delta_t: int = 3,
                 inertia: float = 0.2,
                 use_ocr: bool = True,
                 ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.delta_t = delta_t            # window for direction estimation
        self.inertia = inertia            # weight of OCM term
        self.use_ocr = use_ocr
        self.tracks: List[_OCTrack] = []
        self.frame_count = 0
        self._next_id = 1

    # -- internal helpers ----------------------------------------------------

    def _track_direction(self, t: _OCTrack) -> Optional[np.ndarray]:
        """Direction vector from a past observation (~delta_t back) to last."""
        if len(t.obs_history) < 2:
            return None
        target_frame = t.last_obs_frame - self.delta_t
        prev = None
        for f, b in t.obs_history:
            if f <= target_frame:
                prev = b
            else:
                break
        if prev is None:
            prev = t.obs_history[0][1]
        return _box_center(t.last_obs) - _box_center(prev)

    def _associate(self,
                   dets: np.ndarray,
                   tracks: List[_OCTrack],
                   preds: np.ndarray,
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """First-round association using IoU + OCM."""
        if len(tracks) == 0 or len(dets) == 0:
            return (np.empty((0, 2), dtype=int),
                    np.arange(len(dets), dtype=int),
                    np.arange(len(tracks), dtype=int))

        iou = iou_xyxy(dets, preds)              # (D, T)
        cost = 1.0 - iou

        # OCM term: directional consistency
        if self.inertia > 0:
            det_centers = np.stack([_box_center(d) for d in dets], axis=0)
            for j, t in enumerate(tracks):
                v_track = self._track_direction(t)
                if v_track is None:
                    continue
                last_c = _box_center(t.last_obs)
                for i in range(len(dets)):
                    v_det = det_centers[i] - last_c
                    ang_cost = (1.0 - _cosine(v_track, v_det)) / 2.0  # [0,1]
                    cost[i, j] += self.inertia * ang_cost

        return hungarian(cost, max_cost=1.0 - self.iou_threshold + self.inertia)

    def _observation_centric_reupdate(self, t: _OCTrack,
                                       new_obs: np.ndarray,
                                       cur_frame: int) -> None:
        """ORU: re-step the Kalman state along the virtual trajectory."""
        if t.time_since_update <= 0:
            return
        prev = t.last_obs
        # number of intermediate steps
        n = max(1, cur_frame - t.last_obs_frame)
        for s in range(1, n + 1):
            alpha = s / float(n)
            virt = prev * (1.0 - alpha) + new_obs * alpha
            t.kf.update(virt)

    # -- main step -----------------------------------------------------------

    def update(self,
               dets_xyxy: np.ndarray,
               scores: Optional[np.ndarray] = None,
               ) -> List[Tuple[int, np.ndarray]]:
        self.frame_count += 1

        # Predict
        predicted: List[np.ndarray] = []
        for t in self.tracks:
            predicted.append(t.kf.predict())

        # First round on IoU + OCM
        if len(self.tracks) > 0:
            pred_arr = np.stack(predicted, axis=0)
        else:
            pred_arr = np.zeros((0, 4))
        matched, unm_d, unm_t = self._associate(dets_xyxy, self.tracks, pred_arr)

        # Second round (OCR): match unmatched dets to unmatched tracks via last_obs
        if self.use_ocr and len(unm_d) > 0 and len(unm_t) > 0:
            last_obs_arr = np.stack([self.tracks[j].last_obs for j in unm_t], axis=0)
            sub_dets = dets_xyxy[unm_d]
            iou2 = iou_xyxy(sub_dets, last_obs_arr)
            cost2 = 1.0 - iou2
            m2, u2_d, u2_t = hungarian(cost2, max_cost=1.0 - self.iou_threshold)
            extra: List[Tuple[int, int]] = []
            for r, c in m2:
                extra.append((unm_d[r], unm_t[c]))
            if extra:
                matched = np.vstack([matched, np.array(extra, dtype=int)])
            unm_d = unm_d[u2_d]
            unm_t = unm_t[u2_t]

        # Apply matches
        for d_idx, t_idx in matched:
            track = self.tracks[t_idx]
            if track.time_since_update > 0:
                self._observation_centric_reupdate(track,
                                                   dets_xyxy[d_idx],
                                                   self.frame_count)
            track.kf.update(dets_xyxy[d_idx])
            track.hits += 1
            track.time_since_update = 0
            track.last_obs = dets_xyxy[d_idx].copy()
            track.last_obs_frame = self.frame_count
            track.obs_history.append((self.frame_count, dets_xyxy[d_idx].copy()))
            # keep history bounded
            if len(track.obs_history) > 50:
                track.obs_history = track.obs_history[-50:]

        # Unmatched dets -> new tracks
        for d_idx in unm_d:
            box = dets_xyxy[d_idx]
            new = _OCTrack(KalmanBox(box), self._next_id,
                           last_obs=box.copy(),
                           last_obs_frame=self.frame_count,
                           obs_history=[(self.frame_count, box.copy())])
            self._next_id += 1
            self.tracks.append(new)

        # Unmatched tracks bookkeeping
        for t_idx in unm_t:
            self.tracks[t_idx].time_since_update += 1

        # Age / cleanup
        survivors: List[_OCTrack] = []
        for t in self.tracks:
            t.age += 1
            if t.time_since_update <= self.max_age:
                survivors.append(t)
        self.tracks = survivors

        # Emit
        out: List[Tuple[int, np.ndarray]] = []
        for t in self.tracks:
            if t.time_since_update > 0:
                continue
            if t.hits >= self.min_hits or self.frame_count <= self.min_hits:
                out.append((t.track_id, t.kf.bbox.copy()))
        return out
