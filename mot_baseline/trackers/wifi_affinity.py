"""WiFi-as-ReID OC-SORT — learned affinity edition (Plan D).

Reuses the bind/monopoly/cost machinery from wifi_ocsort, but replaces the
geometric (depth, FTM) Gaussian compatibility with a forward pass through the
pre-trained MultimodalNetwork from my_vifi.

Per-frame procedure:
    1. predict KF, run IoU + OCM associate (vanilla OC-SORT)
    2. apply matches, push (cx, cy, depth) into AffinityHelper.track_buf
    3. push (FTM, IMU9) per phone into AffinityHelper.phone_buf
    4. Forward affinity → (Np, Nt) probability table
    5. EMA-update each track.phone_scores from the affinity row
    6. Resolve binds (greedy + monopoly), emit phone-stable IDs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .ocsort import _box_center, _cosine
from .utils import KalmanBox, hungarian, iou_xyxy
from .affinity_helper import AffinityHelper


@dataclass
class _AffTrack:
    kf: KalmanBox
    track_id: int
    last_obs: np.ndarray
    last_obs_frame: int
    age: int = 0
    hits: int = 1
    time_since_update: int = 0
    obs_history: List[Tuple[int, np.ndarray]] = field(default_factory=list)
    last_depth: float = float("nan")
    phone_scores: Dict[int, float] = field(default_factory=dict)
    bound_phone: Optional[int] = None
    bound_age: int = 0


class WiFiAffinityTracker:
    """OC-SORT with learned-affinity-driven phone soft binding."""

    def __init__(self,
                 max_age: int = 30,
                 min_hits: int = 3,
                 iou_threshold: float = 0.3,
                 delta_t: int = 3,
                 inertia: float = 0.2,
                 use_ocr: bool = True,
                 # WiFi params
                 wifi_weight: float = 0.10,
                 ema_alpha: float = 0.85,
                 bind_threshold: float = 0.5,
                 unbind_threshold: float = 0.10,
                 bind_init_threshold: float = 0.6,
                 phone_id_offset: int = 100000,
                 # Affinity model
                 ckpt_path: Optional[str] = None,
                 norm_stats_path: Optional[str] = None,
                 affinity_k: int = 10,
                 affinity_min_history: int = 3,
                 affinity_device: str = "cpu",
                 ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.delta_t = delta_t
        self.inertia = inertia
        self.use_ocr = use_ocr
        self.wifi_weight = wifi_weight
        self.ema_alpha = ema_alpha
        self.bind_threshold = bind_threshold
        self.unbind_threshold = unbind_threshold
        self.bind_init_threshold = bind_init_threshold
        self.phone_id_offset = phone_id_offset

        if ckpt_path is None or norm_stats_path is None:
            raise ValueError("ckpt_path and norm_stats_path must be provided")
        self.affinity = AffinityHelper(
            ckpt_path=ckpt_path,
            norm_stats_path=norm_stats_path,
            k=affinity_k,
            min_history=affinity_min_history,
            device=affinity_device,
        )

        self.tracks: List[_AffTrack] = []
        self.frame_count = 0
        self._next_id = 1

    # ----------------------------------------------------------------- helpers

    def _track_direction(self, t: _AffTrack) -> Optional[np.ndarray]:
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

    def _phones_in_use(self, exclude: Optional[_AffTrack] = None) -> set:
        return {t.bound_phone for t in self.tracks
                if t.bound_phone is not None and t is not exclude}

    # ------------------------------------------------------------- association

    def _associate(self, dets: np.ndarray, preds: np.ndarray
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(self.tracks) == 0 or len(dets) == 0:
            return (np.empty((0, 2), dtype=int),
                    np.arange(len(dets), dtype=int),
                    np.arange(len(self.tracks), dtype=int))

        iou = iou_xyxy(dets, preds)
        cost = 1.0 - iou

        if self.inertia > 0:
            det_centers = np.stack([_box_center(d) for d in dets], axis=0)
            for j, t in enumerate(self.tracks):
                v_track = self._track_direction(t)
                if v_track is None:
                    continue
                last_c = _box_center(t.last_obs)
                for i in range(len(dets)):
                    v_det = det_centers[i] - last_c
                    ang_cost = (1.0 - _cosine(v_track, v_det)) / 2.0
                    cost[i, j] += self.inertia * ang_cost

        # Wireless penalty on bound tracks: if a det's depth diverges from the
        # bound phone's affinity prediction (we use a track's recent best
        # phone_score as a coarse proxy), increase cost. This keeps the same
        # spirit as the geometric variant but without per-det depth lookups.
        # Actual wifi-aware re-id happens in phone_score updates after match.
        if self.wifi_weight > 0:
            for j, t in enumerate(self.tracks):
                if t.bound_phone is None:
                    continue
                # if we have a recent affinity score for this track, use it as
                # confidence; otherwise no penalty.
                conf = t.phone_scores.get(t.bound_phone, 0.0)
                # higher confidence => more penalty for swaps from other dets
                # we use a uniform per-track penalty bonus added to all dets;
                # this is essentially a track-level bias not changing matching
                # outcomes if the bound track stays best on IoU. Skip explicit
                # add — see resolve_binds for the actual ID-stable effect.
                _ = conf

        max_cost = 1.0 - self.iou_threshold + self.inertia
        return hungarian(cost, max_cost=max_cost)

    def _ocr(self, dets: np.ndarray, unm_d: np.ndarray, unm_t: np.ndarray
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.use_ocr or len(unm_d) == 0 or len(unm_t) == 0:
            return np.empty((0, 2), dtype=int), unm_d, unm_t
        last_obs_arr = np.stack([self.tracks[j].last_obs for j in unm_t], axis=0)
        sub_dets = dets[unm_d]
        iou2 = iou_xyxy(sub_dets, last_obs_arr)
        cost2 = 1.0 - iou2
        m2, u2_d, u2_t = hungarian(cost2, max_cost=1.0 - self.iou_threshold)
        extras = []
        for r, c in m2:
            extras.append((unm_d[r], unm_t[c]))
        new_unm_d = unm_d[u2_d]
        new_unm_t = unm_t[u2_t]
        return (np.array(extras, dtype=int).reshape(-1, 2),
                new_unm_d, new_unm_t)

    def _oru(self, t: _AffTrack, new_obs: np.ndarray, cur_frame: int) -> None:
        if t.time_since_update <= 0:
            return
        prev = t.last_obs
        n = max(1, cur_frame - t.last_obs_frame)
        for s in range(1, n + 1):
            alpha = s / float(n)
            virt = prev * (1.0 - alpha) + new_obs * alpha
            t.kf.update(virt)

    # --------------------------------------------------------------- main step

    def update(self, frame) -> List[Tuple[int, np.ndarray]]:
        """Process one Frame (defined in data/vifi_mot.py)."""
        self.frame_count += 1

        # 1) gather detections + scores (we do not actually need scores here)
        if frame.detections:
            dets = np.array([d.bbox for d in frame.detections], dtype=np.float64)
            depths = np.array([d.depth for d in frame.detections], dtype=np.float64)
            cxcy = np.array([(d.cxcywh[0], d.cxcywh[1])
                             for d in frame.detections], dtype=np.float64)
        else:
            dets = np.zeros((0, 4))
            depths = np.zeros((0,))
            cxcy = np.zeros((0, 2))

        # 2) push phone history (from frame.ftm + frame.imu_agm9)
        if frame.ftm is not None and frame.imu_agm9 is not None:
            N_phone = frame.ftm.shape[0]
            for p in range(N_phone):
                ftm2 = frame.ftm[p]                    # (2,)
                imu9 = frame.imu_agm9[p]               # (9,)
                self.affinity.push_phone(p, ftm2, imu9)
        else:
            N_phone = 0

        # 3) predict
        if self.tracks:
            preds = np.stack([t.kf.predict() for t in self.tracks], axis=0)
        else:
            preds = np.zeros((0, 4))

        # 4) first-round associate (IoU + OCM)
        matched, unm_d, unm_t = self._associate(dets, preds)
        extras, unm_d, unm_t = self._ocr(dets, unm_d, unm_t)
        if len(extras):
            matched = np.vstack([matched, extras])

        # 5) apply matches + push track history
        for d_idx, t_idx in matched:
            track = self.tracks[t_idx]
            if track.time_since_update > 0:
                self._oru(track, dets[d_idx], self.frame_count)
            track.kf.update(dets[d_idx])
            track.hits += 1
            track.time_since_update = 0
            track.last_obs = dets[d_idx].copy()
            track.last_obs_frame = self.frame_count
            track.obs_history.append((self.frame_count, dets[d_idx].copy()))
            if len(track.obs_history) > 50:
                track.obs_history = track.obs_history[-50:]
            depth = float(depths[d_idx])
            track.last_depth = depth
            cx, cy = float(cxcy[d_idx, 0]), float(cxcy[d_idx, 1])
            self.affinity.push_track(track.track_id, cx, cy, depth)

        # 6) new tracks
        new_tracks: List[_AffTrack] = []
        for d_idx in unm_d:
            box = dets[d_idx]
            new = _AffTrack(KalmanBox(box), self._next_id,
                            last_obs=box.copy(),
                            last_obs_frame=self.frame_count,
                            obs_history=[(self.frame_count, box.copy())])
            self._next_id += 1
            depth = float(depths[d_idx])
            new.last_depth = depth
            cx, cy = float(cxcy[d_idx, 0]), float(cxcy[d_idx, 1])
            self.affinity.push_track(new.track_id, cx, cy, depth)
            self.tracks.append(new)
            new_tracks.append(new)

        # 7) unmatched tracks
        for t_idx in unm_t:
            self.tracks[t_idx].time_since_update += 1

        # 8) age + cull
        survivors: List[_AffTrack] = []
        for t in self.tracks:
            t.age += 1
            if t.time_since_update <= self.max_age:
                survivors.append(t)
            else:
                self.affinity.drop_track(t.track_id)
        self.tracks = survivors

        # 9) Forward affinity model on currently-active tracks (those visible
        # this frame, i.e. time_since_update == 0). Phone universe = 0..N-1.
        active_tracks = [t for t in self.tracks if t.time_since_update == 0]
        if N_phone > 0 and active_tracks:
            tids = [t.track_id for t in active_tracks]
            pids = list(range(N_phone))
            aff = self.affinity.predict(tids, pids)   # (N_phone, len(active))
            # 10) EMA-update phone_scores
            for j, t in enumerate(active_tracks):
                new_scores = dict(t.phone_scores)
                for p in range(N_phone):
                    a = float(aff[p, j])
                    prev = new_scores.get(p, 0.0)
                    new_scores[p] = self.ema_alpha * prev + (1.0 - self.ema_alpha) * a
                t.phone_scores = new_scores

        # 11) Resolve binds (greedy by score, monopoly-aware)
        self._resolve_phone_binds()

        # 12) Try to bind brand-new tracks aggressively if model already
        # confident on the very first frame
        for t in new_tracks:
            if t.bound_phone is not None:
                continue
            if not t.phone_scores:
                continue
            in_use = self._phones_in_use(exclude=t)
            best_p, best_s = None, 0.0
            for p, s in t.phone_scores.items():
                if p in in_use:
                    continue
                if s > best_s:
                    best_s = s
                    best_p = p
            if best_p is not None and best_s >= self.bind_init_threshold:
                t.bound_phone = best_p
                t.bound_age = 0

        # 13) Emit
        out: List[Tuple[int, np.ndarray]] = []
        for t in self.tracks:
            if t.time_since_update > 0:
                continue
            if t.hits >= self.min_hits or self.frame_count <= self.min_hits:
                tid = (self.phone_id_offset + t.bound_phone
                       if t.bound_phone is not None else t.track_id)
                out.append((tid, t.kf.bbox.copy()))
        return out

    # ---------------------------------------------------------------- binding

    def _resolve_phone_binds(self) -> None:
        # 1) Unbind low-score tracks
        for t in self.tracks:
            if t.bound_phone is not None:
                t.bound_age += 1
                score = t.phone_scores.get(t.bound_phone, 0.0)
                if score < self.unbind_threshold:
                    t.bound_phone = None
                    t.bound_age = 0
        # 2) Greedy assign unbound tracks to free phones
        bound_phones = self._phones_in_use()
        cands: List[Tuple[float, int, int]] = []
        for ti, t in enumerate(self.tracks):
            if t.bound_phone is not None:
                continue
            for p, s in t.phone_scores.items():
                if p in bound_phones:
                    continue
                if s >= self.bind_threshold:
                    cands.append((s, ti, p))
        cands.sort(reverse=True)
        used_t, used_p = set(), set(bound_phones)
        for s, ti, p in cands:
            if ti in used_t or p in used_p:
                continue
            self.tracks[ti].bound_phone = p
            self.tracks[ti].bound_age = 0
            used_t.add(ti)
            used_p.add(p)
