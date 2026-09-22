"""WiFi-as-ReID enhanced OC-SORT.

Idea
----
ViFi 数据集中每帧的合法用户携带手机持续上报 FTM 距离 / RSSI / IMU。
我们把这些 wireless 流当作每个 track 的"无外观 ReID"信号源：

1. **兼容性** = FTM 距离 (米) 与 detection depth (米) 的差异，转高斯衰减
2. **EMA 软绑定**：track 历次匹配的 detection depth 与每个 phone 的 FTM 距离比对，
   累计 phone_score；超过阈值 → 绑定该 phone，并以 phone-stable ID 输出
3. **Phone Monopoly**：同一个 phone 最多绑定一个 track（防 ID 冲突）
4. **Bind-on-Creation**：新 track 一旦能与某 phone 高度兼容，立刻绑定，避免后期切 ID 引入 IDsw
5. **Wireless cost**：关联代价矩阵中，对已绑定 phone 的 track 加入 depth-FTM 不一致 penalty
6. **旁观者无 wireless** → 走纯 IoU + OCM 路径（与 vanilla OC-SORT 一致）

Notes
-----
- Tracker 不知道 detection 与 phone 的真实对应关系（GT 仅用于评测），
  这与现实部署中"AP 看到 N 个匿名 MAC"的设定一致。
- 输出 track ID：未绑定 → short-term ID；已绑定 → `phone_id_offset + phone_idx`。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .ocsort import _box_center, _cosine
from .utils import KalmanBox, hungarian, iou_xyxy


# ---------------------------------------------------------------------------
# Track data class
# ---------------------------------------------------------------------------

@dataclass
class _WiFiTrack:
    kf: KalmanBox
    track_id: int
    last_obs: np.ndarray
    last_obs_frame: int
    age: int = 0
    hits: int = 1
    time_since_update: int = 0
    obs_history: List[Tuple[int, np.ndarray]] = field(default_factory=list)
    # wireless extension --------------------------------------------------
    last_depth: float = float("nan")
    depth_history: List[float] = field(default_factory=list)
    phone_scores: Dict[int, float] = field(default_factory=dict)
    bound_phone: Optional[int] = None
    bound_age: int = 0


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class WiFiOCSortTracker:
    """OC-SORT with WiFi-FTM-as-ReID enhancement."""

    def __init__(self,
                 # OC-SORT base params
                 max_age: int = 30,
                 min_hits: int = 3,
                 iou_threshold: float = 0.3,
                 delta_t: int = 3,
                 inertia: float = 0.2,
                 use_ocr: bool = True,
                 # WiFi params
                 wifi_weight: float = 0.4,
                 depth_sigma: float = 1.5,         # gaussian sigma in meters
                 ema_alpha: float = 0.85,
                 bind_threshold: float = 0.6,
                 unbind_threshold: float = 0.15,
                 bind_init_threshold: float = 0.7,
                 phone_id_offset: int = 100000,
                 ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.delta_t = delta_t
        self.inertia = inertia
        self.use_ocr = use_ocr
        # wireless
        self.wifi_weight = wifi_weight
        self.depth_sigma = depth_sigma
        self.ema_alpha = ema_alpha
        self.bind_threshold = bind_threshold
        self.unbind_threshold = unbind_threshold
        self.bind_init_threshold = bind_init_threshold
        self.phone_id_offset = phone_id_offset

        self.tracks: List[_WiFiTrack] = []
        self.frame_count = 0
        self._next_id = 1

    # -- helpers -------------------------------------------------------------

    def _track_direction(self, t: _WiFiTrack) -> Optional[np.ndarray]:
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

    def _compat(self, depth: float, ftm_m: float) -> float:
        """Depth-FTM compatibility in [0,1]; e^{-d^2 / 2 sigma^2}."""
        if np.isnan(depth) or np.isnan(ftm_m):
            return 0.0
        d = depth - ftm_m
        return float(np.exp(-(d * d) / (2.0 * self.depth_sigma ** 2)))

    def _phones_in_use(self, exclude: Optional[_WiFiTrack] = None) -> set:
        return {t.bound_phone for t in self.tracks
                if t.bound_phone is not None and t is not exclude}

    def _try_initial_bind(self,
                          track: _WiFiTrack,
                          depth: float,
                          ftm_m: np.ndarray,
                          ftm_valid: np.ndarray) -> None:
        """Bind a freshly-created track if a phone already matches strongly."""
        if np.isnan(depth):
            return
        in_use = self._phones_in_use()
        best_p, best_compat = None, 0.0
        for p in range(len(ftm_m)):
            if not ftm_valid[p] or p in in_use:
                continue
            c = self._compat(depth, ftm_m[p])
            if c > best_compat:
                best_compat = c
                best_p = p
        if best_p is not None and best_compat >= self.bind_init_threshold:
            track.bound_phone = best_p
            track.phone_scores[best_p] = best_compat

    def _associate(self,
                   dets: np.ndarray,
                   tracks: List[_WiFiTrack],
                   preds: np.ndarray,
                   det_depths: np.ndarray,
                   ftm_m: np.ndarray,
                   ftm_valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """First-round association: IoU + OCM + WiFi cost on bound tracks."""
        if len(tracks) == 0 or len(dets) == 0:
            return (np.empty((0, 2), dtype=int),
                    np.arange(len(dets), dtype=int),
                    np.arange(len(tracks), dtype=int))

        iou = iou_xyxy(dets, preds)
        cost = 1.0 - iou

        # OCM directional consistency
        if self.inertia > 0:
            det_centers = np.stack([_box_center(d) for d in dets], axis=0)
            for j, t in enumerate(tracks):
                v_track = self._track_direction(t)
                if v_track is None:
                    continue
                last_c = _box_center(t.last_obs)
                for i in range(len(dets)):
                    v_det = det_centers[i] - last_c
                    ang_cost = (1.0 - _cosine(v_track, v_det)) / 2.0
                    cost[i, j] += self.inertia * ang_cost

        # WiFi cost on bound tracks
        if self.wifi_weight > 0 and len(ftm_m) > 0:
            for j, t in enumerate(tracks):
                if t.bound_phone is None:
                    continue
                p = t.bound_phone
                if not ftm_valid[p]:
                    continue
                ftm = float(ftm_m[p])
                for i in range(len(dets)):
                    if np.isnan(det_depths[i]):
                        continue
                    compat = self._compat(det_depths[i], ftm)
                    cost[i, j] += self.wifi_weight * (1.0 - compat)

        max_cost = 1.0 - self.iou_threshold + self.inertia + self.wifi_weight
        return hungarian(cost, max_cost=max_cost)

    def _ocr(self,
             dets: np.ndarray,
             tracks: List[_WiFiTrack],
             unm_d: np.ndarray,
             unm_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.use_ocr or len(unm_d) == 0 or len(unm_t) == 0:
            return np.empty((0, 2), dtype=int), unm_d, unm_t
        last_obs_arr = np.stack([tracks[j].last_obs for j in unm_t], axis=0)
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

    def _oru(self, t: _WiFiTrack, new_obs: np.ndarray, cur_frame: int) -> None:
        if t.time_since_update <= 0:
            return
        prev = t.last_obs
        n = max(1, cur_frame - t.last_obs_frame)
        for s in range(1, n + 1):
            alpha = s / float(n)
            virt = prev * (1.0 - alpha) + new_obs * alpha
            t.kf.update(virt)

    # -- main step -----------------------------------------------------------

    def update(self,
               dets_xyxy: np.ndarray,
               det_depths: np.ndarray,
               scores: Optional[np.ndarray],
               ftm_m: np.ndarray,
               ftm_valid: np.ndarray) -> List[Tuple[int, np.ndarray]]:
        """Step one frame.

        Parameters
        ----------
        dets_xyxy   : (D, 4) detection boxes
        det_depths  : (D,)   detection depth in meters (NaN allowed)
        scores      : (D,)   detection scores (used only for compat with API)
        ftm_m       : (P,)   FTM range in meters per phone, NaN if missing
        ftm_valid   : (P,)   bool mask
        """
        self.frame_count += 1

        # Predict
        predicted = [t.kf.predict() for t in self.tracks]
        if self.tracks:
            pred_arr = np.stack(predicted, axis=0)
        else:
            pred_arr = np.zeros((0, 4))

        # First round association
        matched, unm_d, unm_t = self._associate(dets_xyxy, self.tracks, pred_arr,
                                                det_depths, ftm_m, ftm_valid)

        # OCR second round
        extras, unm_d, unm_t = self._ocr(dets_xyxy, self.tracks, unm_d, unm_t)
        if len(extras):
            matched = np.vstack([matched, extras])

        # Apply matches & update phone scores --------------------------------
        for d_idx, t_idx in matched:
            track = self.tracks[t_idx]
            if track.time_since_update > 0:
                self._oru(track, dets_xyxy[d_idx], self.frame_count)
            track.kf.update(dets_xyxy[d_idx])
            track.hits += 1
            track.time_since_update = 0
            track.last_obs = dets_xyxy[d_idx].copy()
            track.last_obs_frame = self.frame_count
            track.obs_history.append((self.frame_count, dets_xyxy[d_idx].copy()))
            if len(track.obs_history) > 50:
                track.obs_history = track.obs_history[-50:]

            depth = float(det_depths[d_idx]) if not np.isnan(det_depths[d_idx]) else float("nan")
            if not np.isnan(depth):
                track.last_depth = depth
                track.depth_history.append(depth)
                if len(track.depth_history) > 50:
                    track.depth_history = track.depth_history[-50:]
            self._update_phone_scores(track, depth, ftm_m, ftm_valid)

        # Phone bind / unbind decisions --------------------------------------
        self._resolve_phone_binds()

        # Unmatched detections -> new tracks ---------------------------------
        for d_idx in unm_d:
            box = dets_xyxy[d_idx]
            new = _WiFiTrack(KalmanBox(box), self._next_id,
                             last_obs=box.copy(),
                             last_obs_frame=self.frame_count,
                             obs_history=[(self.frame_count, box.copy())])
            self._next_id += 1
            depth = float(det_depths[d_idx]) if not np.isnan(det_depths[d_idx]) else float("nan")
            if not np.isnan(depth):
                new.last_depth = depth
                new.depth_history.append(depth)
            self.tracks.append(new)
            # try immediate bind
            self._try_initial_bind(new, depth, ftm_m, ftm_valid)

        # Unmatched tracks
        for t_idx in unm_t:
            self.tracks[t_idx].time_since_update += 1

        # Aging / cleanup
        survivors: List[_WiFiTrack] = []
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
                if t.bound_phone is not None:
                    tid = self.phone_id_offset + t.bound_phone
                else:
                    tid = t.track_id
                out.append((tid, t.kf.bbox.copy()))
        return out

    def _update_phone_scores(self,
                             track: _WiFiTrack,
                             depth: float,
                             ftm_m: np.ndarray,
                             ftm_valid: np.ndarray) -> None:
        if np.isnan(depth) or len(ftm_m) == 0:
            return
        new_scores = dict(track.phone_scores)
        for p in range(len(ftm_m)):
            if not ftm_valid[p]:
                # decay
                if p in new_scores:
                    new_scores[p] *= self.ema_alpha
                continue
            compat = self._compat(depth, float(ftm_m[p]))
            prev = new_scores.get(p, 0.0)
            new_scores[p] = self.ema_alpha * prev + (1.0 - self.ema_alpha) * compat
        track.phone_scores = new_scores

    def _resolve_phone_binds(self) -> None:
        """Apply bind / unbind / steal-with-priority logic across tracks."""
        # 1) tracks already bound: check unbind
        for t in self.tracks:
            if t.bound_phone is not None:
                t.bound_age += 1
                score = t.phone_scores.get(t.bound_phone, 0.0)
                if score < self.unbind_threshold:
                    t.bound_phone = None
                    t.bound_age = 0

        # 2) try to bind unbound tracks to free phones, by descending compat
        bound_phones = self._phones_in_use()
        candidates: List[Tuple[float, int, int]] = []   # (score, track_idx, phone)
        for ti, t in enumerate(self.tracks):
            if t.bound_phone is not None:
                continue
            for p, s in t.phone_scores.items():
                if p in bound_phones:
                    continue
                if s >= self.bind_threshold:
                    candidates.append((s, ti, p))
        # greedy by score
        candidates.sort(reverse=True)
        used_tracks: set = set()
        used_phones: set = set(bound_phones)
        for s, ti, p in candidates:
            if ti in used_tracks or p in used_phones:
                continue
            self.tracks[ti].bound_phone = p
            self.tracks[ti].bound_age = 0
            used_tracks.add(ti)
            used_phones.add(p)
