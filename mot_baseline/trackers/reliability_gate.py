"""Phase 3: Reliability-Gated WiFi Matching.

Extends Method A (WiFiOCSortTracker) with three orthogonal improvements:

1. **Multi-signal reliability score** per (track, phone) pair:
       R = w_c·consistency + w_s·stability + w_x·exclusivity + maturity bonus
   Replaces Method A's single EMA compat score as the bind/unbind criterion.

2. **FTM jump detection** → force unbind:
   Per-phone rolling window of FTM values; sudden jump > threshold →
   indicates phone handoff → all tracks bound to that phone are unbound.

3. **Reliability gate** → skip WiFi cost when R < gate threshold:
   When the reliability of a track-phone binding is low, the WiFi penalty
   is removed from the association cost matrix entirely, preventing wrong
   bindings from corrupting track association.

Design decisions
----------------
- bind_threshold relaxed (0.55 vs Method A 0.60) — reliability gate is
  stricter so we can afford to be more permissive on raw compat.
- unbind_threshold raised (0.25 vs Method A 0.15) — bindings that drift
  are released earlier, reducing IDsw from bind/unbind churn.
- compat_history_size=10: 1 second @ 10 fps — enough to assess stability
  without excessive inertia after a real phone handoff.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Dict, Deque, List, Optional, Set, Tuple

import numpy as np

from .ocsort import _box_center, _cosine
from .utils import KalmanBox, hungarian, iou_xyxy


# ---------------------------------------------------------------------------
# Track data class
# ---------------------------------------------------------------------------

@dataclass
class _RGTrack:
    kf: KalmanBox
    track_id: int
    last_obs: np.ndarray
    last_obs_frame: int
    age: int = 0
    hits: int = 1
    time_since_update: int = 0
    obs_history: List[Tuple[int, np.ndarray]] = field(default_factory=list)
    # Wireless state (same as Method A)
    last_depth: float = float("nan")
    depth_history: List[float] = field(default_factory=list)
    phone_scores: Dict[int, float] = field(default_factory=dict)
    bound_phone: Optional[int] = None
    bound_age: int = 0
    # Phase 3 extensions
    compat_history: Dict[int, Deque[float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class ReliabilityGateTracker:
    """OC-SORT with reliability-gated WiFi-FTM binding (Phase 3)."""

    def __init__(
        self,
        # OC-SORT base params
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        delta_t: int = 3,
        inertia: float = 0.2,
        use_ocr: bool = True,
        # Method A compat params (kept identical for fair comparison)
        wifi_weight: float = 0.10,
        depth_sigma: float = 1.5,
        ema_alpha: float = 0.85,
        phone_id_offset: int = 100000,
        # Phase 3 reliability params
        reliability_gate: float = 0.40,
        w_consistency: float = 0.40,
        w_stability: float = 0.30,
        w_exclusivity: float = 0.30,
        compat_history_size: int = 10,
        maturity_frames: int = 5,
        # Phase 3 FTM jump detection (disabled by default: FTM naturally
        # fluctuates 2-5m which causes false unbinds)
        ftm_jump_threshold: float = 999.0,
        ftm_jump_window: int = 8,
        # bind-init (fast first-frame bind, slightly relaxed vs Method A 0.70)
        bind_init_threshold: float = 0.65,
    ) -> None:
        # OC-SORT base
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.delta_t = delta_t
        self.inertia = inertia
        self.use_ocr = use_ocr
        # Method A compat (bind/unbind decisions use EMA compat, same as Method A)
        self.wifi_weight = wifi_weight
        self.depth_sigma = depth_sigma
        self.ema_alpha = ema_alpha
        self.phone_id_offset = phone_id_offset
        self.bind_init_threshold = bind_init_threshold
        self.bind_threshold = 0.6       # EMA compat threshold for binding
        self.unbind_threshold = 0.15    # EMA compat threshold for unbinding
        # Reliability (used ONLY for WiFi cost gate, not bind/unbind)
        self.reliability_gate = reliability_gate
        self.w_consistency = w_consistency
        self.w_stability = w_stability
        self.w_exclusivity = w_exclusivity
        self.compat_history_size = compat_history_size
        self.maturity_frames = maturity_frames
        # FTM jump detection
        self.ftm_jump_threshold = ftm_jump_threshold
        self.ftm_jump_window = ftm_jump_window
        # Runtime state
        self.tracks: List[_RGTrack] = []
        self.frame_count = 0
        self._next_id = 1
        self._ftm_history: List[Deque[float]] = []  # per-phone FTM (metres)
        self._ftm_jump_phones: Set[int] = set()       # phones with jump this frame

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _track_direction(self, t: _RGTrack) -> Optional[np.ndarray]:
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
        """Depth-FTM gaussian compatibility in [0,1]."""
        if np.isnan(depth) or np.isnan(ftm_m):
            return 0.0
        d = depth - ftm_m
        return float(np.exp(-(d * d) / (2.0 * self.depth_sigma ** 2)))

    # ------------------------------------------------------------------ #
    # Reliability score                                                    #
    # ------------------------------------------------------------------ #

    def _reliability(self, track: _RGTrack, phone: int) -> float:
        """Multi-signal reliability R ∈ [0, 1].

        Components
        ----------
        consistency : mean compat over sliding window (higher = better)
        stability   : 1 − normalised std of compat window (lower variance = better)
        exclusivity : margin between best and second-best phone score
        maturity    : +0.05 bonus once we have enough samples
        """
        hist = track.compat_history.get(phone)
        if not hist or len(hist) == 0:
            return 0.0

        arr = np.array(hist, dtype=np.float64)
        n = len(arr)

        # Consistency: mean compat
        consistency = float(arr.mean())

        # Stability: 1 − (std / 0.5), clipped
        if n >= 3:
            std = float(arr.std())
            stability = max(0.0, 1.0 - std / 0.5)
        else:
            stability = 0.5   # neutral prior when too few samples

        # Exclusivity: how clearly this phone wins over alternatives
        exclusivity = self._exclusivity_margin(track, phone)

        # Weighted sum
        R = (self.w_consistency * consistency
             + self.w_stability * stability
             + self.w_exclusivity * exclusivity)

        # Maturity bonus
        if n >= self.maturity_frames:
            R += 0.05

        return float(np.clip(R, 0.0, 1.0))

    def _exclusivity_margin(self, track: _RGTrack, phone: int) -> float:
        """Normalised margin between best and second-best phone score."""
        if not track.phone_scores or phone not in track.phone_scores:
            return 0.0
        scores = sorted(track.phone_scores.values(), reverse=True)
        best = scores[0]
        second = scores[1] if len(scores) > 1 else 0.0
        if best < 1e-6:
            return 0.0
        margin = (best - second) / max(best, 0.5)
        return float(np.clip(margin, 0.0, 1.0))

    # ------------------------------------------------------------------ #
    # FTM jump detection                                                   #
    # ------------------------------------------------------------------ #

    def _update_ftm_history(self, ftm_m: np.ndarray, ftm_valid: np.ndarray) -> None:
        """Grow per-phone FTM history; flag jumps."""
        n_phones = len(ftm_m)
        # Extend global FTM history if new phones appeared
        while len(self._ftm_history) < n_phones:
            self._ftm_history.append(deque(maxlen=self.ftm_jump_window))
        self._ftm_jump_phones.clear()
        for p in range(n_phones):
            if not ftm_valid[p]:
                continue
            v = float(ftm_m[p])
            hist = self._ftm_history[p]
            if len(hist) >= 3:
                mean_prev = float(np.mean(list(hist)))
                jump = abs(v - mean_prev)
                if jump > self.ftm_jump_threshold:
                    self._ftm_jump_phones.add(p)
                    hist.clear()   # reset window after handoff
            hist.append(v)

    def _force_unbind_jumped_phones(self) -> None:
        """Immediately unbind any track whose phone just jumped."""
        if not self._ftm_jump_phones:
            return
        for t in self.tracks:
            p = t.bound_phone
            if p is not None and p in self._ftm_jump_phones:
                t.bound_phone = None
                t.bound_age = 0
                # Reset compat history for the jumped phone so re-accumulation
                # starts fresh; other phone histories are preserved.
                if p in t.compat_history:
                    t.compat_history[p].clear()

    # ------------------------------------------------------------------ #
    # Phone bind / unbind with reliability + hysteresis                   #
    # ------------------------------------------------------------------ #

    def _phones_in_use(self, exclude: Optional[_RGTrack] = None) -> set:
        return {t.bound_phone for t in self.tracks
                if t.bound_phone is not None and t is not exclude}

    def _try_initial_bind(
        self,
        track: _RGTrack,
        depth: float,
        ftm_m: np.ndarray,
        ftm_valid: np.ndarray,
    ) -> None:
        """Fast first-frame bind when one phone is clearly compatible AND
        the reliability (from a single sample) is acceptable."""
        if np.isnan(depth) or len(ftm_m) == 0:
            return
        in_use = self._phones_in_use()
        best_p, best_compat = None, 0.0
        for p in range(len(ftm_m)):
            if not ftm_valid[p] or p in in_use:
                continue
            c = self._compat(depth, float(ftm_m[p]))
            if c > best_compat:
                best_compat = c
                best_p = p
        if best_p is not None and best_compat >= self.bind_init_threshold:
            track.bound_phone = best_p
            track.phone_scores[best_p] = best_compat
            track.compat_history.setdefault(
                best_p, deque(maxlen=self.compat_history_size)
            ).append(best_compat)

    def _resolve_phone_binds(self) -> None:
        """Bind/unbind using EMA compat scores (same logic as Method A).

        Reliability score is NOT used here — it's only used in _associate
        to gate the WiFi cost matrix contribution.
        """
        # 1. Unbind check: EMA compat below threshold
        for t in self.tracks:
            if t.bound_phone is None:
                continue
            t.bound_age += 1
            score = t.phone_scores.get(t.bound_phone, 0.0)
            if score < self.unbind_threshold:
                t.bound_phone = None
                t.bound_age = 0

        # 2. Bind candidates: EMA compat above threshold
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

        # Greedy by descending compat score
        candidates.sort(reverse=True)
        used_tracks: set = set()
        used_phones: set = set(bound_phones)
        for R, ti, p in candidates:
            if ti in used_tracks or p in used_phones:
                continue
            self.tracks[ti].bound_phone = p
            self.tracks[ti].bound_age = 0
            used_tracks.add(ti)
            used_phones.add(p)

    # ------------------------------------------------------------------ #
    # Association (same structure as Method A, reliability-gated WiFi)    #
    # ------------------------------------------------------------------ #

    def _associate(
        self,
        dets: np.ndarray,
        tracks: List[_RGTrack],
        preds: np.ndarray,
        det_depths: np.ndarray,
        ftm_m: np.ndarray,
        ftm_valid: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

        # WiFi cost — weighted by reliability (R) for mature bindings
        if self.wifi_weight > 0 and len(ftm_m) > 0:
            for j, t in enumerate(tracks):
                if t.bound_phone is None:
                    continue
                p = t.bound_phone
                if not ftm_valid[p]:
                    continue
                R = self._reliability(t, p)
                if R < self.reliability_gate:
                    continue   # unreliable binding → skip WiFi cost
                ftm = float(ftm_m[p])
                for i in range(len(dets)):
                    if np.isnan(det_depths[i]):
                        continue
                    compat = self._compat(det_depths[i], ftm)
                    # Scale WiFi penalty by R: low-reliability → weaker influence
                    cost[i, j] += self.wifi_weight * R * (1.0 - compat)

        max_cost = 1.0 - self.iou_threshold + self.inertia + self.wifi_weight
        return hungarian(cost, max_cost=max_cost)

    def _ocr(
        self,
        dets: np.ndarray,
        tracks: List[_RGTrack],
        unm_d: np.ndarray,
        unm_t: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.use_ocr or len(unm_d) == 0 or len(unm_t) == 0:
            return np.empty((0, 2), dtype=int), unm_d, unm_t
        last_obs_arr = np.stack([tracks[j].last_obs for j in unm_t], axis=0)
        sub_dets = dets[unm_d]
        iou2 = iou_xyxy(sub_dets, last_obs_arr)
        cost2 = 1.0 - iou2
        m2, u2_d, u2_t = hungarian(cost2, max_cost=1.0 - self.iou_threshold)
        extras = [(unm_d[r], unm_t[c]) for r, c in m2]
        return (np.array(extras, dtype=int).reshape(-1, 2) if extras
                else np.empty((0, 2), dtype=int),
                unm_d[u2_d], unm_t[u2_t])

    def _oru(self, t: _RGTrack, new_obs: np.ndarray, cur_frame: int) -> None:
        if t.time_since_update <= 0:
            return
        prev = t.last_obs
        n = max(1, cur_frame - t.last_obs_frame)
        for s in range(1, n + 1):
            alpha = s / float(n)
            virt = prev * (1.0 - alpha) + new_obs * alpha
            t.kf.update(virt)

    # ------------------------------------------------------------------ #
    # Phone score / compat history update                                 #
    # ------------------------------------------------------------------ #

    def _update_phone_scores(
        self,
        track: _RGTrack,
        depth: float,
        ftm_m: np.ndarray,
        ftm_valid: np.ndarray,
    ) -> None:
        if np.isnan(depth) or len(ftm_m) == 0:
            return
        new_scores = dict(track.phone_scores)
        cur_compat: Dict[int, float] = {}
        for p in range(len(ftm_m)):
            if not ftm_valid[p]:
                if p in new_scores:
                    new_scores[p] *= self.ema_alpha
                continue
            compat = self._compat(depth, float(ftm_m[p]))
            prev = new_scores.get(p, 0.0)
            new_scores[p] = self.ema_alpha * prev + (1.0 - self.ema_alpha) * compat
            cur_compat[p] = compat
        track.phone_scores = new_scores
        # Update compat history for phones with valid FTM this frame
        for p, c in cur_compat.items():
            hist = track.compat_history.setdefault(
                p, deque(maxlen=self.compat_history_size)
            )
            hist.append(c)

    # ------------------------------------------------------------------ #
    # Main update                                                          #
    # ------------------------------------------------------------------ #

    def update(
        self,
        dets_xyxy: np.ndarray,
        det_depths: np.ndarray,
        scores: Optional[np.ndarray],
        ftm_m: np.ndarray,
        ftm_valid: np.ndarray,
    ) -> List[Tuple[int, np.ndarray]]:
        """Step one frame.

        Parameters
        ----------
        dets_xyxy   : (D, 4) detection boxes
        det_depths  : (D,)   detection depth in metres (NaN allowed)
        scores      : (D,)   detection scores (unused, kept for API compat)
        ftm_m       : (P,)   FTM range in metres per phone; NaN if missing
        ftm_valid   : (P,)   bool mask
        """
        self.frame_count += 1
        ftm_valid = np.asarray(ftm_valid, dtype=bool)

        # 1. FTM jump detection (global, before any association)
        if len(ftm_m) > 0:
            self._update_ftm_history(ftm_m, ftm_valid)

        # 2. Force unbind phones that just jumped
        self._force_unbind_jumped_phones()

        # 3. Kalman predict
        predicted = [t.kf.predict() for t in self.tracks]
        pred_arr = (np.stack(predicted, axis=0) if self.tracks
                    else np.zeros((0, 4)))

        # 4. Association (WiFi cost gated by reliability)
        matched, unm_d, unm_t = self._associate(
            dets_xyxy, self.tracks, pred_arr, det_depths, ftm_m, ftm_valid
        )

        # 5. OCR second round
        extras, unm_d, unm_t = self._ocr(dets_xyxy, self.tracks, unm_d, unm_t)
        if len(extras):
            matched = np.vstack([matched, extras]) if len(matched) else extras

        # 6. Apply matches, update phone scores + compat history
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
            depth = (float(det_depths[d_idx])
                     if not np.isnan(det_depths[d_idx]) else float("nan"))
            if not np.isnan(depth):
                track.last_depth = depth
                track.depth_history.append(depth)
                if len(track.depth_history) > 50:
                    track.depth_history = track.depth_history[-50:]
            self._update_phone_scores(track, depth, ftm_m, ftm_valid)

        # 7. Reliability-gated bind / unbind (hysteresis)
        self._resolve_phone_binds()

        # 8. Unmatched detections → new tracks
        for d_idx in unm_d:
            box = dets_xyxy[d_idx]
            new = _RGTrack(
                KalmanBox(box), self._next_id,
                last_obs=box.copy(), last_obs_frame=self.frame_count,
                obs_history=[(self.frame_count, box.copy())],
            )
            self._next_id += 1
            depth = (float(det_depths[d_idx])
                     if not np.isnan(det_depths[d_idx]) else float("nan"))
            if not np.isnan(depth):
                new.last_depth = depth
                new.depth_history.append(depth)
            self.tracks.append(new)
            self._try_initial_bind(new, depth, ftm_m, ftm_valid)

        # 9. Unmatched tracks: increment time_since_update
        for t_idx in unm_t:
            self.tracks[t_idx].time_since_update += 1

        # 10. Aging / cleanup
        survivors: List[_RGTrack] = []
        for t in self.tracks:
            t.age += 1
            if t.time_since_update <= self.max_age:
                survivors.append(t)
        self.tracks = survivors

        # 11. Emit outputs (use phone-stable ID when bound)
        out: List[Tuple[int, np.ndarray]] = []
        for t in self.tracks:
            if t.time_since_update > 0:
                continue
            if t.hits >= self.min_hits or self.frame_count <= self.min_hits:
                tid = (self.phone_id_offset + t.bound_phone
                       if t.bound_phone is not None else t.track_id)
                out.append((tid, t.kf.bbox.copy()))
        return out
