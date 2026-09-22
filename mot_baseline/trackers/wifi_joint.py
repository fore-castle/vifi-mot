"""WiFi-Enhanced MOT: Joint Phone Assignment + Ghost Pool + FTM Depth Anchor.

Phase 5 tracker — three subsystems on top of Method A's association pipeline:

1. **Global Phone Assignment** — Hungarian joint optimization replaces per-track
   greedy bind/unbind.  All (track, phone) pairs evaluated simultaneously with
   switching penalty and keep bonus.

2. **Ghost Pool** — phone-bound tracks that exceed ``max_age`` without visual
   match are moved to a ghost pool instead of being deleted.  Ghosts don't
   participate in IoU association (avoids Phase 1's drifted-prediction problem).
   They can be reconnected when a new detection matches the ghost's phone and
   spatial location.

3. **FTM Depth Anchor** — ghost tracks receive per-frame FTM updates to keep
   their depth estimate current during occlusion, so reconnection gating is
   accurate and ORU interpolation is better grounded.

Design rationale
----------------
- Phase 1 bridging (max_age=200) failed because long-lived predicted bboxes
  drifted into other persons' territory and corrupted IoU association.
  Ghost pool sidesteps this: ghosts are **invisible** to the cost matrix.
- Phase 2 post-hoc merger reduced IDsw by 24% but lost 2.25pt IDF1 because
  phone binding ≠ physical identity over long gaps.  Ghost pool uses tight
  spatial + FTM gates and short TTL to avoid wrong merges.
- Method A's greedy binding has ordering bias and doesn't handle competition
  between tracks for the same phone.  Hungarian assignment is optimal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .ocsort import _box_center, _cosine
from .utils import KalmanBox, hungarian, iou_xyxy


# ---------------------------------------------------------------------------
# Track data classes
# ---------------------------------------------------------------------------

@dataclass
class _JointTrack:
    kf: KalmanBox
    track_id: int
    last_obs: np.ndarray
    last_obs_frame: int
    age: int = 0
    hits: int = 1
    time_since_update: int = 0
    obs_history: List[Tuple[int, np.ndarray]] = field(default_factory=list)
    # wireless extension
    last_depth: float = float("nan")
    depth_history: List[float] = field(default_factory=list)
    phone_scores: Dict[int, float] = field(default_factory=dict)
    bound_phone: Optional[int] = None
    bound_age: int = 0
    # Phase 5 extension
    reconnect_count: int = 0  # times reconnected from ghost pool


@dataclass
class _GhostTrack:
    """A phone-bound track that became invisible (occluded).

    Kept alive in the ghost pool with FTM depth anchoring.
    Does NOT participate in IoU association.
    """
    kf: KalmanBox                  # Kalman state (predict only, not used for assoc)
    track_id: int                  # original track_id (preserved on reconnect)
    last_obs: np.ndarray           # last matched bbox (xyxy)
    last_obs_frame: int            # frame when last matched
    death_frame: int               # frame when moved to ghost pool
    bound_phone: int               # phone index (always non-None)
    phone_scores: Dict[int, float] # EMA scores at death time
    depth_history: List[float]     # recent depths
    obs_history: List[Tuple[int, np.ndarray]]
    anchor_depth: float = float("nan")  # FTM-anchored depth (updated each frame)
    reconnect_count: int = 0


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class WiFiJointTracker:
    """OC-SORT with Joint Phone Assignment + Ghost Pool + FTM Depth Anchor."""

    def __init__(self,
                 # OC-SORT base params
                 max_age: int = 30,
                 min_hits: int = 3,
                 iou_threshold: float = 0.3,
                 delta_t: int = 3,
                 inertia: float = 0.2,
                 use_ocr: bool = True,
                 # WiFi params (same as Method A defaults)
                 wifi_weight: float = 0.4,
                 depth_sigma: float = 1.5,
                 ema_alpha: float = 0.85,
                 bind_init_threshold: float = 0.7,
                 phone_id_offset: int = 100000,
                 # Global phone assignment params
                 switch_penalty: float = 0.30,
                 keep_bonus: float = 0.20,
                 assign_threshold: float = 0.40,
                 unbind_threshold: float = 0.15,
                 # Ghost pool params
                 ghost_max_age: int = 60,
                 reconnect_compat_gate: float = 0.30,
                 reconnect_pixel_gate: float = 200.0,
                 reconnect_max_gap: int = 60,
                 no_ghost_pool: bool = False,
                 # FTM depth anchor
                 depth_anchor_sigma: float = 1.5,
                 # Phase 6B: learned matching model (optional)
                 matching_model=None,
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
        self.bind_init_threshold = bind_init_threshold
        self.phone_id_offset = phone_id_offset
        # global assignment
        self.switch_penalty = switch_penalty
        self.keep_bonus = keep_bonus
        self.assign_threshold = assign_threshold
        self.unbind_threshold = unbind_threshold
        # ghost pool
        self.ghost_max_age = ghost_max_age
        self.reconnect_compat_gate = reconnect_compat_gate
        self.reconnect_pixel_gate = reconnect_pixel_gate
        self.reconnect_max_gap = reconnect_max_gap
        self.no_ghost_pool = no_ghost_pool
        # FTM depth anchor
        self.depth_anchor_sigma = depth_anchor_sigma
        # Phase 6B: learned matching
        self.matching_model = matching_model
        self._cur_ftm_std_m: Optional[np.ndarray] = None  # set per-frame in update()

        self.tracks: List[_JointTrack] = []
        self.ghost_pool: List[_GhostTrack] = []
        self.frame_count = 0
        self._next_id = 1

    # -- helpers -------------------------------------------------------------

    def _track_direction(self, t: _JointTrack) -> Optional[np.ndarray]:
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

    def _compat(self, depth: float, ftm_m: float, phone_idx: int = -1) -> float:
        """Depth-FTM compatibility in [0,1].

        When matching_model is set, uses learned MLP with 6-dim features.
        Otherwise falls back to Gaussian: e^{-d^2 / 2 sigma^2}.
        """
        if np.isnan(depth) or np.isnan(ftm_m):
            return 0.0
        if self.matching_model is not None and self._cur_ftm_std_m is not None \
                and 0 <= phone_idx < len(self._cur_ftm_std_m):
            import torch
            ftm_std = float(self._cur_ftm_std_m[phone_idx])
            delta = depth - ftm_m
            sigma = self.depth_sigma
            feat = [delta, abs(delta), depth, ftm_m, ftm_std,
                    (delta ** 2) / (2 * sigma ** 2)]
            with torch.no_grad():
                x = torch.tensor([feat], dtype=torch.float32)
                return float(self.matching_model(x).item())
        d = depth - ftm_m
        return float(np.exp(-(d * d) / (2.0 * self.depth_sigma ** 2)))

    def _phones_in_use(self) -> set:
        """Phones currently bound to active tracks."""
        return {t.bound_phone for t in self.tracks if t.bound_phone is not None}

    def _phones_in_ghosts(self) -> set:
        """Phones claimed by ghost tracks."""
        return {g.bound_phone for g in self.ghost_pool}

    # -- association (same as Method A) --------------------------------------

    def _associate(self,
                   dets: np.ndarray,
                   tracks: List[_JointTrack],
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
                    compat = self._compat(det_depths[i], ftm, phone_idx=p)
                    cost[i, j] += self.wifi_weight * (1.0 - compat)

        max_cost = 1.0 - self.iou_threshold + self.inertia + self.wifi_weight
        return hungarian(cost, max_cost=max_cost)

    def _ocr(self,
             dets: np.ndarray,
             tracks: List[_JointTrack],
             unm_d: np.ndarray,
             unm_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Second-round association: IoU with last observation."""
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

    def _oru(self, t: _JointTrack, new_obs: np.ndarray, cur_frame: int) -> None:
        """Observation Recovery Update — interpolate virtual observations."""
        if t.time_since_update <= 0:
            return
        prev = t.last_obs
        n = max(1, cur_frame - t.last_obs_frame)
        for s in range(1, n + 1):
            alpha = s / float(n)
            virt = prev * (1.0 - alpha) + new_obs * alpha
            t.kf.update(virt)

    # -- phone score EMA (same as Method A) ----------------------------------

    def _update_phone_scores(self,
                             track: _JointTrack,
                             depth: float,
                             ftm_m: np.ndarray,
                             ftm_valid: np.ndarray) -> None:
        if np.isnan(depth) or len(ftm_m) == 0:
            return
        new_scores = dict(track.phone_scores)
        for p in range(len(ftm_m)):
            if not ftm_valid[p]:
                if p in new_scores:
                    new_scores[p] *= self.ema_alpha
                continue
            compat = self._compat(depth, float(ftm_m[p]), phone_idx=p)
            prev = new_scores.get(p, 0.0)
            new_scores[p] = self.ema_alpha * prev + (1.0 - self.ema_alpha) * compat
        track.phone_scores = new_scores

    # -- global phone assignment (NEW) ---------------------------------------

    def _global_phone_assign(self,
                             ftm_m: np.ndarray,
                             ftm_valid: np.ndarray) -> None:
        """Hungarian joint phone-track assignment.

        Two-phase approach:
        Phase 1: Unbind tracks whose EMA compat dropped below unbind_threshold
                 (same hysteresis as Method A, prevents spurious unbinds).
        Phase 2: Hungarian assignment for unbound tracks to free phones,
                 with switching penalty for temporal stability.
        """
        if len(self.tracks) == 0 or len(ftm_m) == 0:
            return

        # Phase 1: Unbind (same as Method A)
        for t in self.tracks:
            if t.bound_phone is not None:
                t.bound_age += 1
                ema = t.phone_scores.get(t.bound_phone, 0.0)
                if ema < self.unbind_threshold:
                    t.bound_phone = None
                    t.bound_age = 0

        # Phase 2: Hungarian assignment for unbound tracks
        unbound = [(i, t) for i, t in enumerate(self.tracks)
                   if t.bound_phone is None and not np.isnan(t.last_depth)]
        if not unbound:
            return

        # Available phones (not bound to active tracks, not in ghost pool)
        used = self._phones_in_use()
        ghost_phones = self._phones_in_ghosts()
        free_phones = [p for p in range(len(ftm_m))
                       if ftm_valid[p] and p not in used and p not in ghost_phones]
        if not free_phones:
            return

        nt = len(unbound)
        np_ = len(free_phones)

        # Build cost matrix
        cost = np.full((nt, np_), 10.0, dtype=np.float64)

        for i, (ti, t) in enumerate(unbound):
            for j, p in enumerate(free_phones):
                compat = self._compat(t.last_depth, float(ftm_m[p]), phone_idx=p)
                ema_score = t.phone_scores.get(p, 0.0)
                score = 0.4 * compat + 0.6 * ema_score
                cost[i, j] = -score  # negate for minimization

        # Hungarian assignment
        rows, cols = linear_sum_assignment(cost)

        # Apply assignments
        for r, c in zip(rows, cols):
            ti, t = unbound[r]
            p = free_phones[c]
            score = -cost[r, c]
            if score >= self.assign_threshold:
                t.bound_phone = p
                t.bound_age = 0

    # -- ghost pool management (NEW) -----------------------------------------

    def _demote_to_ghost(self, track: _JointTrack) -> None:
        """Move a phone-bound track to the ghost pool."""
        ghost = _GhostTrack(
            kf=track.kf,
            track_id=track.track_id,
            last_obs=track.last_obs.copy(),
            last_obs_frame=track.last_obs_frame,
            death_frame=self.frame_count,
            bound_phone=track.bound_phone,
            phone_scores=dict(track.phone_scores),
            depth_history=list(track.depth_history),
            obs_history=list(track.obs_history),
            anchor_depth=track.last_depth,
            reconnect_count=track.reconnect_count,
        )
        self.ghost_pool.append(ghost)

    def _update_ghost_depths(self, ftm_m: np.ndarray, ftm_valid: np.ndarray) -> None:
        """FTM depth anchor: update ghost tracks' anchor_depth using FTM.

        Each ghost track continues to receive FTM updates (the phone is still
        broadcasting even though the person is occluded).  We soft-constrain
        the anchor_depth toward the FTM value to keep it current.
        """
        sigma_ftm = self.depth_anchor_sigma
        for ghost in self.ghost_pool:
            p = ghost.bound_phone
            if not ftm_valid[p]:
                continue
            ftm = float(ftm_m[p])
            if np.isnan(ftm):
                continue

            old_depth = ghost.anchor_depth
            if np.isnan(old_depth):
                ghost.anchor_depth = ftm
                continue

            # Simple Kalman-like gain: trust FTM more as occlusion grows
            occlusion_dur = self.frame_count - ghost.death_frame
            # Depth uncertainty grows linearly with occlusion (no visual update)
            sigma_depth = 0.5 + 0.15 * occlusion_dur
            gain = sigma_depth ** 2 / (sigma_depth ** 2 + sigma_ftm ** 2)
            ghost.anchor_depth = old_depth + gain * (ftm - old_depth)

    def _try_ghost_reconnect(self,
                             d_idx: int,
                             dets_xyxy: np.ndarray,
                             det_depths: np.ndarray) -> Optional[_JointTrack]:
        """Try to reconnect an unmatched detection to a ghost track.

        Returns the revived _JointTrack if reconnection succeeds, else None.
        The ghost is removed from the pool on success.
        """
        if len(self.ghost_pool) == 0:
            return None

        depth_d = float(det_depths[d_idx]) if not np.isnan(det_depths[d_idx]) else float("nan")
        if np.isnan(depth_d):
            return None

        det_center = _box_center(dets_xyxy[d_idx])
        best_ghost_idx = None
        best_score = 0.0

        for gi, ghost in enumerate(self.ghost_pool):
            # Gate 1: gap duration
            gap = self.frame_count - ghost.death_frame
            if gap > self.reconnect_max_gap:
                continue

            # Gate 2: phone compatibility (using FTM-anchored depth)
            anchor = ghost.anchor_depth
            compat = self._compat(depth_d, anchor) if not np.isnan(anchor) else 0.0
            if compat < self.reconnect_compat_gate:
                continue

            # Gate 3: spatial proximity (2D pixel distance)
            ghost_center = _box_center(ghost.last_obs)
            pixel_dist = float(np.linalg.norm(det_center - ghost_center))
            if pixel_dist > self.reconnect_pixel_gate:
                continue

            # Score: combine compat and proximity
            spatial_score = 1.0 - min(pixel_dist / self.reconnect_pixel_gate, 1.0)
            score = 0.6 * compat + 0.4 * spatial_score

            if score > best_score:
                best_score = score
                best_ghost_idx = gi

        if best_ghost_idx is None:
            return None

        # Revive the ghost as an active track
        ghost = self.ghost_pool.pop(best_ghost_idx)
        new_det = dets_xyxy[d_idx].copy()

        track = _JointTrack(
            kf=KalmanBox(new_det),  # fresh Kalman from new detection
            track_id=ghost.track_id,  # preserve original ID → no ID switch
            last_obs=new_det,
            last_obs_frame=self.frame_count,
            age=ghost.age if hasattr(ghost, 'age') else 0,
            hits=max(ghost.reconnect_count + 2, 3),  # ensure min_hits satisfied
            time_since_update=0,
            obs_history=ghost.obs_history[-10:] + [(self.frame_count, new_det)],
            last_depth=depth_d,
            depth_history=ghost.depth_history[-10:] + [depth_d],
            phone_scores=ghost.phone_scores,
            bound_phone=ghost.bound_phone,
            bound_age=0,
            reconnect_count=ghost.reconnect_count + 1,
        )

        return track

    # -- initial bind (similar to Method A, uses global context) -------------

    def _try_initial_bind(self,
                          track: _JointTrack,
                          depth: float,
                          ftm_m: np.ndarray,
                          ftm_valid: np.ndarray) -> None:
        """Bind a freshly-created track if a phone matches very strongly."""
        if np.isnan(depth):
            return
        in_use = self._phones_in_use() | self._phones_in_ghosts()
        best_p, best_compat = None, 0.0
        for p in range(len(ftm_m)):
            if not ftm_valid[p] or p in in_use:
                continue
            c = self._compat(depth, float(ftm_m[p]), phone_idx=p)
            if c > best_compat:
                best_compat = c
                best_p = p
        if best_p is not None and best_compat >= self.bind_init_threshold:
            track.bound_phone = best_p
            track.phone_scores[best_p] = best_compat

    # -- main step -----------------------------------------------------------

    def update(self,
               dets_xyxy: np.ndarray,
               det_depths: np.ndarray,
               scores: Optional[np.ndarray],
               ftm_m: np.ndarray,
               ftm_valid: np.ndarray,
               ftm_std_m: Optional[np.ndarray] = None) -> List[Tuple[int, np.ndarray]]:
        """Step one frame.

        Parameters
        ----------
        dets_xyxy   : (D, 4) detection boxes
        det_depths  : (D,)   detection depth in meters (NaN allowed)
        scores      : (D,)   detection scores (unused, for API compat)
        ftm_m       : (P,)   FTM range in meters per phone, NaN if missing
        ftm_valid   : (P,)   bool mask
        ftm_std_m   : (P,)   FTM std in meters per phone (optional, for learned matching)
        """
        self.frame_count += 1
        self._cur_ftm_std_m = ftm_std_m  # Phase 6B: for learned matching

        # 1. Kalman predict (active tracks only)
        predicted = [t.kf.predict() for t in self.tracks]
        if self.tracks:
            pred_arr = np.stack(predicted, axis=0)
        else:
            pred_arr = np.zeros((0, 4))

        # 2. FTM depth anchor (ghost tracks only)
        if not self.no_ghost_pool and len(self.ghost_pool) > 0 and len(ftm_m) > 0:
            self._update_ghost_depths(ftm_m, ftm_valid)

        # 3. First-round association (active tracks only — ghosts are invisible)
        matched, unm_d, unm_t = self._associate(dets_xyxy, self.tracks, pred_arr,
                                                det_depths, ftm_m, ftm_valid)

        # 4. OCR second round
        extras, unm_d, unm_t = self._ocr(dets_xyxy, self.tracks, unm_d, unm_t)
        if len(extras):
            matched = np.vstack([matched, extras])

        # 5. Apply matches & update phone scores
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

        # 6. Global phone assignment (replaces greedy bind/unbind)
        if len(ftm_m) > 0:
            self._global_phone_assign(ftm_m, ftm_valid)

        # 7. Ghost pool reconnection for still-unmatched detections
        reconnected_dets: List[int] = []
        if not self.no_ghost_pool:
            for d_idx in unm_d:
                revived = self._try_ghost_reconnect(d_idx, dets_xyxy, det_depths)
                if revived is not None:
                    # Update phone scores with new detection depth
                    depth = float(det_depths[d_idx]) if not np.isnan(det_depths[d_idx]) else float("nan")
                    self._update_phone_scores(revived, depth, ftm_m, ftm_valid)
                    self.tracks.append(revived)
                    reconnected_dets.append(d_idx)

        # Remove reconnected detections from unm_d
        if reconnected_dets:
            reconnected_set = set(reconnected_dets)
            unm_d = np.array([d for d in unm_d if d not in reconnected_set], dtype=int)

        # 8. Unmatched detections → new tracks
        for d_idx in unm_d:
            box = dets_xyxy[d_idx]
            new = _JointTrack(KalmanBox(box), self._next_id,
                              last_obs=box.copy(),
                              last_obs_frame=self.frame_count,
                              obs_history=[(self.frame_count, box.copy())])
            self._next_id += 1
            depth = float(det_depths[d_idx]) if not np.isnan(det_depths[d_idx]) else float("nan")
            if not np.isnan(depth):
                new.last_depth = depth
                new.depth_history.append(depth)
            self.tracks.append(new)
            # Try immediate bind
            self._try_initial_bind(new, depth, ftm_m, ftm_valid)

        # 9. Unmatched tracks → increment time_since_update
        for t_idx in unm_t:
            self.tracks[t_idx].time_since_update += 1

        # 10. Aging / cleanup / demotion to ghost pool
        survivors: List[_JointTrack] = []
        for t in self.tracks:
            t.age += 1
            if t.time_since_update <= self.max_age:
                survivors.append(t)
            elif not self.no_ghost_pool and t.bound_phone is not None:
                # Phone-bound track exceeded max_age → demote to ghost pool
                # Only if this phone isn't already claimed by another ghost
                ghost_phones = self._phones_in_ghosts()
                if t.bound_phone not in ghost_phones:
                    self._demote_to_ghost(t)
            # else: unbound track dies normally
        self.tracks = survivors

        # 11. Prune expired ghosts
        self.ghost_pool = [
            g for g in self.ghost_pool
            if (self.frame_count - g.death_frame) <= self.ghost_max_age
        ]

        # 12. Emit outputs
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
