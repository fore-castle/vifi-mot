"""WiFi + 3D Spatial Radio Prior + EKF Kalman update OC-SORT (Phase 1 / Phase 2).

Differences vs ``wifi_ocsort.WiFiOCSortTracker``
-----------------------------------------------
1. **3D Mahalanobis spatial compatibility** replaces the scalar
   ``exp(-(depth - r_ftm)^2 / 2 sigma^2)`` Gaussian. We back-project every
   detection bbox center to camera-frame meters and compare it to a phone-state
   estimate that lives on the FTM sphere around a learned AP location. The
   covariance is directional (small along the radial direction, large in the
   tangent plane), giving us proper anisotropic uncertainty.

2. **WiFi cost applies to ALL tracks**, not only bound ones. Because the
   phone-state covariance already encodes huge tangential uncertainty, the
   compatibility automatically drops to ~0 for tracks that are nowhere near the
   FTM sphere -- so unbound tracks pay no penalty.

3. **EKF wireless update**. After the visual update, every track that is bound
   to a phone receives a second Kalman update using the FTM range as a
   nonlinear scalar observation. The Jacobian is derived in the Phase 1 plan
   (``action1_spatial_radio_prior_plan.md`` Sec 2.3). The observation noise
   ``R_wire`` is per-frame adaptive (sigma_ftm^2 + sigma_depth^2 + sigma_ap^2).

4. **Occlusion bridging (Direction β-lite)**. When a bound track has
   ``time_since_update > 0`` (i.e. it failed visual association this frame),
   we still perform the wireless update -- this keeps the predicted bbox
   anchored to the FTM measurement during occlusions, which Phase 0 identified
   as the dominant IDsw driver.

5. **AP position warm-up**. The AP location is not given by the dataset. We
   collect (bbox-center 3D, FTM) pairs from confidently bound tracks during the
   first ``ap_warmup_frames`` frames and solve a least-squares problem to
   localise the AP. Until convergence we fall back to scalar compatibility.

The bind/unbind machinery is borrowed from ``WiFiOCSortTracker`` but uses the
3D Mahalanobis compatibility instead of the scalar gaussian when the AP is
known.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .ocsort import _box_center, _cosine
from .utils import KalmanBox, hungarian, iou_xyxy, xyxy_to_usr
from .spatial_projector import (CX, CY, FX, FY, APEstimate, PhoneState,
                                adaptive_R_wire, estimate_ap_position,
                                estimate_phone_state, mahalanobis_compat,
                                project_bbox_center)


# ---------------------------------------------------------------------------
# Track data class
# ---------------------------------------------------------------------------

@dataclass
class _SpatialTrack:
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
    last_pos3d: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))
    depth_history: List[float] = field(default_factory=list)
    phone_scores: Dict[int, float] = field(default_factory=dict)
    bound_phone: Optional[int] = None
    bound_age: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scalar_compat(depth: float, r_ftm: float, sigma: float) -> float:
    """Fallback scalar gaussian, identical to the legacy method-A."""
    if not (np.isfinite(depth) and np.isfinite(r_ftm)):
        return 0.0
    diff = depth - r_ftm
    return float(np.exp(-(diff * diff) / (2.0 * sigma * sigma)))


# ---------------------------------------------------------------------------
# Main tracker
# ---------------------------------------------------------------------------

class WiFiSpatialOCSortTracker:
    """OC-SORT with 3D Spatial Radio Prior + EKF wireless update."""

    def __init__(self,
                 # OC-SORT base params
                 max_age: int = 30,
                 min_hits: int = 3,
                 iou_threshold: float = 0.3,
                 delta_t: int = 3,
                 inertia: float = 0.2,
                 use_ocr: bool = True,
                 # WiFi / spatial params
                 spatial_weight: float = 0.4,
                 fallback_sigma: float = 1.5,
                 ema_alpha: float = 0.85,
                 bind_threshold: float = 0.6,
                 unbind_threshold: float = 0.15,
                 bind_init_threshold: float = 0.7,
                 phone_id_offset: int = 100000,
                 # 3D / EKF params
                 sigma_perp: float = 2.5,
                 sigma_radial: float = 1.5,
                 sigma_ap: float = 0.5,
                 sigma_ftm_floor: float = 0.4,
                 gate_chi2: float = 9.21,
                 ap_warmup_frames: int = 50,
                 ap_warmup_min_pairs: int = 20,
                 use_ekf_update: bool = True,
                 bridging_max_age: int = 200,
                 compat_mode: str = "scalar",   # "scalar" or "maha"
                 ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.delta_t = delta_t
        self.inertia = inertia
        self.use_ocr = use_ocr
        # wireless
        self.spatial_weight = spatial_weight
        self.fallback_sigma = fallback_sigma
        self.ema_alpha = ema_alpha
        self.bind_threshold = bind_threshold
        self.unbind_threshold = unbind_threshold
        self.bind_init_threshold = bind_init_threshold
        self.phone_id_offset = phone_id_offset
        # 3D / EKF
        self.sigma_perp = sigma_perp
        self.sigma_radial = sigma_radial
        self.sigma_ap = sigma_ap
        self.sigma_ftm_floor = sigma_ftm_floor
        self.gate_chi2 = gate_chi2
        self.ap_warmup_frames = ap_warmup_frames
        self.ap_warmup_min_pairs = ap_warmup_min_pairs
        self.use_ekf_update = use_ekf_update
        self.bridging_max_age = bridging_max_age
        if compat_mode not in ("scalar", "maha"):
            raise ValueError(f"compat_mode must be 'scalar' or 'maha', got {compat_mode!r}")
        self.compat_mode = compat_mode

        self.tracks: List[_SpatialTrack] = []
        self.frame_count = 0
        self._next_id = 1

        # AP estimation state ---------------------------------------------
        self._ap_estimate: Optional[APEstimate] = None
        self._warmup_pos: List[np.ndarray] = []      # camera-frame XYZ
        self._warmup_ftm: List[float] = []           # meters

    # -- helpers -------------------------------------------------------------

    @property
    def ap_known(self) -> bool:
        return self._ap_estimate is not None

    def _track_direction(self, t: _SpatialTrack) -> Optional[np.ndarray]:
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

    def _phones_in_use(self, exclude: Optional[_SpatialTrack] = None) -> set:
        return {t.bound_phone for t in self.tracks
                if t.bound_phone is not None and t is not exclude}

    # ------------------------------------------------------------------
    # AP warm-up
    # ------------------------------------------------------------------

    def _try_estimate_ap(self) -> None:
        if self.ap_known:
            return
        if (self.frame_count < self.ap_warmup_frames or
                len(self._warmup_pos) < self.ap_warmup_min_pairs):
            return
        positions = np.stack(self._warmup_pos, axis=0)
        ftms = np.asarray(self._warmup_ftm, dtype=np.float64)
        est = estimate_ap_position(positions, ftms)
        if est.converged and est.rms_residual < 5.0:
            self._ap_estimate = est

    def _build_phone_states(self,
                            ftm_m: np.ndarray,
                            ftm_valid: np.ndarray) -> List[Optional[PhoneState]]:
        """Per-phone PhoneState (or None if invalid / no AP yet)."""
        if not self.ap_known:
            return [None] * len(ftm_m)
        p_ap = self._ap_estimate.position
        out: List[Optional[PhoneState]] = []
        # Build anchors from current bound tracks (helps disambiguate the sphere).
        anchors: Dict[int, np.ndarray] = {}
        for t in self.tracks:
            if t.bound_phone is None:
                continue
            if np.all(np.isfinite(t.last_pos3d)):
                anchors[t.bound_phone] = t.last_pos3d
        for p in range(len(ftm_m)):
            if not ftm_valid[p]:
                out.append(None)
                continue
            r = float(ftm_m[p])
            # Use sigma_radial (default 1.5m) for the Mahalanobis covariance: this
            # matches Phase 0's empirical median |depth-FTM error| ≈ 1.38m. The
            # tighter sigma_ftm_floor (0.4m) is reserved for adaptive_R_wire in
            # the EKF update where small innovations need to be trusted.
            sigma = self.sigma_radial
            anchor = anchors.get(p, None)
            st = estimate_phone_state(p_ap, r, sigma,
                                       anchor_xyz=anchor,
                                       sigma_perp=self.sigma_perp)
            out.append(st)
        return out

    # ------------------------------------------------------------------
    # Compatibility (3D Mahalanobis when AP known, else scalar fallback)
    # ------------------------------------------------------------------

    def _pair_compat(self,
                     det_pos3d: np.ndarray,
                     det_depth: float,
                     phone_state: Optional[PhoneState],
                     r_ftm: float) -> float:
        if (self.compat_mode == "maha" and self.ap_known
                and phone_state is not None and phone_state.valid):
            compat, _ = mahalanobis_compat(det_pos3d, phone_state,
                                            gate_chi2=self.gate_chi2)
            return compat
        # Fallback / default: legacy scalar gaussian (matches Method A)
        return _scalar_compat(det_depth, r_ftm, self.fallback_sigma)

    # ------------------------------------------------------------------
    # Association
    # ------------------------------------------------------------------

    def _associate(self,
                   dets: np.ndarray,
                   tracks: List[_SpatialTrack],
                   preds: np.ndarray,
                   det_depths: np.ndarray,
                   det_pos3d: np.ndarray,
                   ftm_m: np.ndarray,
                   ftm_valid: np.ndarray,
                   phone_states: List[Optional[PhoneState]],
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

        # Spatial / WiFi compatibility cost on bound tracks
        if self.spatial_weight > 0 and len(ftm_m) > 0:
            for j, t in enumerate(tracks):
                if t.bound_phone is None:
                    continue
                p = t.bound_phone
                if not ftm_valid[p]:
                    continue
                r = float(ftm_m[p])
                state = phone_states[p]
                for i in range(len(dets)):
                    if np.isnan(det_depths[i]):
                        continue
                    compat = self._pair_compat(det_pos3d[i], det_depths[i],
                                               state, r)
                    cost[i, j] += self.spatial_weight * (1.0 - compat)

        max_cost = 1.0 - self.iou_threshold + self.inertia + self.spatial_weight
        return hungarian(cost, max_cost=max_cost)

    def _ocr(self,
             dets: np.ndarray,
             tracks: List[_SpatialTrack],
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

    def _oru(self, t: _SpatialTrack, new_obs: np.ndarray, cur_frame: int) -> None:
        if t.time_since_update <= 0:
            return
        prev = t.last_obs
        n = max(1, cur_frame - t.last_obs_frame)
        for s in range(1, n + 1):
            alpha = s / float(n)
            virt = prev * (1.0 - alpha) + new_obs * alpha
            t.kf.update(virt)

    # ------------------------------------------------------------------
    # EKF wireless update
    # ------------------------------------------------------------------

    def _wireless_update(self,
                         track: _SpatialTrack,
                         r_ftm: float,
                         sigma_ftm: float) -> None:
        """EKF update: scalar observation z = ||p_track - p_AP|| - r_ftm."""
        if not self.ap_known:
            return
        if not np.isfinite(track.last_depth) or track.last_depth <= 0.0:
            return
        if not (np.isfinite(r_ftm) and r_ftm > 0.0):
            return
        p_ap = self._ap_estimate.position
        kf = track.kf
        x = kf.x
        u = float(x[0])
        v = float(x[1])
        depth = track.last_depth

        # 3D position predicted from (u, v, last_depth)
        X = (u - CX) / FX * depth
        Y = -(v - CY) / FY * depth
        Z = depth

        dx = X - p_ap[0]
        dy = Y - p_ap[1]
        dz = Z - p_ap[2]
        d_pred = float(np.sqrt(dx * dx + dy * dy + dz * dz))
        if d_pred < 1e-3:
            return
        innovation = float(r_ftm - d_pred)

        # Reject grossly inconsistent observations via 3-sigma gate using R_wire
        R_w = adaptive_R_wire(sigma_ftm, depth,
                              sigma_ap=self.sigma_ap,
                              sigma_ftm_floor=self.sigma_ftm_floor)
        if abs(innovation) > 3.0 * np.sqrt(R_w) + 1.0:
            # Beyond 3-sigma plus 1 m hard slack -> skip update.
            return

        # Jacobian H_wire (1x7); only u and v entries are non-zero.
        H = np.zeros((1, 7), dtype=np.float64)
        H[0, 0] = (dx * depth) / (d_pred * FX)
        H[0, 1] = (-dy * depth) / (d_pred * FY)

        P = kf.P
        S_mat = H @ P @ H.T + R_w                      # (1,1)
        S = float(S_mat[0, 0])
        if S <= 1e-9:
            return
        K = (P @ H.T) / S                              # 7x1

        kf.x = (x + (K.flatten() * innovation)).astype(np.float64)
        I_KH = np.eye(7) - K @ H
        kf.P = I_KH @ P @ I_KH.T + (K @ K.T) * R_w     # Joseph form

    # ------------------------------------------------------------------
    # Phone score update
    # ------------------------------------------------------------------

    def _update_phone_scores(self,
                             track: _SpatialTrack,
                             det_pos3d: np.ndarray,
                             det_depth: float,
                             ftm_m: np.ndarray,
                             ftm_valid: np.ndarray,
                             phone_states: List[Optional[PhoneState]]) -> None:
        if len(ftm_m) == 0:
            return
        new_scores = dict(track.phone_scores)
        for p in range(len(ftm_m)):
            if not ftm_valid[p]:
                if p in new_scores:
                    new_scores[p] *= self.ema_alpha
                continue
            r = float(ftm_m[p])
            compat = self._pair_compat(det_pos3d, det_depth,
                                       phone_states[p] if self.ap_known else None,
                                       r)
            prev = new_scores.get(p, 0.0)
            new_scores[p] = self.ema_alpha * prev + (1.0 - self.ema_alpha) * compat
        track.phone_scores = new_scores

    def _try_initial_bind(self,
                          track: _SpatialTrack,
                          det_pos3d: np.ndarray,
                          det_depth: float,
                          ftm_m: np.ndarray,
                          ftm_valid: np.ndarray,
                          phone_states: List[Optional[PhoneState]]) -> None:
        if np.isnan(det_depth):
            return
        in_use = self._phones_in_use()
        best_p, best_compat = None, 0.0
        for p in range(len(ftm_m)):
            if not ftm_valid[p] or p in in_use:
                continue
            r = float(ftm_m[p])
            c = self._pair_compat(det_pos3d, det_depth,
                                   phone_states[p] if self.ap_known else None,
                                   r)
            if c > best_compat:
                best_compat = c
                best_p = p
        if best_p is not None and best_compat >= self.bind_init_threshold:
            track.bound_phone = best_p
            track.phone_scores[best_p] = best_compat

    def _resolve_phone_binds(self) -> None:
        # unbind weak bindings
        for t in self.tracks:
            if t.bound_phone is not None:
                t.bound_age += 1
                score = t.phone_scores.get(t.bound_phone, 0.0)
                if score < self.unbind_threshold:
                    t.bound_phone = None
                    t.bound_age = 0

        bound_phones = self._phones_in_use()
        candidates: List[Tuple[float, int, int]] = []
        for ti, t in enumerate(self.tracks):
            if t.bound_phone is not None:
                continue
            for p, s in t.phone_scores.items():
                if p in bound_phones:
                    continue
                if s >= self.bind_threshold:
                    candidates.append((s, ti, p))
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

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    def update(self,
               dets_xyxy: np.ndarray,
               det_depths: np.ndarray,
               scores: Optional[np.ndarray],
               ftm_m: np.ndarray,
               ftm_valid: np.ndarray,
               ftm_std_m: Optional[np.ndarray] = None,
               ) -> List[Tuple[int, np.ndarray]]:
        """Step one frame.

        Parameters
        ----------
        dets_xyxy   : (D, 4) detection boxes
        det_depths  : (D,)   detection depth in meters (NaN allowed)
        scores      : (D,)   detection scores (compat with API; unused)
        ftm_m       : (P,)   FTM range in meters per phone, NaN if missing
        ftm_valid   : (P,)   bool mask
        ftm_std_m   : (P,)   optional FTM std in meters per phone
        """
        self.frame_count += 1

        # Pre-compute detection 3D positions
        D = len(dets_xyxy)
        det_pos3d = np.full((D, 3), np.nan, dtype=np.float64)
        for i in range(D):
            det_pos3d[i] = project_bbox_center(dets_xyxy[i], det_depths[i])

        # Predict
        predicted = [t.kf.predict() for t in self.tracks]
        pred_arr = np.stack(predicted, axis=0) if self.tracks else np.zeros((0, 4))

        # Phone states (need bound tracks' last_pos3d as anchors -> compute before)
        phone_states = self._build_phone_states(ftm_m, ftm_valid)

        # First-round association
        matched, unm_d, unm_t = self._associate(dets_xyxy, self.tracks, pred_arr,
                                                det_depths, det_pos3d,
                                                ftm_m, ftm_valid, phone_states)

        # OCR
        extras, unm_d, unm_t = self._ocr(dets_xyxy, self.tracks, unm_d, unm_t)
        if len(extras):
            matched = np.vstack([matched, extras])

        # Apply matches
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
            pos3d = det_pos3d[d_idx]
            if not np.isnan(depth):
                track.last_depth = depth
                track.last_pos3d = pos3d
                track.depth_history.append(depth)
                if len(track.depth_history) > 50:
                    track.depth_history = track.depth_history[-50:]
            self._update_phone_scores(track, pos3d, depth,
                                       ftm_m, ftm_valid, phone_states)

            # EKF wireless update for bound tracks (Phase 2)
            if self.use_ekf_update and track.bound_phone is not None:
                p = track.bound_phone
                if ftm_valid[p]:
                    sigma_ftm = (float(ftm_std_m[p]) if ftm_std_m is not None
                                 else self.sigma_ftm_floor)
                    self._wireless_update(track, float(ftm_m[p]), sigma_ftm)

        # Resolve binds / unbinds
        self._resolve_phone_binds()

        # New tracks
        for d_idx in unm_d:
            box = dets_xyxy[d_idx]
            new = _SpatialTrack(KalmanBox(box), self._next_id,
                                last_obs=box.copy(),
                                last_obs_frame=self.frame_count,
                                obs_history=[(self.frame_count, box.copy())])
            self._next_id += 1
            depth = float(det_depths[d_idx]) if not np.isnan(det_depths[d_idx]) else float("nan")
            pos3d = det_pos3d[d_idx]
            if not np.isnan(depth):
                new.last_depth = depth
                new.last_pos3d = pos3d
                new.depth_history.append(depth)
            self.tracks.append(new)
            self._try_initial_bind(new, pos3d, depth,
                                   ftm_m, ftm_valid, phone_states)

        # Unmatched tracks (visual aging only)
        for t_idx in unm_t:
            track = self.tracks[t_idx]
            track.time_since_update += 1

        # Direction β — wireless tracklet bridging: every bound track that did
        # NOT receive a visual match this frame gets a second EKF wireless
        # update. This keeps the KF state anchored to the FTM measurement while
        # the phone holder is occluded, so when they reappear they match back
        # to the SAME track and no IDsw is generated.
        if self.use_ekf_update:
            matched_track_indices = {int(t_idx) for _, t_idx in matched}
            for ti, track in enumerate(self.tracks):
                if ti in matched_track_indices:
                    continue
                if track.bound_phone is None:
                    continue
                p = track.bound_phone
                if not (0 <= p < len(ftm_m) and ftm_valid[p]):
                    continue
                sigma_ftm = (float(ftm_std_m[p]) if ftm_std_m is not None
                             else self.sigma_ftm_floor)
                self._wireless_update(track, float(ftm_m[p]), sigma_ftm)

        # Collect warm-up observations for AP estimation
        if not self.ap_known:
            for d_idx, t_idx in matched:
                track = self.tracks[t_idx]
                if track.bound_phone is None:
                    continue
                p = track.bound_phone
                if not ftm_valid[p]:
                    continue
                pos3d = det_pos3d[d_idx]
                if np.all(np.isfinite(pos3d)):
                    self._warmup_pos.append(pos3d.copy())
                    self._warmup_ftm.append(float(ftm_m[p]))
            self._try_estimate_ap()

        # Aging / cleanup — Direction β: bound tracks live up to
        # bridging_max_age frames of occlusion; unbound tracks use max_age.
        survivors: List[_SpatialTrack] = []
        for t in self.tracks:
            t.age += 1
            limit = self.bridging_max_age if t.bound_phone is not None else self.max_age
            if t.time_since_update <= limit:
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
