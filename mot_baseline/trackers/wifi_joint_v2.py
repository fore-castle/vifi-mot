"""WiFiJointTracker V2 — three paper-inspired, training-free extensions.

T1 Trajectory-window correlation (MVTrajecter E6 / ViFiCon / SATM):
   phone-track binding score gains a derivative-correlation term
   rho' = Pearson(diff depth_i, diff ftm_p) over a sliding window.
   Radial-velocity sign disambiguates close crossings where instantaneous
   |depth - ftm| is equal for both people.

T2 Uncertainty-normalized compatibility (UCMCTrack Eq.8 / RCTDistill / WhereArtThou):
   S(i,p) = sigma0^2 + sigma_d(depth)^2 + sigma_f(phone)^2
   compat = sqrt(S_ref/S) * exp(-delta^2 / (2 S))   (likelihood-ratio normalized)
   Large measurement noise no longer masquerades as mismatch, and clean
   measurements enjoy tighter gates.

T3 Occlusion-aware cost modulation + depth-update gating (OA-SORT E1):
   Oc_j = overlapped-area fraction of track j by nearer tracks.
   (a) association cost:  (1 - tau*Oc)*(1-IoU) ... + w*(1 + kappa*Oc)*(1-compat)
   (b) depth EMA gate: d_new = d_old + (1-Oc)*(z - d_old)  -- prevents the
       front person's depth from contaminating an occluded track.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .ocsort import _box_center, _cosine
from .utils import KalmanBox, hungarian, iou_xyxy
from .wifi_joint import WiFiJointTracker, _JointTrack
from scipy.optimize import linear_sum_assignment


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return 0.0
    sa, sb = a.std(), b.std()
    if sa < 1e-6 or sb < 1e-6:
        return 0.0
    return float(np.clip(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb), -1, 1))


class WiFiJointTrackerV2(WiFiJointTracker):

    def __init__(self,
                 # T1: trajectory correlation
                 traj_lambda: float = 0.0,       # 0 = off
                 traj_window: int = 30,          # frames (~3s at 10Hz)
                 traj_min_common: int = 8,
                 # T2: adaptive sigma
                 adaptive_sigma: bool = False,
                 sigma0: float = 0.9,            # base matching noise floor (m)
                 zed_a: float = 0.01,            # sigma_d = zed_a * depth^2
                 lr_norm: bool = False,          # multiply sqrt(S_ref/S) prefactor
                 # T3: occlusion-aware
                 occl_tau: float = 0.0,          # 0 = off (IoU de-weight)
                 occl_kappa: float = 0.0,        # wifi boost under occlusion
                 occl_depth_gate: bool = False,  # gate depth EMA by (1-Oc)
                 # T5: risk-deferred binding (U2MOT verify-refine)
                 bind_margin: float = 0.0,       # 0 = off
                 # T8: SPRT sequential binding (Wald test on MLP odds)
                 sprt: bool = False,
                 llr_clip: float = 2.0,          # per-frame |LLR| cap
                 llr_decay: float = 0.95,        # forgetting factor
                 bind_A: float = 3.0,            # bind when L > A  (~e^3=20:1 odds)
                 unbind_B: float = -2.0,         # unbind when L < B
                 # T11: per-phone wifi-weight gating by calibrated sigma-hat
                 wifi_gate: bool = False,
                 sigma0_gate: float = 0.8,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.traj_lambda = traj_lambda
        self.traj_window = traj_window
        self.traj_min_common = traj_min_common
        self.adaptive_sigma = adaptive_sigma
        self.sigma0 = sigma0
        self.zed_a = zed_a
        self.lr_norm = lr_norm
        self.occl_tau = occl_tau
        self.occl_kappa = occl_kappa
        self.occl_depth_gate = occl_depth_gate
        self.bind_margin = bind_margin
        self.sprt = sprt
        self.llr_clip = llr_clip
        self.llr_decay = llr_decay
        self.bind_A = bind_A
        self.unbind_B = unbind_B
        self.wifi_gate = wifi_gate
        self.sigma0_gate = sigma0_gate
        self._sigma_hat: Optional[np.ndarray] = None

        # frame-aligned histories
        self._phone_hist: Dict[int, List[Tuple[int, float]]] = {}
        self._cur_occ: np.ndarray = np.zeros(0)

    # -- T2: uncertainty-normalized compat ------------------------------------

    def _compat(self, depth: float, ftm_m: float, phone_idx: int = -1) -> float:
        if self.matching_model is not None and getattr(self, "dual_matching", False):
            if np.isnan(depth) or np.isnan(ftm_m):
                return 0.0
            mu_h_arr = getattr(self, "_mu_h", None)
            if (mu_h_arr is None or self._sigma_hat is None
                    or self._cur_ftm_std_m is None
                    or not (0 <= phone_idx < len(mu_h_arr))):
                return 0.0
            mu_h = float(mu_h_arr[phone_idx])
            sig = float(self._sigma_hat[phone_idx])
            if not np.isfinite(mu_h):
                mu_h = ftm_m
            if not np.isfinite(sig) or sig <= 0:
                sig = 1.0
            std_rep = float(self._cur_ftm_std_m[phone_idx])
            import torch
            dc = depth - ftm_m
            dh = depth - mu_h
            feat = [dc, dh, abs(dc), abs(dh), depth, ftm_m, mu_h, std_rep, sig,
                    float(np.log(max(sig, 1e-3))),
                    (dc ** 2) / (2 * 1.3 ** 2),
                    (dh ** 2) / (2 * max(sig, 1e-3) ** 2)]
            with torch.no_grad():
                x = torch.tensor([feat], dtype=torch.float32)
                return float(self.matching_model(x).item())
        if self.matching_model is not None and getattr(self, "hetero_matching", False):
            if np.isnan(depth) or np.isnan(ftm_m):
                return 0.0
            if self._cur_ftm_std_m is None or not (0 <= phone_idx < len(self._cur_ftm_std_m)):
                return 0.0
            import torch
            sig = float(self._cur_ftm_std_m[phone_idx])
            if not np.isfinite(sig) or sig <= 0:
                sig = 1.0
            delta = depth - ftm_m
            feat = [delta, abs(delta), depth, ftm_m, sig,
                    (delta ** 2) / (2 * sig ** 2), float(np.log(sig))]
            with torch.no_grad():
                x = torch.tensor([feat], dtype=torch.float32)
                return float(self.matching_model(x).item())
        if self.matching_model is not None or not self.adaptive_sigma:
            return super()._compat(depth, ftm_m, phone_idx=phone_idx)
        if np.isnan(depth) or np.isnan(ftm_m):
            return 0.0
        sigma_d = self.zed_a * depth * depth
        sigma_f = 0.0
        if self._cur_ftm_std_m is not None and 0 <= phone_idx < len(self._cur_ftm_std_m):
            sigma_f = float(self._cur_ftm_std_m[phone_idx])
        S = self.sigma0 ** 2 + sigma_d ** 2 + sigma_f ** 2
        d = depth - ftm_m
        val = np.exp(-(d * d) / (2.0 * S))
        if self.lr_norm:
            # UCMC Eq.8 ln|S| analogue — penalizes large uncertainty, but
            # shifts the whole compat distribution down (breaks EMA thresholds)
            val *= np.sqrt(self.sigma0 ** 2 / S)
        return float(min(val, 1.0))

    # -- T1: trajectory correlation --------------------------------------------

    def _traj_corr(self, t: _JointTrack, p: int) -> float:
        """Pearson corr of first differences of (track depth, phone ftm)."""
        dh = getattr(t, "depth_frames", None)
        ph = self._phone_hist.get(p)
        if not dh or not ph:
            return 0.0
        lo = self.frame_count - self.traj_window
        dmap = {f: v for f, v in dh if f >= lo}
        pmap = {f: v for f, v in ph if f >= lo}
        common = sorted(dmap.keys() & pmap.keys())
        if len(common) < self.traj_min_common:
            return 0.0
        d = np.array([dmap[f] for f in common])
        r = np.array([pmap[f] for f in common])
        return _pearson(np.diff(d), np.diff(r))

    # -- T8: SPRT sequential binding -------------------------------------------

    def _sprt_assign(self, ftm_m, ftm_valid) -> None:
        """Wald sequential probability-ratio binding.

        Per (track, phone): L <- decay*L + clip(log p/(1-p))  where p is the
        instantaneous match probability (MatchingMLP or Gaussian compat).
        Bind when L > A (strong accumulated evidence), unbind when L < B.
        Replaces EMA + double-threshold hysteresis with a statistically
        grounded sequential test (error rates alpha ~ e^-A, beta ~ e^B).
        """
        eps = 1e-4
        # 1. accumulate LLR for all (track, phone) pairs
        for t in self.tracks:
            llr = getattr(t, "llr_scores", None)
            if llr is None:
                llr = {}
                t.llr_scores = llr
            if np.isnan(t.last_depth):
                continue
            for p in range(len(ftm_m)):
                if not ftm_valid[p] or np.isnan(ftm_m[p]):
                    llr[p] = self.llr_decay * llr.get(p, 0.0)
                    continue
                pr = self._compat(t.last_depth, float(ftm_m[p]), phone_idx=p)
                pr = min(max(pr, eps), 1 - eps)
                inc = float(np.clip(np.log(pr / (1 - pr)),
                                    -self.llr_clip, self.llr_clip))
                llr[p] = self.llr_decay * llr.get(p, 0.0) + inc

        # 2. unbind: accumulated evidence collapsed
        for t in self.tracks:
            if t.bound_phone is not None:
                t.bound_age += 1
                L = getattr(t, "llr_scores", {}).get(t.bound_phone, 0.0)
                if L < self.unbind_B:
                    t.bound_phone = None
                    t.bound_age = 0

        # 3. bind: Hungarian on LLR for unbound tracks x free phones
        unbound = [(i, t) for i, t in enumerate(self.tracks)
                   if t.bound_phone is None and not np.isnan(t.last_depth)]
        if not unbound:
            return
        used = self._phones_in_use()
        ghost_phones = self._phones_in_ghosts()
        free_phones = [p for p in range(len(ftm_m))
                       if ftm_valid[p] and p not in used and p not in ghost_phones]
        if not free_phones:
            return
        cost = np.full((len(unbound), len(free_phones)), 1e3)
        for i, (ti, t) in enumerate(unbound):
            llr = getattr(t, "llr_scores", {})
            for j, p in enumerate(free_phones):
                cost[i, j] = -llr.get(p, 0.0)
        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols):
            if -cost[r, c] > self.bind_A:
                ti, t = unbound[r]
                t.bound_phone = free_phones[c]
                t.bound_age = 0

    def _global_phone_assign(self, ftm_m, ftm_valid) -> None:
        if len(self.tracks) == 0 or len(ftm_m) == 0:
            return
        if self.sprt:
            self._sprt_assign(ftm_m, ftm_valid)
            return
        # Phase 1: unbind (with T1: correlation can veto an unbind)
        for t in self.tracks:
            if t.bound_phone is not None:
                t.bound_age += 1
                ema = t.phone_scores.get(t.bound_phone, 0.0)
                if ema < self.unbind_threshold:
                    if self.traj_lambda > 0:
                        rho = self._traj_corr(t, t.bound_phone)
                        if rho > 0.5:      # trajectory shapes still agree -> keep
                            continue
                    t.bound_phone = None
                    t.bound_age = 0

        unbound = [(i, t) for i, t in enumerate(self.tracks)
                   if t.bound_phone is None and not np.isnan(t.last_depth)]
        if not unbound:
            return
        used = self._phones_in_use()
        ghost_phones = self._phones_in_ghosts()
        free_phones = [p for p in range(len(ftm_m))
                       if ftm_valid[p] and p not in used and p not in ghost_phones]
        if not free_phones:
            return

        cost = np.full((len(unbound), len(free_phones)), 10.0, dtype=np.float64)
        for i, (ti, t) in enumerate(unbound):
            for j, p in enumerate(free_phones):
                compat = self._compat(t.last_depth, float(ftm_m[p]), phone_idx=p)
                ema_score = t.phone_scores.get(p, 0.0)
                score = 0.4 * compat + 0.6 * ema_score
                if self.traj_lambda > 0:
                    score += self.traj_lambda * self._traj_corr(t, p)
                cost[i, j] = -score
        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols):
            ti, t = unbound[r]
            p = free_phones[c]
            if -cost[r, c] < self.assign_threshold:
                continue
            # T5: risk-deferred binding — require score margin over the
            # second-best alternative in BOTH the track row and phone column
            if self.bind_margin > 0:
                score = -cost[r, c]
                row = -cost[r, :]
                col = -cost[:, c]
                row_2nd = np.max(row[np.arange(len(row)) != c]) if len(row) > 1 else -10.0
                col_2nd = np.max(col[np.arange(len(col)) != r]) if len(col) > 1 else -10.0
                rival = max(row_2nd, col_2nd)
                if rival > 0 and (score - rival) < self.bind_margin:
                    continue  # ambiguous this frame -> defer binding
            t.bound_phone = p
            t.bound_age = 0

    # -- T3: occlusion coefficients --------------------------------------------

    def _occlusion_coeffs(self, preds: np.ndarray) -> np.ndarray:
        n = len(self.tracks)
        occ = np.zeros(n)
        if n < 2 or self.occl_tau <= 0 and self.occl_kappa <= 0 \
                and not self.occl_depth_gate:
            return occ
        depths = np.array([t.last_depth for t in self.tracks])
        for j in range(n):
            if np.isnan(depths[j]):
                continue
            bx = preds[j]
            area = max((bx[2] - bx[0]) * (bx[3] - bx[1]), 1e-6)
            covered = 0.0
            for k in range(n):
                if k == j or np.isnan(depths[k]) or depths[k] >= depths[j]:
                    continue
                ox = max(0.0, min(bx[2], preds[k][2]) - max(bx[0], preds[k][0]))
                oy = max(0.0, min(bx[3], preds[k][3]) - max(bx[1], preds[k][1]))
                covered += ox * oy
            occ[j] = min(covered / area, 1.0)
        return occ

    def _associate(self, dets, tracks, preds, det_depths, ftm_m, ftm_valid):
        if len(tracks) == 0 or len(dets) == 0:
            return (np.empty((0, 2), dtype=int),
                    np.arange(len(dets), dtype=int),
                    np.arange(len(tracks), dtype=int))

        iou = iou_xyxy(dets, preds)
        occ = self._cur_occ if len(self._cur_occ) == len(tracks) else np.zeros(len(tracks))

        # T3(a): blend IoU cost toward its neutral value for occluded tracks
        # (OA-SORT Eq.8 convex form — NOT multiplicative scaling, which would
        # make occluded rows uniformly cheap and attract wrong detections)
        if self.occl_tau > 0:
            c_neutral = 1.0 - self.iou_threshold
            blend = self.occl_tau * occ  # in [0, tau]
            cost = (1.0 - blend)[None, :] * (1.0 - iou) + blend[None, :] * c_neutral
        else:
            cost = 1.0 - iou

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

        if self.wifi_weight > 0 and len(ftm_m) > 0:
            for j, t in enumerate(tracks):
                if t.bound_phone is None:
                    continue
                p = t.bound_phone
                if not ftm_valid[p]:
                    continue
                ftm = float(ftm_m[p])
                w = self.wifi_weight * (1.0 + self.occl_kappa * occ[j])
                if self.wifi_gate and self._sigma_hat is not None \
                        and p < len(self._sigma_hat):
                    sg = float(self._sigma_hat[p])
                    if np.isfinite(sg) and sg > 0:
                        s02 = self.sigma0_gate ** 2
                        w *= s02 / (s02 + sg * sg)
                for i in range(len(dets)):
                    if np.isnan(det_depths[i]):
                        continue
                    compat = self._compat(det_depths[i], ftm, phone_idx=p)
                    cost[i, j] += w * (1.0 - compat)

        max_cost = (1.0 - self.iou_threshold + self.inertia
                    + self.wifi_weight * (1.0 + max(self.occl_kappa, 0.0)))
        return hungarian(cost, max_cost=max_cost)

    # -- main step (copied from base with hooks) --------------------------------

    def update(self, dets_xyxy, det_depths, scores, ftm_m, ftm_valid,
               ftm_std_m=None) -> List[Tuple[int, np.ndarray]]:
        self.frame_count += 1
        self._cur_ftm_std_m = ftm_std_m

        # T1: record phone FTM history (frame-aligned)
        if self.traj_lambda > 0 and len(ftm_m) > 0:
            for p in range(len(ftm_m)):
                if ftm_valid[p] and not np.isnan(ftm_m[p]):
                    h = self._phone_hist.setdefault(p, [])
                    h.append((self.frame_count, float(ftm_m[p])))
                    if len(h) > 3 * self.traj_window:
                        del h[:len(h) - 3 * self.traj_window]

        predicted = [t.kf.predict() for t in self.tracks]
        pred_arr = np.stack(predicted, axis=0) if self.tracks else np.zeros((0, 4))

        # T3: occlusion coefficients from predictions + last depths
        self._cur_occ = self._occlusion_coeffs(pred_arr) if len(self.tracks) else np.zeros(0)

        if not self.no_ghost_pool and len(self.ghost_pool) > 0 and len(ftm_m) > 0:
            self._update_ghost_depths(ftm_m, ftm_valid)

        matched, unm_d, unm_t = self._associate(dets_xyxy, self.tracks, pred_arr,
                                                det_depths, ftm_m, ftm_valid)
        extras, unm_d, unm_t = self._ocr(dets_xyxy, self.tracks, unm_d, unm_t)
        if len(extras):
            matched = np.vstack([matched, extras])

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
                # T3(b): occlusion-gated depth update
                if self.occl_depth_gate and not np.isnan(track.last_depth):
                    oc = self._cur_occ[t_idx] if t_idx < len(self._cur_occ) else 0.0
                    depth_eff = track.last_depth + (1.0 - oc) * (depth - track.last_depth)
                else:
                    depth_eff = depth
                track.last_depth = depth_eff
                track.depth_history.append(depth_eff)
                if len(track.depth_history) > 50:
                    track.depth_history = track.depth_history[-50:]
                if self.traj_lambda > 0:
                    df = getattr(track, "depth_frames", None)
                    if df is None:
                        df = []
                        track.depth_frames = df
                    df.append((self.frame_count, depth_eff))
                    if len(df) > 3 * self.traj_window:
                        del df[:len(df) - 3 * self.traj_window]
            self._update_phone_scores(track, track.last_depth if not np.isnan(
                track.last_depth) else float("nan"), ftm_m, ftm_valid)

        if len(ftm_m) > 0:
            self._global_phone_assign(ftm_m, ftm_valid)

        reconnected_dets: List[int] = []
        if not self.no_ghost_pool:
            for d_idx in unm_d:
                revived = self._try_ghost_reconnect(d_idx, dets_xyxy, det_depths)
                if revived is not None:
                    depth = float(det_depths[d_idx]) if not np.isnan(det_depths[d_idx]) else float("nan")
                    self._update_phone_scores(revived, depth, ftm_m, ftm_valid)
                    self.tracks.append(revived)
                    reconnected_dets.append(d_idx)
        if reconnected_dets:
            rs = set(reconnected_dets)
            unm_d = np.array([d for d in unm_d if d not in rs], dtype=int)

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
                if self.traj_lambda > 0:
                    new.depth_frames = [(self.frame_count, depth)]
            self.tracks.append(new)
            self._try_initial_bind(new, depth, ftm_m, ftm_valid)

        for t_idx in unm_t:
            self.tracks[t_idx].time_since_update += 1

        survivors: List[_JointTrack] = []
        for t in self.tracks:
            t.age += 1
            if t.time_since_update <= self.max_age:
                survivors.append(t)
            elif not self.no_ghost_pool and t.bound_phone is not None:
                ghost_phones = self._phones_in_ghosts()
                if t.bound_phone not in ghost_phones:
                    self._demote_to_ghost(t)
        self.tracks = survivors

        self.ghost_pool = [g for g in self.ghost_pool
                           if (self.frame_count - g.death_frame) <= self.ghost_max_age]

        out: List[Tuple[int, np.ndarray]] = []
        for t in self.tracks:
            if t.time_since_update > 0:
                continue
            if t.hits >= self.min_hits or self.frame_count <= self.min_hits:
                tid = (self.phone_id_offset + t.bound_phone
                       if t.bound_phone is not None else t.track_id)
                out.append((tid, t.kf.bbox.copy()))
        return out
