"""Helper that wraps my_vifi's MultimodalNetwork as a phone-track affinity oracle.

Maintains rolling K-frame buffers of (cx, cy, depth) per track and
(FTM_range, FTM_std, IMUagm9 9d) per phone, formats them into the model's
expected input layout, and returns a normalized phone×track affinity matrix in
[0, 1].

Notes
-----
- Phone monopoly / bind logic is handled by the tracker, not here.
- If a track has fewer than `min_history` valid frames, its column is masked out
  (returned affinity = 0). Same for phones.
- Inputs are Z-score normalized using stats produced by
  scripts/precompute_norm_stats.py.
"""

from __future__ import annotations

import os
import sys
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Make my_vifi importable so the saved checkpoint (an instance of
# MultimodalNetwork) can be unpickled.
_MY_VIFI = "/Users/zstar/test/vifi/my_vifi"
if _MY_VIFI not in sys.path:
    sys.path.insert(0, _MY_VIFI)


class AffinityHelper:
    def __init__(self,
                 ckpt_path: str,
                 norm_stats_path: str,
                 k: int = 10,
                 Np: int = 5,
                 Nc: int = 15,
                 min_history: int = 3,
                 device: str = "cpu") -> None:
        self.k = k
        self.Np = Np
        self.Nc = Nc
        self.min_history = min_history
        self.device = torch.device(device)

        self.model = torch.load(ckpt_path, map_location=self.device,
                                weights_only=False)
        self.model.eval()
        self.cam_feat_dim = int(self.model.camera_feat_dim)
        self.ph_feat_dim = int(self.model.phone_feat_dim)
        assert self.cam_feat_dim == 3 and self.ph_feat_dim == 11, \
            "Expected camera_feat_dim=3 (cx,cy,depth) and phone_feat_dim=11 (FTM2+IMUagm9)"

        stats = np.load(norm_stats_path)
        self.cam_mean = stats["cam_mean"].astype(np.float32)
        self.cam_std = stats["cam_std"].astype(np.float32)
        self.ph_mean = stats["ph_mean"].astype(np.float32)
        self.ph_std = stats["ph_std"].astype(np.float32)

        # Rolling buffers
        # track_id -> deque[(cx, cy, depth)]
        self.track_buf: Dict[int, deque] = {}
        # phone_idx -> deque[(ftm_r, ftm_std, imu9...)]
        self.phone_buf: Dict[int, deque] = {}

    # -- buffer maintenance --------------------------------------------------

    def push_track(self, track_id: int, cx: float, cy: float, depth: float) -> None:
        if np.isnan(cx) or np.isnan(cy) or np.isnan(depth):
            return
        buf = self.track_buf.setdefault(track_id, deque(maxlen=self.k))
        buf.append((float(cx), float(cy), float(depth)))

    def push_phone(self, phone_idx: int, ftm_2: np.ndarray,
                   imu9: np.ndarray) -> None:
        if ftm_2 is None or imu9 is None:
            return
        if np.any(np.isnan(ftm_2)) or np.any(np.isnan(imu9)):
            return
        vec = np.concatenate([ftm_2, imu9]).astype(np.float32)  # (11,)
        buf = self.phone_buf.setdefault(phone_idx, deque(maxlen=self.k))
        buf.append(tuple(vec))

    def drop_track(self, track_id: int) -> None:
        self.track_buf.pop(track_id, None)

    # -- inference -----------------------------------------------------------

    def predict(self,
                track_ids: List[int],
                phone_idxs: List[int]) -> np.ndarray:
        """Return affinity (N_phone_query, N_track_query) in [0, 1].

        phone_idxs / track_ids select which buffers to read; entries with
        history shorter than `min_history` are zeroed out in the output.
        """
        N_t_q = len(track_ids)
        N_p_q = len(phone_idxs)
        if N_t_q == 0 or N_p_q == 0:
            return np.zeros((N_p_q, N_t_q), dtype=np.float32)

        # ---------------------------------------------------------- camera
        cam = np.zeros((self.Nc, self.k * self.cam_feat_dim + 1), dtype=np.float32)
        cam_mask = np.zeros(self.Nc + 1, dtype=bool)
        cam_mask[-1] = True   # extra column always valid

        # We can include at most Nc tracks; if more, take the first Nc (the
        # tracker should slice to active ones before calling).
        track_slot = []
        for t_idx, tid in enumerate(track_ids[:self.Nc]):
            buf = self.track_buf.get(tid)
            if not buf or len(buf) < self.min_history:
                track_slot.append(None)
                continue
            seq = np.array(list(buf), dtype=np.float32)         # (n, 3)
            n = seq.shape[0]
            # Z-score
            seq = (seq - self.cam_mean) / (self.cam_std + 1e-8)
            cam[t_idx, :n * self.cam_feat_dim] = seq.reshape(-1)
            cam[t_idx, -1] = n
            cam_mask[t_idx] = True
            track_slot.append(t_idx)

        # ---------------------------------------------------------- phone
        ph = np.zeros((self.Np, self.k * self.ph_feat_dim + 1), dtype=np.float32)
        ph_mask = np.zeros(self.Np + 1, dtype=bool)
        ph_mask[-1] = True

        phone_slot = []
        for p_idx, pid in enumerate(phone_idxs[:self.Np]):
            buf = self.phone_buf.get(pid)
            if not buf or len(buf) < self.min_history:
                phone_slot.append(None)
                continue
            seq = np.array(list(buf), dtype=np.float32)         # (n, 11)
            n = seq.shape[0]
            seq = (seq - self.ph_mean) / (self.ph_std + 1e-8)
            ph[p_idx, :n * self.ph_feat_dim] = seq.reshape(-1)
            ph[p_idx, -1] = n
            ph_mask[p_idx] = True
            phone_slot.append(p_idx)

        # ---------------------------------------------------------- forward
        # Padding rows must have valid_len >= 1 to avoid pack_padded crash.
        for i in range(self.Nc):
            if cam[i, -1] < 1:
                cam[i, -1] = 1.0
        for i in range(self.Np):
            if ph[i, -1] < 1:
                ph[i, -1] = 1.0

        cam_t = torch.from_numpy(cam).unsqueeze(0).to(self.device)
        ph_t = torch.from_numpy(ph).unsqueeze(0).to(self.device)
        cm_t = torch.from_numpy(cam_mask).unsqueeze(0).to(self.device)
        pm_t = torch.from_numpy(ph_mask).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(cam_t, ph_t, cm_t, pm_t)  # (1, 1, Np+1, Nc+1)
        pred = out[0, 0]  # (Np+1, Nc+1)

        # Row-wise softmax (phone -> camera) excluding extra row, but including
        # the extra column so that "no match" can absorb probability mass.
        Np_, Nc_ = self.Np, self.Nc
        mask_pc = torch.ones_like(pred)
        mask_pc[Np_, :] = 0.0
        pred_pc = torch.softmax(mask_pc * pred, dim=1)  # (Np+1, Nc+1)
        # Col-wise softmax (camera -> phone)
        mask_cp = torch.ones_like(pred)
        mask_cp[:, Nc_] = 0.0
        pred_cp = torch.softmax(mask_cp * pred, dim=0)
        sub_pc = pred_pc[:Np_, :Nc_]
        sub_cp = pred_cp[:Np_, :Nc_]
        avg = ((sub_pc + sub_cp) / 2.0).cpu().numpy()  # (Np, Nc)

        # Gather queried (phone, track) entries
        result = np.zeros((N_p_q, N_t_q), dtype=np.float32)
        for qi, pslot in enumerate(phone_slot):
            if pslot is None:
                continue
            for qj, tslot in enumerate(track_slot):
                if tslot is None:
                    continue
                result[qi, qj] = float(avg[pslot, tslot])
        return result
