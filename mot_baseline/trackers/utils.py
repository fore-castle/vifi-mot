"""Common building blocks: IoU, Hungarian matching, constant-velocity Kalman.

The Kalman state matches the canonical SORT formulation:
    x = (u, v, s, r, du, dv, ds)
where u, v are the box center, s is the area, r is the aspect ratio.
This keeps aspect ratio constant during prediction (a known SORT simplification).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# IoU helpers
# ---------------------------------------------------------------------------

def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU. a:(N,4) b:(M,4) -> (N,M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0:1].T, b[:, 1:2].T, b[:, 2:3].T, b[:, 3:4].T
    ix1 = np.maximum(ax1, bx1)
    iy1 = np.maximum(ay1, by1)
    ix2 = np.minimum(ax2, bx2)
    iy2 = np.minimum(ay2, by2)
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    aa = (ax2 - ax1) * (ay2 - ay1)
    ab = (bx2 - bx1) * (by2 - by1)
    union = aa + ab - inter
    return np.where(union > 0, inter / union, 0.0)


def hungarian(cost: np.ndarray, max_cost: float = 0.7
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run Hungarian. Returns matched_pairs, unmatched_rows, unmatched_cols.

    matched_pairs is (K, 2) with row,col indices satisfying cost<=max_cost.
    """
    if cost.size == 0:
        return (np.empty((0, 2), dtype=int),
                np.arange(cost.shape[0], dtype=int),
                np.arange(cost.shape[1], dtype=int))
    rows, cols = linear_sum_assignment(cost)
    matched, unm_r, unm_c = [], [], []
    matched_rows = set()
    matched_cols = set()
    for r, c in zip(rows, cols):
        if cost[r, c] <= max_cost:
            matched.append((r, c))
            matched_rows.add(r)
            matched_cols.add(c)
    matched = np.array(matched, dtype=int).reshape(-1, 2)
    unm_r = np.array([r for r in range(cost.shape[0]) if r not in matched_rows], dtype=int)
    unm_c = np.array([c for c in range(cost.shape[1]) if c not in matched_cols], dtype=int)
    return matched, unm_r, unm_c


# ---------------------------------------------------------------------------
# Box <-> (u, v, s, r) helpers
# ---------------------------------------------------------------------------

def xyxy_to_usr(b: np.ndarray) -> np.ndarray:
    """(x1, y1, x2, y2) -> (u, v, s, r)."""
    w = b[2] - b[0]
    h = b[3] - b[1]
    u = b[0] + w / 2.0
    v = b[1] + h / 2.0
    s = w * h
    r = w / max(h, 1e-6)
    return np.array([u, v, s, r], dtype=np.float64)


def usr_to_xyxy(x: np.ndarray) -> np.ndarray:
    u, v, s, r = x[0], x[1], max(x[2], 1.0), max(x[3], 1e-3)
    w = np.sqrt(s * r)
    h = s / max(w, 1e-6)
    return np.array([u - w / 2.0, v - h / 2.0,
                     u + w / 2.0, v + h / 2.0], dtype=np.float64)


# ---------------------------------------------------------------------------
# 7-state constant-velocity Kalman filter (SORT)
# ---------------------------------------------------------------------------

class KalmanBox:
    """SORT-style Kalman state.

    State x in R^7 = [u, v, s, r, du, dv, ds].
    The aspect ratio r is treated as constant (no derivative).
    """

    def __init__(self, bbox_xyxy: np.ndarray) -> None:
        # Transition F (7x7), constant velocity for u, v, s; r static
        self.F = np.eye(7)
        for i in range(3):
            self.F[i, i + 4] = 1.0
        # Observation H (4x7) - we observe u, v, s, r
        self.H = np.zeros((4, 7))
        for i in range(4):
            self.H[i, i] = 1.0
        # Process noise Q
        self.Q = np.eye(7)
        self.Q[4:, 4:] *= 0.01
        self.Q[3, 3] *= 0.01
        # Measurement noise R
        self.R = np.eye(4)
        self.R[2:, 2:] *= 10.0
        # State covariance P
        self.P = np.eye(7) * 10.0
        self.P[4:, 4:] *= 1000.0
        self.P *= 10.0

        z = xyxy_to_usr(bbox_xyxy)
        self.x = np.zeros(7)
        self.x[:4] = z

    def predict(self) -> np.ndarray:
        # Guard: if s + ds would go non-positive, zero out ds.
        if self.x[2] + self.x[6] <= 0:
            self.x[6] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return usr_to_xyxy(self.x[:4])

    def update(self, bbox_xyxy: np.ndarray) -> None:
        z = xyxy_to_usr(bbox_xyxy)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(7) - K @ self.H
        self.P = I_KH @ self.P

    @property
    def bbox(self) -> np.ndarray:
        return usr_to_xyxy(self.x[:4])
