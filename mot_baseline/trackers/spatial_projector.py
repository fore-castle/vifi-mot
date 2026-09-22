"""3D Spatial Radio Prior utilities.

This module provides the *physics* part of Phase 1: geometric 2D->3D back-projection
from ZED2 camera pixels to camera-frame meters, AP position estimation via
least-squares over a short warm-up window, phone 3D position estimation under
single-AP FTM + heading + height priors, and Mahalanobis 3D spatial compatibility.

These are training-free building blocks consumed by ``wifi_spatial_ocsort``.

Coordinate convention
---------------------
Camera frame, right-handed, Y-up:
    Z = depth (m), positive in front of camera
    X = horizontal lateral (m), positive to the right
    Y = vertical (m), positive up (so a person on the ground at the camera height
        has Y ~= -h_cam_above_ground); for the AP-estimation least-squares we
        do **not** rely on absolute Y because the camera height is unknown --
        we instead solve directly in camera frame and let the optimisation
        absorb whatever vertical offset is present.

Pixel back-projection:
    X = (u - cx) / fx * d
    Y = -(v - cy) / fy * d         (image y goes down; we flip to get Y-up)
    Z =  d
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares


# ---------------------------------------------------------------------------
# ZED2 HD720 intrinsics (Vi-Fi default, verified against depth_to_dist.py)
# ---------------------------------------------------------------------------
FX = 528.365
FY = 527.925
CX = 638.925
CY = 359.2805


def project_pixel_to_camera(u: float, v: float, depth: float
                            ) -> np.ndarray:
    """Back-project a pixel + depth into the camera frame (meters).

    Returns ``np.array([X, Y, Z])``. Returns ``[nan, nan, nan]`` when depth is
    NaN or non-positive (a non-positive depth means an invalid detection).
    """
    if not np.isfinite(depth) or depth <= 0.0:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float64)
    X = (u - CX) / FX * depth
    Y = -(v - CY) / FY * depth
    Z = depth
    return np.array([X, Y, Z], dtype=np.float64)


def project_bbox_center(bbox_xyxy: np.ndarray, depth: float) -> np.ndarray:
    """Back-project the *center* of a bbox at ``depth`` into camera frame."""
    u = 0.5 * (bbox_xyxy[0] + bbox_xyxy[2])
    v = 0.5 * (bbox_xyxy[1] + bbox_xyxy[3])
    return project_pixel_to_camera(u, v, depth)


# ---------------------------------------------------------------------------
# AP position estimation (single-AP, single-scene)
# ---------------------------------------------------------------------------

@dataclass
class APEstimate:
    """Result of ``estimate_ap_position``."""
    position: np.ndarray            # (3,) AP in camera frame, meters
    rms_residual: float             # |dist - r_ftm| RMS in meters
    n_obs: int                      # number of (pos, ftm) pairs used
    converged: bool


def estimate_ap_position(track_positions: np.ndarray,
                         ftm_ranges: np.ndarray,
                         init: Optional[np.ndarray] = None,
                         ) -> APEstimate:
    """Estimate the AP 3D position from track positions + FTM ranges.

    Parameters
    ----------
    track_positions : (N, 3) array of camera-frame XYZ of bound phone holders.
    ftm_ranges      : (N,)   matching FTM ranges in meters.

    The optimisation minimises sum over N of ``(||p - p_AP|| - r_ftm)^2``.
    """
    track_positions = np.asarray(track_positions, dtype=np.float64).reshape(-1, 3)
    ftm_ranges = np.asarray(ftm_ranges, dtype=np.float64).reshape(-1)
    mask = (np.isfinite(track_positions).all(axis=1) & np.isfinite(ftm_ranges)
            & (ftm_ranges > 0.0))
    track_positions = track_positions[mask]
    ftm_ranges = ftm_ranges[mask]
    n_obs = len(ftm_ranges)
    if n_obs < 8:
        # Not enough observations for a stable fit; return a no-op estimate.
        return APEstimate(np.zeros(3), float("inf"), n_obs, False)

    if init is None:
        # Use the centroid of the tracks as the starting point; a reasonable
        # prior because the AP is usually mounted in the scene.
        init = track_positions.mean(axis=0)

    def residuals(p_ap):
        diffs = track_positions - p_ap
        d = np.linalg.norm(diffs, axis=1)
        return d - ftm_ranges

    try:
        res = least_squares(residuals, init, method="lm", max_nfev=200)
    except Exception:
        return APEstimate(init, float("inf"), n_obs, False)

    rms = float(np.sqrt(np.mean(res.fun ** 2)))
    return APEstimate(res.x.astype(np.float64), rms, n_obs, bool(res.success))


# ---------------------------------------------------------------------------
# Phone 3D position estimate with directional covariance
# ---------------------------------------------------------------------------

@dataclass
class PhoneState:
    """Best 3D position estimate of a phone with its uncertainty.

    The covariance ``Sigma`` is directional: small along the AP->phone radial
    direction (FTM is precise on the *distance*) and large perpendicular to it
    (FTM gives no bearing).
    """
    position: np.ndarray            # (3,) camera frame
    Sigma: np.ndarray               # (3, 3) covariance, camera frame, meters^2
    r_ftm: float                    # FTM range used
    sigma_ftm: float                # FTM std used
    valid: bool


def estimate_phone_state(p_ap: np.ndarray,
                         r_ftm: float,
                         sigma_ftm: float,
                         anchor_xyz: Optional[np.ndarray] = None,
                         sigma_perp: float = 2.5,
                         h_floor: float = -1.5,
                         h_phone_above_floor: float = 1.0,
                         ) -> PhoneState:
    """Best phone-position estimate given AP, FTM, and an anchor.

    With a single AP, FTM alone defines a sphere of radius ``r_ftm`` around
    ``p_ap``. We *project* the anchor onto this sphere to get the most likely
    position, then build a 3x3 covariance whose smallest eigenvector points
    along the AP->phone direction (= radial uncertainty = sigma_ftm) and whose
    two larger eigenvectors span the tangent plane (= bearing uncertainty =
    sigma_perp).

    Parameters
    ----------
    p_ap        : (3,) AP position, camera frame
    r_ftm       : FTM range (m)
    sigma_ftm   : FTM stdev (m)
    anchor_xyz  : (3,) best guess of phone position (e.g. last bound track 3D
                  pos). If None, falls back to a synthetic anchor along the +Z
                  axis at the camera origin so the resulting position is in
                  front of the camera.
    sigma_perp  : tangential uncertainty in meters (bearing uncertainty x r).
                  Defaults to 2.5 m which corresponds to ~30 deg at 5 m range.
    h_floor     : camera-frame Y coordinate of the floor (negative because the
                  camera is mounted above the floor). Currently unused but kept
                  for future ground-plane projection.
    h_phone_above_floor : prior phone height (m). Unused here, kept for parity.
    """
    p_ap = np.asarray(p_ap, dtype=np.float64).reshape(3)
    if not np.isfinite(r_ftm) or r_ftm <= 0.0:
        return PhoneState(np.full(3, np.nan), np.eye(3) * 1e6,
                          float("nan"), float("nan"), False)
    if anchor_xyz is None or not np.all(np.isfinite(anchor_xyz)):
        anchor_xyz = p_ap + np.array([0.0, 0.0, max(r_ftm, 1e-3)])
    direction = anchor_xyz - p_ap
    dist = np.linalg.norm(direction)
    if dist < 1e-3:
        # Anchor coincides with AP; pick an arbitrary direction.
        direction = np.array([0.0, 0.0, 1.0])
        dist = 1.0
    unit = direction / dist
    position = p_ap + unit * r_ftm

    # Build directional covariance: variance r_radial^2 along unit, variance
    # sigma_perp^2 along two orthonormal tangent vectors.
    # Construct an orthonormal basis with unit as the first axis.
    if abs(unit[0]) < 0.9:
        helper = np.array([1.0, 0.0, 0.0])
    else:
        helper = np.array([0.0, 1.0, 0.0])
    t1 = np.cross(unit, helper)
    t1 /= np.linalg.norm(t1) + 1e-9
    t2 = np.cross(unit, t1)
    t2 /= np.linalg.norm(t2) + 1e-9
    R = np.stack([unit, t1, t2], axis=1)         # 3x3 rotation
    D = np.diag([max(sigma_ftm, 0.3) ** 2,
                 sigma_perp ** 2,
                 sigma_perp ** 2])
    Sigma = R @ D @ R.T
    return PhoneState(position.astype(np.float64), Sigma, float(r_ftm),
                      float(sigma_ftm), True)


# ---------------------------------------------------------------------------
# Mahalanobis 3D spatial compatibility
# ---------------------------------------------------------------------------

def mahalanobis_compat(p_track: np.ndarray,
                       state: PhoneState,
                       gate_chi2: float = 9.21,
                       ) -> Tuple[float, float]:
    """Return ``(compat, d_maha2)`` for a track position vs a phone state.

    ``compat`` is exp(-0.5 * d^2) clipped to ``[0, 1]``. ``d_maha2`` is the
    squared Mahalanobis distance. Both are returned so the caller can apply a
    chi-squared gate on the squared distance if desired (default 9.21 = 99%
    quantile for 2 DOF, which is conservative for a 3 DOF model).
    """
    if not state.valid:
        return 0.0, float("inf")
    if not np.all(np.isfinite(p_track)):
        return 0.0, float("inf")
    diff = p_track - state.position
    try:
        Sigma_inv = np.linalg.inv(state.Sigma)
    except np.linalg.LinAlgError:
        return 0.0, float("inf")
    d2 = float(diff @ Sigma_inv @ diff)
    if d2 > gate_chi2 * 4.0:       # extra-loose hard gate
        return 0.0, d2
    return float(np.exp(-0.5 * d2)), d2


# ---------------------------------------------------------------------------
# Adaptive FTM observation noise R_wire
# ---------------------------------------------------------------------------

def adaptive_R_wire(sigma_ftm: float,
                    depth: float,
                    sigma_ap: float = 0.5,
                    sigma_ftm_floor: float = 0.4,
                    ) -> float:
    """Per-frame R_wire = sigma_ftm^2 + sigma_depth(depth)^2 + sigma_ap^2.

    ``sigma_ftm_floor`` is the minimum FTM uncertainty we trust (in meters),
    because the per-frame std reported by the device is often optimistic and
    Phase 0 found very weak correlation between it and the actual error.
    """
    if not np.isfinite(sigma_ftm) or sigma_ftm <= 0.0:
        sigma_ftm = sigma_ftm_floor
    sigma_ftm_eff = max(sigma_ftm, sigma_ftm_floor)
    sigma_depth = 0.1 + 0.02 * max(depth, 0.0)
    return float(sigma_ftm_eff ** 2 + sigma_depth ** 2 + sigma_ap ** 2)


# ---------------------------------------------------------------------------
# Public symbols
# ---------------------------------------------------------------------------

__all__ = [
    "FX", "FY", "CX", "CY",
    "APEstimate", "PhoneState",
    "project_pixel_to_camera", "project_bbox_center",
    "estimate_ap_position",
    "estimate_phone_state",
    "mahalanobis_compat",
    "adaptive_R_wire",
]
