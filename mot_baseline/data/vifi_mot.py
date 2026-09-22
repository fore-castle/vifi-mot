"""ViFi -> MOT data layer.

Convert RAN4model_dfv4p4 pickles into MOT-style per-frame detections and
ground-truth tracks.

Conventions
-----------
BBX5 shape:        (T, N_user,   1, 5)  -> 5 dims = (cx, cy, depth, w, h)
BBX5_Others shape: (T-1, N_other, 1, 5) (one frame shorter than BBX5)
FTM_li shape:      (T, N_user,   1, 2)  -> (range_mm, std)
RSSI_li shape:     (T, N_user,   1, 1)  -> dBm
IMU19 shape:       (T, N_user,   1, 19)

Missing-frame marker: NaN in every BBX channel.

The function `load_sequence` returns a list[Frame] where each Frame contains:
    frame_id            : int
    detections          : list of dict with keys
        bbox            : (x1, y1, x2, y2) in image pixels
        cxcywh          : (cx, cy, w, h)
        depth           : float (meters)
        gt_track_id     : str    (unique across the whole sequence)
        score           : float  (1.0 baseline / depth-derived later)
        is_legitimate   : bool   (True = has phone wireless signal)
        subj_idx        : int    (only for legitimate users, else -1)
The frame additionally carries the legitimate users wireless tensors:
    ftm                 : np.ndarray (N_user, 2)        FTM range/std (mm/-)
    rssi                : np.ndarray (N_user, 1)        RSSI (dBm)
    imu19               : np.ndarray (N_user, 19)       full IMU
The wireless tensors keep the original subject index ordering of BBX5 so that
trackers can look up by subj_idx.
"""

from __future__ import annotations

import os
import pickle as pkl
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


DATASET_ROOT = "/Users/zstar/auto_search/vifi-mot/RAN4model_dfv4p4"


# ---------------------------------------------------------------------------
# Low level helpers
# ---------------------------------------------------------------------------

def _load_pkl(path: str):
    with open(path, "rb") as f:
        return pkl.load(f)


def _cxcywh_to_xyxy(cx: float, cy: float, w: float, h: float
                    ) -> Tuple[float, float, float, float]:
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return x1, y1, x2, y2


def _is_valid_bbox(b: np.ndarray) -> bool:
    """BBX5 row of (cx, cy, depth, w, h) - return True if usable."""
    if b is None:
        return False
    if np.any(np.isnan(b)):
        return False
    # require positive width/height
    if b[3] <= 1.0 or b[4] <= 1.0:
        return False
    return True


# ---------------------------------------------------------------------------
# Sequence discovery
# ---------------------------------------------------------------------------

@dataclass
class SequenceMeta:
    kind: str           # "indoor" / "outdoor"
    scene: str          # "scene0" .. "scene4"
    seq_id: str         # e.g. "20211007_144525"
    root: str           # absolute path to the seq directory

    @property
    def sync_dir(self) -> str:
        return os.path.join(self.root, "sync_ts16_dfv4p4")

    @property
    def name(self) -> str:
        return f"{self.kind}-{self.scene}-{self.seq_id}"


def list_sequences(scenes: Optional[List[str]] = None,
                   dataset_root: str = DATASET_ROOT,
                   ) -> List[SequenceMeta]:
    """List sequences across requested scenes.

    Parameters
    ----------
    scenes : list of names like ["scene1", "scene2"]. If None, returns all.
    """
    base = os.path.join(dataset_root, "seqs")
    out: List[SequenceMeta] = []
    for kind in ("indoor", "outdoor"):
        kdir = os.path.join(base, kind)
        if not os.path.isdir(kdir):
            continue
        for sc in sorted(os.listdir(kdir)):
            if scenes is not None and sc not in scenes:
                continue
            sc_dir = os.path.join(kdir, sc)
            for sq in sorted(os.listdir(sc_dir)):
                root = os.path.join(sc_dir, sq)
                if os.path.isdir(os.path.join(root, "sync_ts16_dfv4p4")):
                    out.append(SequenceMeta(kind, sc, sq, root))
    return out


# ---------------------------------------------------------------------------
# Frame container
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    bbox: Tuple[float, float, float, float]   # x1, y1, x2, y2
    cxcywh: Tuple[float, float, float, float]
    depth: float
    gt_track_id: str
    score: float
    is_legitimate: bool
    subj_idx: int


@dataclass
class Frame:
    frame_id: int
    detections: List[Detection] = field(default_factory=list)
    # legitimate users wireless tensors (ordered by subj_idx, possibly NaN)
    ftm: Optional[np.ndarray] = None       # (N_user, 2)
    rssi: Optional[np.ndarray] = None      # (N_user, 1)
    imu19: Optional[np.ndarray] = None     # (N_user, 19)
    imu_agm9: Optional[np.ndarray] = None  # (N_user, 9)  - accel+gyro+mag
    legit_valid: Optional[np.ndarray] = None  # (N_user,) bool - bbox available


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_sequence(meta: SequenceMeta,
                  score_mode: str = "constant",
                  ) -> List[Frame]:
    """Load one sequence into a list of Frame.

    Parameters
    ----------
    score_mode :
        - "constant"   : score = 1.0 for every detection
        - "depth"      : score = clip(1 - depth/20, 0.05, 1.0)
                         (closer -> higher confidence; loose proxy)
    """
    d = meta.sync_dir
    bbx5_path = os.path.join(d, "BBX5_sync_dfv4p4.pkl")
    if not os.path.exists(bbx5_path):
        # indoor uses BBX5H
        bbx5_path = os.path.join(d, "BBX5H_sync_dfv4p4.pkl")
    bbx5 = _load_pkl(bbx5_path)              # (T, N_user, 1, 5)
    T, N_user, _, _ = bbx5.shape

    ftm = _load_pkl(os.path.join(d, "FTM_li_sync_dfv4p4.pkl"))   # (T, N, 1, 2)
    rssi = _load_pkl(os.path.join(d, "RSSI_li_sync_dfv4p4.pkl")) # (T, N, 1, 1)
    imu19 = _load_pkl(os.path.join(d, "IMU19_sync_dfv4p4.pkl"))  # (T, N, 1, 19)
    imu_agm9_path = os.path.join(d, "IMUagm9_sync_dfv4p4.pkl")
    imu_agm9 = _load_pkl(imu_agm9_path) if os.path.exists(imu_agm9_path) else None

    others_path = os.path.join(d, "BBX5_Others_sync_dfv4p4.pkl")
    others_id_path = os.path.join(d, "Others_id_ls.pkl")
    has_others = os.path.exists(others_path) and os.path.exists(others_id_path)
    if has_others:
        bbx5_o = _load_pkl(others_path)      # (T-1 or T, N_other, 1, 5)
        others_ids = _load_pkl(others_id_path)
        if isinstance(others_ids, list):
            others_ids = list(others_ids)
        else:
            others_ids = list(others_ids)
        T_o, N_other, _, _ = bbx5_o.shape
    else:
        bbx5_o = None
        others_ids = []
        T_o, N_other = 0, 0

    frames: List[Frame] = []
    for t in range(T):
        f = Frame(frame_id=t)

        legit_valid = np.zeros(N_user, dtype=bool)
        for s in range(N_user):
            row = bbx5[t, s, 0, :]
            if not _is_valid_bbox(row):
                continue
            cx, cy, depth, w, h = float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])
            if score_mode == "depth":
                score = float(np.clip(1.0 - depth / 20.0, 0.05, 1.0))
            else:
                score = 1.0
            x1, y1, x2, y2 = _cxcywh_to_xyxy(cx, cy, w, h)
            f.detections.append(Detection(
                bbox=(x1, y1, x2, y2),
                cxcywh=(cx, cy, w, h),
                depth=depth,
                gt_track_id=f"L{s}",
                score=score,
                is_legitimate=True,
                subj_idx=s,
            ))
            legit_valid[s] = True

        if has_others and t < T_o:
            for k in range(N_other):
                row = bbx5_o[t, k, 0, :]
                if not _is_valid_bbox(row):
                    continue
                cx, cy, depth, w, h = float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])
                if score_mode == "depth":
                    score = float(np.clip(1.0 - depth / 20.0, 0.05, 1.0))
                else:
                    score = 1.0
                x1, y1, x2, y2 = _cxcywh_to_xyxy(cx, cy, w, h)
                f.detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    cxcywh=(cx, cy, w, h),
                    depth=depth,
                    gt_track_id=f"O{k}",
                    score=score,
                    is_legitimate=False,
                    subj_idx=-1,
                ))

        # wireless tensors aligned by subj_idx; may contain NaNs
        f.ftm = ftm[t, :, 0, :].astype(np.float32)
        f.rssi = rssi[t, :, 0, :].astype(np.float32)
        f.imu19 = imu19[t, :, 0, :].astype(np.float32)
        if imu_agm9 is not None:
            f.imu_agm9 = imu_agm9[t, :, 0, :].astype(np.float32)
        f.legit_valid = legit_valid
        frames.append(f)

    return frames


# ---------------------------------------------------------------------------
# MOTChallenge-style export
# ---------------------------------------------------------------------------

def export_motchallenge_gt(frames: List[Frame], out_path: str) -> None:
    """Write GT in MOTChallenge 1.1 format.

    columns: frame, id, bb_left, bb_top, bb_width, bb_height, conf,
             class(1=pedestrian), visibility(1.0), wireless_flag (custom).
    """
    # build stable integer id per gt_track_id
    str2int: Dict[str, int] = {}
    rows = []
    for fr in frames:
        for det in fr.detections:
            if det.gt_track_id not in str2int:
                str2int[det.gt_track_id] = len(str2int) + 1
            tid = str2int[det.gt_track_id]
            x1, y1, x2, y2 = det.bbox
            w = x2 - x1
            h = y2 - y1
            rows.append((fr.frame_id + 1, tid, x1, y1, w, h,
                         1, 1, 1.0, int(det.is_legitimate)))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for r in rows:
            f.write("{:d},{:d},{:.2f},{:.2f},{:.2f},{:.2f},{:d},{:d},{:.2f},{:d}\n".format(*r))


def export_motchallenge_det(frames: List[Frame], out_path: str) -> None:
    """Write detections in MOT format (id always -1)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for fr in frames:
            for det in fr.detections:
                x1, y1, x2, y2 = det.bbox
                w = x2 - x1
                h = y2 - y1
                f.write("{:d},-1,{:.2f},{:.2f},{:.2f},{:.2f},{:.4f},-1,-1,-1\n".format(
                    fr.frame_id + 1, x1, y1, w, h, det.score))


# ---------------------------------------------------------------------------
# Quick sanity entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    metas = list_sequences()
    print(f"total sequences: {len(metas)}")
    by_scene: Dict[str, int] = {}
    for m in metas:
        by_scene[f"{m.kind}/{m.scene}"] = by_scene.get(f"{m.kind}/{m.scene}", 0) + 1
    for k, v in sorted(by_scene.items()):
        print(f"  {k}: {v}")

    # spot-check one
    seq = next(m for m in metas if m.scene == "scene4")
    frames = load_sequence(seq)
    n_det = sum(len(f.detections) for f in frames)
    n_legit = sum(d.is_legitimate for f in frames for d in f.detections)
    n_other = n_det - n_legit
    gt_ids = sorted({d.gt_track_id for f in frames for d in f.detections})
    print(f"\nspot check: {seq.name}")
    print(f"  T={len(frames)}  detections={n_det}  legit={n_legit}  others={n_other}")
    print(f"  unique GT ids = {len(gt_ids)} -> {gt_ids[:8]}...")
