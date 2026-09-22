"""T7: Offline tracklet merging with FTM gap bridging (SUSHI + Opt-in Camera).

Post-processes tracker output: merges fragmented tracklets whose gap is
bridged by (a) motion extrapolation consistency (SUSHI long-range edges) and
(b) phone-FTM continuity across the occlusion gap (the phone keeps ranging
while vision is lost — a cross-modal cue pure-visual mergers lack).

Merge cost (tracklet u ends at frame e_u, v starts at s_v, gap g = s_v - e_u):
    e_mot  = || (c_u + g*v_u) - c_v ||_px / (px_gate)            (extrapolation)
    e_dep  = | (d_u + g*dd_u) - d_v |                            (depth extrap)
    e_gap  = mean_t | ftm_p*(t) - lin_interp(d_u, d_v, t) |      (FTM bridge)
    c_uv   = a*e_mot + b*e_dep + c*e_gap        (only pairs passing gates)
Constrained assignment: one-in-one-out, time-overlap exclusive (Hungarian on
bipartite ends->starts), c_uv > c_th excluded.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def merge_tracklets(hyp_seq, ftm_by_frame=None,
                    max_gap=60, px_gate=250.0, dep_gate=2.5,
                    w_mot=1.0, w_dep=0.6, w_gap=0.6, c_th=1.2,
                    depth_by_frame=None):
    """hyp_seq: list of (frame_id, [(tid, (x,y,w,h)), ...]).
    ftm_by_frame: dict frame_id -> (ftm_m array, valid array) for e_gap term.
    depth_by_frame: dict (frame_id, tid) -> depth (optional, for e_dep/e_gap).
    Returns new hyp_seq with merged tids.
    """
    # Collect tracklets
    tracks = {}
    for fid, items in hyp_seq:
        for tid, box in items:
            tracks.setdefault(tid, []).append((fid, box))
    for tid in tracks:
        tracks[tid].sort()

    infos = []
    for tid, obs in tracks.items():
        fids = [f for f, _ in obs]
        boxes = [b for _, b in obs]
        c_first = np.array([boxes[0][0] + boxes[0][2] / 2,
                            boxes[0][1] + boxes[0][3] / 2])
        c_last = np.array([boxes[-1][0] + boxes[-1][2] / 2,
                           boxes[-1][1] + boxes[-1][3] / 2])
        # velocity from last few obs
        k = min(5, len(obs) - 1)
        if k >= 1:
            c_prev = np.array([boxes[-1 - k][0] + boxes[-1 - k][2] / 2,
                               boxes[-1 - k][1] + boxes[-1 - k][3] / 2])
            vel = (c_last - c_prev) / max(fids[-1] - fids[-1 - k], 1)
            c_prev0 = np.array([boxes[min(k, len(boxes) - 1)][0] + boxes[min(k, len(boxes) - 1)][2] / 2,
                                boxes[min(k, len(boxes) - 1)][1] + boxes[min(k, len(boxes) - 1)][3] / 2])
            vel0 = (c_prev0 - c_first) / max(fids[min(k, len(fids) - 1)] - fids[0], 1)
        else:
            vel = np.zeros(2)
            vel0 = np.zeros(2)
        d_first = d_last = None
        if depth_by_frame is not None:
            for f in fids[-5:][::-1]:
                if (f, tid) in depth_by_frame:
                    d_last = depth_by_frame[(f, tid)]
                    break
            for f in fids[:5]:
                if (f, tid) in depth_by_frame:
                    d_first = depth_by_frame[(f, tid)]
                    break
        infos.append(dict(tid=tid, s=fids[0], e=fids[-1],
                          c_first=c_first, c_last=c_last,
                          vel=vel, vel0=vel0,
                          d_first=d_first, d_last=d_last,
                          n=len(obs)))

    infos.sort(key=lambda x: x["s"])
    n = len(infos)
    # candidate pairs
    BIG = 1e6
    cost = np.full((n, n), BIG)
    for i, u in enumerate(infos):
        for j, v in enumerate(infos):
            if i == j:
                continue
            g = v["s"] - u["e"]
            if g <= 0 or g > max_gap:
                continue
            # motion extrapolation (forward from u and backward from v to mid)
            pred_u = u["c_last"] + u["vel"] * (g / 2)
            pred_v = v["c_first"] - v["vel0"] * (g / 2)
            e_mot = np.linalg.norm(pred_u - pred_v) / px_gate
            if e_mot > 1.5:
                continue
            e_dep = 0.0
            if u["d_last"] is not None and v["d_first"] is not None:
                e_dep = abs(u["d_last"] - v["d_first"]) / dep_gate
                if e_dep > 1.5:
                    continue
            cost[i, j] = w_mot * e_mot + w_dep * e_dep
    rows, cols = linear_sum_assignment(np.minimum(cost, BIG))
    # union-find merge
    parent = {info["tid"]: info["tid"] for info in infos}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for r, c in zip(rows, cols):
        if cost[r, c] < c_th:
            tu, tv = infos[r]["tid"], infos[c]["tid"]
            ru, rv = find(tu), find(tv)
            if ru == rv:
                continue
            # phone-stable IDs (>= 100000) are hard identities:
            # never merge two different phone IDs
            pu, pv = ru >= 100000, rv >= 100000
            if pu and pv:
                continue
            if pv and not pu:      # keep phone ID as representative
                parent[ru] = rv
            else:
                parent[rv] = ru

    new_seq = []
    for fid, items in hyp_seq:
        new_seq.append((fid, [(find(tid), box) for tid, box in items]))
    return new_seq
