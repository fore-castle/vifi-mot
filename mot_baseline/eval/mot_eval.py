"""MOT evaluator using py-motmetrics.

Produces MOTA / IDF1 / IDsw / MOTP / MT / ML for a single sequence and
aggregates a set of sequences into a summary DataFrame.

We feed the evaluator with per-frame matches built on top of IoU (default
threshold = 0.5), which is the standard MOTChallenge protocol.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import motmetrics as mm
import pandas as pd

IOU_DIST_THRESHOLD = 0.5   # distance = 1 - IoU; rejects matches if dist > 0.5


def _iou_distance_matrix(gt_xywh: np.ndarray,
                          hyp_xywh: np.ndarray) -> np.ndarray:
    """Pairwise 1 - IoU distance between two boxes lists (x, y, w, h).

    NaN where distance > IOU_DIST_THRESHOLD (rejected by motmetrics).
    """
    if len(gt_xywh) == 0 or len(hyp_xywh) == 0:
        return np.zeros((len(gt_xywh), len(hyp_xywh)), dtype=np.float64)

    g = np.asarray(gt_xywh, dtype=np.float64)
    h = np.asarray(hyp_xywh, dtype=np.float64)

    gx1, gy1 = g[:, 0:1], g[:, 1:2]
    gx2, gy2 = gx1 + g[:, 2:3], gy1 + g[:, 3:4]
    hx1, hy1 = h[:, 0:1].T, h[:, 1:2].T
    hx2, hy2 = hx1 + h[:, 2:3].T, hy1 + h[:, 3:4].T

    ix1 = np.maximum(gx1, hx1)
    iy1 = np.maximum(gy1, hy1)
    ix2 = np.minimum(gx2, hx2)
    iy2 = np.minimum(gy2, hy2)
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    a_g = (g[:, 2] * g[:, 3]).reshape(-1, 1)
    a_h = (h[:, 2] * h[:, 3]).reshape(1, -1)
    union = a_g + a_h - inter
    iou = np.where(union > 0, inter / union, 0.0)
    dist = 1.0 - iou
    dist[dist > IOU_DIST_THRESHOLD] = np.nan
    return dist


class Evaluator:
    """Builds and aggregates MOTAccumulator objects across sequences."""

    def __init__(self) -> None:
        self.accumulators: Dict[str, mm.MOTAccumulator] = {}

    def add_sequence(self,
                     seq_name: str,
                     gt: List[Tuple[int, List[Tuple[int, Tuple[float, float, float, float]]]]],
                     hyp: List[Tuple[int, List[Tuple[int, Tuple[float, float, float, float]]]]],
                     ) -> None:
        """Add a sequence.

        gt / hyp items: (frame_id, [(id, (x, y, w, h)), ...])
        frame_id must monotonically increase but does not have to be contiguous.
        """
        acc = mm.MOTAccumulator(auto_id=False)
        # build a frame-indexed map
        gt_map = dict(gt)
        hyp_map = dict(hyp)
        frame_ids = sorted(set(gt_map.keys()) | set(hyp_map.keys()))
        for fid in frame_ids:
            g_items = gt_map.get(fid, [])
            h_items = hyp_map.get(fid, [])
            g_ids = [i for i, _ in g_items]
            h_ids = [i for i, _ in h_items]
            g_boxes = np.array([b for _, b in g_items], dtype=np.float64).reshape(-1, 4)
            h_boxes = np.array([b for _, b in h_items], dtype=np.float64).reshape(-1, 4)
            d = _iou_distance_matrix(g_boxes, h_boxes)
            acc.update(g_ids, h_ids, d, frameid=fid)
        self.accumulators[seq_name] = acc

    def summarize(self) -> pd.DataFrame:
        if not self.accumulators:
            return pd.DataFrame()
        mh = mm.metrics.create()
        metric_names = [
            "num_frames", "num_unique_objects",
            "mota", "motp", "idf1", "idp", "idr",
            "num_switches", "num_fragmentations",
            "mostly_tracked", "partially_tracked", "mostly_lost",
            "num_false_positives", "num_misses",
            "precision", "recall",
        ]
        summary = mh.compute_many(
            list(self.accumulators.values()),
            metrics=metric_names,
            names=list(self.accumulators.keys()),
            generate_overall=True,
        )
        # Make motp interpretable: motmetrics returns it as average distance.
        return summary

    def format(self, summary: pd.DataFrame) -> str:
        mh = mm.metrics.create()
        return mm.io.render_summary(
            summary,
            namemap=mm.io.motchallenge_metric_names,
            formatters=mh.formatters,
        )
