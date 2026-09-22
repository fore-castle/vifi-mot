"""S10: paired significance test of ours vs Phase-6B over ALL 67 sequences.

Motivation: the S5 pilot showed per-sequence IDF1 deltas have std ~2.6pt on
scene4, which is an order of magnitude larger than the aggregate +0.31pt gain.
A 4-fold aggregate number alone cannot tell us whether the gain is real, so we
run a paired per-sequence comparison and test it properly.

Protocol (LOSO hygiene preserved): for every scene, the denoisers and matching
MLPs are the ones trained for that fold (i.e. never saw the test scene).
Detections are GT boxes (same as the main experiments).

Metrics per sequence: IDF1 for
  base = Phase 6B (causal denoiser + 6-dim learned MLP)
  ours = T15 dual-residual arbiter + T11 sigma gate + T3b occlusion depth gate

Tests on the 67 paired deltas:
  - Wilcoxon signed-rank (non-parametric, no normality assumption)
  - paired bootstrap 95% CI of the mean delta
  - sign count

Usage: python -u s10_paired_test.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)

MOT_ROOT = Path("/Users/zstar/auto_search/vifi-mot/mot_baseline")
sys.path.insert(0, str(MOT_ROOT))

import torch                                                      # noqa: E402
from data.vifi_mot import list_sequences, load_sequence           # noqa: E402
from trackers.wifi_joint_v2 import WiFiJointTrackerV2             # noqa: E402
from denoising.inference import FTMDenoiser                       # noqa: E402
from denoising.hetero import HeteroFTMDenoiser                    # noqa: E402
from eval.mot_eval import Evaluator                               # noqa: E402
from scripts.phase6b_integrated_mot import (                      # noqa: E402
    _frame_dets_xyxy_scores, _frame_det_depths, _frame_raw_ftm,
    load_matching_model)
from scripts.phase6b_learned_matching import MatchingMLP          # noqa: E402
from scripts.v2_experiments import BASE_KW                        # noqa: E402

SCENES = ["scene1", "scene2", "scene3", "scene4"]
OUT = Path(__file__).resolve().parent
W = 15


def eval_sequence(meta, variant, models):
    """variant: 'base' | 'ours'. Returns IDF1 (%)."""
    frames = load_sequence(meta)
    if not frames:
        return None

    kw = dict(BASE_KW)
    if variant == "ours":
        kw["wifi_gate"] = True
        kw["occl_depth_gate"] = True
        mm = models["dual"]
    else:
        mm = models["base"]

    tracker = WiFiJointTrackerV2(matching_model=mm, **kw)
    if variant == "ours":
        tracker.dual_matching = True

    den_c = models["den_c"]
    den_h = models["den_h"]
    den_c.reset()
    den_h.reset()

    seq_id_map, gt_seq, hyp_seq = {}, [], []
    for frame in frames:
        gt_items = []
        for d in frame.detections:
            tid = seq_id_map.setdefault(d.gt_track_id, len(seq_id_map) + 1)
            x1, y1, x2, y2 = d.bbox
            gt_items.append((tid, (x1, y1, x2 - x1, y2 - y1)))
        gt_seq.append((frame.frame_id, gt_items))

        dets, scores = _frame_dets_xyxy_scores(frame)
        depths = _frame_det_depths(frame)
        ftm_m, ftm_std_m, ftm_valid = _frame_raw_ftm(frame)

        if len(ftm_m) > 0 and ftm_valid.any():
            denoised = den_c.denoise(ftm_m, ftm_valid, ftm_std_m)
            if variant == "ours":
                mu_h, sig = den_h.denoise(ftm_m, ftm_valid, ftm_std_m)
                tracker._mu_h = mu_h
                tracker._sigma_hat = sig
        else:
            denoised = ftm_m

        outputs = tracker.update(dets, depths, scores, denoised,
                                 ftm_valid, ftm_std_m)
        hyp_seq.append((frame.frame_id,
                        [(int(i), (float(b[0]), float(b[1]),
                                   float(b[2] - b[0]), float(b[3] - b[1])))
                         for i, b in outputs]))

    ev = Evaluator()
    ev.add_sequence(meta.name, gt_seq, hyp_seq)
    s = ev.summarize()
    o = s.loc["OVERALL"].to_dict() if "OVERALL" in s.index else {}
    return float(o.get("idf1", 0)) * 100


def load_models(scene):
    ck = torch.load(MOT_ROOT / "checkpoints_matching" /
                    f"matching_dual_{scene}_best.pth",
                    map_location="cpu", weights_only=False)
    dual = MatchingMLP(in_features=12, hidden=16)
    dual.load_state_dict(ck["model_state"])
    dual.eval()
    return {"den_c": FTMDenoiser("causal_cnn", scene, W),
            "den_h": HeteroFTMDenoiser(scene, W),
            "base": load_matching_model(scene),
            "dual": dual}


def main():
    rows = {}
    print(f"{'scene':>7} {'sequence':>16} | {'base':>7} | {'ours':>7} | {'delta':>7}")
    print("-" * 56)
    for scene in SCENES:
        models = load_models(scene)
        seqs = [s for s in list_sequences(scenes=[scene]) if s.kind == "outdoor"]
        for meta in seqs:
            b = eval_sequence(meta, "base", models)
            o = eval_sequence(meta, "ours", models)
            if b is None or o is None:
                continue
            name = meta.name.split("-")[-1]
            rows[f"{scene}/{name}"] = {"base": b, "ours": o, "delta": o - b}
            print(f"{scene:>7} {name:>16} | {b:>6.2f}% | {o:>6.2f}% | "
                  f"{o - b:>+6.2f}", flush=True)

    d = np.array([v["delta"] for v in rows.values()])
    n = len(d)
    mean, std = float(d.mean()), float(d.std(ddof=1))
    se = std / np.sqrt(n)
    n_pos, n_neg = int((d > 0).sum()), int((d < 0).sum())

    # Wilcoxon signed-rank
    try:
        from scipy.stats import wilcoxon
        stat, p_w = wilcoxon(d, alternative="greater")
        p_w = float(p_w)
    except Exception as e:                                   # pragma: no cover
        stat, p_w = None, None
        print(f"(wilcoxon unavailable: {e})")

    # paired bootstrap CI (fixed seed)
    rng = np.random.RandomState(0)
    boots = np.array([rng.choice(d, n, replace=True).mean()
                      for _ in range(10000)])
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))

    print("-" * 56)
    print(f"n sequences        : {n}")
    print(f"mean delta         : {mean:+.3f} pt   (std {std:.3f}, SE {se:.3f})")
    print(f"sign               : {n_pos} positive / {n_neg} negative")
    print(f"bootstrap 95% CI   : [{ci[0]:+.3f}, {ci[1]:+.3f}]")
    if p_w is not None:
        print(f"Wilcoxon one-sided p: {p_w:.4f}  (H1: ours > base)")
    print(f"P(bootstrap mean>0): {float((boots > 0).mean()):.4f}")

    (OUT / "paired_test.json").write_text(json.dumps({
        "per_sequence": rows, "n": n, "mean_delta": mean, "std": std,
        "se": se, "n_positive": n_pos, "n_negative": n_neg,
        "bootstrap_ci95": ci, "wilcoxon_p_one_sided": p_w,
        "p_bootstrap_mean_gt0": float((boots > 0).mean()),
    }, indent=2))
    print(f"\nsaved -> {OUT/'paired_test.json'}")


if __name__ == "__main__":
    main()
