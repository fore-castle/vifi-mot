"""V2 experiments: paper-inspired training-free extensions, full 4-fold LOSO.

Configs are named ablations over WiFiJointTrackerV2 flags. Baselines
(base_gaussian / base_learned) reproduce Phase 6B numbers with the V2 class
(all extensions off) to guard against implementation drift.

Usage:
    python -u scripts/v2_experiments.py --configs base_learned t1_learned
    python -u scripts/v2_experiments.py            # all configs, all folds
"""
import sys, time, argparse, json
import numpy as np
from pathlib import Path

if not hasattr(np, 'asfarray'):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.vifi_mot import list_sequences, load_sequence
from trackers.wifi_joint_v2 import WiFiJointTrackerV2
from denoising.inference import FTMDenoiser
from eval.mot_eval import Evaluator
from scripts.phase6b_integrated_mot import (
    _frame_dets_xyxy_scores, _frame_det_depths, _frame_raw_ftm,
    load_matching_model)

SCENES = ["scene1", "scene2", "scene3", "scene4"]
ROOT = Path(__file__).resolve().parent.parent

BASE_KW = dict(
    max_age=30, min_hits=3, iou_threshold=0.3, delta_t=3, inertia=0.2,
    wifi_weight=0.10, depth_sigma=1.3, ema_alpha=0.85,
    bind_init_threshold=0.7, unbind_threshold=0.15,
    switch_penalty=0.30, keep_bonus=0.20, assign_threshold=0.40,
    no_ghost_pool=True,
)

CONFIGS = {
    # guards: must reproduce Phase 6B
    "base_gaussian": dict(learned=False),
    "base_learned":  dict(learned=True),
    # T1: trajectory derivative-correlation in phone binding
    "t1_learned":    dict(learned=True,  traj_lambda=0.3, traj_window=30),
    "t1_gaussian":   dict(learned=False, traj_lambda=0.3, traj_window=30),
    # T2: uncertainty-normalized compat (gaussian branch only)
    "t2_gaussian":   dict(learned=False, adaptive_sigma=True, sigma0=0.9, zed_a=0.01),
    "t2b_gaussian":  dict(learned=False, adaptive_sigma=True, sigma0=1.1,
                          zed_a=0.01, lr_norm=False),
    "t2c_gaussian":  dict(learned=False, adaptive_sigma=True, sigma0=1.2,
                          zed_a=0.015, lr_norm=False),
    # T3: occlusion-aware modulation
    "t3_learned":    dict(learned=True,  occl_tau=0.5, occl_kappa=2.0,
                          occl_depth_gate=True),
    "t3a_learned":   dict(learned=True,  occl_tau=0.5, occl_kappa=2.0),
    "t3b_learned":   dict(learned=True,  occl_depth_gate=True),
    "t3c_learned":   dict(learned=True,  occl_tau=0.6, occl_kappa=1.0),  # convex blend fix
    # combos
    "t1t3_learned":  dict(learned=True,  traj_lambda=0.3, traj_window=30,
                          occl_tau=0.5, occl_kappa=2.0, occl_depth_gate=True),
    # T4: heteroscedastic denoiser sigma-hat drives adaptive compat
    "t4_gaussian":   dict(learned=False, hetero=True, adaptive_sigma=True,
                          sigma0=0.9, zed_a=0.01),
    "t4_learned":    dict(learned=True,  hetero=True),
    # T5: risk-deferred binding
    "t5_learned":    dict(learned=True,  bind_margin=0.10),
    # T7: offline tracklet merging (motion + depth bridging)
    "t7_learned":    dict(learned=True,  merge=True),
    # T8: SPRT sequential binding (Wald test replaces EMA hysteresis)
    "t8_learned":    dict(learned=True,  sprt=True),
    "t8t3b_learned": dict(learned=True,  sprt=True, occl_depth_gate=True),
    # T9: uncertainty-aware matching MLP (hetero mu/sigma features, 7-dim)
    "t9":            dict(hetero=True, hetero_matching=True),
    "t9t3b":         dict(hetero=True, hetero_matching=True, occl_depth_gate=True),
    "t9t3bt8":       dict(hetero=True, hetero_matching=True, occl_depth_gate=True,
                          sprt=True),
    # T10/T11: decoupled — hetero mu replaces denoised value (old MLP keeps
    # reported std feature); sigma-hat only gates the wifi weight
    "t10":           dict(learned=True, hetero_mu_only=True),
    "t10t11":        dict(learned=True, hetero_mu_only=True, wifi_gate=True),
    "t10t3b":        dict(learned=True, hetero_mu_only=True, occl_depth_gate=True),
    "t10t11t3b":     dict(learned=True, hetero_mu_only=True, wifi_gate=True,
                          occl_depth_gate=True),
    # T12: 6-dim MLP retrained on hetero-mu features (no distribution shift)
    "t12":           dict(hmu_matching=True, hetero_mu_only=True),
    "t12t11":        dict(hmu_matching=True, hetero_mu_only=True, wifi_gate=True),
    "t12t3b":        dict(hmu_matching=True, hetero_mu_only=True,
                          occl_depth_gate=True),
    "t12t11t3b":     dict(hmu_matching=True, hetero_mu_only=True, wifi_gate=True,
                          occl_depth_gate=True),
    # T14: dual-denoiser mu ensemble (zero retrain, original 6-dim MLP)
    "t14":           dict(learned=True, mu_avg=True),
    "t14t3b":        dict(learned=True, mu_avg=True, occl_depth_gate=True),
    "t14t11":        dict(learned=True, mu_avg=True, wifi_gate=True),
    "t14t11t3b":     dict(learned=True, mu_avg=True, wifi_gate=True,
                          occl_depth_gate=True),
    # T15: dual-denoiser feature-fusion MLP (12-dim, per-sample arbiter)
    "t15":           dict(dual_matching=True),
    "t15t3b":        dict(dual_matching=True, occl_depth_gate=True),
    "t15t11":        dict(dual_matching=True, wifi_gate=True),
    "t15t11t3b":     dict(dual_matching=True, wifi_gate=True,
                          occl_depth_gate=True),
    # ablation completion: sigma-hat gate WITHOUT arbiter matching
    "t11_only":      dict(learned=True, sigma_only=True, wifi_gate=True),
    "t11t3b":        dict(learned=True, sigma_only=True, wifi_gate=True,
                          occl_depth_gate=True),
    # S11 leakage fix: identical to base_learned except the matching MLP was
    # retrained with the test scene EXCLUDED from its training pool
    "base_clean":    dict(clean_matching=True),
    "clean_t3b":     dict(clean_matching=True, occl_depth_gate=True),
}


def run_fold(test_scene: str, cfg: dict, model_name="causal_cnn", window_size=15):
    seqs = [s for s in list_sequences(scenes=[test_scene]) if s.kind == "outdoor"]
    use_hetero = cfg.get("hetero", False) or cfg.get("hetero_mu_only", False)
    use_avg = cfg.get("mu_avg", False)
    use_dual = cfg.get("dual_matching", False)
    use_sigma_only = cfg.get("sigma_only", False)
    if use_avg or use_dual or use_sigma_only:
        from denoising.hetero import HeteroFTMDenoiser
        denoiser = FTMDenoiser(model_name, test_scene, window_size)
        denoiser_h = HeteroFTMDenoiser(test_scene, window_size)
    elif use_hetero:
        from denoising.hetero import HeteroFTMDenoiser
        denoiser = HeteroFTMDenoiser(test_scene, window_size)
    else:
        denoiser = FTMDenoiser(model_name, test_scene, window_size)
    if cfg.get("hetero_matching") or cfg.get("hmu_matching") or use_dual \
            or cfg.get("clean_matching"):
        import torch
        from scripts.phase6b_learned_matching import MatchingMLP
        if use_dual:
            tag, nfeat = "dual", 12
        elif cfg.get("hetero_matching"):
            tag, nfeat = "hetero", 7
        elif cfg.get("clean_matching"):
            tag, nfeat = "clean", 6
        else:
            tag, nfeat = "hmu", 6
        ck = torch.load(ROOT / "checkpoints_matching" /
                        f"matching_{tag}_{test_scene}_best.pth",
                        map_location="cpu", weights_only=False)
        matching_model = MatchingMLP(in_features=nfeat, hidden=16)
        matching_model.load_state_dict(ck["model_state"])
        matching_model.eval()
    else:
        matching_model = load_matching_model(test_scene) if cfg.get("learned") else None

    kw = dict(BASE_KW)
    kw.update({k: v for k, v in cfg.items()
               if k not in ("learned", "hetero", "merge", "hetero_matching",
                            "hetero_mu_only", "hmu_matching", "mu_avg",
                            "dual_matching", "sigma_only", "clean_matching")})

    evaluator = Evaluator()
    for meta in seqs:
        frames = load_sequence(meta)
        if not frames:
            continue
        tracker = WiFiJointTrackerV2(matching_model=matching_model, **kw)
        if cfg.get("hetero_matching"):
            tracker.hetero_matching = True
        if use_dual:
            tracker.dual_matching = True
        seq_id_map, gt_seq, hyp_seq = {}, [], []
        depth_by_frame = {}
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
            std_for_tracker = ftm_std_m
            if len(ftm_m) > 0 and ftm_valid.any():
                if use_dual:
                    denoised = denoiser.denoise(ftm_m, ftm_valid, ftm_std_m)
                    mu_h, sig = denoiser_h.denoise(ftm_m, ftm_valid, ftm_std_m)
                    tracker._mu_h = mu_h
                    tracker._sigma_hat = sig
                elif use_sigma_only:
                    denoised = denoiser.denoise(ftm_m, ftm_valid, ftm_std_m)
                    _, sig = denoiser_h.denoise(ftm_m, ftm_valid, ftm_std_m)
                    tracker._sigma_hat = sig
                elif use_avg:
                    den_c = denoiser.denoise(ftm_m, ftm_valid, ftm_std_m)
                    mu_h, sig = denoiser_h.denoise(ftm_m, ftm_valid, ftm_std_m)
                    denoised = np.where(np.isnan(sig), den_c,
                                        0.5 * (den_c + mu_h))
                    tracker._sigma_hat = sig
                elif use_hetero:
                    denoised, sig = denoiser.denoise(ftm_m, ftm_valid, ftm_std_m)
                    if cfg.get("hetero_mu_only"):
                        # mu replaces range; MLP keeps reported-std feature;
                        # sigma-hat only gates the wifi weight (T11)
                        tracker._sigma_hat = sig
                    else:
                        std_for_tracker = np.where(np.isnan(sig), ftm_std_m, sig)
                else:
                    denoised = denoiser.denoise(ftm_m, ftm_valid, ftm_std_m)
            else:
                denoised = ftm_m
            outputs = tracker.update(dets, depths, scores, denoised,
                                     ftm_valid, std_for_tracker)
            hyp_items = [(int(t), (float(b[0]), float(b[1]),
                                   float(b[2] - b[0]), float(b[3] - b[1])))
                         for t, b in outputs]
            hyp_seq.append((frame.frame_id, hyp_items))
            if cfg.get("merge") and len(dets) > 0:
                cx = (dets[:, 0] + dets[:, 2]) / 2
                cy = (dets[:, 1] + dets[:, 3]) / 2
                for t, b in outputs:
                    bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                    k = int(np.argmin((cx - bx) ** 2 + (cy - by) ** 2))
                    if not np.isnan(depths[k]):
                        depth_by_frame[(frame.frame_id, int(t))] = float(depths[k])
        if cfg.get("merge"):
            from trackers.tracklet_merge import merge_tracklets
            hyp_seq = merge_tracklets(hyp_seq, depth_by_frame=depth_by_frame)
        evaluator.add_sequence(meta.name, gt_seq, hyp_seq)

    summary = evaluator.summarize()
    o = summary.loc["OVERALL"].to_dict() if "OVERALL" in summary.index else {}
    per_seq = {str(i): round(float(summary.loc[i, "idf1"]) * 100, 2)
               for i in summary.index if i != "OVERALL"}
    return {"scene": test_scene,
            "idf1": float(o.get("idf1", 0)) * 100,
            "mota": float(o.get("mota", 0)) * 100,
            "idsw": int(o.get("num_switches", 0)),
            "per_seq": per_seq}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS.keys()))
    ap.add_argument("--scenes", nargs="+", default=SCENES)
    ap.add_argument("--out", default="exps/v2_experiments.json")
    args = ap.parse_args()

    all_res = {}
    prev = {}
    out_path = ROOT / args.out
    if out_path.exists():
        prev = json.loads(out_path.read_text())
        all_res.update(prev)

    for name in args.configs:
        cfg = CONFIGS[name]
        rows = all_res.get(name, [])
        done = {r["scene"] for r in rows}
        for scene in args.scenes:
            if scene in done:
                continue
            t0 = time.time()
            r = run_fold(scene, cfg)
            rows.append(r)
            print(f"{name:>14} | {scene} | IDF1 {r['idf1']:6.2f}% | "
                  f"MOTA {r['mota']:6.2f}% | IDsw {r['idsw']:4d} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        all_res[name] = rows
        if len(rows) == 4:
            print(f"{name:>14} | AVG    | IDF1 {np.mean([r['idf1'] for r in rows]):6.2f}% | "
                  f"MOTA {np.mean([r['mota'] for r in rows]):6.2f}% | "
                  f"IDsw {np.mean([r['idsw'] for r in rows]):6.1f}", flush=True)
        out_path.write_text(json.dumps(all_res, indent=2))

    print("\n==== SUMMARY (4-fold LOSO avg) ====")
    for name, rows in all_res.items():
        if len(rows) == 4:
            print(f"{name:>14}: IDF1 {np.mean([r['idf1'] for r in rows]):.2f}% "
                  f"IDsw {np.mean([r['idsw'] for r in rows]):.1f}")


if __name__ == "__main__":
    main()
