"""Phase 0.3: IDsw distribution + occlusion analysis.

Uses existing OC-SORT results (exps/ocsort_all_constant.json) for IDsw data,
and loads sequence data for occlusion gap statistics only.

Usage:
    /Users/zstar/miniforge3/bin/python3 scripts/phase0_03_idsw_occlusion.py
"""
from __future__ import annotations

import os, sys, json, collections
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.vifi_mot import list_sequences, load_sequence


def load_existing_idsw(scenes: List[str]) -> Dict[str, dict]:
    """Load per-sequence IDsw from existing OC-SORT JSON results."""
    results = {}
    for scene in scenes:
        json_path = os.path.join(os.path.dirname(__file__), "..",
                                 "exps", "ocsort_all_constant.json")
        if not os.path.exists(json_path):
            print(f"  WARNING: {json_path} not found, skipping")
            continue
        with open(json_path) as f:
            data = json.load(f)

        # Find the entry for this scene
        for entry in data:
            if entry.get("test_scene") == scene:
                summary = entry["summary"]
                seq_idsw = {}
                for key in summary.get("num_switches", {}):
                    if key == "OVERALL":
                        continue
                    # key format: "outdoor-scene1-20211004_142306"
                    seq_name = key.split("-", 2)[-1] if key.count("-") >= 2 else key
                    seq_idsw[seq_name] = {
                        "idsw": summary["num_switches"][key],
                        "n_frames": summary["num_frames"].get(key, 0),
                        "n_unique": summary["num_unique_objects"].get(key, 0),
                        "idf1": summary.get("idf1", {}).get(key, 0),
                    }
                results[scene] = {
                    "total_idsw": summary["num_switches"].get("OVERALL", 0),
                    "total_idf1": summary.get("idf1", {}).get("OVERALL", 0),
                    "per_sequence": seq_idsw,
                }
    return results


def compute_occlusion_stats(scenes: List[str]) -> Dict[str, dict]:
    """Compute detection gap statistics per scene (no tracker needed)."""
    results = {}

    for scene in scenes:
        seqs = list_sequences(scenes=[scene])
        all_gaps = []
        long_gap_events = []
        seq_stats = {}

        for meta in seqs:
            frames = load_sequence(meta)
            gt_last_seen: Dict[str, int] = {}
            gt_gaps: Dict[str, List[int]] = collections.defaultdict(list)

            for fr in frames:
                for det in fr.detections:
                    gt_id = det.gt_track_id
                    if gt_id in gt_last_seen:
                        gap = fr.frame_id - gt_last_seen[gt_id] - 1
                        if gap > 0:
                            gt_gaps[gt_id].append(gap)
                    gt_last_seen[gt_id] = fr.frame_id

            gaps_all = []
            for g_list in gt_gaps.values():
                gaps_all.extend(g_list)
            long_gaps = [g for g in gaps_all if g >= 5]
            all_gaps.extend(long_gaps)

            for gt_id, g_list in gt_gaps.items():
                for g in g_list:
                    if g >= 5:
                        long_gap_events.append({
                            "seq": meta.seq_id,
                            "gt_id": gt_id,
                            "gap_frames": g,
                        })

            seq_stats[meta.seq_id] = {
                "n_frames": len(frames),
                "n_gt_tracks": len(gt_last_seen),
                "total_gaps": len(gaps_all),
                "gaps_ge_5": len(long_gaps),
                "gaps_ge_10": len([g for g in gaps_all if g >= 10]),
                "max_gap": max(gaps_all) if gaps_all else 0,
            }

        results[scene] = {
            "n_sequences": len(seqs),
            "total_occlusion_events_ge5": len(long_gap_events),
            "gap_duration_stats_ge5": {
                "count": len(all_gaps),
                "mean": float(np.mean(all_gaps)) if all_gaps else 0,
                "max": max(all_gaps) if all_gaps else 0,
                "median": float(np.median(all_gaps)) if all_gaps else 0,
            },
            "per_sequence": seq_stats,
        }

    return results


def main():
    scenes = ["scene1", "scene2", "scene3", "scene4"]
    out_dir = os.path.join(os.path.dirname(__file__), "..", "exps")

    print("=" * 70)
    print("Experiment 0.3: IDsw Distribution + Occlusion Analysis")
    print("=" * 70)

    # --- IDsw from existing results ---
    print("\n>>> IDsw distribution (from existing OC-SORT results)")
    print("-" * 50)
    idsw_data = load_existing_idsw(scenes)

    for scene, data in idsw_data.items():
        print(f"\n  [{scene}]  total IDsw = {data['total_idsw']}, "
              f"IDF1 = {data['total_idf1']:.4f}")
        per_seq = data["per_sequence"]
        sorted_seqs = sorted(per_seq.items(), key=lambda x: x[1]["idsw"], reverse=True)

        print(f"    Top 5 IDsw sequences:")
        for seq_id, d in sorted_seqs[:5]:
            print(f"      {seq_id}: IDsw={d['idsw']}, frames={d['n_frames']}, "
                  f"unique_obj={d['n_unique']}, IDF1={d['idf1']:.4f}")

        print(f"    Bottom 3 (best) sequences:")
        for seq_id, d in sorted_seqs[-3:]:
            print(f"      {seq_id}: IDsw={d['idsw']}, frames={d['n_frames']}, "
                  f"unique_obj={d['n_unique']}, IDF1={d['idf1']:.4f}")

    # --- Occlusion stats ---
    print("\n\n>>> Occlusion gap statistics (from raw data)")
    print("-" * 50)
    occ_data = compute_occlusion_stats(scenes)

    for scene, data in occ_data.items():
        gap_s = data["gap_duration_stats_ge5"]
        print(f"\n  [{scene}]  {data['n_sequences']} sequences")
        print(f"    occlusion events (gap≥5):  {data['total_occlusion_events_ge5']}")
        print(f"    gap≥5 stats: count={gap_s['count']}, "
              f"mean={gap_s['mean']:.1f}, max={gap_s['max']}, "
              f"median={gap_s['median']:.1f}")

        # Per-sequence occlusion summary
        per_seq = data["per_sequence"]
        sorted_seqs = sorted(per_seq.items(), key=lambda x: x[1]["gaps_ge_5"], reverse=True)
        print(f"    Top 5 occluded sequences:")
        for seq_id, d in sorted_seqs[:5]:
            print(f"      {seq_id}: gaps≥5={d['gaps_ge_5']}, gaps≥10={d['gaps_ge_10']}, "
                  f"max_gap={d['max_gap']}, frames={d['n_frames']}")

    # --- Cross-reference: IDsw vs occlusion ---
    print("\n\n>>> Cross-reference: High-IDsw sequences vs occlusion")
    print("-" * 50)
    for scene in scenes:
        if scene not in idsw_data or scene not in occ_data:
            continue
        idsw_seqs = idsw_data[scene]["per_sequence"]
        occ_seqs = occ_data[scene]["per_sequence"]

        # Find sequences with both high IDsw and high occlusion
        common_seqs = set(idsw_seqs.keys()) & set(occ_seqs.keys())
        combined = []
        for sq in common_seqs:
            combined.append({
                "seq": sq,
                "idsw": idsw_seqs[sq]["idsw"],
                "gaps_ge_5": occ_seqs[sq]["gaps_ge_5"],
                "max_gap": occ_seqs[sq]["max_gap"],
            })
        combined.sort(key=lambda x: x["idsw"], reverse=True)

        print(f"\n  [{scene}] High-IDsw sequences:")
        for c in combined[:5]:
            occl_tag = "OCCLUDED" if c["gaps_ge_5"] > 0 else "no_occlusion"
            print(f"    {c['seq']}: IDsw={c['idsw']}, gaps≥5={c['gaps_ge_5']} [{occl_tag}]")

    # Save results
    combined_output = {
        "idsw_distribution": idsw_data,
        "occlusion_statistics": occ_data,
    }
    out_path = os.path.join(out_dir, "phase0_03_idsw_occlusion.json")
    with open(out_path, "w") as f:
        json.dump(combined_output, f, indent=2, default=str)
    print(f"\n  Saved to exps/phase0_03_idsw_occlusion.json")


if __name__ == "__main__":
    main()
