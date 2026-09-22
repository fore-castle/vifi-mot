"""Decompose the pooled-vs-per-sequence discrepancy.

Observed conflict:
  pooled (count-weighted, standard MOT protocol) : ours 88.22 vs base 87.91  -> +0.31
  per-sequence mean (s10, denoiser reset per seq): mean delta -1.10, CI [-2.02,-0.30]

Two candidate causes:
  (a) WEIGHTING  - pooled IDF1 weights sequences by their identity counts, so long /
      crowded sequences dominate; an unweighted mean over 67 sequences does not.
  (b) DENOISER STATE - the main pipeline creates one FTMDenoiser per fold and never
      calls reset(), so its sliding window carries over between sequences. s10 reset
      it per sequence.

This script isolates (a) by using the MAIN pipeline's own per-sequence IDF1 values
(same no-reset behaviour, same code path), so the only difference from the headline
number is pooled vs unweighted aggregation.

Inputs (produced by scripts/v2_experiments.py, which records per_seq):
  mot_baseline/exps/v2_base_perseq.json  -> base_learned
  mot_baseline/exps/v2_t15.json          -> t15t11t3b
"""
import json
from pathlib import Path

import numpy as np

MOT = Path("/Users/zstar/auto_search/vifi-mot/mot_baseline")
OUT = Path(__file__).resolve().parent


def collect(path, cfg):
    d = json.load(open(path))[cfg]
    out = {}
    for fold in d:
        for seq, idf1 in fold["per_seq"].items():
            out[seq] = idf1
    return out, {f["scene"]: f["idf1"] for f in d}


def main():
    base, base_fold = collect(MOT / "exps/v2_base_perseq.json", "base_learned")
    ours, ours_fold = collect(MOT / "exps/v2_t15.json", "t15t11t3b")

    common = sorted(set(base) & set(ours))
    d = np.array([ours[s] - base[s] for s in common])
    n = len(d)
    mean, std = float(d.mean()), float(d.std(ddof=1))
    se = std / np.sqrt(n)

    rng = np.random.RandomState(0)
    boots = np.array([rng.choice(d, n, replace=True).mean() for _ in range(10000)])
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    try:
        from scipy.stats import wilcoxon
        p_two = float(wilcoxon(d)[1])
    except Exception:
        p_two = None

    print(f"sequences compared        : {n}")
    print(f"pooled fold means  base   : "
          f"{np.mean(list(base_fold.values())):.2f}")
    print(f"pooled fold means  ours   : "
          f"{np.mean(list(ours_fold.values())):.2f}")
    print(f"  -> pooled delta         : "
          f"{np.mean(list(ours_fold.values())) - np.mean(list(base_fold.values())):+.2f} pt")
    print()
    print(f"unweighted per-seq mean   : {mean:+.3f} pt (std {std:.3f}, SE {se:.3f})")
    print(f"sign                      : {int((d>0).sum())} pos / {int((d<0).sum())} neg")
    print(f"bootstrap 95% CI          : [{ci[0]:+.3f}, {ci[1]:+.3f}]")
    if p_two is not None:
        print(f"Wilcoxon two-sided p      : {p_two:.4f}")

    # which sequences drive the pooled gain? sort by |delta|
    rows = sorted(((s, base[s], ours[s], ours[s] - base[s]) for s in common),
                  key=lambda r: r[3])
    print("\nworst 5:")
    for s, b, o, dl in rows[:5]:
        print(f"  {s:>34} {b:6.2f} -> {o:6.2f}  {dl:+6.2f}")
    print("best 5:")
    for s, b, o, dl in rows[-5:]:
        print(f"  {s:>34} {b:6.2f} -> {o:6.2f}  {dl:+6.2f}")

    (OUT / "decomposition.json").write_text(json.dumps({
        "n": n, "per_seq_mean_delta": mean, "std": std, "se": se,
        "bootstrap_ci95": ci, "wilcoxon_p_two_sided": p_two,
        "n_positive": int((d > 0).sum()), "n_negative": int((d < 0).sum()),
        "pooled_base": base_fold, "pooled_ours": ours_fold,
        "per_sequence": {s: {"base": base[s], "ours": ours[s],
                             "delta": ours[s] - base[s]} for s in common},
    }, indent=2))
    print(f"\nsaved -> {OUT/'decomposition.json'}")


if __name__ == "__main__":
    main()
