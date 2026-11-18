"""Random resized crop on the MILK10k clinical images (Chapter 3, Appendix 7.2).

tab:clinical_crop_review: blind review, 100 crops per scale (25 lesions x 4) labeled preserved / partial / lost.
tab:clinical_crop_ablation: ERM, HAM10000 init., scale [0.2, 1.0] vs [0.5, 1.0]; left block seeds 0-2 ([0.2, 1.0] at
200 iterations per epoch vs the main 500-iteration runs, paired by seed, 95 % t), right block seed 0, both at 200.
Inputs: results/clinical_crop_review.csv, clinical_crop_ablation_runs.csv, main predictions (primary test set).
Outputs: tables/clinical_crop_review.csv, tables/clinical_crop_ablation.csv.
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

import common as C


def review():
    r = pd.read_csv(os.path.join(C.RESULTS, "clinical_crop_review.csv"))
    t = pd.crosstab(r.scale_min, r.outcome).reindex(columns=["preserved", "partial", "lost"], fill_value=0)
    t["n"] = t.sum(axis=1)
    assert (t.n == 100).all()
    t["partial_or_lost_pct"] = 100 * (t.partial + t.lost) / t.n
    t.index = [f"[{s}, 1.0]" for s in t.index]
    C.save_table(t.rename_axis("crop_scale").reset_index(), "clinical_crop_review")


def ablation():
    a = pd.read_csv(os.path.join(C.RESULTS, "clinical_crop_ablation_runs.csv"))
    p = C.panel()
    main = p[(p.init == "ham") & (p.target == "milk10k_clinico") & (p.method == "erm")].set_index(["k", "seed"]).ba
    rows = []
    for k in C.KS:
        s02 = a[(a.k == k) & (a.crop_scale_min == 0.2) & (a.iters_per_epoch == 200)].set_index("seed").ba
        s05 = a[(a.k == k) & (a.crop_scale_min == 0.5) & (a.iters_per_epoch == 200)].set_index("seed").ba
        seeds = [0, 1, 2]
        x, y = s02.loc[seeds].values, main.loc[[(k, s) for s in seeds]].values
        d = x - y
        half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
        rows.append({"k": k,
                     "ba_s02_it200_seeds012": x.mean(), "ba_s05_it500_seeds012": y.mean(),
                     "diff_mean": d.mean(), "diff_ci95_half": half, "positive_seeds": f"{(d > 0).sum()}/3",
                     "ba_s02_it200_seed0": s02.loc[0], "ba_s05_it200_seed0": s05.loc[0],
                     "diff_seed0": s02.loc[0] - s05.loc[0]})
    C.save_table(pd.DataFrame(rows), "clinical_crop_ablation")


if __name__ == "__main__":
    review()
    ablation()
