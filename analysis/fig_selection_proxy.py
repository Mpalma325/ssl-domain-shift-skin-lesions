"""Label selection in the frozen-feature proxy, k = 16, Section 5.2.1: figure selection_proxy.

Boxes over the 30 draws of the pool (or the 6 restarts of the oracle search, BCN20000 only); numbers are medians;
dashed line at the median of 80 random images.
Inputs: results/study2/selection_proxy_draws.csv and selection_proxy_oracle.csv (`--recompute` rebuilds them from
features/). Outputs: figures/selection_proxy.pdf, tables/selection_proxy.csv.
"""
import argparse

from matplotlib.patches import Patch

import common as C
import selection as S
from fig_label_selection import boxes

plt = C.plt
GROUPS = {"ref": ("#9ecae1", "#2166ac", "References"), "al": ("#d9d9d9", "#4d4d4d", "Cold-start strategies")}
NAMES = {"random_80": "80 random images", "random_per_class": "16 random per class (Study 1)", "oracle": "Oracle",
         "class_prototypes": "Class-aware prototypes", "kcenter": "k-center greedy",
         "kmeans_centers": "k-means centers", "typiclust": "TypiClust", "probcover": "ProbCover",
         "maxherding": "MaxHerding", "pseudo_class_balance": "Pseudo-class balance"}
ROWS = [("ref", "random_80"), ("ref", "random_per_class"), ("ref", "oracle"), ("ref", "class_prototypes")] + \
    [("al", s) for s in S.COLD_START]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true", help="rebuild the per-draw results from features/ first")
    a = ap.parse_args()
    d = S.results("draws", a.recompute)
    o = S.results("oracle", a.recompute)
    data = {(t, s): g.ba_test.to_numpy() for (t, s), g in d.groupby(["target", "selection"])}
    data["bcn20000", "oracle"] = o[o.selection == "oracle"].ba_test.to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(C.WIDTH, 3.8), sharey=True)
    table = boxes(axes, data, ROWS, NAMES, GROUPS)
    fig.legend(handles=[Patch(facecolor=f, edgecolor=e, label=n) for f, e, n in GROUPS.values()],
               loc="upper center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.6, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=1.5)
    C.save(fig, "selection_proxy")
    C.save_table(table, "selection_proxy")


if __name__ == "__main__":
    main()
