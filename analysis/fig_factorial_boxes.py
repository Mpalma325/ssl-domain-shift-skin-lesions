"""Labeled subset x training seed factorial, Section 4.2 (Labeled Subset): figure factorial_boxes.

k = 16, HAM10000 init., MILK10k dermoscopic and BCN20000, ERM and FixMatch; 3 subset seeds x 3 training seeds, three
cells repeated (averaged first). One box per subset and method over the training seeds.
Input: results/factorial_runs.csv (runs outside the main design; ba on the primary test set).
Outputs: figures/factorial_boxes.pdf, tables/factorial_boxes.csv.
"""
import os

import numpy as np
import pandas as pd
from matplotlib.patches import Patch

import common as C

plt = C.plt
TARGETS = ["milk10k", "bcn20000"]
DX = 0.19
STYLE = {"erm": (-DX, "#d9d9d9", "#4d4d4d"), "fixmatch": (DX, "#9ecae1", "#2166ac")}  # offset, fill, edge


def factorial_runs():
    return pd.read_csv(os.path.join(C.RESULTS, "factorial_runs.csv"))


def main():
    p = factorial_runs().groupby(["target", "method", "subset_seed", "train_seed"], as_index=False).ba.mean()
    fig, axes = plt.subplots(1, 2, figsize=(C.WIDTH, 2.9), sharey=True)
    rows = []
    for ax, t in zip(axes, TARGETS):
        for m, (dx, fill, edge) in STYLE.items():
            data = [p[(p.target == t) & (p.method == m) & (p.subset_seed == ss)].ba.to_numpy() for ss in range(3)]
            pos = [ss + dx for ss in range(3)]
            ax.boxplot(data, positions=pos, widths=0.32, whis=(0, 100), patch_artist=True, showcaps=False,
                       vert=False, boxprops=dict(facecolor=fill, edgecolor=edge, lw=0.7),
                       medianprops=dict(color=edge, lw=1.0), whiskerprops=dict(color=edge, lw=0.7))
            for ss, (y, v) in enumerate(zip(pos, data)):
                ax.scatter(v, [y] * len(v), s=9, color=edge, zorder=3)
                above = m == "erm"  # y axis is inverted
                ax.text(np.median(v), y - 0.19 if above else y + 0.19, "%.1f" % np.median(v), ha="center",
                        va="bottom" if above else "top", fontsize=7, color=edge)
                rows.append(dict(target=t, method=m, subset=ss + 1, n=len(v), median=np.median(v), min=v.min(),
                                 q1=np.percentile(v, 25), q3=np.percentile(v, 75), max=v.max()))
        ax.set_yticks(range(3), ["Subset 1", "Subset 2", "Subset 3"])
        ax.set_ylim(2.6, -0.6)
        ax.set_xlabel("Test balanced accuracy (%)", fontsize=9)
        ax.set_title(C.TARGET_EN[t], color=C.INK, fontsize=10, fontfamily=["cmb10", "DejaVu Serif"])
        ax.tick_params(labelsize=8.5, length=2)
        ax.grid(axis="x", color=C.GRID, lw=0.7)
    handles = [Patch(facecolor=STYLE[m][1], edgecolor=STYLE[m][2], label=C.NAME[m]) for m in STYLE]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.55, 1.01))
    fig.subplots_adjust(left=0.11, right=0.99, top=0.82, bottom=0.16, wspace=0.08)
    C.save(fig, "factorial_boxes")
    C.save_table(pd.DataFrame(rows), "factorial_boxes")


if __name__ == "__main__":
    main()
