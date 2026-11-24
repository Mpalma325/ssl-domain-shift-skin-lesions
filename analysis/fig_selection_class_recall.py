"""Recall per class of the one-epoch ERM networks, each selection minus a reference, Section 5.2.1 and Appendix 7.4.

class_recall: against 80 random images; class_recall_study1: against the protocol of Study 1 (16 random per class).
Difference of means over the 8 sets with a Welch t interval; framed cell if it excludes zero.
Inputs: results/study2/selection_finetune*. Outputs: figures/class_recall*.pdf and tables/class_recall*.csv.
"""
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm

import common as C
import selection as S
from tab_selection_finetune import welch

plt = C.plt
LIMIT = 30
ORDER = ["random_per_class", "class_prototypes", "random_80", "agreement_rule", "mixed_rule", "sequential_rule"]


def differences(runs, ref):
    rows = []
    for t in S.TARGETS:
        r0 = runs[(runs.target == t) & (runs.selection == ref)]
        for s in [x for x in ORDER if x != ref]:
            g = runs[(runs.target == t) & (runs.selection == s)]
            for c in C.CLASSES:
                d, lo, hi = welch(g["rec_" + c], r0["rec_" + c])
                rows.append(dict(target=t, selection=s, cls=c.upper(), diff=d, lo=lo, hi=hi,
                                 excludes_zero=bool(lo > 0 or hi < 0)))
    return pd.DataFrame(rows)


def draw(res, ref):
    rows = [x for x in ORDER if x != ref]
    norm, cmap = TwoSlopeNorm(0, -LIMIT, LIMIT), plt.get_cmap("RdBu")
    fig = plt.figure(figsize=(C.WIDTH, 2.6))
    width, x0, gap = 0.34, 0.21, 0.02
    CL = [c.upper() for c in C.CLASSES]
    for p, t in enumerate(S.TARGETS):
        d = res[res.target == t]
        M = d.pivot(index="selection", columns="cls", values="diff").loc[rows, CL].to_numpy()
        F = d.pivot(index="selection", columns="cls", values="excludes_zero").loc[rows, CL].to_numpy(dtype=bool)
        ax = fig.add_axes([x0 + p * (width + gap), 0.03, width, 0.78])
        ax.pcolormesh(M, cmap=cmap, norm=norm, edgecolors="white", linewidth=1.0)
        ax.invert_yaxis()
        ax.axhline(2, color="white", lw=3)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j + 0.5, i + 0.5, C.r1(M[i, j]).replace("-0.0", "0.0"), ha="center", va="center",
                        fontsize=7, color=C.ink_on(np.clip(M[i, j], -LIMIT, LIMIT), cmap, norm))
                if F[i, j]:
                    C.cell_frame(ax, j, i, sep_y={2: 3}, base=1.0)
        ax.set_xticks(np.arange(5) + 0.5, CL, fontsize=7)
        ax.xaxis.tick_top()
        ax.set_yticks(np.arange(len(rows)) + 0.5, [S.FT_NAMES[s] for s in rows] if p == 0 else [])
        ax.tick_params(axis="y", length=0, labelsize=8.5, pad=2)
        ax.tick_params(axis="x", length=0, pad=2)
        ax.set_title(C.TARGET_EN[t], fontsize=9.5, color=C.INK, fontfamily=["cmb10", "DejaVu Serif"], pad=14)
        for side in ax.spines.values():
            side.set_visible(False)
    cax = fig.add_axes([x0 + 2 * (width + gap) + 0.005, 0.03, 0.016, 0.78])
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, extend="both")
    ticks = [-30, -15, 0, 15, 30]
    bar.set_ticks(ticks, labels=[f"${m}$" for m in ticks])
    bar.ax.tick_params(length=2, width=0.5, labelsize=7.5)
    bar.set_label("Selection $-$ %s, recall (pp)" % ("80 random" if ref == "random_80" else "Study 1"),
                  fontsize=8.5, labelpad=6)
    bar.outline.set_visible(False)
    return fig


def main():
    runs, _ = S.finetune()
    for ref, name in (("random_80", "class_recall"), ("random_per_class", "class_recall_study1")):
        res = differences(runs, ref)
        C.save(draw(res, ref), name)
        C.save_table(res, name)


if __name__ == "__main__":
    main()
