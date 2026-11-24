"""Heuristics against their two controls in the proxy, k = 16, Section 5.2.1 and Appendix 7.4.

Per draw (30), the balanced accuracy of each heuristic minus that of its composition control and of its
recognized-image control (each the mean of 10 sets). selection_controls: MILK10k dermoscopic; selection_controls_bcn:
BCN20000. Number next to each box = mean. Input: results/study2/selection_proxy_controls.csv.
Outputs: figures/selection_controls*.pdf, tables/selection_controls.csv (mean and 95 % t half-width per panel).
"""
import numpy as np
import pandas as pd

import common as C
import selection as S
from tab_selection_controls import per_draw
from tab_selection_proxy import paired

plt = C.plt
RULES = [("agreement_rule", "Agreement rule"), ("mixed_rule", "Mixed rule"), ("sequential_rule", "Sequential rule")]
PANELS = [("composition", "Heuristic $-$ composition control"),
          ("recognized", "Heuristic $-$ recognized-image control")]


def draw(P):
    fill, edge = "#d9d9d9", "#4d4d4d"
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(C.WIDTH, 2.4), sharey=True, sharex=True)
    for ax, (ctl, title) in zip(axes, PANELS):
        ax.axvline(0, color=C.INK_SOFT, lw=0.8, ls="--", zorder=0)
        for i, (r, _) in enumerate(RULES):
            v = (P[r] - P[r + "|" + ctl]).to_numpy()
            ax.boxplot([v], positions=[i], widths=0.45, whis=(0, 100), patch_artist=True, vert=False,
                       boxprops=dict(facecolor=fill, edgecolor=edge, lw=0.7), medianprops=dict(color=edge, lw=1.0),
                       whiskerprops=dict(color=edge, lw=0.7), capprops=dict(color=edge, lw=0.7))
            ax.scatter(v, i + rng.uniform(-0.06, 0.06, len(v)), s=1.5, color=edge, lw=0, alpha=0.8, zorder=3)
            ax.text(v.max() + 0.3, i, "$%+.1f$" % v.mean(), ha="left", va="center", fontsize=7, color=edge)
        ax.set_ylim(len(RULES) - 0.5, -0.95)
        ax.text(0.3, -0.75, r"heuristic better $\rightarrow$", ha="left", va="center", fontsize=7, color=C.INK_SOFT)
        ax.text(-0.3, -0.75, r"$\leftarrow$ control better", ha="right", va="center", fontsize=7, color=C.INK_SOFT)
        ax.set_xlabel("Test balanced accuracy (pp)", fontsize=9)
        ax.set_title(title, color=C.INK, fontsize=10, fontfamily=["cmb10", "DejaVu Serif"])
        ax.tick_params(labelsize=8.5, length=2)
        ax.xaxis.set_major_locator(plt.MultipleLocator(5))
        ax.grid(axis="x", color=C.GRID, lw=0.7)
    axes[0].set_yticks(range(len(RULES)))
    axes[0].set_yticklabels([n for _, n in RULES], fontsize=8.5)
    fig.tight_layout(w_pad=1.5)
    return fig


def main():
    _, P = per_draw()
    rows = []
    for t, name in (("milk10k", "selection_controls"), ("bcn20000", "selection_controls_bcn")):
        C.save(draw(P.loc[t]), name)
        for r, _ in RULES:
            for ctl, _ in PANELS:
                m, lo, hi = paired(P.loc[t][r], P.loc[t][r + "|" + ctl])
                rows.append(dict(target=t, heuristic=r, control=ctl, mean=m, half_width=(hi - lo) / 2,
                                 excludes_zero=bool(lo > 0 or hi < 0)))
    C.save_table(pd.DataFrame(rows), "selection_controls")


if __name__ == "__main__":
    main()
