"""SSL - ERM by method family, HAM10000 init., Section 4.2 (Overall Effect of SSL): figure ssl_families.

Per seed: mean over the family's methods and over k; cell = mean over 5 seeds, frame = 95 % t interval excludes zero.
"All" pools the three targets. Input: common.panel() (primary test set).
Outputs: figures/ssl_families.pdf, tables/ssl_families.csv.
"""
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm
from scipy import stats

import common as C

plt = C.plt
COLUMNS = {"all": "All", "bcn20000": "BCN20000", "milk10k": "MILK10k\ndermoscopic",
           "milk10k_clinico": "MILK10k\nclinical"}


def family_table():
    dd = C.deltas(C.design())
    ham = dd[dd.init == "ham"]
    rows = []
    for family, ms in C.FAMILIES:
        s = ham[ham.method.isin(ms)]
        for t in [None] + C.TARGETS:
            x = s if t is None else s[s.target == t]
            # per seed: mean over methods within each (target, k), then over (target, k)
            v = x.groupby(["target", "k", "seed"]).d_ba.mean().groupby("seed").mean().to_numpy()
            h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
            rows.append(dict(family=family.replace("\n", " "), methods=" + ".join(ms), target=t or "all",
                             n_seeds=len(v), d_ba=v.mean(), t_halfwidth=h, excludes_zero=abs(v.mean()) > h))
    return pd.DataFrame(rows)


def main():
    d = family_table()
    families = [f.replace("\n", " ") for f, _ in C.FAMILIES]
    labels = [f.replace("\n", " ").replace("augmentation and ", "augmentation and\n") for f, _ in C.FAMILIES]
    M = d.pivot(index="family", columns="target", values="d_ba").loc[families, list(COLUMNS)].to_numpy()
    S = d.pivot(index="family", columns="target", values="excludes_zero").loc[families, list(COLUMNS)]
    S = S.to_numpy(dtype=bool)

    norm = TwoSlopeNorm(0, -C.DIVERGING_LIMIT, C.DIVERGING_LIMIT)
    cmap = plt.get_cmap("RdBu")
    fig = plt.figure(figsize=(C.WIDTH, 2.7))
    ax = fig.add_axes([0.29, 0.04, 0.56, 0.80])
    cax = fig.add_axes([0.875, 0.04, 0.02, 0.80])
    ax.pcolormesh(M, cmap=cmap, norm=norm, edgecolors="white", linewidth=1.6)
    ax.invert_yaxis()
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j + 0.5, i + 0.5, C.r1(M[i, j]), ha="center", va="center", fontsize=9,
                    color=C.ink_on(M[i, j], cmap, norm))
            if S[i, j]:
                C.cell_frame(ax, j, i, sep_x={1: 5}, base=1.6)
    ax.axvline(1, color="white", lw=5)
    ax.set_xticks(np.arange(4) + 0.5, list(COLUMNS.values()))
    ax.set_yticks(np.arange(len(families)) + 0.5, labels)
    ax.xaxis.tick_top()
    ax.tick_params(length=0, labelsize=9.5)
    for label in ax.get_xticklabels():
        label.set_fontfamily(["cmb10", "DejaVu Serif"])
    for side in ax.spines.values():
        side.set_visible(False)

    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, extend="both")
    ticks = [-6, -4, -2, 0, 2, 4, 6]
    bar.set_ticks(ticks, labels=[f"${m}$" for m in ticks])
    bar.ax.tick_params(length=2, width=0.5, labelsize=8)
    bar.set_label("SSL $-$ ERM (pp)", fontsize=9.5, labelpad=8)
    bar.outline.set_visible(False)
    C.save(fig, "ssl_families")
    C.save_table(d, "ssl_families")


if __name__ == "__main__":
    main()
