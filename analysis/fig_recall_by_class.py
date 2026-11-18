"""Per-class recall change, SSL - ERM (figures recall_by_class and recall_by_class_imagenet).

Per method, target and class: per seed, mean over k of the paired recall difference on the primary test set; mean
over five seeds and 95 % t interval (frame: excludes zero).
Outputs: figures/recall_by_class[_imagenet].pdf/.png, tables/recall_by_class[_imagenet].csv.
"""
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm
from scipy import stats

import common as C

plt = C.plt
REC = ["rec_" + c for c in C.CLASSES]
LIM = 10


def compute(dd, init):
    d = dd[dd.init == init]
    rows = []
    for t in C.TARGETS:
        for m in C.SSL:
            s = d[(d.method == m) & (d.target == t)]
            for cl, col in zip(C.CLASSES_EN, REC):
                v = s.groupby("seed")["d_" + col].mean().to_numpy()
                h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
                rows.append(dict(init=init, target=t, method=m, cls=cl, n_seeds=len(v), d_recall=v.mean(),
                                 t_half_width=h, excludes_zero=abs(v.mean()) > h))
    return pd.DataFrame(rows)


def figure(r, name):
    norm = TwoSlopeNorm(0, -LIM, LIM)
    cmap = plt.get_cmap("RdBu")
    fig = plt.figure(figsize=(C.WIDTH, 3.9))
    width, x0, gap = 0.246, 0.152, 0.009
    sep_y, acc = {}, 0
    for _, ms in C.FAMILIES[:-1]:
        acc += len(ms)
        sep_y[acc] = 3
    for p, t in enumerate(C.TARGETS):
        d = r[r.target == t]
        M = d.pivot(index="method", columns="cls", values="d_recall").loc[C.SSL, C.CLASSES_EN].to_numpy()
        S = d.pivot(index="method", columns="cls", values="excludes_zero").loc[C.SSL, C.CLASSES_EN].to_numpy(bool)
        ax = fig.add_axes([x0 + p * (width + gap), 0.03, width, 0.83])
        ax.pcolormesh(M, cmap=cmap, norm=norm, edgecolors="white", linewidth=1.0)
        ax.invert_yaxis()
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j + 0.5, i + 0.5, C.r1(M[i, j]), ha="center", va="center", fontsize=6.3,
                        color=C.ink_on(np.clip(M[i, j], -LIM, LIM), cmap, norm))
                if S[i, j]:
                    C.cell_frame(ax, j, i, sep_y=sep_y, base=1.0)
        for pos, lw in sep_y.items():
            ax.axhline(pos, color="white", lw=lw)
        ax.set_xticks(np.arange(5) + 0.5, C.CLASSES_EN, fontsize=6.2)
        ax.xaxis.tick_top()
        ax.set_yticks(np.arange(len(C.SSL)) + 0.5, [C.NAME[m] for m in C.SSL] if p == 0 else [])
        ax.tick_params(axis="y", length=0, labelsize=8.5, pad=2)
        ax.tick_params(axis="x", length=0, pad=2)
        ax.set_title(C.TARGET_EN[t], fontsize=9.5, color=C.INK, fontfamily=["cmb10", "DejaVu Serif"], pad=16)
        for side in ax.spines.values():
            side.set_visible(False)
    cax = fig.add_axes([x0 + 3 * (width + gap) + 0.01, 0.03, 0.016, 0.83])
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, extend="both")
    ticks = [-10, -5, 0, 5, 10]
    bar.set_ticks(ticks, labels=[f"${t}$" for t in ticks])
    bar.ax.tick_params(length=2, width=0.5, labelsize=7.5)
    bar.set_label("SSL $-$ ERM, recall (pp)", fontsize=8.5, labelpad=6)
    bar.outline.set_visible(False)
    C.save(fig, name)


def main():
    dd = C.deltas(C.design(), cols=["ba"] + REC)
    for init, name in (("ham", "recall_by_class"), ("imagenet", "recall_by_class_imagenet")):
        r = compute(dd, init)
        C.save_table(r, name)
        figure(r, name)


if __name__ == "__main__":
    main()
