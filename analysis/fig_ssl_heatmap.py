"""Heatmaps of the 18 conditions (3 targets x 2 inits x 3 k), Section 4.2 (Overall Effect of SSL).

ssl_heatmap: SSL - ERM for the ten methods and their mean, last column = mean over conditions; frame = 95 % t interval
over 5 seeds excludes zero. ssl_ba_heatmap: balanced accuracy of ERM, the ten methods and their mean.
Input: common.panel() (primary test set). Outputs in figures/ and tables/. Stops if the HAM10000 "SSL mean" row does
not reproduce the group table of the thesis.
"""
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm
from scipy import stats

import common as C

plt = C.plt
COLS = [(t, a, k) for t in C.TARGETS for a in ("imagenet", "ham") for k in C.KS]
# SSL mean - ERM, HAM10000 init. (value, 95 % t interval excludes zero), as in the thesis group table
GROUP_TABLE = {("bcn20000", 16): (3.18, True), ("bcn20000", 32): (1.98, True), ("bcn20000", 100): (1.85, True),
               ("milk10k", 16): (-1.55, False), ("milk10k", 32): (-1.15, False), ("milk10k", 100): (-0.61, False),
               ("milk10k_clinico", 16): (0.32, False), ("milk10k_clinico", 32): (1.88, True),
               ("milk10k_clinico", 100): (1.66, False)}


def excludes_zero(v):
    """True if the 95 % t interval over seeds excludes zero."""
    v = np.asarray(v, float)
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return abs(v.mean()) > h


def sel(d, t, a, k):
    return (d.target == t) & (d.init == a) & (d.k == k)


def difference_matrix(dd):
    """Rows: ten SSL methods + mean of the ten. Value = mean over seeds of SSL - ERM."""
    V, S = [], []
    for m in C.SSL:
        dm = dd[dd.method == m]
        V.append([dm[sel(dm, *c)].d_ba.mean() for c in COLS])
        S.append([excludes_zero(dm[sel(dm, *c)].d_ba) for c in COLS])
    per_seed = dd.groupby(["target", "init", "k", "seed"]).d_ba.mean().reset_index()  # mean of the ten per seed
    V.append([per_seed[sel(per_seed, *c)].d_ba.mean() for c in COLS])
    S.append([excludes_zero(per_seed[sel(per_seed, *c)].d_ba) for c in COLS])
    V, S = np.array(V), np.array(S)
    for (t, k), (expected, sig) in GROUP_TABLE.items():
        j = COLS.index((t, "ham", k))
        assert abs(V[-1, j] - expected) < 0.006 and S[-1, j] == sig, (t, k, V[-1, j], S[-1, j])
    V = np.column_stack([V, V.mean(1)])
    S = np.column_stack([S, np.zeros(len(S), bool)])
    return V, S


def ba_matrix(df):
    """Rows: ERM, ten SSL methods, mean of the ten. Value = mean BA over seeds."""
    V = [[df[(df.method == m) & sel(df, *c)].ba.mean() for c in COLS] for m in ["erm"] + C.SSL]
    ssl = df[df.method != "erm"]
    V.append([ssl[sel(ssl, *c)].ba.mean() for c in COLS])
    V = np.array(V)
    return np.column_stack([V, V.mean(1)])


def draw(V, S, labels, row_seps, cmap, norm, clip, bar_label, ticks, height, extend="neither"):
    fig, ax = plt.subplots(figsize=(C.WIDTH, height))
    Vc = np.clip(V, *clip)
    ax.imshow(Vc, cmap=cmap, norm=norm, aspect="auto")
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            ax.text(j, i, C.r1(V[i, j]), ha="center", va="center", fontsize=6.5, color=C.ink_on(Vc[i, j], cmap, norm))
    for x in np.arange(-0.5, V.shape[1] + 0.5):
        ax.axvline(x, color="white", lw=2.0)
    for y in np.arange(-0.5, V.shape[0] + 0.5):
        ax.axhline(y, color="white", lw=2.0)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xticks(range(V.shape[1]), [str(k) for (_, _, k) in COLS] + ["mean"], fontsize=7)
    sep_x, sep_y = {}, {}
    for x in (5.5, 11.5, 17.5):  # between targets and before the mean column
        ax.axvline(x, color="white", lw=5)
        sep_x[x] = 5
    for x in (2.5, 8.5, 14.5):  # between initializations
        ax.axvline(x, color="white", lw=3)
        sep_x[x] = 3
    for y, lw in row_seps:
        ax.axhline(y, color="white", lw=lw)
        sep_y[y] = lw
    for j, t in enumerate(C.TARGETS):
        ax.text(6 * j + 2.5, -1.6, C.TARGET_EN[t], ha="center", color=C.INK, fontsize=9,
                fontfamily=["cmb10", "DejaVu Serif"])
        ax.text(6 * j + 1, -0.85, "ImageNet", ha="center", fontsize=7, color=C.INK_SOFT)
        ax.text(6 * j + 4, -0.85, "HAM10000", ha="center", fontsize=7, color=C.INK_SOFT)
    ax.set_xlabel(r"Labeled images per class ($k$)")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    p = ax.get_position()
    ax.set_position([p.x0, p.y0, p.width * 0.925, p.height])
    cax = fig.add_axes([p.x0 + p.width * 0.945, p.y0, 0.016, p.height])
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, extend=extend)
    bar.set_ticks(ticks, labels=[f"${m}$" for m in ticks])
    bar.ax.tick_params(length=2, width=0.5, labelsize=8)
    bar.set_label(bar_label, fontsize=9, labelpad=8)
    bar.outline.set_visible(False)
    if S is not None:  # frames after the axes position is fixed
        for i in range(S.shape[0]):
            for j in range(S.shape[1]):
                if S[i, j]:
                    C.cell_frame(ax, j - 0.5, i - 0.5, sep_x=sep_x, sep_y=sep_y, base=2.0, lw=1.3)
    return fig


def family_separators(offset):
    out, acc = [], offset
    for _, ms in C.FAMILIES[:-1]:
        acc += len(ms)
        out.append((acc - 0.5, 4))
    return out


def as_table(V, rows, S=None):
    cols = [f"{t}|{a}|k{k}" for t, a, k in COLS] + ["mean"]
    out = []
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            row = {"row": r, "column": c, "value": V[i, j]}
            if S is not None:
                row["excludes_zero"] = bool(S[i, j])
            out.append(row)
    return pd.DataFrame(out)


def main():
    df = C.design()
    dd = C.deltas(df)
    names = [C.NAME[m] for m in C.SSL]
    n = len(C.SSL)

    V, S = difference_matrix(dd)
    fig = draw(V, S, names + ["SSL mean"], family_separators(0) + [(n - 0.5, 6)], plt.get_cmap("RdBu"),
               TwoSlopeNorm(0, -C.DIVERGING_LIMIT, C.DIVERGING_LIMIT), (-C.DIVERGING_LIMIT, C.DIVERGING_LIMIT),
               "SSL $-$ ERM (pp)", [-6, -4, -2, 0, 2, 4, 6], 4.75, extend="both")
    C.save(fig, "ssl_heatmap")
    C.save_table(as_table(V, names + ["SSL mean"], S), "ssl_heatmap")

    B = ba_matrix(df)
    lo, hi = np.floor(B.min() / 2) * 2, np.ceil(B.max() / 2) * 2
    fig = draw(B, None, ["ERM"] + names + ["SSL mean"], [(0.5, 6)] + family_separators(1) + [(n + 0.5, 6)],
               plt.get_cmap("Blues"), Normalize(lo, hi), (lo, hi), "Balanced accuracy (%)",
               list(range(int(lo), int(hi) + 1, 4)), 5.1)
    C.save(fig, "ssl_ba_heatmap")
    C.save_table(as_table(B, ["ERM"] + names + ["SSL mean"]), "ssl_ba_heatmap")
    print("overall SSL mean - ERM", round(V[-1, -1], 2))


if __name__ == "__main__":
    main()
