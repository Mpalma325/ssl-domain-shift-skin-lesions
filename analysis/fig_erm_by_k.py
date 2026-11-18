"""ERM balanced accuracy by k, HAM10000 init., Section 4.2 (Baseline): figure erm_by_k.

Per target: ERM mean over 5 seeds at k = 16, 32, 100 (bars: SD), fully supervised mean (dashed) and the HAM10000
source classifier with the HAM10000 train prior removed from its logits (dotted).
Inputs: common.panel() (primary test set), results/source_only.npz. Outputs: figures/erm_by_k.pdf, tables/erm_by_k.csv.
"""
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import common as C

plt = C.plt
BLUE, BLACK = "#2166ac", "#222222"
X_LINE = 66  # x where the reference labels sit (middle of the 32-100 span)

HAM_TRAIN_COUNTS = np.array([210, 231, 612, 447, 4084], float)  # AKIEC, BCC, BKL, MEL, NV


def source_prior_removed(target):
    """Source-only BA (%) on the primary test set after subtracting log(pi_train) of HAM10000 from the logits."""
    z = np.load(C.os.path.join(C.RESULTS, "source_only.npz"))
    logits, y, ids = z[f"logits_{target}"], z[f"labels_{target}"].astype(int), z[f"ids_{target}"]
    m = C.primary_mask(target, ids)
    adj = logits - np.log(HAM_TRAIN_COUNTS / HAM_TRAIN_COUNTS.sum())
    return C.metrics(adj[m], y[m])["ba"]


def erm_table():
    p = C.panel()
    rows = []
    for t in C.TARGETS:
        for k in C.KS + [0]:
            m = "erm" if k else "fully_supervised"
            v = p[(p.init == "ham") & (p.target == t) & (p.k == k) & (p.method == m)].ba.to_numpy()
            rows.append(dict(target=t, k=k if k else "FS", method=m, n_seeds=len(v), ba=v.mean(), sd=v.std(ddof=1)))
        rows.append(dict(target=t, k="source", method="source_prior_removed", n_seeds=1,
                         ba=source_prior_removed(t), sd=np.nan))
    return pd.DataFrame(rows)


def main():
    df = erm_table()
    df["k"] = df["k"].astype(str)
    levels = ["16", "32", "100"]
    fig, axes = plt.subplots(1, 3, figsize=(C.WIDTH, 2.6), sharey=True)
    for ax, t in zip(axes, C.TARGETS):
        d = df[df.target == t].set_index("k")
        fs, src = d.loc["FS", "ba"], d.loc["source", "ba"]
        ax.axhline(fs, color=BLACK, lw=1.1, ls=(0, (5, 2)), zorder=1)
        ax.axhline(src, color=BLACK, lw=1.1, ls=":", zorder=1)
        xs = [int(k) for k in levels]
        m, sd = d.loc[levels, "ba"].to_numpy(), d.loc[levels, "sd"].to_numpy()
        ax.errorbar(xs, m, yerr=sd, color=BLUE, marker="o", ms=4.5, lw=1.7, capsize=2, zorder=3)
        for x, v, s in zip(xs, m, sd):
            ax.annotate(C.r1(v), (x, v - s), xytext=(0, -3), textcoords="offset points", ha="center", va="top",
                        fontsize=8, color=BLUE, path_effects=C.HALO)
        ax.annotate(C.r1(fs), (X_LINE, fs), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                    fontsize=8, color=BLACK, path_effects=C.HALO)
        # In MILK10k dermoscopic the ERM curve crosses the source line: label above it, left of k = 32
        x_src, dy, va = (19, 2, "bottom") if t == "milk10k" else (X_LINE, -2, "top")
        ax.annotate(C.r1(src), (x_src, src), xytext=(0, dy), textcoords="offset points", ha="center", va=va,
                    fontsize=8, color=BLACK, path_effects=C.HALO)
        ax.set_xticks(xs, levels)
        ax.set_xlim(4, 112)
        ax.set_title(C.TARGET_EN[t], fontsize=10)
        ax.grid(axis="y", color=C.GRID)
    axes[0].set_ylabel("Balanced accuracy (%)")
    axes[1].set_xlabel("Labeled target images per class ($k$)")
    axes[0].set_ylim(30, 65)

    handles = [Line2D([], [], color=BLUE, marker="o", ms=4.5, lw=1.7, label="ERM"),
               Line2D([], [], color=BLACK, lw=1.1, ls=(0, (5, 2)), label="Fully supervised"),
               Line2D([], [], color=BLACK, lw=1.1, ls=":", label="Source classifier, prior removed")]
    fig.legend(handles=handles, frameon=False, fontsize=8.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    C.save(fig, "erm_by_k")
    C.save_table(df, "erm_by_k")


if __name__ == "__main__":
    main()
