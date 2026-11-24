"""Balanced accuracy on MIDAS clinical photographs by number of MILK10k pairs used to fit LFM: fig:e2_midas_curva.

Each point: mean over five seeds (each seed the mean of five nested draws of N pairs; one fit with all 4,427 pairs) and
95 % t interval over seeds. Reference lines: the HAM10000 network on the real dermoscopic images (photographs that
have one) and the clinical ERM alone.
Input: results/study2/midas_metrics.csv (analysis == "pairs_curve"). Outputs: figures/midas_pairs_curve,
tables/midas_pairs_curve.csv.
"""
import numpy as np
import pandas as pd

import clinical_views as V
import common as C

plt = C.plt
SOURCE, TARGET, COMBINATION = "#2166ac", "#7f7f7f", "#222222"
CURVES = [("ham_lfm", TARGET, "D", "HAM10000 network with LFM"),
          ("erm_lfm", SOURCE, "o", "Clinical ERM combined with LFM")]


def main():
    m, _ = V.midas()
    g = m[m.analysis == "pairs_curve"]
    rows = []
    for model, *_ in CURVES:
        x = g[g.model == model]
        for N, v in x.groupby("pairs"):
            mean, lo, hi = V.t_interval(v.sort_values("seed").ba)
            rows.append(dict(model=model, pairs=int(N), ba=mean, t_low=lo, t_high=hi))
    for model in ("ham_real_derm", "erm"):
        mean, lo, hi = V.t_interval(g[g.model == model].ba)
        rows.append(dict(model=model, pairs=np.nan, ba=mean, t_low=lo, t_high=hi))
    t = pd.DataFrame(rows)
    C.save_table(t, "midas_pairs_curve")
    ceiling = t[t.model == "ham_real_derm"].ba.item()
    alone = t[t.model == "erm"].ba.item()
    n_all = int(t.pairs.max())

    fig, ax = plt.subplots(figsize=(C.WIDTH * 0.62, 3.0))
    for model, color, marker, label in CURVES:
        f = t[t.model == model].sort_values("pairs")
        ax.fill_between(f.pairs, f.t_low, f.t_high, color=color, alpha=0.12, lw=0)
        ax.plot(f.pairs, f.ba, color=color, marker=marker, ms=3.5, label=label)
    ax.axhline(ceiling, color=COMBINATION, ls="--", lw=0.9)
    ax.text(110, ceiling + 0.25, f"real dermoscopic image ({C.r1(ceiling)})", fontsize=7, color=COMBINATION)
    ax.axhline(alone, color=SOURCE, ls=":", lw=0.9)
    ax.text(1300, alone - 0.8, f"clinical ERM alone ({C.r1(alone)})", fontsize=7, color=SOURCE)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=7, frameon=False)
    ax.axhline(20, color=C.INK_SOFT, lw=0.7)
    ax.text(110, 20.3, "chance (20)", fontsize=7, color=C.INK_SOFT)
    ax.set_xscale("log")
    ax.set_xticks([100, 250, 500, 1000, 2000, n_all])
    ax.set_xticklabels(["100", "250", "500", "1,000", "2,000", f"{n_all:,}"])
    ax.minorticks_off()
    ax.grid(color=C.GRID)
    ax.set_xlabel("MILK10k pairs used to fit LFM")
    ax.set_ylabel("Balanced accuracy on MIDAS (%)")
    fig.tight_layout()
    C.save(fig, "midas_pairs_curve")
    print(t.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
