"""Share of each expert in the gain of the four-expert combination over chance by k, fig:e2_contribucion_k (5.2.2).

Shapley value as % of the total of each seed, mean of five seeds (tab_expert_contribution.by_k, fixed weights).
Output: figures/expert_contribution_k.pdf and tables/expert_contribution_k.csv (plotted values).
"""
import numpy as np
import pandas as pd

import common as C
import experts as X
from tab_expert_contribution import ORDER, by_k

plt = C.plt
COLOR = {"RH": "#1F4E99", "CH": "#A61C1C", "RE": "#9EC3EC", "CE": "#F0A8A0"}
LABEL = {"RH": "RH (ResNet-50, HAM10000)", "CH": "CH (ConvNeXt-T, HAM10000)",
         "RE": "RE (ResNet-50, ERM)", "CE": "CE (ConvNeXt-T, ERM)"}


def main():
    _, H = by_k()
    H["share"] = 100 * H.shapley / H.total
    assert (H.groupby(["target", "k", "expert"]).seed.nunique() == 5).all()
    sh = H.groupby(["target", "k", "expert"]).share.mean().unstack()
    C.save_table(sh.reset_index(), "expert_contribution_k")

    fig, ax = plt.subplots(1, 2, figsize=(C.WIDTH, 3.0), sharey=True)
    x = np.log(C.KS)
    for a, t in zip(ax, X.TARGETS2):
        Y = np.array([[sh.loc[(t, k), e] for k in C.KS] for e in ORDER])
        assert np.allclose(Y.sum(0), 100)
        a.stackplot(x, Y, colors=[COLOR[e] for e in ORDER], labels=[LABEL[e] for e in ORDER], edgecolor="w", lw=0.8)
        cum = np.cumsum(Y, axis=0)
        for i, e in enumerate(ORDER):
            for j in range(3):
                xx = x[j] + (0.12 if j == 0 else -0.12 if j == 2 else 0)
                a.text(xx, cum[i, j] - Y[i, j] / 2, "%.0f" % Y[i, j], ha="center", va="center", fontsize=7.5,
                       color="w" if e in ("RH", "CH") else "k")
        for xv in x:
            a.axvline(xv, color="w", lw=0.6, alpha=0.6)
        a.set_xlim(x[0], x[-1])
        a.set_ylim(0, 100)
        a.set_xticks(x)
        a.set_xticklabels([str(k) for k in C.KS])
        a.set_xlabel("Labeled images per class (k)")
        a.set_title(X.NAME2[t])
    ax[0].set_ylabel("Share of the gain over chance (%)")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    C.save(fig, "expert_contribution_k")
    print(sh.round(1).to_string())


if __name__ == "__main__":
    main()
