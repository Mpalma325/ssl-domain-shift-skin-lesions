"""Accuracy against confidence for ERM, SSL and fully supervised (figures confidence and confidence_imagenet).

Per target and initialization, the primary-test predictions of each group are pooled (ERM 45 runs, SSL 450, FS five
seeds). Confidence = max softmax in ten equal bins over 20-100 %; each bin is drawn at its mean confidence and
accuracy, only if it holds >= 1 % of the group's predictions.
Outputs: figures/confidence[_imagenet].pdf/.png, tables/confidence[_imagenet].csv.
"""
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import common as C

plt = C.plt
BLUE, RED, BLACK = "#2166ac", "#b2182b", "#222222"
EDGES = np.linspace(0.2, 1.0, 11)
STYLE = {"erm": dict(color=BLUE, marker="o", ls="-", label="ERM"),
         "ssl": dict(color=RED, marker="s", ls="-", label="SSL (mean of ten methods)"),
         "fs": dict(color=BLACK, marker="^", ls=(0, (5, 2)), label="Fully supervised")}


def confidence_and_hit(z, y):
    z = np.asarray(z, float)
    p = np.exp(z - z.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    return p.max(-1), p.argmax(-1) == y


def compute(init):
    R = C.runs()
    rows = []
    for t in C.TARGETS:
        z = C.predictions(t)
        logits = z["logits_test"]
        mask = C.primary_mask(t, z["ids_test"])
        y = z["labels_test"].astype(int)[mask]
        sel = R[(R.target == t) & (R.init == init)]
        groups = {"erm": sel[sel.method == "erm"], "ssl": sel[sel.method.isin(C.SSL)],
                  "fs": sel[sel.method == "fully_supervised"]}
        for g, rr in groups.items():
            cf, ok = confidence_and_hit(logits[rr.row.to_numpy()][:, mask], y)
            cf, ok = cf.ravel(), ok.ravel()
            b = np.clip(np.digitize(cf, EDGES) - 1, 0, 9)
            for i in range(10):
                s = b == i
                rows.append(dict(init=init, target=t, group=g, bin=i, frac=s.mean(),
                                 conf=cf[s].mean() if s.any() else np.nan,
                                 acc=ok[s].mean() if s.any() else np.nan, n_runs=len(rr)))
    return pd.DataFrame(rows)


def figure(D, name):
    fig, axes = plt.subplots(1, 3, figsize=(C.WIDTH, 2.6), sharex=True, sharey=True)
    for ax, t in zip(axes, C.TARGETS):
        ax.plot([20, 100], [20, 100], color="#bbbbbb", lw=0.8, zorder=1)
        for g in ("fs", "erm", "ssl"):
            d = D[(D.target == t) & (D.group == g) & (D.frac >= 0.01)]
            st = STYLE[g]
            ax.plot(100 * d.conf, 100 * d.acc, color=st["color"], ls=st["ls"], lw=1.6, marker=st["marker"],
                    ms=3.8, zorder=3)
        ax.set_xlim(20, 100)
        ax.set_ylim(20, 100)
        ax.set_xticks([20, 40, 60, 80, 100])
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_aspect("equal")
        ax.set_title(C.TARGET_EN[t], fontsize=9.5, color=C.INK)
        ax.grid(color=C.GRID, lw=0.6)
        ax.tick_params(labelsize=8)
        ax.set_xlabel("Confidence (%)", fontsize=9)
    axes[0].set_ylabel("Accuracy (%)", fontsize=9)
    handles = [Line2D([], [], color=STYLE[g]["color"], ls=STYLE[g]["ls"], marker=STYLE[g]["marker"], ms=3.8,
                      lw=1.6, label=STYLE[g]["label"]) for g in ("erm", "ssl", "fs")]
    fig.legend(handles=handles, frameon=False, fontsize=8.5, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 1.02))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.84, bottom=0.17, wspace=0.15)
    C.save(fig, name)


def main():
    for init, name in (("ham", "confidence"), ("imagenet", "confidence_imagenet")):
        D = compute(init)
        C.save_table(D, name)
        figure(D, name)


if __name__ == "__main__":
    main()
