"""Pseudo-label quantity and quality per epoch for FixMatch, FlexMatch, DebiasMatch and DST.

Input: results/pseudolabels_per_epoch.csv (extract_pseudolabels.py). The pool splits into correct = ratio*acc/100,
incorrect = ratio*(100-acc)/100 and none = 100-ratio, averaged over k and seeds (epochs 1-15).
Figures (suffix _imagenet for ImageNet init.): pseudolabels_areas (validation BA dashed) and pseudolabels_quality.
Table: tables/pseudolabels_per_epoch_mean.csv. Pseudo-Label logs only accuracy, so it is left out.
"""
import os

import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from common import DATASET, INK, NAME, RESULTS, TARGET_EN, TARGETS, WIDTH, plt, save, save_table

METHODS = ["fixmatch", "flexmatch", "debiasmatch", "dst"]
N_EP = 15
FILL = {"correct": "#9ecae1", "incorrect": "#f4a582", "none": "#f2f2f2"}
ALPHA = 0.5
LINE = {"correct": "#2166ac", "incorrect": "#b2182b", "val": "#222222"}
TEXT = {"correct": "#174a7c", "incorrect": "#8e1520"}


def load():
    d = pd.read_csv(os.path.join(RESULTS, "pseudolabels_per_epoch.csv"))
    d = d[(d.epoch <= N_EP) & d.method.isin(METHODS)].copy()
    d["correct"] = d.pl_ratio * d.pl_accuracy / 100
    d["incorrect"] = d.pl_ratio * (100 - d.pl_accuracy) / 100
    g = d.groupby(["init", "method", "target", "epoch"]).agg(
        correct=("correct", "mean"), incorrect=("incorrect", "mean"), val_ba=("val_ba", "mean"),
        pl_accuracy=("pl_accuracy", "mean"), n_runs=("run_id", "nunique")).reset_index()
    g["none"] = 100 - g.correct - g.incorrect
    assert g.n_runs.min() >= 13, g.n_runs.min()
    return g


def spread_pair(lo, hi, sep=11):
    """Text positions for two values, at least sep apart and inside [3, 97]."""
    if hi - lo < sep:
        m = (hi + lo) / 2
        lo, hi = m - sep / 2, m + sep / 2
    shift = max(0, 3 - lo) - max(0, hi - 97)
    return lo + shift, hi + shift


def spread(vals, sep=3.6):
    """Text positions for several values: same order, at least sep apart, centred on the values."""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    pos = [vals[i] for i in order]
    for _ in range(50):
        moved = False
        for a in range(len(pos) - 1):
            if pos[a + 1] - pos[a] < sep:
                m = (pos[a] + pos[a + 1]) / 2
                pos[a], pos[a + 1] = m - sep / 2, m + sep / 2
                moved = True
        if not moved:
            break
    out = [0.0] * len(vals)
    for i, p in zip(order, pos):
        out[i] = p
    return out


def style_axis(ax, yticks):
    ax.set_xlim(1, N_EP)
    ax.set_xticks([1, 5, 10, 15])
    ax.set_yticks(yticks)
    ax.tick_params(labelsize=7.5, pad=2)
    ax.tick_params(axis="y", labelleft=False, length=0)  # the numbers give the scale
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#999999")


def areas(G, init):
    fig, axes = plt.subplots(3, len(METHODS), figsize=(WIDTH, 5.0), sharex=True, sharey=True)
    for i, t in enumerate(TARGETS):
        for j, m in enumerate(METHODS):
            ax = axes[i, j]
            d = G[(G.method == m) & (G.target == t)].sort_values("epoch")
            x = d.epoch.to_numpy()
            tot = (d.correct + d.incorrect).to_numpy()
            ax.fill_between(x, 0, d.correct, color=FILL["correct"], alpha=ALPHA, lw=0)
            ax.fill_between(x, d.correct, tot, color=FILL["incorrect"], alpha=ALPHA, lw=0)
            ax.fill_between(x, tot, 100, color=FILL["none"], lw=0)
            ax.plot(x, tot, color=LINE["incorrect"], lw=1.0, marker="o", ms=2.3)
            ax.plot(x, d.correct, color=LINE["correct"], lw=1.0, marker="o", ms=2.3)
            ax.plot(x, d.val_ba, color=LINE["val"], lw=1.0, ls=(0, (3, 1.5)))
            for pos, ep, ha, dx in ((0, 1, "right", -0.45), (-1, N_EP, "left", 0.45)):
                yb, ya = spread_pair(d.correct.iloc[pos], tot[pos])
                ax.text(ep + dx, ya, f"{tot[pos]:.0f}", fontsize=6.8, color=TEXT["incorrect"], ha=ha,
                        va="center", clip_on=False)
                ax.text(ep + dx, yb, f"{d.correct.iloc[pos]:.0f}", fontsize=6.8, color=TEXT["correct"], ha=ha,
                        va="center", clip_on=False)
            ax.set_ylim(0, 100)
            style_axis(ax, [0, 25, 50, 75, 100])
            if i == 0:
                ax.set_title(NAME[m], fontsize=9, color=INK, pad=6)
            if i == 2:
                ax.set_xlabel("Epoch", fontsize=8.5)
            if j == 0:
                ax.set_ylabel(TARGET_EN[t].replace("MILK10k ", "MILK10k\n"), fontsize=8.5, labelpad=16)
    handles = [Patch(color=FILL["none"], label="No pseudo-label"),
               Patch(color=FILL["incorrect"], alpha=ALPHA, label="Incorrect pseudo-label"),
               Patch(color=FILL["correct"], alpha=ALPHA, label="Correct pseudo-label"),
               Line2D([], [], color=LINE["val"], lw=1.0, ls=(0, (3, 1.5)), label="Validation balanced accuracy")]
    fig.legend(handles=handles, frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               columnspacing=1.1, handlelength=1.6)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.9, bottom=0.08, wspace=0.32, hspace=0.14)
    save(fig, "pseudolabels_areas" + ("_imagenet" if init == "imagenet" else ""))


def quality(G, init):
    fig, axes = plt.subplots(1, len(METHODS), figsize=(WIDTH, 2.35), sharey=True)
    for ax, m in zip(axes, METHODS):
        ax.axhline(50, color="#999999", lw=0.7, ls=(0, (1, 2)), zorder=1)
        ends = {0: [], -1: []}
        for t in TARGETS:
            d = G[(G.method == m) & (G.target == t)].sort_values("epoch")
            st = DATASET[t]
            ax.plot(d.epoch, d.pl_accuracy, color=st["color"], ls=st["line"], marker=st["marker"], ms=2.4, lw=1.1,
                    zorder=3)
            for pos in ends:
                ends[pos].append((d.pl_accuracy.iloc[pos], st["color"]))
        for pos, ep, ha, dx in ((0, 1, "right", -0.45), (-1, N_EP, "left", 0.45)):
            ys = spread([v for v, _ in ends[pos]])
            for (v, col), y in zip(ends[pos], ys):
                ax.text(ep + dx, y, f"{v:.0f}", fontsize=6.8, color=col, ha=ha, va="center", clip_on=False)
        ax.set_title(NAME[m], fontsize=9, color=INK, pad=6)
        ax.set_ylim(30, 70)
        style_axis(ax, [30, 40, 50, 60, 70])
        ax.set_xlabel("Epoch", fontsize=8.5)
    axes[0].set_ylabel("Correct among\npseudo-labeled (%)", fontsize=8.5, labelpad=16)
    handles = [Line2D([], [], color=DATASET[t]["color"], ls=DATASET[t]["line"], marker=DATASET[t]["marker"], ms=3,
                      lw=1.1, label=TARGET_EN[t]) for t in TARGETS]
    handles.append(Line2D([], [], color="#999999", lw=0.7, ls=(0, (1, 2)), label="50%"))
    fig.legend(handles=handles, frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.02),
               columnspacing=1.4, handlelength=2.2)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.78, bottom=0.2, wspace=0.32)
    save(fig, "pseudolabels_quality" + ("_imagenet" if init == "imagenet" else ""))


def main():
    G = load()
    for init in ("ham", "imagenet"):
        g = G[G.init == init]
        areas(g, init)
        quality(g, init)
    cols = ["init", "method", "target", "epoch", "n_runs", "correct", "incorrect", "none", "pl_accuracy", "val_ba"]
    save_table(G[cols].round(4), "pseudolabels_per_epoch_mean")


if __name__ == "__main__":
    main()
