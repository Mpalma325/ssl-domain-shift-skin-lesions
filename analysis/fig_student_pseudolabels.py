"""Pseudo-labels of FixMatch started from the first student and from the HAM10000 network (Study 1), BCN20000,
k = 16, Section 5.2.2: fig:e2_pseudo_alumno.

Per epoch (1-15, mean of seeds 0-4): pool share with a correct, an incorrect or no pseudo-label, and validation BA.
Inputs: results/study2/pseudolabels_student_fixmatch.csv, results/pseudolabels_per_epoch.csv.
Outputs: figures/student_pseudolabels.pdf, tables/student_pseudolabels.csv.
"""
import os

import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import common as C
from fig_pseudolabels import ALPHA, FILL, LINE, TEXT, spread_pair

plt = C.plt
N_EP = 15


def summarise(d):
    d = d[d.epoch <= N_EP].copy()
    assert d.seed.nunique() == 5 and d.groupby("seed").epoch.count().min() == N_EP
    d["correct"] = d.pl_ratio * d.pl_accuracy / 100
    d["incorrect"] = d.pl_ratio * (100 - d.pl_accuracy) / 100
    return d.groupby("epoch").agg(correct=("correct", "mean"), incorrect=("incorrect", "mean"),
                                  pl_accuracy=("pl_accuracy", "mean"), val_ba=("val_ba", "mean")).reset_index()


def main():
    s1 = pd.read_csv(os.path.join(C.RESULTS, "pseudolabels_per_epoch.csv"))
    s1 = s1[(s1.init == "ham") & (s1.method == "fixmatch") & (s1.target == "bcn20000") & (s1.k == 16)]
    st = pd.read_csv(os.path.join(C.RESULTS, "study2", "pseudolabels_student_fixmatch.csv"))
    panels = [("From the HAM10000 network (Study 1)", summarise(s1)), ("From the first student", summarise(st))]
    C.save_table(pd.concat([d.assign(start=name) for name, d in panels]), "student_pseudolabels")
    for name, d in panels:
        for ep in (1, 5, 15):
            r = d[d.epoch == ep].iloc[0]
            print(f"  {name:38s} epoch {ep:2d}  correct {r.correct:5.1f}  incorrect {r.incorrect:5.1f}  "
                  f"accuracy {r.pl_accuracy:5.1f}  val {r.val_ba:5.1f}")

    fig, axes = plt.subplots(1, 2, figsize=(C.WIDTH * 0.75, 2.3), sharey=True)
    for ax, (name, d) in zip(axes, panels):
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
            ax.text(ep + dx, ya, f"{tot[pos]:.0f}", fontsize=6.8, color=TEXT["incorrect"], ha=ha, va="center",
                    clip_on=False)
            ax.text(ep + dx, yb, f"{d.correct.iloc[pos]:.0f}", fontsize=6.8, color=TEXT["correct"], ha=ha,
                    va="center", clip_on=False)
        ax.set_xlim(1, N_EP)
        ax.set_ylim(0, 100)
        ax.set_xticks([1, 5, 10, 15])
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.tick_params(labelsize=7.5, pad=2)
        ax.tick_params(axis="y", labelleft=False, length=0)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#999999")
        ax.set_title(name, fontsize=8.5, color=C.INK, pad=5)
        ax.set_xlabel("Epoch", fontsize=8.5)
    handles = [Patch(color=FILL["none"], label="No pseudo-label"),
               Patch(color=FILL["incorrect"], alpha=ALPHA, label="Incorrect pseudo-label"),
               Patch(color=FILL["correct"], alpha=ALPHA, label="Correct pseudo-label"),
               Line2D([], [], color=LINE["val"], lw=1.0, ls=(0, (3, 1.5)), label="Validation balanced accuracy")]
    fig.legend(handles=handles, frameon=False, fontsize=7, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 0.0),
               columnspacing=1.1, handlelength=1.6)
    fig.subplots_adjust(left=0.06, right=0.95, top=0.88, bottom=0.36, wspace=0.25)
    C.save(fig, "student_pseudolabels")


if __name__ == "__main__":
    main()
