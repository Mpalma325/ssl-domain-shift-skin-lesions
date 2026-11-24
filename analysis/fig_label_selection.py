"""Test balanced accuracy of the one-epoch ERM networks per labeled set, Section 5.2.1: figure label_selection.

Boxes over the 8 sets of each selection (whiskers at min and max), points are sets, numbers are medians; dashed line at
the median of 80 random images. Inputs: results/study2/selection_finetune*. Outputs: figures/label_selection.pdf,
tables/label_selection.csv.
"""
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

import common as C
import selection as S

plt = C.plt
GROUPS = {"ref": ("#9ecae1", "#2166ac", "References"), "heu": ("#d9d9d9", "#4d4d4d", "Heuristics")}
ROWS = [("ref", "random_80"), ("ref", "random_per_class"), ("ref", "class_prototypes"),
        ("heu", "agreement_rule"), ("heu", "mixed_rule"), ("heu", "sequential_rule")]


def boxes(axes, data, rows, names, groups=GROUPS):
    """One horizontal box per row and panel; data[(target, key)] -> values. Returns the plotted summary."""
    ypos, y, prev = [], 0, None
    for g, _ in rows:
        y += 0.5 if prev is not None and g != prev else 0
        ypos.append(y)
        y += 1
        prev = g
    rng = np.random.default_rng(0)
    out = []
    for ax, t in zip(axes, S.TARGETS):
        ax.plot([np.median(data[t, rows[0][1]])] * 2, [ypos[1] - 0.6, ypos[-1] + 0.5], color=C.INK_SOFT, lw=0.8,
                ls="--", zorder=0)
        for (g, key), yy in zip(rows, ypos):
            v = data.get((t, key), np.array([]))
            if len(v) == 0:
                continue
            fill, edge, _ = groups[g]
            ax.boxplot([v], positions=[yy], widths=0.45, whis=(0, 100), patch_artist=True, vert=False,
                       boxprops=dict(facecolor=fill, edgecolor=edge, lw=0.7), medianprops=dict(color=edge, lw=1.0),
                       whiskerprops=dict(color=edge, lw=0.7), capprops=dict(color=edge, lw=0.7))
            ax.scatter(v, yy + rng.uniform(-0.06, 0.06, len(v)), s=1.5, color=edge, lw=0, alpha=0.8, zorder=3)
            ax.text(np.median(v), yy - 0.27, "%.1f" % np.median(v), ha="center", va="bottom", fontsize=7, color=edge,
                    bbox=dict(fc="white", ec="none", pad=1.0), zorder=5)
            out.append(dict(target=t, selection=key, n=len(v), median=np.median(v), mean=v.mean(), min=v.min(),
                            max=v.max()))
        ax.set_ylim(ypos[-1] + 0.5, -0.6)
        ax.set_xlabel("Test balanced accuracy (%)", fontsize=9)
        ax.set_title(C.TARGET_EN[t], color=C.INK, fontsize=10, fontfamily=["cmb10", "DejaVu Serif"])
        ax.tick_params(labelsize=8.5, length=2)
        ax.xaxis.set_major_locator(plt.MultipleLocator(5))
        ax.grid(axis="x", color=C.GRID, lw=0.7)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels([names[k] for _, k in rows], fontsize=8)
    return pd.DataFrame(out)


def main():
    runs, _ = S.finetune()
    data = {(t, s): g.ba.to_numpy() for (t, s), g in runs.groupby(["target", "selection"])}
    fig, axes = plt.subplots(1, 2, figsize=(C.WIDTH, 3.2), sharey=True)
    table = boxes(axes, data, ROWS, S.FT_NAMES)
    fig.legend(handles=[Patch(facecolor=f, edgecolor=e, label=n) for f, e, n in GROUPS.values()],
               loc="upper center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.6, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=1.5)
    C.save(fig, "label_selection")
    C.save_table(table, "label_selection")


if __name__ == "__main__":
    main()
