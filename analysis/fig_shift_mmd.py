"""Heatmap of the class-conditional MMD^2 between HAM10000 and each target condition.

Reads results/shift_class_conditional_mmd.csv (mean over 20 seeds of the per-class MMD^2 in the
feature space of the HAM10000 classifier; computed by analysis/compute_shift_mmd.py).
Figure: shift_mmd_class. Table: tables/shift_mmd_class.csv.
"""
import os

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb

from common import CLASSES, FIGURES, RESULTS, TARGET_EN, TARGETS, plt, save_table

INK, MUTED = "#1A1A1A", "#404040"
VMAX = 0.40
# Blue half of RdBu: the same blue as the signed-difference heatmaps.
RAMP = LinearSegmentedColormap.from_list("blue_rdbu", plt.get_cmap("RdBu")(np.linspace(0.5, 1.0, 256)))


def luminance(rgb):
    c = np.asarray(rgb[:3], dtype=float)
    lin = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return float(np.dot(lin, [0.2126, 0.7152, 0.0722]))


def text_colour(bg):
    """Ink or white, whichever has the higher WCAG contrast on the background."""
    def contrast(a, b):
        hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
        return (hi + 0.05) / (lo + 0.05)
    return max((contrast(bg, to_rgb(c)), c) for c in (INK, "#FFFFFF"))[1]


def main():
    d = pd.read_csv(os.path.join(RESULTS, "shift_class_conditional_mmd.csv"))
    table = d.pivot(index="target", columns="class", values="mmd2_mean").loc[TARGETS, CLASSES]
    vals = table.to_numpy(float)

    norm = Normalize(vmin=0, vmax=VMAX)
    fig = plt.figure(figsize=(6.0, 1.95))
    ax = fig.add_axes([0.25, 0.05, 0.595, 0.80])
    cax = fig.add_axes([0.868, 0.05, 0.022, 0.80])
    mesh = ax.pcolormesh(vals, cmap=RAMP, norm=norm, edgecolors="white", linewidth=1.6)
    ax.invert_yaxis()
    for i, row in enumerate(vals):
        for j, v in enumerate(row):
            ax.text(j + 0.5, i + 0.5, f"{v:.3f}", ha="center", va="center", fontsize=10,
                    color=text_colour(RAMP(norm(v))))
    ax.set_xticks(np.arange(len(CLASSES)) + 0.5, [c.upper() for c in CLASSES])
    ax.set_yticks(np.arange(len(TARGETS)) + 0.5, [TARGET_EN[t] for t in TARGETS])
    ax.xaxis.tick_top()
    ax.tick_params(length=0, labelsize=10, labelcolor=INK, pad=4)
    for s in ax.spines.values():
        s.set_visible(False)
    bar = fig.colorbar(mesh, cax=cax)
    bar.set_ticks(np.arange(0, VMAX + 0.01, 0.1))
    bar.ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
    bar.ax.tick_params(length=2, width=0.5, color=MUTED, labelsize=8.5, labelcolor=MUTED, pad=2)
    bar.set_label(r"MMD$^2$", fontsize=10, color=INK, labelpad=12)
    bar.outline.set_visible(False)
    bar.solids.set_edgecolor("face")
    # fixed 6.0 x 1.95 in canvas (no tight bounding box), as in the thesis
    os.makedirs(FIGURES, exist_ok=True)
    base = os.path.join(FIGURES, "shift_mmd_class")
    fig.savefig(base + ".pdf", metadata={"CreationDate": None})
    fig.savefig(base + ".png", dpi=200)
    plt.close(fig)
    print("  -> figures/shift_mmd_class.pdf")
    save_table(table.reset_index().round(3), "shift_mmd_class")


if __name__ == "__main__":
    main()
