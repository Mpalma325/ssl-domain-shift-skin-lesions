"""Class distribution of HAM10000, BCN20000 and MILK10k (all splits, % of images).

Reads data/splits/ham10000.csv, bcn20000.csv and milk10k_dermoscopic.csv (MILK10k has the
same counts in both modalities). Figure: class_distributions. Table: tables/class_distributions.csv.
"""
import os

import numpy as np
import pandas as pd

from common import CLASSES, CLASSES_EN, ROOT, plt, save, save_table

DATASETS = [("HAM10000", "ham10000", "#2166AC"), ("BCN20000", "bcn20000", "#2E7D32"),
            ("MILK10K", "milk10k_dermoscopic", "#E08214")]
INK, MUTED, GRID = "#222222", "#6B6B6B", "#E8E8E8"


def counts():
    rows = []
    for name, split_file, _ in DATASETS:
        d = pd.read_csv(os.path.join(ROOT, "data", "splits", split_file + ".csv"))
        c = d["class"].value_counts()
        for k in CLASSES:
            rows.append(dict(dataset=name, **{"class": k}, n_images=int(c.get(k, 0)),
                             pct=100 * c.get(k, 0) / len(d), n_total=len(d)))
    return pd.DataFrame(rows)


def main():
    t = counts()
    with plt.rc_context({"font.family": "DejaVu Sans"}):
        fig, ax = plt.subplots(figsize=(7.6, 3.3))
        x, w = np.arange(5), 0.26
        for j, (name, _, col) in enumerate(DATASETS):
            d = t[t.dataset == name].set_index("class").loc[CLASSES]
            off = (j - 1) * w
            ax.bar(x + off, d.pct, w * 0.92, color=col, zorder=3, label=f"{name} (n = {d.n_total.iloc[0]:,})")
            for i, (n, p) in enumerate(zip(d.n_images, d.pct)):
                ax.text(i + off, p + 1.8, f"{n:,}\n({p:.1f}%)", ha="center", va="bottom", fontsize=6.0, color=INK,
                        linespacing=1.25)
        ax.set_xticks(x)
        ax.set_xticklabels(CLASSES_EN, fontsize=9, color=INK)
        ax.set_ylim(0, 100)
        ax.set_yticks(range(0, 101, 20))
        ax.set_yticklabels([f"{v}%" for v in range(0, 101, 20)], fontsize=8.5)
        ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#CCCCCC")
            ax.spines[s].set_linewidth(0.7)
        ax.tick_params(colors=MUTED, length=0)
        ax.set_ylabel("% of images", fontsize=9, color=INK)
        ax.set_title("Class Distribution Across Datasets", fontsize=11, color=INK, weight="bold", pad=10, loc="left")
        leg = ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.09),
                        handlelength=1.1, handleheight=0.9, columnspacing=1.8, borderpad=0)
        for txt in leg.get_texts():
            txt.set_color(INK)
        fig.tight_layout()
        save(fig, "class_distributions")
    save_table(t.round(2), "class_distributions")


if __name__ == "__main__":
    main()
