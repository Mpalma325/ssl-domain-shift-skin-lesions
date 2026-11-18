"""Example images: figures ham10000_examples, bcn20000_examples, milk10k_examples_a and milk10k_examples_b.

Images are listed in results/example_images.csv: per diagnosis (by frequency, MILK10k counted per lesion), the image
of median sharpness (Laplacian variance at 300x225) among 60 evenly spaced HAM10000 lesions or the first 80 by
identifier. MILK10k figures show the clinical image of the same lesion below the dermoscopic one. Eleven images are
outside the experiment (vasc, df and excluded diagnoses), so pass the official download folders with --images.

Usage: python analysis/fig_example_images.py --images <HAM10000 dir> --images <BCN20000 dir> --images <MILK10k dir>
"""
import argparse
import os
import re
import sys
import textwrap

import pandas as pd
from PIL import Image

from common import ROOT, plt, save

INK = "#222222"
ISIC = re.compile(r"(ISIC_\d{7})\.(jpe?g|png)$", re.IGNORECASE)


def index(folder):
    found = {}
    for root, _, files in os.walk(folder):
        for name in files:
            m = ISIC.search(name)
            if m:
                found.setdefault(m.group(1), os.path.join(root, name))
    return found


def image_paths(extra):
    """ISIC id -> path. data/ first; MILK10k clinical files are named after their dermoscopic image."""
    paths = {}
    for folder in extra:
        paths.update(index(folder))
    for name in ("HAM10000", "BCN20000", "MILK10K"):
        paths.update(index(os.path.join(ROOT, "data", name)))
    clin = pd.read_csv(os.path.join(ROOT, "data", "splits", "milk10k_clinical.csv"))
    derm_to_clin = dict(zip(clin.derm_isic_id, clin.clinical_isic_id))
    for derm, p in index(os.path.join(ROOT, "data", "MILK10K_CLINICO")).items():
        paths[derm_to_clin[derm]] = p
    return paths


def frame(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_edgecolor("#CCCCCC")
        s.set_linewidth(0.8)


def ham_examples(sel, paths):
    from matplotlib import gridspec
    fig = plt.figure(figsize=(11.0, 3.95))
    gs = gridspec.GridSpec(2, 24, figure=fig, hspace=0.25, wspace=0.30, top=0.95, bottom=0.03)
    spans = [(0, 0), (0, 6), (0, 12), (0, 18), (1, 3), (1, 9), (1, 15)]  # 4 above, 3 centred below
    for (row, col), r in zip(spans, sel.itertuples()):
        ax = fig.add_subplot(gs[row, col:col + 6])
        ax.imshow(Image.open(paths[r.isic_id]))
        frame(ax)
        ax.set_title(r"$\bf{" + r.title + r"}$", fontsize=12, color=INK, pad=6)
    save(fig, "ham10000_examples")


def grid(sel, paths, name, figw=11.0, ncol=5, nrow=2):
    cellw = figw / ncol / 1.28
    fig, axes = plt.subplots(nrow, ncol, figsize=(figw, nrow * cellw * 0.75 + nrow * 0.42), squeeze=False)
    plt.subplots_adjust(wspace=0.28, hspace=0.42, top=0.90, bottom=0.02, left=0.01, right=0.99)
    for k, r in enumerate(sel.itertuples()):
        ax = axes[k // ncol][k % ncol]
        ax.imshow(Image.open(paths[r.isic_id]))
        frame(ax)
        ax.set_title("\n".join(textwrap.wrap(r.title, 21)), fontsize=8.5, color=INK, pad=5, linespacing=1.3)
    save(fig, name)


def paired(sel, paths, name, figw=11.0):
    derm = sel[sel.modality == "dermoscopic"].sort_values("panel")
    clin = sel[sel.modality == "clinical"].sort_values("panel")
    n = len(derm)
    cellw = figw / n / 1.28
    fig, axes = plt.subplots(2, n, figsize=(figw, 2 * cellw * 0.75 + 0.75), squeeze=False)
    plt.subplots_adjust(wspace=0.28, hspace=0.06, top=0.86, bottom=0.02, left=0.055, right=0.99)
    for i, rows in enumerate((derm, clin)):
        for j, r in enumerate(rows.itertuples()):
            ax = axes[i][j]
            ax.imshow(Image.open(paths[r.isic_id]))
            frame(ax)
            if i == 0:
                ax.set_title("\n".join(textwrap.wrap(r.title, 21)), fontsize=8.5, color=INK, pad=5, linespacing=1.3)
        axes[i][0].set_ylabel(("dermoscopic", "clinical")[i], fontsize=9.5, color=INK, labelpad=7)
    save(fig, name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", action="append", default=[],
                    help="folder of an official download, searched recursively for ISIC_*.jpg (repeatable)")
    args = ap.parse_args()

    sel = pd.read_csv(os.path.join(ROOT, "results", "example_images.csv"))
    paths = image_paths(args.images)
    missing = sorted(set(sel.isic_id) - set(paths))
    if missing:
        sys.exit(f"{len(missing)} images not found ({', '.join(missing)}): pass the official download "
                 "folders with --images")
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
                         "axes.titleweight": "bold", "mathtext.fontset": "dejavusans"}):
        ham_examples(sel[sel.figure == "ham10000_examples"].sort_values("panel"), paths)
        grid(sel[sel.figure == "bcn20000_examples"].sort_values("panel"), paths, "bcn20000_examples")
        for name in ("milk10k_examples_a", "milk10k_examples_b"):
            paired(sel[sel.figure == name], paths, name)


if __name__ == "__main__":
    main()
