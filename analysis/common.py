"""Shared loading, metrics and figure style for the analysis scripts.

Inputs: results/runs.csv (one row per run), results/predictions_<target>.npz (validation and test logits,
row-aligned with runs.csv), data/splits/milk10k_test_excluded.txt.
Every metric uses the primary test set: MILK10k without the 29 images that copy a HAM10000 photograph
(885 images) and BCN20000 complete (3199 images).
"""
import logging
import os
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
TABLES = os.path.join(ROOT, "tables")
EXCLUDED = os.path.join(ROOT, "data", "splits", "milk10k_test_excluded.txt")

CLASSES = ["akiec", "bcc", "bkl", "mel", "nv"]
CLASSES_EN = ["AKIEC", "BCC", "BKL", "MEL", "NV"]
KS = [16, 32, 100]
SEEDS = [0, 1, 2, 3, 4]
INITS = ["ham", "imagenet"]
TARGETS = ["bcn20000", "milk10k", "milk10k_clinico"]
TARGET_EN = {"bcn20000": "BCN20000", "milk10k": "MILK10k dermoscopic", "milk10k_clinico": "MILK10k clinical"}
INIT_EN = {"ham": "HAM10000 init.", "imagenet": "ImageNet init."}

# SSL methods grouped by family (Chapter 2). Methods are never ordered by gain.
FAMILIES = [("Prediction consistency", ["pi_model", "mean_teacher"]),
            ("Self-training", ["pseudo_label", "noisy_student"]),
            ("Strong augmentation and\nconfidence filtering", ["uda", "fixmatch", "flexmatch"]),
            ("Bias-aware self-training", ["debiasmatch", "dst"]),
            ("Self-Tuning", ["self_tuning"])]
SSL = [m for _, ms in FAMILIES for m in ms]
NAME = {"erm": "ERM", "fixmatch": "FixMatch", "flexmatch": "FlexMatch", "debiasmatch": "DebiasMatch",
        "dst": "DST", "mean_teacher": "Mean Teacher", "noisy_student": "Noisy Student",
        "pi_model": "Π-Model", "pseudo_label": "Pseudo-Label", "self_tuning": "Self-Tuning", "uda": "UDA",
        "fully_supervised": "Fully supervised"}
NAME_TEX = {**NAME, "pi_model": r"$\Pi$-Model"}


def runs():
    return pd.read_csv(os.path.join(RESULTS, "runs.csv"))


def predictions(target):
    """Arrays of one target; row r belongs to the run with runs.csv row == r and that target."""
    path = os.path.join(RESULTS, f"predictions_{target}.npz")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found: download the predictions (see results/README.md)")
    return np.load(path)


def excluded_ids():
    with open(EXCLUDED, encoding="utf-8") as fh:
        ids = {line.strip() for line in fh if line.strip()}
    assert len(ids) == 29
    return ids


def primary_mask(target, test_ids):
    """True for the images of the primary test set."""
    if target == "bcn20000":
        return np.ones(len(test_ids), bool)
    return ~np.isin(test_ids, sorted(excluded_ids()))


def confusion(logits, labels):
    pred = np.asarray(logits).argmax(1)
    cm = np.zeros((5, 5), np.int64)
    np.add.at(cm, (labels, pred), 1)
    return cm


def metrics(logits, labels):
    """Balanced accuracy, recall and predicted fraction per class, MEL predicted as NV (all in %)."""
    cm = confusion(logits, labels)
    support = cm.sum(1).astype(float)
    recall = np.diag(cm) / support
    out = {"ba": 100 * recall.mean(), "n_test": int(len(labels)), "mel_nv": 100 * cm[3, 4] / support[3]}
    for i, c in enumerate(CLASSES):
        out["rec_" + c] = 100 * recall[i]
        out["pf_" + c] = 100 * cm[:, i].sum() / len(labels)
    for i in range(5):
        for j in range(5):
            out[f"cm_{i}{j}"] = int(cm[i, j])
    return out


_PANEL = None


def panel():
    """One row per run with its test metrics on the primary test set (cached in memory)."""
    global _PANEL
    if _PANEL is None:
        rows = []
        R = runs()
        for target in TARGETS:
            z = predictions(target)
            mask = primary_mask(target, z["ids_test"])
            y = z["labels_test"].astype(int)
            logits = z["logits_test"]  # decompress once per target, not once per run
            for r in R[R.target == target].itertuples():
                lt = logits[r.row]
                m = metrics(lt[mask], y[mask])
                m["ba_full_test"] = metrics(lt, y)["ba"]
                rows.append({"run_id": r.run_id, "init": r.init, "target": target, "k": r.k,
                             "method": r.method, "seed": r.seed, "best_epoch": r.best_epoch, **m})
        _PANEL = pd.DataFrame(rows)
    return _PANEL


def design():
    """The 990 runs of the main design (ERM and the ten SSL methods)."""
    p = panel()
    return p[p.method != "fully_supervised"].reset_index(drop=True)


def deltas(df, cols=("ba",)):
    """SSL - ERM paired by (init, target, k, seed). One row per SSL run."""
    erm = df[df.method == "erm"].set_index(["init", "target", "k", "seed"])
    ssl = df[df.method != "erm"].copy()
    key = ssl.set_index(["init", "target", "k", "seed"]).index
    for c in cols:
        ssl["d_" + c] = ssl[c].values - erm.loc[key, c].values
    return ssl


def cells(dd, col="d_ba"):
    """Mean over seeds of each cell (method, init, target, k)."""
    return dd.groupby(["method", "init", "target", "k"], as_index=False)[col].mean()


def boot_mean_of_cells(dd, col="d_ba", B=2000, seed=0, by=("init", "target", "k")):
    """95 % bootstrap interval of the mean of cells; seeds resampled within each init. (paired: a seed is a subset)."""
    rng = np.random.default_rng(seed)
    piv = dd.pivot_table(index=["method", *by], columns="seed", values=col)
    mean = piv.mean(1).mean()
    inits = piv.index.get_level_values("init")
    blocks = {a: piv[inits == a].dropna(axis=1, how="all").values for a in np.unique(inits)}
    res = np.empty(B)
    for b in range(B):
        vals = []
        for M in blocks.values():
            j = rng.integers(0, M.shape[1], M.shape[1])
            vals.append(np.nanmean(M[:, j], 1))
        res[b] = np.concatenate(vals).mean()
    return mean, np.percentile(res, 2.5), np.percentile(res, 97.5)


import matplotlib  # noqa: E402

matplotlib.use("Agg")
logging.getLogger("fontTools").setLevel(logging.ERROR)
import matplotlib.patheffects as pe  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

WIDTH = 6.5  # in; \textwidth of the thesis
DATASET = {
    "ham10000": dict(name="HAM10000", color="#2166ac", marker="D", line="-"),
    "bcn20000": dict(name="BCN20000", color="#2e7d32", marker="o", line="-"),
    "milk10k": dict(name="MILK10k dermoscopic", color="#e08214", marker="s", line="-"),
    "milk10k_clinico": dict(name="MILK10k clinical", color="#8c4a0f", marker="^", line=(0, (4, 2))),
}
INK, INK_SOFT, GRID = "#333333", "#555555", "#eeeeee"
HALO = [pe.withStroke(linewidth=2.4, foreground="white")]
DIVERGING_LIMIT = 6.0  # pp; values beyond are saturated in colour and still written in the cell

plt.rcParams.update({
    "font.family": ["cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#555555", "axes.linewidth": 0.8,
    "xtick.color": INK, "ytick.color": INK,
    "axes.unicode_minus": False,  # cmr10 has no U+2212
    "axes.formatter.use_mathtext": True,
    "axes.axisbelow": True,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 200,
})


def r1(v):
    """One decimal, rounding half up (Python rounds half to even)."""
    return str(Decimal(str(float(v))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def signed(v):
    s = r1(v)
    return s if s.startswith("-") else "+" + s


def ink_on(value, cmap, norm):
    """White text on dark cells."""
    r, g, b = cmap(norm(value))[:3]
    return "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.5 else INK


def cell_frame(ax, x0, y0, sep_x=None, sep_y=None, base=1.6, lw=1.8):
    """Black frame inside cell (x0, y0), clear of white edges (`base` pt) and separators; axes must be placed."""
    sep_x, sep_y = sep_x or {}, sep_y or {}
    W, H = ax.figure.get_size_inches()
    pos = ax.get_position()
    xl, yl = sorted(ax.get_xlim()), sorted(ax.get_ylim())
    px = 72 * W * pos.width / (xl[1] - xl[0])
    py = 72 * H * pos.height / (yl[1] - yl[0])

    def inset(edge, seps, scale):
        return (max(seps.get(round(edge, 6), 0), base) / 2 + lw / 2 - 0.4) / scale

    left, right = inset(x0, sep_x, px), inset(x0 + 1, sep_x, px)
    top, bottom = inset(y0, sep_y, py), inset(y0 + 1, sep_y, py)
    ax.add_patch(Rectangle((x0 + left, y0 + top), 1 - left - right, 1 - top - bottom, fill=False, ec="black",
                           lw=lw, zorder=6, clip_on=False, joinstyle="miter"))


def save(fig, name):
    os.makedirs(FIGURES, exist_ok=True)
    base = os.path.join(FIGURES, name)
    fig.savefig(base + ".pdf", bbox_inches="tight", metadata={"CreationDate": None})  # reproducible bytes
    fig.savefig(base + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> figures/{name}.pdf")


def save_table(df, name, **kw):
    os.makedirs(TABLES, exist_ok=True)
    path = os.path.join(TABLES, name + ".csv")
    df.to_csv(path, index=False, lineterminator="\n", **kw)
    print(f"  -> tables/{name}.csv")
