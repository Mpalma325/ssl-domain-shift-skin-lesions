"""Class distribution of the MILK10k test lesions before and after translating the clinical features:
fig:e2_lda_mallya (dermoscopic network of Mallya and Hamarneh, seed 0) and fig:e2_lda_virtual (HAM10000 network).

Two-component linear discriminant analysis (shrinkage 0.5) fitted on the dermoscopic validation images; each axis
scaled by the spread of the dermoscopic test images. Ellipses: one standard deviation around each class mean, dashed
for the dermoscopic images, filled for the clinical photographs (left) or their translation (right); arrows go from the
clinical mean to the dermoscopic mean. HAM10000 network: translation by ridge regression from the photograph (alpha
1000) fitted on the unlabeled pairs. Mallya and Hamarneh: guidance network applied to the clinical encoder features.
Input: features/clinical_views_milk10k.npz. Outputs: figures/lda_mallya, figures/lda_lfm, tables/lda_class_means.csv.
"""
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse

import common as C
import compute_lfm

plt = C.plt
COLORS = ["#d97706", "#7c3aed", "#0891b2", "#dc2626", "#475569"]
OFFSET = [(-0.55, -0.15), (-0.55, 0.35), (0.35, -0.25), (0.45, -0.3), (0.4, 0.3)]


def lda(X, y):
    mu = X.mean(0)
    M = np.array([X[y == k].mean(0) for k in range(5)])
    Sw = sum((X[y == k] - M[k]).T @ (X[y == k] - M[k]) for k in range(5)) / len(X)
    Sw = 0.5 * Sw + 0.5 * np.trace(Sw) / Sw.shape[0] * np.eye(Sw.shape[0])
    Sb = ((M - mu).T * np.bincount(y, minlength=5) / len(X)) @ (M - mu)
    w, Vv = np.linalg.eig(np.linalg.solve(Sw, Sb))
    P = Vv[:, np.argsort(-w.real)[:2]].real
    return lambda F: (F - mu) @ P


def ellipse(ax, Z, y, k, **kw):
    z = Z[y == k]
    m = z.mean(0)
    ev, ec = np.linalg.eigh(np.cov(z.T))
    ax.add_patch(Ellipse(m, 2 * np.sqrt(ev[1]), 2 * np.sqrt(ev[0]), angle=np.degrees(np.arctan2(ec[1, 1], ec[0, 1])),
                         **kw))
    return m


def draw(A, Bc, Ct, y, lims, name):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.0))
    for ax, (Z, title) in zip(axes, [(Bc, "Clinical photograph"), (Ct, "Translated photograph")]):
        for k in range(5):
            md = ellipse(ax, A, y, k, fill=False, ec=COLORS[k], lw=1.8, ls="--")
            mz = ellipse(ax, Z, y, k, fc=COLORS[k], ec=COLORS[k], alpha=0.28, lw=1.2)
            ax.scatter(*md, s=40, facecolors="white", edgecolors=COLORS[k], lw=1.8, zorder=6)
            ax.scatter(*mz, s=55, c=COLORS[k], edgecolors="white", lw=1, zorder=7)
            ax.annotate("", xy=md, xytext=mz, zorder=5,
                        arrowprops=dict(arrowstyle="->", color=COLORS[k], lw=1.4, shrinkA=4, shrinkB=4))
            ax.text(md[0] + OFFSET[k][0], md[1] + OFFSET[k][1], C.CLASSES_EN[k], color=COLORS[k], ha="center",
                    va="center", fontsize=10, fontfamily=["cmb10", "DejaVu Serif"], zorder=8)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("discriminant 1")
        ax.tick_params(labelsize=8)
        ax.set_xlim(*lims[0])
        ax.set_ylim(*lims[1])
    axes[0].set_ylabel("discriminant 2")
    axes[0].plot([], [], ls="--", c="k", label=r"dermoscopic image, class mean $\pm$ 1 sd")
    axes[0].fill([], [], c="k", alpha=0.25, label=r"clinical photograph, class mean $\pm$ 1 sd")
    axes[0].plot([], [], marker=r"$\rightarrow$", ls="", c="k", ms=14, label="shift to the dermoscopic mean")
    fig.legend(loc="lower center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    C.save(fig, name)


def class_means(panel, A, Bc, Ct, y):
    return [dict(figure=panel, image=im, cls=C.CLASSES_EN[k], d1=Z[y == k, 0].mean(), d2=Z[y == k, 1].mean())
            for im, Z in (("dermoscopic", A), ("clinical", Bc), ("translated", Ct)) for k in range(5)]


def main():
    f = compute_lfm.load_features()
    ids, y, split = f["ids"].astype(str), f["labels"].astype(int), f["split"].astype(str)
    Fd, Fc = f["ham_derm"].astype(np.float64), f["ham_clinical"].astype(np.float64)
    test = np.flatnonzero(split == "test")
    test = test[C.primary_mask("milk10k", ids[test])]
    val = np.flatnonzero(split == "val")
    Ft = compute_lfm.ridge(Fc, Fd, f["pool"].astype(int))
    proj = lda(Fd[val], y[val])
    A, Bc, Ct = proj(Fd[test]), proj(Fc[test]), proj(Ft[test])
    s = A.std(0)
    lfm = (A / s, Bc / s, Ct / s, y[test])
    draw(*lfm, ((-1.6, 2.6), (-1.9, 1.9)), "lda_lfm")

    keep = C.primary_mask("milk10k", f["mallya_test_ids"].astype(str))
    yt = f["mallya_test_labels"].astype(int)[keep]
    proj = lda(f["mallya_val_derm"].astype(np.float64), f["mallya_val_labels"].astype(int))
    A, Bc, Ct = (proj(f[k].astype(np.float64)[keep]) for k in
                 ("mallya_test_derm", "mallya_test_clinical", "mallya_test_hallucinated"))
    # the sign of each discriminant is arbitrary: NV to the right of zero and above AKIEC, as in the other figure
    s = A.std(0) * np.array([1 if A[yt == 4, 0].mean() > 0 else -1,
                             1 if A[yt == 4, 1].mean() > A[yt == 0, 1].mean() else -1])
    mallya = (A / s, Bc / s, Ct / s, yt)
    draw(*mallya, ((-1.6, 2.6), (-1.9, 2.3)), "lda_mallya")
    C.save_table(pd.DataFrame(class_means("lda_lfm", *lfm) + class_means("lda_mallya", *mallya)), "lda_class_means")


if __name__ == "__main__":
    main()
