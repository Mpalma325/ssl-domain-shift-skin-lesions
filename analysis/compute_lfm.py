"""LFM: clinical-to-dermoscopic linear feature mapping in the feature space of the HAM10000 ResNet-50 (Section 5.1).

Ridge regression from the 2048 features of the clinical photograph and of a 2.5x zoom on its centre (each block
standardised on the pool) to the 2048 features of the dermoscopic image of the same lesion, fitted on the unlabeled
MILK10k pairs (training partition without the lesions copied from HAM10000), with lambda = 3 x mean eigenvalue.
The HAM10000 head reads the mapped features (HAM10000 prior removed). Recomputes the LFM and photograph readings from
features/clinical_views_milk10k.npz and compares them with results/study2/clinical_views_milk10k.npz.
"""
import os

import numpy as np

import clinical_views as V
import common as C

FEATURES = os.path.join(C.ROOT, "features", "clinical_views_milk10k.npz")
LOG_PRIOR_HAM = np.log(np.array([210, 231, 612, 447, 4084], float) / 5584)  # HAM10000 training classes
Q = 3.0


def load_features():
    if not os.path.exists(FEATURES):
        raise SystemExit(f"{FEATURES} not found: download it from the release (see results/README.md)")
    with np.load(FEATURES) as z:
        return {k: z[k] for k in z.files}


def head(F, f):
    """HAM10000 classifier on 2048-d features (bottleneck, batch norm, ReLU, linear); balanced log-posteriors."""
    h = np.asarray(F, np.float64) @ f["head_W1"].T.astype(np.float64) + f["head_b1"]
    h = (h - f["head_rm"]) / np.sqrt(f["head_rv"].astype(np.float64) + 1e-5) * f["head_g"] + f["head_bb"]
    return V.log_softmax(np.maximum(h, 0) @ f["head_W2"].T.astype(np.float64) + f["head_b2"] - LOG_PRIOR_HAM)


def ridge(X, Y, rows, alpha=1000.0):
    """Primal ridge fitted on rows; returns the prediction for every row."""
    mx, my = X[rows].mean(0), Y[rows].mean(0)
    Xc = X[rows] - mx
    W = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(X.shape[1]), Xc.T @ (Y[rows] - my))
    return (X - mx) @ W + my


def lfm(blocks, Y, pool, q=Q):
    """Kernel (dual) ridge on standardised blocks; lambda = q x (sum of eigenvalues / input dimension)."""
    S = []
    for X in blocks:
        X = X.astype(np.float32)
        S.append(((X - X[pool].mean(0)) / (X[pool].std(0) + 1e-6) / np.sqrt(X.shape[1])).astype(np.float32))
    means = [B[pool].mean(0) for B in S]
    K = sum(((B[pool] - m) @ (B[pool] - m).T).astype(np.float64) for B, m in zip(S, means))
    ev, U = np.linalg.eigh(K)
    keep = ev > ev.max() * 1e-10
    ev, U = ev[keep], U[:, keep]
    my = Y[pool].mean(0)
    A = U.T @ (Y[pool] - my)
    lam = q * ev.sum() / sum(B.shape[1] for B in S)
    G = sum(((B - m) @ (B[pool] - m).T).astype(np.float64) for B, m in zip(S, means))
    return ((G @ U) / (ev + lam)) @ A + my


def main():
    f = load_features()
    Fd = f["ham_derm"].astype(np.float64)
    pool = f["pool"].astype(int)
    out = {"clinical_lfm": head(lfm([f["ham_clinical"], f["ham_clinical_zoom"]], Fd, pool), f),
           "clinical_photo": head(f["ham_clinical"], f), "derm_ham": head(Fd, f)}
    R = V.Readings("milk10k")
    assert (R.ids == f["ids"].astype(str)).all()
    for k, lp in out.items():
        ref = R.lp(k, 0)
        print(f"{k:15s} max |difference| {np.abs(lp - ref).max():.2e}, same class "
              f"{100 * (lp.argmax(1) == ref.argmax(1)).mean():.2f} %")


if __name__ == "__main__":
    main()
