"""Class-balanced and marginal shift between HAM10000 and each target (features and kernel of compute_shift_mmd.py).

JSD: per dimension, 50-bin histogram over the joint range, in bits, mean over the 1024 dimensions. MMD^2: biased,
global RBF kernel. Centroid cosine: between class centroids, mean over the five classes (all train images).
Class-balanced: 20 seeds, the same n per class in both domains (smallest class count, 210 for HAM10000), classes
pooled; mean and SD. Marginal: all train images (MMD^2 on at most 2000 per domain, seed 42).
Writes results/shift_representations.csv (read by tab_domain_shift.py).

Usage: python analysis/compute_shift_representations.py [--checkpoint checkpoints/ham10000_source.pth]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import entropy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compute_shift_mmd as S  # noqa: E402

N_BINS, MMD_MAX, MMD_SEED = 50, 2000, 42


def jsd_bits(a, b, n_bins=N_BINS):
    """Mean over dimensions of the JSD between per-dimension histograms."""
    out = []
    for d in range(a.shape[1]):
        lo, hi = min(a[:, d].min(), b[:, d].min()), max(a[:, d].max(), b[:, d].max())
        if hi - lo < 1e-10:
            out.append(0.0)
            continue
        bins = np.linspace(lo, hi, n_bins + 1)
        p = np.histogram(a[:, d], bins=bins)[0].astype(float) + 1e-10
        q = np.histogram(b[:, d], bins=bins)[0].astype(float) + 1e-10
        p, q = p / p.sum(), q / q.sum()
        m = 0.5 * (p + q)
        out.append(0.5 * entropy(p, m, base=2) + 0.5 * entropy(q, m, base=2))
    return float(np.mean(out))


def centroid_cosines(fa, la, fb, lb):
    out = []
    for i in range(len(S.CLASSES)):
        ca, cb = fa[la == i].mean(0), fb[lb == i].mean(0)
        out.append(float(ca @ cb / (np.linalg.norm(ca) * np.linalg.norm(cb))))
    return out


def subsample(x, n, rng):
    return x if n >= len(x) else x[rng.choice(len(x), n, replace=False)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=os.path.join(S.ROOT, "checkpoints", "ham10000_source.pth"))
    ap.add_argument("--cache", default=os.path.join(S.ROOT, "features"))
    ap.add_argument("--out", default=os.path.join(S.ROOT, "results", "shift_representations.csv"))
    args = ap.parse_args()

    S.extract(args.checkpoint, args.cache)
    feats, labels = {}, {}
    for key in S.DOMAINS:
        z = np.load(os.path.join(args.cache, f"feats_{key}.npz"))
        feats[key], labels[key] = z["features"].astype(np.float64), z["labels"]
    gamma = S.gamma_global(feats)
    print(f"gamma = {gamma:.10g}")

    rows = []
    fa, la = feats[S.SOURCE], labels[S.SOURCE]
    for target in S.DOMAINS:
        if target == S.SOURCE:
            continue
        fb, lb = feats[target], labels[target]
        cos = centroid_cosines(fa, la, fb, lb)
        npc = min(S.MAX_PER_CLASS, min(min((la == i).sum(), (lb == i).sum()) for i in range(len(S.CLASSES))))
        jsd, mmd = [], []
        for seed in range(S.N_SEEDS):
            rng = np.random.RandomState(seed)
            pa = np.vstack([subsample(fa[la == i], npc, rng) for i in range(len(S.CLASSES))])
            pb = np.vstack([subsample(fb[lb == i], npc, rng) for i in range(len(S.CLASSES))])
            jsd.append(jsd_bits(pa, pb))
            mmd.append(S.mmd2(pa, pb, gamma))
        rng = np.random.RandomState(MMD_SEED)
        ma, mb = subsample(fa, MMD_MAX, rng), subsample(fb, MMD_MAX, rng)
        jsd, mmd = np.array(jsd), np.array(mmd)
        rows.append(dict(target=target, n_per_class=int(npc),
                         jsd_balanced_mean=jsd.mean(), jsd_balanced_sd=jsd.std(ddof=1),
                         mmd2_balanced_mean=mmd.mean(), mmd2_balanced_sd=mmd.std(ddof=1),
                         centroid_cosine_mean=float(np.mean(cos)),
                         **{f"centroid_cosine_{c}": v for c, v in zip(S.CLASSES, cos)},
                         jsd_marginal=jsd_bits(fa, fb), mmd2_marginal=S.mmd2(ma, mb, gamma)))
    out = pd.DataFrame(rows).round(6)
    out.to_csv(args.out, index=False, lineterminator="\n")
    print(out.T.to_string())
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
