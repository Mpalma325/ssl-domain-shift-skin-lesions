"""Class-conditional MMD^2 between HAM10000 and each target in the 1024-d bottleneck of the HAM10000 classifier.

Features: train split of each domain, validation transform (resize 256, centre crop 224), cached in
features/feats_<domain>.npz. One RBF kernel, gamma = 1 / median non-zero squared distance among 125 images per domain
(seed 42). Per target, class and seed (20 seeds): n = min(400, n_HAM, n_target) images per domain without replacement,
biased MMD^2; mean and SD over seeds. Writes results/shift_class_conditional_mmd.csv (read by fig_shift_mmd.py).

Usage: python analysis/compute_shift_mmd.py [--checkpoint checkpoints/ham10000_source.pth]  (GPU recommended)
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES = ["akiec", "bcc", "bkl", "mel", "nv"]
DOMAINS = {"ham10000": "HAM10000", "bcn20000": "BCN20000", "milk10k": "MILK10K", "milk10k_clinico": "MILK10K_CLINICO"}
SOURCE = "ham10000"
N_SEEDS, MAX_PER_CLASS, GAMMA_N, GAMMA_SEED = 20, 400, 500, 42


def extract(checkpoint, cache):
    if all(os.path.exists(os.path.join(cache, f"feats_{key}.npz")) for key in DOMAINS):
        return
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder
    sys.path.insert(0, os.path.join(ROOT, "train"))
    import utils

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clf = utils.ImageClassifier(utils.get_model("resnet50", pretrained_checkpoint=None), len(CLASSES),
                                bottleneck_dim=1024, pool_layer=None, finetune=True)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    missing, _ = clf.load_state_dict(ck["state_dict"], strict=False)
    assert not [m for m in missing if not m.startswith("backbone.fc")], f"missing weights: {missing[:5]}"
    net = nn.Sequential(clf.backbone, clf.pool_layer, clf.bottleneck).to(dev).eval()
    tf = utils.get_val_transform("default")
    os.makedirs(cache, exist_ok=True)
    for key, folder in DOMAINS.items():
        out = os.path.join(cache, f"feats_{key}.npz")
        if os.path.exists(out):
            continue
        ds = ImageFolder(os.path.join(ROOT, "data", folder, "train"), transform=tf)
        assert ds.classes == CLASSES, ds.classes
        feats, labels = [], []
        with torch.no_grad():
            for x, y in DataLoader(ds, batch_size=32, shuffle=False, num_workers=0):
                feats.append(net(x.to(dev)).cpu())
                labels.append(y)
        np.savez_compressed(out, features=torch.cat(feats).numpy().astype(np.float32),
                            labels=torch.cat(labels).numpy().astype(np.int64))
        print(f"  {folder:16} {len(ds)} images -> {out}")


def gamma_global(feats):
    """1 / median squared distance over an equal sample of each domain (dict order fixes the draw)."""
    rng = np.random.RandomState(GAMMA_SEED)
    per = GAMMA_N // len(feats)
    sample = np.vstack([f[rng.choice(len(f), min(per, len(f)), replace=False)] for f in feats.values()])
    d = cdist(sample, sample, metric="sqeuclidean")
    return 1.0 / np.median(d[d > 0])


def mmd2(a, b, gamma):
    k = lambda x, y: np.exp(-gamma * cdist(x, y, metric="sqeuclidean"))  # noqa: E731
    return float(max(k(a, a).mean() + k(b, b).mean() - 2 * k(a, b).mean(), 0.0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=os.path.join(ROOT, "checkpoints", "ham10000_source.pth"))
    ap.add_argument("--cache", default=os.path.join(ROOT, "features"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "shift_class_conditional_mmd.csv"))
    args = ap.parse_args()

    extract(args.checkpoint, args.cache)
    feats, labels = {}, {}
    for key in DOMAINS:
        z = np.load(os.path.join(args.cache, f"feats_{key}.npz"))
        feats[key], labels[key] = z["features"].astype(np.float64), z["labels"]
    gamma = gamma_global(feats)
    print(f"gamma = {gamma:.10g}")

    rows = []
    for target in DOMAINS:
        if target == SOURCE:
            continue
        vals = {c: [] for c in CLASSES}
        n_used = {}
        for seed in range(N_SEEDS):
            rng = np.random.RandomState(seed)
            for i, c in enumerate(CLASSES):
                fa, fb = feats[SOURCE][labels[SOURCE] == i], feats[target][labels[target] == i]
                n = min(MAX_PER_CLASS, len(fa), len(fb))
                a = fa if n >= len(fa) else fa[rng.choice(len(fa), n, replace=False)]
                b = fb if n >= len(fb) else fb[rng.choice(len(fb), n, replace=False)]
                vals[c].append(mmd2(a, b, gamma))
                n_used[c] = n
        for c in CLASSES:
            v = np.array(vals[c])
            rows.append(dict(target=target, **{"class": c}, n_per_domain=n_used[c],
                             mmd2_mean=round(v.mean(), 6), mmd2_sd=round(v.std(ddof=1), 6)))
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False, lineterminator="\n")
    print(out.to_string(index=False))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
