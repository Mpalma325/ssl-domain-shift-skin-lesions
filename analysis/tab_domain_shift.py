"""Domain shift between HAM10000 and each target, Section 4.2 (Domain Shift Characterization).

class_distance: total variation and JSD (bits) between train class proportions (both MILK10k conditions share them).
representation_shift and shift_marginal: class-balanced (and marginal) JSD, MMD^2 and centroid cosine.
Inputs: data/splits/*.csv, results/shift_representations.csv (compute_shift_representations.py).
Outputs: tables/<name>.csv.
"""
import numpy as np
import pandas as pd
from scipy.stats import entropy

import common as C

SPLITS = {"ham10000": "ham10000", "bcn20000": "bcn20000", "milk10k": "milk10k_dermoscopic"}


def train_proportions(name):
    s = pd.read_csv(f"{C.ROOT}/data/splits/{SPLITS[name]}.csv")
    n = s[s.split == "train"]["class"].value_counts().reindex(C.CLASSES).to_numpy(float)
    return n / n.sum()


def jsd_bits(p, q):
    m = 0.5 * (p + q)
    return float(0.5 * entropy(p, m, base=2) + 0.5 * entropy(q, m, base=2))


def main():
    p = train_proportions("ham10000")
    rows = []
    for t, label in (("bcn20000", "BCN20000"), ("milk10k", "MILK10k")):
        q = train_proportions(t)
        rows.append(dict(target=label, tv=0.5 * np.abs(p - q).sum(), jsd=jsd_bits(p, q)))
    C.save_table(pd.DataFrame(rows), "class_distance")

    s = pd.read_csv(f"{C.RESULTS}/shift_representations.csv").set_index("target").loc[C.TARGETS]
    names = [C.TARGET_EN[t] for t in C.TARGETS]
    C.save_table(pd.DataFrame(dict(target=names, jsd=s.jsd_balanced_mean.values, mmd2=s.mmd2_balanced_mean.values,
                                   centroid_cosine=s.centroid_cosine_mean.values)), "representation_shift")
    C.save_table(pd.DataFrame(dict(target=names, jsd_balanced=s.jsd_balanced_mean.values,
                                   mmd2_balanced=s.mmd2_balanced_mean.values, jsd_marginal=s.jsd_marginal.values,
                                   mmd2_marginal=s.mmd2_marginal.values)), "shift_marginal")
    for name in ("class_distance", "representation_shift", "shift_marginal"):
        print(pd.read_csv(f"{C.TABLES}/{name}.csv").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
