"""Reference performance without unlabeled target images, Section 4.2 (Domain Shift Characterization and
Baseline and Reference Performance).

source_performance: HAM10000 classifier, raw and with the HAM10000 train prior subtracted from its logits.
fs_performance: fully supervised, mean and SD over five seeds. erm_gap: FS - ERM, HAM10000 init. init_contrast
(HAM10000 - ImageNet) and modality_contrast (MILK10k clinical - dermoscopic): paired by seed, 95 % t interval.
Inputs: results/source_only.npz, common.panel() (primary test set), data/splits. Outputs: tables/<name>.csv.
"""
import numpy as np
import pandas as pd
from scipy import stats

import common as C


def t_interval(v):
    """Mean, half-width of the 95 % t interval, and whether it excludes zero."""
    v = np.asarray(v, float)
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return v.mean(), h, bool(abs(v.mean()) > h)


def ham_train_prior():
    s = pd.read_csv(f"{C.ROOT}/data/splits/ham10000.csv")
    n = s[s.split == "train"]["class"].value_counts().reindex(C.CLASSES).to_numpy(float)
    return n / n.sum()


def source_performance():
    z = np.load(f"{C.RESULTS}/source_only.npz")
    log_prior = np.log(ham_train_prior())
    rows = []
    for t in ["ham10000"] + C.TARGETS:
        lg, y = z["logits_" + t].astype(float), z["labels_" + t].astype(int)
        m = np.ones(len(y), bool) if t == "ham10000" else C.primary_mask(t, z["ids_" + t])
        rows.append(dict(test_set=C.DATASET[t]["name"], domain="source" if t == "ham10000" else "target",
                         n_test=int(m.sum()), ba_original=C.metrics(lg[m], y[m])["ba"],
                         ba_prior_removed=C.metrics((lg - log_prior)[m], y[m])["ba"]))
    return pd.DataFrame(rows)


def main():
    p = C.panel()
    ba = p.set_index(["method", "init", "target", "k", "seed"]).ba.sort_index()

    def series(method, init, target, k):
        return ba.loc[(method, init, target, k)]

    C.save_table(source_performance(), "source_performance")

    fs = []
    for t in C.TARGETS:
        row = dict(target=C.TARGET_EN[t])
        for a in ("ham", "imagenet"):
            v = series("fully_supervised", a, t, 0)
            row[f"{a}_mean"], row[f"{a}_sd"], row[f"{a}_n_seeds"] = v.mean(), v.std(ddof=1), len(v)
        fs.append(row)
    C.save_table(pd.DataFrame(fs), "fs_performance")

    gap = []
    for t in C.TARGETS:
        row = dict(target=C.TARGET_EN[t])
        for k in C.KS:
            row[f"k{k}"] = series("fully_supervised", "ham", t, 0).mean() - series("erm", "ham", t, k).mean()
        gap.append(row)
    C.save_table(pd.DataFrame(gap), "erm_gap")

    def contrast_row(label, pairs):
        row = dict(row=label)
        for col, (x, y) in pairs.items():
            d = (x - y).dropna()
            assert len(d) == 5, (label, col)
            row[f"{col}_mean"], row[f"{col}_half_width"], row[f"{col}_excludes_zero"] = t_interval(d)
        return row

    init = [contrast_row(C.TARGET_EN[t],
                         {**{f"erm_k{k}": (series("erm", "ham", t, k), series("erm", "imagenet", t, k)) for k in C.KS},
                          "fully_supervised": (series("fully_supervised", "ham", t, 0),
                                               series("fully_supervised", "imagenet", t, 0))})
            for t in C.TARGETS]
    C.save_table(pd.DataFrame(init).rename(columns={"row": "target"}), "init_contrast")

    mod = [contrast_row(C.INIT_EN[a].replace(" init.", ""),
                        {**{f"erm_k{k}": (series("erm", a, "milk10k_clinico", k), series("erm", a, "milk10k", k))
                            for k in C.KS},
                         "fully_supervised": (series("fully_supervised", a, "milk10k_clinico", 0),
                                              series("fully_supervised", a, "milk10k", 0))})
           for a in ("ham", "imagenet")]
    C.save_table(pd.DataFrame(mod).rename(columns={"row": "initialization"}), "modality_contrast")

    with pd.option_context("display.width", 200):
        for name in ("source_performance", "fs_performance", "erm_gap", "init_contrast", "modality_contrast"):
            print(name)
            print(pd.read_csv(f"{C.TABLES}/{name}.csv").round(2).to_string(index=False))


if __name__ == "__main__":
    main()
