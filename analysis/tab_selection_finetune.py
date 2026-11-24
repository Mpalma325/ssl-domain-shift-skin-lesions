"""Label selection with ERM networks trained for one epoch on each labeled set, Section 5.2.1 and Appendix 7.4.

e2_selection_ft_app: each selection minus 80 random images and minus the protocol of Study 1 (16 random per class):
difference of means over the 8 sets, Welch t interval, and a bootstrap that resamples sets within each selection and
test lesions (one lesion draw per replicate, shared by the selections of a target); asterisk if both exclude zero.
e2_composition_milk, e2_composition_bcn: labeled images per class (mean over the 8 sets).
e2_selection_classes_app: recall per class, labeled images of the class, and MEL sensitivity at 80 % specificity.
Inputs: results/study2/selection_finetune.npz and selection_finetune_runs.csv, data/splits. Outputs: tables/.
"""
import numpy as np
import pandas as pd
from scipy import stats

import common as C
import selection as S

B = 4000
SELECTIONS = ["class_prototypes", "random_per_class", "agreement_rule", "mixed_rule", "sequential_rule"]
BOOT_ORDER = ["agreement_rule", "mixed_rule", "class_prototypes", "random_per_class", "random_80",
              "sequential_rule"]  # order in which the bootstrap draws the sets of each selection (fixed)
SHORT = {"random_80": "80 random", "class_prototypes": "Prototypes", "random_per_class": "Study 1",
         "agreement_rule": "Agreement", "mixed_rule": "Mixed", "sequential_rule": "Sequential"}


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    h = stats.t.ppf(0.975, df) * np.sqrt(va + vb)
    d = a.mean() - b.mean()
    return d, d - h, d + h


def weighted_ba(correct, y, w):
    """correct: (sets, n); w: (B, n) image weights. Returns (B, sets)."""
    return 100 * np.mean([(w[:, y == c] @ correct[:, y == c].T) / w[:, y == c].sum(1, keepdims=True)
                          for c in range(5)], axis=0)


def contrasts(runs, per_target):
    rng = np.random.default_rng(20261118)
    boot = {}
    for t in S.TARGETS:
        correct, y, lesion = per_target[t]
        G = lesion.max() + 1
        w = rng.multinomial(G, np.full(G, 1 / G), size=B).astype(np.float64)[:, lesion]
        for s in BOOT_ORDER:
            M = weighted_ba(correct[s].astype(np.float64), y, w)
            boot[t, s] = np.take_along_axis(M, rng.integers(0, M.shape[1], size=(B, M.shape[1])), 1).mean(1)
    rows = []
    for ref in ("random_80", "random_per_class"):
        for t in S.TARGETS:
            ba = runs[runs.target == t].groupby("selection").ba.apply(np.array)
            for s in SELECTIONS:
                if s == ref:
                    continue
                d, lo, hi = welch(ba[s], ba[ref])
                blo, bhi = np.percentile(boot[t, s] - boot[t, ref], [2.5, 97.5])
                rows.append(dict(against=ref, target=t, selection=s, delta=d, t_lo=lo, t_hi=hi, boot_lo=blo,
                                 boot_hi=bhi, both_exclude_zero=bool((lo > 0 and blo > 0) or (hi < 0 and bhi < 0))))
    return pd.DataFrame(rows)


def main():
    runs, per_target = S.finetune()
    C.save_table(contrasts(runs, per_target), "e2_selection_ft_app")

    means = runs.groupby(["target", "selection"]).mean(numeric_only=True)
    for t, name in (("milk10k", "milk"), ("bcn20000", "bcn")):
        rows = []
        for s, label in (("random_80", "80 random images"), ("random_per_class", "Study 1 and prototypes"),
                         ("agreement_rule", "Agreement rule"), ("mixed_rule", "Mixed rule"),
                         ("sequential_rule", "Sequential rule")):
            rows.append(dict(selection=label, **{c.upper(): means.loc[(t, s), "n_" + c] for c in C.CLASSES}))
        assert (means.loc[(t, "class_prototypes"), ["n_" + c for c in C.CLASSES]] == 16).all()
        C.save_table(pd.DataFrame(rows), f"e2_composition_{name}")

    rows = []
    for t in S.TARGETS:
        for s in ["random_80"] + SELECTIONS:
            m = means.loc[(t, s)]
            rows.append(dict(target=C.TARGET_EN[t], selection=SHORT[s],
                             **{f"{c.upper()}_recall": m["rec_" + c] for c in C.CLASSES},
                             **{f"{c.upper()}_labeled": m["n_" + c] for c in C.CLASSES},
                             mel_sens_at_spec80=m.mel_sens_at_spec80))
    C.save_table(pd.DataFrame(rows), "e2_selection_classes_app")


if __name__ == "__main__":
    main()
