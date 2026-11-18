"""Appendix control tables on the unlabeled pool: tab:control_sisters, tab:control_pool_size, tab:control_prior and
tab:control_prior_classes.

HAM10000 init., primary test set; paired by seed, mean +- 95 % t half-width, `significant` if it excludes zero,
`positive_seeds` = seeds with a positive difference. In tab:control_prior "FlexMatch" is `flexmatch_rerun` (no
warm-up, same machine as the true-composition run), not the main-design FlexMatch.
Inputs: results/controls_pool_sisters.csv, controls_pool_size.csv, controls_class_prior.csv, common.panel().
Outputs: tables/control_sisters.csv, control_pool_size.csv, control_prior.csv, control_prior_classes.csv
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

from common import CLASSES, RESULTS, panel, save_table

SEEDS = [0, 1, 2, 3, 4]


def t_summary(d):
    """Mean, half-width of the 95 % t interval, significance and positive seeds of paired differences."""
    d = np.asarray(d, float)
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return dict(diff=d.mean(), half_width=h, significant=bool(abs(d.mean()) > h), positive_seeds=int((d > 0).sum()),
                n_seeds=len(d))


def by_seed(df, col="ba", **sel):
    x = df
    for k, v in sel.items():
        x = x[x[k] == v]
    x = x.set_index("seed")[col]
    assert sorted(x.index) == SEEDS, sel
    return x.loc[SEEDS].to_numpy(float)


def control_sisters():
    c = pd.read_csv(os.path.join(RESULTS, "controls_pool_sisters.csv"))
    rows = []
    for m, name in [("fixmatch", "FixMatch"), ("pi_model", "Pi-Model")]:
        d = by_seed(c, method=m, pool="without_sisters") - by_seed(c, method=m, pool="complete")
        rows.append(dict(comparison=f"{name}, without - with these images", **t_summary(d)))
    return pd.DataFrame(rows)


def control_pool_size(p):
    c = pd.read_csv(os.path.join(RESULTS, "controls_pool_size.csv"))
    main = p[(p.init == "ham") & (p.target == "bcn20000") & (p.k == 16)]
    erm = by_seed(main, method="erm")
    methods = [("fixmatch", "FixMatch"), ("mean_teacher", "Mean Teacher"), ("pseudo_label", "Pseudo-Label"),
               ("debiasmatch", "DebiasMatch"), ("self_tuning", "Self-Tuning")]
    rows, red_all, com_all = [], [], []
    for m, name in methods:
        red, com = by_seed(c, method=m), by_seed(main, method=m)
        red_all.append(red), com_all.append(com)
        rows.append((name, red, com))
    rows.append(("Mean of the five methods", np.mean(red_all, 0), np.mean(com_all, 0)))
    out = []
    for name, red, com in rows:
        r = dict(method=name)
        for key, d in [("reduced_minus_erm", red - erm), ("complete_minus_erm", com - erm),
                       ("reduced_minus_complete", red - com)]:
            s = t_summary(d)
            r[key], r[key + "_hw"], r[key + "_sig"] = s["diff"], s["half_width"], s["significant"]
            if key == "reduced_minus_erm":
                r["reduced_minus_erm_positive_seeds"] = s["positive_seeds"]
        out.append(r)
    return pd.DataFrame(out)


PRIOR_ROWS = [  # (label, run A, run B, target); runs: (source, method)
    ("FlexMatch, true composition - FlexMatch", ("c", "flexmatch_true_prior"), ("c", "flexmatch_rerun"), "milk10k"),
    ("FlexMatch, true composition - ERM", ("c", "flexmatch_true_prior"), ("p", "erm"), "milk10k"),
    ("FlexMatch, true composition - ERM", ("c", "flexmatch_true_prior"), ("p", "erm"), "bcn20000"),
    ("SimPro - ERM", ("c", "simpro"), ("p", "erm"), "milk10k"),
    ("SimPro - ERM", ("c", "simpro"), ("p", "erm"), "milk10k_clinico"),
    ("SimPro - FixMatch", ("c", "simpro"), ("p", "fixmatch"), "bcn20000"),
]


def control_prior(p):
    c = pd.read_csv(os.path.join(RESULTS, "controls_class_prior.csv"))
    main = p[(p.init == "ham") & (p.k == 16)]
    src = {"c": c, "p": main}
    ba, cls = [], []
    for label, (sa, ma), (sb, mb), t in PRIOR_ROWS:
        A = src[sa][(src[sa].method == ma) & (src[sa].target == t)]
        B = src[sb][(src[sb].method == mb) & (src[sb].target == t)]
        ba.append(dict(comparison=label, target=t, **t_summary(by_seed(A) - by_seed(B))))
        r = dict(comparison=label, target=t)
        for cl in CLASSES:
            s = t_summary(by_seed(A, "rec_" + cl) - by_seed(B, "rec_" + cl))
            r[cl], r[cl + "_hw"], r[cl + "_sig"] = s["diff"], s["half_width"], s["significant"]
        cls.append(r)
    return pd.DataFrame(ba), pd.DataFrame(cls)


def main():
    p = panel()
    tables = {"control_sisters": control_sisters(), "control_pool_size": control_pool_size(p)}
    tables["control_prior"], tables["control_prior_classes"] = control_prior(p)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        for name, df in tables.items():
            save_table(df, name)
            print(df.round(2).to_string(index=False), "\n")


if __name__ == "__main__":
    main()
