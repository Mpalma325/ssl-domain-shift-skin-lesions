"""Appendix tables tab:ba_methods_bcn, tab:ba_methods_milk_derm, tab:ba_methods_milk_clin and tab:init_summary.

ba_methods_*: BA of ERM and each SSL method per init. and k, mean and SD over five seeds. init_summary: HAM10000 -
ImageNet, per seed averaged over k, for ERM, fully supervised (paired by seed) and SSL - ERM (mean over methods);
"All" = per-seed mean of the targets; mean and 95 % t half-width over seeds (excludes_zero = asterisk).
Input: common.panel() (primary test set). Outputs: tables/ba_methods_{bcn,milk_derm,milk_clin}.csv, init_summary.csv.
"""
import numpy as np
import pandas as pd
from scipy import stats

import common as C

OUT = {"bcn20000": "ba_methods_bcn", "milk10k": "ba_methods_milk_derm", "milk10k_clinico": "ba_methods_milk_clin"}


def t_interval(v):
    """Mean, half-width of the 95 % t interval and whether it excludes zero."""
    v = np.asarray(v, float)
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return v.mean(), h, bool(abs(v.mean()) > h)


def ba_by_method(df):
    g = df.groupby(["target", "method", "init", "k"]).ba.agg(["mean", "std", "count"])
    assert (g["count"] == 5).all()
    for t, name in OUT.items():
        rows = []
        for m in ["erm"] + C.SSL:
            row = {"method": C.NAME[m]}
            for a in ("ham", "imagenet"):
                for k in C.KS:
                    row[f"{a}_k{k}_mean"] = g.loc[(t, m, a, k), "mean"]
                    row[f"{a}_k{k}_sd"] = g.loc[(t, m, a, k), "std"]
            rows.append(row)
        C.save_table(pd.DataFrame(rows), name)


def init_summary(p):
    df = p[p.method != "fully_supervised"]
    erm = df[df.method == "erm"].pivot_table(index=["target", "k", "seed"], columns="init", values="ba")
    erm = (erm.ham - erm.imagenet).rename("v").reset_index()
    fs = p[p.method == "fully_supervised"].pivot_table(index=["target", "seed"], columns="init", values="ba")
    fs = (fs.ham - fs.imagenet).rename("v").reset_index()
    dd = C.deltas(df)
    ssl = dd.pivot_table(index=["method", "target", "k", "seed"], columns="init", values="d_ba")
    ssl = (ssl.ham - ssl.imagenet).rename("v").reset_index()
    rows = []
    for label, d in (("ERM", erm), ("Fully supervised", fs), ("SSL - ERM", ssl)):
        per_target_seed = d.groupby(["target", "seed"]).v.mean()  # mean over k (and methods)
        for t in C.TARGETS + ["all"]:
            v = (per_target_seed.groupby("seed").mean() if t == "all" else per_target_seed.loc[t])
            mean, h, sig = t_interval(v.to_numpy())
            rows.append(dict(row=label, target=t, mean=mean, half_width=h, excludes_zero=sig, n_seeds=len(v)))
    r = pd.DataFrame(rows)
    C.save_table(r, "init_summary")
    print(r.round(2).to_string(index=False))


def main():
    p = C.panel()
    ba_by_method(p[p.method != "fully_supervised"])
    init_summary(p)


if __name__ == "__main__":
    main()
