"""Pairwise comparison of the SSL methods in their advantage over ERM (thesis table tab:method_pairs).

Per seed, each method's paired SSL - ERM BA is averaged over the 18 conditions; row - column with a 95 % t interval
over five seeds (no correction for the 45 comparisons). Column "erm": each method vs ERM. Lower triangle, family order.
Input: common.design(). Output: tables/method_pairs.csv (row, column, mean, half_width, excludes_zero).
"""
import numpy as np
import pandas as pd
from scipy import stats

import common as C


def t_interval(v):
    """Mean, half-width of the 95 % t interval and whether it excludes zero."""
    v = np.asarray(v, float)
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return v.mean(), h, bool(abs(v.mean()) > h)


def main():
    dd = C.deltas(C.design())
    assert (dd.groupby("method").apply(lambda g: g.groupby(["init", "target", "k"]).ngroups) == 18).all()
    ps = dd.groupby(["method", "seed"]).d_ba.mean().unstack()  # method x seed, mean over the 18 conditions
    rows = []
    for i, a in enumerate(C.SSL):
        mean, h, sig = t_interval(ps.loc[a])
        rows.append(dict(row=a, column="erm", mean=mean, half_width=h, excludes_zero=sig))
        for b in C.SSL[:i]:
            mean, h, sig = t_interval(ps.loc[a] - ps.loc[b])
            rows.append(dict(row=a, column=b, mean=mean, half_width=h, excludes_zero=sig))
    r = pd.DataFrame(rows)
    C.save_table(r, "method_pairs")
    print(r.pivot(index="row", columns="column", values="mean").loc[C.SSL, ["erm"] + C.SSL[:-1]].round(2)
          .to_string())


if __name__ == "__main__":
    main()
