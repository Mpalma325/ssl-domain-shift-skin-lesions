"""Overall effect of SSL on balanced accuracy, Section 4.2 (Overall Effect of SSL).

ssl_factors, ssl_factors_full: share of the sum of squares of SSL - ERM over the 180 cells (method x target x k x init.,
mean over seeds) per effect of a balanced-design decomposition (main effects, method x k, method x target, rest; or
all 15). ssl_group[_imagenet]: mean of the ten methods minus ERM per seed, mean and 95 % t half-width over five seeds.
ssl_margin: ssl_group / (FS - ERM margin, table erm_gap), %. Input: common.panel(). Outputs: tables/<name>.csv.
"""
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

import common as C

FACTORS = ["target", "method", "k", "init"]
FACTOR_EN = {"target": "Target condition", "method": "Method", "k": "k", "init": "Initialization"}


def decomposition(cells):
    g = cells.d_ba.mean()
    total = ((cells.d_ba - g) ** 2).sum()
    effects, rows = {}, []
    for r in range(1, len(FACTORS) + 1):
        for S in combinations(FACTORS, r):
            e = cells.groupby(list(S)).d_ba.transform("mean") - g
            for T, v in effects.items():
                if set(T) < set(S):
                    e = e - v
            effects[S] = e
            rows.append(dict(effect=" x ".join(FACTOR_EN[f] for f in S), order=r,
                             sum_of_squares=float((e ** 2).sum()), share=float(100 * (e ** 2).sum() / total)))
    full = pd.DataFrame(rows).sort_values("share", ascending=False).reset_index(drop=True)
    assert abs(full.share.sum() - 100) < 1e-6
    return full


def group(dd, init):
    per_seed = dd[dd.init == init].groupby(["target", "k", "seed"]).d_ba.mean()
    rows = []
    for t in C.TARGETS:
        row = dict(target=C.TARGET_EN[t])
        for k in C.KS:
            v = per_seed.loc[(t, k)].to_numpy()
            assert len(v) == 5
            h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
            row[f"k{k}_mean"], row[f"k{k}_half_width"], row[f"k{k}_excludes_zero"] = v.mean(), h, abs(v.mean()) > h
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    p = C.panel()
    dd = C.deltas(C.design())
    cells = C.cells(dd)
    assert len(cells) == 180

    full = decomposition(cells)
    C.save_table(full, "ssl_factors_full")
    keep = ["Target condition", "Method x k", "Method", "Target condition x Method", "k", "Initialization"]
    short = full.set_index("effect").share
    short = pd.DataFrame(dict(factor=keep + ["Other interactions"],
                              share=[short[e] for e in keep] + [100 - short[keep].sum()]))
    C.save_table(short, "ssl_factors")

    g = group(dd, "ham")
    C.save_table(g, "ssl_group")
    C.save_table(group(dd, "imagenet"), "ssl_group_imagenet")

    ba = p[p.init == "ham"].groupby(["method", "target", "k"]).ba.mean()
    margin = []
    for t, row in zip(C.TARGETS, g.itertuples()):
        out = dict(target=C.TARGET_EN[t])
        for k in C.KS:
            gap = ba[("fully_supervised", t, 0)] - ba[("erm", t, k)]
            out[f"k{k}"] = 100 * getattr(row, f"k{k}_mean") / gap
        margin.append(out)
    C.save_table(pd.DataFrame(margin), "ssl_margin")

    with pd.option_context("display.width", 200):
        print(short.round(1).to_string(index=False))
        print(full[["effect", "share"]].round(2).to_string(index=False))
        print(g.round(2).to_string(index=False))
        print(pd.read_csv(f"{C.TABLES}/ssl_group_imagenet.csv").round(2).to_string(index=False))
        print(pd.DataFrame(margin).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
