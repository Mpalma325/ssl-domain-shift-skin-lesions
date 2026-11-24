"""Label selection in the frozen-feature proxy, k = 16, Appendix 7.4: table e2_selection_proxy_app.

Mean and SD of the test balanced accuracy over the 30 draws, and the difference against 80 random images and against
the protocol of Study 1 (16 random per class), paired by draw, 95 % t interval (half-width; excludes_zero marks the
asterisk). Oracle (BCN20000): 6 restarts of the search against 1000 random sets of each reference, Welch interval.
Inputs: results/study2/selection_proxy_draws.csv, selection_proxy_oracle.csv (`--recompute` rebuilds them).
Output: tables/e2_selection_proxy_app.csv.
"""
import argparse

import numpy as np
import pandas as pd
from scipy import stats

import common as C
import selection as S
from fig_selection_proxy import NAMES
from tab_selection_finetune import welch

ROWS = ["random_80", "random_per_class", "oracle", "class_prototypes"] + S.COLD_START


def paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), d.mean() - h, d.mean() + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true", help="rebuild the per-draw results from features/ first")
    a = ap.parse_args()
    d = S.results("draws", a.recompute)
    o = S.results("oracle", a.recompute)
    rows = []
    for t in S.TARGETS:
        P = d[d.target == t].pivot(index="draw", columns="selection", values="ba_test")
        for s in ROWS:
            if s == "oracle":
                if t != "bcn20000":
                    continue
                v = o[o.selection == "oracle"].ba_test.to_numpy()
                diff = {ref: welch(v, o[o.selection == ref].ba_test) for ref in ("random_80", "random_per_class")}
            else:
                v = P[s].to_numpy()
                diff = {ref: paired(P[s], P[ref]) for ref in ("random_80", "random_per_class") if ref != s}
            row = dict(target=C.TARGET_EN[t], selection=NAMES[s], n=len(v), mean=v.mean(), sd=v.std(ddof=1))
            for ref, col in (("random_80", "vs_random"), ("random_per_class", "vs_study1")):
                m, lo, hi = diff.get(ref, (np.nan, np.nan, np.nan))
                row.update({col: m, col + "_half_width": (hi - lo) / 2,
                            col + "_excludes_zero": bool(lo > 0 or hi < 0) if ref in diff else np.nan})
            rows.append(row)
    C.save_table(pd.DataFrame(rows), "e2_selection_proxy_app")


if __name__ == "__main__":
    main()
