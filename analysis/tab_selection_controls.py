"""Composition and recognized-image controls of the three heuristics in the proxy, k = 16, Appendix 7.4:
table e2_selection_controls_app.

Each control averages its 10 sets per draw. Differences against 80 random images, paired by draw (30), 95 % t interval
half-width; correct_pct = share of the chosen images classified correctly by the combined HAM10000 prediction.
Input: results/study2/selection_proxy_controls.csv (`--recompute` rebuilds it). Output: tables/.
"""
import argparse

import numpy as np
import pandas as pd

import common as C
import selection as S
from tab_selection_proxy import paired

RULES = {"agreement_rule": "Agreement", "mixed_rule": "Mixed", "sequential_rule": "Sequential"}


def per_draw(recompute=False):
    """BA per draw: columns <selection> and <selection>|<control>."""
    R = S.results("controls", recompute)
    R["key"] = np.where(R.control == "none", R.selection, R.selection + "|" + R.control)
    return R, R.groupby(["target", "key", "draw"]).ba_test.mean().unstack("key")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true", help="rebuild the per-draw results from features/ first")
    R, P = per_draw(ap.parse_args().recompute)
    correct = R[R.control == "none"].groupby(["target", "selection"]).combined_correct.mean() * 100
    rows = []
    for t in S.TARGETS:
        p = P.loc[t]
        rows.append(dict(target=C.TARGET_EN[t], selection="80 random images", correct_pct=correct[t, "random_80"]))
        for r, name in RULES.items():
            row = dict(target=C.TARGET_EN[t], selection=name)
            for col, x in (("heuristic", r), ("composition", r + "|composition"), ("recognized", r + "|recognized")):
                m, lo, hi = paired(p[x], p["random_80"])
                row.update({col: m, col + "_half_width": (hi - lo) / 2, col + "_excludes_zero": bool(lo > 0 or hi < 0)})
            row["correct_pct"] = correct[t, r]
            rows.append(row)
    C.save_table(pd.DataFrame(rows), "e2_selection_controls_app")


if __name__ == "__main__":
    main()
