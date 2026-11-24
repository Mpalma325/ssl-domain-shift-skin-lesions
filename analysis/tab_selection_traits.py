"""Traits of the 80 images chosen on BCN20000 (k = 16), Section 5.2.1 and Appendix 7.4.

Per set, the mean over its 80 images; per selection, the mean over the 30 draws (6 restarts for the oracle). Pool = all
candidate images. MEL and BCC: labeled images of the class per set of 80. Traits are defined in selection.traits().
e2_traits_source, e2_traits_space: pool, oracle and the cold-start strategies grouped in three families (min and max
over the members). *_app: every selection. e2_traits_relative_app: 100 (value / pool - 1).
Inputs: results/study2/selection_proxy_traits.csv, selection_proxy_oracle_traits.csv. Outputs: tables/.
"""
import pandas as pd

import common as C
import selection as S
from fig_selection_proxy import NAMES

NAMES = {**NAMES, "pool": "Pool (all candidates)"}
FAMILIES = [("Diversity", ["kcenter"]), ("Representativeness", ["kmeans_centers", "typiclust", "pseudo_class_balance"]),
            ("Coverage", ["probcover", "maxherding"])]
SOURCE = ["resnet_correct", "networks_agree", "confidence", "margin"]
SPACE = ["typicality", "centrality", "MEL", "BCC"]


def per_selection():
    t = pd.concat([S.results("traits"), S.results("oracle_traits")], ignore_index=True)
    t = t[t.target == "bcn20000"]
    n = t[["n_" + c for c in C.CLASSES]].sum(axis=1)
    t["MEL"], t["BCC"] = 80 * t.n_mel / n, 80 * t.n_bcc / n
    return t.groupby("selection")[S.TRAITS + ["MEL", "BCC"]].mean()


def main():
    M = per_selection()
    app = ["pool", "random_80", "random_per_class", "oracle"] + S.COLD_START
    for name, cols in (("source", SOURCE), ("space", SPACE)):
        rows = [dict(selection=NAMES[s], **{f"{c}_min": M.loc[s, c] for c in cols},
                     **{f"{c}_max": M.loc[s, c] for c in cols}) for s in ("pool", "oracle")]
        for fam, members in FAMILIES:
            rows.append(dict(selection=fam, **{f"{c}_min": M.loc[members, c].min() for c in cols},
                             **{f"{c}_max": M.loc[members, c].max() for c in cols}))
        C.save_table(pd.DataFrame(rows), f"e2_traits_{name}")
        C.save_table(pd.DataFrame([dict(selection=NAMES[s], **M.loc[s, cols].to_dict()) for s in app]),
                     f"e2_traits_{name}_app")
    rel = [dict(selection=NAMES[s], **(100 * (M.loc[s, S.TRAITS] / M.loc["pool", S.TRAITS] - 1)).to_dict())
           for s in app[1:]]
    C.save_table(pd.DataFrame(rel), "e2_traits_relative_app")


if __name__ == "__main__":
    main()
