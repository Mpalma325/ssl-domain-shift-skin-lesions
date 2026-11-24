"""Errors shared by the experts, the shared blind spot and prediction churn at k = 16 (Section 5.2.2).

e2_errores: R_norm (Eq. e2_rnorm) with E and O_max computed within difficulty groups, for the pairs of experts grouped
  by what they share (same training data: RH-CH, RE-CE; same architecture and initialization: RH-RE, CH-CE; nothing:
  RH-CE, RE-CH), pooled over five seeds, mean of the two pairs. Difficulty group of a test image: how many of the five
  fully supervised ImageNet-initialized networks of Study 1 misclassify it (0, 1-2, 3-4, 5) x their most frequent class.
  PACS columns: aggregate per pair in results/study2/pacs_shared_errors.csv (linear classifiers on frozen ImageNet
  features, ten seeds; the per-image features are not included).
e2_blind_spot: test images misclassified by RE and by all ten SSL networks of Study 1 (same 80 labels), the share the
  combination, CH, RH, CE and the Study 1 ERM run classify correctly, and the share of the combination's errors in them.
e2_churn: share of test images whose prediction changes between two seeds (mean of the ten pairs of seeds 0-4).
Inputs: results/study2/*, predictions of the Study 1 panel. Outputs: tables/<name>.csv.
"""
import os
from itertools import combinations

import numpy as np
import pandas as pd

import common as C
import experts as X

PAIRS = {("RH", "CH"): "Same training data", ("RE", "CE"): "Same training data",
         ("RH", "RE"): "Same architecture and initialization", ("CH", "CE"): "Same architecture and initialization",
         ("RH", "CE"): "Nothing in common", ("RE", "CH"): "Nothing in common"}
GROUPS = ["Same training data", "Same architecture and initialization", "Nothing in common"]
PACS_SHARED = {"same_training_data": GROUPS[0], "same_architecture_and_initialization": GROUPS[1], "nothing": GROUPS[2]}


def difficulty(t, y, mask):
    """Group of each primary test image from the five fully supervised ImageNet networks of Study 1."""
    J = np.array([X.panel_logits(t, "fully_supervised", s, k=0, init="imagenet")[1][mask].argmax(1) for s in C.SEEDS])
    wrong = np.digitize((J != y[None]).sum(0), [1, 3, 5])
    mode = np.stack([(J == c).sum(0) for c in range(5)], 1).argmax(1)
    return wrong * 5 + mode


def o_e_max(y, a, b, groups):
    """Shared errors O, expected E and maximum O_max, with E and O_max summed over groups x true class."""
    O = float(((a == b) & (a != y)).sum())
    E = Om = 0.0
    for h in np.unique(groups):
        m = groups == h
        for c in range(5):
            mc = m & (y == c)
            n = mc.sum()
            if n == 0:
                continue
            ca = np.bincount(a[mc], minlength=5)
            cb = np.bincount(b[mc], minlength=5)
            off = np.arange(5) != c
            E += (ca[off] * cb[off]).sum() / n
            Om += np.minimum(ca, cb)[off].sum()
    return O, E, Om


def main():
    err, blind, churn = [], [], []
    for t in X.TARGETS2:
        y = X.table(t, 0)["yt"]
        mask = X.table(t, 0)["test_mask"]
        groups = difficulty(t, y, mask)
        P = {}
        for s in C.SEEDS:
            M = X.table(t, s)
            assert (M["yt"] == y).all()
            P[s] = {e: M["Lt"][:, i].argmax(1) for i, e in enumerate(X.EXPERTS)}
            P[s]["vote"] = X.combine(M["Lt"]).argmax(1)
            for m in C.SSL + ["erm"]:
                P[s][m] = X.panel_logits(t, m, s)[1][mask].argmax(1)
        for (a, b), g in PAIRS.items():
            O, E, Om = np.sum([o_e_max(y, P[s][a], P[s][b], groups) for s in C.SEEDS], 0)
            err.append(dict(target=t, a=a, b=b, shared=g, r_norm_difficulty=(O - E) / (Om - E)))

        tot = dict(n=0, T=0, vote_fail=0, vote_fail_T=0)
        res = {m: 0 for m in ("vote", "CH", "RH", "CE", "erm")}
        for s in C.SEEDS:
            p = P[s]
            Ts = np.all([p[m] != y for m in ["RE"] + C.SSL], 0)
            tot["n"] += len(y)
            tot["T"] += Ts.sum()
            fail = p["vote"] != y
            tot["vote_fail"] += fail.sum()
            tot["vote_fail_T"] += (fail & Ts).sum()
            for m in res:
                res[m] += (Ts & (p[m] == y)).sum()
        blind.append(dict(target=X.NAME2[t], blind_spot_pct_of_test=100 * tot["T"] / tot["n"],
                          **{f"correct_{m}_pct": 100 * v / tot["T"] for m, v in res.items()},
                          vote_errors_in_blind_spot_pct=100 * tot["vote_fail_T"] / tot["vote_fail"]))
        for system in ("vote", "RE"):
            ch = [100 * np.mean(P[a][system] != P[b][system]) for a, b in combinations(C.SEEDS, 2)]
            churn.append(dict(target=X.NAME2[t], system="Four-expert combination" if system == "vote" else "RE",
                              churn_pct=np.mean(ch), n_pairs=len(ch)))

    E = pd.DataFrame(err)
    rows = {g: {"shared": g} for g in GROUPS}
    for t in X.TARGETS2:
        for g in GROUPS:
            rows[g][X.NAME2[t]] = E[(E.target == t) & (E.shared == g)].r_norm_difficulty.mean()
    pacs = pd.read_csv(os.path.join(X.STUDY2, "pacs_shared_errors.csv"))
    for tp in ("paintings", "sketches", "cartoons"):
        for code, g in PACS_SHARED.items():
            rows[g]["PACS " + tp] = pacs[(pacs.target == tp) & (pacs.shared == code)].r_norm_difficulty.mean()
    C.save_table(pd.DataFrame(list(rows.values())), "e2_errores")
    E["target"] = E.target.map(X.NAME2)
    C.save_table(E, "e2_errores_pairs")
    C.save_table(pd.DataFrame(blind), "e2_blind_spot")
    C.save_table(pd.DataFrame(churn), "e2_churn")
    with pd.option_context("display.width", 200):
        for n in ("e2_errores", "e2_blind_spot", "e2_churn"):
            print(pd.read_csv(f"{C.TABLES}/{n}.csv").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
