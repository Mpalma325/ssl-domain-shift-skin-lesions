"""Contribution of each expert by number of labeled images, margin recovered and sensitivity to lambda and T
(Section 5.2.2 and Appendix 7.4).

Series by k (16, 32, 100): RH and CH calibrated on the k x 5 labeled images, RE = the ERM run of the Study 1 panel at
that k, CE = the ConvNeXt-T ERM run at that k. At k = 16 the combination therefore differs slightly from tab:e2_mesa,
which uses the ResNet ERM run of Study 2.
  app_e2_contribucion: Shapley value of each expert (BA over the 15 sub-combinations, empty set = 20 %) and its share,
                       and four experts minus the three without it (95 % t interval, seeds in favour).
  e2_brecha_mesa:      (combination - ERM) / (FS - ERM), with the Study 1 ERM and fully supervised ResNet-50 means.
  app_e2_lambda_t:     lambda of RE and CE chosen on validation (without copies of HAM10000 images) among 0.25 ... 8
                       against 0.5; and at k = 16 (Study 2 combination) T = 4 against a temperature per expert fitted by
                       NLL on the 16 validation images per class.
Inputs: results/study2/*, predictions of the Study 1 panel. Outputs: tables/<label>.csv.
"""
import itertools
import math

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar

import common as C
import experts as X

ORDER = ["RH", "CH", "RE", "CE"]                 # names of sub-combinations; arrays below use X.EXPERTS order
LAMBDAS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def t_interval(d):
    d = np.asarray(d, float)
    sd = d.std(ddof=1)
    h = stats.t.ppf(0.975, len(d) - 1) * sd / np.sqrt(len(d)) if sd > 0 else 0.0
    return d.mean(), d.mean() - h, d.mean() + h


def name(S):
    return "+".join(e for e in ORDER if e in S)


def values(L, y, lam):
    """BA of every sub-combination (empty = 20) with weight 1 for RH and CH and lam for RE and CE."""
    w = {"RE": lam, "RH": 1.0, "CH": 1.0, "CE": lam}
    out = {frozenset(): 20.0}
    for r in range(1, 5):
        for S in itertools.combinations(X.EXPERTS, r):
            s = sum(w[e] * L[:, X.EXPERTS.index(e)] for e in S)
            out[frozenset(S)] = X.ba(y, s.argmax(1))
    return out


def shapley(val):
    phi = {}
    for e in X.EXPERTS:
        others = [x for x in X.EXPERTS if x != e]
        phi[e] = sum(math.factorial(r) * math.factorial(3 - r) / 24 * (val[frozenset(c) | {e}] - val[frozenset(c)])
                     for r in range(4) for c in itertools.combinations(others, r))
    return phi


def cases():
    """{(t, k, s): (L_val, y_val, L_test, y_test)}, validation without copies, primary test set."""
    out = {}
    for t in X.TARGETS2:
        im = X.images(t)
        keep_v, keep_t = ~im["val"]["copy"], im["test"]["primary"]
        for k in C.KS:
            for s in C.SEEDS:
                E = X.four_experts(t, s, k, re_source="panel")
                out[(t, k, s)] = (E["val"][keep_v], im["val"]["y"][keep_v], E["test"][keep_t], im["test"]["y"][keep_t])
    return out


def by_k():
    """Per seed: BA of the sub-combinations (fixed and chosen lambda) and Shapley values."""
    D = cases()
    ba_rows, sh_rows = [], []
    for t in X.TARGETS2:
        for k in C.KS:
            seeds = [D[(t, k, s)] for s in C.SEEDS]
            val = {lam: np.mean([X.ba(yv, X.combine(Lv, [lam, 1, 1, lam]).argmax(1)) for Lv, yv, _, _ in seeds])
                   for lam in LAMBDAS}
            chosen = max(val, key=val.get)  # first of the grid on ties
            for mode, lam in (("fixed", 0.5), ("chosen", chosen)):
                for s, (_, _, Lt, yt) in zip(C.SEEDS, seeds):
                    v = values(Lt, yt, lam)
                    for S, b in v.items():
                        if S:
                            ba_rows.append(dict(target=t, k=k, mode=mode, lam=lam, seed=s, subset=name(S), ba=b))
                    if mode == "fixed":
                        for e, p in shapley(v).items():
                            sh_rows.append(dict(target=t, k=k, seed=s, expert=e, shapley=p,
                                                total=v[frozenset(X.EXPERTS)] - 20.0))
    return pd.DataFrame(ba_rows), pd.DataFrame(sh_rows)


def fitted_temperature(L, y):
    """Temperature on log-probabilities by NLL (bounded search of log T in [-3, 3])."""
    def nll(lt):
        return -X.log_softmax(L / np.exp(lt))[np.arange(len(y)), y].mean()
    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


def main():
    P, H = by_k()
    full = name(set(ORDER))
    fixed = P[P["mode"] == "fixed"]

    rows = []
    for t in X.TARGETS2:
        for k in C.KS:
            q = fixed[(fixed.target == t) & (fixed.k == k)]
            f = q[q.subset == full].set_index("seed").ba
            for e in ORDER:
                d = (f - q[q.subset == name(set(ORDER) - {e})].set_index("seed").ba.loc[f.index]).values
                m, lo, hi = t_interval(d)
                g = H[(H.target == t) & (H.k == k) & (H.expert == e)]
                rows.append(dict(target=X.NAME2[t], k=k, ba_four_experts=f.mean(), expert=e, shapley=g.shapley.mean(),
                                 share=(100 * g.shapley / g.total).mean(), removal=m, removal_t_lo=lo,
                                 removal_t_hi=hi, seeds_four_higher=int((d > 0).sum())))
    C.save_table(pd.DataFrame(rows), "app_e2_contribucion")

    p = C.panel()
    ba = p[p.init == "ham"].groupby(["method", "target", "k"]).ba.mean()
    gap = []
    for t, label in (("bcn20000", "BCN20000"), ("milk10k", "MILK10k dermoscopic")):
        r = dict(target=label)
        for k in C.KS:
            m = fixed[(fixed.target == t) & (fixed.k == k) & (fixed.subset == full)].ba.mean()
            erm, fs = ba[("erm", t, k)], ba[("fully_supervised", t, 0)]
            r[f"k{k}"] = 100 * (m - erm) / (fs - erm)
        gap.append(r)
    C.save_table(pd.DataFrame(gap), "e2_brecha_mesa")

    lam_rows = []
    for t in X.TARGETS2:
        for k in C.KS:
            q = P[(P.target == t) & (P.k == k) & (P.subset == full)]
            f = q[q["mode"] == "fixed"].set_index("seed").ba
            c = q[q["mode"] == "chosen"].set_index("seed")
            m, lo, hi = t_interval((c.ba - f.loc[c.index]).values)
            lam_rows.append(dict(target=X.NAME2[t], k=k, variant="lambda", chosen_lambda=c.lam.iloc[0],
                                 ba_reference=f.mean(), ba_variant=c.ba.mean(), diff=m, t_lo=lo, t_hi=hi))
    for t in X.TARGETS2:
        ref, fit = [], []
        for s in C.SEEDS:
            M = X.table(t, s)
            Lt = M["Lt"].copy()
            for e in (0, 3):  # RE, CE: their log-probabilities already carry T = 4
                Lt[:, e] = X.log_softmax(Lt[:, e] / fitted_temperature(M["Lv"][M["v16"], e], M["yv"][M["v16"]]))
            ref.append(X.ba(M["yt"], X.combine(M["Lt"]).argmax(1)))
            fit.append(X.ba(M["yt"], X.combine(Lt).argmax(1)))
        m, lo, hi = t_interval(np.array(fit) - np.array(ref))
        lam_rows.append(dict(target=X.NAME2[t], k=16, variant="temperature", ba_reference=np.mean(ref),
                             ba_variant=np.mean(fit), diff=m, t_lo=lo, t_hi=hi))
    C.save_table(pd.DataFrame(lam_rows), "app_e2_lambda_t")
    with pd.option_context("display.width", 200):
        for n in ("app_e2_contribucion", "e2_brecha_mesa", "app_e2_lambda_t"):
            print(pd.read_csv(f"{C.TABLES}/{n}.csv").round(2).to_string(index=False))


if __name__ == "__main__":
    main()
