"""Appendix tables tab:confidence_methods, tab:confidence_gap and tab:confidence_gap_imagenet.

Per run on the primary test set: confidence = mean max softmax, accuracy (not balanced), gap = their difference (%).
confidence_methods: ERM, each SSL method and fully supervised, HAM10000 init., mean over k and seeds. confidence_gap*:
SSL - ERM gap paired by target, k and seed, per seed mean over k, 95 % t interval over five seeds.
Outputs: tables/confidence_methods.csv, confidence_gap.csv, confidence_gap_imagenet.csv.
"""
import numpy as np
import pandas as pd
from scipy import stats

import common as C

KEY = ["init", "target", "k", "seed"]


def t_interval(v):
    """Mean, half-width of the 95 % t interval and whether it excludes zero."""
    v = np.asarray(v, float)
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return v.mean(), h, bool(abs(v.mean()) > h)


def run_confidence():
    R = C.runs()
    rows = []
    for target in C.TARGETS:
        z = C.predictions(target)
        logits = z["logits_test"]
        mask = C.primary_mask(target, z["ids_test"])
        y = z["labels_test"].astype(int)[mask]
        for r in R[R.target == target].itertuples():
            lt = np.asarray(logits[r.row][mask], float)
            p = np.exp(lt - lt.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            conf, ok = p.max(1), p.argmax(1) == y
            rows.append(dict(init=r.init, target=target, k=r.k, method=r.method, seed=r.seed,
                             confidence=100 * conf.mean(), accuracy=100 * ok.mean(),
                             gap=100 * (conf.mean() - ok.mean())))
    return pd.DataFrame(rows)


def main():
    D = run_confidence()
    order = ["erm"] + C.SSL + ["fully_supervised"]
    ham = D[D.init == "ham"]
    rows = []
    for m in order:
        row = {"method": C.NAME[m]}
        for t in C.TARGETS:
            g = ham[(ham.target == t) & (ham.method == m)][["confidence", "accuracy", "gap"]].mean()
            for c in ("confidence", "accuracy", "gap"):
                row[f"{t}_{c}"] = g[c]
        rows.append(row)
    C.save_table(pd.DataFrame(rows), "confidence_methods")

    erm = D[D.method == "erm"].set_index(KEY)
    ssl = D[D.method.isin(C.SSL)].copy()
    ssl["d_gap"] = ssl.gap.to_numpy() - erm.loc[ssl.set_index(KEY).index, "gap"].to_numpy()
    for init, name in (("ham", "confidence_gap"), ("imagenet", "confidence_gap_imagenet")):
        rows = []
        for m in C.SSL:
            g = ssl[(ssl.init == init) & (ssl.method == m)]
            for t in C.TARGETS:
                mean, h, sig = t_interval(g[g.target == t].groupby("seed").d_gap.mean())
                rows.append(dict(method=m, target=t, mean=mean, half_width=h, excludes_zero=sig))
        C.save_table(pd.DataFrame(rows), name)


if __name__ == "__main__":
    main()
