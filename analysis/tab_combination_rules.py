"""Rules that combine the four experts at k = 16, tab:e2_reglas (Section 5.2.2).

Fixed rules on the 20 log-probabilities: weighted power mean (q = 1, -0.5; q -> 0 is the sum of Eq. e2_vote), weighted
majority vote of the top classes (ties broken by the sum) and median of the probabilities. Fitted weights for the sum
with 16 validation images per class (drawn from the validation partition without copies of HAM10000 images): one
weight per expert, or one per expert and class plus class biases; class-balanced cross-entropy with an L2 pull towards
the sum, strength chosen by 5-fold cross-validation grouped by lesion. Logistic regression and gradient boosting on the
20 centred log-probabilities with all the validation labels. Expert oracle: correct if any expert's top class is right.
Test balanced accuracy, mean of five seeds. Inputs: results/study2/*. Output: tables/e2_reglas.csv.
"""
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

import common as C
import experts as X

A = X.WEIGHTS
LAMS = (1e4, 1e3, 1e2, 10.0, 1.0, 0.1)   # most to least regularized; lambda = inf is the sum itself


def features(L):
    Xc = L - L.mean(2, keepdims=True)
    Xt = np.ascontiguousarray(Xc.transpose(0, 2, 1))
    return dict(X=Xc, Xt=Xt, Sb=Xt @ A)


def sub(d, i):
    return {k: v[i] for k, v in d.items()}


def balanced_weights(y):
    w = 1.0 / np.bincount(y, minlength=5)[y]
    return w / w.sum()


def theta0(fam):
    return A.copy() if fam == "expert" else np.r_[np.repeat(A, 5), np.zeros(5)]


def score(fam, th, d):
    if fam == "expert":
        return d["Xt"] @ th
    return (d["X"] * th[:20].reshape(4, 5)).sum(1) + th[20:]


def grad(fam, d, G):
    if fam == "expert":
        return np.tensordot(G, d["Xt"], axes=([0, 1], [0, 1]))
    return np.r_[(d["X"] * G[:, None, :]).sum(0).ravel(), G.sum(0)]


def fit(fam, d, y, lam, th_init, th_0):
    om, i, r = balanced_weights(y), np.arange(len(y)), lam / len(y)

    def f(th):
        s = score(fam, th, d)
        s = s - s.max(1, keepdims=True)
        e = np.exp(s)
        z = e.sum(1, keepdims=True)
        G = e / z
        G[i, y] -= 1
        G *= om[:, None]
        dt = th - th_0
        return -(om * (s[i, y] - np.log(z[:, 0]))).sum() + r * (dt @ dt), grad(fam, d, G) + 2 * r * dt

    return minimize(f, th_init, jac=True, method="L-BFGS-B", options={"maxiter": 500}).x


def fitted_weights(fam, d, y, groups, d_test, seed):
    """Regularization chosen by grouped 5-fold CV (pooled out-of-fold BA; ties -> more regularized)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, groups))
    th_0 = theta0(fam)
    oof = {lam: np.zeros(len(y), int) for lam in LAMS}
    oof[np.inf] = score(fam, th_0, d).argmax(1)
    for tr, te in folds:
        th = th_0
        for lam in LAMS:
            th = fit(fam, sub(d, tr), y[tr], lam, th, th_0)
            oof[lam][te] = score(fam, th, sub(d, te)).argmax(1)
    lam = max(oof, key=lambda l: (X.ba(y, oof[l]), l))
    th = th_0 if np.isinf(lam) else fit(fam, d, y, lam, th_0, th_0)
    return score(fam, th, d_test).argmax(1)


def fixed_rules(L):
    w = A / A.sum()
    lw = np.log(w)[None, :, None]
    S = L.transpose(0, 2, 1) @ A
    top = L.argmax(2)
    votes = np.zeros((len(L), 5))
    for e in range(4):
        np.add.at(votes, (np.arange(len(L)), top[:, e]), A[e])
    return {"Sum of log-probabilities": S.argmax(1),
            "Power mean, q=1": (logsumexp(L + lw, axis=1)).argmax(1),
            "Power mean, q=-0.5": (logsumexp(-0.5 * L + lw, axis=1) / -0.5).argmax(1),
            "Weighted majority vote": (votes * 1e6 + S).argmax(1),
            "Median": np.median(np.exp(L), axis=1).argmax(1)}


def main():
    rows = []
    for ti, t in enumerate(X.TARGETS2):
        for s in C.SEEDS:
            M = X.table(t, s)
            Lt, yt = M["Lt"], M["yt"]
            res = {k: X.ba(yt, p) for k, p in fixed_rules(Lt).items()}
            # 16 per class from the validation partition without the copies of HAM10000 images
            lab = np.flatnonzero(~M["copy_v"])
            yl = M["yv"][lab]
            g = pd.factorize(M["lesion_v"][lab])[0]
            rng = np.random.default_rng([32, 10 + ti, s])
            perm = [rng.permutation(np.flatnonzero(yl == c)) for c in range(5)]
            idx = np.sort(np.concatenate([p[:16] for p in perm]))
            dl, dt = features(M["Lv"][lab][idx]), features(Lt)
            res["One weight per expert"] = X.ba(yt, fitted_weights("expert", dl, yl[idx], g[idx], dt, s))
            res["One weight per expert and class"] = X.ba(yt, fitted_weights("expert_class", dl, yl[idx], g[idx], dt, s))
            # classifiers on the 20 centred log-probabilities, all validation labels
            Fv = (M["Lv"] - M["Lv"].mean(2, keepdims=True)).reshape(len(M["Lv"]), 20)
            Ft = (Lt - Lt.mean(2, keepdims=True)).reshape(len(Lt), 20)
            yv = M["yv"]
            w = (1.0 / np.bincount(yv, minlength=5))[yv]
            for nm, m in (("Logistic regression", LogisticRegression(C=1.0, max_iter=3000)),
                          ("Gradient boosting", HistGradientBoostingClassifier(
                              max_iter=200, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
                              random_state=s))):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m.fit(Fv, yv, sample_weight=w * len(yv) / w.sum())
                res[nm] = X.ba(yt, m.predict(Ft))
            ok = (Lt.argmax(2) == yt[:, None]).any(1)
            res["Expert oracle"] = 100 * np.mean([ok[yt == c].mean() for c in range(5)])
            rows += [dict(target=t, seed=s, rule=k, ba=v) for k, v in res.items()]
        print(f"  {t} done")
    D = pd.DataFrame(rows)
    T = D.pivot_table(index="rule", columns="target", values="ba", aggfunc="mean", sort=False)
    T = T.reset_index().rename(columns={"milk10k": "MILK10k", "bcn20000": "BCN20000"})[["rule", "MILK10k", "BCN20000"]]
    T.insert(0, "group", ["Sum"] + ["Fixed rules"] * 4 + ["Fitted weights for the sum"] * 2 +
             ["Classifiers on the 20 probabilities"] * 2 + ["Upper bound"])
    C.save_table(T, "e2_reglas")
    print(T.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
