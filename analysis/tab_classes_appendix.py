"""Appendix tables tab:auc_class, tab:all_confusions and tab:malignant_benign (primary test set).

SSL - ERM paired by init., target, k and seed; 95 % t interval over five seeds (excludes_zero = asterisk).
auc_class: one-vs-rest AUC per class (softmax probability), per seed mean over methods and k. all_confusions: per
confusion X -> Y and target, over the 60 method x k x init. combinations, mean change, how many increased and how many
with an interval excluding zero. malignant_benign: malignant = AKIEC, BCC, MEL; % of errors crossing groups and of
malignant predicted benign; per-seed means over methods and k, "all" over targets.
Inputs: common.design(), results/predictions_<target>.npz. Outputs: tables/<name>.csv.
"""
import numpy as np
import pandas as pd
from scipy import stats

import common as C

KEY = ["init", "target", "k", "seed"]
MALIGNANT = [C.CLASSES.index(c) for c in ("akiec", "bcc", "mel")]
BENIGN = [C.CLASSES.index(c) for c in ("bkl", "nv")]


def t_interval(v):
    """Mean, half-width of the 95 % t interval and whether it excludes zero."""
    v = np.asarray(v, float)
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return v.mean(), h, bool(abs(v.mean()) > h)


def paired(df, cols):
    """SSL - ERM of the given columns, one row per SSL run (column d_<col>)."""
    erm = df[df.method == "erm"].set_index(KEY)
    ssl = df[df.method != "erm"].copy()
    idx = ssl.set_index(KEY).index
    for c in cols:
        ssl["d_" + c] = ssl[c].to_numpy() - erm.loc[idx, c].to_numpy()
    return ssl


def auc_one_vs_rest(score, positive):
    """Area under the ROC curve (ties count one half), as the Mann-Whitney statistic."""
    r = stats.rankdata(score)
    n1 = positive.sum()
    n0 = len(positive) - n1
    return (r[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def run_auc():
    R = C.runs()
    rows = []
    for target in C.TARGETS:
        z = C.predictions(target)
        logits = z["logits_test"]  # decompress once per target
        mask = C.primary_mask(target, z["ids_test"])
        y = z["labels_test"].astype(int)[mask]
        for r in R[(R.target == target) & (R.method != "fully_supervised")].itertuples():
            lt = np.asarray(logits[r.row][mask], float)
            p = np.exp(lt - lt.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            row = dict(init=r.init, target=target, k=r.k, method=r.method, seed=r.seed)
            for j, c in enumerate(C.CLASSES):
                row["auc_" + c] = 100 * auc_one_vs_rest(p[:, j], y == j)
            rows.append(row)
    return pd.DataFrame(rows)


def auc_class():
    dd = paired(run_auc(), ["auc_" + c for c in C.CLASSES])
    rows = []
    for t in C.TARGETS:
        for init in C.INITS:
            s = dd[(dd.init == init) & (dd.target == t)].groupby("seed")[["d_auc_" + c for c in C.CLASSES]].mean()
            for c in C.CLASSES:
                mean, h, sig = t_interval(s["d_auc_" + c])
                rows.append(dict(target=t, init=init, cls=c, mean=mean, half_width=h, excludes_zero=sig))
    C.save_table(pd.DataFrame(rows), "auc_class")


def all_confusions(df):
    t = df[KEY + ["method"]].copy()
    cols = []
    for i in range(5):
        tot = sum(df[f"cm_{i}{j}"] for j in range(5))
        for j in range(5):
            if i != j:
                t[f"r_{i}{j}"] = 100 * df[f"cm_{i}{j}"] / tot
                cols.append(f"r_{i}{j}")
    dd = paired(t, cols)
    rows = []
    for (init, target, k, m), g in dd.groupby(["init", "target", "k", "method"]):
        for c in cols:
            mean, h, _ = t_interval(g["d_" + c])
            rows.append(dict(target=target, error=c, mean=mean, increased=mean > 0, significant=mean > h))
    R = (pd.DataFrame(rows).groupby(["target", "error"])
         .agg(mean=("mean", "mean"), increased=("increased", "sum"), significant=("significant", "sum"),
              n=("increased", "size")))
    assert (R.n == 60).all()
    out = []
    for c in cols:
        i, j = int(c[2]), int(c[3])
        row = {"confusion": f"{C.CLASSES_EN[i]} -> {C.CLASSES_EN[j]}"}
        for target in C.TARGETS:
            r = R.loc[(target, c)]
            row[f"{target}_mean"] = r["mean"]
            row[f"{target}_increased"] = int(r.increased)
            row[f"{target}_significant"] = int(r.significant)
        out.append(row)
    C.save_table(pd.DataFrame(out), "all_confusions")


def malignant_benign(df):
    cm = lambda i, j: df[f"cm_{i}{j}"]
    err = sum(cm(i, j) for i in range(5) for j in range(5) if i != j)
    mb = sum(cm(i, j) for i in MALIGNANT for j in BENIGN)
    bm = sum(cm(i, j) for i in BENIGN for j in MALIGNANT)
    n_mal = sum(cm(i, j) for i in MALIGNANT for j in range(5))
    d = df[KEY + ["method"]].copy()
    d["crossing"] = 100 * (mb + bm) / err
    d["malignant_as_benign"] = 100 * mb / n_mal
    d["group"] = np.where(d.method == "erm", "erm", "ssl")
    ps = d.groupby(["init", "target", "group", "seed"])[["crossing", "malignant_as_benign"]].mean()
    rows = []
    for init in C.INITS:
        for t in C.TARGETS + ["all"]:
            if t == "all":
                e = ps.loc[(init, slice(None), "erm")].groupby("seed").mean()
                s = ps.loc[(init, slice(None), "ssl")].groupby("seed").mean()
            else:
                e, s = ps.loc[(init, t, "erm")], ps.loc[(init, t, "ssl")]
            row = dict(init=init, target=t)
            for m in ("crossing", "malignant_as_benign"):
                mean, h, sig = t_interval(s[m] - e[m])
                row.update({f"{m}_erm": e[m].mean(), f"{m}_ssl": s[m].mean(), f"{m}_diff": mean,
                            f"{m}_half_width": h, f"{m}_excludes_zero": sig})
            rows.append(row)
    C.save_table(pd.DataFrame(rows), "malignant_benign")


def main():
    df = C.design()
    all_confusions(df)
    malignant_benign(df)
    auc_class()


if __name__ == "__main__":
    main()
