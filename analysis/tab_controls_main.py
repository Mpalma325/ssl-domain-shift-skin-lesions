"""Appendix tables from the main-design runs: tab:pseudo_quality_k, tab:pseudo_selected_epoch,
tab:selected_epoch_methods, tab:disagreement_runs, tab:shared_errors_seed, tab:control_lesion (BCN20000 per image
and per lesion, nine SSL methods), tab:control_accuracy and tab:control_epoch (five rules for choosing the epoch).

Inputs: results/runs.csv, predictions (common.py), results/pseudolabels_per_epoch.csv,
results/pseudolabels_per_epoch_pseudo_label.csv, results/val_curves.csv, data/splits/bcn20000.csv.
Outputs: tables/<label>.csv; intervals are mean +- 95 % t half-width, `significant` when it excludes zero.
"""
import itertools
import os

import numpy as np
import pandas as pd
from scipy import stats

import common as C

SSL9 = [m for m in C.SSL if m != "noisy_student"]  # the nine methods trained alongside ERM
PL_METHODS = ["pseudo_label", "fixmatch", "flexmatch", "debiasmatch", "dst"]
TGT_SHORT = {"bcn20000": "BCN20000", "milk10k": "MILK10k derm.", "milk10k_clinico": "MILK10k clin."}


def t_interval(v):
    """Mean and half-width of the 95 % t interval over the values (seeds)."""
    v = np.asarray(v, float)
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return v.mean(), h


def t_cells(cells):
    """Mean of cells, 95 % t with variance clustered by (init, seed) and Welch df; cells = [(init, per-seed diffs)]."""
    nc = len(cells)
    theta = np.mean([d.mean() for _, d in cells])
    U = {}
    for init, d in cells:
        n = len(d)
        e = (d - d.mean()) * np.sqrt(n / (n - 1))
        for s in range(n):
            U[(init, s)] = U.get((init, s), 0.0) + e[s] / (nc * n)
    var, n_init = {}, {}
    for (init, s), u in U.items():
        var[init] = var.get(init, 0.0) + u ** 2
        n_init[init] = max(n_init.get(init, 0), s + 1)
    total = sum(var.values())
    df = total ** 2 / sum(var[a] ** 2 / (n_init[a] - 1) for a in var if var[a] > 0)
    return theta, stats.t.ppf(0.975, df) * np.sqrt(total)


def cell_mean(d, col):
    """Mean over methods per (init, target, k, seed), then over seeds per cell, then over cells (equal weight)."""
    g = d.groupby(["init", "target", "k", "seed"])[col].mean()
    return g.groupby(["init", "target", "k"]).mean().mean()


def pseudo_tables():
    pl = pd.concat([pd.read_csv(os.path.join(C.RESULTS, f)) for f in
                    ("pseudolabels_per_epoch.csv", "pseudolabels_per_epoch_pseudo_label.csv")], ignore_index=True)
    first = pl[pl.epoch == 1].groupby(["method", "init", "k"]).pl_accuracy.mean()
    rows = []
    for m in PL_METHODS:
        row = {"method": C.NAME[m]}
        for init in C.INITS:
            for k in C.KS:
                row[f"{init}_k{k}"] = first[(m, init, k)]
        rows.append(row)
    C.save_table(pd.DataFrame(rows), "pseudo_quality_k")

    best = C.runs()[["run_id", "best_epoch"]]
    sel = pl.merge(best, on="run_id")
    sel = sel[sel.epoch == sel.best_epoch].groupby(["method", "target", "init"])[["pl_ratio", "pl_accuracy"]].mean()
    rows = []
    for m in PL_METHODS:
        for t in C.TARGETS:
            row = {"method": C.NAME[m], "target": TGT_SHORT[t]}
            for init in C.INITS:
                row[f"{init}_pool_with_pl"] = sel.loc[(m, t, init), "pl_ratio"]  # NaN: Pseudo-Label does not log it
                row[f"{init}_correct"] = sel.loc[(m, t, init), "pl_accuracy"]
            rows.append(row)
    C.save_table(pd.DataFrame(rows), "pseudo_selected_epoch")


def selected_epoch_table():
    r = C.runs()
    r = r[r.method != "fully_supervised"]
    rows = []
    for m in ["erm"] + C.SSL:
        x = r[r.method == m]
        row = {"method": C.NAME[m]}
        for t in C.TARGETS + ["all"]:
            e = x.best_epoch if t == "all" else x[x.target == t].best_epoch
            row[f"{t}_mean"], row[f"{t}_sd"] = e.mean(), e.std(ddof=1)
        rows.append(row)
    C.save_table(pd.DataFrame(rows), "selected_epoch_methods")


def value_at(curve, e):
    """Validation BA at epoch e; the last epoch if the run stopped earlier; NaN before the first logged epoch."""
    eps = curve.index.values
    if e < eps[0]:
        return np.nan
    if e > eps[-1]:
        return curve.iloc[-1]
    return curve.loc[eps[eps <= e].max()]


def control_epoch_table():
    cv = pd.read_csv(os.path.join(C.RESULTS, "val_curves.csv"))
    r = C.runs()
    r = r[r.method.isin(["erm"] + SSL9)].set_index("run_id")
    curves = {rid: g.set_index("epoch").val_ba.sort_index() for rid, g in cv.groupby("run_id") if rid in r.index}
    m = r.loc[list(curves)].reset_index()
    m["e_star"] = [curves[i].idxmax() for i in m.run_id]
    # epoch fixed without the validation set of the target: median over the other two targets of the median epoch
    # selected by the same method, initialization and k
    med = m.groupby(["method", "init", "k", "target"]).e_star.median()
    fixed = [int(round(np.nanmedian([med[(x.method, x.init, x.k, t)] for t in C.TARGETS if t != x.target])))
             for x in m.itertuples()]
    rules = {"best": [curves[i].max() for i in m.run_id],
             "fixed_other_targets": [value_at(curves[i], e) for i, e in zip(m.run_id, fixed)],
             "last": [curves[i].iloc[-1] for i in m.run_id],
             "epoch_15": [value_at(curves[i], 15) for i in m.run_id],
             "mean_from_5": [curves[i][curves[i].index >= 5].mean() for i in m.run_id]}
    for k_, v in rules.items():
        m[k_] = v
    erm = m[m.method == "erm"].set_index(["init", "target", "k", "seed"])
    d = m[m.method != "erm"].copy()
    key = list(zip(d.init, d.target, d.k, d.seed))
    for k_ in rules:
        d["d_" + k_] = d[k_].values - erm.loc[key, k_].values
    names = {"best": "Best validation epoch (main)", "fixed_other_targets": "Fixed from other targets",
             "last": "Last epoch", "epoch_15": "Epoch 15", "mean_from_5": "Mean from epoch 5"}
    rows = []
    for k_, name in names.items():
        row = {"rule": name, "all": cell_mean(d, "d_" + k_)}
        for t in C.TARGETS:
            row[t] = cell_mean(d[d.target == t], "d_" + k_)
        rows.append(row)
    C.save_table(pd.DataFrame(rows), "control_epoch")


def test_preds():
    """{target: (pred[run row] on the primary test set, labels)}; argmax of the logits."""
    out = {}
    for t in C.TARGETS:
        z = C.predictions(t)
        mask = C.primary_mask(t, z["ids_test"])
        out[t] = (z["logits_test"][:, mask].argmax(2), z["labels_test"][mask].astype(int), z["ids_test"][mask])
    return out


def disagreement_and_errors(P):
    r = C.runs()
    r = r[r.method != "fully_supervised"]
    methods = ["erm"] + C.SSL
    dis_rows, err_rows = [], []
    for init in C.INITS:
        for t in C.TARGETS:
            preds, y, _ = P[t]
            for k in C.KS:
                x = r[(r.init == init) & (r.target == t) & (r.k == k)].set_index(["method", "seed"]).row
                pred = {(m, s): preds[x[(m, s)]] for m in methods for s in C.SEEDS}

                def dis(a, b):  # mean over true classes of the share of images predicted differently
                    return 100 * np.mean([np.mean(pred[a][y == c] != pred[b][y == c]) for c in range(5)])

                other_seed = np.mean([dis((m, a), (m, b)) for m in methods for a, b in itertools.combinations(C.SEEDS, 2)])
                other_method = np.mean([dis((a, s), (b, s)) for s in C.SEEDS for a, b in itertools.combinations(methods, 2)])
                dis_rows.append({"target": t, "k": k, "init": init, "same_seed_other_method": other_method,
                                 "same_method_other_seed": other_seed})

                ok = {key: p == y for key, p in pred.items()}
                pairs = [(s, o) for s in C.SEEDS for o in C.SEEDS if o != s]
                erm_also = np.mean([100 * (~ok[("erm", s)][~ok[("erm", o)]]).mean() for s, o in pairs])
                erm_same = np.mean([100 * (pred[("erm", s)][~ok[("erm", o)]] == pred[("erm", o)][~ok[("erm", o)]]).mean()
                                    for s, o in pairs])
                for m in C.SSL:
                    for s in C.SEEDS:
                        q, qok = pred[(m, s)], ok[(m, s)]

                        def shared(o):
                            wrong = ~ok[("erm", o)]
                            return 100 * (~qok[wrong]).mean(), 100 * (q[wrong] == pred[("erm", o)][wrong]).mean()

                        a_same, c_same = shared(s)
                        other = np.array([shared(o) for o in C.SEEDS if o != s])
                        err_rows.append({"init": init, "target": t, "k": k, "method": m, "seed": s,
                                         "also_ssl_same_seed": a_same, "also_ssl_other_seed": other[:, 0].mean(),
                                         "also_erm_other_seed": erm_also, "class_ssl_same_seed": c_same,
                                         "class_ssl_other_seed": other[:, 1].mean(), "class_erm_other_seed": erm_same})
    D = pd.DataFrame(dis_rows)
    D = D.pivot_table(index=["target", "k"], columns="init", values=["same_seed_other_method", "same_method_other_seed"])
    D.columns = [f"{init}_{c}" for c, init in D.columns]
    cols = [f"{i}_{c}" for i in C.INITS for c in ("same_seed_other_method", "same_method_other_seed")]
    D = D[cols].reindex(pd.MultiIndex.from_product([C.TARGETS, C.KS], names=["target", "k"])).reset_index()
    D["target"] = D.target.map(TGT_SHORT)
    C.save_table(D, "disagreement_runs")

    E = pd.DataFrame(err_rows)
    cols = ["also_ssl_same_seed", "also_ssl_other_seed", "also_erm_other_seed",
            "class_ssl_same_seed", "class_ssl_other_seed", "class_erm_other_seed"]
    rows = []
    for init in C.INITS:
        for t in C.TARGETS:
            x = E[(E.init == init) & (E.target == t)]
            rows.append({"init": C.INIT_EN[init].replace(" init.", ""), "target": TGT_SHORT[t], **x[cols].mean()})
    rows.append({"init": "All", "target": "", **E.groupby("seed")[cols].mean().mean()})
    C.save_table(pd.DataFrame(rows), "shared_errors_seed")


def accuracy_table():
    p = C.design().copy()
    p["acc"] = 100 * sum(p[f"cm_{i}{i}"] for i in range(5)) / p.n_test
    d = C.deltas(p, ("ba", "acc"))
    rows = []
    for t in C.TARGETS + [None]:
        x = (d if t is None else d[d.target == t]).groupby("seed")[["d_ba", "d_acc"]].mean()
        row = {"target": "All" if t is None else C.TARGET_EN[t]}
        for c in ("ba", "acc"):
            mean, h = t_interval(x["d_" + c])
            row.update({c: mean, c + "_ci95": h, c + "_significant": abs(mean) > h})
        rows.append(row)
    C.save_table(pd.DataFrame(rows), "control_accuracy")


def lesion_table(P):
    preds, y, ids = P["bcn20000"]
    sp = pd.read_csv(os.path.join(C.ROOT, "data", "splits", "bcn20000.csv")).set_index("isic_id")
    lesion = sp.loc[ids, "lesion_id"].values
    _, inv, counts = np.unique(lesion, return_inverse=True, return_counts=True)
    w = 1.0 / counts[inv]  # each lesion weighs 1, its images 1/n

    def ba_lesion(pred):
        return 100 * np.mean([w[(y == c) & (pred == c)].sum() / w[y == c].sum() for c in range(5)])

    r = C.runs()
    r = r[(r.target == "bcn20000") & r.method.isin(["erm"] + SSL9)].copy()
    p = C.panel().set_index("run_id")
    r["ba"] = p.loc[r.run_id, "ba"].values
    r["ba_lesion"] = [ba_lesion(preds[i]) for i in r.row]
    d = C.deltas(r, ("ba", "ba_lesion"))
    out = []
    for c, name in (("d_ba", "Per image"), ("d_ba_lesion", "Per lesion")):
        cells = []
        for (init, k), g in d.groupby(["init", "k"]):
            cells.append((init, g.groupby("seed")[c].mean().reindex(C.SEEDS).values))
        mean, h = t_cells(cells)
        out.append({"evaluation": name, "ssl_minus_erm": mean, "ci95": h, "significant": abs(mean) > h})
    C.save_table(pd.DataFrame(out), "control_lesion")


def main():
    pseudo_tables()
    selected_epoch_table()
    control_epoch_table()
    accuracy_table()
    P = test_preds()
    lesion_table(P)
    disagreement_and_errors(P)


if __name__ == "__main__":
    main()
