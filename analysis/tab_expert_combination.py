"""The four-expert combination against the baselines at k = 16, Section 5.2.2 and Appendix 7.4.

tab:e2_mesa (BA and four experts - method), tab:e2_mesa_ba_app (mean, SD, seeds) and tab:e2_mesa_dif_app (paired
differences with 95 % t interval, two-level bootstrap over seeds and test lesions, B = 10,000, and seeds in favour).
Methods: RH, CH, Four ERM (RE, the Study 1 ERM run and two ConvNeXt-T ERM runs, log_softmax(z / 4) summed), the
ensemble of the ten SSL methods of Study 1 (same sum), S+T, L2-SP, Co-Tuning, MME, the ResNet-only (RH + 0.5 RE),
ConvNeXt-only (CH + 0.5 CE) and RH + CH combinations, the four-expert combination, and the ConvNeXt-T fully supervised
(paired by seed number). A difference is marked when both intervals exclude zero.
Inputs: results/study2/*, results/predictions_*.npz. Outputs: tables/e2_mesa.csv, e2_mesa_ba_app.csv,
e2_mesa_dif_app.csv.
"""
import numpy as np
import pandas as pd
from scipy import stats

import common as C
import experts as X

B = 10_000
SEED = 20261121
FOUR, CH = "Four-expert combination", "CH (source ConvNeXt)"
FS = "ConvNeXt-T, fully supervised"
SUBSETS = {"RH (source ResNet)": [0, 1, 0, 0], CH: [0, 0, 1, 0], "ResNet-only combination": [.5, 1, 0, 0],
           "ConvNeXt-only combination": [0, 0, 1, .5], FOUR: [.5, 1, 1, .5],
           "HAM10000 networks combined (RH + CH)": [0, 1, 1, 0]}
RUN_MODELS = {"S+T": "source_target", "L2-SP": "l2sp", "Co-Tuning": "cotuning", "MME": "mme", FS: "convnext_fully_supervised"}
# bootstrap column order (fixes the random draws)
METHODS = ["RH (source ResNet)", CH, "Four ERM networks", "Ensemble of the 10 SSL methods", "S+T", "L2-SP",
           "Co-Tuning", "MME", "ResNet-only combination", "ConvNeXt-only combination", FOUR,
           "HAM10000 networks combined (RH + CH)", FS]
MAIN = METHODS[:11]                                               # columns of tab:e2_mesa
APPENDIX = METHODS[:10] + [METHODS[11], FOUR, FS]                 # rows of the appendix tables


def predictions(t, seed):
    """{method: argmax on the primary test set} for one seed."""
    M = X.table(t, seed)
    m = M["test_mask"]
    P = {k: X.combine(M["Lt"], w).argmax(1) for k, w in SUBSETS.items()}
    erm = [X.panel_logits(t, "erm", seed)[1], X.run_logits(t, "convnext_erm", seed)[1],
           X.run_logits(t, "convnext_erm_old", seed)[1]]
    P["Four ERM networks"] = (M["Lt"][:, 0] + sum(X.erm_logprob(z[m]) for z in erm)).argmax(1)
    P["Ensemble of the 10 SSL methods"] = sum(X.erm_logprob(X.panel_logits(t, s, seed)[1][m]) for s in C.SSL).argmax(1)
    runs = X.expert_runs()
    for name, model in RUN_MODELS.items():
        if ((runs.target == t) & (runs.model == model) & (runs.seed == seed)).any():
            P[name] = X.run_logits(t, model, seed, 0 if model == "convnext_fully_supervised" else 16)[1][m].argmax(1)
    return P


def t_interval(d):
    d = np.asarray(d, float)
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), d.mean() - h, d.mean() + h


def ba_boot(correct, y, groups, rng):
    """B x K balanced accuracies with multinomial weights over the test lesions."""
    G = int(groups.max()) + 1
    Y1 = np.eye(5)[y]
    Xc = (Y1[:, :, None] * correct[:, None, :]).reshape(len(y), -1)
    out = np.empty((B, correct.shape[1]))
    for b0 in range(0, B, 250):
        nb = min(250, B - b0)
        W = rng.multinomial(G, np.full(G, 1.0 / G), size=nb).astype(float)[:, groups]
        out[b0:b0 + nb] = 100 * ((W @ Xc).reshape(nb, 5, -1) / (W @ Y1)[:, :, None]).mean(1)
    return out


def main():
    ba_rows, dif_rows = [], []
    for it, t in enumerate(X.TARGETS2):
        P = {m: {} for m in METHODS}
        for s in C.SEEDS:
            for m, p in predictions(t, s).items():
                P[m][s] = p
        M0 = X.table(t, 0)
        y = M0["yt"]
        for m in METHODS:
            v = np.array([X.ba(y, P[m][s]) for s in sorted(P[m])])
            ba_rows.append(dict(target=t, method=m, ba=v.mean(), sd=v.std(ddof=1), n_seeds=len(v)))
        # MILK10k has one image per lesion; BCN20000 lesions keep their images together
        _, groups = np.unique(M0["ids_t"] if t == "milk10k" else M0["lesion_t"], return_inverse=True)
        keys = [(m, s) for m in METHODS for s in sorted(P[m])]
        col = {k: j for j, k in enumerate(keys)}
        BB = ba_boot(np.stack([(P[m][s] == y).astype(float) for m, s in keys], 1), y, groups,
                     np.random.default_rng([SEED, it]))
        draws = {}
        for a, b, kind in [(FOUR, m, "four_minus_method") for m in METHODS if m != FOUR] + \
                          [(m, CH, "method_minus_ch") for m in METHODS if m != CH]:
            seeds = sorted(set(P[a]) & set(P[b]))
            n = len(seeds)
            d = np.array([X.ba(y, P[a][s]) - X.ba(y, P[b][s]) for s in seeds])
            mean, lo, hi = t_interval(d)
            if n not in draws:
                draws[n] = np.random.default_rng([SEED, it, n]).integers(0, n, size=(B, n))
            Db = np.stack([BB[:, col[(a, s)]] - BB[:, col[(b, s)]] for s in seeds], 1)
            blo, bhi = np.percentile(np.take_along_axis(Db, draws[n], 1).mean(1), [2.5, 97.5])
            dif_rows.append(dict(target=t, contrast=kind, method=b if kind == "four_minus_method" else a, diff=mean,
                                 t_lo=lo, t_hi=hi, boot_lo=blo, boot_hi=bhi, seeds_positive=int((d > 0).sum()),
                                 n_seeds=n, significant=bool((lo > 0 or hi < 0) and (blo > 0 or bhi < 0))))
        print(f"  {t} done")
    A, D = pd.DataFrame(ba_rows), pd.DataFrame(dif_rows)

    four = D[D.contrast == "four_minus_method"].set_index(["target", "method"])
    main_rows = []
    for t in X.TARGETS2:
        for m in MAIN:
            r = dict(target=X.NAME2[t], method=m, ba=A[(A.target == t) & (A.method == m)].ba.item())
            if m != FOUR:
                r["four_minus_method"] = four.loc[(t, m), "diff"]
                r["significant"] = four.loc[(t, m), "significant"]
            main_rows.append(r)
    C.save_table(pd.DataFrame(main_rows), "e2_mesa")

    app = []
    for m in APPENDIX:
        r = dict(method=m)
        for t in X.TARGETS2:
            a = A[(A.target == t) & (A.method == m)].iloc[0]
            r.update({f"{t}_ba": a.ba, f"{t}_sd": a.sd, f"{t}_seeds": a.n_seeds})
        app.append(r)
    C.save_table(pd.DataFrame(app), "e2_mesa_ba_app")
    D["_t"] = D.target.map({t: i for i, t in enumerate(X.TARGETS2)})
    D["_m"] = D.method.map({m: i for i, m in enumerate(APPENDIX)})
    D = D.sort_values(["_t", "contrast", "_m"]).drop(columns=["_t", "_m"])
    D["target"] = D.target.map(X.NAME2)
    C.save_table(D, "e2_mesa_dif_app")
    with pd.option_context("display.width", 200):
        print(pd.DataFrame(main_rows).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
