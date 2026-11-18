"""Appendix control tables on MILK10k dermoscopic, k = 16, HAM10000 init.: tab:pseudo_threshold, tab:control_fixmatch,
tab:control_ema, tab:control_natural, tab:control_gpu and tab:repetitions_two.

Inputs: results/controls_fixmatch.csv, controls_mean_teacher_ema.csv, controls_natural_proportions.csv,
controls_gpu.csv, controls_repetitions.npz and the main-design runs. Outputs: tables/<label>.csv.
Paired by seed, mean +- 95 % t half-width (`significant`, `positive_seeds`). Primary test set except tab:control_ema
(complete test set: the teacher was logged only as aggregate metrics).
"""
import itertools
import os

import numpy as np
import pandas as pd
from scipy import stats

import common as C

R = C.RESULTS


def paired(d):
    """Mean, half-width of the 95 % t interval, significance and positive count of paired differences."""
    d = np.asarray(d, float)
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return dict(mean=d.mean(), half_width=h, significant=bool(abs(d.mean()) > h), positive_seeds=int((d > 0).sum()),
                n_seeds=len(d))


def ba_of(pred, y):
    return 100 * np.mean([(pred[y == c] == c).mean() for c in range(5)])


def main_runs(method, seeds):
    """Balanced accuracy (primary test) of main-design runs, HAM10000 init., MILK10k dermoscopic, k = 16."""
    p = C.panel()
    x = p[(p.init == "ham") & (p.target == "milk10k") & (p.k == 16) & (p.method == method)].set_index("seed")
    return x.loc[list(seeds), "ba"].to_numpy()


def pseudo_threshold():
    fm = pd.read_csv(os.path.join(R, "controls_fixmatch.csv"))
    t99 = fm[fm.variant == "threshold_0.99"].sort_values("seed")
    seeds = t99.seed.tolist()
    # threshold 0.95 = main-design FixMatch; same rule as the control runs: mean over epochs 2..last
    pl = pd.read_csv(os.path.join(R, "pseudolabels_per_epoch.csv"))
    pl = pl[(pl.init == "ham") & (pl.target == "milk10k") & (pl.k == 16) & (pl.method == "fixmatch")
            & pl.seed.isin(seeds) & (pl.epoch >= 2)]
    t95 = pl.groupby("seed")[["pl_ratio", "pl_accuracy"]].mean().loc[seeds]
    erm, b95, b99 = main_runs("erm", seeds), main_runs("fixmatch", seeds), t99.ba.to_numpy()
    rows = [dict(row="ERM", pool_with_pseudo_label=np.nan, correct=np.nan, ba=erm.mean()),
            dict(row="Threshold 0.95", pool_with_pseudo_label=t95.pl_ratio.mean(), correct=t95.pl_accuracy.mean(),
                 ba=b95.mean()),
            dict(row="Threshold 0.99", pool_with_pseudo_label=t99.pl_ratio_mean.mean(),
                 correct=t99.pl_accuracy_mean.mean(), ba=b99.mean())]
    for name, d in (("Threshold 0.99 - 0.95 (pp)", b99 - b95), ("Threshold 0.99 - ERM (pp)", b99 - erm)):
        s = paired(d)
        rows.append(dict(row=name, ba=s["mean"], half_width=s["half_width"], significant=s["significant"],
                         positive_seeds=s["positive_seeds"], n_seeds=s["n_seeds"]))
    out = pd.DataFrame(rows)
    out[["positive_seeds", "n_seeds"]] = out[["positive_seeds", "n_seeds"]].astype("Int64")
    C.save_table(out, "pseudo_threshold")


def control_fixmatch():
    fm = pd.read_csv(os.path.join(R, "controls_fixmatch.csv"))
    mu7 = fm[fm.variant == "mu7"].sort_values("seed")
    seeds = mu7.seed.tolist()
    a, b, e = mu7.ba.to_numpy(), main_runs("fixmatch", seeds), main_runs("erm", seeds)
    rows = [dict(comparison=n, **paired(d)) for n, d in
            (("mu=7 - mu=1", a - b), ("mu=7 - ERM", a - e), ("mu=1 - ERM", b - e))]
    C.save_table(pd.DataFrame(rows), "control_fixmatch")


def control_ema():
    m = pd.read_csv(os.path.join(R, "controls_mean_teacher_ema.csv")).sort_values("seed")
    rows = [dict(comparison="Teacher at its best epoch - student",
                 **paired(m.teacher_ba_full_test_at_teacher_epoch - m.student_ba_full_test)),
            dict(comparison="Teacher at the student's epoch - student",
                 **paired(m.teacher_ba_full_test_at_student_epoch - m.student_ba_full_test))]
    C.save_table(pd.DataFrame(rows), "control_ema")


def control_natural():
    n = pd.read_csv(os.path.join(R, "controls_natural_proportions.csv"))
    piv = n.pivot_table(index="seed", columns=["method", "labeling"], values="ba")
    rows = [dict(row="ERM (%)", balanced=piv["erm", "balanced"].mean(), natural=piv["erm", "natural"].mean()),
            dict(row="FixMatch (%)", balanced=piv["fixmatch", "balanced"].mean(),
                 natural=piv["fixmatch", "natural"].mean())]
    for name, d in (("Natural - balanced, ERM (pp)", piv["erm", "natural"] - piv["erm", "balanced"]),
                    ("Natural - balanced, FixMatch (pp)", piv["fixmatch", "natural"] - piv["fixmatch", "balanced"]),
                    ("FixMatch - ERM, balanced (pp)", piv["fixmatch", "balanced"] - piv["erm", "balanced"]),
                    ("FixMatch - ERM, natural (pp)", piv["fixmatch", "natural"] - piv["erm", "natural"])):
        s = paired(d)
        rows.append(dict(row=name, difference=s["mean"], half_width=s["half_width"], significant=s["significant"],
                         n_seeds=s["n_seeds"]))
    out = pd.DataFrame(rows)
    out["n_seeds"] = out["n_seeds"].astype("Int64")
    C.save_table(out, "control_natural")


def control_gpu():
    """Variance components of machine x seed (two repetitions per cell); negative estimates set to zero."""
    g = pd.read_csv(os.path.join(R, "controls_gpu.csv"))
    machines = sorted(g.machine.unique())
    seeds = sorted(g.seed.unique())
    a, b, n = len(machines), len(seeds), 2
    Y = np.array([[g[(g.machine == m) & (g.seed == s)].sort_values("repetition").ba.to_numpy() for s in seeds]
                  for m in machines])
    mu, mi, mj, mij = Y.mean(), Y.mean((1, 2)), Y.mean((0, 2)), Y.mean(2)
    ms_m = b * n * ((mi - mu) ** 2).sum() / (a - 1)
    ms_s = a * n * ((mj - mu) ** 2).sum() / (b - 1)
    ms_ms = n * ((mij - mi[:, None] - mj[None, :] + mu) ** 2).sum() / ((a - 1) * (b - 1))
    ms_e = ((Y - mij[:, :, None]) ** 2).sum() / (a * b * (n - 1))
    comp = {"Machine": max((ms_m - ms_ms) / (b * n), 0.0), "Seed": max((ms_s - ms_ms) / (a * n), 0.0),
            "Machine x seed": max((ms_ms - ms_e) / n, 0.0), "Repetition on the same machine": ms_e}
    tot = sum(comp.values())
    rows = [dict(source=k, variance=v, sd=np.sqrt(v), share=100 * v / tot) for k, v in comp.items()]
    C.save_table(pd.DataFrame(rows), "control_gpu")
    rep = np.abs(Y[:, :, 0] - Y[:, :, 1])
    print("  machine means: " + ", ".join(f"{m} {Y[i].mean():.2f}" for i, m in enumerate(machines))
          + f"; same machine and seed |r1 - r2|: mean {rep.mean():.2f}, max {rep.max():.2f}")


def repetitions_two():
    z = np.load(os.path.join(R, "controls_repetitions.npz"))
    mask = C.primary_mask("milk10k", z["ids_test"])
    y = z["labels_test"].astype(int)[mask]
    lg = z["logits_test"][:, mask].astype(np.float64)
    prob = np.exp(lg - lg.max(2, keepdims=True))
    prob /= prob.sum(2, keepdims=True)
    order = ["erm", "pi_model", "mean_teacher", "pseudo_label", "noisy_student", "uda", "flexmatch", "debiasmatch",
             "dst", "self_tuning"]
    rows = []
    for m in order:
        i = np.where(z["method"] == m)[0]
        assert len(i) == 2
        p = prob[i].argmax(2)
        e = sorted(int(x) for x in z["best_epoch"][i])
        rows.append(dict(method=C.NAME[m], different_predictions=100 * (p[0] != p[1]).mean(),
                         selected_epochs=f"{e[0]} and {e[1]}", ba_single_run=np.mean([ba_of(q, y) for q in p]),
                         ba_average_of_two=ba_of(prob[i].sum(0).argmax(1), y)))
    C.save_table(pd.DataFrame(rows), "repetitions_two")


if __name__ == "__main__":
    pseudo_threshold()
    control_fixmatch()
    control_ema()
    control_natural()
    control_gpu()
    repetitions_two()
