"""Variation between runs, Section 4.2 (Variation Between Runs).

subset_seed_sd: SD of test BA at k=16, HAM10000 init., ERM and FixMatch on MILK10k dermoscopic and BCN20000; repeated
cells averaged into a 3 x 3 grid; "subsets" = SD of row means, "training seeds" = SD of column means, "repetitions" =
sqrt(mean(d^2) / 2) over the three identical pairs. repetitions: five repetitions of one run (MILK10k dermoscopic,
seed 0): mean pairwise disagreement, range of selected epochs, mean single-run BA and BA of the softmax average.
Inputs: results/factorial_runs.csv, repetitions.npz, repetitions_runs.csv. Outputs: tables/<name>.csv.
"""
from itertools import combinations

import numpy as np
import pandas as pd

import common as C


def subset_seed_sd():
    f = pd.read_csv(f"{C.RESULTS}/factorial_runs.csv")
    rows = []
    for (t, m), g in f.groupby(["target", "method"], sort=False):
        grid = g.groupby(["subset_seed", "train_seed"]).ba.mean().unstack()
        assert grid.shape == (3, 3)
        pairs = g[g.replica == 1].merge(g[g.replica == 0], on=["subset_seed", "train_seed"])
        assert len(pairs) == 3
        d = (pairs.ba_x - pairs.ba_y).to_numpy()
        rows.append(dict(target=C.TARGET_EN[t], method=C.NAME[m], subsets=grid.mean(axis=1).std(ddof=1),
                         training_seeds=grid.mean(axis=0).std(ddof=1), repetitions=np.sqrt((d ** 2 / 2).mean())))
    return pd.DataFrame(rows)


def softmax(z):
    z = np.asarray(z, float)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def repetition_rows(runs, logits, y):
    rows = []
    for m, g in runs.groupby("method", sort=False):
        lg = [logits[r] for r in g.row]
        pred = [x.argmax(1) for x in lg]
        ba = [C.metrics(x, y)["ba"] for x in lg]
        avg = C.metrics(sum(softmax(x) for x in lg), y)["ba"]
        diff = np.mean([100 * (pred[a] != pred[b]).mean() for a, b in combinations(range(len(pred)), 2)])
        rows.append(dict(method=C.NAME[m], n_runs=len(lg), different_predictions=diff,
                         epoch_min=int(g.best_epoch.min()), epoch_max=int(g.best_epoch.max()),
                         ba_single_run=np.mean(ba), ba_average=avg))
    return pd.DataFrame(rows)


def main():
    C.save_table(subset_seed_sd(), "subset_seed_sd")

    z = np.load(f"{C.RESULTS}/repetitions.npz")
    runs = pd.read_csv(f"{C.RESULTS}/repetitions_runs.csv")
    mask = C.primary_mask("milk10k", z["ids_test"])
    logits = z["logits_test"][:, mask]
    y = z["labels_test"].astype(int)[mask]
    C.save_table(repetition_rows(runs, logits, y), "repetitions")

    with pd.option_context("display.width", 200):
        for name in ("subset_seed_sd", "repetitions"):
            print(pd.read_csv(f"{C.TABLES}/{name}.csv").round(1).to_string(index=False))


if __name__ == "__main__":
    main()
