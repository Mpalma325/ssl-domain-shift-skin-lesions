"""Class-level effects of SSL, Section 4.2 (Class-Level Effects). Input: common.panel() (primary test set).

ssl_without_akiec, ssl_without_class[_imagenet]: SSL - ERM BA without a class (mean recall of the other four); per
seed, mean over methods and k (and targets in "All"); mean and 95 % t half-width over five seeds.
pair_changes[_imagenet], pair_changes_images[_imagenet]: class pairs whose X->Y and Y->X confusion changes vs ERM each
keep the sign of their mean in >= 4 of 5 seeds with |mean| >= 1 pp (reversed, separated, both worse); images moved =
|mean change in count X->Y| + |mean change in count Y->X|. mel_nv, bcc_recall, mel_nv_methods, bcc_recall_methods
(+ _imagenet): MEL predicted as NV and BCC recall, mean over k and seeds. Outputs: tables/<name>.csv.
"""
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

import common as C

RECALL = ["rec_" + c for c in C.CLASSES]
TYPES = ["reversed", "separated", "both_worse"]
KEY = ["init", "target", "k", "seed"]


def t_interval(v):
    v = np.asarray(v, float)
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return v.mean(), h, bool(abs(v.mean()) > h)


def without_class(dd, init):
    d = dd[dd.init == init].copy()
    for c in C.CLASSES:
        d["without_" + c] = d[["d_rec_" + x for x in C.CLASSES if x != c]].mean(axis=1)
    cols = {"all_classes": "d_ba", **{"without_" + c: "without_" + c for c in C.CLASSES}}
    rows = []
    for t in ["all"] + C.TARGETS:
        x = d if t == "all" else d[d.target == t]
        per_seed = x.groupby("seed")[list(cols.values())].mean()
        assert len(per_seed) == 5
        row = dict(target="All" if t == "all" else C.TARGET_EN[t])
        for name, col in cols.items():
            row[f"{name}_mean"], row[f"{name}_half_width"], row[f"{name}_excludes_zero"] = t_interval(per_seed[col])
        rows.append(row)
    return pd.DataFrame(rows)


def rates(df):
    out = df[["init", "target", "k", "method", "seed"]].copy()
    for a in range(5):
        tot = sum(df[f"cm_{a}{j}"] for j in range(5))
        for b in range(5):
            if a != b:
                out[f"t_{a}{b}"] = 100 * df[f"cm_{a}{b}"] / tot
    return out


def classify_pairs(df, init):
    """One row per consistent class-pair change: target, type and images moved."""
    r = rates(df)
    cols = [c for c in r.columns if c.startswith("t_")]
    erm = r[r.method == "erm"].set_index(KEY)
    ssl = r[(r.method != "erm") & (r.init == init)].copy()
    ssl[cols] = ssl[cols].values - erm.loc[ssl.set_index(KEY).index, cols].values
    cm = [f"cm_{a}{b}" for a in range(5) for b in range(5) if a != b]
    ce = df[df.method == "erm"].set_index(KEY)[cm]
    cs = df[(df.method != "erm") & (df.init == init)].copy()
    cs[cm] = cs[cm].values - ce.loc[cs.set_index(KEY).index].values
    count_change = cs.groupby(["method", "target", "k"])[cm].mean()
    rows = []
    for (m, t, k), g in ssl.groupby(["method", "target", "k"]):
        assert len(g) == 5
        for a, b in combinations(range(5), 2):
            x, y = g[f"t_{a}{b}"].values, g[f"t_{b}{a}"].values
            consistent = ((np.sign(x) == np.sign(x.mean())).sum() >= 4 and (np.sign(y) == np.sign(y.mean())).sum() >= 4)
            if not (consistent and abs(x.mean()) >= 1 and abs(y.mean()) >= 1):
                continue
            kind = "reversed" if np.sign(x.mean()) != np.sign(y.mean()) else ("separated" if x.mean() < 0 else "both_worse")
            moved = abs(count_change.loc[(m, t, k), f"cm_{a}{b}"]) + abs(count_change.loc[(m, t, k), f"cm_{b}{a}"])
            rows.append(dict(method=m, target=t, k=k, pair=f"{C.CLASSES_EN[a]}-{C.CLASSES_EN[b]}", type=kind,
                             images_moved=moved))
    return pd.DataFrame(rows)


def pair_table(pairs, weight):
    rows = []
    for t in C.TARGETS + ["all"]:
        x = pairs if t == "all" else pairs[pairs.target == t]
        v = {tp: (x[x.type == tp].images_moved.sum() if weight else (x.type == tp).sum()) for tp in TYPES}
        tot = sum(v.values())
        row = dict(target="All" if t == "all" else C.TARGET_EN[t])
        for tp in TYPES:
            row[tp], row[tp + "_pct"] = v[tp], 100 * v[tp] / tot
        rows.append(row)
    return pd.DataFrame(rows)


def levels(df, col, init):
    d = df[df.init == init]
    per_method = d.groupby(["method", "target"])[col].mean().unstack()[C.TARGETS].loc[["erm"] + C.SSL]
    per_method.index = [C.NAME[m] for m in per_method.index]
    per_method.columns = [C.TARGET_EN[t] for t in C.TARGETS]
    summary = pd.DataFrame(dict(target=[C.TARGET_EN[t] for t in C.TARGETS],
                                erm=[d[(d.method == "erm") & (d.target == t)][col].mean() for t in C.TARGETS],
                                ssl=[d[(d.method != "erm") & (d.target == t)][col].mean() for t in C.TARGETS]))
    return summary, per_method.rename_axis("method").reset_index()


def main():
    df = C.design()
    dd = C.deltas(df, cols=["ba"] + RECALL)

    for init, suffix in (("ham", ""), ("imagenet", "_imagenet")):
        w = without_class(dd, init)
        C.save_table(w, "ssl_without_class" + suffix)
        if init == "ham":
            keep = ["target"] + [c for c in w.columns if c.startswith(("all_classes", "without_akiec"))]
            C.save_table(w[keep], "ssl_without_akiec")
        pairs = classify_pairs(df, init)
        C.save_table(pair_table(pairs, False), "pair_changes" + suffix)
        C.save_table(pair_table(pairs, True), "pair_changes_images" + suffix)
        for col, name in (("mel_nv", "mel_nv"), ("rec_bcc", "bcc_recall")):
            summary, per_method = levels(df, col, init)
            if init == "ham":
                C.save_table(summary, name)
            C.save_table(per_method, name + "_methods" + suffix)

    with pd.option_context("display.width", 250, "display.max_columns", 40):
        for name in ("ssl_without_akiec", "pair_changes", "pair_changes_images", "pair_changes_imagenet",
                     "mel_nv", "bcc_recall"):
            print(name)
            print(pd.read_csv(f"{C.TABLES}/{name}.csv").round(2).to_string(index=False))


if __name__ == "__main__":
    main()
