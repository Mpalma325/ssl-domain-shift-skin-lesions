"""Copies of HAM10000 photographs in MILK10k and duplicate pairs within MILK10k (Chapter 3, Appendix 7.2).

Tables tab:milk_copies_location, tab:milk_copies_ham, tab:milk_internal_pairs, tab:milk_copies_labeled and
tab:milk_copies_effect (BA and SSL - ERM change when the 29 excluded lesions stay in the test set).
Labeled subsets are rebuilt as train/utils.py:x_u_split draws them (ImageFolder order, same RandomState stream);
both MILK10k modalities share the lesions, so the counts hold for both.
Inputs: data/milk10k_duplicates.csv, data/splits/*.csv, results/runs.csv, predictions. Outputs: tables/<label>.csv.
"""
import os

import numpy as np
import pandas as pd

import common as C

DATA = os.path.join(C.ROOT, "data")
SPLITS = [("train", "Training"), ("val", "Validation"), ("test", "Test")]
SPLIT_EN = dict(SPLITS)


def x_u_split(num_samples_per_class, num_classes, labels, seed):
    """Labeled indices exactly as train/utils.py:x_u_split (same RandomState stream)."""
    labels = np.array(labels)
    rs = np.random.RandomState(seed)
    out = []
    for i in range(num_classes):
        idx = np.where(labels == i)[0]
        rs.shuffle(idx)
        out.extend(idx[:num_samples_per_class])
    return out


def crosstab(rows, cols, row_order, col_order):
    t = pd.crosstab(rows, cols).reindex(index=row_order, columns=col_order, fill_value=0)
    t["total"] = t.sum(axis=1)
    t.loc["total"] = t.sum()
    return t


def main():
    dup = pd.read_csv(os.path.join(DATA, "milk10k_duplicates.csv"))
    derm = pd.read_csv(os.path.join(DATA, "splits", "milk10k_dermoscopic.csv")).set_index("isic_id")
    clin = pd.read_csv(os.path.join(DATA, "splits", "milk10k_clinical.csv")).set_index("clinical_isic_id")
    ham = pd.read_csv(os.path.join(DATA, "splits", "ham10000.csv")).set_index("isic_id")

    pairs = dup[dup.kind == "milk10k_copy_of_ham10000"].copy()
    pairs["ham_split"] = ham.loc[pairs.isic_id_b, "split"].values
    assert pairs.groupby("isic_id_a").ham_split.nunique().max() == 1  # several matches -> same HAM split
    cop = pairs.groupby("isic_id_a").agg(ham_split=("ham_split", "first"), n_matches=("isic_id_b", "size"))
    cop["split"] = derm.loc[cop.index, "split"].values
    cop["class"] = derm.loc[cop.index, "class"].values
    assert len(cop) == 121
    print(f"  {len(cop)} MILK10k images copy {len(pairs)} HAM10000 photographs "
          f"(matches per image: {cop.n_matches.value_counts().sort_index().to_dict()})")

    t = crosstab(cop.split, cop["class"], [s for s, _ in SPLITS], C.CLASSES)
    t.index = [SPLIT_EN.get(i, "Total") for i in t.index]
    t.columns = C.CLASSES_EN + ["Total"]
    C.save_table(t.rename_axis("partition").reset_index(), "milk_copies_location")

    t = crosstab(cop.split, cop.ham_split, [s for s, _ in SPLITS], [s for s, _ in SPLITS])
    t.index = [SPLIT_EN.get(i, "Total") for i in t.index]
    t.columns = ["ham_" + s for s, _ in SPLITS] + ["Total"]
    C.save_table(t.rename_axis("milk10k_partition").reset_index(), "milk_copies_ham")

    internal = dup[dup.kind == "milk10k_internal"].copy()
    look = {"dermoscopic": derm.split, "clinical": clin.split}
    rank = {s: i for i, (s, _) in enumerate(SPLITS)}
    labels, order = [], []
    for r in internal.itertuples():
        a, b = sorted((look[r.modality][r.isic_id_a], look[r.modality][r.isic_id_b]), key=rank.get)
        labels.append(f"Both in {SPLIT_EN[a].lower()}" if a == b else f"{SPLIT_EN[a]} and {SPLIT_EN[b]}")
        # thesis order: dermoscopic first; mixed splits before same split; then by split
        order.append((r.modality != "dermoscopic", a == b, rank[a], rank[b]))
    internal["partitions"], internal["order"] = labels, order
    g = (internal.groupby(["modality", "partitions"]).agg(pairs=("isic_id_a", "size"), order=("order", "first"))
         .reset_index().sort_values("order").drop(columns="order"))
    g["modality"] = g.modality.str.capitalize()
    C.save_table(g, "milk_internal_pairs")

    test_copies = set(cop.index[cop.split == "test"])
    test_internal = {i for r in internal[internal.modality == "dermoscopic"].itertuples()
                     for i in (r.isic_id_a, r.isic_id_b) if derm.split[i] == "test"}
    assert test_copies | test_internal == C.excluded_ids() and len(test_copies) == 26 and len(test_internal) == 3
    print(f"  primary test set: {len(test_copies)} copies + {len(test_internal)} images of internal pairs excluded")

    train = derm[derm.split == "train"].reset_index()
    train = train.assign(c=train["class"].map(C.CLASSES.index)).sort_values(["c", "isic_id"]).reset_index(drop=True)
    is_copy = train.isic_id.isin(cop.index[cop.split == "train"]).values
    rows = []
    for k in C.KS:
        row = {"k": k, "labeled_images": 5 * k}
        for s in C.SEEDS:
            L = x_u_split(k, 5, train.c.values, s)
            assert len(L) == 5 * k
            row[f"seed_{s}"] = int(is_copy[L].sum())
        rows.append(row)
    C.save_table(pd.DataFrame(rows), "milk_copies_labeled")

    p = C.design()
    m = p[p.target.isin(["milk10k", "milk10k_clinico"])].copy()
    m["diff"] = m.ba_full_test - m.ba
    dd = C.deltas(m, cols=("ba", "ba_full_test"))
    c = dd.groupby(["method", "init", "target", "k"])[["d_ba", "d_ba_full_test"]].mean().reset_index()
    c["change"] = (c.d_ba_full_test - c.d_ba).abs()
    c["sign_change"] = np.sign(c.d_ba) != np.sign(c.d_ba_full_test)
    assert len(c) == 120
    rows = []
    for init in C.INITS:
        for tg in ("milk10k", "milk10k_clinico"):
            x = m[(m.init == init) & (m.target == tg)]["diff"]
            y = c[(c.init == init) & (c.target == tg)]
            rows.append({"target": C.TARGET_EN[tg], "init": C.INIT_EN[init], "runs": len(x),
                         "ba_change_mean": x.mean(), "ba_change_min": x.min(), "ba_change_max": x.max(),
                         "ssl_erm_abs_change_mean": y.change.mean(), "ssl_erm_abs_change_max": y.change.max(),
                         "sign_changes": int(y.sign_change.sum())})
    rows.append({"target": "All", "init": "", "runs": len(m), "ba_change_mean": m["diff"].mean(),
                 "ba_change_min": m["diff"].min(), "ba_change_max": m["diff"].max(),
                 "ssl_erm_abs_change_mean": c.change.mean(), "ssl_erm_abs_change_max": c.change.max(),
                 "sign_changes": int(c.sign_change.sum())})
    C.save_table(pd.DataFrame(rows), "milk_copies_effect")
    print(f"  runs with higher BA on the full test set: {(m['diff'] > 0).sum()} of {len(m)}")


if __name__ == "__main__":
    main()
