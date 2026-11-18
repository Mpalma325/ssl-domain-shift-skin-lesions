"""Test images misclassified by all eleven models sharing a labeled subset, Section 4.2 (Shared Misclassifications).

Per init., target, k and seed on the primary test set: "none correct" (no model right), "same incorrect" (all eleven
agree on a wrong class) and "FS correct" (share of those the fully supervised model gets right); means over 30
conditions. Tables: shared_misclassified, shared_fs, shared_misclassified_conditions, shared_misclassified_images
(per ISIC id), shared_misclassified_pairs (true -> predicted). Input: results/runs.csv, predictions.
"""
import numpy as np
import pandas as pd

import common as C

ELEVEN = ["erm"] + C.SSL


def main():
    R = C.runs()
    cond_rows, image_rows, pair_rows = [], [], []
    for t in C.TARGETS:
        z = C.predictions(t)
        logits = z["logits_test"]
        mask = C.primary_mask(t, z["ids_test"])
        y = z["labels_test"].astype(int)[mask]
        ids = z["ids_test"][mask]
        none_n = np.zeros(len(y), int)
        same_n = np.zeros(len(y), int)
        same_class = np.zeros((len(y), 5), int)
        Rt = R[R.target == t].set_index(["init", "k", "method", "seed"]).row
        for init in C.INITS:
            for k in C.KS:
                for s in C.SEEDS:
                    pred = np.stack([logits[Rt[(init, k, m, s)]][mask].argmax(1) for m in ELEVEN])
                    ok = pred == y[None]
                    none = ~ok.any(0)
                    same = (pred == pred[0][None]).all(0) & ~ok[0]
                    fs_ok = logits[Rt[(init, 0, "fully_supervised", s)]][mask].argmax(1) == y
                    none_n += none
                    same_n += same
                    np.add.at(same_class, (np.where(same)[0], pred[0][same]), 1)
                    for a, b in zip(y[same], pred[0][same]):
                        pair_rows.append((t, a, b))
                    cond_rows.append(dict(target=t, init=init, k=k, seed=s, n_test=len(y),
                                          none_correct=100 * none.mean(), same_incorrect=100 * same.mean(),
                                          fs_correct_none=100 * fs_ok[none].mean(),
                                          fs_correct_same=100 * fs_ok[same].mean()))
        for i in np.where(none_n > 0)[0]:
            image_rows.append(dict(target=t, isic_id=ids[i], true_class=C.CLASSES[y[i]], conditions=30,
                                   none_correct=int(none_n[i]), same_incorrect=int(same_n[i]),
                                   same_incorrect_class=C.CLASSES[same_class[i].argmax()] if same_n[i] else ""))

    cond = pd.DataFrame(cond_rows)
    C.save_table(cond, "shared_misclassified_conditions")
    g = cond.groupby("target", sort=False)
    names = [C.TARGET_EN[t] for t in C.TARGETS]
    C.save_table(pd.DataFrame(dict(target=names, none_correct=g.none_correct.mean().values,
                                   same_incorrect=g.same_incorrect.mean().values)), "shared_misclassified")
    C.save_table(pd.DataFrame(dict(target=names, none_correct=g.fs_correct_none.mean().values,
                                   same_incorrect=g.fs_correct_same.mean().values)), "shared_fs")
    C.save_table(pd.DataFrame(image_rows), "shared_misclassified_images")
    pairs = pd.DataFrame(pair_rows, columns=["target", "true", "pred"])
    pairs = pairs.groupby(["target", "true", "pred"]).size().rename("n").reset_index()
    pairs["share"] = 100 * pairs.n / pairs.groupby("target").n.transform("sum")
    pairs["true"] = [C.CLASSES_EN[i] for i in pairs.true]
    pairs["pred"] = [C.CLASSES_EN[i] for i in pairs.pred]
    C.save_table(pairs.sort_values(["target", "share"], ascending=[True, False]), "shared_misclassified_pairs")

    print(pd.read_csv(f"{C.TABLES}/shared_misclassified.csv").round(1).to_string(index=False))
    print(pd.read_csv(f"{C.TABLES}/shared_fs.csv").round(1).to_string(index=False))
    print(cond.groupby(["target", "init"]).none_correct.mean().round(1).to_string())


if __name__ == "__main__":
    main()
