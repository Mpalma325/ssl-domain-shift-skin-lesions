"""Variants of LFM at k = 16, alone and combined with the clinical ERM: tab:app_e2_variantes_lfm (summarised version).

Each variant is compared with LFM on the same test images and seeds (paired by seed; '*' = the t interval and the
two-level bootstrap both exclude zero). MILK10k: clinical test without the 29 HAM10000 copies. MIDAS rows come from the
released aggregates. "Weight chosen on the validation partition": per seed, the weight of LFM in the sum with the ERM
that maximises balanced accuracy on validation without the 80 calibration images and the HAM10000 copies (ties: the
weight closest to 1).
Inputs: results/study2/clinical_views_milk10k.npz, midas_metrics*.csv, predictions of the clinical ERM.
Output: tables/app_e2_variantes_lfm.csv.
"""
import numpy as np
import pandas as pd

import clinical_views as V
import common as C

MILK = [("Input views", "Clinical photograph without zoom", "no_zoom"),
        ("Input views", "Photograph and zoom x2", "zoom2"),
        ("Input views", "Photograph and zoom x3", "zoom3"),
        ("Form of the mapping", "Mapped features rescaled to the dermoscopic spread", "rescaled"),
        ("Form of the mapping", "Mapping to the five log-posteriors", "to_logposteriors"),
        ("Form of the mapping", "Linear distillation of the head", "distilled"),
        ("Form of the mapping", "ResNet-50 trained to produce the features", "resnet_translator"),
        ("Form of the mapping", "Mean features of the 20 nearest pairs", "neighbours20"),
        ("Pairs", "50% of the pairs where the HAM10000 network is most confident", "confident50"),
        ("Pairs", "1,000 random pairs", "random1000"),
        ("Other HAM10000 networks", "ConvNeXt-T instead of ResNet-50", "convnext"),
        ("Other HAM10000 networks", "LFM and ConvNeXt-T, added", "lfm+convnext")]
MIDAS = [("Input views", "Zoom with ink and hair removed", "lfm_clean_zoom"),
         ("Other HAM10000 networks", "Robust network, first camera simulator", "robust_v1_lfm")]
ORDER = ["", "Input views", "Form of the mapping", "Pairs", "Other HAM10000 networks", "Weight of LFM in the sum"]
WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]


def reading(R, key, s):
    return V.log_softmax(sum(R.calibrated("clinical_" + k, s) for k in key.split("+")))


def milk(R):
    P = {}
    for s in V.SEEDS:
        erm = V.ClinicalRun(R, *V.CLINICAL_ERM, s)
        for key in ["lfm"] + [k for *_, k in MILK]:
            cal = reading(R, key, s)
            P.setdefault(key + "|alone", {})[s] = cal[R.test].argmax(1)
            P.setdefault(key + "|erm", {})[s] = erm.test_pred(cal)
        lfm = R.calibrated("clinical_lfm", s)
        w = max(WEIGHTS, key=lambda w: (erm.dev_ba(lfm, w), -abs(w - 1)))
        P.setdefault("weight_dev|erm", {})[s] = erm.test_pred(lfm, w)
    return P


def main():
    R = V.Readings("milk10k")
    K = V.Contrasts(R.y[R.test], milk(R))
    rows = [dict(group="", variant="LFM", evaluated_on="MILK10k", alone=K.level("lfm|alone").mean(),
                 clinical_erm=K.level("lfm|erm").mean())]
    for g, name, key in MILK:
        a, e = K.contrast(key + "|alone", "lfm|alone"), K.contrast(key + "|erm", "lfm|erm")
        rows.append(dict(group=g, variant=name, evaluated_on="MILK10k", alone=K.level(key + "|alone").mean(),
                         alone_diff=a["diff"], alone_star=a["star"], clinical_erm=K.level(key + "|erm").mean(),
                         erm_diff=e["diff"], erm_star=e["star"]))
    e = K.contrast("weight_dev|erm", "lfm|erm")
    rows.append(dict(group="Weight of LFM in the sum", variant="Weight chosen on the validation partition",
                     evaluated_on="MILK10k", clinical_erm=K.level("weight_dev|erm").mean(), erm_diff=e["diff"],
                     erm_star=e["star"]))
    m, b = V.midas()
    lev = m[m.analysis == "variants"].groupby("model").ba.mean()
    rows.append(dict(group="", variant="LFM", evaluated_on="MIDAS", alone=lev["ham_lfm"], clinical_erm=lev["erm_lfm"]))
    for g, name, key in MIDAS:
        a = V.midas_contrast(m, b, "variants", "ham_" + key if key.startswith("lfm") else key, "ham_lfm")
        e = V.midas_contrast(m, b, "variants", "erm_" + key, "erm_lfm")
        rows.append(dict(group=g, variant=name, evaluated_on="MIDAS", alone=lev[a["model"]], alone_diff=a["diff"],
                         alone_star=a["star"], clinical_erm=lev[e["model"]], erm_diff=e["diff"], erm_star=e["star"]))
    t = pd.DataFrame(rows)
    t["order"] = t.group.map(ORDER.index)
    t = t.sort_values("order", kind="stable").drop(columns="order")
    t = t[["group", "variant", "evaluated_on", "alone", "alone_diff", "alone_star", "clinical_erm", "erm_diff",
           "erm_star"]]
    C.save_table(t, "app_e2_variantes_lfm")
    show = t.copy()
    for col, d, st in (("alone", "alone_diff", "alone_star"), ("clinical_erm", "erm_diff", "erm_star")):
        show[col] = [("-" if np.isnan(v) else C.r1(v)) if np.isnan(dd) else V.cell(v, dict(diff=dd, star=bool(s)))
                     for v, dd, s in zip(t[col], t[d], t[st].fillna(False))]
    with pd.option_context("display.width", 200, "display.max_colwidth", 70):
        print(show[["group", "variant", "evaluated_on", "alone", "clinical_erm"]].to_string(index=False))


if __name__ == "__main__":
    main()
