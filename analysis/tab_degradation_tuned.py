"""The degradation-tuned HAM10000 network, k = 16: tab:e2_robusta_clinica, tab:e2_robusta_derm, tab:app_e2_robusta_v1.

Each network reads the image through its own head (HAM10000 prior removed) and is calibrated with BCTS on the 80
labeled images of the seed; with LFM, the clinical features are mapped into its own feature space. In parentheses:
degradation-tuned minus HAM10000 network, paired by seed ('*' = the t interval and the two-level bootstrap both exclude
zero). MIDAS columns come from the released aggregates. Also prints the clean control of the text (ten seeds).
Inputs: results/study2/clinical_views_{milk10k,bcn20000}.npz, midas_metrics*.csv.
Outputs: tables/e2_robusta_clinica.csv, tables/e2_robusta_derm.csv, tables/app_e2_robusta_v1.csv.
"""
import pandas as pd

import clinical_views as V
import common as C


def local(R, pairs, seeds=V.SEEDS):
    """{column: (level of reference, level of network, contrast)} for readings of one dataset."""
    P = {}
    for s in seeds:
        for a, b in pairs.values():
            for k in (a, b):
                P.setdefault(k, {})[s] = R.calibrated(k, s)[R.test].argmax(1)
    K = V.Contrasts(R.y[R.test], P)
    return {col: (K.level(b).mean(), K.level(a).mean(), K.contrast(a, b)) for col, (a, b) in pairs.items()}


def midas(analysis, pairs):
    m, b = V.midas()
    lev = m[m.analysis == analysis].groupby("model").ba.mean()
    return {col: (lev[r], lev[a], V.midas_contrast(m, b, analysis, a, r)) for col, (a, r) in pairs.items()}


def table(cells, rows, name):
    """cells: {column: (reference level, network level, contrast)}; rows: (reference label, network label)."""
    out = [dict(network=rows[0], **{c: v[0] for c, v in cells.items()}),
           dict(network=rows[1], **{c: v[1] for c, v in cells.items()},
                **{c + "_diff": v[2]["diff"] for c, v in cells.items()},
                **{c + "_star": v[2]["star"] for c, v in cells.items()})]
    t = pd.DataFrame(out)
    C.save_table(t, name)
    show = pd.DataFrame([{"": rows[0], **{c: V.cell(v[0]) for c, v in cells.items()}},
                         {"": rows[1], **{c: V.cell(v[1], v[2]) for c, v in cells.items()}}])
    print(show.to_string(index=False))


def main():
    milk, bcn = V.Readings("milk10k"), V.Readings("bcn20000")
    ham, tuned = "HAM10000 network", "Degradation-tuned network"

    c = local(milk, {"milk10k_without_lfm": ("clinical_robust_photo", "clinical_photo"),
                     "milk10k_with_lfm": ("clinical_robust_lfm", "clinical_lfm")})
    c.update(midas("robust", {"midas_without_lfm": ("robust_photo", "ham_photo"),
                              "midas_with_lfm": ("robust_lfm", "ham_lfm")}))
    table({k: c[k] for k in ("milk10k_without_lfm", "midas_without_lfm", "milk10k_with_lfm", "midas_with_lfm")},
          (ham, tuned), "e2_robusta_clinica")

    d = local(milk, {"milk10k": ("derm_robust_v2", "derm_ham")})
    d.update(local(bcn, {"bcn20000": ("derm_robust_v2", "derm_ham")}))
    d.update(midas("derm", {"midas": ("robust_v2_derm", "ham_derm")}))
    table(d, (ham, tuned), "e2_robusta_derm")

    v1 = local(milk, {"milk10k": ("derm_robust_v1", "derm_ham")})
    v1.update(midas("derm", {"midas": ("robust_v1_derm", "ham_derm")}))
    out = [dict(network=ham, milk10k=d["milk10k"][0], midas=d["midas"][0])]
    for label, src in (("First simulator", v1), ("Final simulator", d)):
        out.append(dict(network=label, **{c: src[c][1] for c in ("milk10k", "midas")},
                        **{c + "_diff": src[c][2]["diff"] for c in ("milk10k", "midas")},
                        **{c + "_star": src[c][2]["star"] for c in ("milk10k", "midas")}))
    C.save_table(pd.DataFrame(out), "app_e2_robusta_v1")
    print(pd.DataFrame(out).round(2).to_string(index=False))

    ctrl = local(milk, {"control": ("derm_robust_v2", "derm_control")}, seeds=range(10))["control"]
    print(f"clean control, MILK10k dermoscopic, ten seeds: {ctrl[0]:.2f}; degradation-tuned - control "
          f"{ctrl[2]['diff']:+.2f} t [{ctrl[2]['t_low']:+.2f}, {ctrl[2]['t_high']:+.2f}] "
          f"boot [{ctrl[2]['boot_low']:+.2f}, {ctrl[2]['boot_high']:+.2f}] {'*' if ctrl[2]['star'] else ''}")


if __name__ == "__main__":
    main()
