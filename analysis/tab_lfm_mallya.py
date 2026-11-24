"""LFM and the guidance model of Mallya and Hamarneh on MILK10k clinical photographs, k = 16: tab:e2_lfm_mallya.

Rows: the HAM10000 network alone, the clinical ERM and the four-expert student; columns: without hallucinated features,
with those of the guidance model (fitted on the unlabeled pairs in the HAM10000 feature space) and with LFM.
Also prints the numbers of the text: the published method of Mallya and Hamarneh, and the paired contrasts.
Inputs: results/study2/clinical_views_milk10k.npz, predictions of the clinical ERM and student (clinical_views.py).
Output: tables/e2_lfm_mallya.csv.
"""
import pandas as pd

import clinical_views as V
import common as C

ROWS = [("HAM10000 network only", None), ("Clinical ERM", V.CLINICAL_ERM),
        ("Student of the four-expert combination", V.FOUR_EXPERT_STUDENT)]
COLS = [("without", None), ("guidance_model", "clinical_guidance"), ("lfm", "clinical_lfm")]


def predictions(R):
    P = {}
    for s in V.SEEDS:
        photo = R.calibrated("clinical_photo", s)
        P.setdefault("ham|without", {})[s] = photo[R.test].argmax(1)
        for col, reading in COLS[1:]:
            P.setdefault(f"ham|{col}", {})[s] = R.calibrated(reading, s)[R.test].argmax(1)
        for name, run in ROWS[1:]:
            net = V.ClinicalRun(R, *run, s)
            P.setdefault(f"{name}|without", {})[s] = net.test_pred()
            for col, reading in COLS[1:]:
                P.setdefault(f"{name}|{col}", {})[s] = net.test_pred(R.calibrated(reading, s))
        P.setdefault("mallya", {})[s] = R.z["logits_clinical_mallya"][s][R.test].argmax(1)
    return P


def main():
    R = V.Readings("milk10k")
    P = predictions(R)
    K = V.Contrasts(R.y[R.test], P)
    out = []
    for name, _ in ROWS:
        key = "ham" if name == ROWS[0][0] else name
        out.append(dict(row=name, **{c: K.level(f"{key}|{c}").mean() for c, _ in COLS}))
    C.save_table(pd.DataFrame(out), "e2_lfm_mallya")
    print(pd.DataFrame(out).round(2).to_string(index=False))
    print(f"Mallya and Hamarneh, published code: {K.level('mallya').mean():.2f}")
    pairs = [(f"{k}|lfm", f"{k}|guidance_model") for k in ("ham", ROWS[1][0], ROWS[2][0])]
    pairs += [(f"{k}|lfm", f"{k}|without") for k in ("ham", ROWS[1][0], ROWS[2][0])]
    pairs += [("ham|lfm", "mallya")]
    for metric in ("ba", "mel"):
        for a, b in pairs:
            c = K.contrast(a, b, metric)
            star = "*" if c["star"] else ""
            print(f"  {metric:3s} {a:48s} - {b:48s} {c['diff']:+6.2f}  t [{c['t_low']:+.2f}, {c['t_high']:+.2f}]  "
                  f"boot [{c['boot_low']:+.2f}, {c['boot_high']:+.2f}]  {c['seeds_above']}/5 {star}")


if __name__ == "__main__":
    main()
