"""Dependence of the clinical student on the dermoscopic pairs, MILK10k clinical, k = 16, seeds 0-4, Section 5.2.3:
tab:e2_pares.

Student of RH x ERM (dermoscopic) trained with all pairs, with the pairs of one third of the lesions, or with pairs
shuffled within the class predicted by the teacher; against a student without pairs (teacher RH x ERM on the clinical
photographs). Difference from no pairs paired by seed, 95 % t interval and two-level bootstrap; * when both exclude 0.
Inputs: results/study2/student_runs.csv, predictions_students_milk10k_clinico.npz. Output: tables/e2_pares.csv.
"""
import numpy as np
import pandas as pd

import common as C
from tab_students import Contrasts, argmax_primary, lesion_index, students, test_set

T = "milk10k_clinico"
ROWS = [("pairs_none", "No pairs"), ("pairs_one_third", "One third of the lesions"),
        ("pairs_shuffled", "Pairs shuffled within the predicted class"),
        ("pairs_all", "All lesions with their pair")]
SEED_BASE = 20261142


def main():
    y, ids, mask = test_set(T)
    P = {m: argmax_primary(students(m, T), mask) for m, _ in ROWS}
    K = Contrasts(P, y[mask], lesion_index(T, ids[mask]), np.random.default_rng([SEED_BASE, 0]),
                  np.random.default_rng([SEED_BASE, 1, 5]))
    rows = []
    for m, name in ROWS:
        s = K.summary(m)
        row = dict(pairs=name, ba=s["ba"], ba_sd=s["ba_sd"])
        if m != "pairs_none":
            c = K.contrast(m, "pairs_none")
            row.update(diff_from_no_pairs=c["diff"], t_low=c["t_low"], t_high=c["t_high"], boot_low=c["boot_low"],
                       boot_high=c["boot_high"], star=c["star"])
        rows.append(row)
    df = pd.DataFrame(rows)
    C.save_table(df, "e2_pares")
    print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
