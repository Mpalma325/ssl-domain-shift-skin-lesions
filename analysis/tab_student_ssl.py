"""SSL after the first student, Section 5.2.2: tab:e2_ssl_alumno, k = 16, seeds 0-4.

Balanced accuracy of the first student and of its training continued with ERM or with SSL (Mean Teacher on MILK10k
dermoscopic, FixMatch on BCN20000). Also SSL - ERM paired by seed with its 95 % t interval (quoted in the text).
Inputs: results/study2/student_runs.csv, predictions_students_<target>.npz. Output: tables/e2_ssl_alumno.csv.
"""
import numpy as np
import pandas as pd

import common as C
from tab_students import students, t_interval, test_set

SSL = {"milk10k": "first_student_then_mean_teacher", "bcn20000": "first_student_then_fixmatch"}
ROWS = [("First student", "first_student"), ("Continued with ERM", "first_student_then_erm"),
        ("Continued with SSL", None)]


def per_seed(model, t):
    y, _, mask = test_set(t)
    return np.array([C.metrics(z[mask], y[mask])["ba"] for _, z in sorted(students(model, t).items())])


def main():
    ba = {(t, name): per_seed(m or SSL[t], t) for t in SSL for name, m in ROWS}
    rows = [dict(row=name, **{f"{t}_ba": ba[(t, name)].mean() for t in SSL}) for name, _ in ROWS]
    row = dict(row="SSL - ERM (text)")
    for t in SSL:
        m, lo, hi = t_interval(ba[(t, "Continued with SSL")] - ba[(t, "Continued with ERM")])
        row.update({f"{t}_ba": m, f"{t}_t_low": lo, f"{t}_t_high": hi})
    rows.append(row)
    df = pd.DataFrame(rows)
    C.save_table(df, "e2_ssl_alumno")
    print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
