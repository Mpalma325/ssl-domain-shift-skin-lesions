"""Clinical student of the dermoscopic four-expert combination, MILK10k, k = 16, seeds 0-4, Section 5.2.3 and
appendix: tab:e2_piso_techo and tab:app_e2_escalera.

tab:e2_piso_techo: balanced accuracy of the four-expert combination and of its student, by the images the experts and
the student were trained on (rows) and the test image read (dermoscopic or clinical; columns). The combinations read
the other modality with the calibration learned on their own. The dermoscopic/dermoscopic student on clinical images
is a rerun of the ConvNeXt-T student (same recipe) whose checkpoint was kept.
tab:app_e2_escalera: balanced accuracy and melanoma sensitivity on the clinical photographs by teacher; contrasts =
student of the four-expert combination - model, paired by seed, 95 % t interval and two-level bootstrap.
e2_piso_techo_contrasts: the contrasts quoted in the text.
Inputs: results/study2/student_runs.csv, predictions_students_*.npz, teacher_outputs_*.npz, expert_runs.csv and
predictions_experts_milk10k_clinico.npz (clinical ERM). Outputs: tables/e2_piso_techo.csv,
tables/e2_piso_techo_contrasts.csv, tables/app_e2_escalera.csv.
"""
import numpy as np
import pandas as pd

import common as C
from tab_students import Contrasts, argmax_primary, erm_teacher, lesion_index, students, teacher, test_set

DERM, CLIN = "milk10k", "milk10k_clinico"
SEED_BASE = 20261133


def models():
    """{name: {seed: logits}} on the dermoscopic (DERM) or clinical (CLIN) test image of the same lesions."""
    s4, c4 = "clinical_student_four_dermoscopic_experts", "clinical_student_four_clinical_experts"
    return {
        "comb_derm_on_derm": teacher(DERM, "dermoscopic_experts"),
        "comb_derm_on_clin": teacher(CLIN, "dermoscopic_experts"),
        "comb_clin_on_derm": teacher(DERM, "clinical_experts"),
        "comb_clin_on_clin": teacher(CLIN, "clinical_experts"),
        "student_derm_on_derm": students("four_expert_student", DERM),
        "student_derm_on_clin": students("four_expert_student_rerun", CLIN, trained_on=DERM),
        "student_clin_on_derm": students(c4, DERM, trained_on=CLIN),
        "student_clin_on_clin": students(c4, CLIN),
        "student4_on_derm": students(s4, DERM, trained_on=CLIN),
        "student4": students(s4, CLIN),
        "erm": erm_teacher(CLIN),
        "mean_teacher": students("mean_teacher_rerun", CLIN),
        "erm_imagenet_teacher": students("clinical_student_erm_imagenet_teacher", CLIN),
        "erm_teacher": students("clinical_student_erm_teacher", CLIN),
        "source_teacher": students("clinical_student_source_teacher", CLIN),
        "source_erm_teacher": students("clinical_student_source_erm_teacher", CLIN),
    }


ESCALERA = [("Clinical photograph only", "erm", "ERM"),
            ("Clinical photograph only", "mean_teacher", "Mean Teacher"),
            ("Clinical photograph only", "student_clin_on_clin", "Student of the four experts built on the photograph"),
            ("Students of a teacher that reads the paired dermoscopic image", "erm_imagenet_teacher",
             "ERM initialized from ImageNet"),
            ("Students of a teacher that reads the paired dermoscopic image", "erm_teacher",
             "ERM initialized from HAM10000"),
            ("Students of a teacher that reads the paired dermoscopic image", "source_teacher", "RH"),
            ("Students of a teacher that reads the paired dermoscopic image", "source_erm_teacher", "RH and ERM"),
            ("Students of a teacher that reads the paired dermoscopic image", "student4", "Four-expert combination"),
            ("Upper bound", "comb_derm_on_derm", "Four-expert combination on the real dermoscopic image")]
TEXT = [("comb_derm_on_derm", "comb_clin_on_clin"), ("student4", "student_clin_on_clin"), ("student4", "comb_derm_on_clin"),
        ("student4", "comb_derm_on_derm"), ("student4", "erm"), ("student4", "mean_teacher")]


def main():
    y, ids, mask = test_set(DERM)
    y_c, ids_c, mask_c = test_set(CLIN)
    assert (ids == ids_c).all() and (y == y_c).all() and (mask == mask_c).all()  # same lesions, same order
    P = {m: argmax_primary(d, mask) for m, d in models().items()}
    K = Contrasts(P, y[mask], lesion_index(DERM, ids[mask]), np.random.default_rng([SEED_BASE, 0]),
                  np.random.default_rng([SEED_BASE, 1, 5]))
    ba = lambda m: K.summary(m)["ba"]  # noqa: E731

    grid = [("Dermoscopic / dermoscopic", "comb_derm_on_derm", "comb_derm_on_clin", "student_derm_on_derm",
             "student_derm_on_clin"),
            ("Clinical / clinical", "comb_clin_on_derm", "comb_clin_on_clin", "student_clin_on_derm",
             "student_clin_on_clin"),
            ("Dermoscopic / clinical", "comb_derm_on_derm", "comb_derm_on_clin", "student4_on_derm", "student4")]
    df = pd.DataFrame([dict(trained_on=r, combination_dermoscopic=ba(a), combination_clinical=ba(b),
                            student_dermoscopic=ba(c), student_clinical=ba(d)) for r, a, b, c, d in grid])
    C.save_table(df, "e2_piso_techo")
    print(df.round(2).to_string(index=False))

    rows = [dict(a=a, b=b, metric="ba", **K.contrast(a, b)) for a, b in TEXT]
    df = pd.DataFrame(rows)
    C.save_table(df, "e2_piso_techo_contrasts")
    print(df.round(2).to_string(index=False))

    rows = []
    for group, m, name in ESCALERA:
        s = K.summary(m)
        row = dict(group=group, model=name, ba=s["ba"], mel=s["mel"])
        if m != "student4":
            for metric in ("ba", "mel"):
                c = K.contrast("student4", m, metric)
                row.update({f"{metric}_student_minus_model": c["diff"], f"{metric}_t_low": c["t_low"],
                            f"{metric}_t_high": c["t_high"], f"{metric}_boot_low": c["boot_low"],
                            f"{metric}_boot_high": c["boot_high"], f"{metric}_star": c["star"]})
        rows.append(row)
    df = pd.DataFrame(rows)
    C.save_table(df, "app_e2_escalera")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(df.drop(columns="group").round(2).to_string(index=False))


if __name__ == "__main__":
    main()
