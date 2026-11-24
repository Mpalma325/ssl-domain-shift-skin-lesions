"""Students distilled from the combinations, Section 5.2.2: tab:e2_alumnos (balanced accuracy) and
tab:app_e2_mel_alumnos (melanoma sensitivity), k = 16, HAM10000 init., seeds 0-4.

Rows: ERM (RE, the ERM ResNet-50 teacher), Noisy Student of Study 1, the four-expert combination, the first student
(ResNet-50, teacher RE x RH), the student of the four-expert combination (ConvNeXt-T) and, for melanoma only, the same
student with the teacher averaged by lesion (BCN20000). Contrasts: combination - model, paired by seed, 95 % t interval
and two-level bootstrap (lesions, then seeds); * when both exclude zero.
Inputs: results/study2/student_runs.csv, predictions_students_<target>.npz, teacher_outputs_<target>.npz,
expert_runs.csv and predictions_experts_<target>.npz (RE), common.panel() (Noisy Student).
Outputs: tables/e2_alumnos.csv, tables/app_e2_mel_alumnos.csv, tables/e2_alumnos_contrasts.csv (BA contrasts of
the lesion-averaged student quoted in the text). The helpers are shared by the other Study 2 student
scripts.
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

import common as C

STUDY2 = os.path.join(C.RESULTS, "study2")
EXPERTS = STUDY2
MEL = 3
B = 10_000
ERM_TEACHER = "resnet_erm"  # model name of the ERM ResNet-50 teachers in expert_runs.csv


def student_runs():
    return pd.read_csv(os.path.join(STUDY2, "student_runs.csv"))


def _npz(name):
    path = os.path.join(STUDY2, name)
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found")
    return np.load(path)


def test_set(target):
    """Labels, ISIC ids and primary-test mask of a target (order of the prediction files)."""
    z = _npz(f"predictions_students_{target}.npz")
    ids = z["ids_test"].astype(str)
    return z["labels_test"].astype(int), ids, C.primary_mask(target, ids)


def students(model, target, trained_on=None):
    """{seed: test logits} of a student evaluated on the test images of `target`."""
    r = student_runs()
    r = r[(r.model == model) & (r.target == target) & (r.trained_on == (trained_on or target))]
    z = _npz(f"predictions_students_{target}.npz")["logits_test"]
    out = {int(x.seed): z[x.row] for x in r.itertuples()}
    assert sorted(out) == C.SEEDS, (model, target, sorted(out))
    return out


def teacher(target, experts="dermoscopic_experts"):
    """{seed: test log-probabilities} of a four-expert combination."""
    z = _npz(f"teacher_outputs_{target}.npz")
    return {int(s): z[experts][j] for j, s in enumerate(z["seeds"])}


def erm_teacher(target):
    """{seed: test logits} of the ERM ResNet-50 teacher (RE), seeds 0-4."""
    r = pd.read_csv(os.path.join(EXPERTS, "expert_runs.csv"))
    r = r[(r.model == ERM_TEACHER) & (r.target == target) & (r.k == 16) & r.seed.isin(C.SEEDS)]
    z = np.load(os.path.join(EXPERTS, f"predictions_experts_{target}.npz"))
    ids = z["ids_test"].astype(str)
    _, ref, _ = test_set(target)
    assert (ids == ref).all()
    out = {int(x.seed): z["logits_test"][x.row] for x in r.itertuples()}
    assert sorted(out) == C.SEEDS, (target, sorted(out))
    return out


def study1(method, target):
    """{seed: test logits} of a Study 1 run (HAM10000 init., k = 16)."""
    R = C.runs()
    r = R[(R.method == method) & (R.init == "ham") & (R.k == 16) & (R.target == target)]
    z = C.predictions(target)
    _, ref, _ = test_set(target)
    assert (z["ids_test"].astype(str) == ref).all()
    return {int(x.seed): z["logits_test"][x.row] for x in r.itertuples()}


def lesion_index(target, ids):
    """Lesion of each test image as an integer (BCN20000: lesion_id; MILK10k: each image is its own lesion)."""
    if target == "bcn20000":
        s = pd.read_csv(os.path.join(C.ROOT, "data", "splits", "bcn20000.csv")).set_index("isic_id").lesion_id
        labels = s.loc[ids].to_numpy()
    else:
        labels = np.asarray(ids)
    return np.unique(labels, return_inverse=True)[1]


def recalls(y, pred):
    return np.array([100.0 * np.mean(pred[y == k] == k) for k in range(5)])


def boot_recalls(correct, y, lesion, n_boot, rng):
    """n_boot x columns x 5 recalls with multinomial weights over lesions; correct is images x columns (0/1)."""
    G = int(lesion.max()) + 1
    Y1 = np.eye(5)[y]
    X = (Y1[:, :, None] * correct[:, None, :]).reshape(len(y), -1)
    out = np.empty((n_boot, correct.shape[1], 5))
    for b0 in range(0, n_boot, 250):
        nb = min(250, n_boot - b0)
        W = rng.multinomial(G, np.full(G, 1.0 / G), size=nb).astype(float)[:, lesion]
        num = (W @ X).reshape(nb, 5, -1)
        out[b0:b0 + nb] = 100 * np.transpose(num / (W @ Y1)[:, :, None], (0, 2, 1))
    return out


def t_interval(d):
    d = np.asarray(d, float)
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), d.mean() - h, d.mean() + h


class Contrasts:
    """Per-seed recalls of several models on one test set and paired contrasts with the two-level bootstrap.

    preds: {model: {seed: predicted class}} on the primary test images. The bootstrap draws lesion weights once
    (seed [seed, 0]... given by rng_boot) and one resampling of the five seeds (rng_seeds)."""

    def __init__(self, preds, y, lesion, rng_boot, rng_seeds, n_boot=B):
        self.preds, self.y = preds, y
        self.rec = {m: {s: recalls(y, p) for s, p in d.items()} for m, d in preds.items()}
        keys = [(m, s) for m in preds for s in sorted(preds[m])]
        self.col = {k: j for j, k in enumerate(keys)}
        correct = np.stack([(preds[m][s] == y).astype(float) for m, s in keys], 1)
        self.boot = boot_recalls(correct, y, lesion, n_boot, rng_boot)
        self.seed_draw = rng_seeds.integers(0, 5, size=(n_boot, 5))

    def summary(self, m):
        ba = np.array([self.rec[m][s].mean() for s in C.SEEDS])
        mel = np.array([self.rec[m][s][MEL] for s in C.SEEDS])
        return dict(ba=ba.mean(), ba_sd=ba.std(ddof=1), mel=mel.mean(), mel_sd=mel.std(ddof=1))

    def contrast(self, a, b, metric="ba"):
        f = (lambda r: r.mean()) if metric == "ba" else (lambda r: r[MEL])
        d = np.array([f(self.rec[a][s]) - f(self.rec[b][s]) for s in C.SEEDS])
        m, lo, hi = t_interval(d)
        Bm = self.boot.mean(2) if metric == "ba" else self.boot[:, :, MEL]
        Db = np.stack([Bm[:, self.col[(a, s)]] - Bm[:, self.col[(b, s)]] for s in C.SEEDS], 1)
        blo, bhi = np.percentile(np.take_along_axis(Db, self.seed_draw, 1).mean(1), [2.5, 97.5])
        star = bool((lo > 0 and blo > 0) or (hi < 0 and bhi < 0))
        return dict(diff=m, t_low=lo, t_high=hi, boot_low=blo, boot_high=bhi, star=star)


def argmax_primary(d, mask):
    return {s: np.asarray(v)[mask].argmax(1) for s, v in d.items()}


ROWS = [("Reference conditions", "erm", "ERM (RE)"),
        ("Reference conditions", "noisy_student", "Noisy Student (Study 1)"),
        ("Reference conditions", "combination", "Four-expert combination"),
        ("Distilled students", "first_student", "First student, of RE and RH (ResNet-50)"),
        ("Distilled students", "four_expert_student", "Student of the four-expert combination (ConvNeXt-T)"),
        ("Distilled students", "four_expert_student_lesion_mean", "Same student, by lesion (BCN20000 only)")]
SEED_BASE = 20261131


def main():
    out = {"ba": [], "mel": []}
    cells = {}
    for it, t in enumerate(["milk10k", "bcn20000"]):
        y, ids, mask = test_set(t)
        P = {"erm": erm_teacher(t), "noisy_student": study1("noisy_student", t), "combination": teacher(t),
             "first_student": students("first_student", t), "four_expert_student": students("four_expert_student", t)}
        if t == "bcn20000":
            P["four_expert_student_lesion_mean"] = students("four_expert_student_lesion_mean", t)
        P = {m: argmax_primary(d, mask) for m, d in P.items()}
        K = Contrasts(P, y[mask], lesion_index(t, ids[mask]), np.random.default_rng([SEED_BASE, it]),
                      np.random.default_rng([SEED_BASE, it, 5]))
        for m in P:
            s = K.summary(m)
            cells[(t, m)] = s
            for metric in ("ba", "mel"):
                c = K.contrast("combination", m, metric) if m != "combination" else {}
                cells[(t, m, metric)] = c
        if t == "bcn20000":
            text = [dict(target=t, a=a, b=b, **K.contrast(a, b)) for a, b in
                    (("four_expert_student_lesion_mean", "four_expert_student"),
                     ("four_expert_student_lesion_mean", "combination"))]
    for metric, label in (("ba", "e2_alumnos"), ("mel", "app_e2_mel_alumnos")):
        rows = []
        for group, m, name in ROWS:
            if metric == "ba" and m == "four_expert_student_lesion_mean":
                continue
            row = dict(group=group, model=name)
            for t in ["milk10k", "bcn20000"]:
                if (t, m) not in cells:
                    continue
                row[f"{t}_{metric}"] = cells[(t, m)][metric]
                c = cells[(t, m, metric)]
                if c:
                    row[f"{t}_combination_minus_model"] = c["diff"]
                    row[f"{t}_t_low"], row[f"{t}_t_high"] = c["t_low"], c["t_high"]
                    row[f"{t}_boot_low"], row[f"{t}_boot_high"] = c["boot_low"], c["boot_high"]
                    row[f"{t}_star"] = c["star"]
            rows.append(row)
        df = pd.DataFrame(rows)
        C.save_table(df, label)
        with pd.option_context("display.width", 250, "display.max_columns", 30):
            print(df.round(2).to_string(index=False))
    df = pd.DataFrame(text)
    C.save_table(df, "e2_alumnos_contrasts")
    print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
