"""Shared pieces of the clinical-photograph analyses of Study 2 (Section 5.2.4 and Appendix 7.4).

A "reading" is a set of balanced log-posteriors (HAM10000 prior removed, not yet calibrated) of one image, from a
HAM10000 network reading the photograph, its LFM translation, a variant of LFM, or the dermoscopic image. Each reading
is calibrated per seed with BCTS on the 80 labeled images of that seed, and can be added to a clinical network (ERM or
student) taken at its validation temperature.
Inputs: results/study2/clinical_views_<dataset>.npz, results/study2/predictions_{experts,students}_milk10k_clinico.npz
with expert_runs.csv and student_runs.csv, results/study2/midas_metrics.csv and midas_metrics_bootstrap.csv.
"""
import os

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize, minimize_scalar

import common as C

STUDY2 = os.path.join(C.RESULTS, "study2")
SEEDS = [0, 1, 2, 3, 4]
MEL = 3
BOOT_SEED = 20261122
# clinical networks of MILK10k (k = 16, HAM10000 init.): (file kind, model) in predictions_<kind>_milk10k_clinico.npz
CLINICAL_ERM = ("experts", "resnet_erm")
FOUR_EXPERT_STUDENT = ("students", "clinical_student_four_dermoscopic_experts")
RUNS = {"experts": "expert_runs.csv", "students": "student_runs.csv"}


def log_softmax(z):
    z = np.asarray(z, np.float64)
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def bcts(z, y):
    """Temperature and per-class bias minimising the NLL (small L2 on the bias)."""
    def nll(p):
        q = np.exp(log_softmax(z / np.exp(p[0]) + p[1:]))
        return -np.log(np.clip(q[np.arange(len(y)), y], 1e-12, None)).mean() + 1e-3 * (p[1:] ** 2).sum()
    r = minimize(nll, np.zeros(6), method="L-BFGS-B")
    return float(np.exp(r.x[0])), r.x[1:]


def temperature(z, y):
    """Temperature minimising the class-balanced NLL."""
    w = 1.0 / np.bincount(y, minlength=5).astype(float)[y]

    def f(lt):
        return -(w * log_softmax(z / np.exp(lt))[np.arange(len(y)), y]).sum() / w.sum()
    return float(np.exp(minimize_scalar(f, bounds=(np.log(1e-3), np.log(100.0)), method="bounded").x))


def val_subsample(yv, seed, n=16):
    """The 16 validation images per class of a seed used for calibration (same draw as the trainer)."""
    rng = np.random.RandomState(10_000 + seed)
    out = []
    for c in range(5):
        out.extend(rng.choice(np.flatnonzero(yv == c), n, replace=False).tolist())
    return np.array(sorted(out), dtype=int)


class Readings:
    """One clinical_views_<dataset>.npz: readings of every image, labels, splits and the labeled sets per seed."""

    def __init__(self, dataset="milk10k"):
        path = os.path.join(STUDY2, f"clinical_views_{dataset}.npz")
        with np.load(path) as z:
            self.z = {k: z[k] for k in z.files}
        self.ids = self.z["ids"].astype(str)
        self.y = self.z["labels"].astype(int)
        self.split = self.z["split"].astype(str)
        self.pos = {k: i for i, k in enumerate(self.ids)}
        test = np.flatnonzero(self.split == "test")
        self.test = test[C.primary_mask("milk10k" if dataset == "milk10k" else "bcn20000", self.ids[test])]

    def lp(self, name, seed):
        a = self.z["lp_" + name].astype(np.float64)
        return a[seed] if a.ndim == 3 else a

    def calibrated(self, name, seed):
        """Reading calibrated with BCTS on the 80 labeled images of the seed (all images)."""
        lp = self.lp(name, seed)
        L = self.z["labeled"][seed]
        T, b = bcts(lp[L], self.y[L])
        return log_softmax(lp / T + b)

    def rows(self, ids):
        return np.array([self.pos[k] for k in ids])


class ClinicalRun:
    """Validation and test logits of a clinical network (one seed) at its temperature on the 16 per class."""

    def __init__(self, R, kind, model, seed):
        runs = pd.read_csv(os.path.join(STUDY2, RUNS[kind]))
        r = runs.query("target == 'milk10k_clinico' and k == 16 and model == @model and seed == @seed")
        assert len(r) == 1, (kind, model, seed, len(r))
        with np.load(os.path.join(STUDY2, f"predictions_{kind}_milk10k_clinico.npz")) as z:
            zv, zt = z["logits_val"][r.row.item()].astype(np.float64), z["logits_test"][r.row.item()].astype(np.float64)
            yv, yt = z["labels_val"].astype(int), z["labels_test"].astype(int)
            iv, it = z["ids_val"].astype(str), z["ids_test"].astype(str)
        self.rv, self.rt = R.rows(iv), R.rows(it)
        assert (R.y[self.rv] == yv).all() and (R.y[self.rt] == yt).all()
        sub = val_subsample(yv, seed)
        T = temperature(zv[sub], yv[sub])
        self.lv, self.lt = log_softmax(zv / T), log_softmax(zt / T)
        dev = np.ones(len(yv), bool)
        dev[sub] = False
        self.dev = dev & ~R.z["val_ham_copy"][self.rv]
        self.R = R

    def test_pred(self, reading=None, w=1.0):
        """Predicted class on the primary test images (in the order of Readings.test)."""
        z = self.lt if reading is None else self.lt + w * reading[self.rt]
        p = np.full(len(self.R.ids), -1)
        p[self.rt] = z.argmax(1)
        return p[self.R.test]

    def dev_ba(self, reading, w):
        z = self.lv + w * reading[self.rv]
        y = self.R.y[self.rv][self.dev]
        return balanced_accuracy(y, z[self.dev].argmax(1))


def balanced_accuracy(y, pred):
    return 100.0 * np.mean([(pred[y == c] == c).mean() for c in range(5)])


def recalls(y, pred):
    return 100.0 * np.array([(pred[y == c] == c).mean() for c in range(5)])


def t_interval(d):
    d = np.asarray(d, float)
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), d.mean() - h, d.mean() + h


class Contrasts:
    """Paired contrasts over seeds: 95 % t interval and two-level bootstrap (images, then seeds); '*' when both
    intervals exclude zero. preds: {model: {seed: predicted class on the test images}}."""

    def __init__(self, y, preds, B=10_000, seed=BOOT_SEED):
        self.y, self.P = y, preds
        keys = [(m, s) for m in preds for s in sorted(preds[m])]
        self.col = {k: j for j, k in enumerate(keys)}
        ok = np.stack([(preds[m][s] == y).astype(float) for m, s in keys], 1)
        self.boot = self._recall_boot(ok, y, B, np.random.default_rng([seed, 0]))
        self.B, self.seed = B, seed

    @staticmethod
    def _recall_boot(ok, y, B, rng):
        """B x models x 5 recalls under multinomial image weights."""
        n = len(y)
        Y1 = np.eye(5)[y]
        X = (Y1[:, :, None] * ok[:, None, :]).reshape(n, -1)
        out = np.empty((B, ok.shape[1], 5))
        for b0 in range(0, B, 250):
            nb = min(250, B - b0)
            W = rng.multinomial(n, np.full(n, 1.0 / n), size=nb).astype(float)
            out[b0:b0 + nb] = 100 * np.transpose((W @ X).reshape(nb, 5, -1) / (W @ Y1)[:, :, None], (0, 2, 1))
        return out

    def level(self, m, metric="ba"):
        f = (lambda r: r.mean()) if metric == "ba" else (lambda r: r[MEL])
        return np.array([f(recalls(self.y, self.P[m][s])) for s in sorted(self.P[m])])

    def contrast(self, a, b, metric="ba"):
        seeds = sorted(set(self.P[a]) & set(self.P[b]))
        n = len(seeds)
        d = self.level(a, metric) - self.level(b, metric)
        mean, lo, hi = t_interval(d)
        Bm = self.boot.mean(2) if metric == "ba" else self.boot[:, :, MEL]
        D = np.stack([Bm[:, self.col[(a, s)]] - Bm[:, self.col[(b, s)]] for s in seeds], 1)
        idx = np.random.default_rng([self.seed, 1, n]).integers(0, n, size=(self.B, n))
        blo, bhi = np.percentile(np.take_along_axis(D, idx, 1).mean(1), [2.5, 97.5])
        return dict(model=a, reference=b, metric=metric, diff=mean, t_low=lo, t_high=hi, boot_low=blo, boot_high=bhi,
                    seeds_above=int((d > 0).sum()), star=bool((lo > 0 and blo > 0) or (hi < 0 and bhi < 0)))


def midas():
    """MIDAS aggregates: balanced accuracy per seed, and the bootstrap bounds of the contrasts (computed with the
    images, which are not released)."""
    m = pd.read_csv(os.path.join(STUDY2, "midas_metrics.csv"))
    b = pd.read_csv(os.path.join(STUDY2, "midas_metrics_bootstrap.csv"))
    return m, b


def midas_contrast(m, b, analysis, model, reference):
    x = m[m.analysis == analysis].pivot_table(index="seed", columns="model", values="ba")
    mean, lo, hi = t_interval(x[model] - x[reference])
    r = b[(b.analysis == analysis) & (b.model == model) & (b.reference == reference)]
    assert len(r) == 1, (analysis, model, reference)
    blo, bhi = r.boot_low.item(), r.boot_high.item()
    return dict(model=model, reference=reference, diff=mean, t_low=lo, t_high=hi, boot_low=blo, boot_high=bhi,
                star=bool((lo > 0 and blo > 0) or (hi < 0 and bhi < 0)))


def cell(level, contrast=None):
    """'50.8 (+0.9*)' style: thesis rounding."""
    if contrast is None:
        return C.r1(level)
    return f"{C.r1(level)} ({C.signed(contrast['diff'])}{'*' if contrast['star'] else ''})"
