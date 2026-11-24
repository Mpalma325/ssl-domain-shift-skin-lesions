"""Experts of Study 2 and their combination (Chapter 5, Section "Reusing the Source Networks").

RH and CH: the ResNet-50 and ConvNeXt-T trained on HAM10000 (results/study2/source_models.npz), with the HAM10000
train prior removed and a temperature and one bias per class (BCTS) fitted on the labeled target images of the seed.
RE and CE: ResNet-50 and ConvNeXt-T fine-tuned with ERM on those images (results/study2/predictions_experts_<target>.npz,
or the Study 1 panel for RE at k = 32 and 100), as log_softmax(z / 4). Combination: argmax of sum_e w_e log p_e,
w = 1 for RH and CH and 0.5 for RE and CE. Test = primary test set; dev = validation without the 16 per class used
for fitting (v16) and without the copies of HAM10000 images.
"""
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import common as C

STUDY2 = os.path.join(C.RESULTS, "study2")
EXPERTS = ("RE", "RH", "CH", "CE")
WEIGHTS = np.array([0.5, 1.0, 1.0, 0.5])
T_ERM = 4.0
TARGETS2 = ("milk10k", "bcn20000")
NAME2 = {"milk10k": "MILK10k", "bcn20000": "BCN20000"}


def log_softmax(z):
    z = np.asarray(z, np.float64)
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def ba(y, pred):
    return 100.0 * float(np.mean([(pred[y == c] == c).mean() for c in range(5) if (y == c).any()]))


def recalls(y, pred):
    return np.array([100.0 * (pred[y == c] == c).mean() for c in range(5)])


def ham_log_prior():
    s = pd.read_csv(os.path.join(C.ROOT, "data", "splits", "ham10000.csv"))
    n = s[s.split == "train"]["class"].value_counts().reindex(C.CLASSES).to_numpy(float)
    return np.log(n / n.sum())


def bcts(z, y):
    """Temperature and per-class bias on the labeled images: mean NLL + 1e-3 |b|^2, L-BFGS-B from T = 1, b = 0."""
    def nll(p):
        q = log_softmax(z / np.exp(p[0]) + p[1:])
        return -q[np.arange(len(y)), y].mean() + 1e-3 * (p[1:] ** 2).sum()
    r = minimize(nll, np.zeros(6), method="L-BFGS-B")
    return float(np.exp(r.x[0])), r.x[1:]


def x_u_split(k, labels, seed):
    """Labeled indices as train/utils.py:x_u_split (same RandomState stream)."""
    rs = np.random.RandomState(seed)
    out = []
    for i in range(5):
        idx = np.where(np.asarray(labels) == i)[0]
        rs.shuffle(idx)
        out.extend(idx[:k])
    return np.array(out)


def v16(y_val, seed, n=16):
    """The 16 validation images per class of a seed (train/utils.py:submuestrear_validacion)."""
    rng = np.random.RandomState(10_000 + seed)
    out = []
    for c in range(5):
        out.extend(rng.choice(np.flatnonzero(y_val == c), min(n, int((y_val == c).sum())), replace=False).tolist())
    return np.array(sorted(out), dtype=int)


_CACHE = {}


def _load(name):
    if name not in _CACHE:
        path = os.path.join(STUDY2, name)
        if not os.path.exists(path):
            raise SystemExit(f"{path} not found (see results/README.md)")
        _CACHE[name] = np.load(path) if name.endswith(".npz") else pd.read_csv(path)
    return _CACHE[name]


def expert_runs():
    return _load("expert_runs.csv")


def run_logits(target, model, seed, k=16):
    """(val, test) logits of one run of results/study2/expert_runs.csv."""
    r = expert_runs().query("target == @target and model == @model and seed == @seed and k == @k")
    assert len(r) == 1, (target, model, seed, k, len(r))
    z = _load(f"predictions_experts_{target}.npz")
    row = int(r.row.iloc[0])
    return z["logits_val"][row].astype(np.float64), z["logits_test"][row].astype(np.float64)


def panel_logits(target, method, seed, k=16, init="ham"):
    """(val, test) logits of a Study 1 run (results/predictions_<target>.npz)."""
    R = C.runs()
    r = R[(R.target == target) & (R.method == method) & (R.seed == seed) & (R.k == k) & (R.init == init)]
    assert len(r) == 1, (target, method, seed, k)
    z = C.predictions(target)
    return z["logits_val"][r.row.item()].astype(np.float64), z["logits_test"][r.row.item()].astype(np.float64)


def images(target):
    """Labels, ids, lesions and masks of the validation and test images (ImageFolder order)."""
    key = "images_" + target
    if key not in _CACHE:
        z = _load("source_models.npz")
        sp = "milk10k_dermoscopic" if target.startswith("milk10k") else target
        les = pd.read_csv(os.path.join(C.ROOT, "data", "splits", sp + ".csv")).set_index("isic_id").lesion_id
        d = {p: dict(y=z[f"labels_{target}_{p}"].astype(int), ids=z[f"ids_{target}_{p}"].astype(str))
             for p in ("train", "val", "test")}
        for p in d.values():
            p["lesion"] = les.loc[p["ids"]].to_numpy().astype(str)
        d["test"]["primary"] = C.primary_mask(target, d["test"]["ids"])
        d["val"]["copy"] = np.isin(d["val"]["ids"], sorted(val_copies())) if target.startswith("milk10k") \
            else np.zeros(len(d["val"]["ids"]), bool)
        _CACHE[key] = d
    return _CACHE[key]


def val_copies():
    """MILK10k validation images that copy a HAM10000 photograph (pairs above the NCC threshold)."""
    d = pd.read_csv(os.path.join(C.ROOT, "data", "milk10k_duplicates.csv"))
    d = d[(d.kind == "milk10k_copy_of_ham10000") & (d.ncc_tier == "above threshold")]
    s = pd.read_csv(os.path.join(C.ROOT, "data", "splits", "milk10k_dermoscopic.csv")).set_index("isic_id").split
    return set(i for i in d.isic_id_a if s.loc[i] == "val")


def source_calibrated(target, model, k, seed):
    """Log-probabilities of a source network on val and test: HAM10000 prior removed, BCTS on the labeled images."""
    z = _load("source_models.npz")
    im = images(target)
    lab = x_u_split(k, im["train"]["y"], seed)
    lp = ham_log_prior()
    T, b = bcts(z[f"{model}_{target}_train"][lab].astype(np.float64) - lp, im["train"]["y"][lab])
    return tuple(log_softmax((z[f"{model}_{target}_{p}"].astype(np.float64) - lp) / T + b) for p in ("val", "test"))


def erm_logprob(logits):
    return log_softmax(np.asarray(logits, np.float64) / T_ERM)


def four_experts(target, seed, k=16, re_source="study2"):
    """dict(val, test) with n x 4 x 5 log-probabilities in the order RE, RH, CH, CE (all val and all test images).

    re_source: 'study2' = the ResNet ERM of Study 2 (k = 16 only); 'panel' = the ERM run of the Study 1 panel."""
    rh = source_calibrated(target, "resnet50_ham", k, seed)
    ch = source_calibrated(target, "convnext_tiny_ham", k, seed)
    re_ = run_logits(target, "resnet_erm", seed, k) if re_source == "study2" else panel_logits(target, "erm", seed, k)
    ce = run_logits(target, "convnext_erm", seed, k)
    return {p: np.stack([erm_logprob(re_[i]), rh[i], ch[i], erm_logprob(ce[i])], 1)
            for i, p in enumerate(("val", "test"))}


def combine(L, w=WEIGHTS):
    """Scores of the weighted sum of log-probabilities (... x 4 x 5 -> ... x 5)."""
    return np.einsum("...ek,e->...k", L, np.asarray(w, float))


_TABLE = {}


def table(target, seed):
    """The four-expert combination of the thesis (k = 16): Lt on the primary test set, Lv on all validation images,
    with labels, ids, lesions, v16 and dev (validation without v16 and without the copies of HAM10000 images)."""
    key = (target, seed)
    if key not in _TABLE:
        E = four_experts(target, seed)
        im = images(target)
        m = im["test"]["primary"]
        yv = im["val"]["y"]
        sel = v16(yv, seed)
        dev = np.setdiff1d(np.arange(len(yv)), sel)
        dev = dev[~im["val"]["copy"][dev]]
        _TABLE[key] = dict(Lv=E["val"], yv=yv, ids_v=im["val"]["ids"], lesion_v=im["val"]["lesion"],
                           copy_v=im["val"]["copy"], v16=sel, dev=dev,
                           Lt=E["test"][m], yt=im["test"]["y"][m], ids_t=im["test"]["ids"][m],
                           lesion_t=im["test"]["lesion"][m], test_mask=m)
    return _TABLE[key]
