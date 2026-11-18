"""MEL sensitivity at 80 % specificity, SSL - ERM, HAM10000 init. (figure mel_sensitivity).

Threshold per run at 80 % specificity on its own validation set (MEL log-softmax); sensitivity on the primary test
set. Per seed, mean over k of the paired difference; mean over five seeds and 95 % t interval (frame: excludes zero).
Column "All" and row "All SSL": per-seed means over conditions and over methods.
Outputs: figures/mel_sensitivity.pdf/.png, tables/mel_sensitivity.csv, tables/mel_sensitivity_runs.csv (per run).
"""
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm
from scipy import stats

import common as C

plt = C.plt
MEL = C.CLASSES.index("mel")
COLUMNS = {"all": "All", "bcn20000": "BCN20000", "milk10k": "MILK10k\ndermoscopic",
           "milk10k_clinico": "MILK10k\nclinical"}
LIM = 10


def threshold_at_specificity(neg, spec=0.8):
    """Smallest threshold whose specificity on the negatives is >= spec."""
    s = np.sort(np.asarray(neg, np.float64))
    m = int(np.ceil(spec * len(s) - 1e-9))
    return float(s[max(m, 1) - 1])


def log_softmax(z):
    z = np.asarray(z, float)
    return z - np.logaddexp.reduce(z, axis=1, keepdims=True)


def t_interval(v):
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return v.mean(), h


def run_sensitivity():
    """MEL sensitivity at 80 % validation specificity for every run of the main design."""
    R = C.runs()
    rows = []
    for target in C.TARGETS:
        z = C.predictions(target)
        # read each npz array once: indexing z[...] inside the loop would decompress it on every run
        logits_test, logits_val = z["logits_test"], z["logits_val"]
        mask = C.primary_mask(target, z["ids_test"])
        y = z["labels_test"].astype(int)[mask]
        yv = z["labels_val"].astype(int)
        for r in R[(R.target == target) & (R.method != "fully_supervised")].itertuples():
            lp = log_softmax(logits_test[r.row][mask])[:, MEL]
            lpv = log_softmax(logits_val[r.row])[:, MEL]
            thr = threshold_at_specificity(lpv[yv != MEL])
            rows.append(dict(run_id=r.run_id, init=r.init, target=target, k=r.k, method=r.method, seed=r.seed,
                             sens80=100 * float((lp[y == MEL] > thr).mean()),
                             spec80=100 * float((lp[y != MEL] <= thr).mean())))
    return pd.DataFrame(rows)


def compute(d):
    dd = C.deltas(d, cols=["sens80"])
    ham = dd[dd.init == "ham"]
    rows = []
    for m in ["all"] + C.SSL:
        s = ham if m == "all" else ham[ham.method == m]
        for t in COLUMNS:
            x = s if t == "all" else s[s.target == t]
            v = x.groupby("seed").d_sens80.mean().to_numpy()
            mean, h = t_interval(v)
            rows.append(dict(method=m, target=t, n_seeds=len(v), d_sens80=mean, t_half_width=h,
                             excludes_zero=abs(mean) > h))
    return pd.DataFrame(rows)


def figure(r):
    order = ["all"] + C.SSL
    M = r.pivot(index="method", columns="target", values="d_sens80").loc[order, list(COLUMNS)].to_numpy()
    S = r.pivot(index="method", columns="target", values="excludes_zero").loc[order, list(COLUMNS)].to_numpy(bool)
    norm = TwoSlopeNorm(0, -LIM, LIM)
    cmap = plt.get_cmap("RdBu")
    fig = plt.figure(figsize=(C.WIDTH, 4.1))
    ax = fig.add_axes([0.22, 0.03, 0.60, 0.84])
    cax = fig.add_axes([0.85, 0.03, 0.02, 0.84])
    ax.pcolormesh(M, cmap=cmap, norm=norm, edgecolors="white", linewidth=1.6)
    ax.invert_yaxis()
    sep_y, acc = {1: 5}, 1
    for _, ms in C.FAMILIES[:-1]:
        acc += len(ms)
        sep_y[acc] = 3
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j + 0.5, i + 0.5, C.r1(M[i, j]), ha="center", va="center", fontsize=9,
                    color=C.ink_on(np.clip(M[i, j], -LIM, LIM), cmap, norm))
            if S[i, j]:
                C.cell_frame(ax, j, i, sep_x={1: 5}, sep_y=sep_y, base=1.6)
    ax.axvline(1, color="white", lw=5)
    ax.axhline(1, color="white", lw=5)
    for pos, lw in sep_y.items():
        if pos != 1:
            ax.axhline(pos, color="white", lw=lw)
    ax.set_xticks(np.arange(4) + 0.5, list(COLUMNS.values()))
    ax.set_yticks(np.arange(len(order)) + 0.5, ["All SSL"] + [C.NAME[m] for m in C.SSL])
    ax.xaxis.tick_top()
    ax.tick_params(length=0, labelsize=9.5)
    for label in ax.get_xticklabels():
        label.set_fontfamily(["cmb10", "DejaVu Serif"])
    for side in ax.spines.values():
        side.set_visible(False)
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, extend="both")
    ticks = [-10, -5, 0, 5, 10]
    bar.set_ticks(ticks, labels=[f"${t}$" for t in ticks])
    bar.ax.tick_params(length=2, width=0.5, labelsize=8)
    bar.set_label("SSL $-$ ERM, MEL sensitivity (pp)", fontsize=9.5, labelpad=8)
    bar.outline.set_visible(False)
    C.save(fig, "mel_sensitivity")


def main():
    d = run_sensitivity()
    C.save_table(d, "mel_sensitivity_runs")
    r = compute(d)
    C.save_table(r, "mel_sensitivity")
    print(r.pivot(index="method", columns="target", values="d_sens80").loc[["all"] + C.SSL, list(COLUMNS)]
          .round(2).to_string())
    figure(r)


if __name__ == "__main__":
    main()
