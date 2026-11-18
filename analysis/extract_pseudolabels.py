"""Pseudo-label accuracy and ratio per epoch from the training logs of FixMatch, FlexMatch, DebiasMatch and DST.

Log lines carry running means over the epoch of "Pseudo Label Acc" (accepted pseudo-labels matching the true label)
and "Pseudo Label Ratio" (unlabeled batch above threshold), on the weak view in training mode. The last line of each
epoch is kept; " * Mean Cls" (validation BA) goes to the last epoch seen before it.
Reads outputs/<init>/<target>/k<k>/<method>/s<seed>/train-*.txt; writes results/pseudolabels_per_epoch.csv
(one row per run and epoch, from 1: pl_accuracy, pl_ratio, val_ba in %).

Usage: python analysis/extract_pseudolabels.py [--outputs outputs] [--out results/pseudolabels_per_epoch.csv]
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METHODS = ["fixmatch", "flexmatch", "debiasmatch", "dst"]
PAT = re.compile(r"Epoch: \[(\d+)\]\[\s*(\d+)/(\d+)\].*?Pseudo Label Acc\s+[\d.]+\s+\(\s*([\d.]+)\)"
                 r"(?:\s+Pseudo Label Ratio\s+[\d.]+\s+\(\s*([\d.]+)\))?")
PAT_EPOCH = re.compile(r"Epoch: \[(\d+)\]\[")
PAT_VAL = re.compile(r"^ \* Mean Cls ([\d.]+)")


def read_log(folder):
    """Last line per epoch and val BA per epoch (from 0); later logs overwrite the epoch a resumed run repeats."""
    per_epoch, val = {}, {}
    paths = sorted(glob.glob(os.path.join(folder, "train-*.txt"))) + glob.glob(os.path.join(folder, "train.txt"))
    for path in paths:
        current = None
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                e = PAT_EPOCH.search(line)
                if e:
                    current = int(e.group(1))
                v = PAT_VAL.search(line)
                if v and current is not None:
                    val[current] = float(v.group(1))
                m = PAT.search(line)
                if not m:
                    continue
                ep, it = int(m.group(1)), int(m.group(2))
                if ep not in per_epoch or it >= per_epoch[ep][0]:
                    per_epoch[ep] = (it, float(m.group(4)), float(m.group(5)) if m.group(5) else np.nan)
    return per_epoch, val


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", default=os.path.join(ROOT, "outputs"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "pseudolabels_per_epoch.csv"))
    args = ap.parse_args()

    runs = pd.read_csv(os.path.join(ROOT, "results", "runs.csv"))
    runs = runs[runs.method.isin(METHODS)]
    rows, missing = [], 0
    for r in runs.itertuples():
        folder = os.path.join(args.outputs, r.init, r.target, f"k{r.k}", r.method, f"s{r.seed}")
        per_epoch, val = read_log(folder)
        if not per_epoch:
            missing += 1
            continue
        for ep, (_, acc, ratio) in sorted(per_epoch.items()):
            rows.append(dict(run_id=r.run_id, init=r.init, target=r.target, k=r.k, method=r.method, seed=r.seed,
                             epoch=ep + 1, pl_accuracy=acc, pl_ratio=ratio, val_ba=val.get(ep, np.nan)))
    out = pd.DataFrame(rows).sort_values(["method", "init", "target", "k", "seed", "epoch"])
    out.to_csv(args.out, index=False)
    print(f"{out.run_id.nunique()} runs, {len(out)} rows -> {args.out} ({missing} runs without a log)")


if __name__ == "__main__":
    main()
