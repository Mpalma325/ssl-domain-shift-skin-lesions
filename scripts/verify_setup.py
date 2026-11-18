"""Check that this machine reproduces the setup of the thesis; exits 0 only if every check passes.

1. Versions of python, torch, torchvision, numpy and pillow (another torchvision changes RandAugment).
2. Hash of each split's sorted file list and of the labeled subset of every seed at k = 16, 32, 100.
3. Optional source-only canary (GPU and HAM10000 checkpoint): balanced accuracy on each target test set.

Usage: python scripts/verify_setup.py [--checkpoint checkpoints/ham10000_source.pth]
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINGERPRINTS = os.path.join(ROOT, "data", "fingerprints.json")
CLASSES = ["akiec", "bcc", "bkl", "mel", "nv"]
CANARY_SETS = {"MILK10K dermoscopic": "MILK10K", "MILK10K clinical": "MILK10K_CLINICO",
               "BCN20000": "BCN20000"}

ok_all = True


def header(title):
    print(f"\n{'=' * 66}\n  {title}\n{'=' * 66}")


def report(cond, msg):
    global ok_all
    print(("  [OK]   " if cond else "  [FAIL] ") + msg)
    ok_all &= bool(cond)


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def listing(root):
    """File list in the order produced by torchvision.datasets.ImageFolder."""
    classes = sorted(e.name for e in os.scandir(root) if e.is_dir())
    out = []
    for c in classes:
        for r, _, files in sorted(os.walk(os.path.join(root, c))):
            for f in sorted(files):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    out.append((f, c))
    return classes, out


def check_versions(expected):
    header("1/3  VERSIONS")
    import PIL
    import torch
    import torchvision
    got = {"python": ".".join(map(str, sys.version_info[:2])),
           "torch": torch.__version__.split("+")[0],
           "torchvision": torchvision.__version__.split("+")[0],
           "numpy": np.__version__, "pillow": PIL.__version__}
    for k, exp in expected.items():
        g = got.get(k, "?")
        if k == "python" and not g.startswith(exp):
            print(f"  [WARN] {k:12} expected {exp:10} got {g} (the canary decides)")
            continue
        report(g == exp or (k == "python" and g.startswith(exp)), f"{k:12} expected {exp:10} got {g}")


def check_data(datasets):
    header("2/3  DATA (file lists and labeled subsets)")
    for ds, ref in datasets.items():
        base = os.path.join(ROOT, "data", ds)
        if not os.path.isdir(base):
            report(False, f"{ds}: missing, run scripts/prepare_data.py")
            continue
        for sp in ("train", "val", "test"):
            _, files = listing(os.path.join(base, sp))
            h = sha("|".join(f for f, _ in files))
            report(h == ref[sp]["hash"] and len(files) == ref[sp]["n"],
                   f"{ds:16} {sp:6} {len(files):6} images  {h}")
        classes, files = listing(os.path.join(base, "train"))
        labels = np.array([classes.index(c) for _, c in files])
        names = [f for f, _ in files]
        for seed_txt, per_k in sorted(ref["labeled_by_seed"].items(), key=lambda kv: int(kv[0])):
            for k, exp in sorted(per_k.items(), key=lambda kv: int(kv[0])):
                # same draw as x_u_split in train/utils.py
                rs = np.random.RandomState(int(seed_txt))
                sel = []
                for i in range(len(classes)):
                    idx = np.where(labels == i)[0]
                    rs.shuffle(idx)
                    sel.extend(idx[:int(k)])
                report(sha("|".join(sorted(names[i] for i in sel))) == exp,
                       f"{ds:16} labeled subset seed {seed_txt} k={k}")


def check_canary(checkpoint, expected, tol):
    header("3/3  SOURCE-ONLY CANARY")
    if not ok_all:
        print("  Skipped: fix the checks above first.")
        return
    import torch
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder
    sys.path.insert(0, os.path.join(ROOT, "train"))
    import utils

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = utils.get_model("resnet50", pretrained_checkpoint=None)
    model = utils.ImageClassifier(backbone, len(CLASSES), bottleneck_dim=1024, pool_layer=None,
                                  finetune=True).to(dev)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    missing, _ = model.load_state_dict(ck["state_dict"], strict=False)
    assert not [m for m in missing if not m.startswith("backbone.fc")], f"missing weights: {missing[:5]}"
    model.eval()
    tf = utils.get_val_transform("default")
    for name, folder in CANARY_SETS.items():
        ds = ImageFolder(os.path.join(ROOT, "data", folder, "test"), transform=tf)
        preds, labs = [], []
        with torch.no_grad():
            for x, y in DataLoader(ds, batch_size=32, shuffle=False, num_workers=0):
                preds.append(model(x.to(dev)).argmax(1).cpu())
                labs.append(y)
        p, y = torch.cat(preds).numpy(), torch.cat(labs).numpy()
        bal = float(np.mean([(p[y == c] == c).mean() * 100 for c in range(len(CLASSES))]))
        exp = expected[name]
        report(abs(bal - exp) <= tol, f"{name:22} balanced accuracy {bal:5.1f} (reference {exp:.1f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", help="HAM10000 source checkpoint; enables the canary")
    args = ap.parse_args()
    ref = json.load(open(FINGERPRINTS, encoding="utf-8"))
    check_versions(ref["env"])
    check_data(ref["datasets"])
    if args.checkpoint:
        check_canary(args.checkpoint, ref["canary_source_only"], ref.get("tolerance_pp", 0.5))
    else:
        print("\n  Canary skipped (pass --checkpoint to run it).")
    header("RESULT")
    print("  All checks passed." if ok_all else "  Some checks failed: results would not be comparable.")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
