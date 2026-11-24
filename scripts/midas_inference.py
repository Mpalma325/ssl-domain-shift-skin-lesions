"""Runs trained networks on MRA-MIDAS (external validation of Chapter 5: fig:e2_midas_curva, MIDAS columns of
tab:e2_robusta_clinica, tab:e2_robusta_derm and tab:app_e2_robusta_v1, MIDAS rows of tab:app_e2_variantes_lfm).

MIDAS is not redistributed: download it from its official source (Stanford AIMI) and point --metadata to its
release_midas.csv and --images to the folder with its images. Nothing here trains, and the output stays local.

Image list: lesions whose diagnosis maps to the five classes (AK and SCC in situ -> akiec; BCC; seborrheic keratosis ->
bkl; melanoma; melanocytic nevus -> nv; invasive SCC, other diagnoses and control skin out), one row per image that
is on disk, in the order of the metadata file. Modality from the distance: dermoscope -> derm, 6 in -> clin15,
1 ft -> clin30. A file missing under its listed name is matched to a "_cropped" file with the same stem.
Clinical photographs are first resized to 600 px on the long side (the MILK10k size).

  --features NAME=PATH   pooled features of every view: clinical base, centred zoom x2, x2.5, x3 and x2.5 without
                         ink and hair; dermoscopic base (mixed precision on GPU)
  --classifier NAME=PATH logits (float32) on the base view of the modality the network was trained on; append
                         @clinical or @derm (default clinical)

  python scripts/midas_inference.py --metadata <midas>/release_midas.csv --images <midas>/images \
      --features ham=checkpoints/ham10000_source.pth \
      --classifier erm_clinical_s0=outputs/ham/milk10k_clinico/k16/erm/s0/checkpoints/best.pth@clinical
Output: --out npz (default outputs/midas/midas_inference.npz) with image (file name), lesion, label, modality, ink and hair
fractions, feat_<name>__<view> and logits_<name>. Keep it outside the repository.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract_features as X  # noqa: E402

DIAGNOSIS = {"malignant- ak": "akiec", "malignant- sccis": "akiec", "malignant- bcc": "bcc",
             "benign-seborrheic keratosis": "bkl", "malignant- melanoma": "mel", "benign-melanocytic nevus": "nv"}
DISTANCE = {"dscope": "derm", "6in": "clin15", "1ft": "clin30"}


def image_list(metadata, images):
    D = pd.read_csv(metadata)
    on_disk = sorted(os.listdir(images))
    present = set(on_disk)

    def stem(f):
        s = os.path.splitext(f)[0].lower()
        return s[:-len("_cropped")] if s.endswith("_cropped") else s
    by_stem = {}
    for f in on_disk:
        by_stem.setdefault(stem(f), []).append(f)

    def resolve(f):
        if f in present:
            return f
        c = by_stem.get(stem(f), [])
        return c[0] if len(c) == 1 else None
    D["label"] = D.midas_path.map(DIAGNOSIS)
    D["modality"] = D.midas_distance.map(DISTANCE)
    D = D[D.label.notna() & D.modality.notna() & (D.midas_iscontrol.astype(str).str.lower() != "yes")].copy()
    D["file"] = D.midas_file_name.map(resolve)
    D = D[D.file.notna()].reset_index(drop=True)
    D["lesion"] = D.midas_record_id.astype(str) + "|" + D.midas_location.astype(str)
    return D[["file", "lesion", "label", "modality"]]


def to600(img):
    w, h = img.size
    if max(w, h) <= 600:
        return img
    r = 600.0 / max(w, h)
    return img.resize((max(1, round(w * r)), max(1, round(h * r))))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--metadata", required=True, help="release_midas.csv of the MIDAS download")
    ap.add_argument("--images", required=True, help="folder with the MIDAS images")
    ap.add_argument("--features", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--classifier", action="append", default=[], metavar="NAME=PATH[@clinical|@derm]")
    ap.add_argument("--views", nargs="+", default=X.CLINICAL_VIEWS, choices=X.CLINICAL_VIEWS)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "midas", "midas_inference.npz"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="only N images per modality, for tests")
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit("%s exists; not overwritten" % a.out)
    t0 = time.time()
    device = X.get_device(a.device)
    L = image_list(a.metadata, a.images)
    if a.limit:
        L = pd.concat([L[L.modality == m].head(a.limit) for m in ("clin15", "clin30", "derm")]).reset_index(drop=True)
    n = len(L)
    clin = L.modality.str.startswith("clin").values
    print("MIDAS: %d images (%s), %d lesions | device %s" % (
        n, ", ".join("%s %d" % kv for kv in L.modality.value_counts().sort_index().items()), L.lesion.nunique(),
        device), flush=True)

    V = {v: [None] * n for v in a.views}
    ink, hair = np.full(n, np.nan, np.float32), np.full(n, np.nan, np.float32)
    for i, f in enumerate(L.file):
        img = X.open_rgb(os.path.join(a.images, f))
        if clin[i]:
            vs, ink[i], hair[i] = X.clinical_views(to600(img), a.views)
            for k in a.views:
                V[k][i] = vs[k]
        else:
            V["base"][i] = X.base_view(img)
        if (i + 1) % 250 == 0 or i + 1 == n:
            print("  views %d/%d (%.1f min)" % (i + 1, n, (time.time() - t0) / 60), flush=True)
    res = dict(image=np.array(L.file.tolist(), dtype=str), lesion=np.array(L.lesion.tolist(), dtype=str),
               label=np.array(L.label.tolist(), dtype=str), modality=np.array(L.modality.tolist(), dtype=str),
               ink=ink, hair=hair)
    amp = device.type == "cuda"
    for name, path in X.parse_nets(a.features).items():
        net = X.load_network(path).to(device)
        for k in a.views:
            idx = [i for i in range(n) if V[k][i] is not None]
            F = None
            if idx:
                Fk = X.forward(net, [V[k][i] for i in idx], device, amp, 32)[1]
                F = np.full((n, Fk.shape[1]), np.nan, np.float16)
                F[idx] = Fk
            res["feat_%s__%s" % (name, k)] = F if F is not None else np.zeros((n, 0), np.float16)
        del net
        print("  features %s (%.1f min)" % (name, (time.time() - t0) / 60), flush=True)
    for name, spec in X.parse_nets(a.classifier).items():
        path, _, mod = spec.partition("@")
        rows = np.flatnonzero(~clin if mod == "derm" else clin)
        net = X.load_network(path).to(device)
        Z = np.full((n, 5), np.nan, np.float32)
        if len(rows):
            Z[rows] = X.forward(net, [V["base"][i] for i in rows], device, False, 32)[0]
        res["logits_" + name] = Z
        del net
        print("  logits %s (%.1f min)" % (name, (time.time() - t0) / 60), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, **res)
    print("saved %s (%.1f min)" % (a.out, (time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()
