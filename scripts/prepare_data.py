"""Build data/<DATASET>/<split>/<class>/<image>.jpg (ImageFolder layout) from the official ISIC downloads.

Reads data/splits/; the images are not redistributed. Keep folder and file names: each seed's labeled subset depends
on the sorted file list (checked by scripts/verify_setup.py against data/fingerprints.json). MILK10k clinical images
take the file name of the dermoscopic image of the same lesion, so both modalities share labeled subsets.

Usage: python scripts/prepare_data.py --ham DIR --bcn DIR --milk DIR [--link]  (searched for ISIC_*.jpg; omit any)
"""
import argparse
import csv
import os
import re
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS = os.path.join(ROOT, "data", "splits")
ISIC = re.compile(r"(ISIC_\d{7})\.(jpe?g|png)$", re.IGNORECASE)

# output folder -> (split file, column with the source image, column naming the output file)
TARGETS = {
    "HAM10000": ("ham10000.csv", "isic_id", "isic_id", "ham"),
    "BCN20000": ("bcn20000.csv", "isic_id", "isic_id", "bcn"),
    "MILK10K": ("milk10k_dermoscopic.csv", "isic_id", "isic_id", "milk"),
    "MILK10K_CLINICO": ("milk10k_clinical.csv", "clinical_isic_id", "derm_isic_id", "milk"),
}


def index_images(folder):
    """Map ISIC id -> path for every image under ``folder``."""
    found = {}
    for root, _, files in os.walk(folder):
        for name in files:
            m = ISIC.search(name)
            if m:
                found.setdefault(m.group(1), os.path.join(root, name))
    return found


def build(out_name, split_file, src_col, dst_col, images, link, overwrite):
    out_dir = os.path.join(ROOT, "data", out_name)
    with open(os.path.join(SPLITS, split_file), newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    missing = [r[src_col] for r in rows if r[src_col] not in images]
    if missing:
        sys.exit(f"{out_name}: {len(missing)} of {len(rows)} images not found "
                 f"(first: {missing[:3]}). Check the download folder.")
    for r in rows:
        src = images[r[src_col]]
        ext = os.path.splitext(src)[1].lower()
        dst = os.path.join(out_dir, r["split"], r["class"], r[dst_col] + ext)
        if os.path.exists(dst) and not overwrite:
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if link:
            if os.path.exists(dst):
                os.remove(dst)
            os.link(src, dst)
        else:
            shutil.copy2(src, dst)
    counts = {}
    for r in rows:
        counts[r["split"]] = counts.get(r["split"], 0) + 1
    print(f"{out_name:16} " + "  ".join(f"{s} {counts.get(s, 0)}" for s in ("train", "val", "test")))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ham", help="folder with the HAM10000 images (ISIC 2018 task 3)")
    ap.add_argument("--bcn", help="folder with the BCN20000 images")
    ap.add_argument("--milk", help="folder with the MILK10k training images (both modalities)")
    ap.add_argument("--link", action="store_true", help="hard-link instead of copying")
    ap.add_argument("--overwrite", action="store_true", help="replace files that already exist")
    args = ap.parse_args()

    sources = {"ham": args.ham, "bcn": args.bcn, "milk": args.milk}
    if not any(sources.values()):
        ap.error("pass at least one of --ham, --bcn, --milk")
    cache = {}
    for out_name, (split_file, src_col, dst_col, key) in TARGETS.items():
        if not sources[key]:
            continue
        if key not in cache:
            cache[key] = index_images(sources[key])
        build(out_name, split_file, src_col, dst_col, cache[key], args.link, args.overwrite)
    print("Done. Next: python scripts/verify_setup.py")


if __name__ == "__main__":
    main()
