"""Dataset tables of Chapter 3 and Appendix 7.2: tab:cross_dataset_overlap, tab:ham10000_original_distribution,
tab:ham10000_mapping, tab:bcn20000_distribution, tab:bcn20000_mapping, tab:milk10k_distribution, tab:milk10k_mapping
and tab:data_partitions.

Diagnoses come from diagnosis_3; in HAM10000 and BCN20000 an empty diagnosis_3 with a vascular diagnosis_2 is
"Vascular", any other empty one "Not annotated". The HAM10000 mapping excludes lesions shared with MILK10k.
Inputs: data/metadata/{ham10000,bcn20000,milk10k}.csv, data/diagnosis_to_class.csv, data/splits/*.csv.
Outputs: tables/<label>.csv.
"""
import os

import numpy as np
import pandas as pd

import common as C

DATA = os.path.join(C.ROOT, "data")
NOT_ANNOTATED = "Not annotated"
# Spelling of the thesis tables for a few MILK10k diagnoses
DISPLAY = {"Squamous cell carcinoma, Invasive": "Squamous cell carcinoma, invasive",
           "Melanoma Invasive": "Melanoma, invasive",
           "Lichen planus like keratosis": "Lichen planus-like keratosis"}
# Original seven HAM10000 categories
HAM_DX = {"Nevus": "nv", "Melanoma, NOS": "mel", "Pigmented benign keratosis": "bkl",
          "Basal cell carcinoma": "bcc", "Squamous cell carcinoma, NOS": "akiec",
          "Solar or actinic keratosis": "akiec", "Vascular": "vasc", "Dermatofibroma": "df"}


def metadata(name, vascular_fallback):
    d = pd.read_csv(os.path.join(DATA, "metadata", name + ".csv"))
    dx = d.diagnosis_3.copy()
    if vascular_fallback:
        vasc = dx.isna() & d.diagnosis_2.fillna("").str.contains("Vascular")
        dx[vasc] = "Vascular"
    d["dx"] = dx.fillna(NOT_ANNOTATED)
    return d


def order_key(dx, n):
    """Count descending, ties alphabetical, "Not annotated" last."""
    return [(x == NOT_ANNOTATED, -c, x) for x, c in zip(dx, n)]


def distribution(d, lesions_only=False):
    g = d.groupby("dx").agg(images=("isic_id", "size"), lesions=("lesion_id", "nunique")).reset_index()
    assert g.lesions.sum() == d.lesion_id.nunique()  # one diagnosis per lesion
    keys = order_key(g.dx, g.lesions if lesions_only else g.images)
    g = g.iloc[sorted(range(len(g)), key=lambda i: keys[i])]
    g["images_pct"] = 100 * g.images / g.images.sum()
    g["lesions_pct"] = 100 * g.lesions / g.lesions.sum()
    total = pd.DataFrame([{"dx": "Total", "images": g.images.sum(), "lesions": g.lesions.sum(),
                           "images_pct": 100.0, "lesions_pct": 100.0}])
    return pd.concat([g, total], ignore_index=True)


def mapping(d, dataset, with_images=True):
    """Retained diagnoses by class (count descending within class), then excluded ones."""
    m = pd.read_csv(os.path.join(DATA, "diagnosis_to_class.csv"))
    m = m[m.dataset == dataset].drop_duplicates("diagnosis_3").set_index("diagnosis_3")["class"]
    g = d.groupby("dx").agg(lesions=("lesion_id", "nunique"), images=("isic_id", "size")).reset_index()
    g["class"] = g.dx.map(m).fillna("Excluded")
    g["rank"] = g["class"].map({c: i for i, c in enumerate(C.CLASSES)}).fillna(len(C.CLASSES))
    keys = [(r, *order_key([x], [n])[0]) for r, x, n in zip(g["rank"], g.dx, g.lesions)]
    g = g.iloc[sorted(range(len(g)), key=lambda i: keys[i])].drop(columns="rank")
    kept = g[g["class"] != "Excluded"]
    total = pd.DataFrame([{"dx": "Total retained", "lesions": kept.lesions.sum(), "images": kept.images.sum(),
                           "class": ""}])
    g = pd.concat([g, total], ignore_index=True)
    g["dx"] = g.dx.replace(DISPLAY)
    cols = ["dx", "lesions", "images", "class"] if with_images else ["dx", "lesions", "class"]
    return g[cols].rename(columns={"dx": "diagnosis_3"}), kept


def check_retained(d, kept, split_ids):
    ids = set(d.isic_id[d.dx.isin(kept.dx)])
    assert ids == set(split_ids), "retained images differ from data/splits"


def main():
    ham = metadata("ham10000", vascular_fallback=True)
    bcn = metadata("bcn20000", vascular_fallback=True)
    milk_all = metadata("milk10k", vascular_fallback=False)
    milk = milk_all[milk_all.image_type == "dermoscopic"].copy()
    assert len(milk) == milk.lesion_id.nunique() == 5240

    split = {n: pd.read_csv(os.path.join(DATA, "splits", n + ".csv"))
             for n in ("ham10000", "bcn20000", "milk10k_dermoscopic", "milk10k_clinical")}

    L = {"HAM10000": set(ham.lesion_id), "BCN20000": set(bcn.lesion_id), "MILK10k": set(milk.lesion_id)}
    pairs = [("HAM10000", "BCN20000"), ("HAM10000", "MILK10k"), ("BCN20000", "MILK10k")]
    C.save_table(pd.DataFrame([{"dataset_1": a, "dataset_2": b, "shared_lesions": len(L[a] & L[b])}
                               for a, b in pairs]), "cross_dataset_overlap")
    shared = L["HAM10000"] & L["MILK10k"]
    print(f"  HAM10000 lesions shared with MILK10k: {len(shared)} "
          f"({100 * len(shared) / len(L['HAM10000']):.1f} % of HAM10000, {100 * len(shared) / len(L['MILK10k']):.1f} % "
          f"of MILK10k), {ham.lesion_id.isin(shared).sum()} HAM10000 images")

    g = distribution(ham)
    g.insert(0, "class", g.dx.map(HAM_DX).fillna(""))
    C.save_table(g.rename(columns={"dx": "diagnosis_3"}), "ham10000_original_distribution")
    ham_kept = ham[~ham.lesion_id.isin(shared)]
    print(f"  HAM10000 after removing shared lesions: {ham_kept.lesion_id.nunique()} lesions, {len(ham_kept)} images")
    t, kept = mapping(ham_kept, "HAM10000")
    check_retained(ham_kept, kept, split["ham10000"].isic_id)
    C.save_table(t, "ham10000_mapping")

    C.save_table(distribution(bcn).rename(columns={"dx": "diagnosis_3"}), "bcn20000_distribution")
    t, kept = mapping(bcn, "BCN20000")
    check_retained(bcn, kept, split["bcn20000"].isic_id)
    C.save_table(t, "bcn20000_mapping")

    # -- MILK10k (lesion level; one dermoscopic and one clinical image per lesion)
    g = distribution(milk, lesions_only=True)
    g["dx"] = g.dx.replace(DISPLAY)
    C.save_table(g[["dx", "lesions", "lesions_pct"]].rename(columns={"dx": "diagnosis_3"}), "milk10k_distribution")
    t, kept = mapping(milk, "MILK10K", with_images=False)
    check_retained(milk, kept, split["milk10k_dermoscopic"].isic_id)
    clin = milk_all[milk_all.image_type != "dermoscopic"]
    assert set(clin.isic_id[clin.lesion_id.isin(milk.lesion_id[milk.dx.isin(kept.dx)])]) == \
        set(split["milk10k_clinical"].clinical_isic_id)
    C.save_table(t, "milk10k_mapping")

    rows = []
    for name, s in (("HAM10000", split["ham10000"]), ("BCN20000", split["bcn20000"]),
                    ("MILK10k (per modality)", split["milk10k_dermoscopic"])):
        for sp, spn in (("train", "Train"), ("val", "Validation"), ("test", "Test")):
            x = s[s.split == sp]
            rows.append({"dataset": name, "split": spn, "lesions": x.lesion_id.nunique(), "images": len(x)})
        rows.append({"dataset": name, "split": "Total", "lesions": s.lesion_id.nunique(), "images": len(s)})
        # lesion-level split: no lesion in two partitions
        assert s.groupby("lesion_id").split.nunique().max() == 1
    C.save_table(pd.DataFrame(rows), "data_partitions")


if __name__ == "__main__":
    main()
