# Data

The images are not included. Download them from their official sources and
build the folders with `scripts/prepare_data.py`.

| Dataset | Source | License | Cite |
|---|---|---|---|
| HAM10000 | [Harvard Dataverse](https://doi.org/10.7910/DVN/DBW86T) or ISIC 2018 Task 3 | CC BY-NC 4.0 | Tschandl et al., *Sci. Data* 5, 180161 (2018) |
| BCN20000 | [ISIC Archive](https://www.isic-archive.com), collection 249 | CC BY-NC 4.0 | Hernández-Pérez et al., *Sci. Data* 11, 641 (2024) |
| MILK10k | [ISIC Archive](https://doi.org/10.34970/648456) | CC BY-NC 4.0 | MILK study team, doi:10.34970/648456 |

## Files

- `splits/ham10000.csv`, `splits/bcn20000.csv`, `splits/milk10k_dermoscopic.csv`:
  one row per image with `isic_id`, `lesion_id`, `class` and `split`.
- `splits/milk10k_clinical.csv`: the clinical image of each MILK10k lesion
  (`clinical_isic_id`) next to its dermoscopic image (`derm_isic_id`). Both
  modalities share the lesion, class and split.
- `splits/milk10k_test_excluded.txt`: 29 MILK10k test images whose dermoscopic
  photograph copies a HAM10000 image. They are removed from the primary test
  set (885 images) at evaluation time; the folders keep all 914.
- `diagnosis_to_class.csv`: how the original diagnoses of each dataset map to
  the five classes (`akiec`, `bcc`, `bkl`, `mel`, `nv`).
- `fingerprints.json`: hashes used by `scripts/verify_setup.py`.
- `metadata/ham10000.csv`, `metadata/bcn20000.csv`, `metadata/milk10k.csv`:
  trimmed ISIC metadata (image and lesion identifiers, diagnosis levels and,
  for MILK10k, image type), used for the dataset tables. Same licenses and
  attribution as the datasets above.
- `milk10k_duplicates.csv`: MILK10k images that copy a HAM10000 photograph
  (141 pairs, 121 MILK10k images) and duplicate pairs within MILK10k (8), with
  the image similarity measures and the visual verdict used to confirm each.

Splits are by lesion: all images of a lesion fall in the same partition.
HAM10000 lesions that also appear in MILK10k were removed from HAM10000.
The class labels are grouped from the original diagnoses; this is a change to
the original data in the sense of the CC BY-NC 4.0 license.
