# Tables of the thesis

Each script writes its tables to `tables/<label>.csv` with unrounded values; the thesis rounds them.
The last column compares every number with the table in the thesis after rounding.
Run a script from the repository root, e.g. `python analysis/tab_ssl_overall.py`.

## Chapter 3 and appendix: datasets

| Label | Script | Inputs | Matches the thesis |
|---|---|---|---|
| `tab:cross_dataset_overlap` | `tab_datasets.py` | data/metadata/*.csv | yes (0 / 859 / 0; also 11.5 %, 16.4 %, 1,895 images, 6,611 lesions / 8,120 images of the text) |
| `tab:data_partitions` | `tab_datasets.py` | data/splits/*.csv | yes (all 12 rows) |
| `tab:ham10000_original_distribution` | `tab_datasets.py` | data/metadata/ham10000.csv | yes (counts and % to 2 decimals) |
| `tab:ham10000_mapping` | `tab_datasets.py` | data/metadata/ham10000.csv, data/diagnosis_to_class.csv, data/splits/ham10000.csv | yes; retained images checked equal to the split file |
| `tab:bcn20000_distribution` | `tab_datasets.py` | data/metadata/bcn20000.csv | yes |
| `tab:bcn20000_mapping` | `tab_datasets.py` | data/metadata/bcn20000.csv, data/diagnosis_to_class.csv, data/splits/bcn20000.csv | yes; retained images checked equal to the split file |
| `tab:milk10k_distribution` | `tab_datasets.py` | data/metadata/milk10k.csv | yes (79 "Not annotated" include 2 vascular lesions without diagnosis_3) |
| `tab:milk10k_mapping` | `tab_datasets.py` | data/metadata/milk10k.csv, data/diagnosis_to_class.csv, data/splits/milk10k_*.csv | yes; retained lesions checked equal to both split files |
| `tab:milk_copies_location` | `tab_milk_copies.py` | data/milk10k_duplicates.csv, data/splits/milk10k_dermoscopic.csv | yes |
| `tab:milk_copies_ham` | `tab_milk_copies.py` | data/milk10k_duplicates.csv, data/splits/ham10000.csv | yes |
| `tab:milk_internal_pairs` | `tab_milk_copies.py` | data/milk10k_duplicates.csv, data/splits/milk10k_*.csv | yes; also checks that test copies + test internal pairs = the 29 ids of milk10k_test_excluded.txt |
| `tab:milk_copies_labeled` | `tab_milk_copies.py` | data/milk10k_duplicates.csv, data/splits/milk10k_dermoscopic.csv (x_u_split rebuilt) | yes (15 counts) |
| `tab:milk_copies_effect` | `tab_milk_copies.py` | results/runs.csv, predictions (common.design) | yes (all 5 rows, 2 decimals) |
| `tab:clinical_crop_review` | `tab_clinical_crop.py` | results/clinical_crop_review.csv | yes (81/16/3/19 %, 98/2/0/2 %) |
| `tab:clinical_crop_ablation` | `tab_clinical_crop.py` | results/clinical_crop_ablation_runs.csv, predictions (common.panel) | yes (all 24 numbers, 2 decimals) |

## Chapter 4

| Label | Script | Inputs | Matches the thesis |
|---|---|---|---|
| `tab:experimental_factors` | static, no script | - | yes (text table) |
| `tab:class_distance` | `tab_domain_shift.py` | data/splits/*.csv (train splits) | yes, 4 cells |
| `tab:representation_shift` | `tab_domain_shift.py` | results/shift_representations.csv (compute_shift_representations.py) | yes, 9 cells |
| `tab:source_performance` | `tab_baselines.py` | results/source_only.npz, data/splits/ham10000.csv (HAM10000 train prior) | yes, 8 cells |
| `tab:fs_performance` | `tab_baselines.py` | common.panel() (fully supervised runs) | yes, 12 cells |
| `tab:erm_gap` | `tab_baselines.py` | common.panel() | yes, 9 cells |
| `tab:init_contrast` | `tab_baselines.py` | common.panel() | yes, 24 cells and the 3 asterisks |
| `tab:modality_contrast` | `tab_baselines.py` | common.panel() | yes, 16 cells and the 7 asterisks |
| `tab:ssl_factors` | `tab_ssl_overall.py` | common.design() | yes, 7 cells |
| `tab:ssl_group` | `tab_ssl_overall.py` | common.design() | yes, 18 cells and the 4 asterisks |
| `tab:ssl_margin` | `tab_ssl_overall.py` | common.panel() | no, 8 of 9: MILK10k clinical k=32 gives 20.75 (20.7); the thesis has 20.8, probably from the rounded inputs 1.88 / 9.06 |
| `tab:ssl_without_akiec` | `tab_class_level.py` | common.design() | yes, 16 cells and the 4 asterisks |
| `tab:pair_changes` | `tab_class_level.py` | common.design() | yes, 12 counts and 12 percentages |
| `tab:mel_nv` | `tab_class_level.py` | common.design() | yes, 6 cells |
| `tab:bcc_recall` | `tab_class_level.py` | common.design() | yes, 6 cells |
| `tab:subset_seed_sd` | `tab_run_variation.py` | results/factorial_runs.csv | yes, 12 cells |
| `tab:repetitions` | `tab_run_variation.py` | results/repetitions.npz, results/repetitions_runs.csv | yes, 6 values and the epoch ranges |
| `tab:shared_misclassified` | `tab_shared_errors.py` | common.runs(), common.predictions() | yes, 6 cells; per-image list in tables/shared_misclassified_images.csv |
| `tab:shared_fs` | `tab_shared_errors.py` | common.runs(), common.predictions() | yes, 6 cells |

## Appendix: method-level results

| Label | Script | Inputs | Matches the thesis |
|---|---|---|---|
| `tab:method_specific_configuration` | static, no script | - | static text table (hyperparameters) |
| `tab:shift_marginal` | `tab_domain_shift.py` (chapter-4 group) | results/shift_representations.csv | yes, all 12 cells |
| `tab:ba_methods_bcn` | `tab_ba_methods_appendix.py` | common.panel() | yes, 66 cells (mean and sd) |
| `tab:ba_methods_milk_derm` | `tab_ba_methods_appendix.py` | common.panel() | yes, 66 cells |
| `tab:ba_methods_milk_clin` | `tab_ba_methods_appendix.py` | common.panel() | yes, 66 cells |
| `tab:init_summary` | `tab_ba_methods_appendix.py` | common.panel() (incl. fully supervised) | yes, 12 cells with asterisks |
| `tab:ssl_factors_full` | `tab_ssl_overall.py` (chapter-4 group) | common.design() | yes, 15 effects (names differ only in capitalization) |
| `tab:ssl_group_imagenet` | `tab_ssl_overall.py` (chapter-4 group) | common.design() | yes, 9 cells with asterisks |
| `tab:method_pairs` | `tab_method_pairs.py` | common.design() | yes, 55 cells with asterisks |
| `tab:ssl_without_class` | `tab_class_level.py` (chapter-4 group) | common.design() | yes, 24 cells with asterisks |
| `tab:ssl_without_class_imagenet` | `tab_class_level.py` (chapter-4 group) | common.design() | yes, 24 cells with asterisks |
| `tab:auc_class` | `tab_classes_appendix.py` | predictions_<target>.npz | yes, 30 cells with asterisks |
| `tab:mel_nv_methods` | `tab_class_level.py` (chapter-4 group) | common.design() | yes, 33 cells |
| `tab:mel_nv_methods_imagenet` | `tab_class_level.py` (chapter-4 group) | common.design() | yes, 33 cells |
| `tab:bcc_recall_methods` | `tab_class_level.py` (chapter-4 group) | common.design() | yes, 33 cells |
| `tab:bcc_recall_methods_imagenet` | `tab_class_level.py` (chapter-4 group) | common.design() | yes, 33 cells |
| `tab:all_confusions` | `tab_classes_appendix.py` | common.design() | yes, 180 cells |
| `tab:pair_changes_imagenet` | `tab_class_level.py` (chapter-4 group) | common.design() | yes, 12 cells |
| `tab:pair_changes_images` | `tab_class_level.py` (chapter-4 group) | common.design() | yes, 12 cells |
| `tab:pair_changes_images_imagenet` | `tab_class_level.py` (chapter-4 group) | common.design() | yes, 12 cells |
| `tab:malignant_benign` | `tab_classes_appendix.py` | common.design() | yes, 48 cells with asterisks |
| `tab:confidence_methods` | `tab_confidence_appendix.py` | predictions_<target>.npz | yes, 108 cells |
| `tab:confidence_gap` | `tab_confidence_appendix.py` | predictions_<target>.npz | yes, 30 cells with asterisks |
| `tab:confidence_gap_imagenet` | `tab_confidence_appendix.py` | predictions_<target>.npz | yes, 30 cells with asterisks |

## Appendix: pseudo-labels, variation between runs and control experiments

| Label | Script | Inputs | Matches the thesis |
|---|---|---|---|
| `tab:pseudo_quality_k` | `tab_controls_main.py` | results/pseudolabels_per_epoch.csv, results/pseudolabels_per_epoch_pseudo_label.csv | yes, 30 cells |
| `tab:pseudo_selected_epoch` | `tab_controls_main.py` | the two pseudo-label files, results/runs.csv (best_epoch) | yes, 60 cells |
| `tab:pseudo_threshold` | `tab_controls_runs.py` | results/controls_fixmatch.csv, results/pseudolabels_per_epoch.csv, common.panel() | yes except one half-width: 0.99 - ERM gives 9.93 (unrounded 9.934), thesis 9.94 (taken from the already rounded interval limits) |
| `tab:repetitions_two` | `tab_controls_runs.py` | results/controls_repetitions.npz | yes, 10 rows |
| `tab:selected_epoch_methods` | `tab_controls_main.py` | results/runs.csv | yes except Mean Teacher: MILK10k derm. 15.2 +- 12.5 (thesis 14.8 +- 12.4) and All 20.9 +- 13.4 (thesis 20.8). One run (s3-imagenet-milk10k-k16-mean_teacher) ties at epochs 13 and 23 in the logged validation BA; the thesis took 13, runs.csv has 23 (the run stopped at 38 = 23 + patience 15) |
| `tab:disagreement_runs` | `tab_controls_main.py` | common.runs(), common.predictions() | yes, 36 cells |
| `tab:shared_errors_seed` | `tab_controls_main.py` | common.runs(), common.predictions() | yes, 42 cells |
| `tab:control_ema` | `tab_controls_runs.py` | results/controls_mean_teacher_ema.csv | yes |
| `tab:control_fixmatch` | `tab_controls_runs.py` | results/controls_fixmatch.csv, common.panel() | yes |
| `tab:control_natural` | `tab_controls_runs.py` | results/controls_natural_proportions.csv | yes, with asterisks |
| `tab:control_sisters` | `tab_controls_pool.py` | results/controls_pool_sisters.csv | yes, 6 values |
| `tab:control_pool_size` | `tab_controls_pool.py` | results/controls_pool_size.csv, common.panel() | yes except one half-width: "Mean of the five methods", reduced - ERM 1.454 -> 1.45, thesis 1.46 (from the rounded interval [+0.94, +3.85]) |
| `tab:control_prior` | `tab_controls_pool.py` | results/controls_class_prior.csv, common.panel() | yes, 6 rows |
| `tab:control_prior_classes` | `tab_controls_pool.py` | results/controls_class_prior.csv, common.panel() | yes, 30 cells and asterisks |
| `tab:control_gpu` | `tab_controls_runs.py` | results/controls_gpu.csv | yes (also prints the machine means 45.08-45.59 and the repetition gap 1.76 / 3.12 of the text) |
| `tab:control_lesion` | `tab_controls_main.py` | common.runs(), common.predictions(), data/splits/bcn20000.csv (lesion_id) | yes (2.48; 2.15 +- 0.51*) |
| `tab:control_accuracy` | `tab_controls_main.py` | common.design() | yes, 8 cells and asterisks |
| `tab:control_epoch` | `tab_controls_main.py` | results/val_curves.csv, results/runs.csv | yes, 20 cells |
