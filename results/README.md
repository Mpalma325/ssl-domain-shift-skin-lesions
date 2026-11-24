# Results

Outputs of the 1020 trained models used in Chapter 4: the 990 runs of the
main design (2 initializations x 3 targets x 3 label budgets x 11 methods x
5 seeds) and 30 fully supervised references (2 x 3 x 5).

`runs.csv` is in the repository. The predictions are larger (48 MB) and are
attached to the [GitHub release](https://github.com/Mpalma325/ssl-domain-shift-skin-lesions/releases/tag/v1.0) of this repository: download
`predictions_bcn20000.npz`, `predictions_milk10k.npz` and
`predictions_milk10k_clinico.npz` into this folder.

## runs.csv

| Column | Meaning |
|---|---|
| `run_id` | `s<seed>-<init>-<target>-k<k>-<method>` |
| `init` | `ham` (ResNet-50 fine-tuned on HAM10000) or `imagenet` |
| `target` | `bcn20000`, `milk10k` (dermoscopic) or `milk10k_clinico` (clinical) |
| `k` | labeled images per class; 0 for `fully_supervised` |
| `method` | `erm`, one of the ten SSL methods, or `fully_supervised` |
| `seed` | 0 to 4; the seed also fixes the labeled subset |
| `best_epoch` | epoch selected by validation balanced accuracy |
| `row` | row of this run in `predictions_<target>.npz` |

## predictions_&lt;target&gt;.npz

| Array | Shape | Meaning |
|---|---|---|
| `logits_val`, `logits_test` | runs x images x 5 | logits of the checkpoint selected by validation |
| `labels_val`, `labels_test` | images | class index in the order akiec, bcc, bkl, mel, nv |
| `ids_val`, `ids_test` | images | ISIC image identifier (for clinical images, the identifier of the dermoscopic image of the same lesion, which is the file name used) |

Images follow the order of `torchvision.datasets.ImageFolder` (class, then file
name). The test arrays hold all 914 MILK10k images; the analysis removes the 29
listed in `data/splits/milk10k_test_excluded.txt` (see `analysis/common.py`).

## Other files

| File | Content |
|---|---|
| `source_only.npz` | test logits of the HAM10000 classifier applied to each test set without adaptation (`logits_<target>`, `labels_<target>`, `ids_<target>`, also for the HAM10000 test set) |
| `factorial_runs.csv` | the 48 runs of the labeled subset x training seed factorial (k = 16, HAM10000 init., ERM and FixMatch, MILK10k dermoscopic and BCN20000): balanced accuracy on the primary and full test and on validation |
| `pseudolabels_per_epoch.csv` | accuracy and ratio of the accepted pseudo-labels per epoch, read from the training logs of the pseudo-labelling methods (`analysis/extract_pseudolabels.py`) |
| `shift_class_conditional_mmd.csv` | class-conditional MMD² between HAM10000 and each target (mean and standard deviation over 20 seeds; `analysis/compute_shift_mmd.py`) |
| `example_images.csv` | ISIC identifiers and titles of the images shown in the dataset figures |

## Control experiments and other table inputs

`clinical_crop_review.csv`: blind review of 200 training crops of MILK10k clinical images (25 training lesions, five per class, four crops at each of the scales 0.2 and 0.5). One row per crop: `crop_id`, `derm_isic_id`, `clinical_isic_id`, `class`, `scale_min`, `crop_index` and `outcome` (`preserved`, `partial` or `lost`).

`clinical_crop_ablation_runs.csv`: ERM runs on MILK10k clinical (HAM10000 init., k = 16) of the crop ablation that are outside the main design: crop scale [0.2, 1.0] at 200 iterations per epoch (seeds 0-2) and [0.5, 1.0] at 200 iterations (seed 0). Columns `init`, `target`, `method`, `k`, `seed`, `crop_scale_min`, `iters_per_epoch`, `ba` (primary test), `ba_full_test`.

`shift_representations.csv`: distances between the image representations of HAM10000 and each target, marginal and class-balanced (`analysis/compute_shift_representations.py`).

`repetitions.npz` and `repetitions_runs.csv`: five repetitions each of the same ERM and FixMatch run (MILK10k dermoscopic, HAM10000 init., k = 16, seed 0). `repetitions.npz` holds `logits_test` (10 x 914 x 5), `labels_test` and `ids_test`; `repetitions_runs.csv` gives the `method`, `repetition`, `best_epoch` and `row` of each run. `controls_repetitions.npz` below is a different experiment.

`val_curves.csv` (1.4 MB): validation balanced accuracy (%) logged at every epoch of the 990 runs of the main design.
Columns: `run_id` (as in runs.csv), `epoch` (counted from 1, as `best_epoch`), `val_ba`. A run stops 15 epochs after its
best epoch. In 14 runs the epoch of the highest logged value differs from `best_epoch` (ties at three decimals or
logs of resumed runs); `best_epoch` is the checkpoint whose predictions are released.

`pseudolabels_per_epoch_pseudo_label.csv` (0.3 MB): the same columns as `pseudolabels_per_epoch.csv` for the 90
Pseudo-Label runs, which log only the accuracy of the pseudo-labels (`pl_ratio` is empty).

`controls_fixmatch.csv` (1 KB): FixMatch control runs on MILK10k dermoscopic, k = 16, HAM10000 init. One row per run:
`variant` (`mu7`: seven unlabeled images per labeled image, seeds 0-4; `threshold_0.99`: confidence threshold 0.99,
seeds 0-2), `init`, `target`, `k`, `method`, `seed`, `mu`, `threshold`, `best_epoch`, `ba` (primary test),
`ba_full_test`, `ba_val`, `rec_mel` (MEL recall, primary test), `pl_ratio_mean` and `pl_accuracy_mean` (mean over
epochs 2 to the last of the per-epoch share of the unlabeled batch with a pseudo-label and of correct pseudo-labels,
%), `pl_epochs`. The baselines (FixMatch with mu = 1 and threshold 0.95, and ERM) are the main-design runs.

`controls_mean_teacher_ema.csv` (0.6 KB): Mean Teacher reruns (MILK10k dermoscopic, k = 16, HAM10000 init., seeds
0-4) that also evaluated the teacher (EMA) weights. The teacher was logged only as aggregate metrics on the complete
914-image test set. Columns: `seed`, `student_best_epoch`, `teacher_best_epoch`, `student_ba_val`,
`teacher_ba_val_best`, `student_ba` (primary test), `student_ba_full_test`, `teacher_ba_full_test_at_student_epoch`,
`teacher_ba_full_test_at_teacher_epoch`.

`controls_natural_proportions.csv` (3.3 KB): ERM and FixMatch on MILK10k dermoscopic (HAM10000 init.) with 80
labeled images, `labeling` = `balanced` (16 per class) or `natural` (5 AKIEC, 44 BCC, 10 BKL, 8 MEL, 13 NV), seeds
0-9. Balanced seeds 0-4 are the main-design k = 16 runs (`source` = `main_design`); the rest are control runs.
`decision` gives the rule: natural runs take the argmax of the logits minus the log labeled prior. Columns:
`labeling`, `method`, `seed`, `source`, `decision`, `best_epoch`, `ba`, `ba_full_test`, `ba_val`, `ba_raw_argmax`
(natural runs only, plain argmax).

`controls_gpu.csv` (1.5 KB): Mean Teacher, MILK10k dermoscopic, k = 16, HAM10000 init., seeds 0-2, two runs on each
of three machines (18 runs). Columns: `machine` (GPU model: RTX 3050 Laptop, RTX 3090, RTX 5070), `os`,
`repetition`, `init`, `target`, `k`, `method`, `seed`, `best_epoch`, `ba`, `ba_full_test`, `ba_val`.

`controls_repetitions.npz` (0.35 MB): two repetitions of the same run (MILK10k dermoscopic, k = 16, HAM10000 init.,
seed 0) for ERM and nine SSL methods (no FixMatch), on one machine. Arrays: `logits_test` (20 x 914 x 5, float32,
ImageFolder order), `labels_test`, `ids_test`, `method`, `repetition`, `best_epoch`.

`controls_pool_sisters.csv` (5.7 KB, 20 rows): FixMatch and Pi-Model on BCN20000, HAM10000 init., k = 100, 200
iterations per epoch, seeds 0-4, with the complete pool (`pool` = `complete`) and without the other images of the
labeled lesions (`pool` = `without_sisters`). Columns: `init`, `target`, `k`, `method`, `pool`, `iters_per_epoch`,
`seed`, `best_epoch`, `ba` (primary test), `ba_full_test`, `ba_val`, `n_test`, `rec_<class>` (recall %, primary
test), `cm_ij` (confusion counts on the primary test, true i, predicted j, class order akiec, bcc, bkl, mel, nv).

`controls_pool_size.csv` (7.1 KB, 25 rows): FixMatch, Mean Teacher, Pseudo-Label, DebiasMatch and Self-Tuning on
BCN20000, HAM10000 init., k = 16, 500 iterations per epoch, seeds 0-4. The pool has no other images of the labeled
lesions and is reduced by whole lesions, stratified by class, to the size of the MILK10k dermoscopic pool;
`pool_size` gives the images left (3097-3107). Other columns as in `controls_pool_sisters.csv`.

`controls_class_prior.csv` (8.9 KB, 35 rows): HAM10000 init., k = 16, seeds 0-4. `method` is
`flexmatch_true_prior` (FlexMatch given the true class composition of the pool, MILK10k dermoscopic and BCN20000),
`flexmatch_rerun` (FlexMatch without warm-up, run as its comparator on the same targets and machine) or `simpro`
(the three targets). Columns as in `controls_pool_sisters.csv`.

## study2/

Outputs of Study 2 (Chapter 5 and its appendix), all in the repository. The predictions follow the layout of
`predictions_<target>.npz`; each has a runs CSV whose `row` points into it.

| File | Content |
|---|---|
| `expert_runs.csv`, `predictions_experts_<target>.npz` | ResNet-50 ERM (RE) and ConvNeXt-T ERM (CE) runs, ConvNeXt-T fully supervised, MME, S+T, L2-SP and Co-Tuning (`model`), k = 16, 32, 100 |
| `student_runs.csv`, `predictions_students_<target>.npz` | Students and the runs that continue them; `trained_on` = dermoscopic or clinical images |
| `teacher_outputs_<target>.npz` | Log-probabilities of the four-expert combination on the test set, per seed (`dermoscopic_experts`, `clinical_experts`) |
| `source_models.npz` | Logits of the HAM10000 ResNet-50 (RH) and ConvNeXt-T (CH) on every split of each target |
| `pacs_shared_errors.csv` | Shared-error statistics per pair of PACS classifiers (aggregates only) |
| `pseudolabels_student_fixmatch.csv` | Pseudo-label accuracy and ratio per epoch of FixMatch started from the first student |
| `clinical_views_<target>.npz` | Balanced log-posteriors of the HAM10000 networks reading each image: photograph, LFM, its variants, the guidance model, the degradation-tuned networks (`lp_*`), with ids, split, labels and the labeled images of each seed |
| `midas_metrics.csv`, `midas_metrics_bootstrap.csv` | Balanced accuracy on MIDAS per model, population, seed and number of LFM pairs, and bootstrap bounds; MIDAS itself is not included |
| `selection_sets/<target>_<selection>_s<set>.json` | The 80 labeled images of each set (`ids` relative to the target folder) |
| `selection_finetune_runs.csv`, `selection_finetune.npz` | Test logits of the 96 ERM networks trained for one epoch on those sets (no validation logits) |
| `selection_proxy_*.csv` | Frozen-feature proxy: balanced accuracy per draw, oracle restarts, image traits per set and the two controls (`analysis/selection.py` recomputes them) |

`data/splits/milk10k_pool_excluded.txt` lists the 95 MILK10k images removed from the pool by `--pool-exclude` in
Study 2; 85 of them are copies of HAM10000 images listed in `data/milk10k_duplicates.csv`.

Release [v1.1](https://github.com/Mpalma325/ssl-domain-shift-skin-lesions/releases/tag/v1.1) adds the frozen
features read by three analysis scripts, to be placed in `features/`: `selection_<target>.npz` (ResNet-50 features in
float16 and logits of both HAM10000 networks for every MILK10k dermoscopic or BCN20000 image) and
`clinical_views_milk10k.npz` (features of the views of each MILK10k clinical photograph and of its dermoscopic pair).

## Reading one run

```python
import numpy as np, pandas as pd
runs = pd.read_csv("results/runs.csv")
z = np.load("results/predictions_bcn20000.npz")
r = runs.query("target == 'bcn20000' and init == 'ham' and k == 16 and method == 'fixmatch' and seed == 0").row.item()
pred = z["logits_test"][r].argmax(1)
```
