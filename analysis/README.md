# Analysis

Scripts that regenerate the data figures and tables of the thesis from
`results/` and `data/`. Figures come from `fig_*.py` (below); tables come from
`tab_*.py`, listed by thesis label in `TABLES.md`. Run them from the repository root, for example
`python analysis/fig_ssl_heatmap.py`. Each writes the figure to `figures/`
(PDF, plus a PNG preview) and the plotted numbers to `tables/`. They need the
predictions of the release in `results/` (see `results/README.md`) and run on
a CPU in seconds.

| Output in `figures/` | Script | Inputs | File name in the thesis |
|---|---|---|---|
| `class_distributions` | `fig_class_distributions.py` | `data/splits/` | `fig_classdist_all` |
| `ham10000_examples`, `bcn20000_examples`, `milk10k_examples_a`, `_b` | `fig_example_images.py` | `results/example_images.csv` and the images (see below) | `fig_ham10000_examples`, `fig_bcn20000_top10`, `fig_milk10k_top10_a`, `_b` |
| `shift_mmd_class` | `fig_shift_mmd.py` | `results/shift_class_conditional_mmd.csv` | `e1_shift_mmd_clase` |
| `erm_by_k` | `fig_erm_by_k.py` | predictions, `results/source_only.npz` | `e1_erm_por_k` |
| `factorial_boxes` | `fig_factorial_boxes.py` | `results/factorial_runs.csv` | `e1_factorial_cajas` |
| `ssl_heatmap`, `ssl_ba_heatmap` | `fig_ssl_heatmap.py` | predictions | `e1_ssl_heatmap`, `e1_ssl_ba_heatmap` |
| `ssl_families` | `fig_ssl_families.py` | predictions | `e1_ssl_familias` |
| `recall_by_class`, `_imagenet` | `fig_recall_by_class.py` | predictions | `e1_recall_clase_target`, `_imagenet` |
| `mel_sensitivity` | `fig_mel_sensitivity.py` | predictions | `e1_mel_sens80` |
| `confidence`, `_imagenet` | `fig_confidence.py` | predictions | `e1_confianza`, `_imagenet` |
| `pseudolabels_areas`, `pseudolabels_quality` (and `_imagenet`) | `fig_pseudolabels.py` | `results/pseudolabels_per_epoch.csv` | `e1_pseudoetiquetas_areas`, `e1_pseudoetiquetas_calidad` (and `_imagenet`) |

`common.py` holds the data loading, the metrics on the primary test set (MILK10k
without the 29 copies of HAM10000 images) and the figure style.

Two scripts rebuild intermediate results from your own runs or images:

- `extract_pseudolabels.py` reads the training logs in `outputs/` and writes
  the pseudo-label accuracy and ratio per epoch.
- `compute_shift_mmd.py` extracts features with the HAM10000 checkpoint and
  computes the class-conditional MMD between HAM10000 and each target.
- `compute_shift_representations.py` computes the marginal and class-balanced
  distances between the representations of HAM10000 and each target.

`fig_example_images.py` shows some images outside the five classes or removed
from the splits (vascular lesions, dermatofibroma, excluded categories). Pass
the folders of the official downloads with `--images`.
