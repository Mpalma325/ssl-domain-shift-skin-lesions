# Semi-supervised learning under domain shift and label scarcity in skin lesion classification

Code for the master's thesis *Characterizing Semi-Supervised Learning under
Domain Shift and Label Scarcity in Skin Lesion Classification* (Marcelo Palma,
Universidad de Chile).

The study compares ERM with ten semi-supervised methods (Pi-Model, Mean
Teacher, DebiasMatch, FixMatch, Noisy Student, Pseudo-Label, UDA, FlexMatch,
Self-Tuning and DST) on two target datasets, BCN20000 and MILK10k (dermoscopic
and clinical images of the same lesions), with 16, 32 and 100 labeled images
per class. Each model starts either from ImageNet or from a ResNet-50 already
fine-tuned on HAM10000.

Study 2 reuses the HAM10000 networks: a combination of four experts (ResNet-50
and ConvNeXt-T trained on HAM10000 or on the 80 labeled target images),
students distilled from it, ways to read MILK10k clinical photographs through
the dermoscopic networks (LFM, the guidance model of Mallya and Hamarneh, a
network tuned on simulated camera degradations) and the selection of the
labeled images.

## Contents

| Path | What it is |
|---|---|
| `train/` | The eleven trainers (`erm.py` and one per SSL method), `simpro.py` (used only in a control experiment), the Study 2 trainers and their shared `utils.py` |
| `tllib/` | Modified copy of [TLlib](https://github.com/thuml/Transfer-Learning-Library); changes in `docs/TLLIB_CHANGES.md` |
| `data/` | Lesion-level splits with ISIC image identifiers (no images), see `data/README.md` |
| `scripts/prepare_data.py` | Builds the image folders from the official downloads |
| `scripts/verify_setup.py` | Checks versions, data and labeled subsets against the thesis |
| `scripts/grid.py` | The experimental grid and the exact command of every run |
| `scripts/grid_study2.py` | The commands of the Study 2 networks trained with `train/` |
| `scripts/extract_features.py`, `scripts/midas_inference.py` | Frozen features of the HAM10000 networks; inference on a local copy of MIDAS |
| `results/` | Predictions of the 1020 trained models of Study 1, those of Study 2 (`results/study2/`) and the smaller result tables, see `results/README.md` |
| `analysis/` | Scripts behind every data figure and table of the thesis, see `analysis/README.md` and `analysis/TABLES.md` |

## Reproducing the figures and tables

No GPU needed. Download the three `predictions_*.npz` files of the
[release](https://github.com/Mpalma325/ssl-domain-shift-skin-lesions/releases/tag/v1.0) into `results/`
and run any script in `analysis/`, for example:

```bash
python analysis/fig_ssl_heatmap.py
python analysis/tab_ssl_overall.py
```

Figures go to `figures/` and tables to `tables/`, as CSV.

Study 2 reads `results/study2/` (in the repository). Three scripts also need the
frozen features of release
[v1.1](https://github.com/Mpalma325/ssl-domain-shift-skin-lesions/releases/tag/v1.1)
in `features/`: `selection.py --recompute` (`selection_<target>.npz`),
`compute_lfm.py` and `fig_lda_clinical.py` (`clinical_views_milk10k.npz`).

The analysis also needs `pandas` (3.0.1 was used).

## Setup

Python 3.12 and a CUDA GPU.

```bash
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Use the PyTorch index that matches your CUDA version (`cu118`, `cu126`,
`cu128`). Other versions of torchvision or numpy change the augmentation and
the labeled draw, so results would shift.

## Data

Download HAM10000, BCN20000 and MILK10k (links and licenses in
`data/README.md`), then:

```bash
python scripts/prepare_data.py --ham <HAM10000 images> --bcn <BCN20000 images> --milk <MILK10k images>
python scripts/verify_setup.py
```

`verify_setup.py` hashes every split and the labeled subset that each seed
draws for k = 16, 32 and 100. If they match, every seed selects the same
labeled images as in the thesis. With `--checkpoint` it also applies the
HAM10000 classifier to each target test set and compares its balanced
accuracy with the thesis.

## Training

The HAM10000 classifier is the starting point of the `ham` arm. Download
`ham10000_source.pth` from the [release](https://github.com/Mpalma325/ssl-domain-shift-skin-lesions/releases/tag/v1.0) into `checkpoints/`, or train it:

```bash
python scripts/grid.py run-source
```

One run of the grid (seed, initialization, target, k, method):

```bash
python scripts/grid.py show 0 ham milk10k 16 fixmatch   # print the command
python scripts/grid.py run  0 ham milk10k 16 fixmatch   # run it
python scripts/grid.py list --seeds 0-4                 # all 990 commands
```

Noisy Student uses the ERM run with the same seed, initialization, target and
k as its teacher, so that ERM run has to finish first. Fully supervised
references:

```bash
python scripts/grid.py run-fully-supervised 0 ham bcn20000
```

Each run writes its log and the best checkpoint by validation balanced accuracy
to `outputs/<init>/<target>/k<k>/<method>/s<seed>/`.

### Control experiments

The control experiments of the appendix are not in the grid. Each one is the command printed by `python scripts/grid.py show ...` with the flags below added or changed (same trainer unless stated; `<s>` is the seed). Their results are in `results/` (file in parentheses).

- **FixMatch, mu = 7** (`controls_fixmatch.csv`, `mu7`, seeds 0-4): `grid.py show <s> ham milk10k 16 fixmatch` plus `--mu 7`.
- **FixMatch, threshold 0.99** (`controls_fixmatch.csv`, `threshold_0.99`, seeds 0-2): `grid.py show <s> ham milk10k 16 fixmatch` with `--threshold 0.99`.
- **Mean Teacher with the EMA teacher evaluated** (`controls_mean_teacher_ema.csv`): `grid.py show <s> ham milk10k 16 mean_teacher` plus `--evaluar-teacher` (training is unchanged).
- **Natural labeled proportions** (`controls_natural_proportions.csv`): `grid.py show <s> ham milk10k 16 erm` or `... fixmatch` plus `--label-sampling natural --num-labeled-total 80 --logit-adjust-eval labeled`. The balanced runs with seeds 5-9 are the unchanged command with `<s>` = 5-9.
- **Pool without sisters** (`controls_pool_sisters.csv`): `grid.py show <s> ham bcn20000 100 fixmatch` or `... pi_model` with `--iters-per-epoch 200`, plus `--pool-mode sin_hermanas` for the `without_sisters` arm (the `complete` arm keeps the default `--pool-mode orig`).
- **Pool reduced to the MILK10k size** (`controls_pool_size.csv`): `grid.py show <s> ham bcn20000 16 <method>` (fixmatch, mean_teacher, pseudo_label, debiasmatch, self_tuning) plus `--pool-mode tamano_milk`.
- **FlexMatch given the true pool prior** (`controls_class_prior.csv`): `grid.py show <s> ham <milk10k|bcn20000> 16 flexmatch` with `--warmup false` plus `--pool-prior oracle` (the prior is computed from the pool labels; no input file). `flexmatch_rerun` is the same command with `--warmup false` only.
- **SimPro** (`controls_class_prior.csv`): `grid.py show <s> ham <target> 16 fixmatch` run with `simpro.py` instead of `fixmatch.py` and without `--trade-off-cls-strong 0.1 --trade-off-self-training 1`; SimPro's own flags stay at their defaults (`--tau 2.0 --ema-u 0.9 --alpha 1.0`).
- **Clinical crop ablation** (`clinical_crop_ablation_runs.csv`): `grid.py show <s> ham milk10k_clinico 16 erm` with `--iters-per-epoch 200`; for crop scale [0.2, 1.0] also drop `--train-resizing lesion_preserving`.
- **Labeled subset x training seed** (`factorial_runs.csv`): `grid.py show <t> ham <milk10k|bcn20000> 16 <erm|fixmatch>` plus `--subset-seed <s>`, where `--subset-seed` (0-2) picks the labeled images and `--seed` (0-2) sets the training seed; the 12 `replica` = 1 runs repeat the same command.
- **Repetitions and machines** (`repetitions.npz`, `controls_repetitions.npz`, `controls_gpu.csv`): the unchanged `grid.py show` command (seed 0, or seeds 0-2 for `controls_gpu.csv`) run again, on the same machine or on other machines.

### Study 2

The runs trained with `train/` (ERM and students, ConvNeXt-T ERM, L2-SP,
Co-Tuning, clinical students, label lists) are listed by `scripts/grid_study2.py`,
with the same interface as `grid.py` and outputs in
`outputs/study2/<name>/<target>/k<k>/s<seed>/`:

```bash
python scripts/grid_study2.py list                       # all 365 commands
python scripts/grid_study2.py show 0 milk10k 16 convnext_student
python scripts/grid_study2.py run  0 milk10k 16 erm_teacher
```

They need `checkpoints/ham10000_convnext.pth`, the HAM10000 ConvNeXt-T, and some
take an earlier run as teacher or start (`erm_teacher` and `convnext_erm`
first). The other networks have their own trainers:

```bash
python train/convnext_source.py                                            # HAM10000 ConvNeXt-T
cd train
python mme.py --target milk10k --seed 0                                    # MME
python source_target.py --target milk10k --seed 0                          # S+T
python short_finetune.py --target milk10k --labeled-list ../results/study2/selection_sets/milk10k_*_s*.json
cd ..
python train/camera_simulator.py --version 2 --ham checkpoints/ham10000_source.pth --out outputs/camera_simulator_v2
python train/robust_finetune.py --simulator 2 --mode degraded --epochs 2 --out outputs/robust_v2
python train/feature_translator.py --seed 1 --out features/translator_s1.npz
python train/mallya.py latents --seed 0 && python train/mallya.py guided --seed 0
```

The docstring of each script gives the thesis settings. LFM itself is a ridge
regression on frozen features (`analysis/compute_lfm.py`, from
`scripts/extract_features.py`). The labeled sets of the selection study are in
`results/study2/selection_sets/`; the selection code that chose them is not
included.

MIDAS (external validation) is not redistributed. `scripts/midas_inference.py`
applies the trained networks to a local download; the repository keeps only the
aggregated metrics (`results/study2/midas_metrics*.csv`).

## License

MIT (`LICENSE`). `tllib/` and `train/` derive from TLlib, also MIT
(`third_party/TLLIB_LICENSE`). The datasets keep their own licenses.
