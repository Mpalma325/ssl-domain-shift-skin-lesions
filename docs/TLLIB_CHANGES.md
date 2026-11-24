# Changes to TLlib

`tllib/` and `train/` are a modified copy of
[Transfer-Learning-Library](https://github.com/thuml/Transfer-Learning-Library)
by THUML, MIT license (kept in `third_party/TLLIB_LICENSE`). The base is upstream
commit `c4aa59eb565650a552b809411601d0589efbbfe4`.

Only the semi-supervised image classification trainers
(`examples/semi_supervised_learning/image_classification/`, here `train/`) and
the 30 modules of `tllib/` that they import are included. The other modules of
the package (domain adaptation, ranking, regularization, translation and the
other datasets) were removed, and the `__init__.py` files of `tllib/`,
`tllib/modules/`, `tllib/vision/datasets/` and `tllib/vision/models/` import
only what remains. To see every change, compare `tllib/` and `train/` with the base commit
(`examples/semi_supervised_learning/image_classification/` upstream).

## What changed

**Datasets.** `tllib/vision/datasets/` registers `HAM10000`, `MILK10K` and
`BCN20000`, read as `ImageFolder` trees with lesion-level splits
(see `data/splits/`).

**Logger.** `CompleteLogger.close()` restores the console and closes only the
log file. Upstream it also closed the console, so every run ended with exit
code 120 although its outputs were complete.

**New trainer.** `train/simpro.py` ports SimPro (Du et al., ICML 2024; official
code at https://github.com/LeapLabTHU/SimPro) to the FixMatch recipe of the other
trainers. It is used only in the class-prior control of the appendix; its
docstring lists the differences from the official code.

**Common additions to the eleven trainers** (`erm`, `pi_model`, `mean_teacher`,
`debiasmatch`, `fixmatch`, `noisy_student`, `pseudo_label`, `uda`, `flexmatch`,
`self_tuning`, `dst`):

- resumable checkpoints (`latest.pth`, written atomically; `--resume`);
- early stopping on validation balanced accuracy (`--patience`);
- intermediate initialization with `--pretrained-backbone`, which loads the
  ResNet-50 fine-tuned on HAM10000 (the `ham` arm of the study).

**`train/utils.py`** adds atomic saving, the `lesion_preserving` training crop
(`RandomResizedCrop` with `scale=(0.5, 1.0)`, used for the clinical images) and
the labeled/unlabeled split by class (`x_u_split`).

**Noisy Student** (`train/noisy_student.py`) loads a trained teacher with
`--pretrained-teacher`. In the thesis the teacher is the ERM run of the same
seed, initialization, target and label budget. For Study 2 it also takes
other teachers (source networks, a product of calibrated experts, up to four
experts, a ConvNeXt-T) and the pairing controls of the clinical student, all
off by default (see the docstring and `scripts/grid_study2.py`).

**Corrections after an audit of the methods**: DebiasMatch, the loading of the
Noisy Student teacher, the FlexMatch warm-up and the Mean Teacher EMA.
Deviations from the original papers are listed in Chapter 3 of the thesis.

**Compatibility fixes** with recent NumPy, PyTorch and torchvision, without
which the library no longer imports:

| File | Problem |
|---|---|
| `tllib/modules/grl.py` | `np.float` was removed in NumPy 1.24 |
| `tllib/vision/datasets/imagelist.py` | `T_co` is no longer exported by `torch.utils.data.dataset` |
| `tllib/vision/models/resnet.py` | `model_urls` was removed in torchvision 0.13; the `Weights` API is used |
| `tllib/vision/models/reid/resnet.py` | `load_state_dict_from_url` moved to `torch.hub` |

**Optional flags.** The trainers also contain flags used in exploratory or
control experiments (for example `--mu`, `--ema-eval`, `--pool-mode`). All are
off by default: the commands in `scripts/grid.py` do not use them, and without
them each trainer behaves as described above.
