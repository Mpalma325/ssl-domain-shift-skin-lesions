"""
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataset import Subset, ConcatDataset
import torchvision.transforms as T
import timm
from timm.data.auto_augment import auto_augment_transform, rand_augment_transform

import pool_modes

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, where tllib/ lives
from tllib.modules.classifier import Classifier
import tllib.vision.datasets as datasets
import tllib.vision.models as models
from tllib.vision.transforms import ResizeImage
from tllib.utils.metric import accuracy, ConfusionMatrix
from tllib.utils.meter import AverageMeter, ProgressMeter


def atomic_torch_save(obj, path):
    """Write via a temp file so an interruption never destroys the previous checkpoint."""
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def get_model_names():
    return sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    ) + [CONVNEXT_HAM_ARCH] + timm.list_models()


def get_model(model_name, pretrained=True, pretrained_checkpoint=None):
    if model_name == CONVNEXT_HAM_ARCH:
        return get_convnext_ham(pretrained, pretrained_checkpoint)
    if model_name in models.__dict__:
        backbone = models.__dict__[model_name](pretrained=pretrained)
    else:
        backbone = timm.create_model(model_name, pretrained=pretrained)
        try:
            backbone.out_features = backbone.get_classifier().in_features
            backbone.reset_classifier(0, '')
        except:
            backbone.out_features = backbone.head.in_features
            backbone.head = nn.Identity()
    if pretrained_checkpoint:
        print("=> loading pre-trained model from '{}'".format(pretrained_checkpoint))
        if torch.cuda.is_available():
            _map_loc = "cuda"
        elif torch.backends.mps.is_available():
            _map_loc = "mps"
        else:
            _map_loc = "cpu"
        pretrained_dict = torch.load(pretrained_checkpoint, map_location=_map_loc)
        # oracle checkpoints wrap state_dict and prefix keys with 'backbone.'
        if isinstance(pretrained_dict, dict) and "state_dict" in pretrained_dict:
            pretrained_dict = pretrained_dict["state_dict"]
        pretrained_dict = {k[len("backbone."):]: v for k, v in pretrained_dict.items()
                           if k.startswith("backbone.")}
        result = backbone.load_state_dict(pretrained_dict, strict=False)
        print("   [backbone] {} params loaded | {} missing | {} unexpected".format(
            len(pretrained_dict), len(result.missing_keys), len(result.unexpected_keys)))
        assert len(pretrained_dict) > 0 and len(result.missing_keys) == 0, \
            "FAILED to load the pretrained backbone (check the checkpoint format)"
    return backbone


# Stochastic depth stays at torchvision's default (0.1), the value used to train the HAM model.
CONVNEXT_HAM_ARCH = 'convnext_tiny_ham'
CONVNEXT_HAM_KEYS = ('features', 'norm')


class ConvNeXtTinyHAM(nn.Module):
    """ConvNeXt-T without head, laid out like the HAM ConvNeXt; returns (N, 768, 1, 1)."""

    def __init__(self):
        super(ConvNeXtTinyHAM, self).__init__()
        import torchvision
        try:
            m = torchvision.models.convnext_tiny(weights=None)
        except TypeError:  # torchvision < 0.13
            m = torchvision.models.convnext_tiny(pretrained=False)
        self.features, self.avgpool = m.features, m.avgpool
        self.norm = m.classifier[0]
        self.out_features = m.classifier[2].in_features

    def forward(self, x):
        return self.norm(self.avgpool(self.features(x)))


def get_convnext_ham(pretrained=True, pretrained_checkpoint=None):
    """Loads features.* and norm.* from the HAM ConvNeXt best.pth; the HAM head is dropped."""
    backbone = ConvNeXtTinyHAM()
    if not pretrained_checkpoint:
        if pretrained:
            raise ValueError("-a {} is initialized from the HAM ConvNeXt: --pretrained-backbone with its best.pth is missing "
                             "(it has no ImageNet weights of its own)".format(CONVNEXT_HAM_ARCH))
        return backbone
    print("=> loading pre-trained model from '{}'".format(pretrained_checkpoint))
    checkpoint = torch.load(pretrained_checkpoint, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    subset = {k: v for k, v in checkpoint.items() if k.split(".", 1)[0] in CONVNEXT_HAM_KEYS}
    if not subset:
        raise ValueError("{} lacks the features.* and norm.* keys of the HAM ConvNeXt (-a {})".format(
            pretrained_checkpoint, CONVNEXT_HAM_ARCH))
    backbone.load_state_dict(subset)
    print("   [backbone] {} params loaded (features.* and norm.*) | {} keys left out (the HAM head)".format(
        len(subset), len(checkpoint) - len(subset)))
    return backbone


def get_dataset_names():
    return sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    )


def get_dataset(dataset_name, num_samples_per_class, root, labeled_train_transform, val_transform,
                unlabeled_train_transform=None, seed=0, pool_mode='orig', subset_seed=None,
                val_per_class=None, label_sampling='balanced', num_labeled_total=None,
                pool_exclude=None, labeled_list=None):
    # subset_seed decouples which images get labeled from the training randomness.
    seed_split = seed if subset_seed is None else subset_seed
    validar_muestreo(label_sampling, num_labeled_total, pool_mode)
    if unlabeled_train_transform is None:
        unlabeled_train_transform = labeled_train_transform
    assert pool_mode == 'orig' or dataset_name == 'BCN20000', \
        "pool-mode {} is only defined for BCN20000".format(pool_mode)
    assert pool_exclude is None or dataset_name != 'OxfordFlowers102', \
        "--pool-exclude is not defined for OxfordFlowers102"
    assert labeled_list is None or (dataset_name != 'OxfordFlowers102' and pool_mode == 'orig'
                                    and label_sampling == 'balanced' and num_labeled_total is None), \
        "--labeled-list cannot be combined with OxfordFlowers102, --pool-mode or --label-sampling natural"

    if dataset_name == 'OxfordFlowers102':
        dataset = datasets.__dict__[dataset_name]
        base_dataset = dataset(root=root, split='train', transform=labeled_train_transform, download=True)
        labeled_idxes, unlabeled_idxes = x_u_split(num_samples_per_class, base_dataset.num_classes,
                                                   base_dataset.targets, seed=seed_split)
        if label_sampling != 'balanced':
            labeled_idxes, unlabeled_idxes = muestreo_natural(num_labeled_total, base_dataset, seed_split)
        labeled_train_dataset = Subset(base_dataset, labeled_idxes)
        labeled_train_dataset.num_classes = base_dataset.num_classes
        base_dataset = dataset(root=root, split='train', transform=unlabeled_train_transform, download=True)
        unlabeled_train_dataset = ConcatDataset([
            Subset(base_dataset, unlabeled_idxes),
            dataset(root=root, split='validation', download=True, transform=unlabeled_train_transform)
        ])
        val_dataset = dataset(root=root, split='val', download=True, transform=val_transform)
    else:
        dataset = datasets.__dict__[dataset_name]
        base_dataset = dataset(root=root, split='train', transform=labeled_train_transform, download=True)
        labeled_idxes, unlabeled_idxes = x_u_split(num_samples_per_class, base_dataset.num_classes,
                                                   base_dataset.targets, seed=seed_split)
        if label_sampling != 'balanced':
            labeled_idxes, unlabeled_idxes = muestreo_natural(num_labeled_total, base_dataset, seed_split)
        if labeled_list is not None:
            labeled_idxes, unlabeled_idxes, reporte_lista = indices_de_lista(base_dataset, root, labeled_list)
        if pool_mode != 'orig':
            lesions = pool_modes.lesion_ids(base_dataset.samples, root)
            unlabeled_idxes, rep = pool_modes.select_pool(
                pool_mode, labeled_idxes, unlabeled_idxes, base_dataset.targets, lesions, seed,
                base_dataset.num_classes, num_samples_per_class)
            print(pool_modes.format_report(rep, base_dataset.classes))
        labeled_train_dataset = Subset(base_dataset, labeled_idxes)
        labeled_train_dataset.num_classes = base_dataset.num_classes
        if labeled_list is not None:
            labeled_train_dataset.labeled_list = reporte_lista
        base_dataset = dataset(root=root, split='train', transform=unlabeled_train_transform, download=True)
        unlabeled_train_dataset = Subset(base_dataset, unlabeled_idxes)
        if pool_exclude is not None:
            unlabeled_train_dataset = excluir_del_pool(unlabeled_train_dataset, labeled_idxes, pool_exclude)
        val_dataset = dataset(root=root, split='val', download=True, transform=val_transform)
    if val_per_class is not None:
        val_dataset = submuestrear_validacion(val_dataset, val_per_class, seed)
    return labeled_train_dataset, unlabeled_train_dataset, val_dataset


def submuestrear_validacion(val_dataset, n_por_clase, seed):
    """N val images per class with its own RNG, so every arm of a seed validates on the same images (realistic label budget)."""
    targets = np.asarray(val_dataset.targets)
    rng = np.random.RandomState(10_000 + seed)
    elegidos = []
    for c in range(val_dataset.num_classes):
        de_c = np.flatnonzero(targets == c)
        elegidos.extend(rng.choice(de_c, min(n_por_clase, len(de_c)), replace=False).tolist())
    sub = Subset(val_dataset, sorted(elegidos))
    sub.num_classes = val_dataset.num_classes
    sub.classes = val_dataset.classes
    return sub


def guardar_ids_validacion(val_dataset, root, log_dir):
    """Writes <log>/val_ids.json; no-op without subsampling."""
    return _guardar_ids(val_dataset, root, log_dir, 'val_ids.json')


def guardar_ids_etiquetados(labeled_train_dataset, root, log_dir):
    """Writes <log>/labeled_ids.json (root-relative paths, x_u_split order); no-op for the oracle."""
    return _guardar_ids(labeled_train_dataset, root, log_dir, 'labeled_ids.json')


def _guardar_ids(subset, root, log_dir, nombre):
    import json
    if not isinstance(subset, Subset):
        return None
    base = subset.dataset
    if not hasattr(base, 'samples'):
        return None
    raiz = os.path.abspath(root)
    ids = [os.path.relpath(base.samples[i][0], raiz).replace(os.sep, '/')
           for i in subset.indices]
    ruta = os.path.join(log_dir, nombre)
    with open(ruta, 'w', encoding='utf-8') as f:
        json.dump({'n': len(ids), 'ids': ids}, f)
    return ruta


def x_u_split(num_samples_per_class, num_classes, labels, seed):
    """Class-balanced labeled subset and the rest as unlabeled; deterministic for a given seed."""
    labels = np.array(labels)
    assert num_samples_per_class * num_classes <= len(labels)
    random_state = np.random.RandomState(seed)

    # shuffle + slice keeps the sets nested: 16 ⊂ 32 ⊂ 100
    labeled_idxes = []
    for i in range(num_classes):
        ith_class_idxes = np.where(labels == i)[0]
        random_state.shuffle(ith_class_idxes)
        labeled_idxes.extend(ith_class_idxes[:num_samples_per_class])

    unlabeled_idxes = [i for i in range(len(labels)) if i not in labeled_idxes]
    return labeled_idxes, unlabeled_idxes


# Natural sampling reuses x_u_split's per-class permutation so natural and balanced sets of a seed share images
# (paired contrasts) and natural sets stay nested in N. Logit adjustment follows Menon et al., ICLR 2021.
LABEL_SAMPLING_MODES = ('balanced', 'natural')
LOGIT_ADJUST_MODES = ('none', 'labeled')
SOPORTE_DEFAULTS = {'label_sampling': 'balanced', 'num_labeled_total': None, 'logit_adjust_eval': 'none',
                    'balanced_ce': False}
SOPORTE_JSON = 'label_sampling.json'
SOPORTE_TAG = 'LABEL_SAMPLING'
DECISION_TAG = 'LOGIT_ADJUST_EVAL'


def add_label_sampling_args(parser, balanced_ce=False):
    """default=SUPPRESS keeps the first log line (print(args)) unchanged when unused; read with soporte_opt()."""
    group = parser.add_argument_group('labeled support (default: k per class)')
    group.add_argument('--label-sampling', choices=LABEL_SAMPLING_MODES, default=argparse.SUPPRESS,
                       help='balanced (default): k images per class (x_u_split). natural: --num-labeled-total '
                            'images with the train prevalence, stratified by class (Webster proportional '
                            'allocation, at least 1 per class), with its own RandomState from the split seed '
                            '(does not consume the training RNG)')
    group.add_argument('--num-labeled-total', type=int, default=argparse.SUPPRESS, metavar='N',
                       help='required with --label-sampling natural: total number of labeled images')
    group.add_argument('--logit-adjust-eval', choices=LOGIT_ADJUST_MODES, default=argparse.SUPPRESS,
                       help='none (default): decide with argmax z. labeled: early-stopping validation and the '
                            'final test evaluation decide with argmax(z - log pi_l), pi_l = labeled-set '
                            'frequencies; the model logits are unchanged')
    if balanced_ce:
        group.add_argument('--balanced-ce', action='store_true', default=argparse.SUPPRESS,
                           help='CE on the labeled set with weights N / (K x n_c), inverse to each class frequency '
                                '(all 1 with k per class). Cannot be combined with --logit-adjust-eval labeled')
    return parser


def soporte_opt(args, name):
    return getattr(args, name, SOPORTE_DEFAULTS[name])


def soporte_activo(args):
    return any(soporte_opt(args, name) != default for name, default in SOPORTE_DEFAULTS.items())


def _errores_muestreo(label_sampling, num_labeled_total, pool_mode='orig'):
    errores = []
    if label_sampling not in LABEL_SAMPLING_MODES:
        errores.append('--label-sampling {!r} is not valid ({})'.format(label_sampling, ', '.join(LABEL_SAMPLING_MODES)))
    if label_sampling == 'natural' and num_labeled_total is None:
        errores.append('--label-sampling natural requires --num-labeled-total N')
    if num_labeled_total is not None:
        if label_sampling != 'natural':
            errores.append('--num-labeled-total is only used with --label-sampling natural')
        if not isinstance(num_labeled_total, (int, np.integer)) or isinstance(num_labeled_total, bool) \
                or num_labeled_total < 1:
            errores.append('--num-labeled-total must be an integer >= 1 (got {!r})'.format(num_labeled_total))
    if label_sampling == 'natural' and pool_mode != 'orig':
        errores.append('--pool-mode {} assumes k per class: it cannot be combined with --label-sampling natural'.format(pool_mode))
    return errores


def kw_muestreo(args):
    """{} with the defaults, so the get_dataset call is unchanged (tests spy on it)."""
    modo, n_total = soporte_opt(args, 'label_sampling'), soporte_opt(args, 'num_labeled_total')
    if modo == SOPORTE_DEFAULTS['label_sampling'] and n_total is None:
        return {}
    return {'label_sampling': modo, 'num_labeled_total': n_total}


def kw_eval(log_prior):
    return {} if log_prior is None else {'log_prior': log_prior}


def kw_ce(class_weight):
    return {} if class_weight is None else {'class_weight': class_weight}


def validar_muestreo(label_sampling, num_labeled_total, pool_mode='orig'):
    errores = _errores_muestreo(label_sampling, num_labeled_total, pool_mode)
    if errores:
        raise ValueError('labeled support: ' + '; '.join(errores))


def validar_soporte(args):
    """Fails early (before seeding and loading data) on invalid labeled-support flag combinations."""
    if not soporte_activo(args):
        return
    modo = soporte_opt(args, 'label_sampling')
    ajuste = soporte_opt(args, 'logit_adjust_eval')
    errores = _errores_muestreo(modo, soporte_opt(args, 'num_labeled_total'), getattr(args, 'pool_mode', 'orig'))
    if ajuste not in LOGIT_ADJUST_MODES:
        errores.append('--logit-adjust-eval {!r} is not valid ({})'.format(ajuste, ', '.join(LOGIT_ADJUST_MODES)))
    if soporte_opt(args, 'balanced_ce') and ajuste == 'labeled':
        errores.append('--balanced-ce and --logit-adjust-eval labeled correct the labeled prior twice: '
                       'use only one of them')
    if getattr(args, 'oracle', False):
        errores.append('the labeled-support flags do not apply with --oracle (the whole train set is labeled)')
    if modo == 'natural' and getattr(args, 'source_checkpoint', None):
        errores.append('the teacher with a source (--source-checkpoint) is not validated with --label-sampling natural: '
                       'the ERM teacher carries log pi_l in its logits and the source does not')
    if errores:
        raise ValueError('labeled support: ' + '; '.join(errores))
    if modo == 'natural' and ajuste == 'none' and not soporte_opt(args, 'balanced_ce'):
        print('WARNING label-sampling: natural support with the argmax z decision. Early stopping and the reported BA '
              'are biased towards the frequent labeled classes; for BA use --logit-adjust-eval labeled')


def asignacion_proporcional(total, conteos):
    """Webster / Sainte-Lague allocation: sequential, so raising the total never takes images from a class (nested sets)."""
    conteos = np.asarray(conteos)
    if conteos.ndim != 1 or conteos.size == 0 or not np.issubdtype(conteos.dtype, np.integer) \
            or np.any(conteos < 0) or conteos.sum() == 0:
        raise ValueError('invalid per-class counts: {}'.format(conteos.tolist()))
    conteos = conteos.astype(np.int64)
    presentes = conteos > 0
    total = int(total)
    if not int(presentes.sum()) <= total <= int(conteos.sum()):
        raise ValueError('--num-labeled-total {} outside [{}, {}]: at least 1 image per present class is needed and '
                         'no more than the train set'.format(total, int(presentes.sum()), int(conteos.sum())))
    pi = conteos / conteos.sum()
    asignacion = presentes.astype(np.int64)
    for _ in range(total - int(asignacion.sum())):
        prioridad = np.where(asignacion < conteos, pi / (asignacion + 0.5), -np.inf)
        asignacion[int(np.argmax(prioridad))] += 1
    return asignacion


def x_u_split_natural(num_labeled_total, num_classes, labels, seed):
    """Like x_u_split, with the same per-class permutation, but n_c from asignacion_proporcional."""
    labels = np.array(labels)
    conteos = np.bincount(labels, minlength=num_classes)
    if conteos.size != num_classes or (labels.size and labels.min() < 0):
        raise ValueError('labels outside [0, {})'.format(num_classes))
    asignacion = asignacion_proporcional(num_labeled_total, conteos)
    random_state = np.random.RandomState(seed)
    labeled_idxes = []
    for i in range(num_classes):
        ith_class_idxes = np.where(labels == i)[0]
        random_state.shuffle(ith_class_idxes)
        labeled_idxes.extend(ith_class_idxes[:asignacion[i]])
    elegidas = {int(j) for j in labeled_idxes}
    unlabeled_idxes = [i for i in range(len(labels)) if i not in elegidas]
    return labeled_idxes, unlabeled_idxes


def muestreo_natural(num_labeled_total, base_dataset, seed):
    num_classes = base_dataset.num_classes
    labeled_idxes, unlabeled_idxes = x_u_split_natural(num_labeled_total, num_classes, base_dataset.targets, seed)
    targets = np.asarray(base_dataset.targets)
    print('[label-sampling] natural | N = {} | draw seed {} | train {} | labeled {}'.format(
        num_labeled_total, seed, np.bincount(targets, minlength=num_classes).tolist(),
        np.bincount(targets[np.asarray(labeled_idxes, dtype=np.int64)], minlength=num_classes).tolist()))
    return labeled_idxes, unlabeled_idxes


def conteos_etiquetados(labeled_train_dataset, num_classes):
    if not isinstance(labeled_train_dataset, Subset) or not hasattr(labeled_train_dataset.dataset, 'targets'):
        raise TypeError('labeled support needs the Subset returned by get_dataset')
    targets = np.asarray(labeled_train_dataset.dataset.targets)
    return np.bincount(targets[np.asarray(labeled_train_dataset.indices, dtype=np.int64)], minlength=num_classes)


def soporte_etiquetado(args, labeled_train_dataset, log_dir, device=None):
    """(log_pi, pesos, firma) and <log>/label_sampling.json; (None, None, None) and no output without flags."""
    if not soporte_activo(args):
        return None, None, None
    num_classes = labeled_train_dataset.num_classes
    n = conteos_etiquetados(labeled_train_dataset, num_classes)
    total = int(n.sum())
    ajuste = soporte_opt(args, 'logit_adjust_eval')
    log_pi = pesos = None
    if ajuste == 'labeled':
        if np.any(n == 0):
            raise ValueError('--logit-adjust-eval labeled with classes without labels: {}'.format(n.tolist()))
        log_pi = np.log(n / total)
    if soporte_opt(args, 'balanced_ce'):
        presentes = n > 0
        pesos = np.where(presentes, total / (presentes.sum() * np.maximum(n, 1)), 0.0)
    indices = np.sort(np.asarray(labeled_train_dataset.indices, dtype=np.int64))
    firma = {'label_sampling': soporte_opt(args, 'label_sampling'), 'labeled_counts': n.tolist(),
             'labeled_sha256': hashlib.sha256(indices.tobytes()).hexdigest(),
             'logit_adjust_eval': ajuste, 'balanced_ce': bool(soporte_opt(args, 'balanced_ce'))}
    clases = getattr(labeled_train_dataset.dataset, 'classes', None)
    semilla = getattr(args, 'subset_seed', None)
    registro = dict(firma, version=1, num_labeled_total=total,
                    num_samples_per_class=getattr(args, 'num_samples_per_class', None),
                    seed=getattr(args, 'seed', None), subset_seed=semilla,
                    split_seed=getattr(args, 'seed', None) if semilla is None else semilla,
                    classes=None if clases is None else [str(c) for c in clases],
                    train_counts=np.bincount(np.asarray(labeled_train_dataset.dataset.targets),
                                             minlength=num_classes).tolist(),
                    labeled_prior=(n / total).tolist(),
                    log_prior_eval=None if log_pi is None else log_pi.tolist(),
                    class_weight=None if pesos is None else pesos.tolist())
    ruta = os.path.join(log_dir, SOPORTE_JSON)
    with open(ruta + '.tmp', 'w', encoding='utf-8') as f:
        json.dump(registro, f, indent=1)
    os.replace(ruta + '.tmp', ruta)
    print(SOPORTE_TAG + ' ' + json.dumps(registro, sort_keys=True))

    def tensor(v):
        return None if v is None else torch.as_tensor(np.asarray(v, dtype=np.float32), device=device)
    return tensor(log_pi), tensor(pesos), firma


def soporte_checkpoint(firma):
    return {} if firma is None else {'label_sampling': firma}


def soporte_resume(firma, checkpoint):
    """Resuming with a different labeled set or decision rule is a protocol error."""
    guardada = checkpoint.get('label_sampling') if isinstance(checkpoint, dict) else None
    if firma is None and guardada is None:
        return
    if firma is None:
        raise ValueError('the checkpoint is from a run with labeled support {}; relaunch with the same '
                         'flags'.format(guardada))
    if guardada is None:
        raise ValueError('the checkpoint is from a run with the default support but this one uses {}'.format(firma))
    if guardada != firma:
        raise ValueError('the checkpoint is from another labeled support: {} but this run uses {}'.format(guardada, firma))


def reportar_decision_test(labels, preds_raw, preds_adj, num_classes):
    """Test BA under both decisions; the line format avoids strings the log parsers look for."""
    labels = np.asarray(labels)

    def recalls(preds):
        preds = np.asarray(preds)
        return [100.0 * float(np.mean(preds[labels == c] == c)) if np.any(labels == c) else float('nan')
                for c in range(num_classes)]
    r_adj, r_raw = recalls(preds_adj), recalls(preds_raw)
    print('{} test | BA argmax(z - log pi_l) {:.3f} | argmax z (raw) {:.3f} | adjusted per-class recall [{}] | '
          'raw [{}]'.format(DECISION_TAG, float(np.nanmean(r_adj)), float(np.nanmean(r_raw)),
                              ', '.join('{:.1f}'.format(v) for v in r_adj), ', '.join('{:.1f}'.format(v) for v in r_raw)))
    return float(np.nanmean(r_adj)), float(np.nanmean(r_raw))


def add_epoch_logits_args(parser):
    parser.add_argument('--dump-epoch-logits', default=argparse.SUPPRESS, action='store_true',
                        help='save val and test logits per epoch to <log>/epoch_logits/')
    parser.add_argument('--dump-iter-logits', default=argparse.SUPPRESS, type=int,
                        help='also every N iterations during the first --dump-iter-epochs epochs '
                             '(e<NNN>_i<NNNN>.npz); implies --dump-epoch-logits')
    parser.add_argument('--dump-iter-epochs', default=argparse.SUPPRESS, type=int,
                        help='epochs with fine-grained logging (default 5)')


def _rutas_dataset(ds):
    if isinstance(ds, Subset):
        base = _rutas_dataset(ds.dataset)
        return [base[i] for i in ds.indices]
    return [str(s[0]) for s in getattr(ds, 'samples', [])]


class EpochLogits:
    """Per-epoch val/test logits; RNG state is restored so training is identical to running without the flag."""

    def __init__(self, args, val_loader, val_transform, device):
        self.cada = int(getattr(args, 'dump_iter_logits', 0) or 0)
        self.epocas_finas = int(getattr(args, 'dump_iter_epochs', 5))
        self.iters = args.iters_per_epoch
        self.on = bool(getattr(args, 'dump_epoch_logits', False)) or self.cada > 0
        if not self.on:
            return
        import random as _random
        self._random = _random
        self.device = device
        self.val_loader = val_loader
        test_dataset = datasets.__dict__[args.data](root=args.root, split='test', transform=val_transform)
        # with 0 workers each BCN dump took over a minute
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                       num_workers=int(getattr(args, 'workers', 0)), pin_memory=True)
        self.dir = os.path.join(args.log, 'epoch_logits')
        os.makedirs(self.dir, exist_ok=True)
        with open(os.path.join(self.dir, 'ids.json'), 'w') as f:
            json.dump({'val': _rutas_dataset(val_loader.dataset), 'test': _rutas_dataset(test_dataset)}, f)

    def _logits(self, model, loader):
        out = []
        for images, _ in loader:
            z = model(images.to(self.device, non_blocking=True))
            out.append((z[0] if isinstance(z, tuple) else z).float().cpu().numpy())
        return np.concatenate(out)

    def paso(self, model, epoch, i):
        """Call after step i; the epoch's last step is left to __call__."""
        if self.on and self.cada > 0 and epoch < self.epocas_finas and (i + 1) % self.cada == 0                 and i + 1 < self.iters:
            self._guardar(model, 'e{:03d}_i{:04d}.npz'.format(epoch, i + 1), epoch)

    def __call__(self, model, epoch):
        if not self.on:
            return
        self._guardar(model, 'e{:03d}.npz'.format(epoch), epoch)

    def _guardar(self, model, nombre, epoch):
        estado = (torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                  self._random.getstate(), np.random.get_state())
        entrenando = model.training
        model.eval()
        try:
            with torch.no_grad():
                val, test = self._logits(model, self.val_loader), self._logits(model, self.test_loader)
            np.savez_compressed(os.path.join(self.dir, nombre), val=val, test=test, epoch=epoch)
        finally:
            model.train(entrenando)
            torch.set_rng_state(estado[0])
            if estado[1] is not None:
                torch.cuda.set_rng_state_all(estado[1])
            self._random.setstate(estado[2])
            np.random.set_state(estado[3])


# FixMatch (Sohn et al., 2020) evaluates with a weight EMA, which TLlib lacked; here it is measured only, never trained.
def add_ema_eval_args(parser):
    parser.add_argument('--ema-eval', default=argparse.SUPPRESS, type=float, metavar='DECAY',
                        help='additionally keeps an EMA of the weights (e.g. 0.999) and measures it on val and test '
                             'every epoch; training is unchanged')


def _ba_logits(y, z):
    """BA (%) over the classes present."""
    p = z.argmax(1)
    return 100.0 * float(np.mean([np.mean(p[y == c] == c) for c in np.unique(y)]))


class EMAEval:
    """paso() after each optimizer.step(), (epoch, avg_crudo) after validate(), final() at the end."""

    def __init__(self, args, model, val_loader, val_transform, device):
        self.decay = getattr(args, 'ema_eval', None)
        self.on = self.decay is not None
        if not self.on:
            return
        if not 0.0 < self.decay < 1.0:
            raise ValueError('--ema-eval must be between 0 and 1 (e.g. 0.999)')
        import copy
        import random as _random
        self._random = _random
        self.device = device
        self.model = model
        self.ema = copy.deepcopy(model)
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.ema.eval()
        self.val_loader = val_loader
        test_dataset = datasets.__dict__[args.data](root=args.root, split='test', transform=val_transform)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                       num_workers=int(getattr(args, 'workers', 0)), pin_memory=True)
        self.dir = os.path.join(args.log, 'ema_logits')
        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(os.path.join(args.log, 'checkpoints'), exist_ok=True)
        with open(os.path.join(self.dir, 'ids.json'), 'w') as f:
            json.dump({'val': _rutas_dataset(val_loader.dataset), 'test': _rutas_dataset(test_dataset)}, f)
        self.ckpt = os.path.join(args.log, 'checkpoints', 'ema_best.pth')
        self.resumen = os.path.join(args.log, 'ema_eval.json')
        self.best_avg, self.best_epoch, self.pasos, self.epocas = 0.0, None, 0, []
        print('Weight EMA active: decay {} with the mean_teacher.py ramp, min(decay, 1 - 1/t) (measured on val '
              'and test every epoch; training is unchanged)'.format(self.decay))

    @torch.no_grad()
    def paso(self):
        if not self.on:
            return
        # Ramp min(decay, 1 - 1/t): with a fixed decay the initial network would still weigh 61 % after epoch 0.
        d = min(self.decay, 1.0 - 1.0 / (self.pasos + 1))
        msd = self.model.state_dict()
        for k, v in self.ema.state_dict().items():
            m = msd[k].detach()
            if v.dtype.is_floating_point:
                v.mul_(d).add_(m.to(v.dtype), alpha=1.0 - d)
            else:
                v.copy_(m)
        self.pasos += 1

    def _logits(self, loader):
        zs, ys = [], []
        for images, target in loader:
            z = self.ema(images.to(self.device, non_blocking=True))
            zs.append((z[0] if isinstance(z, tuple) else z).float().cpu().numpy())
            ys.append(np.asarray(target))
        return np.concatenate(zs), np.concatenate(ys)

    def __call__(self, epoch, avg_crudo):
        if not self.on:
            return None
        estado = (torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                  self._random.getstate(), np.random.get_state())
        self.ema.eval()
        t0 = time.time()
        try:
            with torch.no_grad():
                zv, yv = self._logits(self.val_loader)
                zt, yt = self._logits(self.test_loader)
        finally:
            torch.set_rng_state(estado[0])
            if estado[1] is not None:
                torch.cuda.set_rng_state_all(estado[1])
            self._random.setstate(estado[2])
            np.random.set_state(estado[3])
        etiquetas = os.path.join(self.dir, 'labels.npz')
        if not os.path.exists(etiquetas):
            np.savez_compressed(etiquetas, val=yv, test=yt)
        bv, bt = _ba_logits(yv, zv), _ba_logits(yt, zt)
        np.savez_compressed(os.path.join(self.dir, 'e{:03d}.npz'.format(epoch)), val=zv, test=zt, epoch=epoch,
                            val_bal=bv, test_bal=bt, pasos=self.pasos)
        if bv > self.best_avg:
            self.best_avg, self.best_epoch = bv, epoch
            atomic_torch_save(self.ema.state_dict(), self.ckpt)
        self.epocas = [e for e in self.epocas if e['epoch'] != epoch] + [
            {'epoch': epoch, 'pasos': self.pasos, 'val_bal': bv, 'test_bal': bt, 'val_bal_crudo': float(avg_crudo)}]
        self._escribir_resumen()
        print('EMA epoch={} val_bal={:.3f} test_bal={:.3f} | best EMA val {:.3f} (epoch {}) | raw network '
              'val_bal={:.3f} | measured in {:.0f} s'.format(epoch, bv, bt, self.best_avg, self.best_epoch, avg_crudo,
                                                       time.time() - t0))
        return bv

    def _escribir_resumen(self):
        crudas = [e['val_bal_crudo'] for e in self.epocas]
        sel_crudo = self.epocas[int(np.argmax(crudas))]['epoch'] if crudas and max(crudas) > 0 else None
        tmp = self.resumen + '.tmp'
        with open(tmp, 'w') as f:
            json.dump({'decay': self.decay, 'rampa': 'min(decay, 1 - 1/t)', 'pasos': self.pasos,
                       'ema_best_epoch': self.best_epoch,
                       'ema_best_val_bal': self.best_avg, 'epoca_elegida_por_la_red_cruda': sel_crudo,
                       'epocas': self.epocas}, f, indent=1)
        os.replace(tmp, self.resumen)

    def estado(self):
        if not self.on:
            return {}
        return {'ema_state_dict': self.ema.state_dict(), 'ema_best_avg': self.best_avg,
                'ema_best_epoch': self.best_epoch, 'ema_pasos': self.pasos, 'ema_epocas': list(self.epocas)}

    def retomar(self, checkpoint):
        if not self.on:
            return
        if isinstance(checkpoint, dict) and 'ema_state_dict' in checkpoint:
            self.ema.load_state_dict(checkpoint['ema_state_dict'])
            self.best_avg = checkpoint.get('ema_best_avg', 0.0)
            self.best_epoch = checkpoint.get('ema_best_epoch')
            self.pasos = checkpoint.get('ema_pasos', 0)
            self.epocas = list(checkpoint.get('ema_epocas', []))
            print('EMA resumed: {} steps, best val {:.3f} (epoch {})'.format(self.pasos, self.best_avg,
                                                                              self.best_epoch))
        else:
            self.ema.load_state_dict(self.model.state_dict())
            print('WARNING: checkpoint without EMA; the EMA starts from the resumed network')

    def final(self):
        """EMA_TEST at its own best epoch and at the raw network's; format avoids strings the log parsers look for."""
        if not self.on:
            return
        from sklearn.metrics import f1_score, roc_auc_score
        with np.load(os.path.join(self.dir, 'labels.npz')) as z:
            yt = z['test']
        sel_crudo = None
        crudas = [e['val_bal_crudo'] for e in self.epocas]
        if crudas and max(crudas) > 0:
            sel_crudo = self.epocas[int(np.argmax(crudas))]['epoch']
        for etiqueta, ep in (('ema_best', self.best_epoch), ('epoca_de_la_red_cruda', sel_crudo)):
            f = os.path.join(self.dir, 'e{:03d}.npz'.format(ep)) if ep is not None else None
            if f is None or not os.path.exists(f):
                print('EMA_TEST seleccion={} no epoch'.format(etiqueta))
                continue
            with np.load(f) as z:
                zt, bv = z['test'], float(z['val_bal'])
            pr = np.exp(zt - zt.max(1, keepdims=True))
            pr /= pr.sum(1, keepdims=True)
            try:
                auc = roc_auc_score(yt, pr, multi_class='ovr', average='macro')
            except ValueError:
                auc = float('nan')
            print('EMA_TEST seleccion={} epoch={} val_bal={:.4f} bal={:.4f} f1={:.4f} auc={:.4f}'.format(
                etiqueta, ep, bv, _ba_logits(yt, zt), 100 * f1_score(yt, zt.argmax(1), average='macro'), auc))


def add_init_student_args(parser):
    parser.add_argument('--init-student', default=argparse.SUPPRESS, type=str, metavar='CKPT',
                        help="the student's best.pth: the whole network starts from those weights")


def cargar_alumno(classifier, args):
    """Call before Mean Teacher copies its EMA teacher, so the teacher also starts from the student."""
    ruta = getattr(args, 'init_student', None)
    if not ruta:
        return False
    ck = torch.load(ruta, map_location='cpu')
    sd = ck['state_dict'] if isinstance(ck, dict) and 'state_dict' in ck else ck
    classifier.load_state_dict(sd, strict=True)
    h = hashlib.sha256()
    with open(ruta, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    with open(os.path.join(args.log, 'init_student.json'), 'w') as f:
        json.dump({'init_student': os.path.abspath(ruta), 'sha256': h.hexdigest()}, f)
    print('INIT_STUDENT', ruta, h.hexdigest()[:12])
    return True


def add_lesion_pseudo_args(parser):
    parser.add_argument('--lesion-pseudo', default=argparse.SUPPRESS, action='store_true',
                        help='FixMatch pseudo-label from the average over images of the same lesion in the pool')


class _ConIndice(torch.utils.data.Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x, y = self.ds[i]
        return x, y, i


class LesionPseudo:
    """Lesion-level FixMatch: pseudo-label from the bank average over seen images of the same lesion."""

    def __init__(self, args, unlabeled_dataset, num_classes, device):
        self.on = bool(getattr(args, 'lesion_pseudo', False))
        if not self.on:
            return
        rutas = _rutas_dataset(unlabeled_dataset)
        les = pool_modes.lesion_ids([(r, 0) for r in rutas], args.root)
        cod = {l: j for j, l in enumerate(sorted(set(les)))}
        self.g = torch.tensor([cod[l] for l in les], dtype=torch.long, device=device)
        self.n_les = len(cod)
        self.banco = torch.zeros(len(rutas), num_classes, device=device)
        self.visto = torch.zeros(len(rutas), dtype=torch.bool, device=device)
        tam = torch.bincount(self.g)
        info = {'n_pool': len(rutas), 'n_lesiones': self.n_les, 'fotos_en_lesion_multiple': int(tam[self.g].gt(1).sum()),
                'lesiones_multiples': int(tam.gt(1).sum())}
        with open(os.path.join(args.log, 'lesion_pseudo.json'), 'w') as f:
            json.dump(info, f)
        print('LESION_PSEUDO', json.dumps(info))

    def dataset(self, ds):
        return _ConIndice(ds) if self.on else ds

    @torch.no_grad()
    def objetivo(self, y_u, idx):
        """Log of the per-lesion average, so its softmax inside ConfidenceBasedSelfTrainingLoss is that average."""
        idx = idx.to(self.banco.device)
        self.banco[idx] = torch.softmax(y_u.float(), dim=1)
        self.visto[idx] = True
        v = self.visto
        suma = torch.zeros(self.n_les, self.banco.size(1), device=self.banco.device).index_add_(0, self.g[v],
                                                                                                self.banco[v])
        n = torch.bincount(self.g[v], minlength=self.n_les).clamp(min=1).unsqueeze(1)
        return torch.log((suma / n)[self.g[idx]] + 1e-8).to(y_u.dtype)


def add_freno_args(parser):
    parser.add_argument('--freno-mesa', default=argparse.SUPPRESS, type=str,
                        help='json {image: ensemble class on val}; enables the brake')
    parser.add_argument('--freno-delta', default=argparse.SUPPRESS, type=float, help='drop in points (default 3)')
    parser.add_argument('--freno-pausa', default=argparse.SUPPRESS, type=int,
                        help='iterations without the unlabeled loss after a crash (default 500)')
    parser.add_argument('--freno-max', default=argparse.SUPPRESS, type=int, help='maximum number of crashes (default 3)')
    parser.add_argument('--freno-cada', default=argparse.SUPPRESS, type=int,
                        help='iterations between measurements (default 100)')


def _clave_imagen(ruta):
    return '/'.join(str(ruta).replace(chr(92), '/').split('/')[-2:])


class Freno:
    """Rolls back weights and pauses the unlabeled loss when val agreement with the ensemble drops (no val labels used)."""

    def __init__(self, args, val_loader, device):
        self.on = bool(getattr(args, 'freno_mesa', None))
        if not self.on:
            return
        import copy
        import random as _random
        self._copy, self._random = copy, _random
        self.delta = float(getattr(args, 'freno_delta', 3.0))
        self.pausa = int(getattr(args, 'freno_pausa', 500))
        self.max_choques = int(getattr(args, 'freno_max', 3))
        self.cada = int(getattr(args, 'freno_cada', 100))
        self.device, self.val_loader = device, val_loader
        with open(args.freno_mesa) as f:
            mesa = {_clave_imagen(k): int(v) for k, v in json.load(f).items()}
        rutas = [_clave_imagen(r) for r in _rutas_dataset(val_loader.dataset)]
        faltan = [r for r in rutas if r not in mesa]
        assert not faltan, 'freno: {} val images without a class from the ensemble (e.g. {})'.format(len(faltan), faltan[:3])
        self.mesa = torch.tensor([mesa[r] for r in rutas])
        self.ruta = os.path.join(args.log, 'freno.jsonl')
        self.mejor, self.foto, self.restante, self.choques, self.n, self.rebase = None, None, 0, 0, 0, False

    def peso(self):
        return 0.0 if self.on and self.restante > 0 else 1.0

    def estado(self):
        """Without the weight snapshot, to avoid bloating latest.pth and best.pth."""
        return {'freno': {'mejor': self.mejor, 'restante': self.restante, 'choques': self.choques, 'n': self.n,
                       'rebase': self.rebase}} \
            if self.on else {}

    def retomar(self, checkpoint):
        if self.on and 'freno' in checkpoint:
            e = checkpoint['freno']
            self.mejor, self.restante, self.choques, self.n = e['mejor'], e['restante'], e['choques'], e['n']
            self.rebase = e['rebase']

    def _acuerdo(self, model):
        estado = (torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                  self._random.getstate(), np.random.get_state())
        model.eval()
        try:
            pred = []
            with torch.no_grad():
                for images, _ in self.val_loader:
                    z = model(images.to(self.device, non_blocking=True))
                    pred.append((z[0] if isinstance(z, tuple) else z).argmax(1).cpu())
            return 100.0 * (torch.cat(pred) == self.mesa).float().mean().item()
        finally:
            model.train()
            torch.set_rng_state(estado[0])
            if estado[1] is not None:
                torch.cuda.set_rng_state_all(estado[1])
            self._random.setstate(estado[2])
            np.random.set_state(estado[3])

    def _sacar_foto(self, model, optimizer):
        self.foto = ({k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                     self._copy.deepcopy(optimizer.state_dict()))

    def _anotar(self, **kw):
        with open(self.ruta, 'a') as f:
            f.write(json.dumps(kw) + chr(10))

    def paso(self, model, optimizer, epoch, i):
        if not self.on:
            return
        self.n += 1
        if self.restante > 0:
            self.restante -= 1
        if self.foto is None:  # start or resume: the rollback point is the current state
            self._sacar_foto(model, optimizer)
        if self.n % self.cada:
            return
        a = self._acuerdo(model)
        evento = 'medida'
        if self.restante > 0:
            evento = 'pausa'
        elif self.mejor is None or self.rebase:
            self.mejor, self.rebase = a, False
            self._sacar_foto(model, optimizer)
            evento = 'base'
        elif a < self.mejor - self.delta and self.choques < self.max_choques:
            model.load_state_dict(self.foto[0])
            optimizer.load_state_dict(self.foto[1])
            self.choques += 1
            self.restante = self.pausa
            self.rebase = True
            evento = 'choque'
        elif a >= self.mejor:
            self.mejor = a
            self._sacar_foto(model, optimizer)
            evento = 'maximo'
        self._anotar(epoca=epoch, iter=i + 1, n=self.n, acuerdo=round(a, 3), mejor=round(self.mejor, 3),
                     evento=evento, choques=self.choques, restante=self.restante)


# e.g. drops the HAM10000 duplicates hidden inside MILK10k from the pool
POOL_EXCLUDE_JSON = 'pool_exclude.json'
POOL_EXCLUDE_TAG = 'POOL_EXCLUDE'
_ISIC_ID = re.compile(r'ISIC_\d{7}')


def add_pool_exclude_args(parser):
    group = parser.add_argument_group('pool exclusion')
    group.add_argument('--pool-exclude', default=None, type=str,
                       help='file with one ISIC_XXXXXXX per line: those images are removed from the unlabeled pool '
                            'after x_u_split. The labeled set, validation, test and RNG are unchanged')


def kw_pool_exclude(args):
    ruta = getattr(args, 'pool_exclude', None)
    return {} if ruta is None else {'pool_exclude': ruta}


def leer_lista_exclusion(ruta):
    """(ids, sha256); hashed with LF endings so the sha matches on Windows CRLF checkouts."""
    with open(ruta, 'rb') as f:
        contenido = f.read().replace(b'\r\n', b'\n')
    ids = set()
    for n, linea in enumerate(contenido.decode('utf-8-sig').split('\n'), 1):
        linea = linea.strip()
        if not linea or linea.startswith('#'):
            continue
        if not _ISIC_ID.fullmatch(linea):
            raise ValueError('--pool-exclude {}: line {} is not an ISIC_XXXXXXX: {!r}'.format(ruta, n, linea))
        ids.add(linea)
    return ids, hashlib.sha256(contenido).hexdigest()


def isic_de_ruta(path):
    """In clinical MILK the file name carries the derm id of its lesion."""
    m = _ISIC_ID.search(os.path.basename(path))
    return m.group(0) if m else None


def excluir_del_pool(unlabeled_train_dataset, labeled_idxes, ruta):
    """Pool without the listed images; listed images already in the labeled set stay there and are counted."""
    ids, sha = leer_lista_exclusion(ruta)
    base = unlabeled_train_dataset.dataset
    nombres = [isic_de_ruta(p) for p, _ in base.samples]
    presentes = ids.intersection(nombres)
    if ids and not presentes:
        raise ValueError('--pool-exclude {}: no train image matches the list'.format(ruta))
    quitar = [i for i in unlabeled_train_dataset.indices if nombres[i] in ids]
    quitadas = set(quitar)
    filtrado = Subset(base, [i for i in unlabeled_train_dataset.indices if i not in quitadas])
    en_etiquetado = sorted(nombres[i] for i in labeled_idxes if nombres[i] in ids)
    targets = np.asarray(base.targets)
    reporte = {
        'ruta': ruta, 'sha256': sha, 'n_lista': len(ids),
        'n_quitadas': len(quitar),
        'quitadas_por_clase': {c: int(np.sum(targets[quitar] == k)) for k, c in enumerate(base.classes)},
        'n_en_etiquetado': len(en_etiquetado), 'en_etiquetado': en_etiquetado,
        'n_fuera_del_train': len(ids - presentes),
        'pool_antes': len(unlabeled_train_dataset.indices), 'pool_despues': len(filtrado.indices),
    }
    filtrado.pool_exclude = reporte
    print('[pool-exclude] {} (sha256 {}): {} images removed from the pool ({}); {} from the list are in the labeled set '
          'and stay; pool {} -> {}'.format(
              ruta, sha[:12], len(quitar),
              ', '.join('{} {}'.format(c, n) for c, n in reporte['quitadas_por_clase'].items()),
              len(en_etiquetado), reporte['pool_antes'], reporte['pool_despues']))
    if reporte['n_fuera_del_train']:
        print('[pool-exclude] WARNING: {} ids in the list are not in the train split of this dataset'.format(
            reporte['n_fuera_del_train']))
    print('{} {}'.format(POOL_EXCLUDE_TAG, json.dumps(reporte, ensure_ascii=False, sort_keys=True)))
    return filtrado


def guardar_pool_exclude(unlabeled_train_dataset, log_dir):
    """<log>/pool_exclude.json, which the run agent copies into result.json."""
    reporte = getattr(unlabeled_train_dataset, 'pool_exclude', None)
    if reporte is None:
        return None
    ruta = os.path.join(log_dir, POOL_EXCLUDE_JSON)
    with open(ruta, 'w', encoding='utf-8') as f:
        json.dump(reporte, f, ensure_ascii=False, indent=1)
    return ruta


# Ordering matches x_u_split, so feeding a seed's labeled_ids.json reproduces that seed's set exactly.
LABELED_LIST_JSON = 'labeled_list.json'
LABELED_LIST_TAG = 'LABELED_LIST'


def add_labeled_list_args(parser):
    group = parser.add_argument_group('explicit labeled list (default: the usual random draw)')
    group.add_argument('--labeled-list', default=argparse.SUPPRESS, type=str, metavar='JSON',
                       help='JSON with "ids": paths of the labeled images, relative to the dataset root '
                            '(train/<class>/<file>, the labeled_ids.json format). They replace the x_u_split '
                            'draw; --num-samples-per-class becomes only a label. Without the flag nothing changes')
    return parser


def kw_labeled_list(args):
    ruta = getattr(args, 'labeled_list', None)
    return {} if ruta is None else {'labeled_list': ruta}


def leer_lista_etiquetadas(ruta):
    """(ids, sha256); hashed with LF endings so the sha matches on Windows CRLF checkouts."""
    with open(ruta, 'rb') as f:
        contenido = f.read().replace(b'\r\n', b'\n')
    try:
        datos = json.loads(contenido.decode('utf-8-sig'))
    except ValueError as e:
        raise ValueError('--labeled-list {}: not a valid JSON ({})'.format(ruta, e))
    ids = datos.get('ids') if isinstance(datos, dict) else None
    if not isinstance(ids, list) or not ids or not all(isinstance(i, str) and i for i in ids):
        raise ValueError('--labeled-list {}: a key "ids" with a non-empty list of paths is required'.format(ruta))
    if 'n' in datos and datos['n'] != len(ids):
        raise ValueError('--labeled-list {}: "n" is {} but "ids" has {}'.format(ruta, datos['n'], len(ids)))
    vistos, repetidos = set(), set()
    for i in ids:
        (repetidos if i in vistos else vistos).add(i)
    if repetidos:
        raise ValueError('--labeled-list {}: repeated paths: {}'.format(ruta, sorted(repetidos)[:5]))
    return ids, hashlib.sha256(contenido).hexdigest()


def indices_de_lista(base_dataset, root, ruta):
    """Replaces x_u_split: (labeled_idxes, unlabeled_idxes, report)."""
    ids, sha = leer_lista_etiquetadas(ruta)
    raiz = os.path.abspath(root)
    pos = {os.path.relpath(p, raiz).replace(os.sep, '/'): i for i, (p, _) in enumerate(base_dataset.samples)}
    faltan = [i for i in ids if i not in pos]
    if faltan:
        raise ValueError('--labeled-list {}: {} of {} paths are not images of this dataset split (e.g. {})'.format(
            ruta, len(faltan), len(ids), faltan[:3]))
    labeled_idxes = [pos[i] for i in ids]
    targets = np.asarray(base_dataset.targets)
    clases = getattr(base_dataset, 'classes', None)
    if clases is not None:
        mal = [i for i, j in zip(ids, labeled_idxes) if '/' in i and i.split('/')[-2] != clases[targets[j]]]
        if mal:
            raise ValueError('--labeled-list {}: paths outside their class folder: {}'.format(ruta, mal[:3]))
    elegidas = set(labeled_idxes)
    unlabeled_idxes = [i for i in range(len(targets)) if i not in elegidas]
    conteos = np.bincount(targets[np.asarray(labeled_idxes, dtype=np.int64)], minlength=base_dataset.num_classes)
    reporte = {'ruta': ruta, 'sha256': sha, 'n': len(ids), 'conteos': conteos.tolist(),
               'indices_sha256': hashlib.sha256(np.sort(np.asarray(labeled_idxes, dtype=np.int64)).tobytes()).hexdigest()}
    print('[labeled-list] {} (sha256 {}): {} labeled, per class {}'.format(ruta, sha[:12], len(ids),
                                                                               conteos.tolist()))
    print('{} {}'.format(LABELED_LIST_TAG, json.dumps(reporte, sort_keys=True)))
    return labeled_idxes, unlabeled_idxes, reporte


def validar_lista(args):
    """Fails early (before seeding and loading data) on invalid --labeled-list combinations."""
    ruta = getattr(args, 'labeled_list', None)
    if ruta is None:
        return
    errores = []
    if not os.path.isfile(ruta):
        errores.append('no existe --labeled-list {}'.format(ruta))
    else:
        try:
            leer_lista_etiquetadas(ruta)
        except ValueError as e:
            errores.append(str(e))
    if getattr(args, 'oracle', False):
        errores.append('--labeled-list does not apply with --oracle (the whole train set is labeled)')
    if getattr(args, 'subset_seed', None) is not None:
        errores.append('--subset-seed picks the labeled images with x_u_split: it cannot be combined with --labeled-list')
    if soporte_opt(args, 'label_sampling') != 'balanced' or soporte_opt(args, 'num_labeled_total') is not None:
        errores.append('--label-sampling natural and --num-labeled-total draw the labeled images: they cannot be combined with '
                       '--labeled-list')
    if getattr(args, 'pool_mode', 'orig') != 'orig':
        errores.append('--pool-mode assumes the x_u_split draw: it cannot be combined with --labeled-list')
    if errores:
        raise ValueError('labeled list: ' + '; '.join(errores))


def guardar_lista_etiquetadas(labeled_train_dataset, log_dir):
    """Writes <log>/labeled_list.json and returns the signature checked on resume."""
    reporte = getattr(labeled_train_dataset, 'labeled_list', None)
    if reporte is None:
        return None
    ruta = os.path.join(log_dir, LABELED_LIST_JSON)
    with open(ruta + '.tmp', 'w', encoding='utf-8') as f:
        json.dump(reporte, f, indent=1)
    os.replace(ruta + '.tmp', ruta)
    if min(reporte['conteos']) == 0:
        print('WARNING labeled-list: some classes have no labeled images: {}'.format(reporte['conteos']))
    return {k: reporte[k] for k in ('sha256', 'n', 'conteos', 'indices_sha256')}


def lista_checkpoint(firma):
    return {} if firma is None else {'labeled_list': firma}


def lista_resume(firma, checkpoint):
    """Compares content, not path: the list file may move between runs."""
    guardada = checkpoint.get('labeled_list') if isinstance(checkpoint, dict) else None
    if firma is None and guardada is None:
        return
    if firma is None:
        raise ValueError('the checkpoint is from a run with --labeled-list {}; relaunch with the same list'.format(
            guardada))
    if guardada is None:
        raise ValueError('the checkpoint is from a run with the usual random draw but this one uses --labeled-list {}'.format(
            firma))
    if guardada != firma:
        raise ValueError('the checkpoint is from another labeled list: {} but this run uses {}'.format(guardada,
                                                                                                         firma))


def get_train_transform(resizing='default', random_horizontal_flip=True, auto_augment=None,
                        norm_mean=(0.485, 0.456, 0.406), norm_std=(0.229, 0.224, 0.225)):
    if resizing == 'default':
        transform = T.RandomResizedCrop(224, scale=(0.2, 1.))
    elif resizing == 'lesion_preserving':
        # clinical photos: the lesion is small in a large scene, so a 0.2 crop can cut it out
        transform = T.RandomResizedCrop(224, scale=(0.5, 1.))
    elif resizing == 'cifar':
        transform = T.Compose([
            T.RandomCrop(size=32, padding=4, padding_mode='reflect'),
            ResizeImage(224)
        ])
    else:
        raise NotImplementedError(resizing)
    transforms = [transform]
    if random_horizontal_flip:
        transforms.append(T.RandomHorizontalFlip())
    if auto_augment:
        aa_params = dict(
            translate_const=int(224 * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in norm_mean]),
            interpolation=Image.BILINEAR
        )
        if auto_augment.startswith('rand'):
            transforms.append(rand_augment_transform(auto_augment, aa_params))
        else:
            transforms.append(auto_augment_transform(auto_augment, aa_params))
    transforms.extend([
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
    ])
    return T.Compose(transforms)


def get_val_transform(resizing='default', norm_mean=(0.485, 0.456, 0.406), norm_std=(0.229, 0.224, 0.225)):
    if resizing == 'default':
        transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224),
        ])
    elif resizing == 'cifar':
        transform = ResizeImage(224)
    else:
        raise NotImplementedError(resizing)
    return T.Compose([
        transform,
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
    ])


class DatasetWrapper:
    """Module-level so it pickles: DataLoader workers on Windows use spawn."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        return index, self.dataset[index]

    def __len__(self):
        return len(self.dataset)


def convert_dataset(dataset):
    return DatasetWrapper(dataset)


class ImageClassifier(Classifier):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim=1024, **kwargs):
        bottleneck = nn.Sequential(
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        bottleneck[0].weight.data.normal_(0, 0.005)
        bottleneck[0].bias.data.fill_(0.1)
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim, **kwargs)

    def forward(self, x: torch.Tensor):
        f = self.pool_layer(self.backbone(x))
        f = self.bottleneck(f)
        predictions = self.head(f)
        return predictions


def get_cosine_scheduler_with_warmup(optimizer, T_max, num_cycles=7. / 16., num_warmup_steps=0,
                                     last_epoch=-1):
    """Cosine schedule from FixMatch (Sohn et al., NeurIPS 2020)."""

    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            _lr = float(current_step) / float(max(1, num_warmup_steps))
        else:
            num_cos_steps = float(current_step - num_warmup_steps)
            num_cos_steps = num_cos_steps / float(max(1, T_max - num_warmup_steps))
            _lr = max(0.0, math.cos(math.pi * num_cycles * num_cos_steps))
        return _lr

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def validate(val_loader, model, args, device, num_classes, log_prior=None):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    model.eval()
    confmat = ConfusionMatrix(num_classes)
    confmat_crudo = ConfusionMatrix(num_classes) if log_prior is not None else None

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(images)
            loss = F.cross_entropy(output, target)
            if log_prior is not None:
                confmat_crudo.update(target, output.argmax(1))
                output = output - log_prior.to(device=output.device, dtype=output.dtype)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            confmat.update(target, output.argmax(1))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        acc_global, acc_per_class, iu = confmat.compute()
        mean_cls_acc = acc_per_class.mean().item() * 100
        print(' * Mean Cls {:.3f}'.format(mean_cls_acc))
        if log_prior is not None:
            print('{} val | BA argmax(z - log pi_l) {:.3f} | argmax z (raw) {:.3f}'.format(
                DECISION_TAG, mean_cls_acc, confmat_crudo.compute()[1].mean().item() * 100))

    return top1.avg, mean_cls_acc


def empirical_risk_minimization(labeled_train_iter, model, optimizer, lr_scheduler, epoch, args, device,
                                class_weight=None, extra_loss=None, historia=None):
    batch_time = AverageMeter('Time', ':2.2f')
    data_time = AverageMeter('Data', ':2.1f')
    losses = AverageMeter('Loss', ':3.2f')
    cls_accs = AverageMeter('Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    end = time.time()
    batch_size = args.batch_size
    for i in range(args.iters_per_epoch):
        (x_l, x_l_strong), labels_l = next(labeled_train_iter)
        x_l = x_l.to(device, non_blocking=True)
        x_l_strong = x_l_strong.to(device, non_blocking=True)
        labels_l = labels_l.to(device, non_blocking=True)

        data_time.update(time.time() - end)

        y_l = model(x_l)
        y_l_strong = model(x_l_strong)
        loss = F.cross_entropy(y_l, labels_l) + args.trade_off_cls_strong * F.cross_entropy(y_l_strong, labels_l)
        if class_weight is not None:
            loss = (F.cross_entropy(y_l, labels_l, weight=class_weight)
                    + args.trade_off_cls_strong * F.cross_entropy(y_l_strong, labels_l, weight=class_weight))
        if extra_loss is not None:
            loss = loss + extra_loss(model, x_l, labels_l)

        losses.update(loss.item(), batch_size)
        cls_acc = accuracy(y_l, labels_l)[0]
        cls_accs.update(cls_acc.item(), batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        if historia is not None:
            historia.paso(model, epoch, i)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
