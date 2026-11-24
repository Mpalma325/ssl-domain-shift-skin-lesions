"""Noisy Student (Xie et al., CVPR 2020) for semi-supervised skin-lesion classification.

Adapted from TLlib (Baixu Chen). The ERM teacher (--pretrained-teacher, an erm.py best.pth) labels the pool
once; the student trains on labeled CE plus soft-target CE on strong views at temperature T. Without a teacher
the script trains ERM. Optional flags, all off by default, only change the teacher table: source ensemble or
calibrated product of experts, center view, cross teacher, pairing controls, third and fourth experts with a
weight on the ERM terms, per-lesion mean, a different teacher architecture and a dump of the teacher on test.
--labeled-sisters (BCN20000) also moves the pool images of labeled lesions to the labeled set.
"""
import copy
import random
import time
import warnings
import argparse
import shutil
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data import Subset, ConcatDataset

import utils
from tllib.vision.models.reid.loss import CrossEntropyLoss
from tllib.modules.classifier import Classifier
from tllib.vision.transforms import MultipleApply
from tllib.utils.metric import accuracy
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.utils.data import ForeverDataIterator
from tllib.utils.logger import CompleteLogger
import tllib.vision.datasets as datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class _Timer:
    def __init__(self):
        self.records = {}
    def start(self, name):
        self.records[name] = self.records.get(name, {'total': 0.0, 'n': 0})
        self._t = time.perf_counter()
        self._name = name
    def stop(self):
        elapsed = time.perf_counter() - self._t
        self.records[self._name]['total'] += elapsed
        self.records[self._name]['n'] += 1
    def report(self):
        print("\n" + "="*60)
        print(f"{'TIMING REPORT':^60}")
        print("="*60)
        print(f"{'Stage':<35} {'Total(s)':>8} {'N':>6} {'Avg(ms)':>10}")
        print("-"*60)
        for name, v in self.records.items():
            avg_ms = (v['total'] / v['n'] * 1000) if v['n'] > 0 else 0
            print(f"  {name:<33} {v['total']:>8.2f} {v['n']:>6} {avg_ms:>10.1f}")
        print("="*60 + "\n")
_timer = _Timer()


class ImageClassifier(Classifier):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim=1024, **kwargs):
        bottleneck = nn.Sequential(
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        bottleneck[0].weight.data.normal_(0, 0.005)
        bottleneck[0].bias.data.fill_(0.1)
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim, **kwargs)
        self.dropout = nn.Dropout(0.5)
        self.as_teacher_model = False

    def forward(self, x: torch.Tensor):
        f = self.pool_layer(self.backbone(x))
        f = self.bottleneck(f)
        if not self.as_teacher_model:
            f = self.dropout(f)
        predictions = self.head(f)
        return predictions


def calc_teacher_output(classifier_teacher: ImageClassifier, weak_augmented_unlabeled_dataset):
    """Teacher logits on the pool: weak view and no dropout, as in the Noisy Student paper."""

    data_loader = DataLoader(weak_augmented_unlabeled_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, drop_last=False, pin_memory=True)
    batch_time = AverageMeter('Time', ':6.3f')
    progress = ProgressMeter(
        len(data_loader),
        [batch_time],
        prefix='Computing teacher output: ')

    teacher_output = []
    with torch.no_grad():
        end = time.time()
        for i, (images, _) in enumerate(data_loader):
            images = images.to(device, non_blocking=True)
            output = classifier_teacher(images)
            teacher_output.append(output)

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

    teacher_output = torch.cat(teacher_output, dim=0)
    return teacher_output


def calc_outputs_dual(classifier_teacher, classifier_source, weak_augmented_unlabeled_dataset):
    """Teacher and source outputs on the same weak views of the pool, in one pass."""
    data_loader = DataLoader(weak_augmented_unlabeled_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, drop_last=False, pin_memory=True)
    out_t, out_s = [], []
    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device, non_blocking=True)
            out_t.append(classifier_teacher(images))
            out_s.append(classifier_source(images))
    return torch.cat(out_t, dim=0), torch.cat(out_s, dim=0)


def alpha_por_confiabilidad(classifier_source, log_prior, labeled_idxes, val_transform):
    """Source weight from the prior-corrected source BA on the labeled set (see alpha_de_ba)."""
    import numpy as np
    import tllib.vision.datasets as datasets
    base = datasets.__dict__[args.data](root=args.root, split='train', transform=val_transform)
    loader = DataLoader(torch.utils.data.Subset(base, labeled_idxes), batch_size=args.batch_size,
                        shuffle=False, num_workers=0)
    pred, lab = [], []
    with torch.no_grad():
        for images, labels in loader:
            z = classifier_source(images.to(device)) - log_prior
            pred.append(z.argmax(1).cpu().numpy()); lab.append(labels.numpy())
    pred, lab = np.concatenate(pred), np.concatenate(lab)
    ba = 100 * np.mean([np.mean(pred[lab == c] == c) for c in np.unique(lab)])
    alpha, r = alpha_de_ba(ba)
    return alpha, ba, r


def alpha_de_ba(ba):
    """alpha = 0.5 * clip((BA - 20) / (70.7 - 20), 0, 1): 70.7 = source BA on HAM test, 20 = chance."""
    r = (ba - 20.0) / (70.7 - 20.0)
    return 0.5 * min(max(r, 0.0), 1.0), r


def profesor_ensamble(z_t, z_s, log_prior, alpha):
    """log(alpha * softmax(z_s - log_prior) + (1 - alpha) * softmax(z_t)); alpha = 0 is the plain teacher."""
    p_s = torch.softmax(z_s - log_prior, dim=1)
    p_t = torch.softmax(z_t, dim=1)
    return torch.log((alpha * p_s + (1 - alpha) * p_t).clamp_min(1e-12)), p_s, p_t


# train() is unchanged: log-probs work as teacher logits because softmax is invariant to a per-row
# constant, so softmax(log_softmax(z) / T) = softmax(z / T).
CALIB_T_BOUNDS = (0.05, 50.0)  # bounds on T for the NLL fits: avoids T -> 0 if the calibration set is separable
DEFAULT_BCTS_L2 = 1e-3


def add_budget_args(parser: argparse.ArgumentParser):
    """N validation images per class; default None keeps the full validation set."""
    group = parser.add_argument_group('validation budget')
    group.add_argument('--val-per-class', default=None, type=int,
                       help='N validation images per class (own RNG, RandomState(10000 + seed)); '
                            'default: the full validation set')


def add_teacher_args(parser: argparse.ArgumentParser):
    """Configurable-teacher flags; the --source-* flags are defined in the main parser."""
    group = parser.add_argument_group('configurable teacher (default: previous behavior)')
    group.add_argument('--teacher-view', default='weak', choices=['weak', 'center'],
                       help="view used to compute, once, the teacher (and source) outputs: "
                            "weak = weak training view, the default (one fixed random crop per image); "
                            "center = validation transform, no randomness (noise-free teacher, as in the "
                            "original Noisy Student)")
    group.add_argument('--teacher-combine', default='mix', choices=['mix', 'poe'],
                       help="only with --source-checkpoint. mix = log(a*softmax(z_src - log pi_src) + "
                            "(1-a)*softmax(z_teacher)), with a = --source-alpha (weight of the SOURCE); "
                            "poe = calibrated product of experts: log_softmax(z_teacher/T_t) + "
                            "log_softmax((z_src - log pi_src)/T_s + b), renormalized (ignores --source-alpha)")
    group.add_argument('--teacher-temp', default=None, type=float,
                       help='poe: fixed T_t for the teacher (e.g. 4). If omitted, T_t is fit by class-balanced '
                            'NLL on the validation set (the early-stopping set)')
    group.add_argument('--bcts-l2', default=None, type=float,
                       help='poe: L2 penalty on the BCTS bias b of the source (default 1e-3)')
    group.add_argument('--teacher-root', default=None, type=str,
                       help='dataset root seen by the teacher and the source (default: the same root). Images '
                            'are paired with the student images by relative path (cross teacher: '
                            'teacher on MILK10K, student on MILK10K_CLINICO). Requires --teacher-view center')
    return parser


def teacher_flags_active(args) -> bool:
    """True if any configurable-teacher flag differs from its default (configurable path)."""
    return args.teacher_view != 'weak' or args.teacher_combine != 'mix' or args.teacher_root is not None


# Cross-teacher pairing controls: they replace rows of the precomputed teacher table and never touch
# training. The photo teacher runs inside fork_rng so the student RNG stream is unchanged.
PAIRS_JSON = 'pairs.json'
PAIRS_TAG = 'PAIRS'
PAIR_KEYS = ('pair_fraction', 'photo_teacher', 'pair_shuffle', 'pair_seed')


def add_pair_args(parser):
    group = parser.add_argument_group('cross-teacher pairs (default: every image keeps its pair)')
    group.add_argument('--pair-fraction', default=argparse.SUPPRESS, type=float, metavar='F',
                       help='fraction of the pool (strictly between 0 and 1) that keeps the cross teacher of its '
                            'pair; the rest gets --photo-teacher. Requires --photo-teacher and --pair-seed')
    group.add_argument('--photo-teacher', default=argparse.SUPPRESS, type=str, metavar='CKPT',
                       help='best.pth of the photo teacher, only with --pair-fraction')
    group.add_argument('--pair-shuffle', default=argparse.SUPPRESS, choices=['class'],
                       help='class: each image gets the cross-teacher row of another image with the same class '
                            'predicted by that teacher. Requires --pair-seed')
    group.add_argument('--pair-seed', default=argparse.SUPPRESS, type=int,
                       help='seed of the subset (--pair-fraction) or of the shuffle (--pair-shuffle)')
    return parser


def pair_flags(args):
    return {k: getattr(args, k) for k in PAIR_KEYS if hasattr(args, k)}


def validate_pair_args(args):
    f = pair_flags(args)
    if not f:
        return
    errors = []
    frac, shuf = 'pair_fraction' in f, 'pair_shuffle' in f
    if frac == shuf:
        errors.append('use --pair-fraction or --pair-shuffle (exactly one)')
    if 'pair_seed' not in f:
        errors.append('--pair-fraction and --pair-shuffle require --pair-seed')
    if frac:
        if not 0.0 < f['pair_fraction'] < 1.0:
            errors.append('--pair-fraction must be strictly between 0 and 1 (0 is the fused-student run and 1 is the cross-teacher run)')
        if 'photo_teacher' not in f:
            errors.append('--pair-fraction requires --photo-teacher')
        elif not os.path.isfile(f['photo_teacher']):
            errors.append('--photo-teacher {} does not exist'.format(f['photo_teacher']))
    elif 'photo_teacher' in f:
        errors.append('--photo-teacher is only used with --pair-fraction')
    if getattr(args, 'teacher_root', None) is None:
        errors.append('the pairing flags require --teacher-root (the cross teacher)')
    if not getattr(args, 'pretrained_teacher', None):
        errors.append('the pairing flags require --pretrained-teacher')
    if errors:
        raise ValueError('invalid pairing flags:\n  - ' + '\n  - '.join(errors))


def pair_subset(n, fraction, seed):
    """First round(fraction * n) of a seeded permutation: a smaller fraction is a prefix of a larger one."""
    k = int(round(float(fraction) * n))
    mask = np.zeros(n, dtype=bool)
    mask[np.random.RandomState(int(seed)).permutation(n)[:k]] = True
    return mask


def class_shuffle_map(classes, seed):
    """Maps each position to another of the same class (one cycle per class); singletons stay in place."""
    classes = np.asarray(classes)
    rng = np.random.RandomState(int(seed))
    mapa = np.arange(len(classes))
    for c in np.unique(classes):
        idx = np.flatnonzero(classes == c)
        if len(idx) < 2:
            continue
        orden = idx[rng.permutation(len(idx))]
        mapa[orden] = np.roll(orden, -1)
    return mapa


def _sha_indices(indices):
    import hashlib
    return hashlib.sha256(','.join(str(int(i)) for i in indices).encode()).hexdigest()


def apply_pair_flags(args, teacher_output, photo_output_fn):
    """Replaces teacher rows per the pairing flags and writes pairs.json; photo_output_fn runs in fork_rng."""
    f = pair_flags(args)
    if not f:
        return teacher_output
    n, num_classes = teacher_output.size(0), teacher_output.size(1)
    cls_cross = teacher_output.argmax(dim=1).cpu().numpy()
    record = {'n_pool': int(n), 'pair_seed': int(f['pair_seed']), 'pretrained_teacher': args.pretrained_teacher,
              'teacher_root': os.path.abspath(args.teacher_root)}
    if 'pair_fraction' in f:
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            photo = photo_output_fn()
        if tuple(photo.shape) != tuple(teacher_output.shape):
            raise ValueError('the photo teacher has shape {} and the cross teacher {}'.format(
                tuple(photo.shape), tuple(teacher_output.shape)))
        photo = photo.to(device=teacher_output.device, dtype=teacher_output.dtype)
        mask = pair_subset(n, f['pair_fraction'], f['pair_seed'])
        keep = torch.as_tensor(mask, device=teacher_output.device)
        out = torch.where(keep[:, None], teacher_output, photo)
        record.update(mode='fraction', pair_fraction=float(f['pair_fraction']), photo_teacher=f['photo_teacher'],
                      n_paired=int(mask.sum()), paired_sha256=_sha_indices(np.flatnonzero(mask)),
                      argmax_agreement=float((photo.argmax(dim=1).cpu().numpy() == cls_cross).mean()))
    else:
        mapa = class_shuffle_map(cls_cross, f['pair_seed'])
        out = teacher_output[torch.as_tensor(mapa, device=teacher_output.device)]
        record.update(mode='shuffle_' + f['pair_shuffle'], n_fixed=int((mapa == np.arange(n)).sum()),
                      map_sha256=_sha_indices(mapa),
                      class_counts=[int((cls_cross == c).sum()) for c in range(num_classes)])
    path = os.path.join(args.log, PAIRS_JSON)
    with open(path + '.tmp', 'w', encoding='utf-8') as fh:
        json.dump(record, fh, indent=1, ensure_ascii=False)
    os.replace(path + '.tmp', path)
    print(PAIRS_TAG + ' ' + json.dumps(record, ensure_ascii=False, sort_keys=True))
    return out


# Committee teacher (--third-checkpoint: HAM ConvNeXt-T as a third calibrated expert) and per-lesion mean
# of the teacher log-probs (--teacher-lesion-mean). The ConvNeXt is built inside fork_rng so the student
# RNG stream is unchanged.
LESION_JSON = 'lesion_mean.json'
LESION_TAG = 'LESION_MEAN'
COMMITTEE_KEYS = ('third_checkpoint', 'teacher_lesion_mean')


def add_committee_args(parser):
    group = parser.add_argument_group('committee teacher and per-lesion mean (default: off)')
    group.add_argument('--third-checkpoint', default=argparse.SUPPRESS, type=str, metavar='CKPT',
                       help='best.pth of the HAM ConvNeXt-T: third expert of the calibrated product (requires '
                            '--teacher-combine poe and --source-checkpoint; uses the --source-prior-counts prior)')
    group.add_argument('--teacher-lesion-mean', default=argparse.SUPPRESS, type=str, metavar='CSV',
                       help='CSV with columns isic_id and lesion_id: averages the teacher log-probabilities over '
                            'the pool images of the same lesion')
    return parser


def committee_flags(args):
    return {k: getattr(args, k) for k in COMMITTEE_KEYS if hasattr(args, k)}


def validate_committee_args(args):
    f = committee_flags(args)
    if not f:
        return
    errors = []
    if not getattr(args, 'pretrained_teacher', None):
        errors.append('the committee flags require --pretrained-teacher')
    if getattr(args, 'teacher_combine', 'mix') != 'poe':
        errors.append('the committee flags require --teacher-combine poe (the teacher table in log-probs)')
    if not getattr(args, 'source_checkpoint', None):
        errors.append('the committee flags require --source-checkpoint (and --source-prior-counts)')
    for key, flag in (('third_checkpoint', '--third-checkpoint'), ('teacher_lesion_mean', '--teacher-lesion-mean')):
        if key in f and not os.path.isfile(f[key]):
            errors.append('{} {} does not exist'.format(flag, f[key]))
    if errors:
        raise ValueError('invalid committee-teacher flags:\n  - ' + '\n  - '.join(errors))


class ConvNeXtHAM(nn.Module):
    """HAM ConvNeXt-T: torchvision features, original LayerNorm and a linear head; inference only."""

    def __init__(self, num_classes):
        super().__init__()
        import torchvision
        try:
            m = torchvision.models.convnext_tiny(weights=None)
        except TypeError:  # torchvision < 0.13
            m = torchvision.models.convnext_tiny(pretrained=False)
        self.features, self.avgpool = m.features, m.avgpool
        self.norm = m.classifier[0]
        self.head = nn.Linear(768, num_classes)

    def forward(self, x):
        return self.head(torch.flatten(self.norm(self.avgpool(self.features(x))), 1))


def load_third_expert(path, num_classes):
    """Built inside fork_rng because initializing its weights consumes RNG."""
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        model = ConvNeXtHAM(num_classes)
    checkpoint = torch.load(path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    model.load_state_dict(checkpoint)
    model.eval()
    return model.to(device)


# Four-expert teacher: log p = W log_softmax(z_t / T_t) + source BCTS [+ third BCTS] [+ W log_softmax(z_4 / T_t)],
# renormalized. The fourth expert (--fourth-teacher, an erm.py best.pth of the target) enters like the ERM teacher,
# without prior or BCTS. Off by default (SUPPRESS): without the flags the Namespace, RNG and teacher table are unchanged.
FOURTH_KEYS = ('fourth_teacher', 'fourth_teacher_arch', 'erm_weight')


def add_fourth_expert_args(parser):
    group = parser.add_argument_group('four-expert teacher (default: off)')
    group.add_argument('--fourth-teacher', default=argparse.SUPPRESS, type=str, metavar='CKPT',
                       help='best.pth of a second ERM network of the target: fourth expert of the product, at T_t, '
                            'without prior')
    group.add_argument('--fourth-teacher-arch', default=argparse.SUPPRESS, metavar='ARCH',
                       choices=utils.get_model_names(), help='architecture of the fourth expert (convnext_tiny_ham)')
    group.add_argument('--erm-weight', default=argparse.SUPPRESS, type=float, metavar='W',
                       help='weight of the ERM terms (teacher and fourth expert) in the product of experts (1 '
                            'without the flag)')
    group.add_argument('--teacher-test-dump', default=argparse.SUPPRESS, action='store_true',
                       help='also runs the teacher on the test split of its root and writes teacher_test.npz: logits '
                            'of each expert and of the combination on val, test and pool')
    return parser


def fourth_flags(args):
    return {k: getattr(args, k) for k in FOURTH_KEYS if hasattr(args, k)}


def validate_fourth_args(args):
    f = fourth_flags(args)
    if not f and not getattr(args, 'teacher_test_dump', False):
        return
    errors = []
    if f and (getattr(args, 'teacher_combine', 'mix') != 'poe' or not getattr(args, 'source_checkpoint', None)):
        errors.append('the four-expert flags require --teacher-combine poe and --source-checkpoint')
    if f and not getattr(args, 'pretrained_teacher', None):
        errors.append('the four-expert flags require --pretrained-teacher')
    if 'fourth_teacher_arch' in f and 'fourth_teacher' not in f:
        errors.append('--fourth-teacher-arch without --fourth-teacher')
    if 'fourth_teacher' in f and not os.path.isfile(f['fourth_teacher']):
        errors.append('--fourth-teacher {} does not exist'.format(f['fourth_teacher']))
    if getattr(args, 'teacher_test_dump', False) and not (getattr(args, 'pretrained_teacher', None)
                                                          and getattr(args, 'source_checkpoint', None)):
        errors.append('--teacher-test-dump requires the configurable teacher (--pretrained-teacher and '
                      '--source-checkpoint)')
    if 'erm_weight' in f and not (math.isfinite(f['erm_weight']) and f['erm_weight'] > 0):
        errors.append('--erm-weight must be finite and > 0')
    if errors:
        raise ValueError('invalid four-expert flags:\n  - ' + '\n  - '.join(errors))


def load_fourth_expert(args, path, num_classes):
    """ImageClassifier of --fourth-teacher-arch with the student's bottleneck, pool and finetune, built in fork_rng."""
    arch = getattr(args, 'fourth_teacher_arch', 'convnext_tiny_ham')
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        backbone = utils.get_model(arch, pretrained=False)
        template = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim,
                                   pool_layer=nn.Identity() if args.no_pool else None, finetune=args.finetune)
    print('=> fourth expert {} from {}'.format(arch, path))
    return load_classifier_checkpoint(template, path).to(device)


# --labeled-sisters (BCN20000 only): after the split, the pool images of a lesion that already has a labeled image
# join the labeled set with their own class, which must be the class of the lesion. The configurable teacher is
# still calibrated and measured on the k per class. Lesions from <root>/split_manifest.csv. Consumes no RNG.
SISTERS_JSON = 'labeled_sisters.json'


def add_sisters_args(parser):
    parser.add_argument('--labeled-sisters', default=argparse.SUPPRESS, action='store_true',
                        help='BCN20000: the pool images of labeled lesions move to the labeled set')
    return parser


def validate_sisters_args(args):
    if getattr(args, 'labeled_sisters', False) and args.data != 'BCN20000':
        raise ValueError('--labeled-sisters is only defined for BCN20000')


def move_sisters(labeled, pools, root, log_dir):
    """(labeled with the sisters appended, pools without them). pools: weak and strong views, same indices."""
    import pool_modes
    assert all(isinstance(p, Subset) and list(p.indices) == list(pools[0].indices) for p in pools), \
        '--labeled-sisters needs the pool as a Subset, the same in both views'
    base = labeled.dataset
    lesions = pool_modes.lesion_ids(base.samples, root)
    targets = base.targets
    class_of = {}
    for i in labeled.indices:
        class_of.setdefault(lesions[i], set()).add(int(targets[i]))
    sisters = [i for i in pools[0].indices if lesions[i] in class_of]
    wrong = [i for i in sisters if class_of[lesions[i]] != {int(targets[i])}]
    if wrong:
        raise ValueError('{} sisters with a class other than their lesion'.format(len(wrong)))
    out = set(sisters)
    new = Subset(base, list(labeled.indices) + sisters)
    new.num_classes = labeled.num_classes
    new_pools = [Subset(p.dataset, [i for i in p.indices if i not in out]) for p in pools]
    per_class = np.bincount([int(targets[i]) for i in sisters], minlength=labeled.num_classes).tolist()
    record = {'n_labeled_orig': len(labeled.indices), 'n_sisters': len(sisters), 'sisters_per_class': per_class,
              'n_lesions_labeled': len(class_of), 'n_lesions_with_sisters': len({lesions[i] for i in sisters}),
              'pool_before': len(pools[0].indices), 'pool_after': len(new_pools[0].indices)}
    with open(os.path.join(log_dir, SISTERS_JSON), 'w', encoding='utf-8') as fh:
        json.dump(record, fh, indent=1)
    print('LABELED_SISTERS ' + json.dumps(record, sort_keys=True))
    return new, new_pools


def add_expert(log_p, logits, log_prior, temperature, bias):
    """log_p + log_softmax((z - log pi) / T + b), renormalized."""
    log_q = log_p + F.log_softmax((logits - log_prior) / temperature + bias, dim=1)
    return log_q - torch.logsumexp(log_q, dim=1, keepdim=True)


def read_lesion_csv(path):
    """{isic_id: lesion_id} from the --teacher-lesion-mean CSV (columns isic_id and lesion_id)."""
    import csv
    with open(path, newline='', encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))
    if not rows or not {'isic_id', 'lesion_id'} <= set(rows[0]):
        raise ValueError('{}: the --teacher-lesion-mean CSV needs columns isic_id and lesion_id'.format(path))
    return {r['isic_id']: r['lesion_id'] for r in rows}


def lesion_groups(paths, lesion_of):
    """Groups of 2+ pool positions sharing a lesion (id = file stem) and the number of mapped images."""
    by_lesion = {}
    n_mapped = 0
    for i, p in enumerate(paths):
        les = lesion_of.get(os.path.splitext(os.path.basename(p))[0])
        if les is None:
            continue
        n_mapped += 1
        by_lesion.setdefault(les, []).append(i)
    return [g for g in by_lesion.values() if len(g) > 1], n_mapped


def apply_lesion_mean(args, teacher_output, student_pool):
    """Replaces each teacher row by the renormalized mean of its lesion; writes lesion_mean.json."""
    f = committee_flags(args)
    if 'teacher_lesion_mean' not in f:
        return teacher_output
    import hashlib
    paths = dataset_relative_paths(student_pool, args.root)
    if len(paths) != teacher_output.size(0):
        raise ValueError('the pool has {} images but the teacher table has {} rows'.format(
            len(paths), teacher_output.size(0)))
    groups, n_mapped = lesion_groups(paths, read_lesion_csv(f['teacher_lesion_mean']))
    out = teacher_output.detach().to('cpu', torch.float64).clone()
    before = out.argmax(dim=1)
    for g in groups:
        idx = torch.as_tensor(g)
        out[idx] = out[idx].mean(dim=0, keepdim=True)
    out = out - torch.logsumexp(out, dim=1, keepdim=True)
    with open(f['teacher_lesion_mean'], 'rb') as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    record = {'csv': f['teacher_lesion_mean'], 'csv_sha256': sha, 'n_pool': len(paths), 'n_with_lesion': n_mapped,
              'n_groups': len(groups), 'n_in_groups': int(sum(len(g) for g in groups)),
              'argmax_changed': int((out.argmax(dim=1) != before).sum())}
    path = os.path.join(args.log, LESION_JSON)
    with open(path + '.tmp', 'w', encoding='utf-8') as fh:
        json.dump(record, fh, indent=1, ensure_ascii=False)
    os.replace(path + '.tmp', path)
    print(LESION_TAG + ' ' + json.dumps(record, ensure_ascii=False, sort_keys=True))
    return out.to(device=teacher_output.device, dtype=teacher_output.dtype)


# --teacher-arch: teacher and source loaded into an ImageClassifier of another architecture (e.g. ResNet-50
# ERM teacher for a ConvNeXt student), built inside fork_rng. Only on the configurable-teacher path.
TEACHER_ARCH_KEYS = ('teacher_arch',)


def add_teacher_arch_args(parser):
    group = parser.add_argument_group('teacher architecture (default: same as the student)')
    group.add_argument('--teacher-arch', default=argparse.SUPPRESS, metavar='ARCH', choices=utils.get_model_names(),
                       help='architecture of the teacher and the source when it differs from the student (e.g. '
                            'resnet50 with -a convnext_tiny_ham). Only with the configurable teacher')
    return parser


def teacher_arch_flags(args):
    return {k: getattr(args, k) for k in TEACHER_ARCH_KEYS if hasattr(args, k)}


def validate_teacher_arch_args(args):
    if not teacher_arch_flags(args):
        return
    errors = []
    if not getattr(args, 'pretrained_teacher', None):
        errors.append('--teacher-arch requires --pretrained-teacher')
    if not teacher_flags_active(args):
        errors.append('--teacher-arch only applies to the configurable teacher (--teacher-view center, --teacher-combine '
                      'poe or --teacher-root): on the plain path, the teacher and the source are loaded by copying the student')
    if pair_flags(args):
        errors.append('--teacher-arch cannot be combined with the pairing flags (the photo teacher is loaded by copying '
                      'the student)')
    if errors:
        raise ValueError('invalid teacher-architecture flags:\n  - ' + '\n  - '.join(errors))


def build_teacher_template(args, classifier, num_classes):
    """The student itself, or a --teacher-arch ImageClassifier built inside fork_rng."""
    arch = teacher_arch_flags(args).get('teacher_arch')
    if arch is None or arch == args.arch:
        return classifier
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        backbone = utils.get_model(arch, pretrained=False)
        template = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim,
                                   pool_layer=nn.Identity() if args.no_pool else None, finetune=args.finetune)
    print('=> {} template for the teacher or the source model (student {})'.format(arch, args.arch))
    return template.to(device)


def parse_source_prior(text, num_classes=None) -> torch.Tensor:
    """'327,514,1099,1113,6705' -> normalized log(pi) (float64, CPU). Accepts counts or proportions."""
    try:
        values = [float(v) for v in str(text).replace(';', ',').split(',') if v.strip()]
    except ValueError:
        raise ValueError('--source-prior-counts is not a comma-separated list of numbers: {!r}'.format(text))
    if not values or not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError('--source-prior-counts must have finite values > 0: {!r}'.format(text))
    if num_classes is not None and len(values) != num_classes:
        raise ValueError('--source-prior-counts has {} values but the dataset has {} classes'.format(
            len(values), num_classes))
    prior = torch.tensor(values, dtype=torch.float64)
    return torch.log(prior / prior.sum())


def parse_source_alpha(text):
    """--source-alpha: 'auto' (reliability rule, see alpha_de_ba) or a number in [0, 1] (weight of the SOURCE)."""
    if str(text).strip().lower() == 'auto':
        return 'auto'
    try:
        value = float(text)
    except (TypeError, ValueError):
        raise ValueError("--source-alpha must be a number in [0, 1] or 'auto': {!r}".format(text))
    if not 0.0 <= value <= 1.0:
        raise ValueError('--source-alpha must be in [0, 1]: {!r}'.format(text))
    return value


def validate_teacher_args(args):
    """Fails early with all problems together; only checks files, so it does not touch the RNG."""
    combine = args.teacher_combine
    errors = []
    if combine != 'poe' and args.teacher_temp is not None:
        errors.append('--teacher-temp is only used with --teacher-combine poe')
    if combine != 'poe' and args.bcts_l2 is not None:
        errors.append('--bcts-l2 is only used with --teacher-combine poe')
    if args.teacher_temp is not None and not args.teacher_temp > 0:
        errors.append('--teacher-temp must be > 0')
    if args.bcts_l2 is not None and not args.bcts_l2 >= 0:
        errors.append('--bcts-l2 must be >= 0')
    if args.source_checkpoint:
        if not args.pretrained_teacher:
            errors.append('--source-checkpoint requires --pretrained-teacher (the ERM teacher; without it the script '
                          'trains ERM and ignores the source). For a source-only teacher: --source-alpha 1')
        if not os.path.isfile(args.source_checkpoint):
            errors.append('--source-checkpoint {} does not exist'.format(args.source_checkpoint))
        if args.source_prior_counts is None:
            errors.append('--source-checkpoint requires --source-prior-counts')
        else:
            try:
                parse_source_prior(args.source_prior_counts)
            except ValueError as e:
                errors.append(str(e))
        if combine == 'mix':
            try:
                parse_source_alpha(args.source_alpha)
            except ValueError as e:
                errors.append(str(e))
    if teacher_flags_active(args):
        if not args.pretrained_teacher:
            errors.append('the teacher flags require --pretrained-teacher (without a teacher the script trains ERM)')
        if combine == 'poe' and not args.source_checkpoint:
            errors.append('--teacher-combine poe requires --source-checkpoint (and --source-prior-counts)')
        if not args.source_checkpoint and args.source_prior_counts is not None:
            errors.append('--source-prior-counts is only used with --source-checkpoint')
        if args.teacher_root is not None and not os.path.isdir(args.teacher_root):
            errors.append('--teacher-root {} does not exist'.format(args.teacher_root))
        if args.teacher_root is not None and args.teacher_view != 'center':
            errors.append('--teacher-root requires --teacher-view center: with the weak view (the default), '
                          'the teacher would see on its image the random crop of the STUDENT pipeline '
                          '(--train-resizing; on clinical images, lesion_preserving with scale 0.5-1), not its own. '
                          'Add --teacher-view center')
    if errors:
        raise ValueError('invalid teacher flags:\n  - ' + '\n  - '.join(errors))
    if combine == 'poe' and args.bcts_l2 is None:
        args.bcts_l2 = DEFAULT_BCTS_L2


def combine_mix(teacher_logits, source_logits, log_prior, source_alpha):
    """Same as profesor_ensamble() but via logsumexp, so a = 0 or 1 avoids log(0)."""
    log_t = F.log_softmax(teacher_logits, dim=1)
    log_s = F.log_softmax(source_logits - log_prior, dim=1)
    log_w = torch.log(torch.tensor([1.0 - source_alpha, source_alpha], dtype=log_t.dtype,
                                   device=log_t.device))  # log 0 = -inf: ok
    return torch.logsumexp(torch.stack([log_t + log_w[0], log_s + log_w[1]], dim=0), dim=0)


def combine_poe(teacher_logits, source_logits, log_prior, t_teacher=1.0, t_source=1.0, bias=None):
    """Product of experts: log_softmax(z_t / T_t) + log_softmax((z_s - log pi) / T_s + b), renormalized."""
    source_in = (source_logits - log_prior) / t_source
    if bias is not None:
        source_in = source_in + bias
    log_p = F.log_softmax(teacher_logits / t_teacher, dim=1) + F.log_softmax(source_in, dim=1)
    return log_p - torch.logsumexp(log_p, dim=1, keepdim=True)


def calibration_nll(logits, labels, num_classes, temperature=1.0, bias=None, class_balanced=True):
    """NLL of softmax(z / T + b). class_balanced=True weights each class equally (NLL under a uniform prior)."""
    z = torch.as_tensor(logits).detach().to('cpu', torch.float64)
    y = torch.as_tensor(labels).detach().to('cpu', torch.long)
    weights = _calibration_weights(y, num_classes, class_balanced)
    logits_cal = z / temperature
    if bias is not None:
        logits_cal = logits_cal + torch.as_tensor(bias, dtype=torch.float64)
    log_q = F.log_softmax(logits_cal, dim=1)
    return float(-(weights * log_q[torch.arange(len(y)), y]).sum())


def _calibration_weights(y, num_classes, class_balanced):
    if class_balanced:
        counts = torch.bincount(y, minlength=num_classes).to(torch.float64)
        return 1.0 / (int((counts > 0).sum()) * counts[y])
    return torch.full((len(y),), 1.0 / len(y), dtype=torch.float64)


def fit_temperature(logits, labels, num_classes, fit_bias=False, l2=0.0, class_balanced=True,
                    t_bounds=CALIB_T_BOUNDS):
    """Fits T (and with fit_bias a per-class bias b: BCTS) by L-BFGS-B on NLL + l2 * |b|^2."""
    from scipy.optimize import minimize
    z = torch.as_tensor(logits).detach().to('cpu', torch.float64)
    y = torch.as_tensor(labels).detach().to('cpu', torch.long)
    weights = _calibration_weights(y, num_classes, class_balanced)
    rows = torch.arange(len(y))
    n_bias = num_classes if fit_bias else 0

    def objective(x):
        theta = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        logits_cal = z / torch.exp(theta[0])
        if fit_bias:
            logits_cal = logits_cal + theta[1:]
        loss = -(weights * F.log_softmax(logits_cal, dim=1)[rows, y]).sum()
        if fit_bias:
            loss = loss + l2 * (theta[1:] ** 2).sum()
        grad, = torch.autograd.grad(loss, theta)
        return loss.item(), grad.numpy().astype(np.float64)

    log_bounds = (math.log(t_bounds[0]), math.log(t_bounds[1]))
    result = minimize(objective, np.zeros(1 + n_bias), jac=True, method='L-BFGS-B',
                      bounds=[log_bounds] + [(None, None)] * n_bias)
    temperature = float(math.exp(result.x[0]))
    bias = [float(v) for v in result.x[1:]] if fit_bias else None
    return {
        'T': temperature, 'b': bias, 'l2': l2 if fit_bias else None, 'class_balanced': class_balanced,
        'nll_before': calibration_nll(z, y, num_classes, class_balanced=class_balanced),
        'nll_after': calibration_nll(z, y, num_classes, temperature, bias, class_balanced=class_balanced),
        'converged': bool(result.success),
        'T_at_bound': not (log_bounds[0] + 1e-6 < result.x[0] < log_bounds[1] - 1e-6), 'n': int(len(y)),
    }


def balanced_accuracy(scores, labels, num_classes) -> float:
    """Mean per-class recall (in %) over the classes present in labels; argmax of scores (logits or log-probs)."""
    pred = torch.as_tensor(scores).detach().argmax(dim=1).cpu()
    y = torch.as_tensor(labels).detach().cpu()
    recalls = [(pred[y == c] == c).double().mean().item() for c in range(num_classes) if bool((y == c).any())]
    return 100.0 * float(np.mean(recalls)) if recalls else float('nan')


def dataset_relative_paths(dataset, root):
    """Paths relative to root in dataset order; handles nested Subset and ConcatDataset."""
    if isinstance(dataset, Subset):
        inner = dataset_relative_paths(dataset.dataset, root)
        return [inner[i] for i in dataset.indices]
    if isinstance(dataset, ConcatDataset):
        return [p for d in dataset.datasets for p in dataset_relative_paths(d, root)]
    samples = getattr(dataset, 'samples', None)
    if samples is None:
        raise TypeError('{} does not expose .samples: the configurable teacher needs a file-based dataset '
                        '(ImageFolder)'.format(type(dataset).__name__))
    return [os.path.relpath(path, root).replace('\\', '/') for path, _ in samples]


def dataset_classes(dataset):
    while isinstance(dataset, (Subset, ConcatDataset)):
        dataset = dataset.dataset if isinstance(dataset, Subset) else dataset.datasets[0]
    return list(getattr(dataset, 'classes', []))


def check_paired_paths(student_paths, teacher_paths, what):
    """Raises a detailed error unless both path lists match one by one; returns the length."""
    student_paths = [p.replace('\\', '/') for p in student_paths]
    teacher_paths = [p.replace('\\', '/') for p in teacher_paths]
    if student_paths == teacher_paths:
        return len(student_paths)
    lines = ["pairing by relative path: '{}' does not match between the student and the teacher".format(what),
             '  student: {} images; teacher: {}'.format(len(student_paths), len(teacher_paths))]
    differ = [i for i, (a, b) in enumerate(zip(student_paths, teacher_paths)) if a != b]
    if differ:
        i = differ[0]
        lines.append("  {} positions differ; the first is {}: student '{}', teacher '{}'".format(
            len(differ), i, student_paths[i], teacher_paths[i]))
    missing = sorted(set(student_paths) - set(teacher_paths))
    extra = sorted(set(teacher_paths) - set(student_paths))
    if missing:
        lines.append('  missing in the teacher: {} (e.g. {})'.format(len(missing), missing[:3]))
    if extra:
        lines.append('  extra in the teacher: {} (e.g. {})'.format(len(extra), extra[:3]))
    if not missing and not extra:
        lines.append('  same paths, in a different order')
    lines.append('  Check that --teacher-root is the same dataset version as the student root (same splits, '
                 'class folders and file names).')
    raise ValueError('\n'.join(lines))


def load_classifier_checkpoint(template: nn.Module, path):
    """Copy of template with the checkpoint weights, in eval and without the student dropout."""
    model = copy.deepcopy(template)
    checkpoint = torch.load(path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    model.load_state_dict(checkpoint)
    model.eval()
    model.as_teacher_model = True
    return model


def calc_outputs(models, dataset, batch_size, workers, print_freq=100, prefix='Computing teacher output: '):
    """Outputs of several models in one pass, so with the weak view all of them see the same crop."""
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                             num_workers=workers, drop_last=False, pin_memory=True)
    batch_time = AverageMeter('Time', ':6.3f')
    progress = ProgressMeter(len(data_loader), [batch_time], prefix=prefix)
    outputs = [[] for _ in models]
    labels = []
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(data_loader):
            images = images.to(device, non_blocking=True)
            for output, model in zip(outputs, models):
                output.append(model(images))
            labels.append(target)
            batch_time.update(time.time() - end)
            end = time.time()
            if i % print_freq == 0:
                progress.display(i)
    return [torch.cat(output, dim=0) for output in outputs], torch.cat(labels, dim=0)


def calc_configurable_teacher_output(args, classifier, classifier_teacher, student_pool, student_labeled,
                                     student_val, weak_augment, val_transform):
    """Teacher table for the configurable path (logits, or mix/poe log-probs); writes teacher_calib.json."""
    num_classes = student_labeled.num_classes
    teacher_root = args.teacher_root if args.teacher_root is not None else args.root
    pool_transform = val_transform if args.teacher_view == 'center' else weak_augment

    # Same indices as the student: the split is not redrawn.
    for name, subset in (('pool', student_pool), ('labeled set', student_labeled)):
        if not isinstance(subset, Subset) or isinstance(subset.dataset, (Subset, ConcatDataset)):
            raise TypeError('the configurable teacher needs the student {} to be a Subset of a file-based '
                            'dataset (ImageFolder branch of utils.get_dataset)'.format(name))
    dataset_class = datasets.__dict__[args.data]
    teacher_pool = Subset(dataset_class(root=teacher_root, split='train', transform=pool_transform),
                          student_pool.indices)
    teacher_labeled = Subset(dataset_class(root=teacher_root, split='train', transform=val_transform),
                             student_labeled.indices)
    teacher_val = dataset_class(root=teacher_root, split='val', transform=val_transform)
    if isinstance(student_val, Subset):
        teacher_val = Subset(teacher_val, student_val.indices)

    classes = dataset_classes(student_pool)
    if dataset_classes(teacher_pool) != classes:
        raise ValueError('teacher classes {} do not match student classes {}'.format(
            dataset_classes(teacher_pool), classes))
    n_pool = check_paired_paths(dataset_relative_paths(student_pool, args.root),
                                dataset_relative_paths(teacher_pool, teacher_root), 'unlabeled pool')
    n_labeled = check_paired_paths(dataset_relative_paths(student_labeled, args.root),
                                   dataset_relative_paths(teacher_labeled, teacher_root), 'labeled set')
    n_val = check_paired_paths(dataset_relative_paths(student_val, args.root),
                               dataset_relative_paths(teacher_val, teacher_root), 'validation')

    mode = args.teacher_combine if args.source_checkpoint else 'none'
    models = [classifier_teacher]
    log_prior = None
    if mode != 'none':
        classifier = build_teacher_template(args, classifier, num_classes)
        models.append(load_classifier_checkpoint(classifier, args.source_checkpoint))
        log_prior = parse_source_prior(args.source_prior_counts, num_classes)
    third = committee_flags(args).get('third_checkpoint')
    if third is not None:
        models.append(load_third_expert(third, num_classes))
    fourth = fourth_flags(args)
    i_fourth = None
    if 'fourth_teacher' in fourth:
        i_fourth = len(models)
        models.append(load_fourth_expert(args, fourth['fourth_teacher'], num_classes))
    w_erm = float(fourth.get('erm_weight', 1.0))

    # The only pass that consumes RNG; with the weak view teacher and source see the same crop.
    pool_outputs, pool_labels = calc_outputs(models, teacher_pool, args.batch_size, args.workers, args.print_freq)
    for output in pool_outputs:
        if output.dim() != 2 or output.size(1) != num_classes:
            raise ValueError('output of shape {} but the dataset has {} classes'.format(tuple(output.shape), num_classes))

    # Auxiliary passes run in fork_rng so the student's RNG stream does not depend on them.
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        labeled_outputs, labeled_labels = calc_outputs(models, teacher_labeled, args.batch_size, args.workers,
                                                       args.print_freq, prefix='Teacher on labeled set: ')
        val_outputs, val_labels = calc_outputs(models, teacher_val, args.batch_size, args.workers,
                                               args.print_freq, prefix='Teacher on validation set: ')
        dump = getattr(args, 'teacher_test_dump', False)
        if dump:
            teacher_test = dataset_class(root=teacher_root, split='test', transform=val_transform)
            test_outputs, test_labels = calc_outputs(models, teacher_test, args.batch_size, args.workers,
                                                     args.print_freq, prefix='Teacher on test set: ')

        def cpu64(t):
            return t.detach().to('cpu', torch.float64)
        splits = ('val', 'labeled', 'pool')
        labels = {'val': val_labels, 'labeled': labeled_labels, 'pool': pool_labels}
        z_teacher = {'val': cpu64(val_outputs[0]), 'labeled': cpu64(labeled_outputs[0]), 'pool': cpu64(pool_outputs[0])}
        z_source = None
        if log_prior is not None:
            z_source = {'val': cpu64(val_outputs[1]), 'labeled': cpu64(labeled_outputs[1]),
                        'pool': cpu64(pool_outputs[1])}
        z_third = None
        if third is not None:
            z_third = {'val': cpu64(val_outputs[2]), 'labeled': cpu64(labeled_outputs[2]),
                       'pool': cpu64(pool_outputs[2])}
        z_fourth = None
        if i_fourth is not None:
            z_fourth = {'val': cpu64(val_outputs[i_fourth]), 'labeled': cpu64(labeled_outputs[i_fourth]),
                        'pool': cpu64(pool_outputs[i_fourth])}
        if dump:
            splits = splits + ('test',)
            labels['test'] = test_labels
            z_teacher['test'] = cpu64(test_outputs[0])
            if z_source is not None:
                z_source['test'] = cpu64(test_outputs[1])
            if z_third is not None:
                z_third['test'] = cpu64(test_outputs[2])
            if z_fourth is not None:
                z_fourth['test'] = cpu64(test_outputs[i_fourth])

        record = {
            'teacher_view': args.teacher_view, 'teacher_combine': mode,
            'student_root': os.path.abspath(args.root), 'teacher_root': os.path.abspath(teacher_root),
            'cross_teacher': os.path.abspath(teacher_root) != os.path.abspath(args.root),
            'pretrained_teacher': args.pretrained_teacher, 'source_checkpoint': args.source_checkpoint,
            'source_prior': None if log_prior is None else [float(v) for v in log_prior.exp()],
            'source_alpha': None, 'source_alpha_rule': None, 'T_t': None, 'T_t_mode': None, 'T_s': None, 'b': None,
            'bcts_l2': args.bcts_l2, 'calibration': {}, 'T_loss': args.T, 'seed': args.seed,
            'num_samples_per_class': args.num_samples_per_class, 'classes': classes,
            'n': {'pool': n_pool, 'labeled': n_labeled, 'val': n_val},
        }
        if teacher_arch_flags(args):
            record.update(teacher_arch=args.teacher_arch, student_arch=getattr(args, 'arch', None))
        source_bcts = None
        if mode == 'none':
            combined = {s: F.log_softmax(z_teacher[s], dim=1) for s in splits}
            teacher_output = pool_outputs[0]  # raw logits, as in the original path
        elif mode == 'mix':
            alpha = parse_source_alpha(args.source_alpha)
            if alpha == 'auto':
                ba_lab = balanced_accuracy(z_source['labeled'] - log_prior, labels['labeled'], num_classes)
                alpha, r = alpha_de_ba(ba_lab)
                record['source_alpha_rule'] = {'ba_source_labeled': ba_lab, 'r': r}
            record['source_alpha'] = float(alpha)
            combined = {s: combine_mix(z_teacher[s], z_source[s], log_prior, float(alpha)) for s in splits}
        else:
            if args.teacher_temp is not None:
                t_teacher = float(args.teacher_temp)
                record['T_t_mode'] = 'fixed'
                record['calibration']['teacher_val'] = {
                    'T': t_teacher, 'class_balanced': True,
                    'nll_before': calibration_nll(z_teacher['val'], labels['val'], num_classes),
                    'nll_after': calibration_nll(z_teacher['val'], labels['val'], num_classes, t_teacher)}
            else:
                fit = fit_temperature(z_teacher['val'], labels['val'], num_classes, fit_bias=False,
                                      class_balanced=True)
                t_teacher = fit['T']
                record['T_t_mode'] = 'nll_val_balanced'
                record['calibration']['teacher_val'] = fit
            fit = fit_temperature(z_source['labeled'] - log_prior, labels['labeled'], num_classes, fit_bias=True,
                                  l2=args.bcts_l2, class_balanced=True)
            record['calibration']['source_labeled'] = fit
            t_source, bias = fit['T'], torch.tensor(fit['b'], dtype=torch.float64)
            record.update(T_t=t_teacher, T_s=t_source, b=fit['b'])
            source_bcts = {s: (z_source[s] - log_prior) / t_source + bias for s in splits}
            if fourth:
                # W = 1 gives combine_poe up to rounding
                combined = {}
                for s in splits:
                    lp = w_erm * F.log_softmax(z_teacher[s] / t_teacher, dim=1) + source_bcts[s].log_softmax(dim=1)
                    combined[s] = lp - torch.logsumexp(lp, dim=1, keepdim=True)
                record.update(erm_weight=w_erm, fourth_teacher=fourth.get('fourth_teacher'),
                              fourth_teacher_arch=(getattr(args, 'fourth_teacher_arch', 'convnext_tiny_ham')
                                                   if 'fourth_teacher' in fourth else None))
            else:
                combined = {s: combine_poe(z_teacher[s], z_source[s], log_prior, t_teacher, t_source, bias)
                            for s in splits}
            if z_third is not None:
                fit_c = fit_temperature(z_third['labeled'] - log_prior, labels['labeled'], num_classes,
                                        fit_bias=True, l2=args.bcts_l2, class_balanced=True)
                record['calibration']['third_labeled'] = fit_c
                t_third, bias_c = fit_c['T'], torch.tensor(fit_c['b'], dtype=torch.float64)
                record.update(third_checkpoint=third, T_c=t_third, b_c=fit_c['b'])
                third_bcts = {s: (z_third[s] - log_prior) / t_third + bias_c for s in splits}
                combined = {s: add_expert(combined[s], z_third[s], log_prior, t_third, bias_c) for s in splits}
            if z_fourth is not None:
                for s in splits:
                    lp = combined[s] + w_erm * F.log_softmax(z_fourth[s] / t_teacher, dim=1)
                    combined[s] = lp - torch.logsumexp(lp, dim=1, keepdim=True)
        if mode != 'none':
            teacher_output = combined['pool'].to(device=pool_outputs[0].device, dtype=torch.float32)
        if dump:
            box = {'test_ids': np.array(dataset_relative_paths(teacher_test, teacher_root)),
                   'val_ids': np.array(dataset_relative_paths(teacher_val, teacher_root)),
                   'pool_ids': np.array(dataset_relative_paths(teacher_pool, teacher_root)),
                   'classes': np.array(classes)}
            for s in ('val', 'test', 'pool'):
                box[s + '_labels'] = torch.as_tensor(labels[s]).cpu().numpy()
                box[s + '_combined'] = combined[s].numpy()
                for name, z in (('teacher', z_teacher), ('source', z_source), ('third', z_third),
                                ('fourth', z_fourth)):
                    if z is not None:
                        box['%s_%s_logits' % (s, name)] = z[s].numpy()
            np.savez_compressed(os.path.join(args.log, 'teacher_test.npz'), **box)
            print('TEACHER_TEST_DUMP teacher_test.npz test {} val {}'.format(len(box['test_ids']),
                                                                             len(box['val_ids'])))

        # 'pool_diag' uses labels the method never sees; on 'labeled' the teacher and BCTS are in-sample.
        record['ba'] = {}
        for s in splits:
            row = {'teacher': balanced_accuracy(z_teacher[s], labels[s], num_classes)}
            if z_source is not None:
                row['source_prior'] = balanced_accuracy(z_source[s] - log_prior, labels[s], num_classes)
            if source_bcts is not None:
                row['source_bcts'] = balanced_accuracy(source_bcts[s], labels[s], num_classes)
            if z_third is not None:
                row['third_bcts'] = balanced_accuracy(third_bcts[s], labels[s], num_classes)
            if z_fourth is not None:
                row['fourth'] = balanced_accuracy(z_fourth[s], labels[s], num_classes)
            row['combined'] = balanced_accuracy(combined[s], labels[s], num_classes)
            record['ba']['pool_diag' if s == 'pool' else s] = row
        # Diagnostic of the soft target the loss actually sees, softmax(y_u / T_loss).
        record['target_quality'] = {}
        for s in splits:
            y_s = torch.as_tensor(labels[s]).long()
            pred = combined[s].argmax(dim=1)
            q_loss = F.softmax(combined[s] / args.T, dim=1)
            onehot = F.one_hot(y_s, num_classes).to(q_loss.dtype)
            record['target_quality']['pool_diag' if s == 'pool' else s] = {
                'recall': [100.0 * float((pred[y_s == c] == c).double().mean()) if bool((y_s == c).any()) else None
                           for c in range(num_classes)],
                'nll_T1': calibration_nll(combined[s], y_s, num_classes),
                'nll_T_loss': calibration_nll(combined[s], y_s, num_classes, temperature=args.T),
                'brier_T_loss': float(((q_loss - onehot) ** 2).sum(dim=1).mean()),
            }

    print('=> configurable teacher: view={}, combination={}, teacher root={}{}'.format(
        args.teacher_view, mode, teacher_root, ' (cross)' if record['cross_teacher'] else ''))
    print('   pairing by relative path OK: pool {}, labeled {}, validation {}'.format(
        n_pool, n_labeled, n_val))
    if mode == 'mix':
        regla = record['source_alpha_rule']
        print('   alpha (weight of the SOURCE) = {:.4f}{}'.format(record['source_alpha'], '' if regla is None else
              ' (reliability rule: source BA without prior on the labeled set = {:.2f}, r = {:.3f})'.format(
                  regla['ba_source_labeled'], regla['r'])))
    if mode == 'poe':
        cal_t, cal_s = record['calibration']['teacher_val'], record['calibration']['source_labeled']
        print('   T_t = {:.3f} ({}; balanced NLL on validation {:.4f} -> {:.4f})'.format(
            record['T_t'], record['T_t_mode'], cal_t['nll_before'], cal_t['nll_after']))
        print('   BCTS source (labeled set): T_s = {:.3f}, b = [{}] (NLL {:.4f} -> {:.4f}){}'.format(
            record['T_s'], ', '.join('{:+.3f}'.format(v) for v in record['b']), cal_s['nll_before'],
            cal_s['nll_after'], '  WARNING: T at the bound' if cal_s['T_at_bound'] else ''))
        if cal_t.get('T_at_bound'):
            print('   WARNING: T_t ended at the bound {}'.format(CALIB_T_BOUNDS))
        if 'third_labeled' in record['calibration']:
            cal_c = record['calibration']['third_labeled']
            print('   BCTS third expert (labeled set): T_c = {:.3f}, b_c = [{}] (NLL {:.4f} -> {:.4f}){}'.format(
                record['T_c'], ', '.join('{:+.3f}'.format(v) for v in record['b_c']), cal_c['nll_before'],
                cal_c['nll_after'], '  WARNING: T at the bound' if cal_c['T_at_bound'] else ''))
    print('   BA (%)              validation     labeled  pool (diag.)')
    for key, name in (('teacher', 'teacher'), ('source_prior', 'source w/o prior'), ('source_bcts', 'source BCTS'),
                      ('third_bcts', 'third BCTS'), ('combined', 'combined')):
        if key in record['ba']['val']:
            print('   {:<18} {:>11.2f} {:>11.2f} {:>13.2f}'.format(
                name, record['ba']['val'][key], record['ba']['labeled'][key], record['ba']['pool_diag'][key]))
    path = os.path.join(args.log, 'teacher_calib.json')
    with open(path + '.tmp', 'w', encoding='utf-8') as f:
        json.dump(record, f, indent=1, ensure_ascii=False)
    os.replace(path + '.tmp', path)
    print('TEACHER_CALIB ' + json.dumps(record, ensure_ascii=False, sort_keys=True))
    return teacher_output


def main(args: argparse.Namespace):
    import time
    _train_start = time.time()

    logger = CompleteLogger(args.log, args.phase)
    print(args)
    validate_teacher_args(args)
    validate_pair_args(args)
    validate_committee_args(args)
    validate_teacher_arch_args(args)
    validate_fourth_args(args)
    validate_sisters_args(args)
    utils.validar_soporte(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    weak_augment = utils.get_train_transform(args.train_resizing, random_horizontal_flip=True,
                                             norm_mean=args.norm_mean, norm_std=args.norm_std)
    strong_augment = utils.get_train_transform(args.train_resizing, random_horizontal_flip=True,
                                               auto_augment=args.auto_augment,
                                               norm_mean=args.norm_mean, norm_std=args.norm_std)
    labeled_train_transform = MultipleApply([weak_augment, strong_augment])
    val_transform = utils.get_val_transform(args.val_resizing, norm_mean=args.norm_mean, norm_std=args.norm_std)
    print('labeled_train_transform: ', labeled_train_transform)
    print('weak_augment (input transform for teacher model): ', weak_augment)
    print('strong_augment (input transform for student model): ', strong_augment)
    print('val_transform:', val_transform)

    labeled_train_dataset, weak_augmented_unlabeled_dataset, val_dataset = \
        utils.get_dataset(args.data,
                          args.num_samples_per_class,
                          args.root, labeled_train_transform,
                          val_transform,
                          unlabeled_train_transform=weak_augment,
                          seed=args.seed,
                          **utils.kw_muestreo(args),
                          **utils.kw_pool_exclude(args))
    if args.val_per_class is not None:
        val_dataset = utils.submuestrear_validacion(val_dataset, args.val_per_class, args.seed)
    _, strong_augmented_unlabeled_dataset, _ = \
        utils.get_dataset(args.data,
                          args.num_samples_per_class,
                          args.root, labeled_train_transform,
                          val_transform,
                          unlabeled_train_transform=strong_augment,
                          seed=args.seed,
                          **utils.kw_muestreo(args),
                          **utils.kw_pool_exclude(args))

    teacher_labeled_dataset = labeled_train_dataset  # the teacher is calibrated on the k per class
    if getattr(args, 'labeled_sisters', False):
        labeled_train_dataset, (weak_augmented_unlabeled_dataset, strong_augmented_unlabeled_dataset) = move_sisters(
            labeled_train_dataset, [weak_augmented_unlabeled_dataset, strong_augmented_unlabeled_dataset], args.root,
            args.log)
    strong_augmented_unlabeled_dataset = utils.convert_dataset(strong_augmented_unlabeled_dataset)
    utils.guardar_ids_etiquetados(labeled_train_dataset, args.root, args.log)
    utils.guardar_ids_validacion(val_dataset, args.root, args.log)
    utils.guardar_pool_exclude(weak_augmented_unlabeled_dataset, args.log)
    # Both get_dataset calls above draw the same split.
    log_pi_eval, _, firma_soporte = utils.soporte_etiquetado(args, labeled_train_dataset, args.log, device)
    print("labeled_dataset_size: ", len(labeled_train_dataset))
    print('unlabeled_dataset_size: ', len(weak_augmented_unlabeled_dataset))
    print("val_dataset_size: ", len(val_dataset))

    labeled_train_loader = DataLoader(labeled_train_dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.workers, drop_last=True, persistent_workers=(args.workers > 0), pin_memory=True)
    # persistent_workers: on Windows each worker re-creation is a spawn that re-imports torch.
    unlabeled_train_loader = DataLoader(strong_augmented_unlabeled_dataset, batch_size=args.batch_size, shuffle=True,
                                        num_workers=args.workers, drop_last=True,
                                        persistent_workers=(args.workers > 0), pin_memory=True)
    labeled_train_iter = ForeverDataIterator(labeled_train_loader)
    unlabeled_train_iter = ForeverDataIterator(unlabeled_train_loader)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    print("=> using pre-trained model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch, pretrained_checkpoint=args.pretrained_backbone)
    num_classes = labeled_train_dataset.num_classes
    pool_layer = nn.Identity() if args.no_pool else None
    classifier = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim, pool_layer=pool_layer,
                                 finetune=args.finetune).to(device)
    import os as _os
    arch_log = open(_os.path.join(args.log, 'architecture.txt'), 'w')
    arch_log.write(str(classifier))
    arch_log.close()
    print(f"Architecture saved to {args.log}/architecture.txt")

    if args.pretrained_teacher:
        classifier_teacher = copy.deepcopy(classifier)
        if teacher_arch_flags(args):
            classifier_teacher = copy.deepcopy(build_teacher_template(args, classifier, num_classes))
        checkpoint = torch.load(args.pretrained_teacher, map_location='cpu')
        # erm.py's best.pth wraps the weights in a dict; keys match because dropout has no parameters.
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        classifier_teacher.load_state_dict(checkpoint)
        classifier_teacher.eval()
        classifier_teacher.as_teacher_model = True

        if not teacher_flags_active(args):
            if not args.source_checkpoint:
                print('compute outputs of the teacher network')
                teacher_output = calc_teacher_output(classifier_teacher, weak_augmented_unlabeled_dataset)
            else:
                import numpy as np
                classifier_source = copy.deepcopy(classifier)
                ck = torch.load(args.source_checkpoint, map_location='cpu')
                if isinstance(ck, dict) and 'state_dict' in ck:
                    ck = ck['state_dict']
                classifier_source.load_state_dict(ck)
                classifier_source.eval()
                classifier_source.as_teacher_model = True
                conteos = torch.tensor([float(x) for x in args.source_prior_counts.split(',')], device=device)
                assert len(conteos) == num_classes, 'source-prior-counts needs one value per class'
                log_prior = torch.log(conteos / conteos.sum())
                if args.source_alpha == 'auto':
                    alpha, ba, r = alpha_por_confiabilidad(classifier_source, log_prior,
                                                           labeled_train_dataset.indices, val_transform)
                    print('FUENTE reliability rule: source BA without prior on the labeled set = {:.2f}, '
                          'r = {:.3f}, alpha = {:.4f}'.format(ba, r, alpha))
                else:
                    alpha = float(args.source_alpha)
                print('compute outputs of the teacher network and of the source model')
                z_t, z_s = calc_outputs_dual(classifier_teacher, classifier_source, weak_augmented_unlabeled_dataset)
                teacher_output, p_s, p_t = profesor_ensamble(z_t, z_s, log_prior, alpha)
                print('FUENTE alpha={:.4f} prior={} acuerdo_argmax_fuente_erm={:.3f}'.format(
                    alpha, [round(float(x), 4) for x in (conteos / conteos.sum())],
                    float((p_s.argmax(1) == p_t.argmax(1)).float().mean())))
                del classifier_source
        else:
            # train() indexes this pool by idx_u: strong_augmented_unlabeled_dataset.dataset
            teacher_output = calc_configurable_teacher_output(
                args, classifier, classifier_teacher, strong_augmented_unlabeled_dataset.dataset,
                teacher_labeled_dataset, val_dataset, weak_augment, val_transform)

    if args.pretrained_teacher and pair_flags(args):
        # The photo teacher reuses the flags without --teacher-root; its teacher_calib.json goes to <log>/photo_teacher/
        def photo_output():
            photo_args = copy.copy(args)
            photo_args.teacher_root = None
            photo_args.pretrained_teacher = args.photo_teacher
            photo_args.log = os.path.join(args.log, 'photo_teacher')
            os.makedirs(photo_args.log, exist_ok=True)
            photo_teacher = load_classifier_checkpoint(classifier, args.photo_teacher).to(device)
            return calc_configurable_teacher_output(
                photo_args, classifier, photo_teacher, strong_augmented_unlabeled_dataset.dataset,
                teacher_labeled_dataset, val_dataset, weak_augment, val_transform)
        teacher_output = apply_pair_flags(args, teacher_output, photo_output)

    if args.pretrained_teacher and 'teacher_lesion_mean' in committee_flags(args):
        teacher_output = apply_lesion_mean(args, teacher_output, strong_augmented_unlabeled_dataset.dataset)

    if args.lr_scheduler == 'exp':
        optimizer = SGD(classifier.get_parameters(), args.lr, momentum=0.9, weight_decay=args.wd, nesterov=True)
        lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    else:
        optimizer = SGD(classifier.get_parameters(base_lr=args.lr), args.lr, momentum=0.9, weight_decay=args.wd,
                        nesterov=True)
        lr_scheduler = utils.get_cosine_scheduler_with_warmup(optimizer, args.epochs * args.iters_per_epoch)

    if args.phase == 'test':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)
        acc1, avg = utils.validate(val_loader, classifier, args, device, num_classes, **utils.kw_eval(log_pi_eval))
        print(acc1)
        return

    best_acc1 = 0.0
    best_avg = 0.0
    es_counter = 0

    start_epoch = 0
    if args.resume:
        print(f"Resume since {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        classifier.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint.get('best_acc1', 0.0)
        best_avg = checkpoint.get('best_avg', 0.0)
        print(f"Starting on epoch {start_epoch}, best_acc1={best_acc1:.1f}")
        es_counter = checkpoint.get('es_counter', 0)
        utils.soporte_resume(firma_soporte, checkpoint)

    for epoch in range(start_epoch, args.epochs):
        print(lr_scheduler.get_lr())
        _timer.start("1. Train epoch")

        if args.pretrained_teacher:
            train(labeled_train_iter, unlabeled_train_iter, classifier, teacher_output, optimizer, lr_scheduler,
                  epoch, args)
        else:
            utils.empirical_risk_minimization(labeled_train_iter, classifier, optimizer, lr_scheduler, epoch, args,
                                              device)

        _timer.stop()
        _timer.start("2. Validation")
        acc1, avg = utils.validate(val_loader, classifier, args, device, num_classes, **utils.kw_eval(log_pi_eval))

        _timer.stop()
        _timer.start("3. Checkpoint save")
        utils.atomic_torch_save({
            'epoch': epoch + 1,
            'state_dict': classifier.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'best_acc1': max(acc1, best_acc1),
            'best_avg': max(avg, best_avg),
            'es_counter': 0 if avg > best_avg else es_counter + 1,
            **utils.soporte_checkpoint(firma_soporte),
        }, logger.get_checkpoint_path('latest'))
        _timer.stop()
        if avg > best_avg:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))

        if avg > best_avg:
            es_counter = 0
        else:
            es_counter += 1
            if es_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break
        best_acc1 = max(acc1, best_acc1)
        best_avg = max(avg, best_avg)


    _timer.start("4. Final test eval")
    import numpy as np
    import tllib.vision.datasets as datasets
    from sklearn.metrics import (confusion_matrix, f1_score,
                                 precision_score, recall_score, roc_auc_score,
                                 classification_report)
    test_dataset = datasets.__dict__[args.data](root=args.root, split='test', transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        classifier.load_state_dict(checkpoint['state_dict'])
    else:
        classifier.load_state_dict(checkpoint)
    classifier.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    all_preds_crudo = []  # raw argmax z, only with --logit-adjust-eval labeled
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = classifier(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            if log_pi_eval is not None:  # report, matrix and AUC with z - log pi_l (the run's decision rule)
                all_preds_crudo.extend(outputs.argmax(dim=1).cpu().numpy())
                outputs = outputs - log_pi_eval.to(device=outputs.device, dtype=outputs.dtype)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    print(classification_report(all_labels, all_preds, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print("F1 macro:    {:.4f}".format(f1_score(all_labels, all_preds, average='macro')))
    print("F1 weighted: {:.4f}".format(f1_score(all_labels, all_preds, average='weighted')))
    print("AUC-ROC macro: {:.4f}".format(roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')))
    if log_pi_eval is not None:
        utils.reportar_decision_test(all_labels, all_preds_crudo, all_preds, num_classes)
    _timer.stop()
    _timer.report()
    _elapsed = time.time() - _train_start
    print(f"Total training time: {_elapsed/60:.1f} min ({_elapsed:.1f}s)")
    print("best_acc1 = {:3.1f}".format(best_acc1))
    print('best_avg = {:3.1f}'.format(best_avg))
    logger.close()


def train(labeled_train_iter: ForeverDataIterator, unlabeled_train_iter: ForeverDataIterator, model, teacher_output,
          optimizer: SGD, lr_scheduler: LambdaLR, epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':2.2f')
    data_time = AverageMeter('Data', ':2.1f')
    cls_losses = AverageMeter('Cls Loss', ':3.2f')
    self_training_losses = AverageMeter('Self Training Loss', ':3.2f')
    losses = AverageMeter('Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cls_losses, self_training_losses, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    self_training_criterion = CrossEntropyLoss().to(device)
    model.train()

    end = time.time()
    batch_size = args.batch_size
    for i in range(args.iters_per_epoch):
        (x_l, x_l_strong), labels_l = next(labeled_train_iter)
        x_l = x_l.to(device, non_blocking=True)
        x_l_strong = x_l_strong.to(device, non_blocking=True)
        labels_l = labels_l.to(device, non_blocking=True)

        idx_u, (x_u_strong, _) = next(unlabeled_train_iter)
        idx_u = idx_u.to(device, non_blocking=True)
        x_u_strong = x_u_strong.to(device, non_blocking=True)

        data_time.update(time.time() - end)

        optimizer.zero_grad()

        y_l = model(x_l)
        y_l_strong = model(x_l_strong)
        cls_loss = F.cross_entropy(y_l, labels_l) + args.trade_off_cls_strong * F.cross_entropy(y_l_strong, labels_l)
        cls_loss.backward()

        y_u = teacher_output[idx_u]
        y_u_strong = model(x_u_strong)
        self_training_loss = args.trade_off_self_training * self_training_criterion(y_u_strong / args.T, y_u / args.T)
        self_training_loss.backward()

        loss = cls_loss + self_training_loss
        losses.update(loss.item(), batch_size)
        cls_losses.update(cls_loss.item(), batch_size)
        self_training_losses.update(self_training_loss.item(), batch_size)

        cls_acc = accuracy(y_l, labels_l)[0]
        cls_accs.update(cls_acc.item(), batch_size)

        optimizer.step()
        lr_scheduler.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Noisy Student for Semi Supervised Learning')
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA',
                        help='dataset: ' + ' | '.join(utils.get_dataset_names()))
    parser.add_argument('--num-samples-per-class', default=4, type=int,
                        help='number of labeled samples per class')
    parser.add_argument('--train-resizing', default='default', type=str)
    parser.add_argument('--val-resizing', default='default', type=str)
    parser.add_argument('--norm-mean', default=(0.485, 0.456, 0.406), type=float, nargs='+',
                        help='normalization mean')
    parser.add_argument('--norm-std', default=(0.229, 0.224, 0.225), type=float, nargs='+',
                        help='normalization std')
    parser.add_argument('--auto-augment', default='rand-m10-n2-mstd2', type=str,
                        help='AutoAugment policy (default: rand-m10-n2-mstd2)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50', choices=utils.get_model_names(),
                        help='backbone architecture: ' + ' | '.join(utils.get_model_names()) + ' (default: resnet50)')
    parser.add_argument('--bottleneck-dim', default=1024, type=int,
                        help='dimension of bottleneck')
    parser.add_argument('--no-pool', action='store_true', default=False,
                        help='no pool layer after the feature extractor')
    parser.add_argument('--pretrained-backbone', default=None, type=str,
                        help="pretrained checkpoint of the backbone "
                             "(default: None, use the ImageNet supervised pretrained backbone)")
    parser.add_argument('--finetune', action='store_true', default=False,
                        help='whether to use 10x smaller lr for backbone')
    parser.add_argument('--pretrained-teacher', default=None, type=str,
                        help='pretrained checkpoint of the teacher model')
    parser.add_argument('--source-checkpoint', default=None, type=str,
                        help='source model (HAM) for the ensemble teacher; without it, the plain teacher')
    parser.add_argument('--source-prior-counts', default=None, type=str,
                        help='per-class counts of the source training set, comma-separated (pi_source)')
    parser.add_argument('--source-alpha', default='0.5', type=str,
                        help="weight of the source in the teacher: a number, or 'auto' (reliability rule)")
    parser.add_argument('--trade-off-cls-strong', default=0.1, type=float,
                        help='the trade-off hyper-parameter of cls loss on strong augmented labeled data')
    parser.add_argument('--trade-off-self-training', default=1, type=float,
                        help='the trade-off hyper-parameter of self training loss')
    parser.add_argument('--T', default=2, type=float,
                        help='temperature')
    parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.003, type=float, metavar='LR', dest='lr',
                        help='initial learning rate')
    parser.add_argument('--lr-scheduler', default='exp', type=str, choices=['exp', 'cos'],
                        help='learning rate decay strategy')
    parser.add_argument('--lr-gamma', default=0.0004, type=float,
                        help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float,
                        help='parameter for lr scheduler')
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float, metavar='W',
                        help='weight decay (default:5e-4)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=40, type=int, metavar='N',
                        help='number of total epochs to run (default: 40)')
    parser.add_argument('-i', '--iters-per-epoch', default=500, type=int,
                        help='number of iterations per epoch (default: 500)')
    parser.add_argument('-p', '--print-freq', default=100, type=int, metavar='N',
                        help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training ')
    parser.add_argument("--log", default='noisy_student', type=str,
                        help="where to save logs, checkpoints and debugging images")
    parser.add_argument("--phase", default='train', type=str, choices=['train', 'test'],
                        help="when phase is 'test', only test the model")
    parser.add_argument('--patience', default=15, type=int,
                        help='early stopping patience (default: 15)')
    parser.add_argument('--resume', default=None, type=str,
                        help='path to checkpoint to resume training from')
    add_teacher_args(parser)
    add_budget_args(parser)
    utils.add_label_sampling_args(parser)
    utils.add_pool_exclude_args(parser)
    add_pair_args(parser)
    add_committee_args(parser)
    add_teacher_arch_args(parser)
    add_fourth_expert_args(parser)
    add_sisters_args(parser)
    args = parser.parse_args()
    main(args)
