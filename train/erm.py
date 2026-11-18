"""ERM baseline: supervised training on the labeled subset (or on all training data with --oracle).

Adapted from TLlib (Baixu Chen). Optional flags, all off by default: initialization from a full source
classifier (with prior correction of the head bias), evaluation at epoch 0, LP-FT, and L2-SP or Co-Tuning
regularization towards the source model.
"""
import random
import time
import warnings
import argparse
import shutil
import contextlib
import hashlib
import io
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
from torch.utils.data import DataLoader, ConcatDataset

import utils
from tllib.vision.transforms import MultipleApply
from tllib.utils.metric import accuracy
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.utils.data import ForeverDataIterator
from tllib.utils.logger import CompleteLogger

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


# Source initialization (--init-classifier, --init-prior-counts, --eval-epoch0) consumes no RNG: with the
# same seed, data order and augmentations match plain ERM; only the starting point differs.


def add_init_args(parser):
    group = parser.add_argument_group('initialization from the source model (default: previous behavior)')
    group.add_argument('--init-classifier', default=None, type=str,
                       help='best.pth of a FULL ImageClassifier (backbone + bottleneck + head): the erm.py checkpoint '
                            'dict with state_dict, or a bare state_dict. Strict loading. If --pretrained-backbone is '
                            'also given, it must hold the same backbone (compared tensor by tensor); without '
                            '--pretrained-backbone, the backbone comes from this checkpoint')
    group.add_argument('--init-prior-counts', default=None, type=str,
                       help='per-class counts of the source training set, comma separated (e.g. '
                            '210,231,612,447,4084): subtracts log(pi) from the loaded head bias, giving the '
                            'prior-free source (z - log pi). Requires --init-classifier')
    group.add_argument('--eval-epoch0', action='store_true', default=False,
                       help='validate before training and keep that state as a best.pth candidate (epoch 0 in the '
                            'checkpoint). Consumes no RNG. Not repeated when resuming with --resume')
    return parser


def parse_init_prior(text, num_classes=None):
    """Normalized counts -> log(pi) in float64."""
    try:
        values = [float(x) for x in str(text).split(',')]
    except ValueError:
        raise ValueError('--init-prior-counts is not a comma-separated list of numbers: {!r}'.format(text))
    if not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError('--init-prior-counts must have finite values > 0: {!r}'.format(text))
    if num_classes is not None and len(values) != num_classes:
        raise ValueError('--init-prior-counts has {} values but the model has {} classes'.format(len(values), num_classes))
    prior = torch.tensor(values, dtype=torch.float64)
    return torch.log(prior / prior.sum())


def validate_init_args(args):
    init_classifier = getattr(args, 'init_classifier', None)
    prior_counts = getattr(args, 'init_prior_counts', None)
    errors = []
    if prior_counts is not None:
        if init_classifier is None:
            errors.append('--init-prior-counts requires --init-classifier (without the source head there is no bias '
                          'to correct)')
        try:
            parse_init_prior(prior_counts)
        except ValueError as e:
            errors.append(str(e))
    if init_classifier is not None and not os.path.isfile(init_classifier):
        errors.append('no existe --init-classifier {}'.format(init_classifier))
    if errors:
        raise ValueError('invalid source-initialization flags:\n  - ' + '\n  - '.join(errors))


def _plain(value):
    return value.item() if hasattr(value, 'item') else value


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def init_from_source(classifier, args):
    """Strict load of the full source classifier; refuses a different --pretrained-backbone."""
    path = args.init_classifier
    checkpoint = torch.load(path, map_location='cpu')
    meta = {}
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        meta = {k: _plain(checkpoint[k]) for k in ('epoch', 'best_acc1', 'best_avg') if k in checkpoint}
        checkpoint = checkpoint['state_dict']
    if not isinstance(checkpoint, dict) or not checkpoint:
        raise ValueError('--init-classifier {}: no state_dict found'.format(path))

    backbone_check = 'no --pretrained-backbone: the backbone comes from --init-classifier'
    if args.pretrained_backbone:
        own = classifier.backbone.state_dict()
        differ = [k for k, v in own.items() if 'backbone.' + k not in checkpoint
                  or not torch.equal(v.detach().cpu(), checkpoint['backbone.' + k].detach().cpu())]
        if differ:
            raise ValueError('--pretrained-backbone ({}) and --init-classifier ({}) contain different backbones: {} of {} '
                             'tensors differ (e.g. {}). With --init-classifier the backbone comes from that '
                             'checkpoint: pass the same file to --pretrained-backbone or drop it'.format(
                                 args.pretrained_backbone, path, len(differ), len(own), differ[:3]))
        backbone_check = 'same as --pretrained-backbone ({} tensors)'.format(len(own))
    try:
        classifier.load_state_dict(checkpoint)
    except RuntimeError as e:
        raise RuntimeError('--init-classifier {} does not match the ImageClassifier of this run (architecture, '
                           '--bottleneck-dim, --no-pool or number of classes): {}'.format(path, e))

    record = {'init_classifier': os.path.abspath(path), 'sha256': _sha256(path), 'checkpoint_meta': meta,
              'n_tensors': len(checkpoint), 'backbone': backbone_check, 'prior_counts': None}
    print('INIT_CLASSIFIER {} | {} tensors (backbone + bottleneck + head) | checkpoint {} | backbone: {}'.format(
        path, len(checkpoint), meta or 'no metadata', backbone_check))
    if args.init_prior_counts is not None:
        head_bias = getattr(classifier.head, 'bias', None)
        if head_bias is None:
            raise ValueError('--init-prior-counts: the head of --init-classifier has no bias')
        log_prior = parse_init_prior(args.init_prior_counts, classifier.num_classes)
        with torch.no_grad():
            before = head_bias.detach().cpu().double().clone()
            head_bias.sub_(log_prior.to(device=head_bias.device, dtype=head_bias.dtype))
            after = head_bias.detach().cpu().double().clone()
        record.update(prior_counts=[float(x) for x in args.init_prior_counts.split(',')],
                      prior=[float(x) for x in log_prior.exp()], log_prior=[float(x) for x in log_prior],
                      head_bias_before=[float(x) for x in before], head_bias_after=[float(x) for x in after])
        print('INIT_PRIOR pi = {} | head bias {} -> {} (bias - log pi)'.format(
            [round(x, 4) for x in record['prior']], [round(x, 4) for x in record['head_bias_before']],
            [round(x, 4) for x in record['head_bias_after']]))
    with open(os.path.join(args.log, 'init_classifier.json'), 'w', encoding='utf-8') as f:
        json.dump(record, f, indent=1)
    return record


def eval_epoch0(val_loader, classifier, optimizer, lr_scheduler, logger, args, num_classes, log_prior=None,
                firma_soporte=None, firma_lista=None):
    """Validates the initial state and saves it as latest.pth and best.pth with epoch 0."""
    # Creating the val_loader iterator draws from the torch RNG: restore it so training is unchanged.
    cpu_state, py_state, np_state = torch.get_rng_state(), random.getstate(), np.random.get_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        # Hide '* Mean Cls': curve parsers expect one such line per trained epoch.
        with contextlib.redirect_stdout(io.StringIO()):
            acc1, avg = utils.validate(val_loader, classifier, args, device, num_classes, log_prior=log_prior)
    finally:
        torch.set_rng_state(cpu_state)
        random.setstate(py_state)
        np.random.set_state(np_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
    print('EVAL_EPOCH0 acc1={:.3f} mean_cls={:.3f} | initial state, before training: kept in best.pth (epoch 0) '
          'until an epoch beats it'.format(acc1, avg))
    utils.atomic_torch_save({
        'epoch': 0,
        'state_dict': classifier.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'best_acc1': acc1,
        'best_avg': avg,
        'es_counter': 0,
        **utils.soporte_checkpoint(firma_soporte),
        **utils.lista_checkpoint(firma_lista),
    }, logger.get_checkpoint_path('latest'))
    shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
    return acc1, avg


# LP-FT (Kumar et al., ICLR 2022): --lp-epochs N freezes the backbone (no gradient, BatchNorm in eval) for
# the first N epochs, then trains everything with the same optimizer and schedule. Consumes no RNG.


def add_lpft_args(parser):
    group = parser.add_argument_group('LP-FT (default: previous behavior)')
    group.add_argument('--lp-epochs', default=0, type=int,
                       help='LP-FT (Kumar et al. 2022): the first N epochs train only bottleneck and head, with the '
                            'backbone frozen (no gradient and in eval mode: its BatchNorm layers are not updated); '
                            'from epoch N+1 everything trains, with the same optimizer and schedule. 0 (default): '
                            'full fine-tuning from the start. N >= --epochs: LP only')
    return parser


def validate_lpft_args(args):
    lp_epochs = getattr(args, 'lp_epochs', 0)
    if isinstance(lp_epochs, bool) or not isinstance(lp_epochs, int) or lp_epochs < 0:
        raise ValueError('--lp-epochs must be an integer >= 0: {!r}'.format(lp_epochs))


class LinearProbePhase:
    """Freezes the backbone for epochs < lp_epochs; SGD skips parameters without gradient."""

    def __init__(self, classifier, lp_epochs):
        self.backbone = classifier.backbone
        self.lp_epochs = lp_epochs
        self.params = [p for p in self.backbone.parameters() if p.requires_grad]
        self.hook = None
        self.announced = False

    @staticmethod
    def _eval_pre_hook(module, inputs):
        # empirical_risk_minimization calls model.train() each epoch; keep BatchNorm statistics frozen.
        if module.training:
            module.eval()

    def set_epoch(self, epoch):
        if epoch < self.lp_epochs:
            if self.hook is None:
                for p in self.params:
                    p.requires_grad_(False)
                self.hook = self.backbone.register_forward_pre_hook(self._eval_pre_hook)
            print('LP_FT epoch {} of {} of LP: backbone frozen (no gradient, BatchNorm in eval); training '
                  'bottleneck and head'.format(epoch + 1, self.lp_epochs))
        elif self.hook is not None or not self.announced:
            self.release()
            print('LP_FT end of LP (--lp-epochs {}): from epoch {} the whole network is trained'.format(self.lp_epochs,
                                                                                                epoch + 1))
        self.announced = True

    def release(self):
        if self.hook is not None:
            self.hook.remove()
            self.hook = None
            for p in self.params:
                p.requires_grad_(True)


# CT: --reg l2sp (Li et al., ICML 2018; replaces weight decay) or --reg cotuning (You et al., NeurIPS 2020).
# Flags use default=SUPPRESS so the Namespace is unchanged without --reg. Defaults from the papers and TLlib.
CT_DEFECTOS = dict(l2sp_alpha=0.01, l2sp_beta=0.01, cotuning_lambda=2.3)
CT_FLAGS = ('l2sp_alpha', 'l2sp_beta', 'cotuning_lambda', 'cotuning_source')


def add_ct_args(parser):
    group = parser.add_argument_group('CT: regularization towards the source model (default: off)')
    group.add_argument('--reg', default=argparse.SUPPRESS, choices=['l2sp', 'cotuning'],
                       help='l2sp or cotuning; without the flag, plain ERM')
    group.add_argument('--l2sp-alpha', default=argparse.SUPPRESS, type=float, help='weight of ||w - w0||^2 (0.01)')
    group.add_argument('--l2sp-beta', default=argparse.SUPPRESS, type=float,
                       help='weight of ||w||^2 on bottleneck and head (0.01)')
    group.add_argument('--cotuning-lambda', default=argparse.SUPPRESS, type=float, help='Co-Tuning weight (2.3)')
    group.add_argument('--cotuning-source', default=argparse.SUPPRESS, type=str,
                       help='best.pth of the full source ImageClassifier (backbone + bottleneck + head)')
    return parser


def ct_reg(args):
    return getattr(args, 'reg', None)


def ct_param(args, nombre):
    return getattr(args, nombre, CT_DEFECTOS.get(nombre))


def validate_ct_args(args):
    reg = ct_reg(args)
    errores = []
    sueltos = [n for n in CT_FLAGS if hasattr(args, n)]
    if reg is None:
        if sueltos:
            errores.append('CT flags without --reg: ' + ', '.join(sueltos))
    else:
        if not getattr(args, 'pretrained_backbone', None):
            errores.append('--reg {} requires --pretrained-backbone'.format(reg))
        if getattr(args, 'oracle', False):
            errores.append('CT cannot be combined with --oracle')
        if getattr(args, 'eval_epoch0', False) or getattr(args, 'init_classifier', None) or getattr(args, 'lp_epochs', 0):
            errores.append('CT cannot be combined with --eval-epoch0, --init-classifier or LP-FT')
        if reg == 'l2sp':
            if [n for n in sueltos if n.startswith('cotuning')]:
                errores.append('Co-Tuning flags with --reg l2sp')
            for n in ('l2sp_alpha', 'l2sp_beta'):
                v = ct_param(args, n)
                if not (math.isfinite(v) and v >= 0):
                    errores.append('--{} must be finite and >= 0: {!r}'.format(n.replace('_', '-'), v))
        else:
            if [n for n in sueltos if n.startswith('l2sp')]:
                errores.append('L2-SP flags with --reg cotuning')
            src = getattr(args, 'cotuning_source', None)
            if not src or not os.path.isfile(src):
                errores.append('--reg cotuning requires --cotuning-source pointing to an existing file: {!r}'.format(src))
            v = ct_param(args, 'cotuning_lambda')
            if not (math.isfinite(v) and v > 0):
                errores.append('--cotuning-lambda must be finite and > 0: {!r}'.format(v))
    if errores:
        raise ValueError('invalid CT flags:\n  - ' + '\n  - '.join(errores))


class L2SP:
    """alpha/2 sum ||w - w0||^2 (backbone) + beta/2 sum ||w||^2 (everything outside the backbone)."""

    def __init__(self, classifier, w0, alpha, beta):
        self.backbone = classifier.backbone
        propios = dict(self.backbone.named_parameters())
        if set(propios) != set(w0):
            raise ValueError('L2-SP: w0 does not have the same parameters as the backbone ({} vs {})'.format(
                len(w0), len(propios)))
        self.w0 = {n: w0[n].detach().to(device=p.device, dtype=p.dtype).clone() for n, p in propios.items()}
        self.nuevos = [p for n, p in classifier.named_parameters() if not n.startswith('backbone.')]
        self.alpha, self.beta = float(alpha), float(beta)

    def __call__(self, model, x_l, labels_l):
        sp = sum(((p - self.w0[n]) ** 2).sum() for n, p in self.backbone.named_parameters())
        l2 = sum((p ** 2).sum() for p in self.nuevos)
        return 0.5 * self.alpha * sp + 0.5 * self.beta * l2


def relacion_cotuning(fuente, loader, num_classes, device):
    """p(y_source | y_target): softmax of the frozen source averaged per target class (one pass over the loader)."""
    fuente.eval()
    n_src = fuente.head.out_features
    suma = torch.zeros(num_classes, n_src, dtype=torch.float64)
    cuenta = torch.zeros(num_classes, dtype=torch.float64)
    with torch.no_grad():
        for (x, _), y in loader:
            p = torch.softmax(fuente(x.to(device)).double(), dim=1).cpu()
            suma.index_add_(0, y.cpu(), p)
            cuenta += torch.bincount(y.cpu(), minlength=num_classes).double()
    if (cuenta == 0).any():
        raise ValueError('Co-Tuning: some target classes have no labeled samples: {}'.format(cuenta.tolist()))
    return (suma / cuenta[:, None]).float()


class CoTuning:
    """lambda * CE(source head on the model features, p(y_source | y_target)); the head copy also trains."""

    def __init__(self, fuente, relacion, lam, device):
        import copy
        self.bottleneck = copy.deepcopy(fuente.bottleneck).to(device).train()
        self.head = copy.deepcopy(fuente.head).to(device).train()
        for p in list(self.bottleneck.parameters()) + list(self.head.parameters()):
            p.requires_grad_(True)
        self.relacion = relacion.to(device)
        self.lam = float(lam)

    def parametros(self, lr):
        return [{'params': list(self.bottleneck.parameters()) + list(self.head.parameters()), 'lr': lr}]

    def __call__(self, model, x_l, labels_l):
        self.bottleneck.train()
        self.head.train()
        f = model.pool_layer(model.backbone(x_l))
        z = self.head(self.bottleneck(f))
        q = self.relacion[labels_l]
        return self.lam * (-(q * F.log_softmax(z, dim=1)).sum(dim=1).mean())

    def state_dict(self):
        return {'bottleneck': self.bottleneck.state_dict(), 'head': self.head.state_dict(),
                'relacion': self.relacion.cpu()}

    def load_state_dict(self, sd):
        self.bottleneck.load_state_dict(sd['bottleneck'])
        self.head.load_state_dict(sd['head'])
        if not torch.allclose(sd['relacion'].to(self.relacion.device), self.relacion):
            raise ValueError('Co-Tuning: the checkpoint relationship does not match this run')


def construir_ct(args, classifier, labeled_train_dataset, device):
    """The CT term (L2SP or CoTuning), or None without --reg; writes ct.json."""
    reg = ct_reg(args)
    if reg is None:
        return None
    nota = {'reg': reg}
    if reg == 'l2sp':
        w0 = dict(utils.get_model(args.arch, pretrained_checkpoint=args.pretrained_backbone).named_parameters())
        termino = L2SP(classifier, w0, ct_param(args, 'l2sp_alpha'), ct_param(args, 'l2sp_beta'))
        nota.update(alpha=termino.alpha, beta=termino.beta, weight_decay=0.0, w0=args.pretrained_backbone)
    else:
        ck = torch.load(args.cotuning_source, map_location='cpu')
        sd = ck['state_dict'] if isinstance(ck, dict) and 'state_dict' in ck else ck
        n_src = sd['head.weight'].shape[0]
        fuente = utils.ImageClassifier(utils.get_model(args.arch, pretrained_checkpoint=args.pretrained_backbone),
                                       n_src, bottleneck_dim=args.bottleneck_dim,
                                       pool_layer=nn.Identity() if args.no_pool else None,
                                       finetune=args.finetune).to(device)
        fuente.load_state_dict(sd)
        loader = DataLoader(labeled_train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        relacion = relacion_cotuning(fuente, loader, labeled_train_dataset.num_classes, device)
        termino = CoTuning(fuente, relacion, ct_param(args, 'cotuning_lambda'), device)
        del fuente
        nota.update(lam=termino.lam, fuente=args.cotuning_source, fuente_sha256=_sha256(args.cotuning_source),
                    relacion=[[round(float(v), 6) for v in fila] for fila in relacion])
    with open(os.path.join(args.log, 'ct.json'), 'w') as f:
        json.dump(nota, f, indent=1)
    print('CT ' + json.dumps({k: v for k, v in nota.items() if k != 'relacion'}))
    return termino


def ct_weight_decay(args):
    return 0.0 if ct_reg(args) == 'l2sp' else args.wd


def ct_parametros(termino, lr):
    return termino.parametros(lr) if isinstance(termino, CoTuning) else []


def kw_ct(termino):
    return {} if termino is None else {'extra_loss': termino}


def ct_checkpoint(termino):
    return {'ct_state': termino.state_dict()} if isinstance(termino, CoTuning) else {}


def ct_resume(termino, checkpoint):
    if isinstance(termino, CoTuning):
        if 'ct_state' not in checkpoint:
            raise ValueError('--resume with --reg cotuning: the checkpoint has no ct_state')
        termino.load_state_dict(checkpoint['ct_state'])


def main(args: argparse.Namespace):
    import time
    _train_start = time.time()

    logger = CompleteLogger(args.log, args.phase)
    print(args)
    validate_init_args(args)
    validate_lpft_args(args)
    utils.validar_soporte(args)
    utils.validar_lista(args)
    validate_ct_args(args)

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
    train_transform = MultipleApply([weak_augment, strong_augment])
    val_transform = utils.get_val_transform(args.val_resizing, norm_mean=args.norm_mean, norm_std=args.norm_std)
    import os as _os2
    aug_log = open(_os2.path.join(args.log, 'architecture.txt'), 'a')
    aug_log.write(f"\ntrain_transform: {train_transform}\nval_transform: {val_transform}")
    aug_log.close()
    print(f"Transforms saved to {args.log}/architecture.txt")
    labeled_train_dataset, unlabeled_train_dataset, val_dataset = \
        utils.get_dataset(args.data,
                          args.num_samples_per_class,
                          args.root, train_transform,
                          val_transform,
                          seed=args.seed, subset_seed=args.subset_seed,
                          val_per_class=args.val_per_class,
                          **utils.kw_muestreo(args),
                          **utils.kw_labeled_list(args))
    if not args.oracle:
        utils.guardar_ids_etiquetados(labeled_train_dataset, args.root, args.log)
    utils.guardar_ids_validacion(val_dataset, args.root, args.log)
    firma_lista = utils.guardar_lista_etiquetadas(labeled_train_dataset, args.log)
    log_pi_eval, pesos_ce, firma_soporte = utils.soporte_etiquetado(args, labeled_train_dataset, args.log, device)
    if args.oracle:
        num_classes = labeled_train_dataset.num_classes
        labeled_train_dataset = ConcatDataset([labeled_train_dataset, unlabeled_train_dataset])
        labeled_train_dataset.num_classes = num_classes

    print("labeled_dataset_size: ", len(labeled_train_dataset))
    print("val_dataset_size: ", len(val_dataset))

    labeled_train_loader = DataLoader(labeled_train_dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.workers, drop_last=True, persistent_workers=(args.workers > 0), pin_memory=True)
    labeled_train_iter = ForeverDataIterator(labeled_train_loader)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    print("=> using pre-trained model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch, pretrained_checkpoint=args.pretrained_backbone)
    num_classes = labeled_train_dataset.num_classes
    pool_layer = nn.Identity() if args.no_pool else None
    classifier = utils.ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim, pool_layer=pool_layer,
                                       finetune=args.finetune).to(device)
    utils.cargar_alumno(classifier, args)
    import os as _os
    arch_log = open(_os.path.join(args.log, 'architecture.txt'), 'w')
    arch_log.write(str(classifier))
    arch_log.close()
    print(f"Architecture saved to {args.log}/architecture.txt")
    if getattr(args, 'init_classifier', None):
        init_from_source(classifier, args)
    reg_ct = construir_ct(args, classifier, labeled_train_dataset, device)

    if args.lr_scheduler == 'exp':
        optimizer = SGD(classifier.get_parameters() + ct_parametros(reg_ct, 1.0), args.lr, momentum=0.9,
                        weight_decay=ct_weight_decay(args), nesterov=True)
        lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    else:
        optimizer = SGD(classifier.get_parameters(base_lr=args.lr) + ct_parametros(reg_ct, args.lr), args.lr,
                        momentum=0.9, weight_decay=ct_weight_decay(args), nesterov=True)
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
        utils.lista_resume(firma_lista, checkpoint)
        ct_resume(reg_ct, checkpoint)

    if getattr(args, 'eval_epoch0', False) and not args.resume:
        # On resume, epoch 0 is already in best_avg.
        best_acc1, best_avg = eval_epoch0(val_loader, classifier, optimizer, lr_scheduler, logger, args,
                                          num_classes, log_prior=log_pi_eval, firma_soporte=firma_soporte,
                                          firma_lista=firma_lista)
    lp_phase = LinearProbePhase(classifier, args.lp_epochs) if getattr(args, 'lp_epochs', 0) else None

    historia = utils.EpochLogits(args, val_loader, val_transform, device)
    for epoch in range(start_epoch, args.epochs):
        if lp_phase is not None:
            lp_phase.set_epoch(epoch)
        print(lr_scheduler.get_lr())
        _timer.start("1. Train epoch")

        utils.empirical_risk_minimization(labeled_train_iter, classifier, optimizer, lr_scheduler, epoch, args, device,
                                          historia=historia, **utils.kw_ce(pesos_ce), **kw_ct(reg_ct))

        _timer.stop()
        _timer.start("2. Validation")
        acc1, avg = utils.validate(val_loader, classifier, args, device, num_classes, **utils.kw_eval(log_pi_eval))
        historia(classifier, epoch)

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
            **utils.lista_checkpoint(firma_lista),
            **ct_checkpoint(reg_ct),
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
    if lp_phase is not None:
        lp_phase.release()


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
    all_preds_crudo = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = classifier(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            if log_pi_eval is not None:  # report, matrix and AUC use z - log pi_l (the run's decision rule)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Baseline for Semi Supervised Learning')
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
    parser.add_argument('--oracle', action='store_true', default=False,
                        help='use all data as labeled data (oracle)')
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
    parser.add_argument('--trade-off-cls-strong', default=0.1, type=float,
                        help='the trade-off hyper-parameter of cls loss on strong augmented labeled data')
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
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run (default: 20)')
    parser.add_argument('-i', '--iters-per-epoch', default=500, type=int,
                        help='number of iterations per epoch (default: 500)')
    parser.add_argument('-p', '--print-freq', default=100, type=int, metavar='N',
                        help='print frequency (default: 100)')
    parser.add_argument('--subset-seed', default=None, type=int,
                        help='seed used only to pick the labeled images (x_u_split); '
                             'defaults to --seed')
    parser.add_argument('--val-per-class', default=None, type=int,
                        help='N validation images per class (own RNG, '
                             'RandomState(10000 + seed)); defaults to the full validation set')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training ')
    parser.add_argument("--log", default='baseline', type=str,
                        help="where to save logs, checkpoints and debugging images")
    parser.add_argument("--phase", default='train', type=str, choices=['train', 'test'],
                        help="when phase is 'test', only test the model")
    parser.add_argument('--patience', default=15, type=int,
                        help='early stopping patience (default: 15)')
    parser.add_argument('--resume', default=None, type=str,
                        help='path to a checkpoint to resume training from')
    utils.add_epoch_logits_args(parser)
    add_init_args(parser)
    add_lpft_args(parser)
    utils.add_label_sampling_args(parser, balanced_ce=True)
    utils.add_labeled_list_args(parser)
    add_ct_args(parser)
    utils.add_init_student_args(parser)
    args = parser.parse_args()
    main(args)
