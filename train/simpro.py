"""
SimPro (Du, Han and Huang, ICML 2024; arXiv 2402.13505) on this repository's FixMatch recipe.
Official code: github.com/LeapLabTHU/SimPro @ aa2425dc, SimPro/train.py; hyperparameters from sh/SimPro.sh.
The E-step, logit-adjusted losses and M-step follow the official code.

Deviations from the official code:
  * SGD nesterov, lr 0.003 exp decay, --finetune, batch 16, 500 iters/epoch, early stopping on val BA; no EMA model.
  * mu = 1 (the paper uses mu = M/N); alpha = 1 by default, --alpha auto = official 1 / scale_ratio.
  * tau = 2, threshold 0.95: the paper's values for 10 classes.
  * One forward pass per view (BatchNorm sees batches of 16).
"""
import random
import time
import warnings
import argparse
import shutil
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

import utils
import pool_modes
import prior_pool
from tllib.vision.transforms import MultipleApply
from tllib.utils.metric import accuracy
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.utils.data import ForeverDataIterator
from tllib.utils.logger import CompleteLogger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

JSON_TAG = 'SIMPRO_JSON'


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


def compute_adjustment_by_py(py, tro):
    """log(py ** tro + 1e-12), as in the official code."""
    return torch.log(py ** tro + 1e-12)


def parse_alpha(value):
    """--alpha: 'auto' (the official code's rule, 1 / scale_ratio) or a number > 0."""
    if str(value).strip().lower() == 'auto':
        return 'auto'
    a = float(value)
    if not a > 0:
        raise argparse.ArgumentTypeError('--alpha must be > 0 or auto: {}'.format(value))
    return a


class SimProState:
    """pi_u, phi and per-epoch accumulators; float32 and the official order of operations, to match it bit for bit."""

    def __init__(self, num_classes, pi_lab, tau=2.0, threshold=0.95, ema_u=0.9, alpha=1.0, scale_ratio=None,
                 device='cpu'):
        self.num_classes = int(num_classes)
        self.tau = float(tau)
        self.threshold = float(threshold)
        self.ema_u = float(ema_u)
        if not 0.0 <= self.ema_u < 1.0:
            raise ValueError('--ema-u must be in [0, 1): {}'.format(ema_u))
        self.alpha = alpha
        if alpha == 'auto':
            if scale_ratio is None or not scale_ratio > 0:
                raise ValueError('--alpha auto needs scale_ratio = u_len / l_len > 0')
            self.scale_ratio = float(scale_ratio)
            self.alpha_eff = 1.0 / self.scale_ratio
        else:
            self.scale_ratio = None
            self.alpha_eff = float(alpha)
        self.device = torch.device(device)
        if isinstance(pi_lab, torch.Tensor):
            pi_lab = pi_lab.detach().float().cpu()
        else:
            pi_lab = torch.as_tensor(np.asarray(pi_lab, dtype=np.float32))
        if pi_lab.shape != (self.num_classes,) or not torch.all(pi_lab > 0):
            raise ValueError('pi_lab must have {} probabilities > 0: {}'.format(self.num_classes, pi_lab.tolist()))
        self.pi_lab = pi_lab.to(self.device)                                           # args.py_con = args.py_l
        self.pi_u = (torch.ones(self.num_classes) / self.num_classes).to(self.device)  # args.estimated_dis = py_uni
        self.phi = self.pi_lab                                                         # args.adapt_dis = args.py_con
        self.n_mstep = 0
        self.start_epoch()

    # adjustments stay fixed during the epoch, as in the official code
    def start_epoch(self):
        C = self.num_classes
        self.adjustment_u = compute_adjustment_by_py(self.pi_u, self.tau)   # for the pseudo-labels (E-step)
        self.adjustment = compute_adjustment_by_py(self.phi, self.tau)      # for Lx and Lu (logit adjustment)
        self.dis_unlabeled = torch.zeros(C).to(self.device)
        self.count_labeled = torch.zeros(C).to(self.device)
        self.pi_u_used = self.pi_u.clone()
        self.phi_used = self.phi.clone()

    @torch.no_grad()
    def pseudo_label(self, y_u_weak):
        """Soft q and mask (max q >= threshold, as in the official code)."""
        pseudo_label = torch.softmax(y_u_weak.detach() + self.adjustment_u.to(y_u_weak.dtype), dim=-1)
        max_probs, _ = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(self.threshold)
        return pseudo_label, mask

    @torch.no_grad()
    def accumulate(self, labels_l, pseudo_label, mask):
        """M-step accumulators: labeled batch counts and soft mass of the accepted samples."""
        self.count_labeled += torch.bincount(labels_l, minlength=self.num_classes)
        self.dis_unlabeled += torch.sum(pseudo_label[mask], dim=0)

    def labeled_loss(self, y_l_views, labels_l):
        """Lx without alpha, averaged over the weak and strong labeled views."""
        logits_l = torch.cat(list(y_l_views), dim=0)
        targets_l = torch.cat([labels_l] * len(y_l_views), dim=0)
        return F.cross_entropy(logits_l + self.adjustment.to(logits_l.dtype), targets_l, reduction="mean")

    def weighted_labeled_loss(self, lx):
        """alpha * Lx; with auto, Lx / scale_ratio as in the official code."""
        if self.alpha == 'auto':
            return lx / self.scale_ratio
        return lx if self.alpha_eff == 1.0 else lx * self.alpha_eff

    def unlabeled_loss(self, y_u_strong_views, pseudo_label, mask):
        """Lu with the soft q over the strong views."""
        n = len(y_u_strong_views)
        mask = mask.float()
        mask_twice = torch.cat([mask] * n, dim=0)
        logits_u_s_twice = torch.cat(list(y_u_strong_views), dim=0)
        targets_u_twice = torch.cat([pseudo_label] * n, dim=0)
        return (F.cross_entropy(logits_u_s_twice + self.adjustment.to(logits_u_s_twice.dtype), targets_u_twice,
                                reduction="none") * mask_twice).mean()

    @torch.no_grad()
    def m_step(self):
        """Sets the new pi_u and phi; returns the raw estimate of the epoch."""
        estimated_dis = self.dis_unlabeled / (self.dis_unlabeled.sum() + 1)
        self.pi_u = self.pi_u * self.ema_u + (estimated_dis) * (1 - self.ema_u)
        if self.alpha_eff == 1.0:
            count_forward = self.count_labeled + self.dis_unlabeled                # the official code, as is
        else:
            count_forward = self.alpha_eff * self.count_labeled + self.dis_unlabeled  # Eq. 9 of the paper
        self.phi = self.phi * self.ema_u + (count_forward / count_forward.sum()) * (1 - self.ema_u)
        self.n_mstep += 1
        return estimated_dis

    def hyper(self):
        return {'tau': self.tau, 'threshold': self.threshold, 'ema_u': self.ema_u,
                'alpha': self.alpha if self.alpha == 'auto' else float(self.alpha),
                'alpha_eff': self.alpha_eff, 'num_classes': self.num_classes}

    def state_dict(self):
        return dict(self.hyper(), pi_u=self.pi_u.cpu().tolist(), phi=self.phi.cpu().tolist(),
                    pi_lab=self.pi_lab.cpu().tolist(), n_mstep=self.n_mstep)

    def load_state_dict(self, state):
        mine = self.hyper()
        for key in ('tau', 'threshold', 'ema_u', 'alpha', 'num_classes'):
            if state.get(key) != mine[key]:
                raise ValueError('the checkpoint has {} = {!r} and this run uses {!r}; relaunch with the same '
                                 'flags'.format(key, state.get(key), mine[key]))
        if not np.allclose(state['pi_lab'], self.pi_lab.cpu().numpy(), rtol=0, atol=1e-7):
            raise ValueError('checkpoint pi_lab differs from this run')
        self.pi_u = torch.tensor(state['pi_u'], dtype=torch.float32).to(self.device)
        self.phi = torch.tensor(state['phi'], dtype=torch.float32).to(self.device)
        self.n_mstep = int(state['n_mstep'])
        self.start_epoch()


def symmetric_kl(estimated, true, eps=1e-12):
    """The official 'KL_estim' (logging only)."""
    kl = nn.KLDivLoss(reduction="sum")
    return float(0.5 * kl((estimated + eps).log(), true) + 0.5 * kl(true.log(), estimated))


class _PriorLog:
    """Adapter so PseudoLabelClassStats.report prints the pi_u used as pi_pool."""

    def __init__(self, state):
        self.mode = 'simpro'
        self.pi_pool = state.pi_u_used.double().cpu().numpy()
        self.pi_lab = state.pi_lab.double().cpu().numpy()
        self.n_ema_updates = state.n_mstep


def _fmt(v):
    return '[' + ', '.join('{:.4f}'.format(float(x)) for x in v) + ']'


def epoch_report(epoch, state, estimated, pi_real, names):
    """Readable line + SIMPRO_JSON line with the M-step state of the epoch (pi_real is only measured)."""
    est = estimated.double().cpu()
    real = torch.as_tensor(pi_real, dtype=torch.float64)
    payload = {
        'epoca': int(epoch), 'clases': list(names), 'alpha': state.hyper()['alpha'], 'alpha_eff': state.alpha_eff,
        'tau': state.tau, 'umbral': state.threshold, 'ema_u': state.ema_u,
        'pi_u_usado': [round(float(x), 6) for x in state.pi_u_used], 'phi_usado': [round(float(x), 6) for x in
                                                                                  state.phi_used],
        'pi_u_epoca': [round(float(x), 6) for x in est], 'masa_aceptada': round(float(state.dis_unlabeled.sum()), 3),
        'n_lab_vistas': int(state.count_labeled.sum()),
        'pi_u_nuevo': [round(float(x), 6) for x in state.pi_u], 'phi_nuevo': [round(float(x), 6) for x in state.phi],
        'pi_pool_real': [round(float(x), 6) for x in real],
        'tv_pi_u_real_pp': round(100 * prior_pool.tv(state.pi_u.double().cpu().numpy(), real.numpy()), 3),
        'tv_pi_u_epoca_real_pp': round(100 * prior_pool.tv(est.numpy() / max(float(est.sum()), 1e-12),
                                                           real.numpy()), 3),
        'kl_estim': round(symmetric_kl(est.float(), real.float()), 6),
        'm_steps': state.n_mstep,
    }
    lines = ['[SimPro] epoch {} | accepted mass {:.1f} | pi_u used {} -> new {} | phi new {} | '
             'TV(pi_u, true) = {:.1f} pp | KL_estim {:.4f}'.format(
                 epoch, payload['masa_aceptada'], _fmt(state.pi_u_used), _fmt(state.pi_u), _fmt(state.phi),
                 payload['tv_pi_u_real_pp'], payload['kl_estim']),
             JSON_TAG + ' ' + json.dumps(payload)]
    return '\n'.join(lines)


def main(args: argparse.Namespace):
    import time
    _train_start = time.time()

    logger = CompleteLogger(args.log, args.phase)
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)  # timm's RandAugment uses the global np.random
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    # pool: weak + TWO strong views, as in the official code
    weak_augment = utils.get_train_transform(args.train_resizing, random_horizontal_flip=True,
                                             norm_mean=args.norm_mean, norm_std=args.norm_std)
    strong_augment = utils.get_train_transform(args.train_resizing, random_horizontal_flip=True,
                                               auto_augment=args.auto_augment,
                                               norm_mean=args.norm_mean, norm_std=args.norm_std)
    labeled_train_transform = MultipleApply([weak_augment, strong_augment])
    unlabeled_train_transform = MultipleApply([weak_augment, strong_augment, strong_augment])
    val_transform = utils.get_val_transform(args.val_resizing, norm_mean=args.norm_mean, norm_std=args.norm_std)
    print('labeled_train_transform: ', labeled_train_transform)
    print('unlabeled_train_transform: ', unlabeled_train_transform)
    print('val_transform:', val_transform)
    labeled_train_dataset, unlabeled_train_dataset, val_dataset = \
        utils.get_dataset(args.data,
                          args.num_samples_per_class,
                          args.root, labeled_train_transform,
                          val_transform,
                          unlabeled_train_transform=unlabeled_train_transform,
                          seed=args.seed, pool_mode=args.pool_mode,
                          subset_seed=args.subset_seed,
                          val_per_class=args.val_per_class)
    utils.guardar_ids_etiquetados(labeled_train_dataset, args.root, args.log)
    utils.guardar_ids_validacion(val_dataset, args.root, args.log)
    print("labeled_dataset_size: ", len(labeled_train_dataset))
    print('unlabeled_dataset_size: ', len(unlabeled_train_dataset))
    print("val_dataset_size: ", len(val_dataset))
    num_classes = labeled_train_dataset.num_classes
    class_names = prior_pool.class_names(labeled_train_dataset, num_classes)
    lab_counts = prior_pool.class_counts(labeled_train_dataset, num_classes)
    pool_counts = prior_pool.class_counts(unlabeled_train_dataset, num_classes)
    if np.any(lab_counts == 0):
        raise ValueError('some classes have no labeled examples: {}'.format(lab_counts.tolist()))
    pi_pool_real = pool_counts / max(pool_counts.sum(), 1)

    labeled_train_loader = DataLoader(labeled_train_dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.workers, drop_last=True, persistent_workers=(args.workers > 0), pin_memory=True)
    unlabeled_train_loader = DataLoader(unlabeled_train_dataset, batch_size=args.batch_size * args.mu, shuffle=True,
                                        num_workers=args.workers, drop_last=True, persistent_workers=(args.workers > 0), pin_memory=True)
    labeled_train_iter = ForeverDataIterator(labeled_train_loader)
    unlabeled_train_iter = ForeverDataIterator(unlabeled_train_loader)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    l_len, u_len = len(labeled_train_loader), len(unlabeled_train_loader)
    cls_num_list_l = torch.Tensor(lab_counts.tolist()).float()  # as in the official code: py_l = counts / sum, float32
    simpro = SimProState(num_classes, cls_num_list_l / cls_num_list_l.sum(), tau=args.tau, threshold=args.threshold,
                         ema_u=args.ema_u, alpha=args.alpha, scale_ratio=u_len / max(l_len, 1), device=device)
    print('[SimPro] tau {} | threshold {} | ema_u {} | alpha {} (effective {:.6f}; l_len {}, u_len {}, mu {}) | '
          'pi_lab {} | true pi_pool {} (measured only)'.format(
              args.tau, args.threshold, args.ema_u, args.alpha, simpro.alpha_eff, l_len, u_len, args.mu,
              _fmt(lab_counts / lab_counts.sum()), _fmt(pi_pool_real)))

    print("=> using pre-trained model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch, pretrained_checkpoint=args.pretrained_backbone)
    pool_layer = nn.Identity() if args.no_pool else None
    classifier = utils.ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim, pool_layer=pool_layer,
                                       finetune=args.finetune).to(device)
    import os as _os
    arch_log = open(_os.path.join(args.log, 'architecture.txt'), 'w')
    arch_log.write(str(classifier))
    arch_log.close()
    print(f"Architecture saved to {args.log}/architecture.txt")

    if args.lr_scheduler == 'exp':
        optimizer = SGD(classifier.get_parameters(), args.lr, momentum=0.9, weight_decay=args.wd, nesterov=True)
        lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    else:
        optimizer = SGD(classifier.get_parameters(base_lr=args.lr), args.lr, momentum=0.9, weight_decay=args.wd,
                        nesterov=True)
        lr_scheduler = utils.get_cosine_scheduler_with_warmup(optimizer, args.epochs * args.iters_per_epoch)

    # evaluate the best checkpoint only (in fixmatch.py this branch is broken: it loads the whole dict)
    if args.phase == 'test':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
        acc1, avg = utils.validate(val_loader, classifier, args, device, num_classes)
        print(acc1, avg)
        return

    best_acc1 = 0.0
    best_avg = 0.0
    es_counter = 0

    start_epoch = 0
    if args.resume:
        print(f"Resume since {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'simpro' not in checkpoint:
            raise ValueError('the checkpoint is not from simpro.py (it has no "simpro" entry)')
        classifier.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint.get('best_acc1', 0.0)
        best_avg = checkpoint.get('best_avg', 0.0)
        print(f"Starting on epoch {start_epoch}, best_acc1={best_acc1:.1f}")
        es_counter = checkpoint.get('es_counter', 0)
        simpro.load_state_dict(checkpoint['simpro'])
        print('[SimPro] resumed from checkpoint: pi_u {} | phi {} | {} M-steps'.format(
            _fmt(simpro.pi_u), _fmt(simpro.phi), simpro.n_mstep))

    for epoch in range(start_epoch, args.epochs):
        print(lr_scheduler.get_lr())
        _timer.start("1. Train epoch")

        train(labeled_train_iter, unlabeled_train_iter, classifier, optimizer, lr_scheduler, epoch, args,
              simpro, class_names, pi_pool_real)

        _timer.stop()
        _timer.start("2. Validation")
        acc1, avg = utils.validate(val_loader, classifier, args, device, num_classes)

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
            'simpro': simpro.state_dict(),
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
    # same output as fixmatch.py: export_logits.py parses it
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
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = classifier(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
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
    _timer.stop()
    _timer.report()
    _elapsed = time.time() - _train_start
    print(f"Total training time: {_elapsed/60:.1f} min ({_elapsed:.1f}s)")
    print("best_acc1 = {:3.1f}".format(best_acc1))
    print('best_avg = {:3.1f}'.format(best_avg))
    logger.close()


def train(labeled_train_iter: ForeverDataIterator, unlabeled_train_iter: ForeverDataIterator, model, optimizer: SGD,
          lr_scheduler: LambdaLR, epoch: int, args: argparse.Namespace, simpro: SimProState, class_names=None,
          pi_pool_real=None):
    batch_time = AverageMeter('Time', ':2.2f')
    data_time = AverageMeter('Data', ':2.1f')
    cls_losses = AverageMeter('Cls Loss', ':3.2f')
    self_training_losses = AverageMeter('Self Training Loss', ':3.2f')
    losses = AverageMeter('Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    pseudo_label_ratios = AverageMeter('Pseudo Label Ratio', ':3.1f')
    pseudo_label_accs = AverageMeter('Pseudo Label Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cls_losses, self_training_losses, cls_accs, pseudo_label_accs,
         pseudo_label_ratios],
        prefix="Epoch: [{}]".format(epoch))

    simpro.start_epoch()
    pl_stats = None
    model.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        (x_l, x_l_strong), labels_l = next(labeled_train_iter)
        x_l = x_l.to(device, non_blocking=True)
        x_l_strong = x_l_strong.to(device, non_blocking=True)
        labels_l = labels_l.to(device, non_blocking=True)

        (x_u, x_u_strong, x_u_strong1), labels_u = next(unlabeled_train_iter)
        x_u = x_u.to(device, non_blocking=True)
        x_u_strong = x_u_strong.to(device, non_blocking=True)
        x_u_strong1 = x_u_strong1.to(device, non_blocking=True)
        labels_u = labels_u.to(device, non_blocking=True)

        data_time.update(time.time() - end)

        optimizer.zero_grad()

        y_l = model(x_l)
        y_l_strong = model(x_l_strong)
        cls_loss = simpro.labeled_loss([y_l, y_l_strong], labels_l)
        simpro.weighted_labeled_loss(cls_loss).backward()

        with torch.no_grad():
            y_u = model(x_u)
        pseudo_label, mask = simpro.pseudo_label(y_u)
        simpro.accumulate(labels_l, pseudo_label, mask)
        y_u_strong = model(x_u_strong)
        y_u_strong1 = model(x_u_strong1)
        self_training_loss = simpro.unlabeled_loss([y_u_strong, y_u_strong1], pseudo_label, mask)
        self_training_loss.backward()

        hard = pseudo_label.argmax(dim=1)
        if pl_stats is None:
            pl_stats = prior_pool.PseudoLabelClassStats(y_u.size(1), y_u.device)
        pl_stats.update(hard, mask.float(), labels_u)

        loss = simpro.weighted_labeled_loss(cls_loss.detach()) + self_training_loss.detach()
        losses.update(loss.item(), x_l.size(0))
        cls_losses.update(cls_loss.item(), x_l.size(0))
        self_training_losses.update(self_training_loss.item(), x_l.size(0))

        cls_acc = accuracy(y_l, labels_l)[0]
        cls_accs.update(cls_acc.item(), x_l.size(0))

        n_pseudo_labels = mask.sum()
        ratio = n_pseudo_labels.float() / x_u.size(0)
        pseudo_label_ratios.update(ratio.item() * 100, x_u.size(0))
        if n_pseudo_labels > 0:
            n_correct = ((hard == labels_u) & mask).float().sum()
            pseudo_label_acc = n_correct / n_pseudo_labels * 100
            pseudo_label_accs.update(pseudo_label_acc.item(), n_pseudo_labels.item())

        optimizer.step()
        lr_scheduler.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    if pl_stats is not None:
        print(pl_stats.report(epoch, class_names, _PriorLog(simpro)))
    estimated = simpro.m_step()
    if pi_pool_real is not None:
        print(epoch_report(epoch, simpro, estimated, pi_pool_real, class_names))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SimPro (ICML 2024) for Semi Supervised Learning')
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA',
                        help='dataset: ' + ' | '.join(utils.get_dataset_names()))
    parser.add_argument('--num-samples-per-class', default=4, type=int,
                        help='number of labeled samples per class')
    parser.add_argument('--pool-mode', default='orig', choices=pool_modes.POOL_MODES,
                        help='sibling-image ablation for BCN20000 (see pool_modes.py)')
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
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='threshold on the adjusted posterior softmax(z + tau log pi_u) (paper: 0.95 on CIFAR10/STL10)')
    parser.add_argument('--tau', default=2.0, type=float,
                        help='strength of the log pi_u and log phi adjustment (paper: 2 on CIFAR10/STL10, 1 on '
                             'CIFAR100/ImageNet-127)')
    parser.add_argument('--ema-u', default=0.9, type=float,
                        help='per-epoch EMA momentum of pi_u and phi (sh/SimPro.sh: 0.9)')
    parser.add_argument('--alpha', default=1.0, type=parse_alpha,
                        help='weight of Lx in L = alpha Lx + Lu: a number (default 1, the paper value with mu = M/N) or '
                             'auto = l_len / u_len, the official code rule (mu N / M)')
    parser.add_argument('--mu', default=1, type=int,
                        help='pool batch = mu x labeled batch (this protocol: 1; paper: M/N)')
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
    parser.add_argument('--epochs', default=60, type=int, metavar='N',
                        help='number of total epochs to run (default: 60)')
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
    parser.add_argument("--log", default='simpro', type=str,
                        help="where to save logs, checkpoints and debugging images")
    parser.add_argument("--phase", default='train', type=str, choices=['train', 'test'],
                        help="when phase is 'test', only test the model")
    parser.add_argument('--patience', default=15, type=int,
                        help='early stopping patience (default: 15)')
    parser.add_argument('--resume', default=None, type=str,
                        help='path to a checkpoint to resume training from')
    args = parser.parse_args()
    if args.mu < 1:
        parser.error('--mu must be >= 1')
    main(args)
