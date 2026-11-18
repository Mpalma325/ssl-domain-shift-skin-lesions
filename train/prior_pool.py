"""
Pool prior for FixMatch and FlexMatch (off by default): tests whether pseudo-labels fail by base rate.

The labeled set is balanced but the pool is not, so the model over-predicts rare classes on the pool. With
log_ratio = log pi_pool - log pi_lab, selection and pseudo-labels use softmax(z_weak + log_ratio) and the unlabeled
loss is CE(z_strong + log_ratio, pseudo) (logit adjustment, Menon et al., ICLR 2021); z itself stays balanced.
FlexMatch divides the learning status by pi_pool / pi_lab so it becomes a per-class rate.
True pool labels are used only for per-class logging. Only torch and numpy, so tests can import it.
"""
import argparse
import json
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, Subset

MODES = ('none', 'file', 'oracle')
# Flags use default=SUPPRESS so print(args) matches the original script; read them with opt().
DEFAULTS = {
    'pool_prior': 'none',
    'pool_prior_file': None,
    'pool_prior_ema': False,
    'pool_prior_ema_momentum': 0.999,
    'pool_prior_ema_start_epoch': 1,
    'log_pseudo_per_class': False,
    'pool_prior_status': 'tasa',  # FlexMatch only
}
STATUS_MODES = ('tasa', 'conteo')
PRIOR_FLOOR = 1e-4  # floor of the EM re-estimated prior (avoids log 0)
JSON_TAG = 'PL_POR_CLASE_JSON'


def add_arguments(parser, flexmatch=False):
    g = parser.add_argument_group('pseudo-labeling corrected by the pool prior (default: off)')
    g.add_argument('--pool-prior', choices=MODES, default=argparse.SUPPRESS,
                   help='none (default): original method, unchanged. file: pool prior read from --pool-prior-file. '
                        'oracle: true pool prior, from its true labels (control arm only). '
                        'With file or oracle, threshold and pseudo-label use softmax(z_w + log pi_pool - log pi_lab) '
                        'and the unlabeled CE is logit-adjusted')
    g.add_argument('--pool-prior-file', type=str, default=argparse.SUPPRESS, metavar='JSON',
                   help='list with one probability per class (in dataset class order), {class: p} '
                        'or {"prior": ..., "classes": [...], "seed": s, "num_samples_per_class": k, "data": d}; '
                        'any metadata present is checked against the run')
    g.add_argument('--pool-prior-ema', action='store_true', default=argparse.SUPPRESS,
                   help='re-estimate pi_pool online by EM from the adjusted pool predictions (experimental, '
                        'off by default); starts from the --pool-prior prior (from pi_lab with none)')
    g.add_argument('--pool-prior-ema-momentum', type=float, default=argparse.SUPPRESS, metavar='M',
                   help='per-iteration momentum of the online EM (default: 0.999)')
    g.add_argument('--pool-prior-ema-start-epoch', type=int, default=argparse.SUPPRESS, metavar='E',
                   help='first epoch in which the online EM updates (default: 1)')
    g.add_argument('--log-pseudo-per-class', action='store_true', default=argparse.SUPPRESS,
                   help='log per epoch the accepted fraction and per-class precision of the pseudo-labels, '
                        'using the true pool labels for measurement only (always on with a pool prior)')
    if flexmatch:
        g.add_argument('--pool-prior-status', choices=STATUS_MODES, default=argparse.SUPPRESS,
                       help='FlexMatch with a pool prior only: tasa (default) divides the learning status '
                            'sigma(c) by pi_pool(c)/pi_lab(c), so the curriculum equalizes per-class rates instead of '
                            'counts; conteo keeps the original count (to decompose the effect)')


def opt(args, name):
    """Value of a pool-prior flag, or its default if not passed."""
    return getattr(args, name, DEFAULTS[name])


def dataset_targets(ds):
    """True labels without loading images (through Subset, ConcatDataset and .dataset wrappers)."""
    if isinstance(ds, Subset):
        inner = dataset_targets(ds.dataset)
        return [inner[i] for i in ds.indices]
    if isinstance(ds, ConcatDataset):
        return [y for d in ds.datasets for y in dataset_targets(d)]
    if hasattr(ds, 'targets'):
        return [int(y) for y in ds.targets]
    if hasattr(ds, 'dataset'):
        return dataset_targets(ds.dataset)
    raise TypeError('cannot read the labels of a {}'.format(type(ds).__name__))


def class_names(ds, num_classes):
    """Class names in index order (ImageFolder.classes); '0', '1', ... if there are none."""
    base = ds
    while isinstance(base, (Subset, ConcatDataset)) or (not hasattr(base, 'classes') and hasattr(base, 'dataset')):
        base = base.datasets[0] if isinstance(base, ConcatDataset) else base.dataset
    names = getattr(base, 'classes', None)
    if names is None or len(names) != num_classes:
        names = [str(c) for c in range(num_classes)]
    return [str(n) for n in names]


def class_counts(ds, num_classes):
    y = np.asarray(dataset_targets(ds), dtype=np.int64)
    if y.size and (y.min() < 0 or y.max() >= num_classes):
        raise ValueError('labels outside [0, {})'.format(num_classes))
    return np.bincount(y, minlength=num_classes)


def read_prior_file(path, names, expected=None):
    """Pool prior from a JSON file; requires p > 0 and renormalizes."""
    with open(path, encoding='utf-8') as fh:
        obj = json.load(fh)
    if isinstance(obj, dict) and 'prior' in obj:
        meta, obj = obj, obj['prior']
        if 'classes' in meta and [str(c) for c in meta['classes']] != list(names):
            raise ValueError('{}: "classes" = {} does not match the dataset classes {}'.format(
                path, meta['classes'], list(names)))
        for key, value in (expected or {}).items():
            if key in meta and value is not None and meta[key] != value:
                raise ValueError('{}: the JSON has {} = {!r} but the run uses {!r}'.format(path, key, meta[key], value))
    if isinstance(obj, dict):
        if set(obj) != set(names):
            raise ValueError('{}: keys {} differ from the dataset classes {}'.format(path, sorted(obj), list(names)))
        obj = [obj[c] for c in names]
    p = np.asarray(obj, dtype=np.float64)
    if p.shape != (len(names),):
        raise ValueError('{}: expected {} probabilities, got {}'.format(path, len(names), p.size))
    if not np.all(np.isfinite(p)) or np.any(p <= 0):
        raise ValueError('{}: the prior must be finite and > 0 for every class (put a floor in the estimator): '
                         '{}'.format(path, p.tolist()))
    s = p.sum()
    if abs(s - 1.0) > 1e-3:
        print('WARNING pool-prior: {} sums to {:.6f}; renormalizing'.format(path, s))
    return p / s


def tv(a, b):
    """Total variation distance between two distributions."""
    return 0.5 * float(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)).sum())


def _fmt(v):
    return '[' + ', '.join('{:.4f}'.format(x) for x in v) + ']'


class PoolPrior:
    """Pool-prior state of a run. log_ratio() = log pi_pool - log pi_lab (float64)."""

    def __init__(self, mode, pi_pool, pi_lab, names, ema_momentum=None, ema_start_epoch=1):
        assert mode in MODES, mode
        self.mode = mode
        self.pi_pool = np.array(pi_pool, dtype=np.float64)
        self.pi_lab = np.array(pi_lab, dtype=np.float64)
        self.pi_init = self.pi_pool.copy()
        self.names = [str(n) for n in names]
        assert self.pi_pool.shape == self.pi_lab.shape == (len(self.names),)
        assert np.all(self.pi_pool > 0) and np.all(self.pi_lab > 0)
        if ema_momentum is not None and not 0.0 <= ema_momentum < 1.0:
            raise ValueError('--pool-prior-ema-momentum must be in [0, 1): {}'.format(ema_momentum))
        self.ema_momentum = ema_momentum
        self.ema_start_epoch = int(ema_start_epoch)
        self.n_ema_updates = 0

    def log_ratio(self):
        return np.log(self.pi_pool) - np.log(self.pi_lab)

    def ema_active(self, epoch):
        return self.ema_momentum is not None and epoch >= self.ema_start_epoch

    @torch.no_grad()
    def ema_update(self, y_target):
        """One online EM step (Saerens et al., 2002) with the weak-view logits."""
        log_ratio = torch.as_tensor(self.log_ratio(), dtype=y_target.dtype, device=y_target.device)
        q = F.softmax(y_target.detach() + log_ratio, dim=1).mean(dim=0).double().cpu().numpy()
        new = self.ema_momentum * self.pi_pool + (1.0 - self.ema_momentum) * q
        new = np.maximum(new, PRIOR_FLOOR)
        self.pi_pool = new / new.sum()
        self.n_ema_updates += 1

    def state_dict(self):
        return {'mode': self.mode, 'pi_pool': self.pi_pool.tolist(), 'pi_lab': self.pi_lab.tolist(),
                'pi_init': self.pi_init.tolist(), 'classes': list(self.names),
                'ema_momentum': self.ema_momentum, 'n_ema_updates': self.n_ema_updates}

    def load_state_dict(self, state):
        if state.get('mode') != self.mode:
            raise ValueError('the checkpoint is from --pool-prior {} but this run uses {}'.format(state.get('mode'), self.mode))
        if list(state.get('classes', self.names)) != self.names:
            raise ValueError('checkpoint classes {} differ from {}'.format(state.get('classes'), self.names))
        if not np.allclose(np.asarray(state['pi_lab'], dtype=np.float64), self.pi_lab, rtol=0, atol=1e-12):
            raise ValueError('checkpoint pi_lab differs from this run')
        saved = np.asarray(state['pi_pool'], dtype=np.float64)
        if self.ema_momentum is None:
            if not np.allclose(saved, self.pi_pool, rtol=0, atol=1e-12):
                raise ValueError('checkpoint prior {} differs from this run {}'.format(
                    _fmt(saved), _fmt(self.pi_pool)))
        else:
            self.pi_pool = saved
            self.n_ema_updates = int(state.get('n_ema_updates', 0))

    def describe(self, lab_counts=None, pool_counts=None):
        em = 'no' if self.ema_momentum is None else 'momentum {} from epoch {}'.format(
            self.ema_momentum, self.ema_start_epoch)
        lines = ['[pool-prior] mode {} | online EM: {}'.format(self.mode, em),
                 '  {:<8s} {:>6s} {:>8s} {:>7s} {:>8s} {:>9s} {:>10s}'.format(
                     'class', 'n_lab', 'pi_lab', 'n_pool', 'pi_real', 'pi_used', 'log_ratio')]
        pi_real = None if pool_counts is None else pool_counts / max(pool_counts.sum(), 1)
        for c, name in enumerate(self.names):
            lines.append('  {:<8s} {:>6s} {:8.4f} {:>7s} {:>8s} {:9.4f} {:+10.4f}'.format(
                name[:8], '-' if lab_counts is None else str(int(lab_counts[c])), self.pi_lab[c],
                '-' if pool_counts is None else str(int(pool_counts[c])),
                '-' if pi_real is None else '{:.4f}'.format(pi_real[c]), self.pi_pool[c], self.log_ratio()[c]))
        if pi_real is not None:
            lines.append('  TV(pi_used, pi_real) = {:.1f} pp | TV(pi_lab, pi_real) = {:.1f} pp '
                         '(pi_real comes from the true labels of the pool; measured only)'.format(
                             100 * tv(self.pi_pool, pi_real), 100 * tv(self.pi_lab, pi_real)))
        if self.mode == 'oracle':
            lines.append('  WARNING: oracle uses the true labels of the pool; it is a control arm, not a method.')
        return '\n'.join(lines)


def build(args, labeled_ds, unlabeled_ds):
    """PoolPrior of the run, or None when the pool prior is off."""
    mode = opt(args, 'pool_prior')
    path = opt(args, 'pool_prior_file')
    ema = opt(args, 'pool_prior_ema')
    if mode not in MODES:
        raise ValueError('--pool-prior {} is not valid'.format(mode))
    if path is not None and mode != 'file':
        raise ValueError('--pool-prior-file is only used with --pool-prior file')
    if mode == 'none' and not ema:
        if hasattr(args, 'pool_prior_status'):
            raise ValueError('--pool-prior-status is only used with a pool prior (--pool-prior file or oracle)')
        return None
    num_classes = getattr(labeled_ds, 'num_classes', None) or int(max(dataset_targets(labeled_ds))) + 1
    names = class_names(labeled_ds, num_classes)
    lab_counts = class_counts(labeled_ds, num_classes)
    pool_counts = class_counts(unlabeled_ds, num_classes)
    if np.any(lab_counts == 0):
        raise ValueError('some classes have no labeled examples: {}'.format(lab_counts.tolist()))
    pi_lab = lab_counts / lab_counts.sum()
    if mode == 'file':
        if path is None:
            raise ValueError('--pool-prior file requires --pool-prior-file')
        expected = {'seed': getattr(args, 'seed', None), 'data': getattr(args, 'data', None),
                    'num_samples_per_class': getattr(args, 'num_samples_per_class', None)}
        pi_pool = read_prior_file(path, names, expected)
    elif mode == 'oracle':
        pi_pool = pool_counts / pool_counts.sum()
        if np.any(pi_pool <= 0):
            print('WARNING pool-prior: some classes are absent from the pool; oracle gives them a floor of {}'.format(PRIOR_FLOOR))
            pi_pool = np.maximum(pi_pool, PRIOR_FLOOR)
            pi_pool = pi_pool / pi_pool.sum()
    else:  # none + online EM: starts without adjustment (pi_pool = pi_lab)
        pi_pool = pi_lab
    state = PoolPrior(mode, pi_pool, pi_lab, names,
                      ema_momentum=float(opt(args, 'pool_prior_ema_momentum')) if ema else None,
                      ema_start_epoch=opt(args, 'pool_prior_ema_start_epoch'))
    print(state.describe(lab_counts, pool_counts))
    return state


def checkpoint_entry(state):
    """Extra checkpoint entry; empty when the pool prior is off."""
    return {} if state is None else {'pool_prior': state.state_dict()}


def resume(state, checkpoint):
    """Restores the prior on resume; mixing runs with different priors is an error."""
    saved = checkpoint.get('pool_prior') if isinstance(checkpoint, dict) else None
    if state is None:
        if saved is not None:
            raise ValueError('the checkpoint is from a run with --pool-prior {}; relaunch with the same '
                             'flags'.format(saved.get('mode')))
        return
    if saved is None:
        raise ValueError('the checkpoint is from a run without pool prior but this one uses --pool-prior {}'.format(state.mode))
    state.load_state_dict(saved)
    print('[pool-prior] resumed from checkpoint: pi_pool = {} ({} EM steps)'.format(
        _fmt(state.pi_pool), state.n_ema_updates))


class PriorAdjustedSelfTrainingLoss(nn.Module):
    """ConfidenceBasedSelfTrainingLoss with log_ratio added; log_ratio = 0 reproduces it bit for bit."""

    def __init__(self, threshold, log_ratio):
        super(PriorAdjustedSelfTrainingLoss, self).__init__()
        self.threshold = threshold
        self.register_buffer('log_ratio', torch.as_tensor(np.asarray(log_ratio, dtype=np.float32)))

    def set_log_ratio(self, log_ratio):
        self.log_ratio.copy_(torch.as_tensor(np.asarray(log_ratio, dtype=np.float32)))

    def forward(self, y, y_target):
        log_ratio = self.log_ratio.to(dtype=y.dtype)
        confidence, pseudo_labels = F.softmax(y_target.detach() + log_ratio, dim=1).max(dim=1)
        mask = (confidence > self.threshold).float()
        self_training_loss = (F.cross_entropy(y + log_ratio, pseudo_labels, reduction='none') * mask).mean()
        return self_training_loss, mask, pseudo_labels


class PseudoLabelClassStats:
    """Per-class pseudo-label counts for one epoch (measurement only, uses the pool's true labels)."""

    def __init__(self, num_classes, device=None):
        self.num_classes = int(num_classes)
        C = self.num_classes
        self._conf = torch.zeros(C * C, dtype=torch.long, device=device)
        self._pred = torch.zeros(C, dtype=torch.long, device=device)
        self._true = torch.zeros(C, dtype=torch.long, device=device)
        self.n_seen = 0

    @torch.no_grad()
    def update(self, pseudo_labels, mask, labels_u):
        C = self.num_classes
        pseudo_labels = pseudo_labels.long()
        labels_u = labels_u.to(pseudo_labels.device).long()
        accepted = mask > 0
        self._pred += torch.bincount(pseudo_labels, minlength=C)
        self._true += torch.bincount(labels_u, minlength=C)
        self._conf += torch.bincount(labels_u[accepted] * C + pseudo_labels[accepted], minlength=C * C)
        self.n_seen += int(pseudo_labels.numel())

    def counts(self):
        C = self.num_classes
        return {'n_vistas': self.n_seen, 'pred': self._pred.cpu().tolist(), 'verdad': self._true.cpu().tolist(),
                'conf': self._conf.view(C, C).cpu().tolist()}

    def report(self, epoch, names=None, prior=None):
        """Readable table plus one JSON line (prefix JSON_TAG) with the raw counts of the epoch."""
        C = self.num_classes
        names = [str(n) for n in names] if names is not None else [str(c) for c in range(C)]
        k = self.counts()
        n = k['n_vistas']
        conf = np.asarray(k['conf'], dtype=np.int64)
        pred, true = np.asarray(k['pred']), np.asarray(k['verdad'])
        acc, ok = conf.sum(axis=0), np.diag(conf)

        def pct(a, b):
            return 100.0 * a / b if b else float('nan')

        lines = ['[PL por clase] epoch {} | pool seen {} | accepted {} ({:.1f} %) | precision {:.1f} %'.format(
                     epoch, n, int(acc.sum()), pct(acc.sum(), n), pct(ok.sum(), acc.sum())),
                 '  {:<8s} {:>7s} {:>7s} {:>7s} {:>11s} {:>7s} {:>10s} {:>7s} | true composition of the '
                 'accepted (%): {}'.format('class', 'true%', 'pred%', 'acc%', 'acc/pred%', 'prec%',
                                            'coverage%', 'n_acc', ' '.join('{:>6s}'.format(x[:6]) for x in names))]
        for c in range(C):
            comp = ' '.join('{:6.1f}'.format(pct(conf[t, c], acc[c])) for t in range(C))
            lines.append('  {:<8s} {:7.1f} {:7.1f} {:7.1f} {:11.1f} {:7.1f} {:10.1f} {:7d} | {}'.format(
                names[c][:8], pct(true[c], n), pct(pred[c], n), pct(acc[c], n), pct(acc[c], pred[c]),
                pct(ok[c], acc[c]), pct(ok[c], true[c]), int(acc[c]), comp))
        payload = {'epoca': int(epoch), 'clases': names, 'n_vistas': n, 'pred': k['pred'], 'verdad': k['verdad'],
                   'conf_verdad_x_pseudo': k['conf']}
        if prior is not None:
            payload.update({'modo': prior.mode, 'pi_pool': [round(float(x), 6) for x in prior.pi_pool],
                            'pi_lab': [round(float(x), 6) for x in prior.pi_lab], 'pasos_em': prior.n_ema_updates})
        lines.append(JSON_TAG + ' ' + json.dumps(payload))
        return '\n'.join(lines)


class LogitAdjust:
    """Adds log pi_pool - log pi_lab; without a prior returns the same tensor."""

    def __init__(self, state):
        self.state = state
        self._t, self._v = None, None

    def __call__(self, y):
        if self.state is None:
            return y
        clave = (self.state.n_ema_updates, y.device, y.dtype)
        if self._t is None or self._v != clave:
            self._t = torch.as_tensor(np.asarray(self.state.log_ratio(), dtype=np.float32)).to(device=y.device,
                                                                                               dtype=y.dtype)
            self._v = clave
        return y + self._t


class PriorAwareThresholding:
    """FlexMatch thresholding with the learning status divided by w(c) = pi_pool(c) / pi_lab(c); w = 1 is the original."""

    def __init__(self, base, state):
        self.base = base
        self.state = state

    def __getattr__(self, name):
        return getattr(self.base, name)

    def class_weight(self):
        return (self.state.pi_pool / self.state.pi_lab).tolist()

    def get_threshold(self, pseudo_labels):
        b = self.base
        pseudo_counter = Counter(b.net_outputs.tolist())
        if max(pseudo_counter.values()) == b.n_unlabeled_samples:
            status = torch.zeros(b.num_classes).to(b.device)
        else:
            w = self.class_weight()
            rates = [pseudo_counter[c] / w[c] for c in range(b.num_classes)]
            candidatos = rates + ([pseudo_counter[-1]] if b.warmup and -1 in pseudo_counter else [])
            max_num = max(candidatos)
            status = torch.FloatTensor([r / max_num for r in rates]).to(b.device)
        return b.threshold * b.mapping_func(status[pseudo_labels])


def flexmatch_thresholding(module, state, args):
    """The original module, or the prior-aware one when a prior is used."""
    if state is None or opt(args, 'pool_prior_status') == 'conteo':
        return module
    return PriorAwareThresholding(module, state)
