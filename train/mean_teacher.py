"""
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
import random
import time
import warnings
import argparse
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

import utils
import pool_modes
from tllib.self_training.pi_model import sigmoid_warm_up, L2ConsistencyLoss
from tllib.self_training.mean_teacher import update_bn, EMATeacher
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


def main(args: argparse.Namespace):
    import time
    _train_start = time.time()

    logger = CompleteLogger(args.log, args.phase)
    print(args)

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
    unlabeled_train_transform = MultipleApply([weak_augment, weak_augment])
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
                          **utils.kw_pool_exclude(args))
    utils.guardar_pool_exclude(unlabeled_train_dataset, args.log)
    print("labeled_dataset_size: ", len(labeled_train_dataset))
    print('unlabeled_dataset_size: ', len(unlabeled_train_dataset))
    print("val_dataset_size: ", len(val_dataset))

    labeled_train_loader = DataLoader(labeled_train_dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.workers, drop_last=True, persistent_workers=(args.workers > 0), pin_memory=True)
    unlabeled_train_loader = DataLoader(unlabeled_train_dataset, batch_size=args.batch_size, shuffle=True,
                                        num_workers=args.workers, drop_last=True, persistent_workers=(args.workers > 0), pin_memory=True)
    labeled_train_iter = ForeverDataIterator(labeled_train_loader)
    unlabeled_train_iter = ForeverDataIterator(unlabeled_train_loader)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    print("=> using pre-trained model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch, pretrained_checkpoint=args.pretrained_backbone)
    num_classes = labeled_train_dataset.num_classes
    pool_layer = nn.Identity() if args.no_pool else None
    classifier = utils.ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim, pool_layer=pool_layer,
                                       finetune=args.finetune).to(device)
    utils.cargar_alumno(classifier, args)
    teacher = EMATeacher(classifier, alpha=args.alpha)
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

    if args.phase == 'test':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)
        acc1, avg = utils.validate(val_loader, classifier, args, device, num_classes)
        print(acc1)
        return

    best_acc1 = 0.0
    best_avg = 0.0
    es_counter = 0
    best_teacher_avg = 0.0

    start_epoch = 0
    if args.resume:
        print(f"Resume since {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        classifier.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        teacher.load_state_dict(checkpoint.get('teacher_state_dict', checkpoint['state_dict']))
        if 'teacher_state_dict' not in checkpoint:
            print("WARNING: old checkpoint without EMA teacher; initialized from the resumed student.")
        start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint.get('best_acc1', 0.0)
        best_avg = checkpoint.get('best_avg', 0.0)
        print(f"Starting on epoch {start_epoch}, best_acc1={best_acc1:.1f}")
        es_counter = checkpoint.get('es_counter', 0)
        best_teacher_avg = checkpoint.get('best_teacher_avg', 0.0)

    for epoch in range(start_epoch, args.epochs):
        print(lr_scheduler.get_lr())
        _timer.start("1. Train epoch")

        train(labeled_train_iter, unlabeled_train_iter, classifier, teacher, optimizer, lr_scheduler, epoch, args)

        _timer.stop()
        _timer.start("2. Validation")
        acc1, avg = utils.validate(val_loader, classifier, args, device, num_classes)
        if args.evaluar_teacher:
            t_avg = _evaluar_teacher(val_loader, teacher, args, device, num_classes)
            print("TEACHER epoch={} val_bal={:.3f} student_val_bal={:.3f}".format(epoch, t_avg, avg))
            if t_avg > best_teacher_avg:
                best_teacher_avg = t_avg
                utils.atomic_torch_save(teacher.state_dict(), logger.get_checkpoint_path('teacher_best'))

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
            'teacher_state_dict': teacher.state_dict(),
            'best_teacher_avg': best_teacher_avg,
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
    if args.evaluar_teacher:
        # Both teachers on test (student-chosen epoch and the teacher's own best val), in a format log parsers skip.
        import copy as _copy
        modelo_t = _copy.deepcopy(classifier)
        if isinstance(checkpoint, dict) and 'teacher_state_dict' in checkpoint:
            modelo_t.load_state_dict(checkpoint['teacher_state_dict'])
            b, f1t, auct = _test_teacher(test_loader, modelo_t, device)
            print("TEACHER_TEST seleccion=student_best bal={:.4f} f1={:.4f} auc={:.4f}".format(b, f1t, auct))
        if _os.path.exists(logger.get_checkpoint_path('teacher_best')):
            modelo_t.load_state_dict(torch.load(logger.get_checkpoint_path('teacher_best'), map_location='cpu'))
            b, f1t, auct = _test_teacher(test_loader, modelo_t, device)
            print("TEACHER_TEST seleccion=teacher_best val_bal={:.4f} bal={:.4f} f1={:.4f} auc={:.4f}".format(
                best_teacher_avg, b, f1t, auct))
    _timer.stop()
    _timer.report()
    _elapsed = time.time() - _train_start
    print(f"Total training time: {_elapsed/60:.1f} min ({_elapsed:.1f}s)")
    print("best_acc1 = {:3.1f}".format(best_acc1))
    print('best_avg = {:3.1f}'.format(best_avg))
    logger.close()


def _evaluar_teacher(val_loader, teacher, args, device, num_classes):
    """Validates the EMA teacher, restoring the RNG state and silencing utils.validate so training and parsing are unchanged."""
    import contextlib, io
    import numpy as np
    estado = (torch.get_rng_state(), random.getstate(), np.random.get_state(),
              torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    with contextlib.redirect_stdout(io.StringIO()):
        _, avg = utils.validate(val_loader, teacher.teacher, args, device, num_classes)
    torch.set_rng_state(estado[0]); random.setstate(estado[1]); np.random.set_state(estado[2])
    if estado[3] is not None:
        torch.cuda.set_rng_state_all(estado[3])
    return avg


def _test_teacher(test_loader, modelo, device):
    """Balanced accuracy, macro F1 and AUC of a model on test, on a single line."""
    import numpy as np
    from sklearn.metrics import f1_score, recall_score, roc_auc_score
    modelo.eval()
    p, y, pr = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            out = modelo(images.to(device, non_blocking=True))
            out = out[0] if isinstance(out, tuple) else out
            pr.append(torch.softmax(out, dim=1).cpu().numpy()); p.append(out.argmax(1).cpu().numpy()); y.append(labels.numpy())
    p, y, pr = np.concatenate(p), np.concatenate(y), np.concatenate(pr)
    return (100 * recall_score(y, p, average='macro'), 100 * f1_score(y, p, average='macro'),
            roc_auc_score(y, pr, multi_class='ovr', average='macro'))


def train(labeled_train_iter: ForeverDataIterator, unlabeled_train_iter: ForeverDataIterator, model, teacher,
          optimizer: SGD, lr_scheduler: LambdaLR, epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':2.2f')
    data_time = AverageMeter('Data', ':2.1f')
    cls_losses = AverageMeter('Cls Loss', ':3.2f')
    con_losses = AverageMeter('Con Loss', ':3.2f')
    losses = AverageMeter('Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cls_losses, con_losses, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    consistency_criterion = L2ConsistencyLoss(reduction='sum').to(device)
    model.train()
    teacher.train()

    end = time.time()
    batch_size = args.batch_size
    for i in range(args.iters_per_epoch):
        (x_l, x_l_strong), labels_l = next(labeled_train_iter)
        x_l = x_l.to(device, non_blocking=True)
        x_l_strong = x_l_strong.to(device, non_blocking=True)
        labels_l = labels_l.to(device, non_blocking=True)

        (x_u, x_u_teacher), _ = next(unlabeled_train_iter)
        x_u = x_u.to(device, non_blocking=True)
        x_u_teacher = x_u_teacher.to(device, non_blocking=True)

        data_time.update(time.time() - end)

        optimizer.zero_grad()

        y_l = model(x_l)
        y_l_strong = model(x_l_strong)
        cls_loss = F.cross_entropy(y_l, labels_l) + args.trade_off_cls_strong * F.cross_entropy(y_l_strong, labels_l)
        cls_loss.backward()

        with torch.no_grad():
            y_u_teacher = teacher(x_u_teacher)
            p_u_teacher = torch.softmax(y_u_teacher, dim=1)
        y_u = model(x_u)
        p_u = torch.softmax(y_u, dim=1)
        con_loss = args.trade_off_con * sigmoid_warm_up(epoch, args.warm_up_epochs) * \
                   consistency_criterion(p_u, p_u_teacher)
        con_loss.backward()

        loss = cls_loss + con_loss
        losses.update(loss.item(), batch_size)
        cls_losses.update(cls_loss.item(), batch_size)
        con_losses.update(con_loss.item(), batch_size)

        cls_acc = accuracy(y_l, labels_l)[0]
        cls_accs.update(cls_acc.item(), batch_size)

        optimizer.step()
        lr_scheduler.step()

        global_step = epoch * args.iters_per_epoch + i + 1
        teacher.set_alpha(min(args.alpha, 1 - 1 / global_step))
        teacher.update()
        update_bn(model, teacher.teacher)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mean Teacher for Semi Supervised Learning')
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
    parser.add_argument('--evaluar-teacher', action='store_true', default=False,
                        help='also evaluate the EMA teacher every epoch and on test (training is unchanged)')
    parser.add_argument('--alpha', default=0.999, type=float,
                        help='ema decay factor')
    parser.add_argument('--trade-off-cls-strong', default=0.1, type=float,
                        help='the trade-off hyper-parameter of cls loss on strong augmented labeled data')
    parser.add_argument('--trade-off-con', default=0.1, type=float,
                        help='the trade-off hyper-parameter of consistency loss')
    parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', dest='lr',
                        help='initial learning rate')
    parser.add_argument('--lr-scheduler', default='exp', type=str, choices=['exp', 'cos'],
                        help='learning rate decay strategy')
    parser.add_argument('--lr-gamma', default=0.0001, type=float,
                        help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float,
                        help='parameter for lr scheduler')
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float, metavar='W',
                        help='weight decay (default:5e-4)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=40, type=int, metavar='N',
                        help='number of total epochs to run (default: 40)')
    parser.add_argument('--warm-up-epochs', default=10, type=int,
                        help='number of epochs to warm up (default: 10)')
    parser.add_argument('-i', '--iters-per-epoch', default=500, type=int,
                        help='number of iterations per epoch (default: 500)')
    parser.add_argument('-p', '--print-freq', default=100, type=int, metavar='N',
                        help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training ')
    parser.add_argument("--log", default='mean_teacher', type=str,
                        help="where to save logs, checkpoints and debugging images")
    parser.add_argument("--phase", default='train', type=str, choices=['train', 'test'],
                        help="when phase is 'test', only test the model")
    parser.add_argument('--patience', default=15, type=int,
                        help='early stopping patience (default: 15)')
    parser.add_argument('--resume', default=None, type=str,
                        help='path to a checkpoint to resume training from')
    utils.add_pool_exclude_args(parser)
    utils.add_init_student_args(parser)
    args = parser.parse_args()
    main(args)
