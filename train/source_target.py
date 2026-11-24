"""S+T: ERM on the labeled target images plus cross-entropy on HAM10000 train (Study 2 baseline).

Each iteration: 16 labeled target images, loss CE(weak) + 0.1 CE(strong, RandAugment m10 n2), and 16 HAM10000 train
images (weak view), plain CE with weight 1. No unlabeled data. Optimizer and schedule of the Mean Teacher recipe of the
grid: SGD Nesterov lr 0.01, 'exp' decay (gamma 0.0001, decay 0.75), backbone at 0.1x, wd 5e-4; ResNet-50 from the
HAM10000 classifier, bottleneck 1024, up to 100 epochs of 500 iterations, patience 15 on the target validation
balanced accuracy. Same labeled subset as the grid (utils.get_dataset, same seed). The best state is kept in memory.
Writes <log>/predictions.npz (val/test logits, labels, ids relative to the target root), result.json, labeled_ids.json.

Usage (from train/): python source_target.py --target milk10k|bcn20000|milk10k_clinico --seed 0 [--data-root ../data]
                     [--backbone ../checkpoints/ham10000_source.pth] [--log ../outputs/ham/<target>/k16/source_target/s0]
"""
import argparse
import copy
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset

import utils
import tllib.vision.datasets as datasets
from tllib.utils.data import ForeverDataIterator
from tllib.vision.transforms import MultipleApply

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# target -> (dataset flag, data folder, train resizing)
TARGETS = {"milk10k": ("MILK10K", "MILK10K", "default"),
           "milk10k_clinico": ("MILK10K", "MILK10K_CLINICO", "lesion_preserving"),
           "bcn20000": ("BCN20000", "BCN20000", "default")}


def predict(model, ds, root, device, bs):
    model.eval()
    z, y = [], []
    with torch.no_grad():
        for xb, yb in DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0, pin_memory=True):
            z.append(model(xb.to(device)).float().cpu().numpy())
            y.append(yb.numpy())
    base = ds.dataset if isinstance(ds, Subset) else ds
    idx = ds.indices if isinstance(ds, Subset) else range(len(base.samples))
    ids = np.array([os.path.relpath(base.samples[i][0], root).replace("\\", "/") for i in idx])
    return np.concatenate(z).astype(np.float32), np.concatenate(y).astype(np.int64), ids


def balanced_accuracy(z, y):
    p = z.argmax(1)
    return 100.0 * np.mean([np.mean(p[y == c] == c) for c in range(5)])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--target", required=True, choices=sorted(TARGETS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--num-samples-per-class", type=int, default=16)
    ap.add_argument("--data-root", default=os.path.join(ROOT, "data"))
    ap.add_argument("--backbone", default=os.path.join(ROOT, "checkpoints", "ham10000_source.pth"))
    ap.add_argument("--log", default=None, help="default: outputs/ham/<target>/k<k>/source_target/s<seed>")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--lr-gamma", type=float, default=0.0001)
    ap.add_argument("--lr-decay", type=float, default=0.75)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--trade-off-cls-strong", type=float, default=0.1)
    ap.add_argument("--source-weight", type=float, default=1.0)
    ap.add_argument("--auto-augment", default="rand-m10-n2-mstd2")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--iters-per-epoch", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--print-freq", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dflag, folder, resizing = TARGETS[args.target]
    root = os.path.join(args.data_root, folder)
    out = args.log or os.path.join(ROOT, "outputs", "ham", args.target, "k%d" % args.num_samples_per_class,
                                   "source_target", "s%d" % args.seed)
    if os.path.exists(os.path.join(out, "predictions.npz")):
        print("already done:", out)
        return
    os.makedirs(out, exist_ok=True)
    print(args, flush=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)

    weak = utils.get_train_transform(resizing)
    strong = utils.get_train_transform(resizing, auto_augment=args.auto_augment)
    val_tf = utils.get_val_transform("default")
    labeled, _, val = utils.get_dataset(dflag, args.num_samples_per_class, root, MultipleApply([weak, strong]),
                                        val_tf, weak, args.seed)
    utils.guardar_ids_etiquetados(labeled, root, out)
    test = datasets.__dict__[dflag](root, split="test", transform=val_tf)
    source = datasets.HAM10000(os.path.join(args.data_root, "HAM10000"), split="train",
                               transform=utils.get_train_transform("default"))
    print("source %d, labeled %d, val %d, test %d" % (len(source), len(labeled), len(val), len(test)), flush=True)
    kw = dict(batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0, pin_memory=True)
    # The labeled iterator is built before the model and the source iterator after it, as in the thesis runs.
    labeled_iter = ForeverDataIterator(DataLoader(labeled, **kw))
    backbone = utils.get_model("resnet50", pretrained=False, pretrained_checkpoint=args.backbone)
    model = utils.ImageClassifier(backbone, 5, bottleneck_dim=1024, finetune=True).to(device)
    optimizer = SGD(model.get_parameters(), args.lr, momentum=0.9, weight_decay=args.wd, nesterov=True)
    scheduler = LambdaLR(optimizer, lambda step: args.lr * (1 + args.lr_gamma * float(step)) ** (-args.lr_decay))
    source_iter = ForeverDataIterator(DataLoader(source, **kw))

    t0, best, best_epoch, stale, history, state = time.time(), 0., -1, 0, [], None
    step = 0
    for epoch in range(args.epochs):
        model.train()
        for _ in range(args.iters_per_epoch):
            (x_weak, x_strong), y = next(labeled_iter)
            xs, ys = next(source_iter)
            x_weak, x_strong, y, xs, ys = (v.to(device, non_blocking=True) for v in (x_weak, x_strong, y, xs, ys))
            optimizer.zero_grad()
            target_loss = F.cross_entropy(model(x_weak), y) + \
                args.trade_off_cls_strong * F.cross_entropy(model(x_strong), y)
            target_loss.backward()
            source_loss = F.cross_entropy(model(xs), ys)
            (args.source_weight * source_loss).backward()
            optimizer.step()
            scheduler.step()
            step += 1
            if step % args.print_freq == 0:
                print("epoch %d step %d  target ce %.3f  source ce %.3f" % (epoch, step, target_loss.item(),
                                                                          source_loss.item()), flush=True)
        zv, yv, _ = predict(model, val, root, device, args.batch_size)
        ba = balanced_accuracy(zv, yv)
        history.append(round(ba, 3))
        print("epoch %d  val BA %.2f  (best %.2f)  %.1f min" % (epoch, ba, best, (time.time() - t0) / 60), flush=True)
        if ba > best:
            best, best_epoch, stale = ba, epoch, 0
            state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(state)
    zv, yv, iv = predict(model, val, root, device, args.batch_size)
    zt, yt, it = predict(model, test, root, device, args.batch_size)
    meta = dict(method="source_target", target=args.target, k=args.num_samples_per_class, seed=args.seed,
                best_epoch=best_epoch, val_balanced_accuracy=balanced_accuracy(zv, yv),
                test_balanced_accuracy=balanced_accuracy(zt, yt), val_history=history, minutes=(time.time() - t0) / 60)
    tmp = os.path.join(out, "predictions.tmp.npz")
    np.savez_compressed(tmp, val_logits=zv, val_labels=yv, val_ids=iv, test_logits=zt, test_labels=yt, test_ids=it,
                        meta=json.dumps(meta))
    os.replace(tmp, os.path.join(out, "predictions.npz"))
    with open(os.path.join(out, "result.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
