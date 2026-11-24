"""MME, semi-supervised domain adaptation by minimax entropy (Saito et al., ICCV 2019), Study 2 baseline.

Trains on HAM10000 train (labeled source), the k labeled target images per class and the unlabeled target pool.
Prototype head: cosine between the L2-normalized bottleneck feature and the normalized class weights, divided by
T = 0.05. On the pool the head maximizes the entropy and the feature extractor minimizes it (gradient reversal,
weight lambda = 0.1). Each iteration takes 16 source, 16 labeled target and 16 pool images.
Protocol of the Study 1 grid: ResNet-50 from the HAM10000 classifier, bottleneck 1024, SGD Nesterov lr 0.003 with
the 'exp' decay (gamma 0.0004, decay 0.75), backbone at 0.1x, wd 5e-4, up to 100 epochs of 500 iterations, patience
15 on the target validation balanced accuracy, the same labeled subset and pool (utils.get_dataset, same seed), weak
augmentation (random resized crop and flip). No checkpoint is written: the best state is kept in memory.
Writes <log>/predictions.npz (val/test logits, labels, ids relative to the target root), result.json, labeled_ids.json.

Usage (from train/): python mme.py --target milk10k|bcn20000 --seed 0 [--data-root ../data]
                     [--backbone ../checkpoints/ham10000_source.pth] [--log ../outputs/ham/<target>/k16/mme/s0]
"""
import argparse
import copy
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import utils
import tllib.vision.datasets as datasets
from tllib.utils.data import ForeverDataIterator
from tllib.modules.grl import GradientReverseFunction

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGETS = {"milk10k": ("MILK10K", "MILK10K"), "bcn20000": ("BCN20000", "BCN20000")}
LAMBDA_ENT, TEMPERATURE = 0.1, 0.05


class PrototypeHead(nn.Module):
    def __init__(self, dim, num_classes, temperature=TEMPERATURE):
        super().__init__()
        self.fc = nn.Linear(dim, num_classes, bias=False)
        self.T = temperature

    def forward(self, f):
        return F.linear(F.normalize(f, dim=1), F.normalize(self.fc.weight, dim=1)) / self.T


def reversed_logits(model, x):
    """Gradient reversal between bottleneck and head (the minimax step)."""
    f = model.bottleneck(model.pool_layer(model.backbone(x)))
    return model.head(GradientReverseFunction.apply(f, 1.0))


def predict(model, ds, root, device, bs):
    model.eval()
    z, y = [], []
    with torch.no_grad():
        for xb, yb in DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0):
            z.append(model(xb.to(device)).float().cpu().numpy())
            y.append(yb.numpy())
    base = ds.dataset if isinstance(ds, Subset) else ds
    idx = ds.indices if isinstance(ds, Subset) else range(len(base.samples))
    ids = np.array([os.path.relpath(base.samples[i][0], root).replace("\\", "/") for i in idx])
    return np.concatenate(z).astype(np.float32), np.concatenate(y).astype(np.int64), ids


def balanced_accuracy(z, y):
    p = z.argmax(1)
    return 100.0 * np.mean([np.mean(p[y == c] == c) for c in np.unique(y)])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--target", required=True, choices=sorted(TARGETS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--num-samples-per-class", type=int, default=16)
    ap.add_argument("--data-root", default=os.path.join(ROOT, "data"))
    ap.add_argument("--backbone", default=os.path.join(ROOT, "checkpoints", "ham10000_source.pth"))
    ap.add_argument("--log", default=None, help="default: outputs/ham/<target>/k<k>/mme/s<seed>")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--iters-per-epoch", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dflag, folder = TARGETS[args.target]
    root = os.path.join(args.data_root, folder)
    out = args.log or os.path.join(ROOT, "outputs", "ham", args.target, "k%d" % args.num_samples_per_class, "mme",
                                   "s%d" % args.seed)
    if os.path.exists(os.path.join(out, "predictions.npz")):
        print("already done:", out)
        return
    os.makedirs(out, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    dev = torch.device(args.device)

    weak = utils.get_train_transform("default", random_horizontal_flip=True)
    val_tf = utils.get_val_transform("default")
    lab, unl, val = utils.get_dataset(dflag, args.num_samples_per_class, root, weak, val_tf,
                                      unlabeled_train_transform=weak, seed=args.seed)
    utils.guardar_ids_etiquetados(lab, root, out)
    src = datasets.HAM10000(root=os.path.join(args.data_root, "HAM10000"), split="train", transform=weak)
    test = datasets.__dict__[dflag](root=root, split="test", transform=val_tf)
    print("source %d, labeled %d, pool %d, val %d, test %d" % (len(src), len(lab), len(unl), len(val), len(test)))
    kw = dict(batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.workers,
              persistent_workers=args.workers > 0, pin_memory=dev.type == "cuda")
    it_s, it_l, it_u = (ForeverDataIterator(DataLoader(d, **kw)) for d in (src, lab, unl))

    backbone = utils.get_model("resnet50", pretrained_checkpoint=args.backbone)
    model = utils.ImageClassifier(backbone, 5, bottleneck_dim=1024, head=PrototypeHead(1024, 5), finetune=True).to(dev)
    opt = torch.optim.SGD(model.get_parameters(), 0.003, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda x: 0.003 * (1. + 0.0004 * float(x)) ** (-0.75))

    t0, best, best_epoch, stale, history, state = time.time(), -1.0, -1, 0, [], None
    for epoch in range(args.epochs):
        model.train()
        for i in range(args.iters_per_epoch):
            (xs, ys), (xl, yl) = next(it_s), next(it_l)
            x = torch.cat([xs, xl]).to(dev, non_blocking=True)
            y = torch.cat([ys, yl]).to(dev, non_blocking=True)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            xu, _ = next(it_u)
            p = F.softmax(reversed_logits(model, xu.to(dev, non_blocking=True)), dim=1)
            ent = LAMBDA_ENT * torch.mean(torch.sum(p * torch.log(p + 1e-5), 1))  # -lambda * H: the head raises H
            ent.backward()
            opt.step()
            sched.step()
            if i % 250 == 0:
                print("epoch %d it %d  ce %.3f  ent %.4f" % (epoch, i, loss.item(), ent.item()), flush=True)
        zv, yv, _ = predict(model, val, root, dev, 64)
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
    zv, yv, iv = predict(model, val, root, dev, 64)
    zt, yt, it = predict(model, test, root, dev, 64)
    meta = dict(method="mme", lambda_entropy=LAMBDA_ENT, temperature=TEMPERATURE, target=args.target,
                k=args.num_samples_per_class, seed=args.seed, best_epoch=best_epoch, val_balanced_accuracy=balanced_accuracy(zv, yv),
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
