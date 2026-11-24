"""One-epoch fine-tuning used to compare labeled-set selection strategies (Study 2, label selection).

For each labeled set (80 images, a JSON from results/study2/selection_sets/): the ResNet-50 HAM10000 classifier with a
new bottleneck (1024) and head, all trainable (backbone at 0.1x); 500 iterations of batch 16, SGD Nesterov lr 0.003,
'exp' decay (gamma 0.0004, decay 0.75), wd 5e-4; loss CE(weak) + 0.1 CE(strong, RandAugment m10 n2) with class
weights N / (K n_c) (all 1 for a balanced set); mixed precision. No validation or early stopping: the network after
iteration 500 predicts the full test set. The 80 images are held in memory (BCN20000 images decoded at 512 px).
Writes <out>/<target>_<strategy>_s<set>.npz (test logits, labels, ids, labeled ids) and appends to <out>/runs.csv.

Usage (from train/): python short_finetune.py --target milk10k --labeled-list ../results/study2/selection_sets/milk10k_*_s*.json
                     [--data-root ../data] [--backbone ../checkpoints/ham10000_source.pth] [--out ../outputs/short_finetune]
"""
import argparse
import copy
import csv
import glob
import json
import os
import queue
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

import utils
from tllib.vision.transforms import ResizeImage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOLDERS = {"milk10k": "MILK10K", "bcn20000": "BCN20000"}
CLASSES = ("akiec", "bcc", "bkl", "mel", "nv")
ITERS, BATCH, LR, GAMMA, DECAY, WD, STRONG = 500, 16, 0.003, 0.0004, 0.75, 5e-4, 0.1
BCN_SIDE = 512


def open_image(root, rel, bcn):
    im = Image.open(os.path.join(root, rel))
    if bcn:
        im.draft("RGB", (BCN_SIDE, BCN_SIDE))
    im = im.convert("RGB")
    if bcn and max(im.size) > BCN_SIDE * 1.25:
        f = BCN_SIDE / max(im.size)
        im = im.resize((round(im.size[0] * f), round(im.size[1] * f)), Image.BILINEAR)
    return im


def load_test(root):
    """uint8 [N, 224, 224, 3] with the validation transform (resize 256, center crop 224), ImageFolder order."""
    ids = ["test/%s/%s" % (c, f) for c in CLASSES for f in sorted(os.listdir(os.path.join(root, "test", c)))]
    tf = T.Compose([ResizeImage(256), T.CenterCrop(224)])

    def one(rel):
        return np.asarray(tf(Image.open(os.path.join(root, rel)).convert("RGB")), dtype=np.uint8)
    with ThreadPoolExecutor(6) as ex:
        x = np.stack(list(ex.map(one, ids)))
    return x, ids


class Runner:
    def __init__(self, backbone, iters, device):
        self.iters, self.dev = iters, device
        self.amp = device.type == "cuda"
        torch.backends.cudnn.benchmark = True
        self.backbone0 = utils.get_model("resnet50", pretrained_checkpoint=backbone)
        weak = utils.get_train_transform("default", random_horizontal_flip=True)
        strong = utils.get_train_transform("default", random_horizontal_flip=True, auto_augment="rand-m10-n2-mstd2")
        # drop ToTensor + Normalize: images go to the device as uint8 and are normalized there
        self.weak, self.strong = T.Compose(weak.transforms[:-2]), T.Compose(strong.transforms[:-2])
        self.mean = (torch.tensor([0.485, 0.456, 0.406], device=device) * 255).view(1, 3, 1, 1)
        self.std = (torch.tensor([0.229, 0.224, 0.225], device=device) * 255).view(1, 3, 1, 1)

    def to_device(self, x):
        t = torch.from_numpy(np.ascontiguousarray(x)).to(self.dev, non_blocking=True)
        return (t.permute(0, 3, 1, 2).contiguous().float() - self.mean) / self.std

    def producer(self, imgs, y, q):
        """A single thread draws the batch order and the augmentations, so the seed fixes both."""
        n, it = len(imgs), 0
        while it < self.iters:
            perm = torch.randperm(n).tolist()
            for b in range(n // BATCH):
                if it >= self.iters:
                    break
                idx = perm[b * BATCH:(b + 1) * BATCH]
                xw, xs = [], []
                for i in idx:  # weak then strong per image: this order fixes the random stream
                    xw.append(np.asarray(self.weak(imgs[i]), dtype=np.uint8))
                    xs.append(np.asarray(self.strong(imgs[i]), dtype=np.uint8))
                q.put((np.stack(xw), np.stack(xs), y[idx]))
                it += 1
        q.put(None)

    def run(self, root, bcn, ids, y, seed, x_test):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        imgs = [open_image(root, i, bcn) for i in ids]
        model = utils.ImageClassifier(copy.deepcopy(self.backbone0), 5, bottleneck_dim=1024, pool_layer=None,
                                      finetune=True).to(self.dev)
        opt = torch.optim.SGD(model.get_parameters(), LR, momentum=0.9, weight_decay=WD, nesterov=True)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda x: LR * (1. + GAMMA * float(x)) ** (-DECAY))
        n = np.bincount(y, minlength=5)
        present = n > 0
        weights = torch.tensor(np.where(present, len(y) / (present.sum() * np.maximum(n, 1)), 0.0),
                               dtype=torch.float32, device=self.dev)
        scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        q = queue.Queue(maxsize=6)
        thread = threading.Thread(target=self.producer, args=(imgs, y, q), daemon=True)
        thread.start()
        model.train()
        it = 0
        while True:
            batch = q.get()
            if batch is None:
                break
            xw, xs, yl = batch
            xw, xs, yl = self.to_device(xw), self.to_device(xs), torch.from_numpy(yl).to(self.dev)
            with torch.autocast(self.dev.type, dtype=torch.float16, enabled=self.amp):
                zw = model(xw)
                zs = model(xs)
            loss = F.cross_entropy(zw.float(), yl, weight=weights) + STRONG * F.cross_entropy(zs.float(), yl, weight=weights)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            it += 1
        thread.join()
        model.eval()
        z = []
        with torch.no_grad():
            for i in range(0, len(x_test), 64):
                with torch.autocast(self.dev.type, dtype=torch.float16, enabled=self.amp):
                    z.append(model(self.to_device(x_test[i:i + 64])).float().cpu().numpy())
        return np.concatenate(z).astype(np.float32), float(loss.item())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--target", required=True, choices=sorted(FOLDERS))
    ap.add_argument("--labeled-list", nargs="+", required=True, help="JSON files (globs allowed)")
    ap.add_argument("--seed", type=int, default=None, help="default: the 'seed' of each list")
    ap.add_argument("--data-root", default=os.path.join(ROOT, "data"))
    ap.add_argument("--backbone", default=os.path.join(ROOT, "checkpoints", "ham10000_source.pth"))
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "short_finetune"))
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    lists = sorted({p for pat in args.labeled_list for p in (glob.glob(pat) or [pat])})
    root = os.path.join(args.data_root, FOLDERS[args.target])
    bcn = args.target == "bcn20000"
    os.makedirs(args.out, exist_ok=True)
    runner = Runner(args.backbone, args.iters, torch.device(args.device))
    x_test, test_ids = load_test(root)
    y_test = np.array([CLASSES.index(i.split("/")[1]) for i in test_ids])
    for path in lists:
        d = json.load(open(path, encoding="utf-8"))
        ids = d["ids"]
        assert len(ids) == len(set(ids)) and all(os.path.isfile(os.path.join(root, i)) for i in ids), path
        y = np.array([CLASSES.index(i.split("/")[1]) for i in ids])
        seed = d.get("seed", 0) if args.seed is None else args.seed
        name = os.path.splitext(os.path.basename(path))[0]
        out = os.path.join(args.out, name + ".npz")
        if os.path.exists(out):
            print("already done:", out)
            continue
        t0 = time.time()
        z, last_loss = runner.run(root, bcn, ids, y, seed, x_test)
        recall = [float((z[y_test == k].argmax(1) == k).mean()) for k in range(5)]
        ba = 100 * float(np.mean(recall))
        np.savez_compressed(out, logits=z, labels=y_test, ids=np.array(test_ids), labeled_ids=np.array(ids),
                            labeled_labels=y)
        row = dict(target=args.target, labeled_list=os.path.basename(path), strategy=d.get("strategy", ""),
                   set=d.get("set", ""), seed=seed, iters=args.iters, test_balanced_accuracy=round(ba, 3),
                   **{"recall_" + c: round(100 * r, 2) for c, r in zip(CLASSES, recall)},
                   **{"n_" + c: int(v) for c, v in zip(CLASSES, np.bincount(y, minlength=5))},
                   last_loss=round(last_loss, 4), seconds=round(time.time() - t0, 1))
        csv_path = os.path.join(args.out, "runs.csv")
        new = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        print("%s  test BA %.2f  (%.0f s)" % (name, ba, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
