"""ConvNeXt-T source model on HAM10000 (Study 2, the second source expert).

ImageNet-1k ConvNeXt-T (torchvision) with a new 5-class head, trained on HAM10000 train: 8 epochs, AdamW
(body 2e-4, head 1e-3, wd 0.05) with a one-cycle schedule, batch 32, label smoothing 0.1, mixed precision, resize to
256 + random 224 crop + horizontal and vertical flips. The epoch with the best HAM10000 validation balanced accuracy is
kept. Writes --checkpoint ({"state_dict": features.*, norm.*, head.*}; what ``-a convnext_tiny_ham
--pretrained-backbone`` loads) and a .json with the same name.

Usage (from the repo root): python train/convnext_source.py [--data-root data/HAM10000] [--checkpoint checkpoints/ham10000_convnext.pth]
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES = ["akiec", "bcc", "bkl", "mel", "nv"]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class ConvNeXtHAM(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        m = torchvision.models.convnext_tiny(weights=weights)
        self.features, self.avgpool = m.features, m.avgpool
        self.norm = m.classifier[0]  # LayerNorm2d
        self.head = nn.Linear(768, 5)

    def forward(self, x):
        f = torch.flatten(self.norm(self.avgpool(self.features(x))), 1)
        return self.head(f), f


def list_split(root, split):
    out = []
    for k, c in enumerate(CLASSES):
        d = os.path.join(root, split, c)
        if os.path.isdir(d):
            out += [(os.path.join(d, f), k) for f in sorted(os.listdir(d)) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    return out


def open_rgb(path):
    with open(path, "rb") as f:
        return Image.open(f).convert("RGB")


def center_view(im):
    return im.resize((256, 256)).crop((16, 16, 240, 240))


def augment(im, rng):
    im = im.resize((256, 256))
    x0, y0 = rng.integers(0, 33, 2)
    im = im.crop((x0, y0, x0 + 224, y0 + 224))
    if rng.random() < 0.5:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() < 0.5:
        im = im.transpose(Image.FLIP_TOP_BOTTOM)
    return im


def to_tensor(images, device):
    u8 = torch.from_numpy(np.stack([np.asarray(im, dtype=np.uint8) for im in images])).to(device)
    x = u8.permute(0, 3, 1, 2).float().div_(255)
    return (x - MEAN.to(device)) / STD.to(device)


def balanced_accuracy(y, p):
    return 100 * float(np.mean([(p[y == k] == k).mean() for k in range(5) if (y == k).any()]))


def evaluate(model, items, device, batch=48):
    model.eval()
    y = np.array([k for _, k in items])
    pred = []
    with torch.no_grad():
        for i in range(0, len(items), batch):
            x = to_tensor([center_view(open_rgb(p)) for p, _ in items[i:i + batch]], device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                pred.append(model(x)[0].float().argmax(1).cpu().numpy())
    return balanced_accuracy(y, np.concatenate(pred))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", default=os.path.join(ROOT, "data", "HAM10000"))
    ap.add_argument("--checkpoint", default=os.path.join(ROOT, "checkpoints", "ham10000_convnext.pth"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--max-steps", type=int, default=None, help="stop each epoch after this many steps (smoke test)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    t0 = time.time()
    device = torch.device(args.device)
    train, val, test = (list_split(args.data_root, s) for s in ("train", "val", "test"))
    if not (train and val and test):
        raise SystemExit("no HAM10000 train/val/test class folders under %s" % args.data_root)
    print("HAM10000: train %d, val %d, test %d | %s" % (len(train), len(val), len(test), device), flush=True)
    if args.max_steps:
        rs = np.random.RandomState(0)
        val, test = ([x[i] for i in rs.choice(len(x), 24, replace=False)] for x in (val, test))
    os.makedirs(os.path.dirname(os.path.abspath(args.checkpoint)), exist_ok=True)

    torch.manual_seed(args.seed)
    model = ConvNeXtHAM(pretrained=True).to(device)
    body = list(model.features.parameters()) + list(model.norm.parameters())
    opt = torch.optim.AdamW([{"params": body, "lr": 2e-4}, {"params": model.head.parameters(), "lr": 1e-3}],
                            weight_decay=0.05)
    steps = int(np.ceil(len(train) / args.batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[2e-4, 1e-3], total_steps=args.epochs * steps,
                                                pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    rng = np.random.default_rng(args.seed)  # drives the batch order and the augmentation
    best, history = None, []
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(len(train))
        losses = []
        for s in range(steps if not args.max_steps else min(steps, args.max_steps)):
            idx = order[s * args.batch_size:(s + 1) * args.batch_size]
            x = to_tensor([augment(open_rgb(train[i][0]), rng) for i in idx], device)
            y = torch.tensor([train[i][1] for i in idx], device=device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = F.cross_entropy(model(x)[0].float(), y, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(float(loss))
            if (s + 1) % 50 == 0:
                print("  epoch %d step %d/%d loss %.3f (%.1f min)" % (epoch + 1, s + 1, steps, np.mean(losses[-50:]),
                                                                    (time.time() - t0) / 60), flush=True)
        ba = evaluate(model, val, device)
        history.append(dict(epoch=epoch + 1, loss=float(np.mean(losses)), val_balanced_accuracy=ba))
        print("epoch %d: HAM10000 val BA %.2f (%.1f min)" % (epoch + 1, ba, (time.time() - t0) / 60), flush=True)
        if best is None or ba > best[0]:
            best = (ba, epoch + 1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[2])
    path = args.checkpoint
    torch.save({"state_dict": best[2]}, path)
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    test_ba = evaluate(model, test, device)
    print("ConvNeXt-T on HAM10000: epoch %d, val %.2f, test %.2f" % (best[1], best[0], test_ba), flush=True)
    with open(os.path.splitext(path)[0] + ".json", "w") as f:
        json.dump(dict(history=history, best_epoch=best[1], val_balanced_accuracy=best[0],
                       test_balanced_accuracy=test_ba, sha256=sha, minutes=(time.time() - t0) / 60), f, indent=1)


if __name__ == "__main__":
    main()
