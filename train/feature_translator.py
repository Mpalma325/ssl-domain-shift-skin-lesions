"""ResNet-50 trained to produce dermoscopic features from the clinical photograph (row "ResNet-50 trained to
produce the features" of tab:app_e2_variantes_lfm; thesis runs: --seed 1 to 5).

A ResNet-50 that starts from the HAM10000 weights looks at the clinical photograph and learns to output the HAM10000
features of the paired dermoscopic image; the frozen HAM10000 head reads them. No labels: only the MILK10k pool pairs
(training split without the images listed in --pool-exclude). A linear adapter initialized with the ridge map of LFM
(alpha 1000, fitted on the training pairs) follows the backbone, so epoch 0 is the ridge. 10 % of the pairs are held
out (RandomState(0)) and the epoch with the lowest KL between the head on the real dermoscopic features and on the
translation is kept (patience 6). Loss: feature MSE / pool variance + T^2 KL(head(derm) / T || head(translation) / T),
T = 4. AdamW (backbone 3e-5, adapter 1e-4, wd 1e-4), cosine schedule, batch 16, BatchNorm statistics frozen.

Inputs: the HAM10000-network features of MILK10K and MILK10K_CLINICO from scripts/extract_features.py folder
(validation view, float32). Output: --out npz with the translated features F (float16) of every MILK10k clinical
photograph in the same row order, the head logits Z, and the training history.

  python train/feature_translator.py --seed 1 --out features/translator_s1.npz
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
import torchvision.transforms as T
from PIL import Image

warnings.filterwarnings("ignore", message=r"Detected call of `lr_scheduler\.step\(\)` before `optimizer\.step\(\)`")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import extract_features as X  # noqa: E402

IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


class ResizeImage:
    def __init__(self, size):
        self.size = (int(size), int(size))

    def __call__(self, img):
        return img.resize(self.size)


def val_transform():
    return T.Compose([ResizeImage(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(*IMAGENET)])


def train_transform():
    return T.Compose([ResizeImage(256), T.RandomResizedCrop(224, scale=(0.6, 1.0)), T.RandomHorizontalFlip(),
                      T.RandomVerticalFlip(), T.ToTensor(), T.Normalize(*IMAGENET)])


class Photos(torch.utils.data.Dataset):
    def __init__(self, paths, tf, target=None):
        self.paths, self.tf, self.target = list(paths), tf, target

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with open(self.paths[i], "rb") as f:
            x = self.tf(Image.open(f).convert("RGB"))
        if self.target is None:
            return x, i
        return x, torch.from_numpy(self.target[i]), i


@torch.no_grad()
def extract(net, paths, device, bs, amp):
    net.eval()
    dl = torch.utils.data.DataLoader(Photos(paths, val_transform()), batch_size=bs, shuffle=False, num_workers=0)
    out = np.zeros((len(paths), 2048), np.float32)
    for x, i in dl:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            f = net(x.to(device))
        out[i.numpy()] = f.float().cpu().numpy()
    return out


@torch.no_grad()
def head_kl(head, Fd, Fh, device):
    """Mean KL (T = 1) from softmax(head(real derm)) to softmax(head(translation)) and argmax agreement."""
    head.eval()
    a = head(torch.from_numpy(Fd).to(device)).float()
    b = head(torch.from_numpy(Fh).to(device)).float()
    kl = Fnn.kl_div(Fnn.log_softmax(b, 1), Fnn.log_softmax(a, 1), log_target=True, reduction="none").sum(1)
    return kl.mean().item(), (a.argmax(1) == b.argmax(1)).float().mean().item()


def load_features(path, key):
    z = np.load(path)
    return z["ids"].astype(str), z["split"].astype(str), z["y"], z["F_" + key].astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ham", default=os.path.join(ROOT, "checkpoints", "ham10000_source.pth"))
    ap.add_argument("--data-root", default=os.path.join(ROOT, "data"))
    ap.add_argument("--derm-features", default=os.path.join(ROOT, "features", "ham__milk10k.npz"))
    ap.add_argument("--clinical-features", default=os.path.join(ROOT, "features", "ham__milk10k_clinico.npz"))
    ap.add_argument("--features-key", default="ham", help="network name inside the feature files (F_<key>)")
    ap.add_argument("--pool-exclude", default=os.path.join(ROOT, "data", "splits", "milk10k_pool_excluded.txt"))
    ap.add_argument("--out", required=True, help="output npz")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--lr-adapter", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--w-kl", type=float, default=1.0)
    ap.add_argument("--t-kl", type=float, default=4.0)
    ap.add_argument("--alpha", type=float, default=1000.0, help="ridge alpha of the adapter initialization")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", action="store_true", help="64 training and 16 held-out pairs, 2 epochs, for tests")
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit("%s exists; not overwritten" % a.out)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = X.get_device(a.device)
    amp = device.type == "cuda"
    t0 = time.time()

    ids, split, y, Fd = load_features(a.derm_features, a.features_key)
    ids_c, _, _, Fc = load_features(a.clinical_features, a.features_key)
    if not (ids_c == ids).all():
        sys.exit("dermoscopic and clinical feature rows differ")
    clin = [os.path.join(a.data_root, "MILK10K_CLINICO", *i.split("/")) for i in ids]
    derm = [os.path.join(a.data_root, "MILK10K", *i.split("/")) for i in ids]
    pool = X.milk_pool(ids, split, a.pool_exclude)
    held = np.sort(np.random.RandomState(0).choice(pool, len(pool) // 10, replace=False))
    train = np.setdiff1d(pool, held)
    if a.limit:
        train, held, a.epochs = train[:64], held[:16], 2
    print("pairs: pool %d, train %d, held out %d | device %s" % (len(pool), len(train), len(held), device), flush=True)

    net = X.load_network(a.ham)
    bb, head = net.backbone.to(device), nn.Sequential(net.bottleneck, net.head).to(device).eval()
    for p in head.parameters():
        p.requires_grad_(False)
    sample = pool[:16]  # the code must reproduce the feature files
    ed = np.abs(extract(bb, [derm[i] for i in sample], device, 16, amp) - Fd[sample]).max() / (np.abs(Fd[sample]).max() + 1e-6)
    ec = np.abs(extract(bb, [clin[i] for i in sample], device, 16, amp) - Fc[sample]).max() / (np.abs(Fc[sample]).max() + 1e-6)
    print("check against the feature files: max relative error derm %.4f, photo %.4f" % (ed, ec), flush=True)
    if ed > 0.02 or ec > 0.02:
        sys.exit("features not reproduced: check checkpoint, data or feature files")
    var_ref = float(Fd[pool].var(0).mean())

    Xa, Ya = Fc[train].astype(np.float64), Fd[train].astype(np.float64)
    mx, my = Xa.mean(0), Ya.mean(0)
    W = np.linalg.solve((Xa - mx).T @ (Xa - mx) + a.alpha * np.eye(2048), (Xa - mx).T @ (Ya - my))
    adapter = nn.Linear(2048, 2048)
    with torch.no_grad():
        adapter.weight.copy_(torch.from_numpy(W.T.astype(np.float32)))
        adapter.bias.copy_(torch.from_numpy((my - mx @ W).astype(np.float32)))
    model = nn.Sequential(bb, adapter).to(device)

    def evaluate():
        fh = extract(model, [clin[i] for i in held], device, a.bs, amp)
        ref = Fd[held]
        r2 = 1 - ((fh - ref) ** 2).sum() / ((ref - ref.mean(0)) ** 2).sum()
        kl, agree = head_kl(head, ref, fh, device)
        return kl, agree, float(r2)
    kl, agree, r2 = evaluate()
    history = [dict(epoch=0, mse_train=None, kl_train=None, kl_held=kl, agree_held=agree, r2_held=r2)]
    print("epoch  0 (ridge) | held out KL %.4f agreement %.3f R2 %+.3f" % (kl, agree, r2), flush=True)
    best, stale = history[0], 0
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    dl = torch.utils.data.DataLoader(Photos([clin[i] for i in train], train_transform(), target=Fd[train]),
                                     batch_size=a.bs, shuffle=True, drop_last=True, num_workers=0)
    opt = torch.optim.AdamW([dict(params=bb.parameters(), lr=a.lr), dict(params=adapter.parameters(), lr=a.lr_adapter)],
                            weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, a.epochs * len(dl)))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    for ep in range(a.epochs):
        model.train()
        for m in model.modules():  # BatchNorm statistics stay those of HAM10000; affine weights are trained
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        te, n, sm, sk = time.time(), 0, 0.0, 0.0
        for x, f, _ in dl:
            x, f = x.to(device), f.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                s = model(x)
            s = s.float()
            mse = ((s - f) ** 2).mean(1).mean() / var_ref
            zt, zs = head(f) / a.t_kl, head(s) / a.t_kl
            klb = Fnn.kl_div(Fnn.log_softmax(zs, 1), Fnn.log_softmax(zt, 1), log_target=True, reduction="none").sum(1).mean()
            loss = mse + a.w_kl * a.t_kl ** 2 * klb
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            sm, sk, n = sm + mse.item() * len(x), sk + klb.item() * len(x), n + len(x)
        kl, agree, r2 = evaluate()
        h = dict(epoch=ep + 1, mse_train=sm / n, kl_train=sk / n, kl_held=kl, agree_held=agree, r2_held=r2,
                 seconds=time.time() - te)
        history.append(h)
        print("epoch %2d | train mse %.3f kl %.4f | held out KL %.4f agreement %.3f R2 %+.3f | %.0f s" % (
            ep + 1, h["mse_train"], h["kl_train"], kl, agree, r2, h["seconds"]), flush=True)
        if kl < best["kl_held"] - 1e-6:
            best, stale = h, 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= a.patience:
                break
    model.load_state_dict(state)
    print("chosen epoch %d (held out KL %.4f)" % (best["epoch"], best["kl_held"]), flush=True)

    rows = np.arange(len(ids)) if not a.limit else np.concatenate([pool[:16], np.flatnonzero(split == "test")[:16]])
    Fh = extract(model, [clin[i] for i in rows], device, a.bs, amp)
    with torch.no_grad():
        Zh = head(torch.from_numpy(Fh).to(device)).float().cpu().numpy()
    F = np.zeros((len(ids), 2048), np.float16)
    Z = np.zeros((len(ids), 5), np.float32)
    F[rows], Z[rows] = Fh.astype(np.float16), Zh
    meta = dict(args=vars(a), history=history, chosen=best, minutes=(time.time() - t0) / 60)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, ids=ids, split=split, y=y, F=F, Z=Z, rows=rows, meta=json.dumps(meta))
    print("saved %s (%.1f min)" % (a.out, meta["minutes"]), flush=True)


if __name__ == "__main__":
    main()
