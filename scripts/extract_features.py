"""Frozen features of the HAM10000 networks, the input of LFM and its variants (Chapter 5, Appendix 7.4).

Used by tab:e2_robusta_clinica, tab:e2_robusta_derm, tab:app_e2_variantes_lfm, tab:app_e2_robusta_v1,
fig:e2_midas_curva and every LFM row.

  folder  every image of <data-root>/<FOLDER>/{train,val,test} in ImageFolder order, validation view (resize to
          256 x 256, centre crop 224): pooled features (input of the bottleneck) and logits of each network.
          Float32 by default; --amp gives the mixed-precision pass used for the BCN20000 features of the robust
          networks and their clean control.
  views   MILK10k: views of each clinical photograph (base; centred zoom x2, x2.5, x3; zoom x2.5 after removing
          marker ink and hair) and the base view of its dermoscopic pair ("derm"). Rows follow the ImageFolder order
          of MILK10K_CLINICO (train, val, test). Mixed precision on GPU.

--net NAME=PATH (repeatable): a tllib ImageClassifier checkpoint (ResNet-50 or ConvNeXt-T backbone: the HAM10000
classifier, the robust networks, ERM runs) or the HAM10000 ConvNeXt-T (features.*, norm.*, head.*).
Output: one npz (features float16, logits float32). The other Study 2 scripts import the view and loading helpers.

  python scripts/extract_features.py folder MILK10K --net ham=checkpoints/ham10000_source.pth
  python scripts/extract_features.py views --net ham=checkpoints/ham10000_source.pth --views base z2 z25 z3 z25_TP
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES = ["akiec", "bcc", "bkl", "mel", "nv"]
SPLITS = ["train", "val", "test"]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
HAM_TRAIN_COUNTS = np.array([210, 231, 612, 447, 4084], float)  # class counts of the HAM10000 training split
CLINICAL_VIEWS = ["base", "z2", "z25", "z3", "z25_TP"]
ZOOM = {"z2": 2.0, "z25": 2.5, "z3": 3.0}


# ------------------------------------------------------------------------------------------------------- views
def open_rgb(path):
    with open(path, "rb") as f:
        return Image.open(f).convert("RGB")


def base_view(img):
    """tllib ResizeImage(256) (no aspect ratio kept) + CenterCrop(224)."""
    return img.resize((256, 256)).crop((16, 16, 240, 240))


def window(img, cw, ch, cx, cy):
    w, h = img.size
    cw, ch = min(cw, w), min(ch, h)
    x0 = min(max(cx - cw / 2, 0), w - cw)
    y0 = min(max(cy - ch / 2, 0), h - ch)
    return (x0, y0, x0 + cw, y0 + ch)


def crop224(img, box):
    return img.crop(tuple(int(round(v)) for v in box)).resize((224, 224))


def zoom(img, z):
    w, h = img.size
    return crop224(img, window(img, w / z, h / z, w / 2, h / 2))


def ink_mask(a):
    """Marker ink: strongly saturated violet or blue."""
    hsv = np.asarray(Image.fromarray(a).convert("HSV")).astype(np.int16)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = (h >= 140) & (h <= 215) & (s >= 90) & (v >= 40) & (v <= 235) & (a[..., 2].astype(np.int16) > a[..., 0] + 10)
    m = ndimage.binary_opening(m, iterations=1)
    lab, n = ndimage.label(m)
    if n:
        size = ndimage.sum(m, lab, index=np.arange(1, n + 1))
        m = np.isin(lab, 1 + np.flatnonzero(size >= 25))
    return ndimage.binary_dilation(m, iterations=3)


def hair_mask(a):
    """Hair: black-hat on luminance, keeping thin elongated blobs only."""
    g = a.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    m = (ndimage.grey_closing(g, size=(11, 11)) - g) > 30
    lab, n = ndimage.label(m)
    if not n:
        return m
    area = ndimage.sum(m, lab, index=np.arange(1, n + 1))
    ok = np.zeros(n + 1, bool)
    for k, sl in enumerate(ndimage.find_objects(lab)):
        L = max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start)
        ok[k + 1] = area[k] >= 40 and L >= 40 and area[k] / float(L * L) < 0.08
    return ndimage.binary_dilation(ok[lab], iterations=1)


def fill(a, m):
    """Replaces masked pixels with the normalized Gaussian average of their unmasked surroundings."""
    if not m.any():
        return a
    s = max(2.0, 0.02 * max(a.shape[:2]))
    ok = ndimage.gaussian_filter((~m).astype(np.float32), s)
    out = a.astype(np.float32).copy()
    for c in range(3):
        rel = ndimage.gaussian_filter(out[..., c] * (~m), s) / np.maximum(ok, 1e-3)
        out[..., c][m] = rel[m]
    return np.clip(out, 0, 255).astype(np.uint8)


def clinical_views(img, names=CLINICAL_VIEWS):
    """{view: 224 x 224 PIL image} for one clinical photograph; also the ink and hair fractions."""
    v = {}
    for name in names:
        if name == "base":
            v[name] = base_view(img)
        elif name in ZOOM:
            v[name] = zoom(img, ZOOM[name])
    ink = hair = np.nan
    if "z25_TP" in names:
        a = np.asarray(img)
        mi, mh = ink_mask(a), hair_mask(a)
        v["z25_TP"] = zoom(Image.fromarray(fill(a, mi | mh)), 2.5)
        ink, hair = float(mi.mean()), float(mh.mean())
    return v, ink, hair


def to_tensor(imgs, device):
    u8 = torch.from_numpy(np.stack([np.asarray(im, dtype=np.uint8) for im in imgs])).to(device)
    x = u8.permute(0, 3, 1, 2).float().div_(255)
    return (x - MEAN.to(device)) / STD.to(device)


# ---------------------------------------------------------------------------------------------------- networks
class ResNetClassifier(nn.Module):
    """tllib ImageClassifier with a ResNet-50: pooled 2048-d features, bottleneck 1024 (BN, ReLU, dropout), head."""

    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.resnet50(weights=None)
        self.backbone.fc = nn.Identity()
        self.bottleneck = nn.Sequential(nn.Linear(2048, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.5))
        self.head = nn.Linear(1024, 5)

    def forward(self, x):
        f = self.backbone(x)
        return self.head(self.bottleneck(f)), f


class ConvNeXtClassifier(nn.Module):
    """tllib ImageClassifier with -a convnext_tiny_ham: 768-d features, bottleneck 1024, head."""

    def __init__(self):
        super().__init__()
        m = torchvision.models.convnext_tiny(weights=None)
        self.backbone = nn.Module()
        self.backbone.features, self.backbone.norm = m.features, m.classifier[0]
        self.avg = m.avgpool
        self.bottleneck = nn.Sequential(nn.Linear(768, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.5))
        self.head = nn.Linear(1024, 5)

    def forward(self, x):
        f = self.backbone.norm(self.avg(self.backbone.features(x))).flatten(1)
        return self.head(self.bottleneck(f)), f


class ConvNeXtHAM(nn.Module):
    """ConvNeXt-T trained on HAM10000 (ImageNet-1k init.), linear head on the 768-d features."""

    def __init__(self, imagenet=False):
        super().__init__()
        w = torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if imagenet else None
        m = torchvision.models.convnext_tiny(weights=w)
        self.features, self.avgpool = m.features, m.avgpool
        self.norm = m.classifier[0]
        self.head = nn.Linear(768, 5)

    def forward(self, x):
        f = torch.flatten(self.norm(self.avgpool(self.features(x))), 1)
        return self.head(f), f


def state_dict(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck


def load_network(path):
    """Module whose forward returns (logits, features), architecture read from the checkpoint keys."""
    sd = state_dict(path)
    if "backbone.conv1.weight" in sd:
        m, sd = ResNetClassifier(), {k: v for k, v in sd.items() if not k.startswith("backbone.fc.")}
    elif "backbone.features.0.0.weight" in sd:
        m = ConvNeXtClassifier()
    elif "features.0.0.weight" in sd and "head.weight" in sd:
        m = ConvNeXtHAM()
    else:
        sys.exit("unknown checkpoint layout: %s" % path)
    res = m.load_state_dict(sd, strict=False)
    if res.missing_keys or res.unexpected_keys:
        sys.exit("%s: missing %s, unexpected %s" % (path, res.missing_keys[:5], res.unexpected_keys[:5]))
    return m.eval()


def head_arrays(sd):
    """Bottleneck and head of a ResNet ImageClassifier as float arrays, for the CPU head used by LFM."""
    g = lambda k: sd[k].float().numpy()
    return dict(W1=g("bottleneck.0.weight"), b1=g("bottleneck.0.bias"), rm=g("bottleneck.1.running_mean"),
                rv=g("bottleneck.1.running_var"), g=g("bottleneck.1.weight"), bb=g("bottleneck.1.bias"),
                W2=g("head.weight"), b2=g("head.bias"))


def apply_head(F, H):
    """Eval-mode bottleneck + head in float64."""
    h = np.asarray(F, np.float64) @ H["W1"].T + H["b1"]
    h = (h - H["rm"]) / np.sqrt(H["rv"] + 1e-5) * H["g"] + H["bb"]
    return np.maximum(h, 0) @ H["W2"].T + H["b2"]


def log_softmax(z):
    z = np.asarray(z, np.float64)
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def balanced_log_posterior(z):
    """Log-posterior with the HAM10000 training prior removed."""
    return log_softmax(z - np.log(HAM_TRAIN_COUNTS / HAM_TRAIN_COUNTS.sum()))


@torch.no_grad()
def forward(net, imgs, device, amp, batch=48):
    """(logits float32, features float16) of a list of 224 x 224 images."""
    Z, F = [], []
    for i in range(0, len(imgs), batch):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            z, f = net(to_tensor(imgs[i:i + batch], device))
        Z.append(z.float().cpu().numpy())
        F.append(f.float().cpu().numpy())
    return np.concatenate(Z).astype(np.float32), np.concatenate(F).astype(np.float16)


def parse_nets(specs):
    out = {}
    for s in specs:
        name, _, path = s.partition("=")
        if not path:
            sys.exit("--net expects NAME=PATH, got %r" % s)
        out[name] = path
    return out


def get_device(name):
    return torch.device(name if name == "cpu" or torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------------------------------------- image lists
def list_folder(root):
    """ImageFolder order of root/{train,val,test}: relative ids 'split/class/file', paths, labels, split."""
    from torchvision.datasets import ImageFolder
    ids, paths, y, split = [], [], [], []
    for sp in SPLITS:
        d = ImageFolder(os.path.join(root, sp))
        if list(d.classes) != CLASSES:
            sys.exit("classes of %s/%s are %s" % (root, sp, d.classes))
        for p, k in d.samples:
            ids.append(os.path.relpath(p, root).replace(os.sep, "/"))
            paths.append(p)
            y.append(k)
            split.append(sp)
    return np.array(ids), paths, np.array(y), np.array(split)


def read_id_list(path):
    with open(path, encoding="utf-8-sig") as fh:
        return {line.strip() for line in fh if line.strip() and not line.startswith("#")}


def isic_of(rel_id):
    return os.path.splitext(os.path.basename(rel_id))[0]


def milk_pool(ids, split, pool_exclude):
    """MILK10k unlabeled pairs: training images not listed in pool_exclude (copies of HAM10000 lesions)."""
    excl = read_id_list(pool_exclude)
    itr = np.flatnonzero(split == "train")
    return itr[~np.array([isic_of(i) in excl for i in ids[itr]])]


# ------------------------------------------------------------------------------------------------- subcommands
def cmd_folder(a, device):
    root = os.path.join(a.data_root, a.folder)
    ids, paths, y, split = list_folder(root)
    if a.limit:
        keep = np.concatenate([np.flatnonzero(split == sp)[:a.limit] for sp in SPLITS])
        ids, paths, y, split = ids[keep], [paths[i] for i in keep], y[keep], split[keep]
    nets = {k: load_network(p).to(device) for k, p in parse_nets(a.net).items()}
    out = {k: ([], []) for k in nets}
    t0 = time.time()
    for i in range(0, len(paths), a.batch):
        imgs = [base_view(open_rgb(p)) for p in paths[i:i + a.batch]]
        for k, net in nets.items():
            z, f = forward(net, imgs, device, a.amp, a.batch)
            out[k][0].append(z)
            out[k][1].append(f)
        if (i // a.batch) % 25 == 0:
            print("  %s %d/%d (%.0f s)" % (a.folder, i + len(imgs), len(paths), time.time() - t0), flush=True)
    res = dict(ids=ids, split=split, y=y)
    for k, (Z, F) in out.items():
        res["Z_" + k], res["F_" + k] = np.concatenate(Z), np.concatenate(F)
    dest = a.output or os.path.join(a.out, "%s__%s.npz" % ("_".join(nets), a.folder.lower()))
    return dest, res


def cmd_views(a, device):
    clin_root, derm_root = os.path.join(a.data_root, "MILK10K_CLINICO"), os.path.join(a.data_root, "MILK10K")
    ids, paths, y, split = list_folder(clin_root)
    rows = np.arange(len(ids))[:a.limit] if a.limit else np.arange(len(ids))
    nets = {k: load_network(p).to(device) for k, p in parse_nets(a.net).items()}
    clin = [v for v in a.views if v != "derm"]
    acc = {(k, v): [] for k in nets for v in a.views}
    t0 = time.time()
    for b0 in range(0, len(rows), a.block):
        V = {v: [] for v in a.views}
        for i in rows[b0:b0 + a.block]:
            vs, _, _ = clinical_views(open_rgb(paths[i]), clin)
            for v in clin:
                V[v].append(vs[v])
            if "derm" in a.views:
                V["derm"].append(base_view(open_rgb(os.path.join(derm_root, *ids[i].split("/")))))
        for k, net in nets.items():
            for v in a.views:
                acc[(k, v)].append(forward(net, V[v], device, device.type == "cuda", a.batch)[1])
        done = b0 + a.block
        if done % 500 < a.block or done >= len(rows):
            print("  views %d/%d (%.1f min)" % (min(done, len(rows)), len(rows), (time.time() - t0) / 60), flush=True)
    res = dict(ids=ids[rows], split=split[rows], y=y[rows])
    for (k, v), F in acc.items():
        res["%s__%s" % (k, v)] = np.concatenate(F)
    dest = a.output or os.path.join(a.out, "%s__milk10k_views.npz" % "_".join(nets))
    return dest, res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", default=os.path.join(ROOT, "data"))
    ap.add_argument("--out", default=os.path.join(ROOT, "features"))
    ap.add_argument("--output", help="exact output file (overrides --out)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0, help="only the first N images (per split for 'folder'), for tests")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("folder")
    f.add_argument("folder", help="HAM10000, BCN20000, MILK10K or MILK10K_CLINICO")
    f.add_argument("--net", action="append", required=True, metavar="NAME=PATH")
    f.add_argument("--amp", action="store_true", help="float16 autocast on GPU")
    v = sub.add_parser("views")
    v.add_argument("--net", action="append", required=True, metavar="NAME=PATH")
    v.add_argument("--views", nargs="+", default=CLINICAL_VIEWS + ["derm"],
                   choices=CLINICAL_VIEWS + ["derm"])
    v.add_argument("--block", type=int, default=8, help="photographs read per step")
    a = ap.parse_args()
    device = get_device(a.device)
    dest, res = {"folder": cmd_folder, "views": cmd_views}[a.cmd](a, device)
    if os.path.exists(dest):
        sys.exit("%s exists; not overwritten" % dest)
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    np.savez_compressed(dest, **res)
    print("saved", dest, flush=True)


if __name__ == "__main__":
    main()
