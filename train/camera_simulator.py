"""Camera simulator: degrades a dermoscopic image with the physics of a clinical photograph (Chapter 5, robust
network; tab:e2_robusta_clinica, tab:e2_robusta_derm, tab:app_e2_robusta_v1).

A dermoscope removes surface glare, sees below the stratum corneum, brings its own uniform light and looks from
contact distance. The simulator removes that information step by step, each factor with an intensity t in [0, 1]
(t = 0 does nothing), in physical order: tissue (scatter) -> surface (veil, specular) -> light and camera (light)
-> sensor (resolution, noise). Version 1 crops in for the field of view; version 2 zooms out over skin of the
lesion's own border colour and adds a colour shift towards the mean colour of the clinical photographs.

Run as a script it calibrates the intensities without labels: 600 MILK10k pool pairs, dermoscopic image degraded
versus the x2.5 zoom of the real clinical photograph, compared in the HAM10000 network (1 - cosine of the pooled
features); one factor at a time, 40 Latin-hypercube combinations, then +-0.15 around the best. CALIBRATED holds the
values found for the thesis.

  python train/camera_simulator.py --version 2 --ham checkpoints/ham10000_source.pth --out outputs/camera_simulator_v2
"""
import argparse
import csv
import io
import json
import os
import sys
import time

import numpy as np
from PIL import Image
from scipy import ndimage

SIDE = 224
FACTORS = {1: ["field", "scatter", "veil", "specular", "light", "resolution", "noise"],
           2: ["field", "color", "scatter", "veil", "specular", "light", "resolution", "noise"]}
# Calibration of the thesis (simulador_calibrado.json of each version).
CALIBRATED = {
    1: dict(t={"field": 0.0, "scatter": 0.0, "veil": 0.0, "specular": 1.0, "light": 0.0, "resolution": 0.15,
               "noise": 0.0}),
    2: dict(t={"field": 0.0, "color": 0.74, "scatter": 0.048, "veil": 0.006, "specular": 0.439, "light": 0.4,
               "resolution": 0.269, "noise": 0.832},
            color={"md": [0.6764313578605652, 0.546265184879303, 0.538263201713562],
                   "sd": [0.12722985446453094, 0.1323661059141159, 0.14603359997272491],
                   "mc": [0.6935784220695496, 0.4938623011112213, 0.40459102392196655],
                   "sc": [0.12441429495811462, 0.1352568119764328, 0.13546444475650787]}),
}


def _smooth(rng, shape, sigma):
    """Smooth random field (scale ~sigma px), mean 0 and sd 1."""
    k = max(2, int(round(shape[0] / sigma)))
    z = ndimage.zoom(rng.standard_normal((k + 2, k + 2)), shape[0] / k, order=3)[:shape[0], :shape[1]]
    return (z - z.mean()) / (z.std() + 1e-8)


def remove_vignette(img):
    """Crops the dark circular lens vignette, if any, to the largest central rectangle without it."""
    a = np.asarray(img.convert("L").resize((96, 72)), np.float32)
    dark = ndimage.binary_opening(a < 25, iterations=1)
    lab, n = ndimage.label(dark)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    if not border or np.isin(lab, list(border)).mean() < 0.03:
        return img
    bad = np.isin(lab, list(border))
    w, h = img.size
    for f in np.linspace(0.98, 0.5, 25):
        cw, ch = int(96 * f), int(72 * f)
        x0, y0 = (96 - cw) // 2, (72 - ch) // 2
        if not bad[y0:y0 + ch, x0:x0 + cw].any():
            return img.crop((int(w * x0 / 96), int(h * y0 / 72), int(w * (x0 + cw) / 96), int(h * (y0 + ch) / 72)))
    return img.crop((int(w * 0.25), int(h * 0.25), int(w * 0.75), int(h * 0.75)))


def prepare(img, side=448):
    """Dermoscopic image without vignette, squared (aspect ratio not kept) at 448 px."""
    return remove_vignette(img.convert("RGB")).resize((side, side), Image.BICUBIC)


class CameraSimulator:
    def __init__(self, version, color=None):
        assert version in FACTORS
        self.version, self.factors = version, FACTORS[version]
        self.color = None if color is None else {k: np.asarray(v, np.float32) for k, v in color.items()}

    def random_intensities(self, cap, rng):
        """Each factor uniform between 0 and its cap."""
        return {k: float(rng.uniform(0, cap.get(k, 0.0))) for k in self.factors}

    def degrade(self, img448, t, rng):
        """img448: output of prepare(); t: factor -> intensity; rng: np.random.Generator. Returns a 224 x 224 image."""
        g = {k: float(np.clip(t.get(k, 0.0), 0.0, 1.0)) for k in self.factors}
        w = img448.size[0]
        c = w * (0.875 * (1 - 0.4 * g["field"]) if self.version == 1 else 0.875)
        x0 = (w - c) / 2
        im = img448.crop((int(round(x0)), int(round(x0)), int(round(x0 + c)), int(round(x0 + c)))).resize(
            (SIDE, SIDE), Image.BICUBIC)
        x = np.asarray(im).astype(np.float32) / 255.0
        h = SIDE
        if self.version == 2 and g["field"] > 0:  # zoom out over skin of the border colour, soft edge
            m = max(32, int(round(h * (1 - 0.6 * g["field"]))))
            small = np.asarray(Image.fromarray((x * 255 + 0.5).astype(np.uint8)).resize((m, m), Image.BICUBIC)
                               ).astype(np.float32) / 255
            p0 = (h - m) // 2
            ring = np.concatenate([x[:h // 10].reshape(-1, 3), x[-h // 10:].reshape(-1, 3),
                                   x[:, :h // 10].reshape(-1, 3), x[:, -h // 10:].reshape(-1, 3)])
            skin = np.median(ring, 0)
            bg = np.clip(skin[None, None, :] * (1 + 0.04 * _smooth(rng, (h, h), h / 4)[..., None]), 0, 1)
            alpha = np.zeros((h, h), np.float32)
            alpha[p0:p0 + m, p0:p0 + m] = 1
            alpha = ndimage.gaussian_filter(alpha, max(2.0, m / 12))[..., None]
            canvas = bg.copy()
            canvas[p0:p0 + m, p0:p0 + m] = small
            x = alpha * canvas + (1 - alpha) * bg
        if self.version == 2 and g["color"] > 0 and self.color is not None:  # towards the photographs' colour
            s_ = g["color"]
            mu, sd = x.reshape(-1, 3).mean(0), x.reshape(-1, 3).std(0) + 1e-4
            C = self.color
            tm = mu + s_ * (C["mc"] - C["md"]) + rng.normal(0, 0.02 * s_, 3).astype(np.float32)
            ts = sd * (1 + s_ * (C["sc"] / (C["sd"] + 1e-4) - 1))
            x = np.clip((x - mu) / sd * ts + tm, 0, 1)
        if g["scatter"] > 0:
            a = 0.6 * g["scatter"]
            s = 4 * (1 + g["scatter"])
            x = (1 - a) * x + a * ndimage.gaussian_filter(x, sigma=(s, s, 0))
        illum = np.ones(3, np.float32)
        if g["light"] > 0:
            tone = rng.normal(0, 1)
            illum = np.array([1 + 0.06 * tone, 1.0, 1 - 0.06 * tone], np.float32)
        if g["veil"] > 0:
            v = 0.30 * g["veil"]
            m = np.clip(1 + 0.35 * _smooth(rng, (h, h), h / 2), 0.3, 1.8)[..., None]
            x = x * (1 - v * m) + v * m * 0.92 * illum
        if g["specular"] > 0:
            p = 0.02 * g["specular"]
            n = ndimage.gaussian_filter(rng.standard_normal((h // 2, h // 2)), 1.0)
            n = ndimage.zoom(n, 2, order=1) + 0.5 * _smooth(rng, (h, h), h / 4)
            u = np.quantile(n, 1 - p)
            m = ndimage.gaussian_filter(1 / (1 + np.exp(-(n - u) / (0.1 * n.std() + 1e-8))), 1.0)
            x = x + (0.95 * illum - x) * (0.85 * m[..., None])
        if g["light"] > 0:  # white balance, exposure, gamma, saturation and gradient
            s = g["light"]
            gain = np.exp(rng.normal(0, 0.12 * s, 3)).astype(np.float32)
            gain /= gain.mean()
            x = x * gain * np.exp(rng.normal(0, 0.18 * s))
            x = np.clip(x, 0, None) ** np.exp(rng.normal(0, 0.12 * s))
            lum = (x * np.array([0.299, 0.587, 0.114], np.float32)).sum(-1, keepdims=True)
            x = lum + (x - lum) * (1 - 0.35 * s * rng.uniform(0.3, 1.0))
            phi = rng.uniform(0, 2 * np.pi)
            yy, xx = np.mgrid[0:h, 0:h] / (h - 1) - 0.5
            x = x * (1 + 0.35 * s * rng.uniform(0.3, 1.0) * (xx * np.cos(phi) + yy * np.sin(phi)))[..., None]
        x = np.clip(x, 0, 1)
        if g["resolution"] > 0 or g["noise"] > 0:
            im = Image.fromarray((x * 255 + 0.5).astype(np.uint8))
            if g["resolution"] > 0:
                r = max(16, int(round(h / (1 + 3 * g["resolution"]))))
                im = im.resize((r, r), Image.BOX).resize((h, h), Image.BICUBIC)
            x = np.asarray(im).astype(np.float32) / 255.0
            if g["noise"] > 0:
                x = np.clip(x + rng.normal(0, 4 / 255 * g["noise"], x.shape), 0, 1)
            im = Image.fromarray((x * 255 + 0.5).astype(np.uint8))
            if g["resolution"] > 0:
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=int(round(95 - 45 * g["resolution"])))
                im = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
            return im
        return Image.fromarray((x * 255 + 0.5).astype(np.uint8))


def from_calibration(version, calibration=None):
    """(simulator, calibrated intensities) from a calibration json or the thesis values."""
    cal = CALIBRATED[version] if calibration is None else json.load(open(calibration, encoding="utf-8"))
    if calibration is not None and cal.get("version", 1) != version:
        sys.exit("%s is not a version %d calibration" % (calibration, version))
    return CameraSimulator(version, cal.get("color")), dict(cal["t"])


# ----------------------------------------------------------------------------------------------- calibration
def calibrate(a):
    import torch
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    import extract_features as X
    t0 = time.time()
    device = X.get_device(a.device)
    amp = device.type == "cuda"
    os.makedirs(a.out, exist_ok=True)
    if os.path.exists(os.path.join(a.out, "calibration.json")):
        sys.exit("%s already holds a calibration" % a.out)
    ids, paths, _, split = X.list_folder(os.path.join(a.data_root, "MILK10K_CLINICO"))
    pool = X.milk_pool(ids, split, a.pool_exclude)
    held = np.sort(np.random.RandomState(0).choice(pool, len(pool) // 10, replace=False))
    inside = np.setdiff1d(pool, held)
    sel = np.sort(np.random.RandomState(22).choice(inside, a.n, replace=False))
    print("pairs: %d (of %d; %d held out are not used) | device %s" % (a.n, len(inside), len(held), device), flush=True)
    derm = [prepare(X.open_rgb(os.path.join(a.data_root, "MILK10K", *ids[i].split("/")))) for i in sel]
    photos = [X.zoom(X.open_rgb(paths[i]), 2.5) for i in sel]
    sim = CameraSimulator(a.version)
    color = None
    if a.version == 2:  # mean and sd per channel of the dermoscopic base view and of the photographs, no labels
        def moments(ims):
            px = np.concatenate([np.asarray(im, np.float32).reshape(-1, 3)[::7] / 255 for im in ims])
            return px.mean(0).tolist(), px.std(0).tolist()
        md, sd = moments([sim.degrade(d, {}, np.random.default_rng(0)) for d in derm])
        mc, sc = moments(photos)
        color = dict(md=md, sd=sd, mc=mc, sc=sc)
        sim = CameraSimulator(2, color)
    net = X.load_network(a.ham).to(device)
    H = X.head_arrays(X.state_dict(a.ham))
    log_pi = np.log(X.HAM_TRAIN_COUNTS / X.HAM_TRAIN_COUNTS.sum())
    head = lambda F: X.log_softmax(X.apply_head(F, H) - log_pi)
    Fc = X.forward(net, photos, device, amp, 32)[1].astype(np.float32)
    nc = Fc / np.linalg.norm(Fc, axis=1, keepdims=True)
    lc = head(Fc.astype(np.float64))

    def measure(t):
        imgs = [sim.degrade(d, t, np.random.default_rng(1000 + k)) for k, d in enumerate(derm)]
        F = X.forward(net, imgs, device, amp, 32)[1].astype(np.float32)
        ns = F / np.linalg.norm(F, axis=1, keepdims=True)
        ls = head(F.astype(np.float64))
        return dict(dcos=float(1 - (ns * nc).sum(1).mean()), kl=float((np.exp(lc) * (lc - ls)).sum(1).mean()),
                    agree=100 * float((ls.argmax(1) == lc.argmax(1)).mean()))

    rows = []

    def trial(stage, t):
        m = measure(t)
        rows.append(dict(stage=stage, **{k: round(t.get(k, 0.0), 3) for k in sim.factors}, **m))
        return m

    base = trial("none", {})
    for f in sim.factors:
        for t in a.single_levels:
            m = trial("single", {f: t})
        print("  %-10s t=%.2f: dcos %.4f (%.1f min)" % (f, t, m["dcos"], (time.time() - t0) / 60), flush=True)
    rs = np.random.RandomState(5)
    lhs = np.stack([(rs.permutation(a.random) + rs.uniform(0, 1, a.random)) / a.random for _ in sim.factors], 1)
    for j in range(a.random):
        trial("random", dict(zip(sim.factors, lhs[j])))
    best = min([r for r in rows if r["stage"] != "none"], key=lambda r: r["dcos"])
    tb = {f: best[f] for f in sim.factors}
    for f in sim.factors:
        for d in (-0.15, 0.15):
            t2 = dict(tb)
            t2[f] = float(np.clip(t2[f] + d, 0, 1))
            if t2[f] != tb[f]:
                trial("refine", t2)
    best = min([r for r in rows if r["stage"] != "none"], key=lambda r: r["dcos"])
    tb = {k: float(best[k]) for k in sim.factors}
    with open(os.path.join(a.out, "calibration.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    out = dict(version=a.version, t=tb, dcos=best["dcos"], kl=best["kl"], agree=best["agree"], none=base, n=a.n)
    if color is not None:
        out["color"] = color
    json.dump(out, open(os.path.join(a.out, "calibration.json"), "w"), indent=1)
    print("best: %s | dcos %.4f (undegraded %.4f) | %.1f min" % (
        ", ".join("%s %.2f" % kv for kv in tb.items()), best["dcos"], base["dcos"], (time.time() - t0) / 60), flush=True)


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--version", type=int, choices=[1, 2], default=2)
    ap.add_argument("--ham", default=os.path.join(root, "checkpoints", "ham10000_source.pth"))
    ap.add_argument("--data-root", default=os.path.join(root, "data"))
    ap.add_argument("--pool-exclude", default=os.path.join(root, "data", "splits", "milk10k_pool_excluded.txt"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=600, help="pairs")
    ap.add_argument("--random", type=int, default=40, help="Latin-hypercube combinations")
    ap.add_argument("--single-levels", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--device", default="cuda")
    calibrate(ap.parse_args())


if __name__ == "__main__":
    main()
