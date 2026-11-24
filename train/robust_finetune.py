"""Fine-tunes the HAM10000 classifier to read dermoscopic images degraded by the camera simulator (Chapter 5, robust
network: tab:e2_robusta_clinica, tab:e2_robusta_derm, tab:app_e2_robusta_v1, row of tab:app_e2_variantes_lfm).

Uses only HAM10000 labels. Each step takes --batch training images twice: one clean augmented view and, with
--mode degraded, the same image degraded with random intensities between 0 and --slack times the calibrated ones
(--mode clean: a second clean augmented view, the control). Loss: cross-entropy on both halves (unweighted, so the
head keeps the HAM10000 prior) + KL(clean || second half). AdamW (backbone 2e-5, bottleneck and head 2e-4), one-cycle
schedule, mixed precision. The epoch is chosen by the mean of clean and degraded balanced accuracy on HAM10000 val.

Thesis runs: robust v1  --simulator 1 --mode degraded --epochs 6
             robust v2  --simulator 2 --mode degraded --epochs 2
             control    --simulator 2 --mode clean    --epochs 2  (the simulator only enters the epoch choice)
Outputs in --out: best.pth, head.npz (bottleneck and head as arrays) and summary.json (val/test balanced accuracy,
clean and degraded, before and after). Features of the result: scripts/extract_features.py and scripts/midas_inference.py.

  python train/robust_finetune.py --simulator 2 --mode degraded --epochs 2 --out outputs/robust_v2
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as Fn
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import extract_features as X  # noqa: E402
from camera_simulator import from_calibration, prepare  # noqa: E402


def list_split(root, split):
    out = []
    for k, c in enumerate(X.CLASSES):
        d = os.path.join(root, split, c)
        if os.path.isdir(d):
            out += [(os.path.join(d, f), k) for f in sorted(os.listdir(d)) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    return out


def balanced_accuracy(y, p):
    return 100 * float(np.mean([(p[y == k] == k).mean() for k in range(5) if (y == k).any()]))


def evaluate(model, items, device, sim, cap, batch=48):
    """Clean (base view) and degraded balanced accuracy; image j is degraded with seed 777 + j in every call."""
    model.eval()
    y = np.array([k for _, k in items])
    res = {}
    for name in ("clean", "degraded"):
        pred = []
        with torch.no_grad():
            for i in range(0, len(items), batch):
                ims = []
                for j, (path, _) in enumerate(items[i:i + batch]):
                    im = X.open_rgb(path)
                    if name == "clean":
                        ims.append(X.base_view(im))
                    else:
                        g = np.random.default_rng(777 + i + j)
                        ims.append(sim.degrade(prepare(im), sim.random_intensities(cap, g), g))
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    pred.append(model(X.to_tensor(ims, device))[0].float().argmax(1).cpu().numpy())
        res[name] = balanced_accuracy(y, np.concatenate(pred))
    return res


def augment_clean(im, g):
    im = im.resize((256, 256))
    x0, y0 = g.integers(0, 33, 2)
    im = im.crop((x0, y0, x0 + 224, y0 + 224))
    if g.random() < 0.5:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    if g.random() < 0.5:
        im = im.transpose(Image.FLIP_TOP_BOTTOM)
    return im


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ham", default=os.path.join(ROOT, "checkpoints", "ham10000_source.pth"),
                    help="HAM10000 classifier to start from")
    ap.add_argument("--ham-root", default=os.path.join(ROOT, "data", "HAM10000"))
    ap.add_argument("--simulator", type=int, choices=[1, 2], default=2)
    ap.add_argument("--calibration", default=None, help="calibration.json of camera_simulator.py (default: thesis values)")
    ap.add_argument("--mode", choices=["degraded", "clean"], default="degraded")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=24, help="clean images per step (plus as many in the second half)")
    ap.add_argument("--slack", type=float, default=1.3, help="training intensities up to slack x calibrated")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="only N train and N/2 val/test images, for tests")
    a = ap.parse_args()
    t0 = time.time()
    device = X.get_device(a.device)
    if os.path.exists(os.path.join(a.out, "best.pth")):
        sys.exit("%s already holds a checkpoint" % a.out)
    os.makedirs(a.out, exist_ok=True)
    sim, t_cal = from_calibration(a.simulator, a.calibration)
    cap = {k: min(1.0, v) for k, v in t_cal.items()}
    cap_train = {k: min(1.0, a.slack * v) for k, v in t_cal.items()}
    tr, va, te = (list_split(a.ham_root, s) for s in ("train", "val", "test"))
    if not tr or not va or not te:
        sys.exit("HAM10000 not found in %s/{train,val,test}/<class>" % a.ham_root)
    if a.limit:
        rs = np.random.RandomState(0)
        tr, va, te = ([x[i] for i in rs.choice(len(x), min(len(x), m), replace=False)]
                      for x, m in ((tr, a.limit), (va, a.limit // 2), (te, a.limit // 2)))
    print("HAM10000: train %d, val %d, test %d | %s, simulator v%d | device %s" % (
        len(tr), len(va), len(te), a.mode, a.simulator, device), flush=True)
    original = X.load_network(a.ham).to(device)
    ceiling = {"original": {"val": evaluate(original, va, device, sim, cap), "test": evaluate(original, te, device, sim, cap)}}
    print("original network: %s" % ceiling["original"], flush=True)
    del original

    model = X.load_network(a.ham).to(device)
    head_params = list(model.bottleneck.parameters()) + list(model.head.parameters())
    opt = torch.optim.AdamW([{"params": list(model.backbone.parameters()), "lr": 2e-5},
                             {"params": head_params, "lr": 2e-4}], weight_decay=1e-4)
    steps = int(np.ceil(len(tr) / a.batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[2e-5, 2e-4], total_steps=a.epochs * steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    g = np.random.default_rng(2026)
    best, history = None, []
    for ep in range(a.epochs):
        model.train()
        order = g.permutation(len(tr))
        losses = []
        for p in range(steps):
            idx = order[p * a.batch:(p + 1) * a.batch]
            ims, ys = [], []
            for i in idx:
                ims.append(augment_clean(X.open_rgb(tr[i][0]), g))
                ys.append(tr[i][1])
            for i in idx:
                im = X.open_rgb(tr[i][0])
                if a.mode == "clean":
                    ims.append(augment_clean(im, g))
                else:
                    d = sim.degrade(prepare(im), sim.random_intensities(cap_train, g), g)
                    if g.random() < 0.5:
                        d = d.transpose(Image.FLIP_LEFT_RIGHT)
                    ims.append(d)
            x = X.to_tensor(ims, device)
            y = torch.tensor(ys, device=device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                z = model(x)[0].float()
                z1, z2 = z[:len(idx)], z[len(idx):]
                loss = Fn.cross_entropy(z1, y) + Fn.cross_entropy(z2, y) + \
                    Fn.kl_div(Fn.log_softmax(z2, 1), Fn.softmax(z1.detach(), 1), reduction="batchmean")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(float(loss))
            if (p + 1) % 50 == 0:
                print("    epoch %d step %d/%d loss %.3f (%.1f min)" % (ep + 1, p + 1, steps, np.mean(losses[-50:]),
                                                                    (time.time() - t0) / 60), flush=True)
        ev = evaluate(model, va, device, sim, cap)
        score = (ev["clean"] + ev["degraded"]) / 2
        history.append(dict(epoch=ep + 1, loss=float(np.mean(losses)), **ev))
        print("  epoch %d: HAM10000 val clean %.2f, degraded %.2f (%.1f min)" % (ep + 1, ev["clean"], ev["degraded"],
                                                                             (time.time() - t0) / 60), flush=True)
        if best is None or score > best[0]:
            best = (score, ep + 1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[2])
    path = os.path.join(a.out, "best.pth")
    torch.save({"state_dict": best[2]}, path)
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    np.savez(os.path.join(a.out, "head.npz"), **X.head_arrays(best[2]))
    ceiling["finetuned"] = {"val": evaluate(model, va, device, sim, cap), "test": evaluate(model, te, device, sim, cap)}
    print("fine-tuned network (epoch %d): %s" % (best[1], ceiling["finetuned"]), flush=True)
    json.dump(dict(args=vars(a), ceiling=ceiling, history=history, best_epoch=best[1], sha256=sha, cap=cap,
                   minutes=(time.time() - t0) / 60), open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print("saved in %s (%.1f min)" % (a.out, (time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()
