"""Mallya and Hamarneh (2022) guidance model on MILK10k, k = 16 (tab:e2_lfm_mallya, fig:e2_lda_mallya, rows "G" of
tab:app_e2_variantes_lfm).

PyTorch port of their Keras code (DeepGuide, derm7pt): frozen 2048-d latents, a decoder G from the clinical latent to
the dermoscopic latent, and a head on [clinical latent, G(clinical latent)].
  G     Dense 2048-1024-512-512-1024-2048, ReLU everywhere (output too), dropout 0.25 after each hidden layer; Glorot
        uniform init, zero bias; SGD lr 0.5, momentum 0.9, Nesterov, decay 1e-6 per step (lr / (1 + d t)); MSE;
        batch 100; 500 epochs.
  head  dropout 0.25 + Dense 4096-5; SGD lr 1e-4 (same momentum and decay); cross-entropy with 'balanced' class
        weights; batch 100; 500 epochs; early stopping on validation loss, patience 200, best weights restored.

  latents   per seed: pooled latents and logits of the clinical ERM run (E_I) on the 80 labeled photographs, the pool,
            validation and test, and of the dermoscopic ERM run (E_S) on the dermoscopic pair of each (GPU, inference).
  guided    per seed: G trained on the pool + labeled pairs (no labels), head on the 80 labeled photographs; also the
            'photo' (clinical latent only) and 'guide' (G output only) heads. Logits and balanced accuracy (CPU).
  ham-space G in the HAM10000 network space instead of the ridge of LFM: MILK10k pool pairs (photo or its x2.5 zoom ->
            dermoscopic image) with the features of scripts/extract_features.py; output the balanced log-posteriors
            of the HAM10000 head on G(photo) for every image. Thesis: base s0, s1 on CPU, the other eight on GPU
            (same recipe; only the dropout masks change with the device).

  python train/mallya.py latents --seed 0
  python train/mallya.py guided --seed 0
  python train/mallya.py ham-space --view base --seed 0 --device cpu
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
OUT = os.path.join(ROOT, "features", "mallya")
G_EPOCHS, HEAD_EPOCHS = 500, 500


def balanced_accuracy(y, p):
    return 100 * np.mean([np.mean(p[y == c] == c) for c in np.unique(y)])


def keras_init(m):
    for c in m.modules():
        if isinstance(c, nn.Linear):
            nn.init.xavier_uniform_(c.weight)
            nn.init.zeros_(c.bias)
    return m


def sgd(m, lr):
    return torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9, nesterov=True)


def decay(opt, lr, t):
    for g in opt.param_groups:
        g["lr"] = lr / (1 + 1e-6 * t)


def train_g(x, y, seed, device="cpu", epochs=G_EPOCHS):
    """G built on CPU after manual_seed and moved to the device; batch order from the CPU generator."""
    torch.manual_seed(seed)
    layers, dims = [], [2048, 1024, 512, 512, 1024]
    for a, b in zip(dims, dims[1:]):
        layers += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(0.25)]
    g = keras_init(nn.Sequential(*layers, nn.Linear(1024, 2048), nn.ReLU())).to(device)
    opt, t = sgd(g, 0.5), 0
    x, y = torch.tensor(x).to(device), torch.tensor(y).to(device)
    for _ in range(epochs):
        g.train()
        for i in torch.randperm(len(x)).split(100):
            i = i.to(device)
            decay(opt, 0.5, t)
            loss = ((g(x[i]) - y[i]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            t += 1
    return g.eval()


def train_head(x, y, xv, yv, seed, epochs=HEAD_EPOCHS):
    torch.manual_seed(seed)
    h = keras_init(nn.Sequential(nn.Dropout(0.25), nn.Linear(x.shape[1], 5)))
    n = np.bincount(y, minlength=5)
    w = torch.tensor([len(y) / (5 * c) if c else 0.0 for c in n], dtype=torch.float32)
    opt, t = sgd(h, 1e-4), 0
    x, y, xv, yv = torch.tensor(x), torch.tensor(y), torch.tensor(xv), torch.tensor(yv)
    best, state, stale = np.inf, None, 0
    for _ in range(epochs):
        h.train()
        for i in torch.randperm(len(x)).split(100):
            decay(opt, 1e-4, t)
            # Keras with class_weight: mean of the per-sample weighted losses
            loss = (nn.functional.cross_entropy(h(x[i]), y[i], reduction="none") * w[y[i]]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            t += 1
        h.eval()
        with torch.no_grad():
            vl = nn.functional.cross_entropy(h(xv), yv).item()  # Keras val_loss: no class weights
        if vl < best:
            best, state, stale = vl, {k: v.clone() for k, v in h.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= 200:
                break
    h.load_state_dict(state)
    return h.eval()


# ---------------------------------------------------------------------------------------------------- latents
def cmd_latents(a):
    sys.path.insert(0, os.path.join(ROOT, "train"))
    import utils
    device = torch.device(a.device if a.device == "cpu" or torch.cuda.is_available() else "cpu")

    def load(path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
        m = utils.ImageClassifier(utils.get_model("resnet50", pretrained=False), 5, bottleneck_dim=1024, finetune=True)
        m.load_state_dict(sd, strict=True)
        return m.to(device).eval()

    @torch.no_grad()
    def extract(m, ds):
        z, lg = [], []
        for x, _ in DataLoader(ds, batch_size=32, shuffle=False, num_workers=a.workers):
            f = m.pool_layer(m.backbone(x.to(device)))
            z.append(f.cpu().numpy().astype(np.float32))
            lg.append(m.head(m.bottleneck(f)).cpu().numpy().astype(np.float32))
        return np.concatenate(z), np.concatenate(lg)

    def labels(ds):
        if isinstance(ds, Subset):
            return labels(ds.dataset)[np.asarray(ds.indices)]
        return np.array([y for _, y in ds.samples])

    tf = utils.get_val_transform("default")
    clin, derm = os.path.join(a.data_root, "MILK10K_CLINICO"), os.path.join(a.data_root, "MILK10K")
    cls = utils.datasets.__dict__["MILK10K"]
    s = a.seed
    lab, pool, val = utils.get_dataset("MILK10K", 16, clin, tf, tf, seed=s)
    test = cls(root=clin, split="test", transform=tf)
    pair = {"lab": Subset(cls(root=derm, split="train", transform=tf), lab.indices),
            "pool": Subset(cls(root=derm, split="train", transform=tf), pool.indices),
            "val": cls(root=derm, split="val", transform=tf)}
    C = {"lab": lab, "pool": pool, "val": val, "test": test}
    for k in pair:  # the pair has the same relative path in both folders
        rc = [os.path.relpath(p, clin) for p in utils._rutas_dataset(C[k])]
        rd = [os.path.relpath(p, derm) for p in utils._rutas_dataset(pair[k])]
        assert rc == rd, k
    ei, es = load(a.erm_clinical.format(seed=s)), load(a.erm_derm.format(seed=s))
    out = {}
    for k, ds in C.items():
        out["zi_" + k], out["li_" + k] = extract(ei, ds)
        out["y_" + k] = labels(ds)
        out["id_" + k] = np.array([os.path.basename(p) for p in utils._rutas_dataset(ds)])
    for k, ds in pair.items():
        out["zs_" + k], out["ls_" + k] = extract(es, ds)
    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(os.path.join(a.out, "latents_s%d.npz" % s), **out)
    print("s%d: lab %d pool %d val %d test %d" % (s, *(len(C[k]) for k in ("lab", "pool", "val", "test"))), flush=True)


# ----------------------------------------------------------------------------------------------------- guided
def cmd_guided(a):
    torch.set_num_threads(4)
    s = a.seed
    d = np.load(os.path.join(a.out, "latents_s%d.npz" % s))
    g = train_g(np.concatenate([d["zi_pool"], d["zi_lab"]]), np.concatenate([d["zs_pool"], d["zs_lab"]]), s,
                epochs=a.g_epochs)
    with torch.no_grad():
        G = {k: g(torch.tensor(d["zi_" + k])).numpy() for k in ("lab", "val", "test")}
    row = dict(seed=s, mse_g_val=float(((G["val"] - d["zs_val"]) ** 2).mean()),
               erm_clinical=balanced_accuracy(d["y_test"], d["li_test"].argmax(1)))
    logits = {}
    for name, f in (("mallya", lambda k: np.concatenate([d["zi_" + k], G[k]], 1)),
                    ("photo", lambda k: d["zi_" + k]), ("guide", lambda k: G[k])):
        h = train_head(f("lab"), d["y_lab"], f("val"), d["y_val"], s, epochs=a.head_epochs)
        with torch.no_grad():
            for k in ("val", "test"):
                logits[name + "_" + k] = h(torch.tensor(f(k))).numpy()
        row[name] = balanced_accuracy(d["y_test"], logits[name + "_test"].argmax(1))
        row[name + "_val"] = balanced_accuracy(d["y_val"], logits[name + "_val"].argmax(1))
    np.savez_compressed(os.path.join(a.out, "guided_s%d.npz" % s), id_test=d["id_test"], y_test=d["y_test"], **logits)
    print(row, flush=True)


# -------------------------------------------------------------------------------------------------- ham-space
def cmd_ham_space(a):
    import extract_features as X
    torch.set_num_threads(4)
    device = torch.device(a.device if a.device == "cpu" or torch.cuda.is_available() else "cpu")
    rd, rc = np.load(a.derm_features), np.load(a.clinical_features)
    ids = rd["ids"].astype(str)
    assert (rc["ids"].astype(str) == ids).all()
    Fd = rd["F_" + a.features_key].astype(np.float64)
    if a.view == "base":
        Xv = rc["F_" + a.features_key]
    else:
        V = np.load(a.views)
        assert (V["ids"].astype(str) == ids).all()
        Xv = V["%s__z25" % a.features_key]
    Xv = Xv.astype(np.float64).astype(np.float32)
    split = rc["split"].astype(str)
    pool = X.milk_pool(ids, split, a.pool_exclude)
    excl = X.read_id_list(a.test_exclude)
    ite = np.flatnonzero(split == "test")
    te = ite[~np.array([X.isic_of(i) in excl for i in ids[ite]])]
    H = X.head_arrays(X.state_dict(a.ham))
    t0 = time.time()
    g = train_g(Xv[pool], Fd[pool].astype(np.float32), a.seed, device, a.g_epochs)
    with torch.no_grad():
        T = np.concatenate([g(torch.tensor(Xv[i:i + 1000]).to(device)).cpu().numpy() for i in range(0, len(Xv), 1000)])
    T = T.astype(np.float64)
    mse_pool, mse_te = float(((T[pool] - Fd[pool]) ** 2).mean()), float(((T[te] - Fd[te]) ** 2).mean())
    os.makedirs(a.out, exist_ok=True)
    dest = os.path.join(a.out, "g_%s_s%d.npz" % (a.view, a.seed))
    np.savez_compressed(dest, ids=ids, lp=X.balanced_log_posterior(X.apply_head(T, H)), T_te=T[te].astype(np.float32),
                        mse_pool=mse_pool, mse_te=mse_te, device=str(device))
    print("G %s s%d (%s) in %.1f min: MSE pool %.4f, test %.4f -> %s" % (
        a.view, a.seed, device, (time.time() - t0) / 60, mse_pool, mse_te, dest), flush=True)


def main():
    feats = os.path.join(ROOT, "features")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--device", default="cuda")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("latents")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--data-root", default=os.path.join(ROOT, "data"))
    p.add_argument("--erm-clinical", default=os.path.join(ROOT, "outputs", "ham", "milk10k_clinico", "k16", "erm",
                                                          "s{seed}", "checkpoints", "best.pth"))
    p.add_argument("--erm-derm", default=os.path.join(ROOT, "outputs", "ham", "milk10k", "k16", "erm",
                                                      "s{seed}", "checkpoints", "best.pth"))
    p.add_argument("--workers", type=int, default=4)
    p = sub.add_parser("guided")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--g-epochs", type=int, default=G_EPOCHS)
    p.add_argument("--head-epochs", type=int, default=HEAD_EPOCHS)
    p = sub.add_parser("ham-space")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--view", choices=["base", "z25"], default="base")
    p.add_argument("--ham", default=os.path.join(ROOT, "checkpoints", "ham10000_source.pth"))
    p.add_argument("--derm-features", default=os.path.join(feats, "ham__milk10k.npz"))
    p.add_argument("--clinical-features", default=os.path.join(feats, "ham__milk10k_clinico.npz"))
    p.add_argument("--views", default=os.path.join(feats, "ham__milk10k_views.npz"))
    p.add_argument("--features-key", default="ham")
    p.add_argument("--pool-exclude", default=os.path.join(ROOT, "data", "splits", "milk10k_pool_excluded.txt"))
    p.add_argument("--test-exclude", default=os.path.join(ROOT, "data", "splits", "milk10k_test_excluded.txt"))
    p.add_argument("--g-epochs", type=int, default=G_EPOCHS)
    a = ap.parse_args()
    {"latents": cmd_latents, "guided": cmd_guided, "ham-space": cmd_ham_space}[a.cmd](a)


if __name__ == "__main__":
    main()
