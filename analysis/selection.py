"""Label selection in the frozen-feature proxy, Study 2 (Section 5.2.1 and Appendix 7.4, label selection).

The proxy is a balanced logistic regression (C = 1) on frozen features of the HAM10000 ResNet-50, trained on the 80
labeled images of a set and scored on the target test set. A draw takes 90 % of the candidate images as the pool; every
selection of that draw chooses from the same pool. Random generators are seeded per draw and per selection, so the
sets do not depend on the order in which selections run.

Inputs: features/selection_<target>.npz (release asset): ResNet-50 features (float16), logits of the HAM10000
ResNet-50 and ConvNeXt-T, labels, split, lesion and pool exclusions of every image, in ImageFolder order.
`compute()` writes the per-draw results to results/study2/selection_proxy_*.csv; the table and figure scripts read them.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # k-means moves an image of a set when run multithreaded

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

import common as C  # noqa: E402

FEATURES = os.path.join(C.ROOT, "features")
STUDY2 = os.path.join(C.RESULTS, "study2")
TARGETS = ["milk10k", "bcn20000"]
K = 16
N_DRAWS = 30
HAM_TRAIN = np.array([210, 231, 612, 447, 4084], float)  # HAM10000 train images per class
LOG_PRIOR_HAM = np.log(HAM_TRAIN / HAM_TRAIN.sum())
# Index of each selection in the seed of its generator (fixed; do not reorder).
SEED_INDEX = {"random_per_class": 0, "random_80": 1, "kcenter": 2, "kmeans_centers": 3, "typiclust": 4,
              "probcover": 5, "maxherding": 6, "pseudo_class_balance": 7}
COLD_START = ["kcenter", "kmeans_centers", "typiclust", "probcover", "maxherding", "pseudo_class_balance"]
RULES = {"agreement_rule": 3, "mixed_rule": 4, "sequential_rule": 5}
TRAITS = ["resnet_correct", "networks_agree", "confidence", "margin", "typicality", "centrality"]


# ------------------------------------------------------------------------------------------------- data
def log_softmax(z):
    z = np.asarray(z, np.float64)
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def softmax(z):
    return np.exp(log_softmax(z))


def ba(y, pred):
    return 100.0 * float(np.mean([(pred[y == c] == c).mean() for c in range(5) if (y == c).any()]))


_DATA = {}


def load(t):
    if t in _DATA:
        return _DATA[t]
    path = os.path.join(FEATURES, f"selection_{t}.npz")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found: download the features of the release (see results/README.md)")
    z = np.load(path)
    F = z["features"].astype(np.float32)
    F /= np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-12)
    sp = z["split"].astype(str)
    D = dict(t=t, ids=z["ids"].astype(str), y=z["labels"].astype(int), F=F, lesion=z["lesion"].astype(str),
             pool_excluded=z["pool_excluded"], test_ok=C.primary_mask(t, z["ids"].astype(str)))
    D["itr"], D["iva"], D["ite"] = (np.flatnonzero(sp == s) for s in ("train", "val", "test"))
    p_res = softmax(z["logits_resnet"].astype(np.float64) - LOG_PRIOR_HAM)  # HAM10000 prior removed
    p_cnx = softmax(z["logits_convnext"].astype(np.float64) - LOG_PRIOR_HAM)
    D["pred_res"], D["pred_cnx"] = p_res.argmax(1), p_cnx.argmax(1)
    D["score"] = (p_res + p_cnx) / 2  # combined prediction of the two HAM10000 networks
    D["pred"] = D["score"].argmax(1)
    _DATA[t] = D
    return D


def candidates(D):
    return D["itr"][~D["pool_excluded"][D["itr"]]]


def draw_pool(D, s):
    rs = np.random.default_rng(77_000 + 100 * s + K)
    cand = candidates(D)
    return np.sort(rs.choice(cand, int(0.9 * len(cand)), replace=False))


def one_per_lesion(D, cand, rng):
    les = D["lesion"][cand]
    order = rng.permutation(len(cand))
    _, first = np.unique(les[order], return_index=True)
    return cand[order[first]]


# ------------------------------------------------------------------------------------------------- proxy
class Proxy:
    """Validation split into 16 per class (v16) and the rest (dev); test = primary test set."""

    def __init__(self, D, seed):
        y, iva = D["y"], D["iva"]
        rng = np.random.RandomState(10_000 + seed)
        yv = y[iva]
        v = sorted(sum((rng.choice(np.flatnonzero(yv == c), min(16, (yv == c).sum()), replace=False).tolist()
                        for c in range(5)), []))
        self.v16 = iva[np.array(v, int)]
        self.dev = iva[~np.isin(iva, self.v16)]
        self.test = D["ite"][D["test_ok"][D["ite"]]]


def probe(F, y, L):
    """Weights (5 x d) and biases of the balanced logistic regression; absent classes get bias -1e4."""
    present = np.unique(y[L])
    W, b = np.zeros((5, F.shape[1])), np.full(5, -1e4)
    if len(present) == 1:
        b[present[0]] = 0.0
        return W, b
    lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, tol=1e-6).fit(F[L], y[L])
    Wp, bp = lr.coef_.copy(), lr.intercept_.copy()
    if len(present) == 2:  # binary: one row
        Wp, bp = np.vstack([-Wp[0] / 2, Wp[0] / 2]), np.array([-bp[0] / 2, bp[0] / 2])
    W[present], b[present] = Wp, bp
    return W, b


def evaluate(D, proxy, L):
    y, F = D["y"], D["F"]
    W, b = probe(F, y, L)
    out = {}
    for name, idx in (("dev", proxy.dev), ("test", proxy.test)):
        p = (F[idx] @ W.T + b).argmax(1)
        out["ba_" + name] = ba(y[idx], p)
        if name == "test":
            out["rec_mel"] = 100.0 * float((p[y[idx] == 3] == 3).mean())
    return out


# ------------------------------------------------------------------------------------------------- strategies
def dist2(F):
    return np.maximum(2.0 - 2.0 * (F @ F.T), 0.0).astype(np.float32)


def kmeans(F, n, rng, iters=30):
    km = KMeans(n_clusters=n, n_init=1, max_iter=iters, random_state=int(rng.integers(1 << 31)))
    return km.fit_predict(F), km.cluster_centers_


def prototypes(F, stratum, n, groups, rng):
    """n images of the stratum: spherical k-means with `groups` groups, the n/groups closest to each centre."""
    sub = stratum if len(stratum) <= 2500 else rng.choice(stratum, 2500, replace=False)
    Fs = F[sub]
    groups = max(1, min(groups, len(sub)))
    cen = Fs[rng.choice(len(sub), groups, replace=False)].copy()
    for _ in range(25):
        a = (Fs @ cen.T).argmax(1)
        for j in range(groups):
            if (a == j).any():
                m = Fs[a == j].mean(0)
                cen[j] = m / np.linalg.norm(m)
    a = (Fs @ cen.T).argmax(1)
    quota = [n // groups + (1 if j < n % groups else 0) for j in range(groups)]
    chosen = []
    for j in range(groups):
        members = np.flatnonzero(a == j)
        chosen.extend(members[np.argsort(-(Fs[members] @ cen[j]))][:quota[j]].tolist())
    if n > len(chosen):
        rest = np.setdiff1d(np.arange(len(sub)), chosen)
        chosen.extend(rng.choice(rest, min(n - len(chosen), len(rest)), replace=False).tolist())
    return sub[np.array(chosen, dtype=int)]


def kcenter(F, B, rng):
    """k-center greedy (Core-Set), random start."""
    chosen = [int(rng.integers(len(F)))]
    dmin = np.maximum(2.0 - 2.0 * (F @ F[chosen[0]]), 0.0)
    for _ in range(B - 1):
        j = int(dmin.argmax())
        chosen.append(j)
        dmin = np.minimum(dmin, np.maximum(2.0 - 2.0 * (F @ F[j]), 0.0))
    return np.array(chosen)


def kmeans_centers(F, B, rng):
    a, cen = kmeans(F, B, rng)
    return np.array([m[((F[m] - cen[j]) ** 2).sum(1).argmin()] for j in range(B)
                     for m in [np.flatnonzero(a == j)] if len(m)])


def typiclust(F, B, rng, knn=20, minimum=5):
    """TypiClust: B clusters; the most typical image of the clusters with fewest picks, largest first."""
    a, _ = kmeans(F, B, rng)
    size = np.bincount(a, minlength=B)
    typ = np.zeros(len(F))
    for j in range(B):
        m = np.flatnonzero(a == j)
        if len(m) < 2:
            continue
        kk = min(knn, len(m) - 1)
        d = np.sqrt(np.partition(dist2(F[m]), kk, axis=1)[:, 1:kk + 1]).mean(1)
        typ[m] = 1.0 / (d + 1e-8)
    chosen, used = [], np.zeros(B, int)
    while len(chosen) < B:
        ok = np.flatnonzero((size >= minimum) & (size > used))
        if len(ok) == 0:
            ok = np.flatnonzero(size > used)
        j = ok[np.lexsort((-size[ok], used[ok]))[0]]
        m = np.flatnonzero(a == j)
        m = m[~np.isin(m, chosen)]
        chosen.append(m[typ[m].argmax()])
        used[j] += 1
    return np.array(chosen)


def probcover(F, B, rng, purity=0.95):
    """ProbCover: balls of the largest radius with purity >= 0.95 against k-means (5 groups); greedy coverage."""
    D2 = dist2(F)
    a, _ = kmeans(F, 5, rng)
    delta = None
    for d in np.linspace(0.05, 1.4, 55):
        mixed = ((D2 <= d * d) & (a[None, :] != a[:, None])).any(1)
        if 1.0 - mixed.mean() >= purity:
            delta = d
        else:
            break
    delta = 0.05 if delta is None else delta
    adj = D2 <= delta * delta
    covered = np.zeros(len(F), bool)
    chosen = []
    for _ in range(B):
        gain = (adj & ~covered[None, :]).sum(1).astype(float)
        gain[chosen] = -1
        if gain.max() <= 0:  # all covered: the rest at random
            rest = np.setdiff1d(np.arange(len(F)), chosen)
            chosen.extend(rng.choice(rest, B - len(chosen), replace=False).tolist())
            break
        j = int(rng.choice(np.flatnonzero(gain == gain.max())))
        chosen.append(j)
        covered |= adj[j]
    return np.array(chosen)


def maxherding(F, B, rng, knn=20):
    """MaxHerding: greedy on sum_x max_l k(x, l), Gaussian kernel with sigma = median distance to the 20th neighbour."""
    D2 = dist2(F)
    sigma = float(np.median(np.sqrt(np.partition(D2, knn, axis=1)[:, knn])))
    Kx = np.exp(-D2 / (2.0 * sigma * sigma)).astype(np.float32)
    cur = np.zeros(len(F), np.float32)
    chosen = []
    for _ in range(B):
        gain = np.maximum(Kx, cur[None, :]).sum(1)
        gain[chosen] = -np.inf
        j = int(gain.argmax())
        chosen.append(j)
        cur = np.maximum(cur, Kx[j])
    return np.array(chosen)


def pseudo_class_balance(F, k, rng):
    a, _ = kmeans(F, 5, rng)
    idx = np.arange(len(F))
    return np.concatenate([prototypes(F, idx[a == j], k, 4, rng) for j in range(5)])


def select(name, D, pool, rng):
    """80 images of the pool. Only random_per_class looks at the labels of the candidates."""
    y, F = D["y"], D["F"]
    if name == "random_per_class":
        return np.concatenate([rng.choice(pool[y[pool] == c], K, replace=False) for c in range(5)])
    if name == "random_80":
        return rng.choice(pool, 5 * K, replace=False)
    if name == "agreement_rule":
        return agreement_rule(D, pool, rng)
    base = one_per_lesion(D, pool, rng) if D["t"] == "bcn20000" else pool
    if name == "pseudo_class_balance":
        return base[pseudo_class_balance(F[base], K, rng)]
    fn = dict(kcenter=kcenter, kmeans_centers=kmeans_centers, typiclust=typiclust, probcover=probcover,
              maxherding=maxherding)[name]
    return base[fn(F[base], 5 * K, rng)]


def class_prototypes(D, pool, rng):
    """Class-aware prototypes: per true class (one image per lesion on BCN20000), 8 groups, 2 per group."""
    y, F = D["y"], D["F"]
    out = []
    for c in range(5):
        st = pool[y[pool] == c]
        if D["t"] == "bcn20000":
            st = one_per_lesion(D, st, rng)
        out.append(prototypes(F, st, K, 8, rng))
    return np.concatenate(out)


# ------------------------------------------------------------------------------------------------- heuristics
def agreement_rule(D, pool, rng, k=K):
    """Per predicted class: images where both HAM10000 networks agree, above the median score; 4 prototype groups."""
    F, score = D["F"], D["score"]
    base = one_per_lesion(D, pool, rng) if D["t"] == "bcn20000" else pool
    agree = D["pred_res"][base] == D["pred_cnx"][base]
    L = []
    for c in range(5):
        st = base[agree & (D["pred_res"][base] == c)]
        p = score[st, c]
        cand = st[p >= np.median(p)] if len(st) else st
        if len(cand) < 2 * k:  # fill: rest of the agreement, then the class of the ResNet-50 alone
            extra = st[~np.isin(st, cand)]
            cand = np.concatenate([cand, extra[np.argsort(-score[extra, c])][:2 * k - len(cand)]])
        if len(cand) < 2 * k:
            alone = base[(D["pred_res"][base] == c) & ~np.isin(base, cand)]
            cand = np.concatenate([cand, alone[np.argsort(-score[alone, c])][:2 * k - len(cand)]])
        cand = cand[~np.isin(cand, np.concatenate(L) if L else [])]
        L.append(prototypes(F, cand, k, 4, rng))
    L = np.concatenate(L)
    if len(L) < 5 * k:
        rest = base[~np.isin(base, L)]
        L = np.concatenate([L, rng.choice(rest, 5 * k - len(L), replace=False)])
    return L


def kmeans_prototypes(F, cand, groups, per_group, rng):
    if len(cand) <= groups * per_group:
        return list(cand)
    km = KMeans(groups, n_init=1, random_state=int(rng.integers(1 << 31))).fit(F[cand])
    out = []
    for g in range(groups):
        m = cand[km.labels_ == g]
        out += list(m[np.argsort(((F[m] - km.cluster_centers_[g]) ** 2).sum(1))][:per_group])
    return out


def mixed_rule(D, base, rng):
    """Per predicted class (combined prediction): 8 prototypes of the more confident half and 8 of the less."""
    F, P = D["F"], D["score"]
    pred = P.argmax(1)
    L = []
    for c in range(5):
        st = base[pred[base] == c]
        conf = P[st, c]
        doubtful, confident = st[conf < np.median(conf)], st[conf >= np.median(conf)]
        sel = kmeans_prototypes(F, confident, 2, 4, rng) + kmeans_prototypes(F, doubtful, 2, 4, rng)
        if len(sel) < 16:
            rest = st[~np.isin(st, sel)]
            sel += list(rest[np.argsort(P[rest, c])][:16 - len(sel)])
        L += sel[:16]
    return np.array(L)


def complete(L, base, n, rng):
    L = list(dict.fromkeys(int(i) for i in L))
    if len(L) < n:
        L += rng.choice(base[~np.isin(base, L)], n - len(L), replace=False).tolist()
    return np.array(L[:n])


def allocate(m, short):
    if short.sum() <= 0:
        short = np.ones(5)
    q = m * short / short.sum()
    a = np.floor(q).astype(int)
    for i in np.argsort(-(q - a))[:m - a.sum()]:
        a[i] += 1
    return a


def sequential_rule(D, base, rng, rounds=(40, 20, 20)):
    """Rounds of 40 + 20 + 20. Round 1: agreement rule. Later rounds: probe on the labels received so far times the
    HAM10000 consensus; the quota goes to the classes still short, prototypes in the more confident half."""
    F, y, P0 = D["F"], D["y"], D["score"]
    L = list(agreement_rule(D, base, rng, k=rounds[0] // 5))
    for m in rounds[1:]:
        La = np.array(L)
        n = np.bincount(y[La], minlength=5)  # labels of the images already chosen
        W, b = probe(F, y, La)
        Pb = softmax(log_softmax(F[base] @ W.T + b) + np.log(P0[base] + 1e-12))
        pred = Pb.argmax(1)
        free = ~np.isin(base, La)
        quota = allocate(m, np.maximum((len(La) + m) / 5.0 - n, 0))
        for c in np.argsort(-quota):
            a = int(quota[c])
            if a == 0:
                continue
            if n[c] == 0:  # class not labeled yet: the probe does not know it, use HAM10000 alone
                ix = np.flatnonzero(free)
                order = ix[np.argsort(-P0[base[ix], c])][:max(2 * a, 30)]
            else:
                ix = np.flatnonzero(free & (pred == c))
                order = ix[np.argsort(-Pb[ix, c])]
                if len(order) < a:
                    rest = np.flatnonzero(free & (pred != c))
                    order = np.concatenate([order, rest[np.argsort(-Pb[rest, c])][:a - len(order)]])
            cand = base[order]
            sel = list(cand) if len(cand) <= a else list(
                prototypes(F, cand[:max(2 * a, len(cand) // 2)], a, min(4, a), rng))
            L += [int(i) for i in sel]
            free[np.isin(base, sel)] = False
    return complete(L, base, sum(rounds), rng)


def heuristic(name, D, pool, rng):
    if name == "agreement_rule":
        return agreement_rule(D, pool, rng)
    base = one_per_lesion(D, pool, rng) if D["t"] == "bcn20000" else pool
    return mixed_rule(D, base, rng) if name == "mixed_rule" else sequential_rule(D, base, rng)


# ------------------------------------------------------------------------------------------------- oracle
class Centroids:
    """Balanced accuracy of a nearest-centroid (cosine) classifier on fixed images, from per-class sums."""

    def __init__(self, D, idx):
        self.Fe, self.ye = D["F"][idx], D["y"][idx]

    def ba(self, S, n):
        if (n == 0).any():
            return 0.0
        return ba(self.ye, (self.Fe @ (S / np.linalg.norm(S, axis=1, keepdims=True)).T).argmax(1))


def oracle_search(D, L0, pool, guide, steps, rng):
    """Hill climbing by swaps (one labeled image out, one pool image in, kept if the guide does not drop)."""
    F, y = D["F"], D["y"]
    L = list(L0)
    out = np.setdiff1d(pool, L)
    S, n = np.zeros((5, F.shape[1])), np.zeros(5)
    for i in L:
        S[y[i]] += F[i]
        n[y[i]] += 1
    current = guide.ba(S, n)
    for _ in range(steps):
        a = int(rng.integers(len(L)))
        i, j = L[a], int(out[rng.integers(len(out))])
        S[y[i]] -= F[i]
        n[y[i]] -= 1
        S[y[j]] += F[j]
        n[y[j]] += 1
        new = guide.ba(S, n)
        if new >= current:
            current = new
            L[a] = j
            out[out == j] = i
        else:
            S[y[j]] -= F[j]
            n[y[j]] -= 1
            S[y[i]] += F[i]
            n[y[i]] += 1
    return np.array(L)


def oracle_sets(D, seed_base, restarts=6, steps=4000):
    """Search guided by dev, any 80 of the candidates; restarts alternate a random and an agreement-rule start."""
    pool = candidates(D)
    guide = Centroids(D, Proxy(D, 0).dev)
    out = []
    for r in range(restarts):
        rng = np.random.default_rng(seed_base + 1000 * r)
        start = "random_80" if r % 2 == 0 else "agreement_rule"
        L0 = rng.choice(pool, 5 * K, replace=False) if start == "random_80" else agreement_rule(D, pool, rng)
        out.append((r, start, oracle_search(D, L0, pool, guide, steps, rng)))
    return out


# ------------------------------------------------------------------------------------------------- traits
def traits(D, pool):
    """Per image: ResNet-50 correct, networks agree, confidence and margin of the combined prediction, typicality
    (mean cosine to its 20 nearest candidates) and centrality (cosine to the centre of its true class in the pool)."""
    F, y, P = D["F"], D["y"], D["score"]
    top = np.sort(P, axis=1)
    R = pd.DataFrame(dict(resnet_correct=(D["pred_res"] == y).astype(float),
                          networks_agree=(D["pred_res"] == D["pred_cnx"]).astype(float),
                          confidence=top[:, -1], margin=top[:, -1] - top[:, -2]))
    Fp = F[pool]
    typ = np.full(len(y), np.nan)
    for i0 in range(0, len(pool), 2000):
        block = pool[i0:i0 + 2000]
        S = F[block] @ Fp.T
        S[np.arange(len(block)), np.arange(i0, i0 + len(block))] = -np.inf
        typ[block] = np.sort(S, axis=1)[:, -20:].mean(1)
    R["typicality"] = typ
    cen = np.full(len(y), np.nan)
    for c in range(5):
        m = pool[y[pool] == c]
        mu = F[m].mean(0)
        cen[m] = F[m] @ (mu / np.linalg.norm(mu))
    R["centrality"] = cen
    return R


# ------------------------------------------------------------------------------------------------- compute
def composition(D, L):
    return {"n_" + c: int(v) for c, v in zip(C.CLASSES, np.bincount(D["y"][L], minlength=5))}


def compute_draws(log=print):
    """Random, Study 1 protocol, class-aware prototypes and the six cold-start strategies, 30 draws per target.
    Also the traits of each set on BCN20000."""
    rows, trait_rows = [], []
    for t in TARGETS:
        D = load(t)
        R = traits(D, candidates(D)) if t == "bcn20000" else None
        for s in range(N_DRAWS):
            proxy, pool = Proxy(D, s % 5), draw_pool(D, s)
            sets = {n: select(n, D, pool, np.random.default_rng(10_000 * s + 100 * K + i))
                    for n, i in SEED_INDEX.items()}
            sets["class_prototypes"] = class_prototypes(D, pool, np.random.default_rng(10_000 * s + 100 * K + 99))
            for n, L in sets.items():
                assert len(np.unique(L)) == len(L) == 5 * K, (t, n, s)
                rows.append(dict(target=t, selection=n, draw=s, **evaluate(D, proxy, L), **composition(D, L)))
                if R is not None:
                    trait_rows.append(dict(target=t, selection=n, set=s, **composition(D, L),
                                           **R.loc[L, TRAITS].mean().to_dict()))
            log(f"{t} draw {s}")
    return pd.DataFrame(rows), pd.DataFrame(trait_rows)


def compute_oracle(log=print):
    """BCN20000: six restarts of the oracle search and 1000 random sets of each reference on the full candidate set,
    all scored with the proxy of seed 0; plus the six restarts whose images are described in the trait tables."""
    D = load("bcn20000")
    proxy, pool = Proxy(D, 0), candidates(D)
    rows = []
    for r, start, L in oracle_sets(D, 9_000_010):
        rows.append(dict(target="bcn20000", selection="oracle", set=r, start=start, **evaluate(D, proxy, L)))
    log("oracle search done")
    for n, i in (("random_80", 0), ("random_per_class", 1)):
        for s in range(1000):
            L = select(n, D, pool, np.random.default_rng(7_000_000 + 10_000 * i + s))
            rows.append(dict(target="bcn20000", selection=n, set=s, start="", **evaluate(D, proxy, L)))
        log(f"{n} sets done")
    R = traits(D, pool)
    trait_rows = [dict(target="bcn20000", selection="pool", set=0, **composition(D, pool),
                       **R.loc[pool, TRAITS].mean().to_dict())]
    for r, _, L in oracle_sets(D, 9_100_000):
        trait_rows.append(dict(target="bcn20000", selection="oracle", set=r, **composition(D, L),
                               **R.loc[L, TRAITS].mean().to_dict()))
    log("oracle traits done")
    return pd.DataFrame(rows), pd.DataFrame(trait_rows)


def compute_controls(reps=10, log=print, targets=TARGETS):
    """Each heuristic, a composition control (random images with the same true-class counts) and a recognized-image
    control (same counts, drawn among the images the combined HAM10000 prediction gets right); 10 controls per draw."""
    rows = []
    for t in targets:
        D = load(t)
        y = D["y"]
        for s in range(N_DRAWS):
            proxy, pool = Proxy(D, s % 5), draw_pool(D, s)
            sets = {("random_80", "none", 0): select("random_80", D, pool, np.random.default_rng(
                10_000 * s + 100 * K + SEED_INDEX["random_80"]))}
            for n, i in RULES.items():
                sets[(n, "none", 0)] = heuristic(n, D, pool, np.random.default_rng(31_000_000 + 10_000 * s + i))
            right = pool[D["pred"][pool] == y[pool]]
            for rep in range(reps):
                for j, n in enumerate(RULES):
                    rng = np.random.default_rng(39_000_000 + 1000 * s + 100 * rep + j)
                    cnt = np.bincount(y[sets[(n, "none", 0)]], minlength=5)
                    sets[(n, "composition", rep)] = np.concatenate(
                        [rng.choice(pool[y[pool] == c], cnt[c], replace=False) for c in range(5)])
                    L3 = []
                    for c in range(5):
                        b = right[y[right] == c]
                        if len(b) < cnt[c]:
                            b = pool[y[pool] == c]
                        L3 += rng.choice(b, cnt[c], replace=False).tolist()
                    sets[(n, "recognized", rep)] = np.array(L3)
            for (n, ctl, rep), L in sets.items():
                assert len(L) == 80 and len(np.unique(L)) == 80, (t, s, n, ctl)
                rows.append(dict(target=t, selection=n, control=ctl, draw=s, rep=rep,
                                 ba_test=evaluate(D, proxy, L)["ba_test"],
                                 combined_correct=float((D["pred"][L] == y[L]).mean()), **composition(D, L)))
            log(f"{t} draw {s}")
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------- fine-tuning
FT_NAMES = {"random_80": "80 random images", "random_per_class": "16 random per class (Study 1)",
            "class_prototypes": "Class-aware prototypes", "agreement_rule": "Agreement rule",
            "mixed_rule": "Mixed rule", "sequential_rule": "Sequential rule"}


def finetune():
    """One-epoch ERM networks trained on each labeled set (8 sets per selection and target).
    Returns the runs table (results/study2/selection_finetune_runs.csv) with test metrics on the primary test set,
    and per target (correct per run and image, labels, lesion index) for the bootstrap."""
    runs = pd.read_csv(os.path.join(STUDY2, "selection_finetune_runs.csv"))
    z = np.load(os.path.join(STUDY2, "selection_finetune.npz"))
    rows, per_target = [], {}
    for t in TARGETS:
        ids, y = z["ids_test_" + t].astype(str), z["labels_test_" + t].astype(int)
        ok = C.primary_mask(t, ids)
        logits = z["logits_test_" + t]
        split = pd.read_csv(os.path.join(C.ROOT, "data", "splits", {"milk10k": "milk10k_dermoscopic.csv",
                                                                    "bcn20000": "bcn20000.csv"}[t]))
        lesion = split.set_index("isic_id").loc[ids[ok], "lesion_id"].to_numpy()
        correct = {}
        for r in runs[runs.target == t].itertuples():
            lg = logits[r.row].astype(np.float64)[ok]
            m = C.metrics(lg, y[ok])
            pm = softmax(lg)[:, 3]
            pos, neg = pm[y[ok] == 3], pm[y[ok] != 3]
            rows.append(dict(run_id=r.run_id, target=t, selection=r.selection, set=r.set, ba=m["ba"],
                             **{"rec_" + c: m["rec_" + c] for c in C.CLASSES},
                             mel_sens_at_spec80=100 * float((pos > np.quantile(neg, 0.80)).mean()),
                             **{"n_" + c: getattr(r, "n_" + c) for c in C.CLASSES}))
            correct.setdefault(r.selection, []).append(lg.argmax(1) == y[ok])
        per_target[t] = ({k: np.array(v) for k, v in correct.items()}, y[ok], np.unique(lesion, return_inverse=True)[1])
    return pd.DataFrame(rows), per_target


FILES = {"draws": "selection_proxy_draws.csv", "traits": "selection_proxy_traits.csv",
         "oracle": "selection_proxy_oracle.csv", "oracle_traits": "selection_proxy_oracle_traits.csv",
         "controls": "selection_proxy_controls.csv"}


def compute(parts=("draws", "oracle", "controls")):
    """Recompute the per-draw results (single-threaded CPU) and write them to results/study2/."""
    from threadpoolctl import threadpool_limits
    os.makedirs(STUDY2, exist_ok=True)
    out = {}
    with threadpool_limits(1):
        if "draws" in parts:
            out["draws"], out["traits"] = compute_draws()
        if "oracle" in parts:
            out["oracle"], out["oracle_traits"] = compute_oracle()
        if "controls" in parts:
            out["controls"] = compute_controls()
    for name, df in out.items():
        df.round(6).to_csv(os.path.join(STUDY2, FILES[name]), index=False, lineterminator="\n")
        print(f"  -> results/study2/{FILES[name]}")


def results(name, recompute=False):
    """Per-draw results of the proxy; with recompute=True they are rebuilt from the features first."""
    path = os.path.join(STUDY2, FILES[name])
    if recompute:
        compute(({"traits": "draws", "oracle_traits": "oracle"}.get(name, name),))
    return pd.read_csv(path)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Recompute the per-draw results of the label-selection proxy.")
    ap.add_argument("parts", nargs="*", default=["draws", "oracle", "controls"],
                    choices=["draws", "oracle", "controls"])
    compute(tuple(ap.parse_args().parts))
