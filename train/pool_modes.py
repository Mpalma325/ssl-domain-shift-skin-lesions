"""
Unlabeled-pool variants for the sibling-image ablation on BCN20000 (the labeled set is unchanged).

  orig                 default pool
  sin_hermanas         drops every image of lesions with a labeled image
  remocion_emparejada  per class, drops random whole lesions until it matches what sin_hermanas removed
  tamano_milk          sin_hermanas, then keeps random whole lesions per class down to the MILK10k pool size

Each mode draws from its own RandomState(seed + OFFSET), so the training random sequence is unchanged.
"""
import csv
import os
from collections import defaultdict

import numpy as np

POOL_MODES = ("orig", "sin_hermanas", "remocion_emparejada", "tamano_milk")
OFFSET_B = 1000003
OFFSET_C = 2000003
# images in MILK10K/train; the MILK pool at a given k is this minus 5*k (3103 at k=16)
MILK_TRAIN_IMAGES = 3183


def lesion_ids(samples, root):
    """lesion_id of each image from split_manifest.csv; an image without one is its own lesion."""
    with open(os.path.join(root, "split_manifest.csv"), newline="", encoding="utf-8") as fh:
        les = {r["isic_id"]: (r["lesion_id"] or r["isic_id"]) for r in csv.DictReader(fh)}
    ids = [os.path.splitext(os.path.basename(p))[0] for p, _ in samples]
    faltan = [i for i in ids if i not in les]
    assert not faltan, "{} images without a row in split_manifest.csv (e.g. {})".format(
        len(faltan), faltan[:3])
    return [les[i] for i in ids]


def _acumular_lesiones(orden, imgs_de, meta):
    """Takes whole lesions in order until the image count is closest to meta (ties: stop)."""
    tomadas, total = [], 0
    for l in orden:
        if total >= meta:
            break
        m = len(imgs_de[l])
        if total + m <= meta:
            tomadas.append(l)
            total += m
            continue
        if abs(total + m - meta) < abs(total - meta):
            tomadas.append(l)
        break
    return tomadas


def select_pool(mode, labeled_idxes, unlabeled_idxes, labels, lesions, seed, num_classes,
                num_samples_per_class):
    """Returns (filtered unlabeled_idxes, report). The relative order of the pool is preserved."""
    assert mode in POOL_MODES, mode
    if mode == "orig":
        return list(unlabeled_idxes), None
    assert seed is not None, "pool-mode requires --seed"
    labels = np.asarray(labels)
    lab_les = {lesions[i] for i in labeled_idxes}
    es_hermana = {i: lesions[i] in lab_les for i in unlabeled_idxes}
    hermanas = [i for i in unlabeled_idxes if es_hermana[i]]
    quitar = set()

    if mode == "sin_hermanas":
        quitar = set(hermanas)
        meta = None

    elif mode == "remocion_emparejada":
        rng = np.random.RandomState(seed + OFFSET_B)
        meta = {c: 0 for c in range(num_classes)}
        for i in hermanas:
            meta[labels[i]] += 1
        for c in range(num_classes):
            restantes = [i for i in unlabeled_idxes if labels[i] == c and not es_hermana[i]]
            imgs_de = defaultdict(list)
            for i in restantes:
                imgs_de[lesions[i]].append(i)
            total = 0
            while total < meta[c] and restantes:
                l = lesions[restantes[rng.randint(len(restantes))]]
                m = len(imgs_de[l])
                if total + m > meta[c] and abs(total + m - meta[c]) >= abs(total - meta[c]):
                    break
                quitar.update(imgs_de[l])
                total += m
                restantes = [i for i in restantes if lesions[i] != l]

    elif mode == "tamano_milk":
        # starts from the pool without siblings and shrinks it per class at the same rate
        rng = np.random.RandomState(seed + OFFSET_C)
        base = [i for i in unlabeled_idxes if not es_hermana[i]]
        objetivo = MILK_TRAIN_IMAGES - num_classes * num_samples_per_class
        tasa = objetivo / float(len(base))
        meta = {"tasa": tasa, "objetivo": objetivo}
        estratos = defaultdict(lambda: defaultdict(list))
        for i in base:
            estratos[int(labels[i])][lesions[i]].append(i)
        conservar = set()
        for clave in sorted(estratos):
            imgs_de = estratos[clave]
            claves = sorted(imgs_de)
            orden = [claves[j] for j in rng.permutation(len(claves))]
            n_imgs = sum(len(v) for v in imgs_de.values())
            for l in _acumular_lesiones(orden, imgs_de, int(round(tasa * n_imgs))):
                conservar.update(imgs_de[l])
        quitar = set(unlabeled_idxes) - conservar

    nuevo = [i for i in unlabeled_idxes if i not in quitar]
    return nuevo, _reporte(mode, unlabeled_idxes, nuevo, quitar, es_hermana, labels, lesions,
                           num_classes, meta)


def _reporte(mode, antes, despues, quitar, es_hermana, labels, lesions, num_classes, meta):
    por_clase = []
    for c in range(num_classes):
        a = [i for i in antes if labels[i] == c]
        d = [i for i in despues if labels[i] == c]
        q = [i for i in quitar if labels[i] == c]
        fila = {"clase": c, "pool_antes": len(a), "pool_despues": len(d),
                "imgs_quitadas": len(q), "lesiones_quitadas": len({lesions[i] for i in q}),
                "hermanas_despues": sum(es_hermana[i] for i in d)}
        if mode == "remocion_emparejada":
            fila["meta"] = meta[c]
        if mode == "tamano_milk":
            fila["base_e1"] = sum(1 for i in a if not es_hermana[i])
        por_clase.append(fila)
    return {"modo": mode, "pool_antes": len(antes), "pool_despues": len(despues),
            "meta": meta if mode == "tamano_milk" else None, "por_clase": por_clase}


def format_report(rep, classes):
    """Human-readable report, for the run log and for the dry run."""
    if rep is None:
        return "pool-mode orig: pool unchanged"
    out = ["pool-mode {}: pool {} -> {}".format(rep["modo"], rep["pool_antes"], rep["pool_despues"])]
    if rep["meta"]:
        out[0] += "  (target {}, rate {:.4f})".format(rep["meta"]["objetivo"], rep["meta"]["tasa"])
    for f in rep["por_clase"]:
        pct = 100.0 * f["hermanas_despues"] / f["pool_despues"] if f["pool_despues"] else 0.0
        s = "  {:6s} pool {:5d} -> {:5d}  removed {:5d} imgs / {:4d} lesions  siblings {:5d} ({:5.1f}%)".format(
            classes[f["clase"]], f["pool_antes"], f["pool_despues"], f["imgs_quitadas"],
            f["lesiones_quitadas"], f["hermanas_despues"], pct)
        if "meta" in f:
            s += "  target {:5d} deviation {:+d}".format(f["meta"], f["imgs_quitadas"] - f["meta"])
        if "base_e1" in f:
            s += "  from no-siblings pool {:5d} -> {:5d} ({:.3f})".format(
                f["base_e1"], f["pool_despues"], f["pool_despues"] / float(f["base_e1"]))
        out.append(s)
    return "\n".join(out)
