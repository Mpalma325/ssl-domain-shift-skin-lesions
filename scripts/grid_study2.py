"""Training commands of the Study 2 networks (Chapter 5 and Appendix 7.4) trained with train/.

A run is (seed, target, k, name). Inputs: checkpoints/ham10000_source.pth (RH, as in grid.py) and
checkpoints/ham10000_convnext.pth (CH, the HAM10000 ConvNeXt-T). Teachers and initializations are the best.pth
of earlier runs in this list (erm_teacher first, then convnext_erm, first_student_repeat). The MILK10k pool
exclusion is data/splits/milk10k_pool_excluded.txt; the labeled sets of the selection study are in
results/study2/selection_sets/. Commands run from
train/ and write to outputs/study2/<name>/<target>/k<k>/s<seed>/.

Usage: python scripts/grid_study2.py [--data-root data --log-root outputs --backbone ... --convnext ...]
       {list [--name NAME] | show|run SEED TARGET K NAME}
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grid  # noqa: E402

ROOT = grid.ROOT
LISTS = os.path.join(ROOT, "results", "study2", "selection_sets")
POOL_EXCLUDED = os.path.join(ROOT, "data", "splits", "milk10k_pool_excluded.txt")
HAM_COUNTS = "210,231,612,447,4084"  # HAM10000 train images per class (source prior of the BCTS)
CNX = "convnext_tiny_ham"


class Paths:
    def __init__(self, args):
        self.data, self.logs, self.rh = grid.defaults(args)
        self.ch = args.convnext or os.path.join(ROOT, "checkpoints", "ham10000_convnext.pth")

    def run_dir(self, name, target, k, seed):
        return f"{self.logs}/study2/{name}/{target}/k{k}/s{seed}"

    def ckpt(self, name, target, seed, k=16):
        return f"{self.run_dir(name, target, k, seed)}/checkpoints/best.pth"


def base(p, trainer, target, k, arch="resnet50", init="ham", method=None, oracle=False):
    dflag, folder, extra = grid.TARGETS[target]
    argv = [sys.executable, f"{trainer}.py", f"{p.data}/{folder}", "-d", dflag, "--num-samples-per-class", str(k)]
    argv += grid.COMMON.replace("-a resnet50", f"-a {arch}").split()
    if oracle:
        argv.append("--oracle")
    if init == "ham":
        argv += ["--pretrained-backbone", p.rh if arch == "resnet50" else p.ch]
    return argv + extra.split() + grid.METHODS[method or trainer].split()


def pool_exclusion(target):
    # MILK10k images that copy a HAM10000 photograph or another pool image (the semi-supervised runs on MILK10k)
    return ["--pool-exclude", POOL_EXCLUDED] if target.startswith("milk") else []


def product_of_experts(p, teacher):
    return ["--pretrained-teacher", teacher, "--teacher-view", "center", "--teacher-combine", "poe",
            "--teacher-temp", "4", "--source-checkpoint", p.rh, "--source-prior-counts", HAM_COUNTS]


def four_experts(p, fourth, erm_weight="0.5"):
    return ["--third-checkpoint", p.ch, "--fourth-teacher", fourth, "--fourth-teacher-arch", CNX,
            "--erm-weight", erm_weight]


def teacher(p, target, seed, name="erm_teacher"):
    return p.ckpt(name, target, seed)


# name -> (cells as (target, k, seeds), builder(p, target, k, seed) -> argv without --seed/--log)
S5, S10, S20 = range(5), range(10), range(20)
DERM_CLIN = "milk10k_clinico"


def erm(p, t, k, s):
    return base(p, "erm", t, k)


def erm_imagenet(p, t, k, s):
    return base(p, "erm", t, k, init="imagenet")


def convnext_erm(p, t, k, s):
    return base(p, "erm", t, k, arch=CNX)


def convnext_fully_supervised(p, t, k, s):
    return base(p, "erm", t, k, arch=CNX, oracle=True)


def first_student(p, t, k, s):
    return base(p, "noisy_student", t, k) + product_of_experts(p, teacher(p, t, s)) + pool_exclusion(t)


def student_erm_teacher_only(p, t, k, s):
    return base(p, "noisy_student", t, k) + ["--pretrained-teacher", teacher(p, t, s), "--teacher-view", "center"] \
        + pool_exclusion(t)


def continue_student(trainer, exclusion=False):
    def build(p, t, k, s):
        return base(p, trainer, t, k) + ["--init-student", p.ckpt("first_student_repeat", t, s)] \
            + (pool_exclusion(t) if exclusion else [])
    return build


def convnext_student(p, t, k, s):
    return base(p, "noisy_student", t, k, arch=CNX) + product_of_experts(p, teacher(p, t, s)) \
        + ["--teacher-arch", "resnet50"] + four_experts(p, p.ckpt("convnext_erm", t, s)) + pool_exclusion(t)


def convnext_student_per_lesion(p, t, k, s):
    return convnext_student(p, t, k, s) + ["--teacher-lesion-mean", os.path.join(ROOT, "data", "metadata",
                                                                                 "bcn20000.csv"), "--labeled-sisters"]


def clinical(teacher_flags):
    """Student on the clinical photographs, teacher on the paired dermoscopic images."""
    def build(p, t, k, s):
        return base(p, "noisy_student", t, k) + teacher_flags(p, s) \
            + ["--teacher-root", f"{p.data}/{grid.TARGETS['milk10k'][1]}"] + pool_exclusion(t)
    return build


def derm_re(p, s):
    return ["--pretrained-teacher", teacher(p, "milk10k", s), "--teacher-view", "center"]


def derm_re_imagenet(p, s):
    return ["--pretrained-teacher", teacher(p, "milk10k", s, "erm_teacher_imagenet"), "--teacher-view", "center"]


def derm_rh(p, s):
    return derm_re(p, s) + ["--source-checkpoint", p.rh, "--source-prior-counts", HAM_COUNTS, "--source-alpha", "1"]


def derm_re_rh(p, s):
    return product_of_experts(p, teacher(p, "milk10k", s))


def derm_four(p, s):
    return derm_re_rh(p, s) + four_experts(p, p.ckpt("convnext_erm", "milk10k", s)) + ["--teacher-test-dump"]


def clinical_derm_four_experts_convnext(p, t, k, s):
    argv = clinical(derm_four)(p, t, k, s)
    argv[argv.index("-a") + 1] = CNX
    argv[argv.index("--pretrained-backbone") + 1] = p.ch
    return argv + ["--teacher-arch", "resnet50"]


def clinical_four_experts(p, t, k, s):
    """Floor: the four experts built on the clinical photographs themselves."""
    return base(p, "noisy_student", t, k) + product_of_experts(p, teacher(p, t, s)) \
        + four_experts(p, p.ckpt("convnext_erm", t, s)) + ["--teacher-test-dump"] + pool_exclusion(t)


def clinical_mean_teacher(p, t, k, s):
    return base(p, "mean_teacher", t, k) + pool_exclusion(t)


def pairs_shuffled(p, t, k, s):
    return clinical(derm_re_rh)(p, t, k, s) + ["--pair-shuffle", "class", "--pair-seed", str(s)]


def pairs_third(p, t, k, s):
    return clinical(derm_re_rh)(p, t, k, s) + ["--pair-fraction", "0.3333333333", "--photo-teacher",
                                               teacher(p, t, s), "--pair-seed", str(s)]


def regularized(reg):
    def build(p, t, k, s):
        return base(p, "erm", t, k) + ["--reg", reg] + (["--cotuning-source", p.rh] if reg == "cotuning" else [])
    return build


def labeled_list(selection):
    def build(p, t, k, s):
        return base(p, "erm", t, k) + ["--labeled-list", os.path.join(LISTS, f"{t}_{selection}_s{s}.json")]
    return build


def val16(builder, teacher_name=None):
    def build(p, t, k, s):
        argv = builder(p, t, k, s)
        if teacher_name:
            argv[argv.index("--pretrained-teacher") + 1] = teacher(p, t, s, teacher_name)
        return argv + ["--val-per-class", "16"]
    return build


DM_BCN = ("milk10k", "bcn20000")
RUNS = {
    # ERM teachers (RE), the same command as the ERM of grid.py
    "erm_teacher": ([("bcn20000", 16, S5), ("milk10k", 16, S10), (DERM_CLIN, 16, S10)], erm),
    "erm_teacher_imagenet": ([("milk10k", 16, S5)], erm_imagenet),
    # ConvNeXt-T ERM from CH (CE); the repeat at k = 16 is the second CE of the four-ERM ensemble
    "convnext_erm": ([(t, 16, S5) for t in DM_BCN + (DERM_CLIN,)] + [(t, k, S5) for k in (32, 100) for t in DM_BCN],
                     convnext_erm),
    "convnext_erm_repeat": ([(t, 16, S5) for t in DM_BCN], convnext_erm),
    "convnext_fully_supervised": ([(t, 16, S5) for t in DM_BCN], convnext_fully_supervised),
    # students
    "first_student": ([("milk10k", 16, S10), ("bcn20000", 16, S5), (DERM_CLIN, 16, S5)], first_student),
    "first_student_repeat": ([(t, 16, S5) for t in DM_BCN], first_student),
    "student_then_erm": ([(t, 16, S5) for t in DM_BCN], continue_student("erm")),
    "student_then_mean_teacher": ([("milk10k", 16, S5)], continue_student("mean_teacher")),
    "student_then_fixmatch": ([("bcn20000", 16, S5)], continue_student("fixmatch")),
    "convnext_student": ([(t, 16, S5) for t in DM_BCN], convnext_student),
    "convnext_student_per_lesion": ([("bcn20000", 16, S5)], convnext_student_per_lesion),
    "student_erm_teacher_only": ([("milk10k", 16, S10), ("bcn20000", 16, S5), (DERM_CLIN, 16, S5)],
                                 student_erm_teacher_only),
    # clinical photographs, dermoscopic teachers
    "clinical_derm_re": ([(DERM_CLIN, 16, S5)], clinical(derm_re)),
    "clinical_derm_re_imagenet": ([(DERM_CLIN, 16, S5)], clinical(derm_re_imagenet)),
    "clinical_derm_rh": ([(DERM_CLIN, 16, S5)], clinical(derm_rh)),
    "clinical_derm_re_rh": ([(DERM_CLIN, 16, S10)], clinical(derm_re_rh)),
    "clinical_derm_four_experts": ([(DERM_CLIN, 16, S5)], clinical(derm_four)),
    "clinical_derm_four_experts_convnext": ([(DERM_CLIN, 16, S5)], clinical_derm_four_experts_convnext),
    "clinical_four_experts": ([(DERM_CLIN, 16, S5)], clinical_four_experts),
    "clinical_mean_teacher": ([(DERM_CLIN, 16, S10)], clinical_mean_teacher),
    # pairing controls of the clinical student: all pairs, shuffled within class, one third, none (photo teacher)
    "pairs_all": ([(DERM_CLIN, 16, S5)], clinical(derm_re_rh)),
    "pairs_shuffled": ([(DERM_CLIN, 16, S5)], pairs_shuffled),
    "pairs_third": ([(DERM_CLIN, 16, S5)], pairs_third),
    "pairs_none": ([(DERM_CLIN, 16, S5)], first_student),
    # baselines that use the source network
    "l2sp": ([(t, 16, S5) for t in DM_BCN], regularized("l2sp")),
    "cotuning": ([(t, 16, S5) for t in DM_BCN], regularized("cotuning")),
    # label selection: full ERM on the random sets and on class-aware prototype selection
    "erm_random_list": ([(t, 16, S20) for t in DM_BCN], labeled_list("study1")),
    "erm_prototypes_list": ([(t, 16, S10) for t in DM_BCN], labeled_list("class_prototypes")),
    # training epoch chosen on 16 validation images per class
    "erm_teacher_val16": ([(t, 16, S5) for t in DM_BCN + (DERM_CLIN,)], val16(erm)),
    "first_student_val16": ([(t, 16, S5) for t in DM_BCN + (DERM_CLIN,)], val16(first_student, "erm_teacher_val16")),
    "student_erm_teacher_only_val16": ([(t, 16, S5) for t in DM_BCN + (DERM_CLIN,)],
                                       val16(student_erm_teacher_only, "erm_teacher_val16")),
}


def build_argv(p, name, target, k, seed):
    cells, builder = RUNS[name]
    if not any(t == target and kk == k and seed in seeds for t, kk, seeds in cells):
        raise SystemExit(f"{name} was not run for {target} k={k} seed {seed}")
    return builder(p, target, k, seed) + ["--seed", str(seed), "--log", p.run_dir(name, target, k, seed)]


def all_runs(names=None):
    for name in names or RUNS:
        for target, k, seeds in RUNS[name][0]:
            for seed in seeds:
                yield name, target, k, seed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root")
    ap.add_argument("--log-root")
    ap.add_argument("--backbone")
    ap.add_argument("--convnext")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list")
    ls.add_argument("--name", choices=list(RUNS), action="append")
    for cmd in ("show", "run"):
        q = sub.add_parser(cmd)
        q.add_argument("seed", type=int)
        q.add_argument("target", choices=list(grid.TARGETS))
        q.add_argument("k", type=int, choices=grid.KS)
        q.add_argument("name", choices=list(RUNS))
    args = ap.parse_args()
    p = Paths(args)
    if args.cmd == "list":
        for name, target, k, seed in all_runs(args.name):
            print(" ".join(build_argv(p, name, target, k, seed)))
    else:
        argv = build_argv(p, args.name, args.target, args.k, args.seed)
        print(" ".join(argv)) if args.cmd == "show" else grid.execute(argv)


if __name__ == "__main__":
    main()
