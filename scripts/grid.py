"""Experimental grid of the thesis and the exact training command of each run.

A run is (seed, init, target, k, method): init ``ham`` (ResNet-50 fine-tuned on HAM10000, ``--pretrained-backbone``)
or ``imagenet``; target ``milk10k``, ``milk10k_clinico`` or ``bcn20000``; k = 16, 32, 100 labeled images per class;
ERM or one of ten SSL methods. Seeds 0-4 give 990 runs; batch size 16 is part of the protocol. ``run-source`` and
``run-fully-supervised`` train the ``erm.py --oracle`` references. Commands run from ``train/``.

Usage: python scripts/grid.py [--data-root data --log-root outputs --backbone checkpoints/ham10000_source.pth]
       {list --seeds 0-4 | show|run SEED INIT TARGET K METHOD | run-source | run-fully-supervised SEED INIT TARGET}
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COMMON = ("-a resnet50 --bottleneck-dim 1024 --finetune --batch-size 16 --wd 0.0005 "
          "--epochs 100 --iters-per-epoch 500 --patience 15")

# Method-specific hyperparameters (TLlib defaults unless stated in Chapter 3).
METHODS = {
    "erm": "--trade-off-cls-strong 0.1 --lr 0.003 --lr-scheduler exp --lr-gamma 0.0004 --lr-decay 0.75",
    "pi_model": "--trade-off-cls-strong 0.1 --trade-off-con 0.1 --lr 0.003 --lr-scheduler exp "
                "--lr-gamma 0.0004 --lr-decay 0.75 --warm-up-epochs 10",
    "mean_teacher": "--alpha 0.999 --trade-off-cls-strong 0.1 --trade-off-con 0.1 --lr 0.01 "
                    "--lr-scheduler exp --lr-gamma 0.0001 --lr-decay 0.75 --warm-up-epochs 10",
    "debiasmatch": "--momentum 0.999 --tau 1 --trade-off-cls-strong 0.1 --trade-off-self-training 1 "
                   "--threshold 0.95 --lr 0.003 --lr-scheduler exp --lr-gamma 0.0004 --lr-decay 0.75",
    "fixmatch": "--trade-off-cls-strong 0.1 --trade-off-self-training 1 --threshold 0.95 --lr 0.003 "
                "--lr-scheduler exp --lr-gamma 0.0004 --lr-decay 0.75",
    "noisy_student": "--trade-off-cls-strong 0.1 --trade-off-self-training 1 --T 2 --lr 0.003 "
                     "--lr-scheduler exp --lr-gamma 0.0004 --lr-decay 0.75",
    "pseudo_label": "--trade-off-cls-strong 0.1 --trade-off-self-training 1 --threshold 0.95 --lr 0.003 "
                    "--lr-scheduler exp --lr-gamma 0.0004 --lr-decay 0.75",
    "uda": "--trade-off-cls-strong 0.1 --trade-off-con 1 --threshold 0.7 --T 0.85 --lr 0.003 "
           "--lr-scheduler exp --lr-gamma 0.0004 --lr-decay 0.75",
    # threshold warm-up of the original FlexMatch (off by default in the trainer)
    "flexmatch": "--warmup true --trade-off-cls-strong 0.1 --trade-off-self-training 1 --threshold 0.95 "
                 "--lr 0.003 --lr-scheduler exp --lr-gamma 0.0004 --lr-decay 0.75",
    "self_tuning": "--projection-dim 1024 --T 0.07 --K 32 --m 0.999 --lr 0.003 --lr-gamma 0.1 "
                   "--milestones 12 24 36 48",
    "dst": "--width 2048 --trade-off-cls-strong 0.1 --trade-off-self-training 1 --eta 1 --eta-prime 2 "
           "--threshold 0.7 --lr 0.01 --lr-scheduler exp --lr-gamma 0.0002 --lr-decay 0.75",
}

# target -> (dataset flag, data folder, extra transform flags).
# Clinical images use RandomResizedCrop scale (0.5, 1.0) so the crop keeps the lesion.
TARGETS = {
    "milk10k": ("MILK10K", "MILK10K", ""),
    "milk10k_clinico": ("MILK10K", "MILK10K_CLINICO", "--train-resizing lesion_preserving"),
    "bcn20000": ("BCN20000", "BCN20000", ""),
}
INITS = ["ham", "imagenet"]
KS = [16, 32, 100]


def defaults(args):
    return (args.data_root or os.path.join(ROOT, "data"),
            args.log_root or os.path.join(ROOT, "outputs"),
            args.backbone or os.path.join(ROOT, "checkpoints", "ham10000_source.pth"))


def build_argv(seed, init, target, k, method, data_root, log_root, backbone):
    dflag, folder, extra = TARGETS[target]
    logdir = f"{log_root}/{init}/{target}/k{k}/{method}/s{seed}"
    argv = [sys.executable, f"{method}.py", f"{data_root}/{folder}", "-d", dflag,
            "--num-samples-per-class", str(k)] + COMMON.split()
    if init == "ham":
        argv += ["--pretrained-backbone", backbone]
    argv += extra.split() + METHODS[method].split()
    if method == "noisy_student":
        # The teacher is the ERM run of the same seed, initialization, target and k,
        # so ERM has to finish first. Without a teacher the run reduces to ERM.
        argv += ["--pretrained-teacher", f"{log_root}/{init}/{target}/k{k}/erm/s{seed}/checkpoints/best.pth"]
    argv += ["--seed", str(seed), "--log", logdir]
    return argv


def build_source(data_root, log_root):
    # HAM10000 classifier (seed 0). It was trained with 200 iterations per epoch.
    return [sys.executable, "erm.py", f"{data_root}/HAM10000", "-d", "HAM10000", "--oracle",
            "-a", "resnet50", "--bottleneck-dim", "1024", "--finetune", "--batch-size", "16",
            "--wd", "0.0005", "--epochs", "100", "--iters-per-epoch", "200", "--patience", "15",
            *METHODS["erm"].split(), "--seed", "0", "--log", f"{log_root}/source_ham10000"]


def build_fully_supervised(seed, init, target, data_root, log_root, backbone):
    dflag, folder, extra = TARGETS[target]
    argv = [sys.executable, "erm.py", f"{data_root}/{folder}", "-d", dflag, "--oracle"] + COMMON.split()
    if init == "ham":
        argv += ["--pretrained-backbone", backbone]
    return argv + extra.split() + METHODS["erm"].split() + [
        "--seed", str(seed), "--log", f"{log_root}/fully_supervised/{init}/{target}/s{seed}"]


def parse_seeds(text):
    out = []
    for part in str(text).split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out


def execute(argv):
    print(" ".join(argv), flush=True)
    sys.exit(subprocess.call(argv, cwd=os.path.join(ROOT, "train")))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root")
    ap.add_argument("--log-root")
    ap.add_argument("--backbone")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list")
    ls.add_argument("--seeds", default="0-4")
    for name in ("show", "run"):
        p = sub.add_parser(name)
        p.add_argument("seed", type=int)
        p.add_argument("init", choices=INITS)
        p.add_argument("target", choices=list(TARGETS))
        p.add_argument("k", type=int, choices=KS)
        p.add_argument("method", choices=list(METHODS))
    sub.add_parser("run-source")
    fs = sub.add_parser("run-fully-supervised")
    fs.add_argument("seed", type=int)
    fs.add_argument("init", choices=INITS)
    fs.add_argument("target", choices=list(TARGETS))
    args = ap.parse_args()
    data_root, log_root, backbone = defaults(args)

    if args.cmd == "list":
        for seed in parse_seeds(args.seeds):
            for init in INITS:
                for target in TARGETS:
                    for k in KS:
                        for method in METHODS:
                            print(" ".join(build_argv(seed, init, target, k, method,
                                                      data_root, log_root, backbone)))
    elif args.cmd in ("show", "run"):
        argv = build_argv(args.seed, args.init, args.target, args.k, args.method,
                          data_root, log_root, backbone)
        print(" ".join(argv)) if args.cmd == "show" else execute(argv)
    elif args.cmd == "run-source":
        execute(build_source(data_root, log_root))
    else:
        execute(build_fully_supervised(args.seed, args.init, args.target, data_root, log_root, backbone))


if __name__ == "__main__":
    main()
