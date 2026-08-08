#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
f1_lr_sweep.py - is the retraining lottery a LEARNING-RATE artefact?

THE THREAT. The paper's premise is that identically configured trainings of the flat
controller span a 3.4-fold range in sustained RLF, and attributes that to the geometry of
the action space against a measured power-reliability cliff. No learning-rate sweep exists.
If sigma_train(flat) collapses at a different learning rate, the lottery is a tuning
artefact and the premise dissolves. This is the standard referee reflex AFTER "just fix
the seed" - and that one is already answered, so this is the next thing they reach for.

THE TEST. Retrain the flat arm at lr in {1e-4, 1e-3}, 5 seeds each, everything else held
identical to the published protocol. The published lr = 3e-4 arm already exists from the
A1 campaign (results/a1_selection, sigma_train(RLF) = 4.908 at k=5, R=8), measured under
the same pipeline on the same day, so it is the correct comparator.

PRE-COMMITTED READING (fixed before the run):
  * sigma_train stays large at both lrs        -> the lottery is not a tuning artefact. Premise holds.
  * sigma_train falls materially at some lr    -> the claim NARROWS to "at a standard
                                                  hyperparameter setting", and that must be said.
  * sigma_train collapses to guarded levels    -> the premise dissolves. Report it.

PROVENANCE. No published source file is modified. run_multiseed_rel.train() is called
unmodified; only the Adam constructor is wrapped to inject the learning rate, and the
network builder is wrapped to capture the final weights (same mechanism as a1_dual_checkpoint).

Usage:
    python -u f1_lr_sweep.py 1e-4 0
    python -u f1_lr_sweep.py 1e-3 4
"""
import sys, os, time, csv, hashlib

LR   = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-4
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 800_000
TAG = f"{LR:.0e}".replace("-", "m").replace("+", "p")   # 1e-04 -> 1em04

import torch
torch.set_num_threads(1)
import numpy as np
import train_phase2 as tp

OUTDIR = "results/f1_lr"
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)
LOG = os.path.join("logs", f"f1_lr_{TAG}_s{SEED}.log")


def say(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    say(f"F1 LR SWEEP | lr={LR:g} seed={SEED} steps={STEPS:,}")
    say(f"toolchain | python {sys.version.split()[0]} | torch {torch.__version__} | numpy {np.__version__}")
    slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH)
    say(f"data | slow={len(slow)} med={len(med)} fast(held-out)={len(fast)} pool={len(slow)+len(med)}")

    saved_argv = sys.argv[:]
    sys.argv = ["run_multiseed_rel.py", "1", str(STEPS)]
    import importlib
    import run_multiseed_rel as M
    importlib.reload(M)
    sys.argv = saved_argv

    # inject the learning rate without editing the published trainer
    orig_adam = M.optim.Adam

    def adam_with_lr(params, *a, **kw):
        kw["lr"] = LR
        return orig_adam(params, *a, **kw)

    # capture the live network so the final-step weights can be saved too
    captured = {}
    orig_builder = M.build_dndqn_networks

    def builder(*a, **kw):
        online, target = orig_builder(*a, **kw)
        captured["online"] = online
        return online, target

    M.optim.Adam = adam_with_lr
    M.build_dndqn_networks = builder

    best_path  = os.path.join(OUTDIR, f"f1_flat_{TAG}_best_s{SEED}.pt")
    final_path = os.path.join(OUTDIR, f"f1_flat_{TAG}_final_s{SEED}.pt")
    say(f"train START flat lr={LR:g} s{SEED}")
    try:
        mins = M.train(SEED, best_path, slow, med)
    finally:
        M.optim.Adam = orig_adam
        M.build_dndqn_networks = orig_builder

    online = captured.get("online")
    if online is None:
        raise RuntimeError("network not captured")
    torch.save({"q_net_state": online.state_dict(), "obs_dim": M.OBS_DIM,
                "n_actions": M.N_ACTIONS, "net_arch": M.NET_ARCH, "algorithm": "Dn-DQN",
                "n_steps": M.NSTEP, "seed": SEED, "lr": LR,
                "training_mode": "nyu_reliability_tilt"}, final_path)

    man = os.path.join(OUTDIR, "f1_manifest.csv")
    new = not os.path.exists(man)
    with open(man, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["lr", "tag", "seed", "steps", "selection", "ckpt", "md5", "train_min", "utc"])
        utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for sel, p in (("best", best_path), ("final", final_path)):
            w.writerow([LR, TAG, SEED, STEPS, sel, os.path.basename(p), md5(p), f"{mins:.1f}", utc])
    say(f"train DONE  lr={LR:g} s{SEED} in {mins:.1f}m -> {os.path.basename(best_path)} / {os.path.basename(final_path)}")


if __name__ == "__main__":
    main()
