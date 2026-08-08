#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
f2_guard_lr_sweep.py - does the guard remove sensitivity to the LEARNING RATE, not just to seeds?

WHY THIS RUN EXISTS. F1 showed the flat controller's sustained RLF is strongly learning-rate
dependent: mean 4.62 / 10.38 / 16.13 at lr = 1e-4 / 3e-4 / 1e-3, with sigma_train(RLF)
0.704 / 4.908 / 3.751. A flat controller at lr = 1e-4 therefore reaches 4.62% - better than the
published guarded arm's 4.93% - which defeats any claim that the guard is what achieves that level.

But the direction of the effect matches the paper's own mechanism: sigma_train(Power) moves with it
(5.6 / 19.9 / 14.1), i.e. the learning rate is displacing the controller along the SAME
power-reliability cliff that the seed displaces it along.

THE HYPOTHESIS THIS RUN TESTS, stated before the run:

    H-F2: the guarded controller's sustained RLF is INSENSITIVE to the learning rate.
          Concretely, across lr in {1e-4, 3e-4, 1e-3} its mean RLF stays within ~1 point and its
          sigma_train stays below 0.5, while the flat arm swings 4.62 -> 16.13.

    If H-F2 HOLDS, the contribution generalises: the cliff converts ANY source of training
    variation into reliability variation, and the guard removes sensitivity to the source in
    general - not merely to the random seed. That is strictly stronger than the current claim.

    If H-F2 FAILS - the guarded arm also moves with lr - then the guard does not do what the
    paper says it does, and the claim must be rewritten. Report either outcome.

The lr = 3e-4 guarded arm already exists from the A1 campaign (results/a1_selection,
mean RLF 4.927, sigma_train 0.000 at k=5, R=8), same pipeline, same day. Only 1e-4 and 1e-3 are run.

PROVENANCE. No published source is modified. train_srdndqn.train() is called unmodified; only the
Adam constructor is wrapped to inject lr, and the network builder is wrapped to capture the final
weights (same mechanism as a1_dual_checkpoint.py).

Usage:
    python -u f2_guard_lr_sweep.py 1e-4 0
"""
import sys, os, time, csv, hashlib

LR    = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-4
SEED  = int(sys.argv[2]) if len(sys.argv) > 2 else 0
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 800_000
TAG   = f"{LR:.0e}".replace("-", "m").replace("+", "p")

import torch
torch.set_num_threads(1)
import numpy as np
import train_phase2 as tp

OUTDIR = "results/f2_guard_lr"
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)
LOG = os.path.join("logs", f"f2_guard_{TAG}_s{SEED}.log")


def say(m):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {m}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    say(f"F2 GUARDED LR SWEEP | lr={LR:g} seed={SEED} steps={STEPS:,}")
    say(f"toolchain | python {sys.version.split()[0]} | torch {torch.__version__} | numpy {np.__version__}")
    slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH)
    say(f"data | pool={len(slow)+len(med)} (slow={len(slow)} med={len(med)} held-out fast={len(fast)})")

    saved = sys.argv[:]
    sys.argv = ["train_srdndqn.py", "postguard", "1", str(STEPS)]
    import train_srdndqn as S
    sys.argv = saved

    orig_adam = S.optim.Adam

    def adam_with_lr(params, *a, **kw):
        kw["lr"] = LR
        return orig_adam(params, *a, **kw)

    captured = {}
    orig_builder = S.build_branched_networks

    def builder(*a, **kw):
        online, target = orig_builder(*a, **kw)
        captured["online"] = online
        return online, target

    S.optim.Adam = adam_with_lr
    S.build_branched_networks = builder

    best_path  = os.path.join(OUTDIR, f"f2_guard_{TAG}_best_s{SEED}.pt")
    final_path = os.path.join(OUTDIR, f"f2_guard_{TAG}_final_s{SEED}.pt")
    say(f"train START postguard lr={LR:g} s{SEED} | guard {S.GUARD_DB:.0f} dB, save line {S.SAVE_LINE:.0f} dBm")
    try:
        done, mins = S.train(SEED, "postguard", best_path, slow, med, STEPS)
    finally:
        S.optim.Adam = orig_adam
        S.build_branched_networks = orig_builder

    online = captured.get("online")
    if online is None:
        raise RuntimeError("network not captured")
    S.save_branched(online, final_path,
                    {"seed": SEED, "binary_power": True, "arm": "postguard",
                     "backup_shield": S.ARMS["postguard"][1], "save_line_dbm": S.SAVE_LINE,
                     "algorithm": "SR-Dn-DQN", "lr": LR, "selection": "final-step"})

    man = os.path.join(OUTDIR, "f2_manifest.csv")
    new = not os.path.exists(man)
    with open(man, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["lr", "tag", "seed", "steps", "selection", "ckpt", "md5", "train_min", "utc"])
        utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for sel, p in (("best", best_path), ("final", final_path)):
            w.writerow([LR, TAG, SEED, STEPS, sel, os.path.basename(p), md5(p), f"{mins:.1f}", utc])
    say(f"train DONE  lr={LR:g} s{SEED} in {mins:.1f}m")


if __name__ == "__main__":
    main()
