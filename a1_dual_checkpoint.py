#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
a1_dual_checkpoint.py - is the reported between-training variance an artefact of the
                        best-checkpoint save rule?

THE CONFOUND. Every training in this project saves the checkpoint with the best 20-episode
moving-average reward, computed on an UNSEEDED channel. An argmax over a noisy criterion
injects a run-specific positive bias whose magnitude scales with how much that criterion
fluctuates. The flat arm's reward fluctuates far more than the guarded arm's. The pipeline
may therefore inject more selection noise into exactly the arm that reports more
between-training variance - which is sufficient, on its own, to produce the headline.

THE TEST. Train each arm normally, but keep TWO checkpoints from the SAME run:
    <arm>_best_s<k>.pt    the usual best-moving-average checkpoint  (unchanged rule)
    <arm>_final_s<k>.pt   the weights at the last training step     (no selection at all)
Then evaluate both sets and compare sigma_train. No extra training cost: one run, two saves.

    if the flat-vs-guarded sigma_train gap SURVIVES under final-checkpoint saving
        -> the confound is not driving the result; report as a positive control
    if the gap SHRINKS materially
        -> part of the headline is a selection artefact; must be disclosed and re-derived

PROVENANCE. No existing source file is modified. The algorithm code path is literally the
same objects: this driver imports train() from the published modules and wraps only the
NETWORK BUILDER, so it holds a reference to the same `online` network the trainer mutates
in place. After train() returns, that object carries the FINAL weights.

Usage:
    python -u a1_dual_checkpoint.py flat   0-4
    python -u a1_dual_checkpoint.py guard  0-4
    python -u a1_dual_checkpoint.py both   0-4  [steps]
"""
import sys, os, time, csv, hashlib

ARM_ARG  = (sys.argv[1] if len(sys.argv) > 1 else "both").lower()
SEED_ARG = sys.argv[2] if len(sys.argv) > 2 else "0-4"
STEPS    = int(sys.argv[3]) if len(sys.argv) > 3 else 800_000

if "-" in SEED_ARG:
    a, b = SEED_ARG.split("-"); SEEDS = list(range(int(a), int(b) + 1))
else:
    SEEDS = [int(x) for x in SEED_ARG.split(",")]

import torch
torch.set_num_threads(1)                       # paper protocol: single-threaded CPU
import numpy as np
import train_phase2 as tp

OUTDIR = "results/a1_selection"
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)
LOG = os.path.join("logs", f"a1_dual_{ARM_ARG}_{SEEDS[0]}_{SEEDS[-1]}.log")


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


# ----------------------------------------------------------------------------------
# FLAT arm  -  run_multiseed_rel.train(seed, out, slow, med)
# ----------------------------------------------------------------------------------
def run_flat(seed, slow, med):
    saved_argv = sys.argv[:]
    sys.argv = ["run_multiseed_rel.py", "1", str(STEPS)]
    import importlib
    import run_multiseed_rel as M
    importlib.reload(M)                         # pick up STEPS from argv
    sys.argv = saved_argv

    captured = {}
    orig_builder = M.build_dndqn_networks

    def wrapped(*a, **kw):
        online, target = orig_builder(*a, **kw)
        captured["online"] = online             # same object the trainer mutates in place
        return online, target

    M.build_dndqn_networks = wrapped
    best_path  = os.path.join(OUTDIR, f"a1_flat_best_s{seed}.pt")
    final_path = os.path.join(OUTDIR, f"a1_flat_final_s{seed}.pt")
    try:
        mins = M.train(seed, best_path, slow, med)
    finally:
        M.build_dndqn_networks = orig_builder

    online = captured.get("online")
    if online is None:
        raise RuntimeError("did not capture the flat network - builder was not called")
    # replicate run_multiseed_rel._save() exactly, so the published evaluator can load it
    torch.save({"q_net_state": online.state_dict(), "obs_dim": M.OBS_DIM,
                "n_actions": M.N_ACTIONS, "net_arch": M.NET_ARCH, "algorithm": "Dn-DQN",
                "n_steps": M.NSTEP, "seed": seed,
                "training_mode": "nyu_reliability_tilt"}, final_path)
    return mins, best_path, final_path


# ----------------------------------------------------------------------------------
# GUARDED arm  -  train_srdndqn.train(seed, arm, out, slow, med, steps)
# ----------------------------------------------------------------------------------
def run_guard(seed, slow, med):
    saved_argv = sys.argv[:]
    sys.argv = ["train_srdndqn.py", "postguard", "1", str(STEPS)]
    import train_srdndqn as S
    sys.argv = saved_argv

    captured = {}
    orig_builder = S.build_branched_networks

    def wrapped(*a, **kw):
        online, target = orig_builder(*a, **kw)
        captured["online"] = online
        return online, target

    S.build_branched_networks = wrapped
    best_path  = os.path.join(OUTDIR, f"a1_guard_best_s{seed}.pt")
    final_path = os.path.join(OUTDIR, f"a1_guard_final_s{seed}.pt")
    try:
        done, mins = S.train(seed, "postguard", best_path, slow, med, STEPS)
    finally:
        S.build_branched_networks = orig_builder

    online = captured.get("online")
    if online is None:
        raise RuntimeError("did not capture the branched network - builder was not called")
    S.save_branched(online, final_path,
                    {"seed": seed, "binary_power": True, "arm": "postguard",
                     "backup_shield": S.ARMS["postguard"][1], "save_line_dbm": S.SAVE_LINE,
                     "algorithm": "SR-Dn-DQN", "selection": "final-step (no selection)"})
    return mins, best_path, final_path


def main():
    say(f"A1 DUAL-CHECKPOINT | arm={ARM_ARG} seeds={SEEDS} steps={STEPS:,}")
    say(f"toolchain | python {sys.version.split()[0]} | torch {torch.__version__} | numpy {np.__version__}")
    slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH)
    say(f"data | slow={len(slow)} med={len(med)} fast(held-out)={len(fast)} skipped={skip} pool={len(slow)+len(med)}")

    arms = ["flat", "guard"] if ARM_ARG == "both" else [ARM_ARG]
    manifest = os.path.join(OUTDIR, "a1_manifest.csv")
    new = not os.path.exists(manifest)
    fh = open(manifest, "a", newline="", encoding="utf-8")
    w = csv.writer(fh)
    if new:
        w.writerow(["arm", "seed", "steps", "selection", "ckpt", "md5", "train_min", "utc"])

    for arm in arms:
        for seed in SEEDS:
            say(f"train START {arm} s{seed} ({STEPS:,} steps)")
            t0 = time.time()
            if arm == "flat":
                mins, bp, fp = run_flat(seed, slow, med)
            elif arm == "guard":
                mins, bp, fp = run_guard(seed, slow, med)
            else:
                say(f"unknown arm {arm}"); continue
            utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            for sel, p in (("best", bp), ("final", fp)):
                if os.path.exists(p):
                    w.writerow([arm, seed, STEPS, sel, os.path.basename(p), md5(p), f"{mins:.1f}", utc])
                else:
                    say(f"  MISSING {sel} checkpoint at {p}")
            fh.flush()
            say(f"train DONE  {arm} s{seed} in {mins:.1f}m -> best={os.path.basename(bp)} "
                f"final={os.path.basename(fp)} (elapsed {(time.time()-t0)/60:.1f}m)")

    fh.close()
    say(f"ALL DONE. manifest -> {manifest}")
    say("NEXT: evaluate both checkpoint sets at R=8 CRN and compare sigma_train "
        "(best vs final) within each arm.")


if __name__ == "__main__":
    main()
