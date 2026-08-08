#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
f1_eval_unit.py - evaluate ONE learning-rate-sweep checkpoint at R = 8 CRN.

Same provenance discipline as a1_eval_unit.py: imports tablevii_r8 and calls its run_unit(),
so the evaluation loop, the CRN channel-seed patch, the eval seeds and the metric definitions
are the published objects. Only the checkpoint paths and the output directory are overridden.

Usage:
    python -u f1_eval_unit.py lr1em04_best 0
    python -u f1_eval_unit.py lr1em03_final 4
"""
import os, sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ARM = sys.argv[1]
SEED = int(sys.argv[2])

_saved = sys.argv
sys.argv = [_saved[0]]
import tablevii_r8 as T
sys.argv = _saved

CKPT_DIR = "results/f1_lr"
T.OUTDIR = "results/f1_lr_eval"
os.makedirs(T.OUTDIR, exist_ok=True)

_MAP = {}
for tag, lr in (("1em04", "1e-4"), ("1em03", "1e-3")):
    for sel in ("best", "final"):
        key = f"lr{tag}_{sel}"
        _MAP[key] = (f"f1_flat_{tag}_{sel}_s{{s}}.pt", f"f1-lr{lr}-{sel}-20260808")
        T.ARMS[key] = (f"Flat Dn-DQN, lr={lr}, {sel}-checkpoint", None, None, None, "flat")

_orig = T.ckpt_for


def _ckpt_for(arm, seed):
    if arm in _MAP:
        fn, origin = _MAP[arm]
        return os.path.join(CKPT_DIR, fn.format(s=seed)), origin
    return _orig(arm, seed)


T.ckpt_for = _ckpt_for

if __name__ == "__main__":
    if ARM not in _MAP:
        print(f"unknown arm {ARM}; expected one of {sorted(_MAP)}")
        sys.exit(2)
    sys.exit(T.run_unit(ARM, SEED) or 0)
