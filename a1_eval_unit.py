#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
a1_eval_unit.py - evaluate ONE A1 checkpoint (arm, selection, seed) at R = 8 CRN.

PROVENANCE. This does NOT reimplement the evaluation. It imports tablevii_r8 and calls its
run_unit(), so the evaluation loop, the CRN channel-seed patch, the eval seeds, the episode
count and the metric definitions are literally the published objects. The ONLY things
overridden are:
    * ckpt_for()  -> points at the A1 dual-checkpoint files instead of the published ones
    * ARMS        -> four keys added: flat_best, flat_final, guard_best, guard_final
    * OUTDIR      -> a separate directory, so nothing published is touched

Usage:
    python -u a1_eval_unit.py flat_best 0
    python -u a1_eval_unit.py guard_final 3
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

CKPT_DIR = "results/a1_selection"
T.OUTDIR = "results/a1_selection_eval"
os.makedirs(T.OUTDIR, exist_ok=True)

# four new arms: (label, binary_power, guard_db, guard_mode, evaluator family)
T.ARMS.update({
    "flat_best":   ("A1 flat, best-checkpoint",      None, None, None,    "flat"),
    "flat_final":  ("A1 flat, final-step",           None, None, None,    "flat"),
    "guard_best":  ("A1 guarded, best-checkpoint",   True, 9.0, "state",  "branched"),
    "guard_final": ("A1 guarded, final-step",        True, 9.0, "state",  "branched"),
})

_MAP = {
    "flat_best":   ("a1_flat_best_s{s}.pt",   "a1-best-20260808"),
    "flat_final":  ("a1_flat_final_s{s}.pt",  "a1-final-20260808"),
    "guard_best":  ("a1_guard_best_s{s}.pt",  "a1-best-20260808"),
    "guard_final": ("a1_guard_final_s{s}.pt", "a1-final-20260808"),
}

_orig_ckpt_for = T.ckpt_for


def _ckpt_for(arm, seed):
    if arm in _MAP:
        fn, origin = _MAP[arm]
        return os.path.join(CKPT_DIR, fn.format(s=seed)), origin
    return _orig_ckpt_for(arm, seed)


T.ckpt_for = _ckpt_for

if __name__ == "__main__":
    if ARM not in _MAP:
        print(f"unknown arm {ARM}; expected one of {sorted(_MAP)}")
        sys.exit(2)
    rc = T.run_unit(ARM, SEED)
    sys.exit(rc or 0)
