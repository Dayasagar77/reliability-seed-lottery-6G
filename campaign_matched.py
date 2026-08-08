#!/usr/bin/env python3
"""
campaign_matched.py — MATCHED-OBJECTIVE comparison at 5 TRAINING seeds x 800k steps.

WHY THIS EXISTS
---------------
Results in Engineering AE (RINENG-D-26-14706, 1 Aug 2026):
    "some baselines are represented by a single trained model and others by evaluation-seed
     variability, making the comparisons methodologically inconsistent"

That is correct. Table X mixes (a) single trainings with +/- across 5 EVALUATION seeds for the
learned baselines against (b) +/- across 5 TRAINING seeds for the proposed controller. This
campaign removes the inconsistency by retraining every learned method at 5 TRAINING seeds under
one identical objective and one identical budget, so every learned row in the revised comparison
carries training-seed sigma.

PROVENANCE DISCIPLINE
---------------------
run_multiseed_matched.py is NOT modified. It stays byte-identical as the record of what produced
the published 3-seed / 400k table (results/matched_rel_nyu.csv). This driver IMPORTS its train(),
evaluate() and load_net() so the algorithm code path is literally the same objects, and only
changes what must change:

  1. seeds 0..4 instead of 0..2
  2. 800,000 steps instead of 400,000 (matches every other headline result in the paper, so the
     matched study stops being the one study on a different budget)
  3. a `steps` column in the output CSV, and a resume key of (method, seed, steps), so 400k and
     800k rows can never be silently pooled
  4. separate checkpoint names (m5_*) and a separate results file, so nothing overwrites the
     published artefacts
  5. append-on-completion CSV writes + a heartbeat line per seed, so a container recycle costs at
     most the seed in flight

Nothing about the algorithm, reward, channel, network, evaluation protocol or metric definitions
is changed. Single-variable change: number of training seeds (and the budget alignment above).

Usage:
    python -u campaign_matched.py                 # seeds 0-4, all 3 methods, 800k steps, resumable
    python -u campaign_matched.py 5 800000        # same, explicit (what the cloud watchdog launches)
    python -u campaign_matched.py dndqn:0 800000  # SHARD: one method, one seed, own CSV
    python -u campaign_matched.py ddqn:0,1 800000 # SHARD: one method, listed seeds
    python -u campaign_matched.py all:3-4 800000  # SHARD: every method, seeds 3 and 4
"""
import sys, os, time, csv

# ---------------------------------------------------------------------------
# WORK SELECTION — two accepted forms so one driver serves both venues.
#
#   bare integer      -> a seed COUNT over all methods, canonical ledger, byte-identical to before.
#                        The cloud watchdog launches this form; do not change it.
#   method:seedspec   -> a SHARD. Its own CSV and heartbeat, so up to 15 shards (3 methods x 5
#                        seeds) run concurrently without touching the same file. Checkpoint names
#                        are already unique per (method, seed), so they cannot collide either.
#
# Sharding changes only WHICH units a process does, never WHAT a training is: train() still calls
# set_seed(seed) and builds fresh networks, so a given (method, seed) yields the same policy
# whether it ran alone or alongside siblings.
# ---------------------------------------------------------------------------
_ARG1 = sys.argv[1] if len(sys.argv) > 1 else '5'
_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 800_000

if ':' in _ARG1:
    _MSEL, _SEEDSPEC = _ARG1.split(':', 1)
    if '-' in _SEEDSPEC:
        _a, _b = _SEEDSPEC.split('-')
        SEEDS = list(range(int(_a), int(_b) + 1))
    else:
        SEEDS = [int(x) for x in _SEEDSPEC.split(',')]
    SHARD = f"_{_MSEL}_s{SEEDS[0]}_{SEEDS[-1]}"
else:
    _MSEL = 'all'
    SEEDS = list(range(int(_ARG1)))
    SHARD = ''

_N_SEEDS = len(SEEDS)
# run_multiseed_matched reads N_SEEDS/STEPS from sys.argv at import time; set them before import.
sys.argv = ['run_multiseed_matched.py', str(_N_SEEDS), str(_STEPS)]

import torch
torch.set_num_threads(1)                       # paper protocol: single-threaded CPU

import numpy as np
import run_multiseed_matched as M              # <-- identical train()/evaluate()/load_net()
import train_phase2 as tp

assert M.N_SEEDS == _N_SEEDS and M.STEPS == _STEPS, "argv override did not take effect"

# Restrict to the selected method(s) AFTER import, so the key->label map comes from the unmodified
# published module rather than being duplicated here.
METHODS = M.METHODS if _MSEL == 'all' else [(l, m) for l, m in M.METHODS if m == _MSEL]
assert METHODS, f"unknown method key {_MSEL!r}; known keys: {[m for _, m in M.METHODS]}"

if SHARD:
    ROWS_PATH = f'results/matched_rel_{_STEPS // 1000}k{SHARD}.csv'
    HEARTBEAT = f'campaign_matched{SHARD}.heartbeat'
else:
    ROWS_PATH = f'results/matched_rel_{_N_SEEDS}seed_{_STEPS // 1000}k.csv'
    HEARTBEAT = 'campaign_matched.heartbeat'

# Which machine + toolchain produced the row. The two venues run different PyTorch builds (laptop
# 2.5.1, cloud 2.13.0) and give measurably different policies from the same seed, so venue is DATA.
# Legacy rows written before this column existed read back as '' — honestly blank, never guessed.
PIPELINE = os.environ.get('PIPELINE_TAG', 'cloud-singlepass')
FIELDS = ['method', 'seed', 'steps', 'pipeline', 'HO', 'Power', 'Spec', 'RLF', 'Tput', 'train_min']


def beat(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
    print(line, flush=True)
    with open(HEARTBEAT, 'a') as f:
        f.write(line + '\n')


def load_done():
    """Resume key is (method, seed, steps) so budgets can never be pooled by accident.

    The key deliberately excludes `pipeline`: a ledger is per-venue (own CSV per shard, own
    container), so re-keying on venue would only ever cause needless retraining. Rows written
    before the pipeline column existed are read back with a blank value rather than a guessed one.
    """
    rows = []
    done = set()
    if os.path.exists(ROWS_PATH):
        with open(ROWS_PATH, newline='') as f:
            for r in csv.DictReader(f):
                r.setdefault('pipeline', '')       # pre-instrumentation row: blank, not invented
                rows.append(r)
                done.add((r['method'], int(r['seed']), int(r['steps'])))
    return rows, done


def write_rows(rows):
    tmp = ROWS_PATH + '.tmp'
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, ROWS_PATH)                 # atomic: a crash mid-write cannot corrupt the file


def main():
    os.makedirs('results', exist_ok=True)
    beat(f"CAMPAIGN matched | seeds {SEEDS} x {len(METHODS)} method(s) {[m for _, m in METHODS]} "
         f"| {_STEPS:,} steps | reliability-tilted reward | pipeline={PIPELINE} -> {ROWS_PATH}")
    beat(f"toolchain | python {sys.version.split()[0]} | torch {torch.__version__} | numpy {np.__version__}")
    slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH)
    pool = slow + med
    beat(f"data | slow={len(slow)} med={len(med)} fast(held-out)={len(fast)} skipped={skip} pool={len(pool)}")

    rows, done = load_done()
    todo = [(lbl, m, s) for lbl, m in METHODS for s in SEEDS
            if (lbl, s, _STEPS) not in done]
    beat(f"plan | {len(done)} already complete, {len(todo)} to train")

    for label, method, seed in todo:
        ck = f"m5_{method}_s{seed}.pt"
        beat(f"train START {label} seed {seed} -> {ck}")
        t0 = time.time()
        M.train(method, seed, ck, slow, med)
        mins = (time.time() - t0) / 60.0
        met = M.evaluate(M.load_net(method, ck), pool)
        rows.append({'method': label, 'seed': seed, 'steps': _STEPS, 'pipeline': PIPELINE,
                     'HO': f"{met['HO']:.2f}", 'Power': f"{met['Power']:.2f}",
                     'Spec': f"{met['Spec']:.2f}", 'RLF': f"{met['RLF']:.2f}",
                     'Tput': f"{met['Tput']:.1f}", 'train_min': f"{mins:.1f}"})
        write_rows(rows)
        beat(f"train DONE  {label} s{seed}: HO {met['HO']:.2f}% Pwr {met['Power']:.2f}% "
             f"Spec {met['Spec']:.2f}% RLF {met['RLF']:.2f}% Tput {met['Tput']:.0f} ({mins:.1f}m)")

    # ---- summary (population-form sigma, matching the manuscript's stated convention) ----
    beat("ALL SEEDS COMPLETE — summary follows")
    print("=" * 100)
    print(f"  MATCHED @ RELIABLE CORNER — mean +/- population sigma over {_N_SEEDS} TRAINING seeds "
          f"({_STEPS:,} steps)")
    print("=" * 100)
    print(f"  {'method':>16} | {'HOsucc%':>13} | {'Power%':>13} | {'Spec%':>13} | {'RLF%':>13} | {'Tput':>12}")
    print("  " + "-" * 94)
    for label, method in METHODS:
        v = {k: [float(r[k]) for r in rows if r['method'] == label and int(r['steps']) == _STEPS]
             for k in ['HO', 'Power', 'Spec', 'RLF', 'Tput']}
        if not v['HO']:
            continue
        g = lambda a: (float(np.mean(a)), float(np.std(a)))   # np.std = population form
        ho, pw, sp, rl, tp_ = g(v['HO']), g(v['Power']), g(v['Spec']), g(v['RLF']), g(v['Tput'])
        print(f"  {label:>16} | {ho[0]:6.2f}+/-{ho[1]:4.2f} | {pw[0]:6.2f}+/-{pw[1]:4.2f} | "
              f"{sp[0]:6.2f}+/-{sp[1]:4.2f} | {rl[0]:6.2f}+/-{rl[1]:4.2f} | {tp_[0]:6.1f}+/-{tp_[1]:4.1f} "
              f"(n={len(v['HO'])})")
    print(f"\n  -> {ROWS_PATH}")


if __name__ == "__main__":
    main()
