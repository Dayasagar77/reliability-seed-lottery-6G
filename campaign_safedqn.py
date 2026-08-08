#!/usr/bin/env python3
"""
campaign_safedqn.py — Shielded Safe-DQN baseline retrained at 5 TRAINING seeds x 800k steps,
in the same cloud pipeline and at the same budget as every other learned row in the revision.

WHY THIS EXISTS (and why it is a RETRAIN, not a top-up)
------------------------------------------------------
results/safe_dqn_nyu.csv holds 3 seeds trained on the author's laptop at 400,000 steps
(train_min ~19, consistent with the 400k default in train_safe_dqn.main()). Two problems if it is
merely topped up to 5 seeds here:

  1. BUDGET. Every other learned row in the revised comparison is retrained at 800k steps, which is
     the budget used for the paper's headline results. Leaving one baseline at 400k reintroduces
     exactly the kind of non-comparability the AE objected to.
  2. PIPELINE. Appending cloud-trained seeds 3-4 to laptop-trained seeds 0-2 would put two pipelines
     inside one reported +/-.

So all 5 seeds are retrained here, single pipeline, single budget. The published 3-seed / 400k set
is kept untouched as an independent cross-pipeline, cross-budget reference.

PROVENANCE DISCIPLINE
---------------------
train_safe_dqn.py is NOT modified. This driver imports its train(), evaluate() and shielded_action()
so the algorithm code path is literally the same objects. The shield stays applied at BOTH training
and evaluation, exactly as in the published baseline.

Usage:
    python -u campaign_safedqn.py             # seeds 0-4, 800k steps, resumable
    python -u campaign_safedqn.py 5 800000    # same, explicit (what the cloud watchdog launches)
    python -u campaign_safedqn.py 3 800000    # NOTE: bare int is a COUNT, not a seed -> seeds 0-2
    python -u campaign_safedqn.py 3,4 800000  # explicit seed LIST -> only seeds 3 and 4, own shard
    python -u campaign_safedqn.py 0-4 800000  # explicit seed RANGE
"""
import sys, os, time, csv

# ---------------------------------------------------------------------------
# SEED SELECTION — two accepted forms so one driver serves both venues.
#
#   bare integer  -> a COUNT (seeds 0..n-1), canonical ledger, byte-identical behaviour to before.
#                    The cloud watchdog launches this form; do not change it.
#   list or range -> a SHARD. Writes its own CSV and heartbeat, so N shards can run concurrently on
#                    N cores without ever touching the same file. merge_shards.py concatenates them.
#
# Sharding changes only WHICH seeds a process does, never WHAT a training is: train() still calls
# set_seed(seed) and builds fresh networks, so seed s yields the same policy whether it ran alone
# or alongside siblings on the same box. Checkpoint names are already unique per seed.
# ---------------------------------------------------------------------------
_SEED_ARG = sys.argv[1] if len(sys.argv) > 1 else '5'
_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 800_000

if ',' in _SEED_ARG or '-' in _SEED_ARG:
    if '-' in _SEED_ARG:
        _a, _b = _SEED_ARG.split('-')
        SEEDS = list(range(int(_a), int(_b) + 1))
    else:
        SEEDS = [int(x) for x in _SEED_ARG.split(',')]
    SHARD = f"_s{SEEDS[0]}_{SEEDS[-1]}"
else:
    SEEDS = list(range(int(_SEED_ARG)))
    SHARD = ''

_N_SEEDS = len(SEEDS)
sys.argv = ['train_safe_dqn.py', str(_N_SEEDS), str(_STEPS)]

import torch
torch.set_num_threads(1)                       # paper protocol: single-threaded CPU

import numpy as np
import train_safe_dqn as SD                    # <-- identical train()/evaluate()/shielded_action()
from dndqn import load_dndqn_checkpoint
import train_phase2 as tp

if SHARD:
    ROWS_PATH = f'results/safedqn_{_STEPS // 1000}k{SHARD}.csv'
    HEARTBEAT = f'campaign_safedqn{SHARD}.heartbeat'
else:
    ROWS_PATH = f'results/safedqn_{_N_SEEDS}seed_{_STEPS // 1000}k.csv'
    HEARTBEAT = 'campaign_safedqn.heartbeat'

# Which machine + toolchain produced the row. This paper's subject is reproducibility, so the venue
# is data, not metadata: it goes in the CSV, not in a README. Default preserves cloud behaviour.
PIPELINE = os.environ.get('PIPELINE_TAG', 'cloud-singlepass')
FIELDS = ['seed', 'steps', 'pipeline', 'HO', 'Power', 'Spec', 'RLF', 'Tput', 'train_min']


def beat(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
    print(line, flush=True)
    with open(HEARTBEAT, 'a') as f:
        f.write(line + '\n')


def load_done():
    rows, done = [], set()
    if os.path.exists(ROWS_PATH):
        with open(ROWS_PATH, newline='') as f:
            for r in csv.DictReader(f):
                rows.append(r)
                done.add((int(r['seed']), int(r['steps']), r['pipeline']))
    return rows, done


def write_rows(rows):
    tmp = ROWS_PATH + '.tmp'
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, ROWS_PATH)                 # atomic


def main():
    os.makedirs('results', exist_ok=True)
    beat(f"CAMPAIGN safedqn | Shielded Safe-DQN (action mask at selection, save line "
         f"{SD.SAVE_LINE:.0f} dBm) | seeds {SEEDS} | {_STEPS:,} steps | pipeline={PIPELINE} "
         f"-> {ROWS_PATH}")
    beat(f"toolchain | python {sys.version.split()[0]} | torch {torch.__version__} | numpy {np.__version__}")
    slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH)
    pool = slow + med
    beat(f"data | slow={len(slow)} med={len(med)} fast(held-out)={len(fast)} skipped={skip} pool={len(pool)}")

    rows, done = load_done()
    todo = [s for s in SEEDS if (s, _STEPS, PIPELINE) not in done]
    beat(f"plan | {len(done)} already complete, {len(todo)} to train")

    for seed in todo:
        ck = f"sd5_safe_s{seed}.pt"             # own namespace: never touches published ms_safe_*
        beat(f"train START Safe-DQN seed {seed} -> {ck}")
        mins = SD.train(seed, ck, slow, med, _STEPS)
        met = SD.evaluate(load_dndqn_checkpoint(ck)[0], pool)
        rows.append({'seed': seed, 'steps': _STEPS, 'pipeline': PIPELINE,
                     'HO': f"{met['HO']:.2f}", 'Power': f"{met['Power']:.2f}",
                     'Spec': f"{met['Spec']:.2f}", 'RLF': f"{met['RLF']:.2f}",
                     'Tput': f"{met['Tput']:.1f}", 'train_min': f"{mins:.1f}"})
        rows.sort(key=lambda r: int(r['seed']))
        write_rows(rows)
        beat(f"train DONE  Safe-DQN s{seed}: HO {met['HO']:.2f}% Pwr {met['Power']:.2f}% "
             f"Spec {met['Spec']:.2f}% RLF {met['RLF']:.2f}% Tput {met['Tput']:.0f} ({mins:.1f}m)")

    beat("ALL SEEDS COMPLETE — summary follows")
    v = {k: [float(r[k]) for r in rows] for k in ['HO', 'Power', 'Spec', 'RLF', 'Tput']}
    print("=" * 84)
    print(f"  SHIELDED Safe-DQN — mean +/- population sigma over {len(v['RLF'])} TRAINING seeds "
          f"({_STEPS:,} steps)")
    print("=" * 84)
    for k in ['HO', 'Power', 'Spec', 'RLF', 'Tput']:
        print(f"  {k:>6} : {np.mean(v[k]):8.2f} +/- {np.std(v[k]):.2f}   "
              f"[min {min(v[k]):.2f}, max {max(v[k]):.2f}]")
    print(f"\n  published laptop reference (3 seeds @ 400k): RLF 6.00 +/- 1.63 — different budget "
          f"AND different pipeline, do not pool")
    print(f"  -> {ROWS_PATH}")


if __name__ == "__main__":
    main()
