#!/usr/bin/env python3
"""
campaign_lottery.py — the SEED LOTTERY at 10 training seeds, all from ONE pipeline.

WHY THIS EXISTS
---------------
Results in Engineering AE (RINENG-D-26-14706, 1 Aug 2026):
    "the evidence base is limited to five training seeds"

Correct, and it bites hardest on the paper's CENTRAL claim, which is a statement about VARIANCE
(five identical trainings of the flat controller span 4.27-16.80% RLF; sigma = 4.18 against the
guarded controller's 0.92). Variance estimates converge far more slowly than means: at n = 5 you
can display a spread but you cannot run a credible test on it. At n = 10 per arm a
Brown-Forsythe / Levene test and a bootstrap CI on the sigma ratio become reportable.

PIPELINE-CONFOUND HAZARD (the reason all 10 seeds are retrained, not just 5-9)
-----------------------------------------------------------------------------
results/multiseed_rel_nyu.csv seeds 0-4 were trained on the author's laptop. Appending cloud-
trained seeds 5-9 to them would mix TRAINING-SEED variance with PIPELINE variance inside the very
sample whose variance is the paper's headline number. That is not acceptable in a variance paper.
All 10 seeds are therefore trained here in one pipeline, and the published laptop 5-seed set is
kept untouched as an independent cross-pipeline reproduction check.

PROVENANCE DISCIPLINE
---------------------
run_multiseed_rel.py is NOT modified. It stays byte-identical as the record of what produced the
published Section V numbers. This driver imports its train() and evaluate() so the algorithm code
path is the same objects, and only changes the seed range, the output path, the checkpoint names
and the provenance columns.

Usage:
    python -u campaign_lottery.py            # seeds 0-9, 800k steps, resumable
    python -u campaign_lottery.py 10 800000  # same, explicit  (what the cloud watchdog launches)
    python -u campaign_lottery.py 3 800000   # NOTE: bare int is a COUNT, not a seed -> seeds 0-2
    python -u campaign_lottery.py 3,4 800000 # explicit seed LIST -> only seeds 3 and 4, own shard
    python -u campaign_lottery.py 0-4 800000 # explicit seed RANGE
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
# or alongside seven siblings on the same box.
# ---------------------------------------------------------------------------
_SEED_ARG = sys.argv[1] if len(sys.argv) > 1 else '10'
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
sys.argv = ['run_multiseed_rel.py', str(_N_SEEDS), str(_STEPS)]

import torch
torch.set_num_threads(1)                       # paper protocol: single-threaded CPU

import numpy as np
import run_multiseed_rel as R                  # <-- identical train()/evaluate()
from dndqn import load_dndqn_checkpoint
import train_phase2 as tp

assert R.N_SEEDS == _N_SEEDS and R.STEPS == _STEPS, "argv override did not take effect"

if SHARD:
    ROWS_PATH = f'results/lottery_flat_{_STEPS // 1000}k{SHARD}.csv'
    HEARTBEAT = f'campaign_lottery{SHARD}.heartbeat'
else:
    ROWS_PATH = f'results/lottery_flat_{_N_SEEDS}seed_{_STEPS // 1000}k.csv'
    HEARTBEAT = 'campaign_lottery.heartbeat'

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
    os.replace(tmp, ROWS_PATH)


def main():
    os.makedirs('results', exist_ok=True)
    beat(f"CAMPAIGN lottery | flat reliability-tilted Dn-DQN | seeds {SEEDS} "
         f"| {_STEPS:,} steps | pipeline={PIPELINE} -> {ROWS_PATH}")
    beat(f"toolchain | python {sys.version.split()[0]} | torch {torch.__version__} | numpy {np.__version__}")
    slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH)
    pool = slow + med
    beat(f"data | slow={len(slow)} med={len(med)} fast(held-out)={len(fast)} skipped={skip} pool={len(pool)}")

    rows, done = load_done()
    todo = [s for s in SEEDS if (s, _STEPS, PIPELINE) not in done]
    beat(f"plan | {len(done)} already complete, {len(todo)} to train")

    for seed in todo:
        ck = f"lot_flat_s{seed}.pt"
        beat(f"train START flat seed {seed} -> {ck}")
        mins = R.train(seed, ck, slow, med)
        met = R.evaluate(load_dndqn_checkpoint(ck)[0], pool)
        rows.append({'seed': seed, 'steps': _STEPS, 'pipeline': PIPELINE,
                     'HO': f"{met['HO']:.2f}", 'Power': f"{met['Power']:.2f}",
                     'Spec': f"{met['Spec']:.2f}", 'RLF': f"{met['RLF']:.2f}",
                     'Tput': f"{met['Tput']:.1f}", 'train_min': f"{mins:.1f}"})
        write_rows(rows)
        beat(f"train DONE  flat s{seed}: HO {met['HO']:.2f}% Pwr {met['Power']:.2f}% "
             f"Spec {met['Spec']:.2f}% RLF {met['RLF']:.2f}% Tput {met['Tput']:.0f} ({mins:.1f}m)")

    beat("ALL SEEDS COMPLETE — summary follows")
    v = {k: [float(r[k]) for r in rows] for k in ['HO', 'Power', 'Spec', 'RLF', 'Tput']}
    print("=" * 80)
    print(f"  FLAT reliability-tilted Dn-DQN — mean +/- population sigma over {len(v['RLF'])} TRAINING seeds")
    print("=" * 80)
    for k in ['HO', 'Power', 'Spec', 'RLF', 'Tput']:
        print(f"  {k:>6} : {np.mean(v[k]):8.2f} +/- {np.std(v[k]):.2f}   "
              f"[min {min(v[k]):.2f}, max {max(v[k]):.2f}]")
    print(f"\n  published laptop 5-seed reference: RLF 11.47 +/- 4.18 (span 4.27-16.80)")
    print(f"  -> {ROWS_PATH}")


if __name__ == "__main__":
    main()
