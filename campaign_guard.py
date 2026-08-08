#!/usr/bin/env python3
"""
campaign_guard.py — the GUARDED side of the seed-lottery variance claim, 10 training seeds,
single-pass, same cloud pipeline as campaign_lottery.py.

WHY THIS EXISTS
---------------
The paper's central claim is a statement about VARIANCE: the flat controller's sustained-RLF
scatters across identical trainings (sigma = 4.18) while the measurement-guarded branched
controller does not (sigma = 0.92). The Results in Engineering AE objected that "the evidence base
is limited to five training seeds". A variance claim at n = 5 can be displayed but not tested.
n = 10 per arm makes Brown-Forsythe / Levene and a bootstrap CI on the sigma RATIO reportable.

BOTH ARMS MUST COME FROM ONE PIPELINE
-------------------------------------
campaign_lottery.py trains the flat arm at 10 seeds, single-pass, in this container. This driver
trains the guarded arm at 10 seeds, single-pass, in this container. Neither reuses the published
laptop numbers (Table X) or the chunked cold-restart numbers (results/srdndqn_postguard.csv, the
Section IX-K reproduction) INSIDE the tested sample. Both of those stay untouched and serve as
independent cross-pipeline reproduction checks.

Mixing pipelines inside a sample whose VARIANCE is the headline number would confound training-seed
variance with pipeline variance. That is not acceptable in a variance paper.

PROVENANCE DISCIPLINE
---------------------
train_srdndqn.py is NOT modified. This driver imports its train() and evaluate() so the algorithm
code path is literally the same objects. Arm is 'postguard' == the published proposed controller
(branched binary action space + 9 dB measured state guard, backup shield OFF).

Each seed records its own `pipeline` value, `<venue>-<continuity>`:
    venue       cloud | laptop — set by the PIPELINE_TAG environment variable, default cloud. The
                two venues run different PyTorch builds (laptop 2.5.1, cloud 2.13.0) and produce
                measurably different policies from the same seed, so venue is DATA, not metadata.
    continuity  singlepass — trained start-to-finish in one process (the intended condition)
                resumed    — a recycle forced a mid-seed restart from the .resume checkpoint, i.e.
                             a cold restart with a warm-restarting replay buffer. Recorded, not
                             hidden; the analysis script reports how many seeds carry it.

Usage:
    python -u campaign_guard.py 0,1,2,3,4        # shard A
    python -u campaign_guard.py 5,6,7,8,9        # shard B (safe to run concurrently: separate CSV)
    python -u campaign_guard.py 0-9 800000
    PIPELINE_TAG=laptop-singlepass python -u campaign_guard.py 3   # venue-tagged single-seed shard
"""
import sys, os, time, csv

_SEED_ARG = sys.argv[1] if len(sys.argv) > 1 else '0-9'
_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 800_000

if '-' in _SEED_ARG:
    a, b = _SEED_ARG.split('-')
    SEEDS = list(range(int(a), int(b) + 1))
else:
    SEEDS = [int(x) for x in _SEED_ARG.split(',')]

sys.argv = ['train_srdndqn.py', 'postguard', str(len(SEEDS)), str(_STEPS)]

import torch
torch.set_num_threads(1)                       # paper protocol: single-threaded CPU

import numpy as np
import train_srdndqn as S                      # <-- identical train()/evaluate()
from dndqn_branched import load_branched
from dndqn import OBS_DIM
import train_phase2 as tp

ARM = 'postguard'
SHARD = f"s{SEEDS[0]}_{SEEDS[-1]}"

# VENUE is the machine+toolchain half of the pipeline tag; the continuity half (singlepass/resumed)
# can only be known per seed at runtime, so it is appended there. Accepting a full tag like
# "laptop-singlepass" keeps ONE env var with ONE meaning across all four campaign drivers.
VENUE = os.environ.get('PIPELINE_TAG', 'cloud-singlepass').split('-')[0]
_VSUF = '' if VENUE == 'cloud' else f'_{VENUE}'          # cloud paths stay byte-identical

ROWS_PATH = f'results/lottery_guard_800k_{SHARD}{_VSUF}.csv'
HEARTBEAT = f'campaign_guard_{SHARD}{_VSUF}.heartbeat'
FIELDS = ['seed', 'steps', 'arm', 'pipeline', 'HO', 'Power', 'Spec', 'RLF', 'Override',
          'Tput', 'train_min']


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
                done.add((int(r['seed']), int(r['steps'])))
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
    beat(f"CAMPAIGN guard | arm={ARM} (branched binary + {S.GUARD_DB:.0f} dB measured state guard, "
         f"save line {S.SAVE_LINE:.0f} dBm) | seeds {SEEDS} | {_STEPS:,} steps | venue={VENUE} "
         f"-> {ROWS_PATH}")
    beat(f"toolchain | python {sys.version.split()[0]} | torch {torch.__version__} | numpy {np.__version__}")
    slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH)
    pool = slow + med
    beat(f"data | slow={len(slow)} med={len(med)} fast(held-out)={len(fast)} skipped={skip} pool={len(pool)}")

    rows, done = load_done()
    todo = [s for s in SEEDS if (s, _STEPS) not in done]
    beat(f"plan | {len(done)} already complete, {len(todo)} to train")

    for seed in todo:
        ck = f"gd_s{seed}.pt"                              # own namespace: never touches srdndqn_*
        resume_existed = os.path.exists(ck + '.resume')    # honest pipeline label
        beat(f"train START {ARM} seed {seed} -> {ck}" + (" (RESUMING mid-seed)" if resume_existed else ""))
        finished, mins = S.train(seed, ARM, ck, slow, med, _STEPS)   # wall=None -> single pass
        if not finished:
            beat(f"train INCOMPLETE {ARM} s{seed} — unexpected early return; will retry on restart")
            continue
        met = S.evaluate(load_branched(ck, OBS_DIM)[0], pool, ARM)
        rows.append({'seed': seed, 'steps': _STEPS, 'arm': ARM,
                     'pipeline': f"{VENUE}-resumed" if resume_existed else f"{VENUE}-singlepass",
                     'HO': f"{met['HO']:.2f}", 'Power': f"{met['Power']:.2f}",
                     'Spec': f"{met['Spec']:.2f}", 'RLF': f"{met['RLF']:.2f}",
                     'Override': f"{met.get('Override', 0.0):.2f}",
                     'Tput': f"{met['Tput']:.1f}", 'train_min': f"{mins:.1f}"})
        rows.sort(key=lambda r: int(r['seed']))
        write_rows(rows)
        beat(f"train DONE  {ARM} s{seed}: RLF {met['RLF']:.2f}% Override {met.get('Override',0.0):.2f}% "
             f"HO {met['HO']:.2f}% Pwr {met['Power']:.2f}% Spec {met['Spec']:.2f}% ({mins:.1f}m)")

    beat("ALL SEEDS COMPLETE — shard summary follows")
    v = {k: [float(r[k]) for r in rows] for k in ['HO', 'Power', 'Spec', 'RLF', 'Override', 'Tput']}
    print("=" * 84)
    print(f"  GUARDED branched Dn-DQN — mean +/- population sigma over {len(v['RLF'])} TRAINING seeds "
          f"(shard {SHARD})")
    print("=" * 84)
    for k in ['HO', 'Power', 'Spec', 'RLF', 'Override', 'Tput']:
        print(f"  {k:>9} : {np.mean(v[k]):8.2f} +/- {np.std(v[k]):.2f}   "
              f"[min {min(v[k]):.2f}, max {max(v[k]):.2f}]")
    n_res = sum(1 for r in rows if not r['pipeline'].endswith('-singlepass'))
    print(f"\n  pipeline | {len(rows) - n_res} single-pass, {n_res} resumed mid-seed")
    print(f"  published references: branched-guarded 4.61 +/- 0.92 (laptop) | "
          f"postguard 4.08 +/- 0.42 (cloud chunked)")
    print(f"  -> {ROWS_PATH}")


if __name__ == "__main__":
    main()
