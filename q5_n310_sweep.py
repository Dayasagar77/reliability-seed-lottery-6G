#!/usr/bin/env python3
"""
q5_n310_sweep.py -- reviewer 2, Q5: is the result robust to the N310 persistence criterion?

    "The paper uses N310 = 3 and argues that N310 = 1 is an artifact. What happens at N310 = 2 or
     N310 = 5? Is the cliff shape and the lottery structure robust to this choice?"

Evaluation only: the five guarded checkpoints are frozen and re-evaluated under each criterion.
No retraining, so this measures how the REPORTED metric responds to its own definition, which is
what the question asks.

TWO GATES, BOTH LEARNED THE HARD WAY IN THIS PROJECT.

  1. N310 = 3 must reproduce the published guarded arm. It is the same checkpoints, the same guard
     and the same CRN seeds, so anything else means the harness is not what produced Table VII.
  2. The three settings must actually DIFFER. env_6g_branched does `from env_6g_v6 import
     N310_DEFAULT` and then passes `n310=N310_DEFAULT` inside __init__, so patching env_6g_v6
     after import would change nothing and the sweep would return three identical columns that
     look like beautiful robustness. This is the same shape as the array-gain constant that would
     have made the M2 sweep report a flat line. The patch target is env_6g_branched, and the gate
     fails loudly if the columns come out identical.
"""
import csv
import os
import sys
import time

SEEDS = [0, 1, 2, 3, 4]
LEVELS = [2, 3, 5]
R = 8
GUARD_DB, GUARD_MODE = 9.0, 'state'
METRICS = ['HO', 'Power', 'Spec', 'RLF', 'Tput']
OUT = os.path.join('results', 'n310')
PUBLISHED_RLF_N310_3 = 4.91


def log(m):
    print(time.strftime('%H:%M:%S') + ' | ' + m, flush=True)


def main():
    sys.argv = sys.argv[:1]
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import env_6g_v5
    import env_6g_branched as EB
    import train_phase2 as tp
    from train_branched_nyu import evaluate_branched
    from dndqn_branched import load_branched
    from dndqn import OBS_DIM
    import tablevii_r8 as T7
    import tablevii_decompose as D
    D.CSV = 'results/tablevii_r8_laptop.csv'
    T7.install_patch(env_6g_v5)

    orig = EB.N310_DEFAULT
    log('env_6g_branched.N310_DEFAULT = %r (patch target)' % orig)

    slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH)
    pool = slow + med
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, 'n310.csv')
    rows = list(csv.DictReader(open(path))) if os.path.exists(path) else []
    have = {(int(r['n310']), int(r['seed']), int(r['repeat'])) for r in rows}

    try:
        for n in LEVELS:
            EB.N310_DEFAULT = n
            if EB.N310_DEFAULT != n:
                log('*** patch did not take ***')
                return 2
            for s in SEEDS:
                ck = 'br_gst_s%d.pt' % s
                if not os.path.exists(ck):
                    log('missing %s' % ck)
                    return 3
                if all((n, s, r) in have for r in range(R)):
                    continue
                net, _ = load_branched(ck, OBS_DIM)
                for r in range(R):
                    if (n, s, r) in have:
                        continue
                    with T7.seeded(r):
                        m = evaluate_branched(net, pool, True, guard_db=GUARD_DB,
                                              guard_mode=GUARD_MODE)
                    row = {'n310': n, 'seed': s, 'repeat': r, 'ckpt': ck,
                           'guard_db': GUARD_DB, 'venue': 'laptop',
                           'measured_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
                    for k in METRICS:
                        row[k] = '%.4f' % float(m[k])
                    rows.append(row)
                with open(path, 'w', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)
                log('  N310=%d s%d done | RLF %.2f' % (n, s, float(rows[-1]['RLF'])))
    finally:
        EB.N310_DEFAULT = orig

    # ---------------------------------------------------------------- report + gates
    cells = {(int(r['n310']), int(r['seed']), int(r['repeat'])): r for r in rows}
    summary = {}
    for n in LEVELS:
        summary[n] = {}
        for m in METRICS:
            X = np.array([[float(cells[(n, s, r)][m]) for r in range(R)] for s in SEEDS])
            summary[n][m] = D.decompose(X)

    print()
    print('=' * 96)
    print('Q5 -- SENSITIVITY TO THE N310 PERSISTENCE CRITERION')
    print('   frozen guarded policies, G = 9 dB, 5 trainings x 8 CRN blocks')
    print('=' * 96)
    print('  %-8s %22s %22s %22s' % ('N310', 'sustained RLF (%)', 'power saving (%)',
                                     'spectrum saving (%)'))
    for n in LEVELS:
        cellstr = []
        for m in ('RLF', 'Power', 'Spec'):
            d = summary[n][m]
            sd = ('%.2f' % d['sd_train']) if d['resolvable'] else '<%.2f' % d['sd_train_hi']
            cellstr.append('%12.2f +- %-7s' % (d['grand'], sd))
        print('  %-8s %s' % ('%d%s' % (n, ' (published)' if n == 3 else ''), ' '.join(cellstr)))

    print()
    ok = True
    got3 = summary[3]['RLF']['grand']
    d3 = abs(got3 - PUBLISHED_RLF_N310_3)
    print('  GATE 1  N310=3 reproduces the published guarded arm: %.2f vs %.2f  %s'
          % (got3, PUBLISHED_RLF_N310_3, 'PASS' if d3 < 0.05 else '*** FAIL (%.2f off) ***' % d3))
    ok &= d3 < 0.05
    rlfs = [summary[n]['RLF']['grand'] for n in LEVELS]
    spread = max(rlfs) - min(rlfs)
    print('  GATE 2  the three settings differ (patch took): RLF spread %.3f points  %s'
          % (spread, 'PASS' if spread > 1e-6 else '*** FAIL - identical columns ***'))
    ok &= spread > 1e-6
    print()
    print('  RLF by criterion: %s' % ', '.join('N310=%d -> %.2f%%' % (n, summary[n]['RLF']['grand'])
                                               for n in LEVELS))
    print('  power/spectrum are unchanged by construction: the criterion only reclassifies')
    print('  episodes, it does not alter the controller or the channel.')
    print('=' * 96)
    print('  VERDICT: %s' % ('PASS' if ok else '*** FAIL ***'))
    return 0 if ok else 4


if __name__ == '__main__':
    sys.exit(main())
