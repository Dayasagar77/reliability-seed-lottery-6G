#!/usr/bin/env python3
"""
review_M4_gsweep.py -- M4: make the step from "sigma = 8.2 dB is measured" to "G = 9 dB" visible.

THE REVIEWER'S POINT, WHICH IS CORRECT. Upper-bounding a shadow-fading standard deviation is not
the same as deriving a guard from a target outage probability. If X_sigma ~ N(0, sigma^2) then
P(X_sigma > G) = Q(G/sigma); at G = 9, sigma = 8.2 that is Q(1.098) ~ 13.6%, which is not a small
exceedance. The oracle sweep of Fig. 4c already shows what a larger guard COSTS in saveable
budget; it does not show what a smaller guard costs in RELIABILITY. This supplies that.

WHAT THIS MEASURES, AND WHAT IT DOES NOT. The five checkpoints were TRAINED at G = 9. Evaluating
them at other G measures how a G=9-trained policy behaves when the guard it learned against is
changed underneath it. It does NOT measure what a controller trained at G = 6 would do -- that
needs six retraining campaigns, and the difference matters because the policy has learned to lean
on a particular guard. The manuscript text must say so; a sweep silently read as "trained at each
G" would overclaim. Reported as guard ROBUSTNESS, not as guard SELECTION.

Evaluation-only, R = 8 CRN at the published channel seeds, published evaluator called unmodified.
"""
import csv
import hashlib
import os
import sys
import time

import numpy as np
from scipy import stats

G_GRID = [6.0, 7.0, 8.0, 9.0, 10.0, 12.0]
SEEDS = [0, 1, 2, 3, 4]
R = 8
OUT = 'results/gsweep'
METRICS = ['HO', 'Power', 'Spec', 'RLF', 'Tput']
SIG_NLOS = 8.2


def log(m):
    print(time.strftime('%Y-%m-%d %H:%M:%S') + ' | ' + m, flush=True)


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    saved = sys.argv
    sys.argv = [saved[0]]
    import torch
    torch.set_num_threads(1)
    import env_6g_v5
    import train_phase2 as tp
    import train_branched_nyu as B
    from dndqn_branched import load_branched
    from dndqn import OBS_DIM
    import tablevii_r8 as T7
    sys.argv = saved

    T7.install_patch(env_6g_v5)
    slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH)
    pool = slow + med
    os.makedirs(OUT, exist_ok=True)

    log('M4 GUARD-CONSTANT SWEEP | G in %s | %d seeds x R=%d | pool %d'
        % (G_GRID, len(SEEDS), R, len(pool)))
    log('policies were TRAINED at G = 9.0; this is robustness of a G=9 policy, not selection of G')

    path = os.path.join(OUT, 'gsweep.csv')
    done = set()
    if os.path.exists(path):
        for r in csv.DictReader(open(path)):
            done.add((float(r['guard_db']), int(r['seed']), int(r['repeat'])))
        log('resuming: %d rows already present' % len(done))
    rows = list(csv.DictReader(open(path))) if os.path.exists(path) else []

    for G in G_GRID:
        for s in SEEDS:
            ck = 'br_gst_s%d.pt' % s
            if not os.path.exists(ck):
                log('MISSING %s -- refusing' % ck)
                return 3
            if all((G, s, r) in done for r in range(R)):
                continue
            net = load_branched(ck, OBS_DIM)[0]
            for r in range(R):
                if (G, s, r) in done:
                    continue
                t0 = time.time()
                with T7.seeded(r):
                    m = B.evaluate_branched(net, pool, True, guard_db=G, guard_mode='state')
                row = {'guard_db': G, 'seed': s, 'ckpt': ck, 'ckpt_md5': md5(ck), 'repeat': r,
                       'chan_seed_first': T7.BASE + 100000 * r,
                       'secs': round(time.time() - t0, 1), 'venue': 'cloud-hw-torch2.5.1',
                       'torch': torch.__version__, 'numpy': np.__version__,
                       'env_v5_md5': md5('env_6g_v5.py'),
                       'env_branched_md5': md5('env_6g_branched.py'),
                       'measured_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
                for k in METRICS:
                    row[k] = '%.4f' % float(m[k])
                rows.append(row)
                with open(path + '.tmp', 'w', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)
                os.replace(path + '.tmp', path)
            log('G=%4.1f s%d done | RLF %.2f Power %.2f'
                % (G, s, np.mean([float(x['RLF']) for x in rows
                                  if float(x['guard_db']) == G and int(x['seed']) == s]),
                   np.mean([float(x['Power']) for x in rows
                            if float(x['guard_db']) == G and int(x['seed']) == s])))

    # ---------------------------------------------------------------- report
    sys.path.insert(0, '/home/claude/work')
    import tablevii_decompose as D
    print('\n' + '=' * 96)
    print('M4 -- SUSTAINED RLF AND POWER SAVING vs THE GUARD CONSTANT G')
    print('=' * 96)
    print('  exceedance probability of a Gaussian shadow term, Q(G/sigma) at sigma = %.1f dB:'
          % SIG_NLOS)
    for G in G_GRID:
        print('    G = %4.1f dB  ->  Q(G/sigma) = %6.2f%%' % (G, 100 * stats.norm.sf(G / SIG_NLOS)))
    print()
    print('  %-8s %22s %22s %22s' % ('G (dB)', 'sustained RLF', 'power saving', 'spectrum saving'))
    for G in G_GRID:
        cells = []
        for m in ('RLF', 'Power', 'Spec'):
            cell = {}
            for x in rows:
                if float(x['guard_db']) == G:
                    cell[(int(x['seed']), int(x['repeat']))] = float(x[m])
            X = np.array([[cell[(s, r)] for r in range(R)] for s in SEEDS])
            d = D.decompose(X)
            sd = ('%.2f' % d['sd_train']) if d['resolvable'] else '<%.2f' % d['sd_train_hi']
            cells.append('%8.2f +- %-8s' % (d['grand'], sd))
        print('  %-8.1f %22s %22s %22s' % (G, cells[0], cells[1], cells[2]))
    print('=' * 96)
    return 0


if __name__ == '__main__':
    sys.exit(main())
