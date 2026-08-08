"""Q5 summary: throughput-floor arm vs the published guarded arm.

PART 0 is an equivalence GATE, not a formality. Q5's evaluation used the published
evaluate_branched, which resolves BranchedEnv6G in its own module globals -- exactly the path that
silently broke Q2. Q5 should be immune, because it changes only the reward and evaluate_branched
discards reward. That is reasoning, and reasoning is what failed on Q2, so it is checked: one
checkpoint through both environments must give bit-identical metrics.
"""
import csv
import sys
import types

import numpy as np
import tablevii_decompose as D

R = 8
SEEDS = [0, 1, 2, 3, 4]
METRICS = ['RLF', 'Power', 'Spec', 'Tput', 'HO']


def gate():
    sys.argv = sys.argv[:1]
    import torch
    torch.set_num_threads(1)
    import env_6g_v5
    import train_phase2 as tp
    import train_branched_nyu as B
    import train_branched_q5 as B5
    from dndqn_branched import load_branched
    from dndqn import OBS_DIM
    import tablevii_r8 as T7
    T7.install_patch(env_6g_v5)
    slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH)
    pool = slow + med
    g = dict(B.evaluate_branched.__globals__)
    g['BranchedEnv6G'] = B5.BranchedEnv6G
    ev_q5 = types.FunctionType(B.evaluate_branched.__code__, g, 'ev_q5',
                               B.evaluate_branched.__defaults__, B.evaluate_branched.__closure__)
    net = load_branched('q5_fl_s0.pt', OBS_DIM)[0]
    with T7.seeded(0):
        a = B.evaluate_branched(net, pool, True, guard_db=9.0, guard_mode='state')
    with T7.seeded(0):
        b = ev_q5(net, pool, True, guard_db=9.0, guard_mode='state')
    same = all(abs(float(a[k]) - float(b[k])) < 1e-9 for k in METRICS)
    print('=' * 92)
    print('GATE -- was Q5 evaluated in the right environment?')
    print('=' * 92)
    for k in METRICS:
        print('  %-6s published-env %12.4f   floor-env %12.4f   %s'
              % (k, a[k], b[k], 'same' if abs(a[k] - b[k]) < 1e-9 else '*** DIFFERS ***'))
    print('  VERDICT: %s' % ('PASS - the floor changes only the reward, which evaluation discards; '
                             'the grid is valid'
                             if same else 'FAIL - Q5 metrics were measured in the wrong env'))
    print('=' * 92)
    return same


def grid(path, pick):
    cell = {}
    for r in csv.DictReader(open(path)):
        if pick(r):
            cell[(int(r['seed']), int(r['repeat']))] = r
    return cell


def arm(cell, m):
    X = np.array([[float(cell[(s, r)][m]) for r in range(R)] for s in SEEDS])
    d = D.decompose(X)
    sd = ('%.2f' % d['sd_train']) if d['resolvable'] else '<%.2f' % d['sd_train_hi']
    return d['grand'], sd, X.mean(axis=1), d


if not gate():
    sys.exit(2)

new = grid('results/q5floor/q5_campaign.csv', lambda r: True)
pub = grid('results/tablevii_r8_laptop.csv', lambda r: r['arm'] == 'guarded')
print()
print('=' * 92)
print('Q5 -- THROUGHPUT-FLOOR ARM (T_floor = 239.90 Mbps, p33) vs PUBLISHED GUARDED ARM')
print('   guarded, G = 9 dB, k = 5 trainings x R = 8 CRN, venue laptop')
print('=' * 92)
print('  %-8s %26s %26s' % ('metric', 'published (no floor)', 'floor arm (Q5)'))
for m in METRICS:
    g1, s1, _, _ = arm(pub, m)
    g2, s2, _, _ = arm(new, m)
    print('  %-8s %16.2f +- %-7s %16.2f +- %-7s' % (m, g1, s1, g2, s2))

print()
print('  per-training means -- did the lottery narrow?')
for m in ('Spec', 'Tput', 'RLF', 'Power'):
    _, _, a, _ = arm(pub, m)
    _, _, b, _ = arm(new, m)
    print('   %-5s no floor  %s   [range %7.2f]'
          % (m, '  '.join('%8.2f' % v for v in a), a.max() - a.min()))
    print('   %-5s floor     %s   [range %7.2f]'
          % ('', '  '.join('%8.2f' % v for v in b), b.max() - b.min()))

print()
print('  per-step throughput of the trained floor policies:')
for r in csv.DictReader(open('results/q5floor/q5_trained_tput.csv')):
    print('    s%s  mean %9.1f Mbps   p25 %8.1f   below floor %5.1f%%'
          % (r['seed'], float(r['mean_mbps']), float(r['p25']), float(r['below_floor_pct'])))
print('    (published arm, pooled: mean 465.7 Mbps, p25 203.8, 33%% below by construction)')
print('=' * 92)
