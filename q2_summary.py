"""Q2 summary: three-level actuator vs the published binary arm, same venue, same CRN seeds.

Uses the paper's own decomposition (tablevii_decompose.decompose) so the +- reported here is the
same between-training component reported everywhere else. No new measurement -- CSV arithmetic.
"""
import csv
import numpy as np
import tablevii_decompose as D

R = 8
SEEDS = [0, 1, 2, 3, 4]
METRICS = ['RLF', 'Power', 'Spec', 'Tput', 'HO']
LBL = ['PROTECT', 'SAVE', 'HANDOVER', 'MID']


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
    return d['grand'], sd, X.mean(axis=1)


new = grid('results/q2_3level/q2_3level.csv', lambda r: True)
pub = grid('results/tablevii_r8_laptop.csv', lambda r: r['arm'] == 'guarded')
print('cells: three-level %d, published binary %d' % (len(new), len(pub)))
print()
print('=' * 92)
print('Q2 -- THREE-LEVEL ACTUATOR (23 / 26.5 / 30 dBm) vs PUBLISHED BINARY (23 / 30 dBm)')
print('   guarded arm, G = 9 dB, k = 5 trainings x R = 8 CRN, venue laptop')
print('=' * 92)
print('  %-8s %26s %26s' % ('metric', 'binary (published)', 'three-level (Q2)'))
for m in METRICS:
    g1, s1, _ = arm(pub, m)
    g2, s2, _ = arm(new, m)
    print('  %-8s %16.2f +- %-7s %16.2f +- %-7s' % (m, g1, s1, g2, s2))

print()
print('  per-training means (the lottery, seed by seed):')
for m in ('RLF', 'Power', 'Spec'):
    _, _, a = arm(pub, m)
    _, _, b = arm(new, m)
    print('   %-6s binary      %s' % (m, '  '.join('%7.2f' % v for v in a)))
    print('   %-6s three-level %s' % ('', '  '.join('%7.2f' % v for v in b)))

print()
print('=' * 92)
print('  ACTION INTENT OF THE TRAINED POLICY  (what the net asked for, before the guard)')
print('=' * 92)
print('  %-6s %10s %10s %10s %10s %12s' % ('seed', 'PROTECT', 'SAVE', 'HANDOVER', 'MID', 'guard ovr'))
rows = list(csv.DictReader(open('results/q2_3level/q2_actions.csv')))
mids = []
for r in rows:
    tot = sum(int(r['a%d' % i]) for i in range(4))
    pct = [100.0 * int(r['a%d' % i]) / tot for i in range(4)]
    mids.append(pct[3])
    print('  s%-5s %9.1f%% %9.1f%% %9.1f%% %9.1f%% %11.1f%%'
          % (r['seed'], pct[0], pct[1], pct[2], pct[3],
             100.0 * int(r['guard_overrode_intent']) / tot))
print()
print('  MID usage across the five trainings: %s' % ', '.join('%.1f%%' % v for v in sorted(mids)))
print('  range %.1f pp   |   trainings never using MID: %d of 5'
      % (max(mids) - min(mids), sum(1 for v in mids if v < 0.5)))
print('=' * 92)
