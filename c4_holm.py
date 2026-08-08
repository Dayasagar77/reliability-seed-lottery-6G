#!/usr/bin/env python3
"""
c4_holm.py -- what the retrained BDQ arm does to the comparison, with and without Holm.

The published branched-incremental row mixed two training origins. Retraining all five seeds under
one protocol changed it on every axis, so every statistic that involved that arm has to be redone:
the CRN-corrected Welch test, the effect size, and the family-wise correction reviewer 2 asked for.

Both arms are compared on the SAME CRN blocks (T7.seeded(r), r = 0..7), so the correction term
welch_pair estimates from the per-block arm-mean difference is still valid across the two files.
"""
import csv
import sys

import numpy as np

sys.path.insert(0, '.')
import tablevii_decompose as D                                          # noqa: E402

R = 8
SEEDS = [0, 1, 2, 3, 4]
METRICS = ['HO', 'Power', 'Spec', 'RLF', 'Tput']


def load_arm(path, arm_filter, name, data, store):
    rows = [r for r in csv.DictReader(open(path)) if arm_filter(r)]
    if not rows:
        return False
    data[name], store[name] = {}, {}
    for m in METRICS:
        cell = {(int(r['seed']), int(r['repeat'])): float(r[m]) for r in rows}
        X = np.array([[cell[(s, r)] for r in range(R)] for s in SEEDS])
        data[name][m] = X
        store[name][m] = D.decompose(X)
    return True


def holm(pairs, alpha=0.05):
    order = sorted(range(len(pairs)), key=lambda i: pairs[i][1])
    n, adj, run = len(pairs), [0.0] * len(pairs), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (n - rank) * pairs[i][1]))
        adj[i] = run
    return [(pairs[i][0], pairs[i][1], adj[i], adj[i] < alpha) for i in range(n)]


data, store = {}, {}
T7 = 'results/tablevii_r8_laptop.csv'
ok = all([
    load_arm(T7, lambda r: r['arm'] == 'guarded', 'guarded', data, store),
    load_arm(T7, lambda r: r['arm'] == 'incremental', 'inc_published', data, store),
    load_arm('results/bdq5/bdq5.csv', lambda r: True, 'inc_retrained', data, store),
])
if not ok:
    print('missing an arm'); sys.exit(2)

print('=' * 100)
print('C4 -- THE BDQ ARM, PUBLISHED (MIXED ORIGIN) vs RETRAINED (ONE ORIGIN)')
print('=' * 100)
print('  %-7s %26s %26s' % ('metric', 'published', 'retrained'))
for m in METRICS:
    a, b = store['inc_published'][m], store['inc_retrained'][m]
    sa = ('%.2f' % a['sd_train']) if a['resolvable'] else '<%.2f' % a['sd_train_hi']
    sb = ('%.2f' % b['sd_train']) if b['resolvable'] else '<%.2f' % b['sd_train_hi']
    print('  %-7s %16.2f +- %-7s %16.2f +- %-7s   sigma %.2fx'
          % (m, a['grand'], sa, b['grand'], sb,
             (b['sd_train'] / a['sd_train']) if a['sd_train'] else float('nan')))

for label, arm in (('PUBLISHED (mixed origin)', 'inc_published'),
                   ('RETRAINED (one origin)', 'inc_retrained')):
    print()
    print('-' * 100)
    print('proposed (guarded) vs BDQ fine dial -- %s' % label)
    print('-' * 100)
    ps, det = [], {}
    for m in METRICS:
        r = D.welch_pair(data, store, 'guarded', arm, m)
        ps.append((m, float(r['p_corr'])))
        det[m] = r
    res = holm(ps)
    print('  %-6s %10s %10s %9s %11s %10s  %s'
          % ('metric', 'proposed', 'BDQ', 'd', 'p (CRN)', 'p (Holm)', 'verdict'))
    for m, p, padj, rej in res:
        d = det[m]
        print('  %-6s %10.2f %10.2f %9.2f %11.4f %10.4f  %s%s'
              % (m, d['mean_a'], d['mean_b'], d['d'], p, padj,
                 'reject' if rej else 'not rejected',
                 '   <-- survives Holm' if rej else ''))

print()
print('=' * 100)
print('WHAT CHANGES IN THE MANUSCRIPT')
print('=' * 100)
gp, ip, ir = store['guarded'], store['inc_published'], store['inc_retrained']
print('  spectrum saving, BDQ : %.2f -> %.2f   (proposed %.2f)'
      % (ip['Spec']['grand'], ir['Spec']['grand'], gp['Spec']['grand']))
print('     the published gap of %.1f points becomes %.1f -- any claim that BDQ-style branching'
      % (gp['Spec']['grand'] - ip['Spec']['grand'], gp['Spec']['grand'] - ir['Spec']['grand']))
print('     gives up spectrum was an artefact of mixed provenance and must be withdrawn.')
print('  RLF sigma_train, BDQ : %.2f -> %.2f   (proposed <%.2f)'
      % (ip['RLF']['sd_train'], ir['RLF']['sd_train'], gp['RLF']['sd_train_hi']))
print('     the variance-collapse ratio against BDQ falls from ~%.0fx to ~%.0fx -- still an order'
      % (ip['RLF']['sd_train'] / gp['RLF']['sd_train_hi'],
         ir['RLF']['sd_train'] / gp['RLF']['sd_train_hi']))
print('     of magnitude, but the headline number changes.')
print('  power saving, BDQ    : %.2f -> %.2f   (proposed %.2f)'
      % (ip['Power']['grand'], ir['Power']['grand'], gp['Power']['grand']))
print('     the power-axis separation GROWS, from %.1f to %.1f points.'
      % (gp['Power']['grand'] - ip['Power']['grand'],
         gp['Power']['grand'] - ir['Power']['grand']))
print('=' * 100)
