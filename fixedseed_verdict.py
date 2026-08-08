#!/usr/bin/env python3
"""
fixedseed_verdict.py -- is the seed a CAUSE or a LABEL?

Compares the between-training dispersion of K flat controllers trained at THE SAME SEED against
the published dispersion of five flat controllers trained at DIFFERENT seeds.

    If sigma_between(fixed seed) ~ sigma_train(varying seed), the declared seed explains little
    of the lottery and the phenomenon is a RETRAINING lottery.

    If sigma_between(fixed seed) ~ 0, the seed really is the driver and the manuscript's original
    wording was right.

Same estimator (md5-pinned tablevii_decompose.decompose), same R = 8 CRN design, same channel
seeds (BASE = 20260801), same evaluator (run_multiseed_rel.evaluate), same pool. The ONLY thing
that differs from the published flat arm is that the five trainings share a seed instead of
having five different ones.

VENUE. Evaluation in this container is bit-exact against the laptop (40/40 cells). Training is
not reproducible anywhere -- that is the finding under test -- so the fixed-seed arm is compared
against the published arm on DISPERSION, not on means. A venue offset adds a constant to every
training run and cancels exactly out of a standard deviation; it does not cancel out of a mean.
That asymmetry is what makes this comparison legitimate across venues and it is the same
argument the R = 8 campaign already rests on.
"""
import csv
import hashlib
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, '/home/claude/work')
EST_MD5 = '4d30eacb161d92df09821ef755569f85'
METRICS = ['HO', 'Power', 'Spec', 'RLF', 'Tput']
R = 8


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


got = md5('/home/claude/work/tablevii_decompose.py')
print('estimator md5 %s %s' % (got, 'OK' if got == EST_MD5 else 'MISMATCH'))
if got != EST_MD5:
    sys.exit(2)
import tablevii_decompose as D                                       # noqa: E402

# ---- fixed-seed arm
import glob
g = defaultdict(dict)
ck = {}
for p in sorted(glob.glob('/home/claude/work/sim/results/fixedseed/unit_fs_r*.csv')):
    for r in csv.DictReader(open(p)):
        k = int(r['replicate'])
        ck[k] = r['ckpt_md5']
        for m in METRICS:
            g[k][(m, int(r['repeat']))] = float(r[m])
ks = sorted(k for k in g if len(g[k]) == len(METRICS) * R)
K = len(ks)
FS = {m: np.array([[g[k][(m, j)] for j in range(R)] for k in ks]) for m in METRICS}
print('fixed-seed replicates complete: %d %s' % (K, ks))
print('checkpoint md5s: %s' % {k: ck[k][:8] for k in ks})
print('distinct checkpoints: %d of %d' % (len({ck[k] for k in ks}), K))
if K < 2:
    sys.exit('not enough replicates yet')

# ---- published varying-seed arms
def grid(path, enc='utf-8'):
    rows = list(csv.DictReader(open(path, encoding=enc)))
    gg = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        for m in METRICS:
            gg[r['arm']][int(r['seed'])][(m, int(r['repeat']))] = float(r[m])
    out = {}
    for arm in gg:
        seeds = sorted(gg[arm])
        out[arm] = {m: np.array([[gg[arm][s][(m, j)] for j in range(R)] for s in seeds])
                    for m in METRICS}
    return out


T = grid('/home/claude/work/results/tablevii_r8_laptop.csv')
L = grid('/home/claude/work/results/lottery_r8_laptop.csv', 'latin-1')

BAR = '=' * 104
print('\n' + BAR)
print('SIGMA BETWEEN TRAININGS -- SAME SEED (k=%d, cloud) vs DIFFERENT SEEDS (published)' % K)
print(BAR)
print('  %-8s %26s %26s %26s' % ('', 'FIXED seed 0 (k=%d)' % K,
                                 'Table VII flat (k=5)', 'ten-training flat (k=10)'))
for m in METRICS:
    row = '  %-8s' % m
    for X in (FS[m], T['flat'][m], L['lot_flat'][m]):
        d = D.decompose(X)
        s = ('%.3f' % d['sd_train']) if d['resolvable'] else '<%.3f (bound)' % d['sd_train_hi']
        row += ' %12.2f  %-12s' % (d['grand'], s)
    print(row)
print('\n  (each cell: mean, then the between-training sigma or its 97.5%% upper bound)')

print('\n' + BAR)
print('THE RELIABILITY AXIS, WHICH IS WHERE THE LOTTERY LIVES')
print(BAR)
fs = FS['RLF'].mean(axis=1)
t5 = T['flat']['RLF'].mean(axis=1)
t10 = L['lot_flat']['RLF'].mean(axis=1)
for name, v in (('FIXED seed 0 (this container)', fs),
                ('published, 5 different seeds', t5),
                ('published, 10 different seeds', t10)):
    print('  %-32s n=%2d  span %5.2f - %5.2f  (%.1fx)  per-run %s'
          % (name, len(v), v.min(), v.max(), v.max() / v.min(),
             np.array2string(np.sort(v), precision=2)))

dfs = D.decompose(FS['RLF'])
d5 = D.decompose(T['flat']['RLF'])
d10 = D.decompose(L['lot_flat']['RLF'])
print('\n  sigma_between, fixed seed      : %.3f' % dfs['sd_train'] if dfs['resolvable']
      else '\n  sigma_between, fixed seed      : < %.3f (not resolvable)' % dfs['sd_train_hi'])
print('  sigma_train, 5 varying seeds   : %.3f' % d5['sd_train'])
print('  sigma_train, 10 varying seeds  : %.3f' % d10['sd_train'])
if dfs['resolvable']:
    print('\n  ratio fixed/varying(5)  : %.2f' % (dfs['sd_train'] / d5['sd_train']))
    print('  ratio fixed/varying(10) : %.2f' % (dfs['sd_train'] / d10['sd_train']))
    frac = dfs['v_train'] / d10['v_train']
    print('\n  VARIANCE share reproduced at a FIXED seed, against the 10-seed arm: %.0f%%'
          % (100 * frac))
    print('  => the declared seed can account for at most about %.0f%% of the lottery variance.'
          % (100 * max(0.0, 1 - frac)))
print('\n  F test on the fixed-seed arm: F = %.3f, p = %.4g  (resolvable: %s)'
      % (dfs['f_obs'], dfs['p'], dfs['resolvable']))
print(BAR)
