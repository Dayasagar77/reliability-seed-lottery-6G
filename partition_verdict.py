#!/usr/bin/env python3
"""
partition_verdict.py -- the confound-free partition, both arms trained in THIS container.

    sigma_vary^2  = between-training variance when the seed DOES change  (seed + run entropy)
    sigma_fixed^2 = between-training variance when the seed is HELD       (run entropy alone)
    sigma_seed^2  = sigma_vary^2 - sigma_fixed^2                          (what the seed adds)

Both arms: flat controller, 8e5 steps, same driver, same pool, same machine, same toolchain,
evaluated under the same R = 8 CRN protocol at the same channel seeds. They differ in exactly
one thing -- whether the five runs share a declared seed. The earlier cross-venue estimate
(<=9%) compared a cloud fixed-seed arm against a laptop varying-seed arm, and the argument that
licenses that (a constant offset cancels out of a standard deviation) covers an additive shift,
not a change of scale. This has no such gap.

sigma_seed^2 can come out NEGATIVE. That is not an error: it is the moment estimator saying the
seed contributes nothing distinguishable, exactly as a negative variance component elsewhere in
this paper is reported as a bound rather than as zero. It is reported that way here too.
"""
import csv
import glob
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, '/home/claude/work')
import tablevii_decompose as D                                       # noqa: E402

R = 8
METRICS = ['HO', 'Power', 'Spec', 'RLF', 'Tput']


def arm(pattern, key='replicate'):
    g = defaultdict(dict)
    for p in sorted(glob.glob(pattern)):
        for r in csv.DictReader(open(p)):
            for m in METRICS:
                g[int(r[key])][(m, int(r['repeat']))] = float(r[m])
    ks = sorted(k for k in g if len(g[k]) == len(METRICS) * R)
    return ks, {m: np.array([[g[k][(m, j)] for j in range(R)] for k in ks]) for m in METRICS}


kf, FS = arm('/home/claude/work/sim/results/fixedseed/unit_fs_r*.csv')
kv, VS = arm('/home/claude/work/sim/results/varyseed/unit_vs_r*.csv')
print('fixed-seed arm  : k = %d %s' % (len(kf), kf))
print('varying-seed arm: k = %d %s' % (len(kv), kv))
if not kv:
    sys.exit('varying-seed arm has no complete replicates yet')

BAR = '=' * 100
print('\n' + BAR)
print('CONFOUND-FREE PARTITION -- both arms trained and evaluated in this container')
print(BAR)
print('  %-8s %26s %26s %20s' % ('', 'seed HELD (k=%d)' % len(kf),
                                 'seed VARIES (k=%d)' % len(kv), 'seed contribution'))
for m in METRICS:
    df, dv = D.decompose(FS[m]), D.decompose(VS[m])
    vf = df['v_train'] if df['v_train'] > 0 else 0.0
    vv = dv['v_train'] if dv['v_train'] > 0 else 0.0
    vs_ = vv - vf
    sf = ('%.3f' % df['sd_train']) if df['resolvable'] else '<%.3f' % df['sd_train_hi']
    sv = ('%.3f' % dv['sd_train']) if dv['resolvable'] else '<%.3f' % dv['sd_train_hi']
    if vs_ > 0:
        share = 100.0 * vs_ / vv if vv > 0 else float('nan')
        seedtxt = 'sigma %.3f  (%.0f%% of var)' % (np.sqrt(vs_), share)
    else:
        seedtxt = 'not distinguishable from 0'
    print('  %-8s %10.2f +- %-12s %10.2f +- %-12s %20s'
          % (m, df['grand'], sf, dv['grand'], sv, seedtxt))

print('\n' + BAR)
print('THE RELIABILITY AXIS')
print(BAR)
f, v = FS['RLF'].mean(axis=1), VS['RLF'].mean(axis=1)
print('  seed HELD   : %s   span %.2f-%.2f' % (np.array2string(np.sort(f), precision=2),
                                               f.min(), f.max()))
print('  seed VARIES : %s   span %.2f-%.2f' % (np.array2string(np.sort(v), precision=2),
                                               v.min(), v.max()))
df, dv = D.decompose(FS['RLF']), D.decompose(VS['RLF'])
vf = max(df['v_train'], 0.0)
vv = max(dv['v_train'], 0.0)
print('\n  var(between trainings), seed held   : %.4f  -> sigma %.3f' % (vf, np.sqrt(vf)))
print('  var(between trainings), seed varies : %.4f  -> sigma %.3f' % (vv, np.sqrt(vv)))
delta = vv - vf
print('  difference attributable to the seed : %+.4f' % delta)
if delta > 0:
    print('  => the seed accounts for %.0f%% of the lottery variance; retraining alone for %.0f%%.'
          % (100 * delta / vv, 100 * vf / vv))
else:
    print('  => NEGATIVE: holding the seed did not reduce the dispersion at all. The declared')
    print('     seed contributes nothing this design can distinguish from zero, and retraining')
    print('     alone accounts for the whole lottery.')
from scipy import stats                                              # noqa: E402
Ff = vf / max(dv['v_train'], 1e-12)
print('\n  variance ratio fixed/varying = %.3f' % (vf / vv if vv > 0 else float('nan')))
nf, nv = len(kf) - 1, len(kv) - 1
if vv > 0 and nf > 0 and nv > 0:
    ratio = vf / vv
    p = 2 * min(stats.f.cdf(ratio, nf, nv), 1 - stats.f.cdf(ratio, nf, nv))
    print('  F(%d,%d) test that the two dispersions are equal: p = %.4f  (%s)'
          % (nf, nv, p, 'no evidence they differ' if p > 0.05 else 'they differ'))
print(BAR)
