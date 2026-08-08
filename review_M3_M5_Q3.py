#!/usr/bin/env python3
"""
review_M3_M5_Q3.py -- the three review points answerable from grids already on disk.

M3  POWER ANALYSIS. The reviewer asks: given the observed sigma_train, how many training runs
    would be needed to resolve a 2-point RLF difference at 80% power? This converts "five is
    enough to expose the lottery" into a quantitative statement of what five CANNOT resolve.

M5  THROUGHPUT VARIANCE DECOMPOSITION. The relocated dispersion lands on throughput
    (sigma_train 250.3 / 323.3 Mbps against a mean near 500). The reviewer asks for that axis to
    be characterised with the same rigour as the reliability axis: how much of the throughput
    dispersion is spectrum-driven and how much power-driven.

Q3  CORRELATION AT n = 10. r = 0.985 between per-seed power saving and RLF rests on 5 points.
    The ten-training grid exists; recompute there.

Every number comes from the md5-pinned grids and the md5-pinned estimator. Read-only.

WHAT THE VARIANCE COMPONENTS MEAN HERE. A seed mean is an average over R = 8 CRN blocks, so
    Var(seed mean) = sigma_train^2 + sigma_resid^2 / R
and an arm mean over k trainings has Var = (sigma_train^2 + sigma_resid^2/R)/k. The power
analysis below uses exactly that, not sigma_train alone -- using sigma_train alone would
understate the noise and overstate the power.
"""
import csv
import hashlib
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

sys.path.insert(0, '/home/claude/work')
EST = '/home/claude/work/tablevii_decompose.py'
EST_MD5 = '4d30eacb161d92df09821ef755569f85'
T7 = '/home/claude/work/results/tablevii_r8_laptop.csv'
LOT = '/home/claude/work/results/lottery_r8_laptop.csv'
METRICS = ['HO', 'Power', 'Spec', 'RLF', 'Tput']
R = 8


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


got = md5(EST)
print('estimator md5 %s  %s' % (got, 'OK' if got == EST_MD5 else 'MISMATCH -- STOP'))
if got != EST_MD5:
    sys.exit(2)
import tablevii_decompose as D                                       # noqa: E402


def grid(path, encoding='utf-8'):
    rows = list(csv.DictReader(open(path, encoding=encoding)))
    g = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        for m in METRICS:
            g[r['arm']][int(r['seed'])][(m, int(r['repeat']))] = float(r[m])
    out = {}
    for arm in g:
        seeds = sorted(g[arm])
        out[arm] = {m: np.array([[g[arm][s][(m, j)] for j in range(R)] for s in seeds])
                    for m in METRICS}
    return out


T = grid(T7)
L = grid(LOT, 'latin-1')
print('Table VII arms: %s (k=%d)' % (sorted(T), T['guarded']['RLF'].shape[0]))
print('lottery   arms: %s (k=%d)' % (sorted(L), L['lot_guard']['RLF'].shape[0]))

BAR = '=' * 104

# ---------------------------------------------------------------- M3
print('\n' + BAR)
print('M3 -- SEED-COUNT POWER ANALYSIS')
print(BAR)
print("""
  Model: seed mean has Var = s_tr^2 + s_e^2/R ; arm mean over k has that over k.
  Two-arm difference, unpaired and therefore conservative under CRN:
      SE(diff) = sqrt( (V_A + V_B) / k ),  V = s_tr^2 + s_e^2/R
  MDE at 80% power, two-sided alpha = 0.05, df = 2(k-1):
      delta = ( t_{0.975,df} + t_{0.80,df} ) * SE(diff)
  Where sigma_train is NOT resolvable the 97.5% one-sided UPPER bound is used in its place --
  the conservative choice, since it inflates the arm's assumed noise.
""")


def V_of(X):
    d = D.decompose(X)
    s_tr = d['sd_train'] if d['resolvable'] else d['sd_train_hi']
    tag = '' if d['resolvable'] else ' (bound)'
    return d, s_tr, d['sd_e'], tag


def mde(V_a, V_b, k):
    df = 2 * (k - 1)
    if df < 1:
        return float('nan')
    se = np.sqrt((V_a + V_b) / k)
    return (stats.t.ppf(0.975, df) + stats.t.ppf(0.80, df)) * se


def k_needed(V_a, V_b, target, kmax=400):
    for k in range(3, kmax + 1):
        if mde(V_a, V_b, k) <= target:
            return k
    return None


print('  per-arm noise on the RLF axis (Table VII, k = 5, R = 8):')
print('    %-14s %10s %10s %12s' % ('arm', 'sd_train', 'sd_resid', 'V=sd_seedmean'))
Vr = {}
for arm in ['flat', 'incremental', 'binary', 'guarded']:
    d, s_tr, s_e, tag = V_of(T[arm]['RLF'])
    V = s_tr ** 2 + s_e ** 2 / R
    Vr[arm] = V
    print('    %-14s %10.3f %10.3f %12.3f   %s' % (arm, s_tr, s_e, np.sqrt(V), tag))

print('\n  MDE on sustained RLF (percentage points) for each pair the reviewer names:')
PAIRS = [('guarded', 'incremental', 'M3(i)  proposed vs BDQ-style fine dial'),
         ('guarded', 'flat', '       proposed vs flat baseline'),
         ('guarded', 'binary', '       proposed vs binary unguarded')]
print('    %-46s %9s %9s %9s   %s' % ('comparison', 'MDE k=5', 'MDE k=10', 'MDE k=20',
                                      'k for 2.0 pp'))
for a, b, name in PAIRS:
    row = [mde(Vr[a], Vr[b], k) for k in (5, 10, 20)]
    kn = k_needed(Vr[a], Vr[b], 2.0)
    print('    %-46s %9.2f %9.2f %9.2f   %s' % (name, row[0], row[1], row[2],
                                                kn if kn else '>400'))

print('\n  M3(iii) shielded Safe-DQN null (Table X arms, k = 5):')
TX = grid('/home/claude/work/results/tablex_r8_laptop.csv')
d1, s1, e1, t1 = V_of(TX['m5_dndqn']['RLF'])
d2, s2, e2, t2 = V_of(TX['sd5_safe']['RLF'])
V1, V2 = s1 ** 2 + e1 ** 2 / R, s2 ** 2 + e2 ** 2 / R
print('    proposed  sd_train %.2f%s  sd_resid %.2f' % (s1, t1, e1))
print('    masked    sd_train %.2f%s  sd_resid %.2f' % (s2, t2, e2))
print('    observed difference of means: %.2f pp'
      % abs(TX['m5_dndqn']['RLF'].mean() - TX['sd5_safe']['RLF'].mean()))
for k in (5, 10, 20, 40):
    print('    MDE at k = %-3d : %6.2f pp' % (k, mde(V1, V2, k)))
kn = k_needed(V1, V2, 2.0)
print('    trainings needed to resolve 2.0 pp at 80%% power: %s' % (kn if kn else '>400'))

print('\n  the ten-training set (lottery grid), for comparison:')
for arm in ['lot_flat', 'lot_guard']:
    d, s_tr, s_e, tag = V_of(L[arm]['RLF'])
    print('    %-10s k=%d  sd_train %.3f%s  sd_resid %.3f  MDE vs itself-at-k: '
          % (arm, L[arm]['RLF'].shape[0], s_tr, tag, s_e), end='')
    V = s_tr ** 2 + s_e ** 2 / R
    print('k=10 %.2f pp' % mde(V, V, 10))

# ---------------------------------------------------------------- M5
print('\n' + BAR)
print('M5 -- THROUGHPUT VARIANCE DECOMPOSITION ON THE RELOCATED AXIS')
print(BAR)


def tput_decomp(X_t, X_s, X_p, name, k):
    t = X_t.mean(axis=1)
    s = X_s.mean(axis=1)
    p = X_p.mean(axis=1)
    dt = D.decompose(X_t)
    print('\n  %s  (k = %d)' % (name, k))
    print('    throughput  mean %.1f Mbps   sd_train %s   sd across seed means %.1f'
          % (dt['grand'],
             ('%.1f' % dt['sd_train']) if dt['resolvable'] else '< %.1f (bound)' % dt['sd_train_hi'],
             t.std(ddof=0)))
    print('    per-seed throughput: %s' % np.array2string(np.sort(t)[::-1], precision=1))
    print('    per-seed spectrum  : %s' % np.array2string(s[np.argsort(t)[::-1]], precision=2))
    print('    per-seed power     : %s' % np.array2string(p[np.argsort(t)[::-1]], precision=2))
    out = {}
    for lab, x in (('spectrum saving', s), ('power saving', p)):
        if x.std() == 0:
            print('    vs %-16s : constant across seeds, no covariation' % lab)
            out[lab] = 0.0
            continue
        r, pv = stats.pearsonr(x, t)
        # variance of throughput seed means attributable to this axis, univariate
        beta = np.polyfit(x, t, 1)[0]
        expl = (beta ** 2) * x.var(ddof=0)
        out[lab] = expl
        print('    vs %-16s : r = %+.4f (p = %.4f)  slope %+.2f Mbps per point  '
              'explains %.0f%% of Var(Tput seed means)'
              % (lab, r, pv, beta, 100 * expl / t.var(ddof=0)))
    # joint model
    A = np.column_stack([np.ones_like(s), s, p])
    if np.linalg.matrix_rank(A) == 3 and k > 3:
        coef, *_ = np.linalg.lstsq(A, t, rcond=None)
        fit = A @ coef
        r2 = 1 - ((t - fit) ** 2).sum() / ((t - t.mean()) ** 2).sum()
        print('    joint linear model Tput ~ Spec + Power : R^2 = %.4f  (df resid = %d)'
              % (r2, k - 3))
    return out


tput_decomp(T['guarded']['Tput'], T['guarded']['Spec'], T['guarded']['Power'],
            'Table VII proposed arm', 5)
tput_decomp(L['lot_guard']['Tput'], L['lot_guard']['Spec'], L['lot_guard']['Power'],
            'ten-training guarded set', 10)
tput_decomp(L['lot_flat']['Tput'], L['lot_flat']['Spec'], L['lot_flat']['Power'],
            'ten-training flat set', 10)

# ---------------------------------------------------------------- Q3
print('\n' + BAR)
print('Q3 -- POWER-SAVING vs RLF CORRELATION AT n = 10')
print(BAR)


def corr(X_p, X_r, name):
    p = X_p.mean(axis=1)
    r_ = X_r.mean(axis=1)
    n = len(p)
    r, pv = stats.pearsonr(p, r_)
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
    rs, ps = stats.spearmanr(p, r_)
    print('  %-34s n=%2d  r = %+.4f (p = %.4f)  95%% CI [%+.4f, %+.4f]  Spearman %+.4f (p=%.4f)'
          % (name, n, r, pv, lo, hi, rs, ps))
    print('      power saving per seed: %s' % np.array2string(np.sort(p), precision=2))
    print('      sustained RLF per seed: %s' % np.array2string(r_[np.argsort(p)], precision=2))


corr(T['flat']['Power'], T['flat']['RLF'], 'flat, Table VII (published n=5)')
corr(L['lot_flat']['Power'], L['lot_flat']['RLF'], 'flat, ten-training set')
corr(L['lot_guard']['Power'], L['lot_guard']['RLF'], 'guarded, ten-training set')
allp = np.concatenate([T['flat']['Power'].mean(axis=1), L['lot_flat']['Power'].mean(axis=1)])
allr = np.concatenate([T['flat']['RLF'].mean(axis=1), L['lot_flat']['RLF'].mean(axis=1)])
r, pv = stats.pearsonr(allp, allr)
print('  %-34s n=%2d  r = %+.4f (p = %.4f)   [two disjoint campaigns pooled]'
      % ('flat, both campaigns', len(allp), r, pv))
print('\n' + BAR)
