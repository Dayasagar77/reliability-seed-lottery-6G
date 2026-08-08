#!/usr/bin/env python3
"""
m3_retrained.py -- redo the M3 power analysis with the RETRAINED BDQ arm.

WHY THIS SCRIPT EXISTS. The review-2 edit spec lists four passages the retrained arm invalidates.
It misses a fifth: the M3 power paragraph in IX-Q is a function of the BDQ arm's variance
components, and those changed (RLF sigma_train 5.88 -> 2.30). The MDE, the required training
counts, the resolution floor and the achieved power for every proposed-vs-BDQ statement therefore
move with the row.

CONTROL FIRST. Every published M3 number is recomputed from the published arm and must reproduce
what the manuscript prints BEFORE any retrained number is believed. Math copied verbatim from
review_M3_exact.py.
"""
import csv, sys
from collections import defaultdict
import numpy as np
from scipy import stats
sys.path.insert(0, '.')
import tablevii_decompose as D

R = 8
METRICS = ['HO', 'Power', 'Spec', 'RLF', 'Tput']

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

T  = grid('results/tablevii_r8_laptop.csv')
TX = grid('results/tablex_r8_laptop.csv')
L  = grid('results/lottery_r8_laptop.csv', 'latin-1')
B5 = grid('results/bdq5/bdq5.csv')
BDQ_NEW = B5['incremental_retrained']

def parts(A, B):
    ma, mb = A.mean(axis=1), B.mean(axis=1)
    k = A.shape[0]
    sa, sb = ma.var(ddof=1), mb.var(ddof=1)
    b = A.mean(axis=0) - B.mean(axis=0)
    dA, dB = D.decompose(A), D.decompose(B)
    resid = (dA['ms_e'] + dB['ms_e']) / k
    v_blockdiff = max((b.var(ddof=1) - resid) / R, 0.0)
    return ma.mean() - mb.mean(), sa, sb, v_blockdiff, k

def mde_at(sa, sb, vbd, k):
    va, vb = sa / k, sb / k
    se = np.sqrt(va + vb + vbd)
    num = (va + vb + vbd) ** 2
    den = va**2/(k-1) + vb**2/(k-1) + (vbd**2/(R-1) if vbd > 0 else 0.0)
    df = num/den if den > 0 else np.nan
    if not np.isfinite(df) or df < 1:
        return np.nan, np.nan
    return (stats.t.ppf(0.975, df) + stats.t.ppf(0.80, df)) * se, df

def k_for(sa, sb, vbd, target):
    for kk in range(3, 5001):
        m, _ = mde_at(sa, sb, vbd, kk)
        if m <= target:
            return kk
    return None

rng = np.random.default_rng(20260802)
def achieved(A, B):
    diff, sa, sb, vbd, k = parts(A, B)
    va, vb = sa/k, sb/k
    se = np.sqrt(va+vb+vbd)
    num = (va+vb+vbd)**2
    den = va**2/(k-1) + vb**2/(k-1) + (vbd**2/(R-1) if vbd > 0 else 0)
    df = num/den
    tc = stats.t.ppf(0.975, df)
    ncp = abs(diff)/se
    power = stats.t.sf(tc, df, loc=ncp) + stats.t.cdf(-tc, df, loc=ncp)
    sim = rng.standard_t(df, 400000)*se + abs(diff)
    sig = np.abs(sim) >= tc*se
    typem = np.abs(sim[sig]).mean()/abs(diff) if sig.any() else float('nan')
    return abs(diff), df, 100*power, typem

def row(A, B, name):
    diff, sa, sb, vbd, k = parts(A, B)
    m5,_  = mde_at(sa, sb, vbd, 5)
    m20,_ = mde_at(sa, sb, vbd, 20)
    m100,_= mde_at(sa, sb, vbd, 100)
    floor = 2.802*np.sqrt(vbd)
    return dict(name=name, diff=abs(diff), m5=m5, m20=m20, m100=m100,
                floor=floor, k2=k_for(sa, sb, vbd, 2.0))

BAR = '='*104
print(BAR); print('CONTROL -- reproduce the M3 numbers the manuscript prints (published BDQ arm)'); print(BAR)
CTL = [
    row(T['guarded']['RLF'],  T['incremental']['RLF'], 'proposed vs BDQ (published)'),
    row(T['guarded']['RLF'],  T['flat']['RLF'],        'proposed vs flat'),
    row(TX['m5_dndqn']['RLF'],TX['sd5_safe']['RLF'],   'proposed vs shielded Safe-DQN'),
    row(TX['m5_dndqn']['RLF'],TX['m5_ddqn']['RLF'],    'proposed vs DDQN+PER'),
]
print('  %-32s %8s %8s %8s %8s %8s' % ('comparison','MDE k=5','k=20','k=100','floor','k for 2pp'))
for r in CTL:
    print('  %-32s %8.2f %8.2f %8.2f %8.3f %8s' % (r['name'], r['m5'], r['m20'], r['m100'], r['floor'], r['k2']))

MS = {'MDE k=5 vs BDQ':9.75, 'MDE k=20 vs BDQ':4.00, 'MDE k=100 vs BDQ':1.97,
      'k for 2pp vs flat':35, 'k for 2pp vs BDQ':96, 'k for 2pp vs SafeDQN':124,
      'k for 2pp vs DDQN+PER':148, 'floor vs BDQ':1.0}
GOT = {'MDE k=5 vs BDQ':CTL[0]['m5'], 'MDE k=20 vs BDQ':CTL[0]['m20'], 'MDE k=100 vs BDQ':CTL[0]['m100'],
       'k for 2pp vs flat':CTL[1]['k2'], 'k for 2pp vs BDQ':CTL[0]['k2'],
       'k for 2pp vs SafeDQN':CTL[2]['k2'], 'k for 2pp vs DDQN+PER':CTL[3]['k2'],
       'floor vs BDQ':CTL[0]['floor']}
ok = True
print('\n  manuscript claim vs recomputed:')
for kk, want in MS.items():
    got = GOT[kk]
    good = abs(got-want) < (0.06 if isinstance(want,float) else 0.5)
    ok &= good
    print('    %-24s manuscript %-8s recomputed %-10.4f %s' % (kk, want, got, 'OK' if good else '*** MISMATCH ***'))
d,df,pw,tm = achieved(T['guarded']['RLF'], T['incremental']['RLF'])
good = abs(pw-54.6) < 0.6
ok &= good
print('    %-24s manuscript %-8s recomputed %-10.4f %s' % ('achieved power vs BDQ','54.6',pw,'OK' if good else '*** MISMATCH ***'))
print('\n  CONTROL: %s' % ('PASS -- the published M3 paragraph reproduces exactly' if ok else '*** FAIL -- stop ***'))
if not ok:
    sys.exit(3)

print('\n'+BAR); print('RETRAINED -- the same quantities with the one-origin BDQ arm'); print(BAR)
NEW = row(T['guarded']['RLF'], BDQ_NEW['RLF'], 'proposed vs BDQ (retrained)')
print('  %-32s %8s %8s %8s %8s %8s' % ('comparison','MDE k=5','k=20','k=100','floor','k for 2pp'))
for r in (CTL[0], NEW):
    print('  %-32s %8.2f %8.2f %8.2f %8.3f %8s' % (r['name'], r['m5'], r['m20'], r['m100'], r['floor'], r['k2']))
d2,df2,pw2,tm2 = achieved(T['guarded']['RLF'], BDQ_NEW['RLF'])
print('\n  achieved power at the observed effect:')
print('    published  |diff| %6.2f  df %6.2f  power %5.1f%%  Type-M x%.2f' % (d,df,pw,tm))
print('    retrained  |diff| %6.2f  df %6.2f  power %5.1f%%  Type-M x%.2f' % (d2,df2,pw2,tm2))
print('\n  Type-M range across the four significant results (retrained BDQ substituted):')
tms=[tm2]
for A,Bb,nm in ((TX['m5_dndqn']['Spec'],TX['m5_ddqn']['Spec'],'Spec vs DDQN+PER'),
                (TX['m5_dndqn']['Tput'],TX['m5_ddqn']['Tput'],'Tput vs DDQN+PER'),
                (L['lot_guard']['RLF'],L['lot_flat']['RLF'],'RLF guarded vs flat k=10')):
    a,b,c,t = achieved(A,Bb); tms.append(t)
    print('    %-28s power %5.1f%%  Type-M x%.2f' % (nm,c,t))
print('    range x%.2f - x%.2f' % (min(tms), max(tms)))
print(BAR)
