#!/usr/bin/env python3
"""ratio174_provenance.py -- READ-ONLY.

Settles ONE question, from source, before it is written into a correction record:

    Is the "17.4x" in the venue-provisional banner of
    DnDQN_Paper1_ManuscriptEdits_RelocationFraming_20260801.md (labelled CLOUD, and marked
    STALE by CorrectionSpec 14 item 2)

    the SAME number as the "17.4x" in CorrectionSpec 6/6.1 (labelled LAPTOP, and treated
    there as LIVE)?

The banner quotes exact 17.476348.  CorrectionSpec 6.1 quotes 17.4761.  Agreement to five
significant figures between two supposedly different venues is not a coincidence one may assume
away, and it is not a coincidence one may assert either.  So it is recomputed here from the
laptop CSV, from scratch, without importing tablevii_decompose.py -- an independent re-derivation,
the same discipline CorrectionSpec 6.1 item 1 applied to the headline.

Writes nothing but its own stdout.
"""
import hashlib
import numpy as np
import pandas as pd
from scipy import stats

CSV = '/root/work/exp/results/tablevii_r8_laptop.csv'
EXPECT = '2234f1ccfd763518a0027712274f22a8'

h = hashlib.md5(open(CSV, 'rb').read()).hexdigest()
print(f'csv md5 {h}  {"OK" if h == EXPECT else "DRIFT -- REFUSING"}')
if h != EXPECT:
    raise SystemExit(1)

df = pd.read_csv(CSV)
print(f'{len(df)} rows; columns: {list(df.columns)}\n')
for c in ('arm', 'tag', 'label'):
    if c in df.columns:
        print(f'  distinct {c}: {sorted(df[c].unique())}')
print()


def decompose(sub, metric, seed_col, rep_col):
    """Two-way crossed random effects, no replication.  Returns the pieces the bounds need."""
    piv = sub.pivot_table(index=seed_col, columns=rep_col, values=metric, aggfunc='mean')
    Y = piv.values
    k, R = Y.shape
    gm = Y.mean()
    ms_a = R * ((Y.mean(axis=1) - gm) ** 2).sum() / (k - 1)
    ms_b = k * ((Y.mean(axis=0) - gm) ** 2).sum() / (R - 1)
    resid = Y - Y.mean(axis=1, keepdims=True) - Y.mean(axis=0, keepdims=True) + gm
    ms_e = (resid ** 2).sum() / ((k - 1) * (R - 1))
    return dict(k=k, R=R, ms_a=ms_a, ms_b=ms_b, ms_e=ms_e,
                F=ms_a / ms_e, dfA=k - 1, dfE=(k - 1) * (R - 1),
                var_train=(ms_a - ms_e) / R, mean=gm)


def bounds(d, alpha):
    """One-sided bounds on sigma_train at level (1-alpha), via the exact pivot
       F_obs/(1+R*gamma) ~ F(dfA, dfE),  gamma = sigma_train^2 / sigma_e^2."""
    F, R, dfA, dfE = d['F'], d['R'], d['dfA'], d['dfE']
    q_a = stats.f.ppf(alpha, dfA, dfE)          # lower tail
    q_1a = stats.f.ppf(1 - alpha, dfA, dfE)      # upper tail
    g_u = (F / q_a - 1) / R                      # UPPER bound on gamma
    g_l = (F / q_1a - 1) / R                     # LOWER bound on gamma
    se2 = d['ms_e']                              # sigma_e^2 estimate
    su = np.sqrt(max(g_u, 0) * se2)
    sl = np.sqrt(max(g_l, 0) * se2)
    return sl, su


# locate the two arms by whatever the arm column is called
armcol = next(c for c in ('arm', 'tag', 'label', 'method') if c in df.columns)
seedcol = next(c for c in ('seed', 'train_seed', 'training_seed') if c in df.columns)
repcol = next(c for c in ('repeat', 'rep', 'crn_repeat', 'block') if c in df.columns)
metric = next(c for c in ('rlf', 'RLF', 'rlf_rate', 'rlf_pct') if c in df.columns)
print(f'using arm={armcol!r} seed={seedcol!r} repeat={repcol!r} metric={metric!r}\n')

arms = sorted(df[armcol].unique())
res = {}
for a in arms:
    d = decompose(df[df[armcol] == a], metric, seedcol, repcol)
    p = 1 - stats.f.cdf(d['F'], d['dfA'], d['dfE'])
    l95, u95 = bounds(d, 0.05)     # 95 % one-sided
    l975, u975 = bounds(d, 0.025)  # 97.5 % one-sided
    res[a] = dict(d=d, p=p, l95=l95, u95=u95, l975=l975, u975=u975)
    print(f'{a:28s} mean {d["mean"]:8.4f}  MS_A {d["ms_a"]:.5f}  MS_E {d["ms_e"]:.5f}  '
          f'F({d["dfA"]},{d["dfE"]}) = {d["F"]:.4f}  p = {p:.5f}')
    print(f'{"":28s}   sigma_train pt {np.sqrt(max(d["var_train"],0)):.4f}   '
          f'95%L {l95:.4f}  95%U {u95:.4f}   97.5%L {l975:.4f}  97.5%U {u975:.4f}')

print('\n' + '=' * 100)
print('  THE TWO RATIOS, from these laptop bounds')
print('=' * 100)


def show(flat_key, guard_key):
    f, g = res[flat_key], res[guard_key]
    r95 = f['l975'] / g['u975']
    r90 = f['l95'] / g['u95']
    print(f'  joint >=95 %  = {f["l975"]:.4f} / {g["u975"]:.4f} = {r95:.6f}   '
          f'-> floored "at least {np.floor(r95*10)/10:.1f}x"')
    print(f'  joint >=90 %  = {f["l95"]:.4f} / {g["u95"]:.4f} = {r90:.6f}   '
          f'-> floored "at least {np.floor(r90*10)/10:.1f}x"')
    print()
    print(f'  CorrectionSpec 6 states  12.6563  and  17.4761')
    print(f'  banner (RelocationFraming line 170) states exact  17.476348')
    print(f'  this recomputation gives  {r95:.6f}  and  {r90:.6f}')
    print()
    print(f'  |recomputed_r90 - 17.4761 |  = {abs(r90 - 17.4761):.6f}')
    print(f'  |recomputed_r90 - 17.476348| = {abs(r90 - 17.476348):.6f}')


cand_flat = [a for a in arms if 'flat' in str(a).lower()]
cand_guard = [a for a in arms if 'guard' in str(a).lower()]
print(f'flat candidates {cand_flat}; guarded candidates {cand_guard}\n')
if cand_flat and cand_guard:
    show(cand_flat[0], cand_guard[0])
else:
    print('  arm names did not match flat/guarded; inspect the distinct-values list above')
