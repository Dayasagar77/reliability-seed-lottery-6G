#!/usr/bin/env python3
"""
review_M3_exact.py -- M3 power analysis against THE TEST THE PAPER ACTUALLY REPORTS.

A first pass modelled the comparison as an unpaired two-sample Welch. That is conservative, and
conservative in a way that would have read as a contradiction: it puts the MDE for proposed-vs-
BDQ at 8.41 pp while the paper reports that same difference resolved at p = 0.0442. Both can be
true -- the paper's test exploits CRN pairing and therefore has more power than an unpaired model
-- but quoting the unpaired number as "what five seeds can resolve" would understate the design.

So this computes the MDE for the ACTUAL statistic: the CRN-corrected Welch on seed means, with
Satterthwaite df, exactly as tablevii_decompose.welch_pair implements it.

    SE(k)^2 = (s_A^2 + s_B^2)/k  +  v_blockdiff
    MDE(k)  = ( t_{0.975,df(k)} + t_{0.80,df(k)} ) * SE(k)

THE STRUCTURAL POINT THIS EXPOSES. v_blockdiff -- the variance of the block-level arm-mean
difference -- does NOT shrink with k. It is divided by R, not by k. So SE(k) does not go to zero
as trainings are added; it goes to sqrt(v_blockdiff), and the MDE goes to 2.80 * sqrt(v_blockdiff).

    THERE IS A RESOLUTION FLOOR THAT NO NUMBER OF TRAINING SEEDS CAN BEAT.
    Beating it requires more CRN BLOCKS (larger R), not more trainings.

That is a sharper and more useful answer to the reviewer than a seed count, and it is not
something the paper currently says.
"""
import hashlib
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, '/home/claude/work')
EST_MD5 = '4d30eacb161d92df09821ef755569f85'


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
from review_M3_M5_Q3 import grid, R                                  # noqa: E402

T = grid('/home/claude/work/results/tablevii_r8_laptop.csv')
TX = grid('/home/claude/work/results/tablex_r8_laptop.csv')
L = grid('/home/claude/work/results/lottery_r8_laptop.csv', 'latin-1')


def parts(A, B):
    """The three variance ingredients of the CRN-corrected Welch, at the observed k."""
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
    den = va ** 2 / (k - 1) + vb ** 2 / (k - 1) + (vbd ** 2 / (R - 1) if vbd > 0 else 0.0)
    df = num / den if den > 0 else np.nan
    if not np.isfinite(df) or df < 1:
        return np.nan, np.nan
    return (stats.t.ppf(0.975, df) + stats.t.ppf(0.80, df)) * se, df


def report(A, B, name, obs_p=None):
    diff, sa, sb, vbd, k = parts(A, B)
    floor = 2.802 * np.sqrt(vbd)
    print('\n  %s' % name)
    print('    observed |difference| %.2f pp   at k = %d' % (abs(diff), k), end='')
    print('   (paper reports p = %s)' % obs_p if obs_p else '')
    print('    variance parts: s_A^2 = %.3f  s_B^2 = %.3f  v_blockdiff = %.5f'
          % (sa, sb, vbd))
    line = '    MDE(80%% power):'
    for kk in (5, 10, 20, 50, 100):
        m, df = mde_at(sa, sb, vbd, kk)
        line += '  k=%d %.2f' % (kk, m)
    print(line)
    print('    irreducible floor as k -> inf : %.3f pp   (set by R = %d blocks, not by k)'
          % (floor, R))
    for target in (2.0, 1.0):
        kn = None
        for kk in range(3, 2001):
            m, _ = mde_at(sa, sb, vbd, kk)
            if m <= target:
                kn = kk
                break
        print('    trainings to resolve %.1f pp : %s'
              % (target, kn if kn else 'UNREACHABLE at any k (floor %.2f pp)' % floor))


print('=' * 104)
print('M3 -- MINIMUM DETECTABLE EFFECT under the CRN-corrected Welch, sustained RLF')
print('=' * 104)
report(T['guarded']['RLF'], T['incremental']['RLF'],
       'M3(i)   proposed vs BDQ-style fine dial  [Table VII]', '0.0442')
report(T['guarded']['RLF'], T['flat']['RLF'],
       '        proposed vs flat baseline        [Table VII]')
report(TX['m5_dndqn']['RLF'], TX['sd5_safe']['RLF'],
       'M3(iii) proposed vs shielded Safe-DQN    [Table X]', '0.7454')
report(TX['m5_dndqn']['RLF'], TX['m5_ddqn']['RLF'],
       'M3(ii)  proposed vs DDQN+PER             [Table X]', '0.4180')
report(L['lot_guard']['RLF'], L['lot_flat']['RLF'],
       '        guarded vs flat                  [ten-training set]')

print('\n' + '=' * 104)
print('THE SAME QUESTION ON THE OTHER METRICS, k = 5, Table X proposed vs DDQN+PER')
print('=' * 104)
for m, unit, p in (('Power', 'points', '0.0639'), ('Spec', 'points', '0.0466'),
                   ('Tput', 'Mbps', '0.0421'), ('HO', 'points', None)):
    diff, sa, sb, vbd, k = parts(TX['m5_dndqn'][m], TX['m5_ddqn'][m])
    m5, _ = mde_at(sa, sb, vbd, 5)
    m20, _ = mde_at(sa, sb, vbd, 20)
    print('  %-6s observed %8.2f %-6s  MDE k=5 %8.2f   k=20 %8.2f   floor %8.2f %s'
          % (m, abs(diff), unit, m5, m20, 2.802 * np.sqrt(vbd),
             ('(paper p = %s)' % p) if p else ''))
print('=' * 104)


# ------------------------------------------------------------------ achieved power & Type M
print('\n' + '=' * 104)
print('ACHIEVED POWER AT THE OBSERVED EFFECT, and the magnitude bias that follows')
print('=' * 104)
print("""
  An MDE larger than an observed-but-significant difference is not a contradiction. A study
  powered at 80% for 9.75 pp still returns p < 0.05 on a smaller observed difference -- but
  only on the draws where the observed difference happens to OVERSTATE the truth. So the
  p-values stand; the EFFECT SIZES attached to the significant ones are upward-biased. This is
  the Type-M (magnitude) error, and it is the precise form of the reviewer's M3 worry.

  "Power at observed" below treats the observed difference as if it were the truth and asks how
  often this design would detect it. Type-M is the ratio E[|observed| | significant] / |true|,
  by simulation under that same assumption.
""")
rng = np.random.default_rng(20260802)


def achieved(A, B, name):
    diff, sa, sb, vbd, k = parts(A, B)
    va, vb = sa / k, sb / k
    se = np.sqrt(va + vb + vbd)
    num = (va + vb + vbd) ** 2
    den = va ** 2 / (k - 1) + vb ** 2 / (k - 1) + (vbd ** 2 / (R - 1) if vbd > 0 else 0)
    df = num / den
    tc = stats.t.ppf(0.975, df)
    ncp = abs(diff) / se
    power = stats.t.sf(tc, df, loc=ncp) + stats.t.cdf(-tc, df, loc=ncp)
    sim = rng.standard_t(df, 400000) * se + abs(diff)
    sig = np.abs(sim) >= tc * se
    typem = np.abs(sim[sig]).mean() / abs(diff) if sig.any() else float('nan')
    print('  %-44s |diff| %7.2f  df %5.2f  power@observed %5.1f%%  Type-M x%.2f'
          % (name, abs(diff), df, 100 * power, typem))


achieved(T['guarded']['RLF'], T['incremental']['RLF'], 'RLF  proposed vs BDQ fine dial (p=0.0442)')
achieved(T['guarded']['RLF'], T['flat']['RLF'], 'RLF  proposed vs flat')
achieved(TX['m5_dndqn']['Spec'], TX['m5_ddqn']['Spec'], 'Spec proposed vs DDQN+PER (p=0.0466)')
achieved(TX['m5_dndqn']['Tput'], TX['m5_ddqn']['Tput'], 'Tput proposed vs DDQN+PER (p=0.0421)')
achieved(TX['m5_dndqn']['Power'], TX['m5_ddqn']['Power'], 'Pwr  proposed vs DDQN+PER (p=0.0639)')
achieved(L['lot_guard']['RLF'], L['lot_flat']['RLF'], 'RLF  guarded vs flat, k=10')
print('=' * 104)
