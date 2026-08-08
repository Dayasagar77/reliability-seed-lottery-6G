#!/usr/bin/env python3
"""
review2_c2_m2.py -- C2 (Holm across the metric family) and M2 (the 22.6x ratio), from source.

C2. Reviewer 2 asks which significance claims survive a family-wise correction "across the
multiple metrics reported per comparison". The word "Holm" occurs exactly ONCE in the manuscript,
in the ten-training passage -- it is not applied to the tournament comparisons at all. This applies
Holm-Bonferroni within each comparison's own family of five metrics, using the manuscript's own
CRN-corrected Welch test (imported from tablevii_decompose, which is left unmodified so its md5
gate still means something).

M2. The abstract says the power axis varies 22.6x more than the spectrum axis run to run. The
reviewer back-computes 20.70 / 0.91 = 22.75 and says it should read 22.7. BOTH of his inputs are
already rounded to two decimals, so his arithmetic cannot settle it. This recomputes the ratio from
the grid at full precision. Whichever way it lands, the manuscript should carry the number the data
gives, not the one either of us inferred from rounded text.
"""
import csv
import sys

import numpy as np

sys.path.insert(0, '.')
import tablevii_decompose as D                                          # noqa: E402

D.CSV = 'results/tablevii_r8_laptop.csv'
R = 8
SEEDS = [0, 1, 2, 3, 4]
METRICS = ['HO', 'Power', 'Spec', 'RLF', 'Tput']


def build(path, arms):
    """data[arm][metric] = k x R array; store[arm][metric] = decompose() output (carries ms_e)."""
    rows = list(csv.DictReader(open(path)))
    data, store = {}, {}
    for a in arms:
        sel = [r for r in rows if r['arm'] == a]
        if not sel:
            continue
        data[a], store[a] = {}, {}
        for m in METRICS:
            cell = {(int(r['seed']), int(r['repeat'])): float(r[m]) for r in sel}
            X = np.array([[cell[(s, r)] for r in range(R)] for s in SEEDS])
            data[a][m] = X
            store[a][m] = D.decompose(X)
    return data, store


def holm(pairs, alpha=0.05):
    """Holm-Bonferroni. pairs = [(label, p)]. Returns [(label, p, p_adj, reject)] in input order."""
    order = sorted(range(len(pairs)), key=lambda i: pairs[i][1])
    n = len(pairs)
    adj = [0.0] * n
    running = 0.0
    for rank, i in enumerate(order):
        a = min(1.0, (n - rank) * pairs[i][1])
        running = max(running, a)          # enforce monotonicity
        adj[i] = running
    return [(pairs[i][0], pairs[i][1], adj[i], adj[i] < alpha) for i in range(n)]


print('=' * 104)
print('C2 -- HOLM-BONFERRONI WITHIN EACH COMPARISON\'S FAMILY OF FIVE METRICS')
print('   test: the manuscript\'s own CRN-corrected Welch (tablevii_decompose.welch_pair, p_corr)')
print('=' * 104)

FAMILIES = [
    ('results/tablevii_r8_laptop.csv', ['flat', 'incremental', 'binary', 'guarded'],
     [('guarded', 'flat', 'proposed vs flat'),
      ('guarded', 'incremental', 'proposed vs BDQ fine dial'),
      ('guarded', 'binary', 'proposed vs binary-no-guard')]),
    ('results/tablex_r8_laptop.csv', ['m5_dndqn', 'm5_ddqn', 'm5_std', 'sd5_safe'],
     [('m5_dndqn', 'm5_ddqn', 'Dn-DQN vs DDQN+PER'),
      ('m5_dndqn', 'm5_std', 'Dn-DQN vs Std DQN'),
      ('m5_dndqn', 'sd5_safe', 'Dn-DQN vs shielded Safe-DQN')]),
]

flips = []
for path, arms, comps in FAMILIES:
    try:
        data, store = build(path, arms)
    except FileNotFoundError:
        print('\n  %s not found - skipped' % path)
        continue
    for a, b, label in comps:
        if a not in data or b not in data:
            print('\n  %-32s arms missing - skipped' % label)
            continue
        ps = []
        detail = {}
        for m in METRICS:
            r = D.welch_pair(data, store, a, b, m)
            ps.append((m, float(r['p_corr'])))
            detail[m] = r
        res = holm(ps)
        print('\n  %s   (family of %d metrics)' % (label, len(ps)))
        print('    %-6s %10s %10s %12s %10s   %s'
              % ('metric', 'mean A', 'mean B', 'p (CRN)', 'p (Holm)', 'verdict'))
        for m, p, padj, rej in res:
            d = detail[m]
            was = p < 0.05
            mark = ''
            if was and not rej:
                mark = '   <-- LOSES significance under Holm'
                flips.append((label, m, p, padj))
            print('    %-6s %10.2f %10.2f %12.4f %10.4f   %s%s'
                  % (m, d['mean_a'], d['mean_b'], p, padj,
                     'reject' if rej else 'not rejected', mark))

print('\n' + '-' * 104)
if flips:
    print('  %d claim(s) significant at p<0.05 do NOT survive Holm within their own metric family:'
          % len(flips))
    for label, m, p, padj in flips:
        print('    %-34s %-6s p = %.4f -> %.4f' % (label, m, p, padj))
    print('  These must be reworded in the manuscript, not merely footnoted.')
else:
    print('  every claim significant at p<0.05 also survives Holm within its family')
print('=' * 104)

print()
print('=' * 104)
print('M2 -- THE 22.6x RATIO, RECOMPUTED AT FULL PRECISION')
print('=' * 104)
data, store = build('results/tablevii_r8_laptop.csv', ['flat'])
if 'flat' not in data:
    print('  flat arm not present in tablevii_r8_laptop.csv')
else:
    dp = store['flat']['Power']
    ds = store['flat']['Spec']
    print('  flat arm, power saving   : mean %.4f   sigma_train %.6f  (resolvable=%s)'
          % (dp['grand'], dp['sd_train'], dp['resolvable']))
    print('  flat arm, spectrum saving: mean %.4f   sigma_train %.6f  (resolvable=%s)'
          % (ds['grand'], ds['sd_train'], ds['resolvable']))
    ratio = dp['sd_train'] / ds['sd_train'] if ds['sd_train'] else float('nan')
    print()
    print('  ratio from FULL PRECISION sigma_train : %.4f  -> rounds to %.1f' % (ratio, ratio))
    print('  ratio from the ROUNDED text (20.70/0.91) : %.4f  -> rounds to %.1f'
          % (20.70 / 0.91, 20.70 / 0.91))
    print()
    print('  manuscript prints 22.6; reviewer computed 22.7 from rounded inputs.')
    print('  VERDICT: %s'
          % ('the manuscript is correct - the reviewer rounded twice'
             if abs(round(ratio, 1) - 22.6) < 1e-9 else
             'the manuscript should read %.1f - this is a real correction' % ratio))
print('=' * 104)
