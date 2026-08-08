#!/usr/bin/env python3
"""
tableviii_ix.py -- replacement values for TABLE VIII and TABLE IX, laptop venue.

WHY THIS EXISTS
---------------
Dumping the manuscript's XML on 2026-08-02 revealed something no planning doc had recorded:
the four arms of results/tablex_r8_laptop.csv

    m5_dndqn  m5_ddqn  m5_std  sd5_safe

are EXACTLY the four rows of the manuscript's TABLE VIII

    "Dn-DQN (dueling + n-step + PER) / DDQN+PER / Standard DQN / Shielded Safe-DQN (action mask)"

whose caption currently reads "3 training seeds" and whose body text says "3 training seeds x
4x10^5 steps".  The grid is 5 training seeds x 8x10^5 steps x 8 CRN evaluation repeats.  So the
grid does not merely close Table X's learned-baseline rows -- it SUPERSEDES Table VIII, and with
it TABLE IX, whose Welch t / Cohen's d are computed on n = 3 training seeds.

Table IX at n = 3 is the weakest inferential object in the paper: two of its six p-values sit
either side of 0.05 on 2 degrees of freedom.  Recomputing at n = 5, with the CRN correction that
the shared channel blocks require, is not cosmetic.

ARCHITECTURAL DISCIPLINE
------------------------
Imports load() from tablex_decompose (which already enforces every provenance gate on this CSV)
and decompose / cohens_d / welch_pair / rd / ru from tablevii_decompose.  It re-implements
nothing.  There must be exactly ONE implementation of the bound algebra in this project.

Arithmetic carries no venue effect: the INPUTS are laptop-produced, so where the sums happen is
irrelevant.  This is NOT a cross-venue comparison and must not be described as one.

THE TEST
--------
welch_pair runs Welch's t on the k = 5 SEED MEANS -- the correct unit of replication, since a
training seed is the thing being sampled -- and then adds the CRN block term.  The two arms are
evaluated on the SAME eight channel draws, so their arm means share a block offset that the
naive SE^2 = s_A^2/k + s_B^2/k omits.  Adding it back gives a strictly CONSERVATIVE p.  Where
naive and corrected disagree, the corrected one is what the paper reports.

Cohen's d is computed on the seed means with pooled ddof = 1 sd -- the manuscript's convention,
unchanged, so the d column is comparable to the published one.
"""
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, '/home/claude')
from tablevii_decompose import decompose, cohens_d, welch_pair, rd, ru   # ONE implementation
from tablex_decompose import load, ARMS, LABEL, METRICS, CSV

# Manuscript TABLE VIII as it currently stands, read verbatim from word/document.xml of
# DnDQN_manuscript_updated_IXrenumbered.docx (inner md5 4972f22c676787a95b2192e2d207c814)
# on 2026-08-02.  n = 3 training seeds x 4x10^5 steps, +- = sd across those three.
PUBLISHED_VIII = {
    'm5_dndqn': dict(HO=(94.44, 0.61), Power=(20.92, 4.01),  Spec=(90.40, 0.28), RLF=(8.98, 1.73),  Tput=(450.4, 2.0)),
    'm5_ddqn':  dict(HO=(91.64, 3.72), Power=(65.86, 16.35), Spec=(65.51, 6.83), RLF=(13.87, 6.43), Tput=(711.2, 68.4)),
    'm5_std':   dict(HO=(89.86, 1.55), Power=(76.40, 5.72),  Spec=(74.52, 3.13), RLF=(19.29, 1.51), Tput=(559.5, 26.7)),
    'sd5_safe': dict(HO=(94.47, 0.04), Power=(19.41, 4.47),  Spec=(85.04, 4.33), RLF=(6.00, 1.63),  Tput=(529.4, 57.0)),
}

# Manuscript TABLE IX as it currently stands, same source.  (t, d, p) at n = 3.
# SIGN CONVENTION: proposed arm first, so every t and d is (proposed - other).
PUBLISHED_IX = {
    ('m5_dndqn', 'm5_std',  'RLF'):   (-5.18,  -6.35, 0.0034),
    ('m5_dndqn', 'm5_std',  'Spec'):  (+5.83,  +7.13, 0.0183),
    ('m5_dndqn', 'm5_std',  'Power'): (-9.17, -11.23, 0.0006),
    ('m5_dndqn', 'm5_ddqn', 'RLF'):   (-0.85,  -1.04, 0.3961),
    ('m5_dndqn', 'm5_ddqn', 'Spec'):  (+4.20,  +5.15, 0.0355),
    ('m5_dndqn', 'm5_ddqn', 'Power'): (-3.08,  -3.78, 0.0531),
}
PUB_ORDER = [('m5_dndqn', 'm5_std', 'RLF'), ('m5_dndqn', 'm5_std', 'Spec'),
             ('m5_dndqn', 'm5_std', 'Power'), ('m5_dndqn', 'm5_ddqn', 'RLF'),
             ('m5_dndqn', 'm5_ddqn', 'Spec'), ('m5_dndqn', 'm5_ddqn', 'Power')]

SHORT = {'m5_dndqn': 'Dn-DQN', 'm5_ddqn': 'DDQN+PER',
         'm5_std': 'Standard DQN', 'sd5_safe': 'Shielded Safe-DQN'}
DEC = {'HO': 2, 'Power': 2, 'Spec': 2, 'RLF': 2, 'Tput': 1}


def cell(d, m):
    """Render one table cell on the sigma_train basis, honouring resolvability."""
    n = DEC[m]
    if d['resolvable']:
        return f'{d["grand"]:.{n}f} $\\pm$ {d["sd_train"]:.{n}f}'
    return f'{d["grand"]:.{n}f} ($\\sigma_{{train}} < {ru(d["sd_train_hi"], n):.{n}f}$)'


def main():
    data, meta, blocks, torch_v = load()

    print('=' * 118)
    print('TABLE VIII and TABLE IX -- REPLACEMENT VALUES, laptop venue (torch %s)' % torch_v)
    print('  source : %s' % CSV)
    print('  design : per arm, k = 5 training seeds x R = 8 CRN evaluation repeats')
    print('           each repeat = 5 evaluation seeds x 150 episodes;  df_A = 4, df_B = 7, df_E = 28')
    print('  NOTE   : these are the SAME four arms and the SAME 160 evaluations that close')
    print('           Table X\'s learned-baseline rows.  Tables VIII and X must print IDENTICAL')
    print('           values wherever they share a controller, or the paper contradicts itself.')
    print('=' * 118)

    # --------------------------------------------------------------- provenance, one line/arm
    print('\n[0] CHECKPOINT PROVENANCE (arm -> the five trained models actually evaluated)')
    for arm in ARMS:
        print('  %-9s %-36s %s' % (arm, LABEL[arm],
                                   ' '.join(meta[arm][s][0] for s in range(5))))
    print('  blocks: ' + ' '.join('r%d=%s' % (rep, sorted(s)[0]) for rep, s in blocks))

    # ------------------------------------------------------------------------- decomposition
    store = {}
    print('\n[1] PER-ARM DECOMPOSITION  (sd_train = sqrt((MS_A - MS_E)/R); '
          'resolvable requires v_train > 0 AND p < 0.05)')
    print('  %-9s %-7s %9s %10s %9s %9s %9s %9s  %-11s %s'
          % ('arm', 'metric', 'mean', 'sd_train', 'sd_eval', 'sd_blk', 'F(4,28)', 'p',
             'verdict', 'published n=3'))
    print('  ' + '-' * 128)
    for arm in ARMS:
        store[arm] = {}
        for m in METRICS:
            d = decompose(data[arm][m])
            store[arm][m] = d
            pm, ps = PUBLISHED_VIII[arm][m]
            sdt = f'{d["sd_train"]:.4f}' if d['resolvable'] else f'<{d["sd_train_hi"]:.4f}'
            print('  %-9s %-7s %9.4f %10s %9.4f %9.4f %9.3f %9.5f  %-11s %8.2f+-%-6.2f'
                  % (arm, m, d['grand'], sdt, d['sd_e'], np.sqrt(max(d['v_block'], 0)),
                     d['f_obs'], d['p'],
                     'RESOLVABLE' if d['resolvable'] else 'not resolv.', pm, ps))
        print('  ' + '-' * 128)

    unres = [(a, m) for a in ARMS for m in METRICS if not store[a][m]['resolvable']]
    print('  -> %d of %d cells RESOLVABLE.  Not resolvable: %s'
          % (20 - len(unres), 20, unres if unres else 'none'))

    # --------------------------------------------------------------- per-seed means, on record
    print('\n[2] PER-SEED MEANS (each is a mean over 8 CRN repeats; these are the units of')
    print('    replication for every test in [4])')
    for arm in ARMS:
        print('  %-9s' % arm)
        for m in METRICS:
            print('      %-6s ' % m + '  '.join('%9.3f' % v for v in store[arm][m]['seed_means']))

    # ------------------------------------------------------------------ TABLE VIII, typeset
    print('\n[3] TABLE VIII -- REPLACEMENT ROWS (sigma_train basis)')
    print('  | Algorithm | HO (%) | Power (%) | Spec (%) | RLF (%) | Tput (Mbps) |')
    for arm in ARMS:
        print('  | %-32s | %s |' % (LABEL[arm],
                                    ' | '.join(cell(store[arm][m], m) for m in METRICS)))

    print('\n  side-by-side with the published n=3 row, to make the size of the change visible:')
    print('  %-9s %-6s | %-22s | %-22s | %s' % ('arm', 'metric', 'published (n=3, 4e5)',
                                                'measured (n=5, 8e5)', 'mean shift'))
    print('  ' + '-' * 100)
    for arm in ARMS:
        for m in METRICS:
            d, (pm, ps) = store[arm][m], PUBLISHED_VIII[arm][m]
            n = DEC[m]
            print('  %-9s %-6s | %*.*f +- %-*.*f      | %-22s | %+8.2f'
                  % (arm, m, 8, n, pm, 6, n, ps, cell(d, m).replace('$\\pm$', '+-')
                     .replace('$\\sigma_{train} < ', 'sd_tr<').replace('$', '').replace(')', ')'),
                     d['grand'] - pm))
        print('  ' + '-' * 100)

    # ------------------------------------------------------------------- TABLE IX, recomputed
    print('\n[4] TABLE IX -- RECOMPUTED at n = 5 training seeds, with the CRN block correction')
    print('    Welch t on the k=5 seed means.  naive SE^2 = s_A^2/k + s_B^2/k omits')
    print('    Var(beta_A - beta_B); the corrected SE adds it and is strictly conservative.')
    print('    Cohen\'s d on the seed means, pooled ddof=1 -- the manuscript\'s convention.')
    print()
    print('  %-28s %-6s | %-17s | %7s %7s %8s | %7s %7s %6s %8s | %-20s'
          % ('comparison', 'metric', 'means (A vs B)', 't_naive', 'd', 'p_naive',
             't_corr', 'df', 'p_corr', 'v_blk', 'published (n=3)'))
    print('  ' + '-' * 150)
    ix = {}
    for (a, b, m) in PUB_ORDER:
        w = welch_pair(data, store, a, b, m)
        ix[(a, b, m)] = w
        pt, pd_, pp = PUBLISHED_IX[(a, b, m)]
        n = DEC[m]
        print('  %-28s %-6s | %8.*f vs %6.*f | %7.2f %7.2f %8.4f | %7.2f %7.2f %6.4f %8.3f | '
              't=%+.2f d=%+.2f p=%.4f'
              % (f'{SHORT[a]} vs {SHORT[b]}', m, n, w['mean_a'], n, w['mean_b'],
                 w['t'], w['d'], w['p'], w['t_corr'], w['df_corr'], w['p_corr'],
                 w['v_blockdiff'], pt, pd_, pp))
    print('  ' + '-' * 150)

    print('\n  SIGNIFICANCE AT alpha = 0.05 -- published vs naive vs corrected')
    print('  %-28s %-6s | %-14s | %-14s | %-14s | %s'
          % ('comparison', 'metric', 'published n=3', 'naive n=5', 'corrected n=5', 'verdict'))
    print('  ' + '-' * 118)
    flips = []
    for (a, b, m) in PUB_ORDER:
        w, (_, _, pp) = ix[(a, b, m)], PUBLISHED_IX[(a, b, m)]
        sig = lambda p: ('sig    p=%.4f' % p) if p < 0.05 else ('n.s.   p=%.4f' % p)
        was, now = pp < 0.05, w['p_corr'] < 0.05
        v = 'unchanged' if was == now else ('LOST significance' if was else 'GAINED significance')
        if was != now:
            flips.append((a, b, m, v))
        print('  %-28s %-6s | %-14s | %-14s | %-14s | %s'
              % (f'{SHORT[a]} vs {SHORT[b]}', m, sig(pp), sig(w['p']), sig(w['p_corr']), v))
    print('  ' + '-' * 118)
    if flips:
        print('  -> %d of 6 change significance status. Every one must be reflected in the body text.'
              % len(flips))
        for a, b, m, v in flips:
            print('       %s / %s : %s' % (f'{SHORT[a]} vs {SHORT[b]}', m, v))
    else:
        print('  -> no comparison changes significance status.')

    # ------------------------------------------------- the two comparisons the paper does NOT run
    print('\n[5] COMPLETENESS -- every arm-pair x metric, corrected test only.')
    print('    Table IX prints six of these thirty.  The rest are recorded so that the six')
    print('    printed are visibly a stated selection, not a search over outcomes.')
    print('  %-30s | ' % 'comparison' + ' | '.join('%-22s' % m for m in METRICS))
    print('  ' + '-' * 150)
    pairs = [('m5_dndqn', 'm5_std'), ('m5_dndqn', 'm5_ddqn'), ('m5_dndqn', 'sd5_safe'),
             ('sd5_safe', 'm5_std'), ('sd5_safe', 'm5_ddqn'), ('m5_ddqn', 'm5_std')]
    for a, b in pairs:
        cells = []
        for m in METRICS:
            w = welch_pair(data, store, a, b, m)
            star = '*' if w['p_corr'] < 0.05 else ' '
            cells.append('t=%+6.2f p=%.4f%s' % (w['t_corr'], w['p_corr'], star))
        print('  %-30s | ' % f'{SHORT[a]} vs {SHORT[b]}' + ' | '.join(cells))
    print('  ' + '-' * 150)
    print('  * = p_corr < 0.05.  Six of these are the Table IX rows; the selection is by')
    print('    METRIC (RLF, Spec, Power = the three reward terms) and by PAIR (proposed vs the')
    print('    two flat learned baselines), fixed before the grid was run.')

    # ---------------------------------------------------------------- evaluation-noise floor
    print('\n[6] EVALUATION-NOISE FLOOR per arm -- what one more channel draw moves.')
    print('    pooled sd_eval = sqrt(sigma_block^2 + sigma_e^2); this is the estimator the')
    print('    tables quote, NOT sqrt(MS_E) alone.')
    print('  %-9s | ' % 'arm' + ' | '.join('%-14s' % m for m in METRICS))
    print('  ' + '-' * 96)
    for arm in ARMS:
        cells = []
        for m in METRICS:
            d = store[arm][m]
            cells.append('%14.4f' % np.sqrt(max(d['v_block'], 0) + d['ms_e']))
        print('  %-9s | ' % arm + ' | '.join(cells))
    print('  ' + '-' * 96)

    print('\n[7] CAPTION FACTS (to be transcribed verbatim, not paraphrased)')
    print('    training  : 5 independent training seeds per arm, 8x10^5 environment steps')
    print('    evaluation: 8 common-random-number repeats per model,')
    print('                each repeat = 5 evaluation seeds x 150 episodes  (%d evaluations total)'
          % (4 * 5 * 8))
    print('    reported +-: sigma_train, the training-seed sd with the evaluation-noise')
    print('                 contribution subtracted; "(sigma_train < x)" where the training-seed')
    print('                 component is not resolvable at this k and R.')
    print('    venue     : laptop, torch %s.  Same grid as Table X\'s learned rows.' % torch_v)
    return 0


if __name__ == '__main__':
    sys.exit(main())
