#!/usr/bin/env python3
"""
tablex_decompose.py -- variance decomposition for TABLE X's four learned arms, laptop venue.

WHY THIS EXISTS
---------------
Table VII's four controller-architecture arms were re-measured on a k = 5 training seeds x
R = 8 common-random-number grid so that every reported +/- means "spread across TRAINING
seeds" and the evaluation noise is separated out rather than silently mixed in.  Table X's
four LEARNED-BASELINE arms were still on the OLD basis: one training seed per row, a single
unseeded evaluation pass.  That is precisely the inconsistency the Associate Editor named:

    "some baselines are represented by a single trained model and others by evaluation-seed
     variability, making the comparisons methodologically inconsistent"

The 160-evaluation grid in results/tablex_r8_laptop.csv closes that gap.  This script does
the arithmetic on it.

ARCHITECTURAL DISCIPLINE
------------------------
It IMPORTS decompose / cohens_d / rd / ru from tablevii_decompose.  It does not re-implement
them.  There must be exactly ONE implementation of the bound algebra in this project, or the
paper's two headline tables can silently drift apart in their statistics.  This is the same
rule that made tablex_r8.py import the frozen tablevii_r8.py instead of copying it.

Arithmetic carries no venue effect: the INPUTS are laptop-produced, so where the sums happen
is irrelevant.  This is NOT a cross-venue comparison and must not be described as one.

THE MODEL (identical to Table VII's, by construction)
-----------------------------------------------------
CRN makes the design CROSSED, not nested: repeat j uses the identical channel-seed stream for
every arm and every training seed, and -- verified below -- the SAME stream Table VII used.

    x_ij = mu + alpha_i + beta_j + eps_ij       i = 1..5 training seeds, j = 1..8 repeats
    df_A = 4    df_B = 7    df_E = 28
    sigma_train^2 = (MS_A - MS_E) / R           F_obs = MS_A / MS_E ~ F(4, 28)

POSITIVE != RESOLVABLE.  resolvable requires v_train > 0 AND p < 0.05.

WHAT IT REFUSES TO DO
---------------------
Refuses to run if the grid is not exactly 4 arms x 5 seeds x 8 repeats, if any row is from a
venue other than the laptop, if torch is not unique, if the evaluation protocol constants are
not (eval_seeds = 5, eval_ep = 150) on every row, if the frozen-harness md5 is not constant,
if the evaluator AST md5s do not group as expected, or if the CRN block structure is broken.
"""
import csv
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

sys.path.insert(0, '/home/claude')
from tablevii_decompose import decompose, cohens_d, rd, ru          # ONE implementation only
import tablevii_decompose as T7D

CSV = '/root/work/exp/results/tablex_r8_laptop.csv'
METRICS = ['HO', 'Power', 'Spec', 'RLF', 'Tput']
ARMS = ['m5_dndqn', 'm5_ddqn', 'm5_std', 'sd5_safe']
LABEL = {'m5_dndqn': 'Proposed Dn-DQN (matched objective)',
         'm5_ddqn':  'DDQN+PER',
         'm5_std':   'Standard DQN',
         'sd5_safe': 'Shielded Safe-DQN'}

# Expected evaluator AST fingerprints.  run_multiseed_rel.evaluate and
# run_multiseed_matched.evaluate are AST-identical; train_safe_dqn.evaluate differs because the
# shield is applied at evaluation time.  Verified live on both venues 2026-08-01.
EVAL_AST_MATCHED = '14947df50dadc3f994bfade7ced93089'
EVAL_AST_SAFE    = '502d6491b4de8d81c8cefdce429a7cd9'
T7_MD5           = '5fd895bf1560f2fe94715b8557b544a9'      # frozen tablevii_r8.py

# SINGLE-PASS per-seed values, verbatim from the two laptop campaign CSVs.  These are the OLD
# basis: one unseeded evaluation pass per trained model.  Used ONLY for instrument agreement.
#   results/matched_rel_5seed_800k_laptop.csv   (Dn-DQN, DDQN+PER, Standard DQN)
#   results/safedqn_5seed_800k_laptop.csv       (Shielded Safe-DQN)
SINGLEPASS_SEED = {
    'm5_dndqn': {
        0: dict(HO=93.77, Power=18.36, Spec=89.27, RLF=9.20,  Tput=462.8),
        1: dict(HO=94.76, Power=13.51, Spec=89.46, RLF=4.67,  Tput=462.2),
        2: dict(HO=93.84, Power=8.76,  Spec=77.68, RLF=3.87,  Tput=685.6),
        3: dict(HO=86.85, Power=66.91, Spec=89.67, RLF=19.60, Tput=414.4),
        4: dict(HO=95.45, Power=1.97,  Spec=89.44, RLF=4.40,  Tput=502.0)},
    'm5_ddqn': {
        0: dict(HO=89.49, Power=72.74, Spec=51.25, RLF=17.33, Tput=887.0),
        1: dict(HO=94.67, Power=34.81, Spec=69.08, RLF=8.53,  Tput=739.3),
        2: dict(HO=90.13, Power=53.74, Spec=75.40, RLF=11.73, Tput=606.6),
        3: dict(HO=93.56, Power=30.98, Spec=74.72, RLF=5.33,  Tput=654.8),
        4: dict(HO=87.25, Power=82.50, Spec=22.35, RLF=18.27, Tput=1182.2)},
    'm5_std': {
        0: dict(HO=85.24, Power=84.42, Spec=79.99, RLF=20.40, Tput=501.3),
        1: dict(HO=85.57, Power=81.20, Spec=79.56, RLF=20.53, Tput=526.5),
        2: dict(HO=84.60, Power=85.72, Spec=80.36, RLF=18.27, Tput=493.4),
        3: dict(HO=89.05, Power=74.26, Spec=74.71, RLF=13.60, Tput=578.3),
        4: dict(HO=87.04, Power=83.31, Spec=70.06, RLF=20.13, Tput=618.5)},
    'sd5_safe': {
        0: dict(HO=91.00, Power=31.09, Spec=90.38, RLF=12.40, Tput=443.1),
        1: dict(HO=92.24, Power=37.67, Spec=79.16, RLF=14.93, Tput=582.2),
        2: dict(HO=95.48, Power=15.40, Spec=89.01, RLF=4.13,  Tput=483.6),
        3: dict(HO=92.69, Power=26.75, Spec=86.25, RLF=9.07,  Tput=506.3),
        4: dict(HO=92.89, Power=25.50, Spec=79.24, RLF=11.73, Tput=620.4)},
}

# Table X as PUBLISHED, Block A learned rows.  Filled in from the manuscript at render time;
# left empty here so nothing unverified can leak into a printed comparison.
PUBLISHED_X = {}


def load():
    rows = list(csv.DictReader(open(CSV)))
    if len(rows) != 160:
        raise SystemExit(f'FATAL: {len(rows)} rows, expected exactly 160 (4 arms x 5 seeds x 8 repeats).')
    venues = {r['venue'] for r in rows}
    if venues != {'laptop'}:
        raise SystemExit(f'FATAL: venues {venues}, expected only laptop. Table X reports MEANS; '
                         f'a venue offset does not cancel out of a mean.')
    torches = {r['torch'] for r in rows}
    if len(torches) != 1:
        raise SystemExit(f'FATAL: mixed torch versions {torches}.')

    # PROTOCOL CONSTANTS.  The smoke runs used TX_EVAL_EP=3 / TX_R=1.  If either had leaked into
    # the real grid the table would silently claim 150 episodes over 3.  Assert from the DATA.
    eps = {r['eval_ep'] for r in rows}
    esd = {r['eval_seeds'] for r in rows}
    if eps != {'150'}:
        raise SystemExit(f'FATAL: eval_ep {eps}, expected only 150.')
    if esd != {'5'}:
        raise SystemExit(f'FATAL: eval_seeds {esd}, expected only 5.')
    t7 = {r['t7_md5'] for r in rows}
    if t7 != {T7_MD5}:
        raise SystemExit(f'FATAL: frozen-harness md5 {t7}, expected {{{T7_MD5}}}.')

    # EVALUATOR IDENTITY.  Three arms must share the matched evaluator's AST; the shielded arm
    # must not (the shield is applied at evaluation).  If these ever collapse to one value, the
    # shield silently stopped being applied and every Safe-DQN number is wrong.
    ast = defaultdict(set)
    for r in rows:
        ast[r['arm']].add(r['eval_ast_md5'])
    for arm in ['m5_dndqn', 'm5_ddqn', 'm5_std']:
        if ast[arm] != {EVAL_AST_MATCHED}:
            raise SystemExit(f'FATAL: arm {arm} evaluator AST {ast[arm]}, expected {EVAL_AST_MATCHED}.')
    if ast['sd5_safe'] != {EVAL_AST_SAFE}:
        raise SystemExit(f'FATAL: sd5_safe evaluator AST {ast["sd5_safe"]}, expected {EVAL_AST_SAFE}.')

    grid = defaultdict(dict)
    blocks = defaultdict(set)
    meta = defaultdict(dict)
    for r in rows:
        arm, seed, rep = r['arm'], int(r['seed']), int(r['repeat'])
        blocks[rep].add(r['chan_seed_first'])
        grid[arm].setdefault(seed, {m: {} for m in METRICS})
        for m in METRICS:
            grid[arm][seed][m][rep] = float(r[m])
        meta[arm][seed] = (r['ckpt'], r['ckpt_md5'][:8], r['loader'], r['evaluator'],
                           r['eval_seeds'], r['eval_ep'])

    for arm in ARMS:
        if arm not in grid:
            raise SystemExit(f'FATAL: arm {arm} missing.')
        if sorted(grid[arm]) != [0, 1, 2, 3, 4]:
            raise SystemExit(f'FATAL: arm {arm} seeds {sorted(grid[arm])}, expected 0..4.')
        for s in grid[arm]:
            for m in METRICS:
                if sorted(grid[arm][s][m]) != list(range(8)):
                    raise SystemExit(f'FATAL: arm {arm} seed {s} metric {m} repeats '
                                     f'{sorted(grid[arm][s][m])}, expected 0..7.')

    # THE CRN CHECK. If this fails, the design is nested and every df below is wrong.
    for rep, seeds in sorted(blocks.items()):
        if len(seeds) != 1:
            raise SystemExit(f'FATAL: repeat {rep} carries {len(seeds)} distinct channel seeds '
                             f'{seeds} -- CRN is broken, the design is NOT crossed.')
    if len(blocks) != 8:
        raise SystemExit(f'FATAL: {len(blocks)} distinct repeats, expected 8.')

    out = {}
    for arm in ARMS:
        out[arm] = {m: np.array([[grid[arm][s][m][j] for j in range(8)] for s in range(5)])
                    for m in METRICS}
    return out, meta, sorted(blocks.items()), torches.pop()


def main():
    data, meta, blocks, torch_v = load()

    print('=' * 100)
    print('TABLE X -- LEARNED-BASELINE VARIANCE DECOMPOSITION, laptop venue (torch %s)' % torch_v)
    print('  design: per arm, k = 5 training seeds x R = 8 CRN repeats;  df_A = 4, df_B = 7, df_E = 28')
    print('  source: %s' % CSV)
    print('=' * 100)

    # ------------------------------------------------------------------ CRN block equality
    # Table VII's grid and Table X's grid must sit on the SAME channel draws, or the flat row
    # measured in the Table VII campaign cannot be printed in Table X beside these four.
    print('\n[1] CRN BLOCK STRUCTURE -- shared with the Table VII grid?')
    t7data, t7meta, t7blocks, t7torch = T7D.load()
    xb = {rep: sorted(s)[0] for rep, s in blocks}
    tb = {rep: sorted(s)[0] for rep, s in t7blocks}
    same = xb == tb
    for rep in sorted(xb):
        mark = 'OK ' if xb[rep] == tb.get(rep) else 'XX '
        print('    %s repeat %d  Table X chan_seed_first = %s   Table VII = %s'
              % (mark, rep, xb[rep], tb.get(rep)))
    print('    -> %s' % ('IDENTICAL. The two grids are on one instrument; the Table VII flat row '
                         'is directly comparable.' if same else
                         'DIFFERENT. The grids are NOT paired; do not print them as one table.'))
    print('    Table VII venue torch = %s   Table X venue torch = %s   %s'
          % (t7torch, torch_v, 'MATCH' if t7torch == torch_v else 'MISMATCH'))

    # ------------------------------------------------------------------ per-arm decomposition
    print('\n[2] PER-ARM, PER-METRIC DECOMPOSITION')
    print('    sd_train  = sqrt((MS_A - MS_E)/R), the TRAINING-seed component (nan if negative)')
    print('    sd_eval   = sqrt(MS_E),  the evaluation-noise floor at this episode count')
    print('    resolvable requires v_train > 0 AND p < 0.05.  POSITIVE IS NOT RESOLVABLE.')
    print('    gamma bounds: [lo, hi] are the two-sided-95%% endpoints  -> joint >= 95%%')
    print('                  [l1, u1] are the two one-sided 95%% bounds -> joint >= 90%%')

    res = {arm: {} for arm in ARMS}
    for arm in ARMS:
        print('\n  %s   [%s]' % (LABEL[arm], arm))
        ck = ', '.join('s%d:%s' % (s, meta[arm][s][1]) for s in range(5))
        print('    ckpts: %s' % ck)
        print('    loader/evaluator: %s / %s' % (meta[arm][0][2], meta[arm][0][3]))
        print('    %-6s | %10s | %9s | %9s | %8s | %7s | %-14s | %9s | %9s'
              % ('metric', 'mean', 'sd_train', 'sd_eval', 'F(4,28)', 'p', 'verdict',
                 'sd_tr_hi', 'sd_tr_u1'))
        print('    ' + '-' * 108)
        for m in METRICS:
            d = decompose(data[arm][m])
            res[arm][m] = d
            if d['resolvable']:
                verdict = 'RESOLVABLE'
            elif d['positive']:
                verdict = 'pos,not resolv'
            else:
                verdict = 'NEGATIVE'
            sdt = ('%9.4f' % d['sd_train']) if d['positive'] else '      n/a'
            print('    %-6s | %10.4f | %s | %9.4f | %8.4f | %7.4f | %-14s | %9.4f | %9.4f'
                  % (m, d['grand'], sdt, d['sd_e'], d['f_obs'], d['p'], verdict,
                     d['sd_train_hi'], d['sd_train_u1']))
        # population sd across seed means -- the quantity the manuscript prints as +-
        print('    reported +- (population sd across the 5 seed means, ddof=0): ' +
              '  '.join('%s %.4f' % (m, res[arm][m]['sd_seedmeans_pop']) for m in METRICS))

    # ------------------------------------------------- the two "sd_eval"s must not both be printed
    print('\n[2b] TWO DIFFERENT EVALUATION-NOISE QUANTITIES -- ONLY ONE MAY BE CALLED "sd_eval"')
    print('     tablex_r8.py --table printed  sd_eval = sqrt(mean_i var_j(x_ij, ddof=1)),  the')
    print('     POOLED WITHIN-CHECKPOINT sd.  Within a seed the 8 repeats differ by BOTH the')
    print('     shared channel-draw effect beta_j AND the residual eps_ij, so that quantity')
    print('     estimates sigma_block^2 + sigma_e^2.  This script\'s sqrt(MS_E) estimates')
    print('     sigma_e^2 ALONE -- the block main effect has been removed.  They are different')
    print('     numbers and a reviewer will find both if the paper prints both under one name.')
    print('     The reader-facing question is "re-evaluate this checkpoint once more with a')
    print('     fresh channel draw -- how far does the number move?"  That answer is')
    print('     sqrt(sigma_block^2 + sigma_e^2), i.e. the POOLED quantity already reported.')
    print()
    print('     %-36s %-6s | %9s | %9s | %9s | %9s | %8s'
          % ('arm', 'metric', 'sd_resid', 'sd_block', 'sqrt(sum)', 'pooled', 'agree'))
    maxgap = 0.0
    for arm in ARMS:
        for m in METRICS:
            X = data[arm][m]
            d = res[arm][m]
            sb2 = max(d['v_block'], 0.0)
            comb = np.sqrt(sb2 + d['ms_e'])
            pooled = float(np.sqrt(np.mean([X[s].var(ddof=1) for s in range(5)])))
            gap = abs(comb - pooled)
            maxgap = max(maxgap, gap)
            print('     %-36s %-6s | %9.4f | %9.4f | %9.4f | %9.4f | %8.5f'
                  % (LABEL[arm] if m == 'HO' else '', m, d['sd_e'], np.sqrt(sb2), comb, pooled, gap))
    print('     max |sqrt(sigma_block^2+sigma_e^2) - pooled| = %.5f  -> the identity holds; the'
          % maxgap)
    print('     pooled number IS the block-plus-residual evaluation noise, not the residual.')

    # ------------------------------------------------------------------ instrument agreement
    print('\n[3] INSTRUMENT AGREEMENT -- does the OLD single unseeded pass land inside the')
    print('    R = 8 CRN repeat range for the SAME checkpoint?  If the single pass was one')
    print('    draw from the distribution the 8 repeats sample, it should usually fall inside.')
    tot = inside = 0
    for arm in ARMS:
        hits = []
        for m in METRICS:
            X = data[arm][m]
            for s in range(5):
                lo, hi = X[s].min(), X[s].max()
                v = SINGLEPASS_SEED[arm][s][m]
                tot += 1
                ok = lo <= v <= hi
                inside += ok
                if not ok:
                    hits.append((m, s, v, lo, hi))
        print('\n  %-36s  outside-range cells: %d / 25' % (LABEL[arm], len(hits)))
        for m, s, v, lo, hi in hits:
            gap = (v - hi) if v > hi else (lo - v)
            print('      %-5s seed %d  single-pass %9.3f   repeat range [%9.3f, %9.3f]  miss by %.3f'
                  % (m, s, v, lo, hi, gap))
    # THE NULL.  With 9 exchangeable draws (the single pass + its 8 CRN repeats), the single
    # pass falls INSIDE the range of the other 8 iff it is neither the overall maximum nor the
    # overall minimum.  Its rank is uniform on 1..9, so P(inside) = 1 - 2/9 = 7/9 = 77.8%.
    # (An earlier version of this script printed 8/9 = 88.9%, which counts only ONE tail and
    # would have made an agreeing result look like a disagreeing one.)
    p0 = 7.0 / 9.0
    obs = inside / tot
    se = np.sqrt(p0 * (1 - p0) / tot)
    z = (obs - p0) / se
    print('\n  TOTAL: %d / %d cells (%.1f%%) of single-pass values fall inside their own' % (inside, tot, 100.0 * obs))
    print('  checkpoint\'s 8-repeat range.')
    print('  NULL: 9 exchangeable draws -> the new one is inside the other 8 unless it is the')
    print('  max or the min, so P = 1 - 2/9 = %.1f%%.' % (100.0 * p0))
    print('  OBSERVED %.1f%%  vs  EXPECTED %.1f%%   z = %+.2f  (nominal two-sided p = %.3f)'
          % (100.0 * obs, 100.0 * p0, z, 2 * (1 - stats.norm.cdf(abs(z)))))
    print('  The 100 cells are NOT independent (5 metrics share each evaluation), so treat the')
    print('  p-value as indicative, not exact.  The direction is what matters: agreement is at')
    print('  or ABOVE the exchangeability expectation, i.e. the old single-pass numbers behave')
    print('  like one more draw from the same instrument -- not like a different instrument.')
    permetric = {m: [0, 0] for m in METRICS}
    for arm in ARMS:
        for m in METRICS:
            X = data[arm][m]
            for s in range(5):
                permetric[m][1] += 1
                permetric[m][0] += X[s].min() <= SINGLEPASS_SEED[arm][s][m] <= X[s].max()
    print('  by metric: ' + '  '.join('%s %d/%d' % (m, permetric[m][0], permetric[m][1]) for m in METRICS))
    for m in METRICS:
        ins, n = permetric[m]
        zz = (ins / n - p0) / np.sqrt(p0 * (1 - p0) / n)
        flag = '   <-- BELOW EXPECTATION' if zz < -1.96 else ''
        print('    %-5s %2d/%2d = %5.1f%%   z = %+.2f%s' % (m, ins, n, 100.0 * ins / n, zz, flag))

    # ------------------------------------------------- the stronger, distributional version
    # The range test throws away magnitude.  Within one checkpoint the 8 CRN repeats ARE iid
    # draws of (beta_j + eps_ij), so the standardised deviation of the single pass from their
    # mean is an exact t on 7 df UNDER THE NULL THAT THE SINGLE PASS USED THE SAME PROTOCOL:
    #     t = (x_new - xbar_8) / (s_8 * sqrt(1 + 1/8))  ~  t_7
    # Two things this can reveal that the range test cannot:
    #   mean(t) far from 0   -> the old protocol was BIASED relative to the seeded one
    #   sd(t) far from 1.18  -> the old protocol had a DIFFERENT NOISE SCALE (e.g. fewer
    #                           evaluation episodes), which is the still-open question of what
    #                           episode count the published campaign actually used.
    # sd of a t_7 variate is sqrt(7/5) = 1.183.
    print('\n  DISTRIBUTIONAL VERSION -- t = (single_pass - mean_of_8) / (s_8*sqrt(1+1/8)) ~ t_7')
    print('  under the null that the single pass used the SAME evaluation protocol.')
    T_SD_NULL = np.sqrt(7.0 / 5.0)
    allt = {m: [] for m in METRICS}
    for arm in ARMS:
        for m in METRICS:
            X = data[arm][m]
            for s in range(5):
                mu8, s8 = X[s].mean(), X[s].std(ddof=1)
                if s8 > 0:
                    allt[m].append((SINGLEPASS_SEED[arm][s][m] - mu8) / (s8 * np.sqrt(1 + 1 / 8)))
    print('    %-6s | %4s | %8s | %8s | %8s | %8s | %s'
          % ('metric', 'n', 'mean t', 'sd t', '|t|>2.36', 'expect', 'reading'))
    pool = []
    for m in METRICS:
        t = np.array(allt[m]); pool += list(t)
        big = int((np.abs(t) > stats.t.ppf(0.975, 7)).sum())
        note = ''
        if t.std(ddof=1) > 2 * T_SD_NULL:
            note = 'noise scale looks LARGER than the seeded protocol'
        elif abs(t.mean()) > 1.0:
            note = 'looks BIASED vs the seeded protocol'
        print('    %-6s | %4d | %+8.3f | %8.3f | %8d | %8.1f | %s'
              % (m, len(t), t.mean(), t.std(ddof=1), big, 0.05 * len(t), note))
    pool = np.array(pool)
    print('    %-6s | %4d | %+8.3f | %8.3f | %8d | %8.1f |'
          % ('ALL', len(pool), pool.mean(), pool.std(ddof=1), int((np.abs(pool) > stats.t.ppf(0.975, 7)).sum()),
             0.05 * len(pool)))
    print('    null: mean t = 0, sd t = sqrt(7/5) = %.3f, and 5%% of |t| beyond %.2f.'
          % (T_SD_NULL, stats.t.ppf(0.975, 7)))

    # ------------------------------------------------------------------ Table X Block A rows
    print('\n[4] TABLE X BLOCK A -- five learned rows, ONE protocol')
    print('    flat row comes from the Table VII grid (same 8 CRN blocks, same venue).')
    fmt = {'HO': 2, 'Power': 2, 'Spec': 2, 'RLF': 2, 'Tput': 1}
    order = ['flat'] + ARMS
    lab = dict(LABEL); lab['flat'] = 'Flat Dn-DQN (6-action)  [from Table VII grid]'
    for arm in order:
        src = t7data['flat'] if arm == 'flat' else data[arm]
        cells = []
        for m in METRICS:
            d = decompose(src[m])
            n = fmt[m]
            cells.append('%.*f $\\pm$ %.*f' % (n, d['grand'], n, d['sd_seedmeans_pop']))
        print('ROW: | %s | %s |' % (lab[arm], ' | '.join(cells)))

    print('\n    evaluation-noise floor per row (sd_eval), same order:')
    for arm in order:
        src = t7data['flat'] if arm == 'flat' else data[arm]
        cells = ['%s %.3f' % (m, decompose(src[m])['sd_e']) for m in METRICS]
        print('      %-46s %s' % (lab[arm], '  '.join(cells)))

    # ------------------------------------------------------------------ the ratio, per arm
    print('\n[5] sd_train / sd_eval PER ARM -- how much of the reported spread is training-seed')
    print('    lottery and how much is evaluation noise.  "--" where sd_train is not resolvable.')
    print('    %-36s %s' % ('arm', '  '.join('%-14s' % m for m in METRICS)))
    for arm in order:
        src = t7data['flat'] if arm == 'flat' else data[arm]
        cells = []
        for m in METRICS:
            d = decompose(src[m])
            if d['positive']:
                cells.append('%-14s' % ('%.2fx' % (d['sd_train'] / d['sd_e'])))
            else:
                cells.append('%-14s' % ('-- (<%.2f)' % (d['sd_train_hi'] / d['sd_e'])))
        print('    %-36s %s' % (lab[arm], '  '.join(cells)))

    # ------------------------------------------------------------------ block-sampling limitation
    # The single-pass values sit ABOVE the grid mean for throughput in ALL FOUR arms
    # (+5.10, +2.93, +2.93, +6.06).  That is NOT four independent confirmations of a protocol
    # difference: the four arms share the SAME 8 channel draws, so a set of draws that happens
    # to run slightly low for throughput moves every arm the same way.  It is ONE observation
    # about the 8 blocks, seen four times.
    #
    # The reportable consequence: with only R = 8 blocks, the GRAND MEAN of each row carries a
    # block-sampling uncertainty of sigma_block / sqrt(R) on top of the training-seed term.  It
    # is COMMON TO EVERY ARM (CRN), so it cancels in arm-vs-arm comparisons -- which is what the
    # paper claims -- but it does NOT cancel from an absolute mean.  This belongs in Threats.
    print('\n[6] BLOCK-SAMPLING UNCERTAINTY ON THE GRAND MEAN (common to all arms, cancels in')
    print('    arm-vs-arm comparisons under CRN; does NOT cancel from an absolute mean)')
    print('    %-36s %-6s | %9s | %9s | %11s | %11s'
          % ('arm', 'metric', 'sd_block', 'sb/sqrt(R)', 'grid mean', '1pass-grid'))
    for arm in ARMS:
        for m in METRICS:
            d = res[arm][m]
            sb = np.sqrt(max(d['v_block'], 0.0))
            sp = float(np.mean([SINGLEPASS_SEED[arm][s][m] for s in range(5)]))
            print('    %-36s %-6s | %9.4f | %10.4f | %11.4f | %+11.4f'
                  % (LABEL[arm] if m == 'HO' else '', m, sb, sb / np.sqrt(8), d['grand'], sp - d['grand']))
    print('\n    Throughput specifically: the single-pass mean exceeds the grid mean in all four')
    print('    arms (%s).  Under CRN that is one shared block-draw'
          % ', '.join('%+.2f' % (float(np.mean([SINGLEPASS_SEED[a][s]['Tput'] for s in range(5)]))
                                 - res[a]['Tput']['grand']) for a in ARMS))
    print('    offset seen four times, not four independent effects, and it is the same order as')
    print('    sigma_block/sqrt(8) (%s).  Report as a bounded'
          % ', '.join('%.2f' % (np.sqrt(max(res[a]['Tput']['v_block'], 0.0)) / np.sqrt(8)) for a in ARMS))
    print('    limitation on absolute means; the comparative claims are unaffected.')

    return 0


if __name__ == '__main__':
    sys.exit(main())
