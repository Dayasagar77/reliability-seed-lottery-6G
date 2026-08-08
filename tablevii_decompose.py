#!/usr/bin/env python3
"""
tablevii_decompose.py -- the variance decomposition for Table VII, on REPORTING-VENUE data.

WHY THIS EXISTS
---------------
The Associate Editor's second criticism was, verbatim:

    "some baselines are represented by a single trained model and others by evaluation-seed
     variability, making the comparisons methodologically inconsistent"

That is correct and it is fatal to Table VII as submitted. The fix is not an argument, it is a
measurement: every learned arm re-evaluated under ONE protocol -- k = 5 training seeds x R = 8
common-random-number evaluation repeats -- so that every reported +/- in the table means the
same thing (spread across TRAINING seeds) and the evaluation noise is separated out rather than
silently mixed in.

The prior decomposition was computed on the cloud (torch 2.13). Table VII reports MEANS, and a
venue offset does not cancel out of a mean. So the whole 160-evaluation grid was re-run on the
laptop, which is the reporting venue. This script does the arithmetic. Arithmetic carries no
venue effect: the INPUTS are laptop-produced, so where the sums happen is irrelevant. This is
NOT a cross-venue comparison and must not be described as one.

THE MODEL
---------
CRN makes the design CROSSED, not nested: repeat j uses the identical channel-seed stream for
every arm and every training seed. So

    x_ij = mu + alpha_i + beta_j + eps_ij        i = 1..k training seeds, j = 1..R repeats

with alpha ~ (0, sigma_train^2) the training-seed effect, beta ~ (0, sigma_block^2) the shared
channel-draw effect, and eps ~ (0, sigma_e^2). Two-way random effects, no replication:

    df_A = k-1 = 4      df_B = R-1 = 7      df_E = (k-1)(R-1) = 28
    sigma_train^2 = (MS_A - MS_E) / R
    F_obs = MS_A / MS_E ~ F(4, 28) under H0: sigma_train^2 = 0

Exact pivot for gamma = sigma_train^2 / sigma_e^2:   F_obs / (1 + R*gamma) ~ F(4, 28)
    one-sided UPPER bound at 1-a :  gamma_U = (F_obs / q_a     - 1) / R
    one-sided LOWER bound at 1-a :  gamma_L = (F_obs / q_{1-a} - 1) / R
so the endpoints of the two-sided 95% interval ARE the two 97.5% one-sided bounds. That matters
for the ratio claim: pairing two 95% one-sided bounds gives only joint >= 90% by Bonferroni,
whereas pairing the two-sided-95% endpoints gives joint >= 95%. The paper quotes the latter.

A NEGATIVE sigma_train^2 estimate is not a bug and must not be clipped and forgotten: it means
the between-seed mean square is smaller than the residual, i.e. the training-seed component is
NOT RESOLVABLE at this k and R. The honest report is "not resolvable, bounded above by X".

WHAT IT REFUSES TO DO
---------------------
It refuses to run at all if the grid is not exactly 4 arms x 5 seeds x 8 repeats, if any row is
from a venue other than the laptop, if the torch version is not unique, or if the CRN block
structure is broken (repeat j must carry the same channel seed everywhere -- otherwise the
design is nested, not crossed, and every df below is wrong).
"""
import csv
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

CSV = '/root/work/exp/results/tablevii_r8_laptop.csv'
METRICS = ['HO', 'Power', 'Spec', 'RLF', 'Tput']
ARMS = ['flat', 'incremental', 'binary', 'guarded']
LABEL = {'flat': 'Flat Dn-DQN (6-action)',
         'incremental': 'Branched incremental (BDQ-style)',
         'binary': 'Branched binary, unguarded',
         'guarded': 'Branched binary + state guard (proposed)'}

# Table VII as PUBLISHED (mean, sd).  VERIFIED verbatim against DnDQN_v15_IEEE_source.md:315-345
# and manuscript_in.md:375 on 2026-08-01.  An earlier draft of this script carried a WRONG
# 'binary' row (94.29/61.35/78.71/7.32/600.3) which produced a false "+149.8% mean shift" alarm.
PUBLISHED = {
    'flat':        {'HO': (92.40, 2.02), 'Power': (27.37, 18.28), 'Spec': (88.36, 0.91),
                    'RLF': (11.47, 4.18), 'Tput': (479.5, 16.5)},
    'incremental': {'HO': (93.15, 3.30), 'Power': (26.88, 15.20), 'Spec': (54.99, 29.94),
                    'RLF': (9.04, 4.23), 'Tput': (981.6, 445.5)},
    'binary':      {'HO': (89.11, 3.88), 'Power': (68.14, 19.43), 'Spec': (72.36, 28.09),
                    'RLF': (18.59, 6.06), 'Tput': (599.6, 325.2)},
    'guarded':     {'HO': (95.18, 0.45), 'Power': (63.61, 4.38), 'Spec': (80.63, 17.02),
                    'RLF': (4.61, 0.92), 'Tput': (505.9, 223.5)},
}

# The published Welch tests of Findings 2 and 3, for side-by-side comparison.
# (arm_a, arm_b, metric) -> (t, p, d) as printed in the submitted manuscript.
#
# SIGN CONVENTION: the manuscript always writes the PROPOSED arm first, so every t and d is
# (proposed - other). An earlier draft of this script keyed the last two as ('binary','guarded')
# and ('incremental','guarded'), which produced t = +4.80 and +2.88 against published -4.56 and
# -2.04 -- the same tests with the sign flipped, which looks like a reversal and is not one.
PUBLISHED_TESTS = {
    ('guarded', 'flat', 'RLF'):        (-3.20, 0.0288, -2.03),
    ('guarded', 'flat', 'Power'):      (+3.86, 0.0148, +2.44),
    ('guarded', 'flat', 'HO'):         (+2.68, 0.0499, +1.70),
    ('guarded', 'binary', 'RLF'):      (-4.56, 0.0093, -2.88),
    ('guarded', 'incremental', 'RLF'): (-2.04, 0.1044, -1.29),
}

# Table IV as PUBLISHED -- the per-seed flat ("seed lottery") table, read verbatim from
# DnDQN_manuscript_v13.md:210-226 on 2026-08-01. These are the SAME five checkpoints
# (ms_rel_s0..s4.pt) as Table VII row 1, so a direct per-seed published-vs-measured comparison
# is licensed: same weights, same metric definitions, different evaluation protocol.
PUBLISHED_TABLE_IV = {
    0: {'HO': 93.00, 'Power': 18.96, 'Spec': 88.52, 'RLF': 10.67, 'Tput': 485.9},
    1: {'HO': 89.94, 'Power': 53.75, 'Spec': 86.72, 'RLF': 16.80, 'Tput': 471.3},
    2: {'HO': 93.07, 'Power': 23.86, 'Spec': 89.31, 'RLF': 11.60, 'Tput': 469.2},
    3: {'HO': 90.47, 'Power': 39.98, 'Spec': 89.04, 'RLF': 14.00, 'Tput': 462.3},
    4: {'HO': 95.53, 'Power':  0.31, 'Spec': 88.21, 'RLF':  4.27, 'Tput': 508.8},
}

ORACLE_CEILING = 84.51   # % of steps SAVE-able at zero reliability cost, G = 9 dB (Sec. III-C)


def load():
    rows = list(csv.DictReader(open(CSV)))
    if len(rows) != 160:
        raise SystemExit(f'FATAL: {len(rows)} rows, expected exactly 160 (4 arms x 5 seeds x 8 repeats).')
    venues = {r['venue'] for r in rows}
    if venues != {'laptop'}:
        raise SystemExit(f'FATAL: venues {venues}, expected only laptop. Table VII reports MEANS; '
                         f'a venue offset does not cancel out of a mean.')
    torches = {r['torch'] for r in rows}
    if len(torches) != 1:
        raise SystemExit(f'FATAL: mixed torch versions {torches}.')

    grid = defaultdict(dict)          # grid[arm][seed] -> {metric: {repeat: value}}
    blocks = defaultdict(set)         # repeat -> set of chan_seed_first seen
    meta = defaultdict(dict)
    for r in rows:
        arm, seed, rep = r['arm'], int(r['seed']), int(r['repeat'])
        blocks[rep].add(r['chan_seed_first'])
        grid[arm].setdefault(seed, {m: {} for m in METRICS})
        for m in METRICS:
            grid[arm][seed][m][rep] = float(r[m])
        meta[arm][seed] = (r['ckpt'], r['ckpt_origin'], r['ckpt_md5'][:8],
                           r['binary_power'], r['guard_db'], r['guard_mode'],
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


def decompose(X, conf=0.95):
    """X is k x R. Two-way random effects, no replication, crossed by CRN."""
    k, R = X.shape
    gm = X.mean()
    row, col = X.mean(axis=1), X.mean(axis=0)

    ss_a = R * ((row - gm) ** 2).sum()
    ss_b = k * ((col - gm) ** 2).sum()
    ss_t = ((X - gm) ** 2).sum()
    ss_e = ss_t - ss_a - ss_b

    df_a, df_b, df_e = k - 1, R - 1, (k - 1) * (R - 1)
    ms_a, ms_b, ms_e = ss_a / df_a, ss_b / df_b, ss_e / df_e

    v_train = (ms_a - ms_e) / R                       # may be negative -> not resolvable
    v_block = (ms_b - ms_e) / k
    f_obs = ms_a / ms_e
    p = 1.0 - stats.f.cdf(f_obs, df_a, df_e)

    a = 1.0 - conf
    # Pivot: F_obs / (1 + R*gamma) ~ F(df_a, df_e).
    #   UPPER bound at conf 1-a :  gamma <= (F_obs / ppf(a)   - 1)/R
    #   LOWER bound at conf 1-a :  gamma >= (F_obs / ppf(1-a) - 1)/R
    # BOTH one-sided bounds are needed at BOTH levels. An earlier version computed only the 95%
    # UPPER (g_u1) and then paired it with the 97.5% LOWER to form the "joint >= 90%" ratio -- a
    # mismatched pairing that printed 15.86 under a label claiming two 95% bounds. The correct
    # joint->=90% pairing is g_l1 with g_u1. (The error was conservative, not inflationary, and
    # the joint->=95% claim was never affected because it pairs g_lo with g_hi correctly.)
    g_u1 = (f_obs / stats.f.ppf(a, df_a, df_e) - 1.0) / R          # 95%   one-sided UPPER
    g_l1 = (f_obs / stats.f.ppf(1 - a, df_a, df_e) - 1.0) / R      # 95%   one-sided LOWER
    q_lo, q_hi = stats.f.ppf(a / 2, df_a, df_e), stats.f.ppf(1 - a / 2, df_a, df_e)
    g_lo, g_hi = (f_obs / q_hi - 1.0) / R, (f_obs / q_lo - 1.0) / R
    # g_lo is simultaneously the 97.5% one-sided LOWER bound; g_hi the 97.5% one-sided UPPER.

    sd_e = np.sqrt(ms_e)
    return dict(
        k=k, R=R, grand=gm, seed_means=row, block_means=col,
        sd_seedmeans_pop=row.std(ddof=0), sd_seedmeans_samp=row.std(ddof=1),
        ms_a=ms_a, ms_b=ms_b, ms_e=ms_e, f_obs=f_obs, p=p,
        v_train=v_train, sd_train=np.sqrt(v_train) if v_train > 0 else float('nan'),
        v_block=v_block, sd_block=np.sqrt(v_block) if v_block > 0 else float('nan'),
        sd_e=sd_e,
        # POSITIVE means the moment estimator did not go negative. RESOLVABLE means more: the
        # F test actually rejects sigma_train^2 = 0. Guarded RLF has F = 1.017, so its v_train
        # is positive by a hair and sqrt() of it is 0.008 -- a number that READS as a precision
        # claim and is not one. Printing "4.91 +- 0.01" would assert a training-seed spread we
        # cannot distinguish from zero. The honest render is "4.91 (sigma_train < bound)".
        positive=v_train > 0,
        resolvable=bool(v_train > 0 and p < 0.05),
        gamma_hat=v_train / ms_e,
        gamma_u1=g_u1, gamma_l1=max(g_l1, 0.0), gamma_lo=max(g_lo, 0.0), gamma_hi=g_hi,
        sd_train_u1=sd_e * np.sqrt(max(g_u1, 0.0)),      # 95%  one-sided upper
        sd_train_l1=sd_e * np.sqrt(max(g_l1, 0.0)),      # 95%  one-sided lower
        sd_train_lo=sd_e * np.sqrt(max(g_lo, 0.0)),      # 97.5% one-sided lower
        sd_train_hi=sd_e * np.sqrt(max(g_hi, 0.0)),      # 97.5% one-sided upper
        df=(df_a, df_b, df_e),
    )


def cohens_d(a, b):
    """Pooled-sd effect size, ddof = 1 (the manuscript's convention for d)."""
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else float('nan')


def welch_pair(data, store, arm_a, arm_b, metric):
    """Welch t on the k=5 SEED MEANS, plus a CRN-corrected variant.

    The naive Welch SE^2 = s_A^2/k + s_B^2/k covers sigma_train^2/k + sigma_e^2/(kR) for each
    arm, but omits Var(betabar_A - betabar_B): the two arms share channel-seed blocks, so their
    arm means carry a common (and partly cancelling) block offset. That term is estimated
    directly from the per-block arm-mean difference and added, giving a strictly conservative
    p-value. If the naive and corrected tests disagree, the corrected one is what gets reported.
    """
    A, B = data[arm_a][metric], data[arm_b][metric]      # k x R each
    ma, mb = A.mean(axis=1), B.mean(axis=1)              # seed means, k = 5
    k, R = A.shape

    t, p = stats.ttest_ind(ma, mb, equal_var=False)
    d = cohens_d(ma, mb)

    # --- CRN correction -------------------------------------------------------------------
    # b_j = (arm-A block-j mean) - (arm-B block-j mean).  Var_j(b_j) = Var(beta_A - beta_B)
    #       + (sigma_eA^2 + sigma_eB^2)/k, so subtract the known residual part and divide by R.
    b = A.mean(axis=0) - B.mean(axis=0)                  # length R
    var_b = b.var(ddof=1)
    resid = (store[arm_a][metric]['ms_e'] + store[arm_b][metric]['ms_e']) / k
    v_blockdiff = max((var_b - resid) / R, 0.0)

    va, vb = ma.var(ddof=1) / k, mb.var(ddof=1) / k
    se_naive = np.sqrt(va + vb)
    se_corr = np.sqrt(va + vb + v_blockdiff)
    t_corr = (ma.mean() - mb.mean()) / se_corr if se_corr > 0 else float('nan')
    # Satterthwaite df, treating the block term as a single variance estimate on R-1 df.
    num = (va + vb + v_blockdiff) ** 2
    den = va ** 2 / (k - 1) + vb ** 2 / (k - 1) + (v_blockdiff ** 2 / (R - 1) if v_blockdiff > 0 else 0.0)
    df_corr = num / den if den > 0 else float('nan')
    p_corr = 2 * (1 - stats.t.cdf(abs(t_corr), df_corr)) if np.isfinite(df_corr) else float('nan')

    return dict(mean_a=ma.mean(), mean_b=mb.mean(), sd_a=ma.std(ddof=0), sd_b=mb.std(ddof=0),
                t=t, p=p, d=d, se_naive=se_naive, se_corr=se_corr, v_blockdiff=v_blockdiff,
                t_corr=t_corr, p_corr=p_corr, df_corr=df_corr, seed_means_a=ma, seed_means_b=mb)


def rd(x, n=1):
    """Outward rounding DOWN -- for a lower bound or a separation margin."""
    f = 10.0 ** n
    return np.floor(x * f) / f


def ru(x, n=1):
    """Outward rounding UP -- for an upper bound."""
    f = 10.0 ** n
    return np.ceil(x * f) / f


def main():
    data, meta, blocks, torch_v = load()
    print('=' * 118)
    print('  TABLE VII VARIANCE DECOMPOSITION -- REPORTING VENUE (laptop)')
    print(f'  source : {CSV}  md5 2234f1ccfd763518a0027712274f22a8')
    print(f'  grid   : 4 arms x 5 training seeds x 8 CRN repeats = 160 evaluations, torch {torch_v}')
    print(f'  CRN    : repeat -> channel seed  ' +
          ', '.join(f'{r}:{list(s)[0]}' for r, s in blocks))
    print('  model  : x_ij = mu + alpha_i + beta_j + eps_ij   df_A=4  df_B=7  df_E=28')
    print('=' * 118)

    print('\n  ---- checkpoint provenance, per arm ----')
    for arm in ARMS:
        c = meta[arm][0]
        print(f'  {arm:>12s}  binary_power={c[3] or "-":<5s} guard_db={c[4] or "-":<4s} '
              f'guard_mode={c[5] or "-":<6s} eval={c[6]}x{c[7]} ep')
        print(f'  {"":>12s}  ' + '  '.join(f's{s}:{meta[arm][s][0]}({meta[arm][s][1][:11]})'
                                           for s in range(5)))

    store = {}
    for arm in ARMS:
        print('\n' + '-' * 118)
        print(f'  ARM: {arm}')
        print('-' * 118)
        print(f'  {"metric":<7s} {"mean":>9s} {"sd_train":>10s} {"sd_eval":>9s} {"sd_blk":>8s} '
              f'{"F(4,28)":>9s} {"p":>9s} {"sd_train 95%CI":>20s} {"published":>16s}')
        store[arm] = {}
        for m in METRICS:
            d = decompose(data[arm][m])
            store[arm][m] = d
            pm, ps = PUBLISHED[arm][m]
            sdt = f'{d["sd_train"]:.4f}' if d['resolvable'] else f'<{d["sd_train_hi"]:.4f}'
            print(f'  {m:<7s} {d["grand"]:9.4f} {sdt:>10s} {d["sd_e"]:9.4f} '
                  f'{(np.sqrt(max(d["v_block"],0))):8.4f} '
                  f'{d["f_obs"]:9.3f} {d["p"]:9.5f} '
                  f'[{d["sd_train_lo"]:8.4f},{d["sd_train_hi"]:9.4f}] '
                  f'{pm:8.2f}+-{ps:<6.2f}')
        for m in METRICS:
            print(f'  per-seed means ({m:>5s}): ' +
                  '  '.join(f'{v:9.3f}' for v in store[arm][m]['seed_means']))

    # ---------------- the headline: RLF spread ratio, flat vs guarded ----------------
    print('\n' + '=' * 118)
    print('  THE HEADLINE CLAIM: training-seed spread in RLF, flat vs guarded')
    print('=' * 118)
    f, g = store['flat']['RLF'], store['guarded']['RLF']
    print(f'  flat     sd_train = {f["sd_train"]:.4f}   (F = {f["f_obs"]:.3f}, p = {f["p"]:.6f}, '
          f'resolvable = {f["resolvable"]})')
    print(f'           95%   one-sided LOWER bound = {f["sd_train_l1"]:.4f}')
    print(f'           97.5% one-sided LOWER bound = {f["sd_train_lo"]:.4f}   '
          f'(this is the one the headline claim uses)')
    print(f'  guarded  sd_train {"= %.4f" % g["sd_train"] if g["resolvable"] else "NOT RESOLVABLE"}   '
          f'(F = {g["f_obs"]:.3f}, p = {g["p"]:.6f})')
    print(f'           MS_A = {g["ms_a"]:.5f} is SMALLER than MS_E = {g["ms_e"]:.5f}')
    print(f'           95%   one-sided UPPER bound = {g["sd_train_u1"]:.4f} RLF points')
    print(f'           97.5% one-sided UPPER bound = {g["sd_train_hi"]:.4f} RLF points')

    r90 = f['sd_train_l1'] / g['sd_train_u1'] if g['sd_train_u1'] > 0 else float('inf')
    r95 = f['sd_train_lo'] / g['sd_train_hi'] if g['sd_train_hi'] > 0 else float('inf')
    print()
    print('  THE RATIO, at two joint confidence levels (Bonferroni over the two bounds):')
    print(f'    pairing two 95%   one-sided bounds -> joint >= 90% : '
          f'{r90:8.4f}  ->  "at least {rd(r90):.1f}x"')
    print(f'    pairing two 97.5% one-sided bounds -> joint >= 95% : '
          f'{r95:8.4f}  ->  "at least {rd(r95):.1f}x"   <== THE CLAIM')
    print(f'  Both numerator and denominator round OUTWARD (lower bound down, upper bound up),')
    print(f'  so the quoted ratio is floor()-ed: it can only understate the separation.')
    pr = f['sd_train'] / g['sd_train'] if g['positive'] else float('nan')
    print(f'\n  point ratio of sd_train estimates : {pr:.3f}   '
          f'(NOT reportable: the guarded denominator is 0.008, an estimate that the F test')
    print(f'                                                cannot distinguish from zero. '
          f'Quote the bounded ratio, never this.)')
    print(f'  naive ratio of seed-mean sds (pop): '
          f'{f["sd_seedmeans_pop"]/g["sd_seedmeans_pop"]:.3f}   '
          f'(also not reportable: both numerator and denominator still carry sigma_e/sqrt(8))')

    # ---------------- published vs measured, flagged ----------------
    print('\n' + '=' * 118)
    print('  PUBLISHED vs MEASURED -- every cell, with the discrepancy flagged')
    print('=' * 118)
    print(f'  {"arm":<12s} {"metric":<7s} {"pub mean":>9s} {"new mean":>9s} {"d_mean":>9s} '
          f'{"rel%":>7s} {"pub sd":>8s} {"new sd_train":>13s} {"sd(seedmeans)":>14s}  flag')
    for arm in ARMS:
        for m in METRICS:
            d = store[arm][m]
            pm, ps = PUBLISHED[arm][m]
            dm = d['grand'] - pm
            rel = 100.0 * dm / pm if pm else float('nan')
            sdt = f'{d["sd_train"]:.3f}' if d['resolvable'] else f'<{d["sd_train_hi"]:.3f}'
            flag = ''
            if abs(rel) > 20:
                flag = '<== MEAN MOVED >20%'
            elif abs(rel) > 5:
                flag = '<-- mean moved >5%'
            print(f'  {arm:<12s} {m:<7s} {pm:9.2f} {d["grand"]:9.2f} {dm:+9.2f} {rel:+7.1f} '
                  f'{ps:8.2f} {sdt:>13s} {d["sd_seedmeans_pop"]:14.3f}  {flag}')

    # ---------------- Table IV: the per-seed published-vs-measured comparison ----------------
    print('\n' + '=' * 118)
    print('  THE FLAT ARM, SEED BY SEED -- published Table IV vs the 160-evaluation grid')
    print('=' * 118)
    print('  Same five checkpoints (ms_rel_s0..s4.pt), same metric definitions, different')
    print('  evaluation protocol. So any per-seed delta is a PROTOCOL effect, not a training')
    print('  effect -- the weights are byte-identical. This is the only place in the paper where')
    print('  the old and new protocols can be compared on a fixed policy.')

    # Integrity check: Table IV must reproduce Table VII row 1 exactly. If my transcription of
    # Table IV is wrong, this fails loudly rather than quietly poisoning the comparison below.
    print('\n  [integrity] Table IV per-seed values must reproduce published Table VII row 1:')
    ok = True
    for m in METRICS:
        col = np.array([PUBLISHED_TABLE_IV[s][m] for s in range(5)])
        pm, ps = PUBLISHED['flat'][m]
        dp = 1 if m == 'Tput' else 2
        good_m = abs(col.mean() - pm) <= 0.5 * 10 ** -dp + 1e-9
        good_s = abs(col.std(ddof=0) - ps) <= 0.5 * 10 ** -dp + 1e-9
        ok = ok and good_m and good_s
        print(f'    {m:<6s} mean {col.mean():8.3f} vs published {pm:8.2f}  {"ok" if good_m else "MISMATCH"}'
              f'   |  sigma_pop {col.std(ddof=0):7.3f} vs published {ps:6.2f}  '
              f'{"ok" if good_s else "MISMATCH"}')
    if not ok:
        raise SystemExit('FATAL: transcribed Table IV does not reproduce published Table VII row 1. '
                         'Re-read the manuscript before trusting anything below.')
    print('    -> Table IV transcription verified. The comparison below is on solid ground.')

    R_ = store['flat']['RLF']['R']
    for m in METRICS:
        d = store['flat'][m]
        se = d['sd_e'] / np.sqrt(R_)        # standard error of a seed mean in the NEW grid
        print(f'\n  {m}   (new-grid standard error on each seed mean = sigma_e/sqrt(R) = {se:.4f})')
        print(f'    {"seed":>4s} {"published":>10s} {"measured":>10s} {"delta":>9s} '
              f'{"delta/SE":>9s}')
        deltas = []
        for s in range(5):
            pv = PUBLISHED_TABLE_IV[s][m]
            mv = d['seed_means'][s]
            dl = mv - pv
            deltas.append(dl)
            print(f'    s{s:<3d} {pv:10.2f} {mv:10.3f} {dl:+9.3f} {dl/se:+9.1f}')
        deltas = np.array(deltas)
        signs = 'all negative' if (deltas < 0).all() else \
                'all positive' if (deltas > 0).all() else \
                f'{int((deltas < 0).sum())} negative / {int((deltas > 0).sum())} positive'
        print(f'    mean delta {deltas.mean():+.3f}   sign pattern: {signs}')

    print('\n  ---- is the RLF shift structured, or is it noise? ----')
    rlf_d = np.array([store['flat']['RLF']['seed_means'][s] - PUBLISHED_TABLE_IV[s]['RLF']
                      for s in range(5)])
    pw_pub = np.array([PUBLISHED_TABLE_IV[s]['Power'] for s in range(5)])
    r_, p_ = stats.pearsonr(pw_pub, rlf_d)
    rs_, ps_ = stats.spearmanr(pw_pub, rlf_d)
    print('    Hypothesis under test: the shift tracks CLIFF EXPOSURE. A seed that saves power')
    print('    operates near the reliability cliff, where a hairline numerical difference can')
    print('    flip an argmax; a seed that saves no power sits far from it and cannot move.')
    print(f'    {"seed":>4s} {"published power saving":>23s} {"RLF delta":>10s}')
    for s in np.argsort(pw_pub):
        print(f'    s{s:<3d} {pw_pub[s]:23.2f} {rlf_d[s]:+10.3f}')
    print(f'    Pearson  r = {r_:+.4f}  p = {p_:.4f}   (n = 5)')
    print(f'    Spearman rho = {rs_:+.4f}  p = {ps_:.4f}')
    print('    NOTE: n = 5 -- this correlation is descriptive, not a test. What actually carries')
    print('    the "structured" claim is the sign pattern plus the delta/SE column above: the')
    print('    deltas are tens of new-grid standard errors wide, so they are not measurement')
    print('    noise, and the one seed that does not move is the one that saves ~0% power.')
    print('    The CAUSE is NOT established. The original evaluation ledger for these five')
    print('    checkpoints has not been located, so the old protocol\'s venue and episode count')
    print('    are unknown. The manuscript must state the shift and state that it is unexplained.')

    print('\n  ---- how many distinct effects is this? (checked against the metric code) ----')
    print('    train_srdndqn.py:evaluate() defines the five metrics exactly as:')
    print('      Spec  = (RB_MAX - mean(rb_count)) / (RB_MAX - RB_MIN) * 100')
    print('      Tput  = mean over steps of  rb_count * RB_BW_MHz * log2(1 + SINR_linear)')
    print('      RLF   = 100 * (episodes ending in sustained_rlf) / episodes')
    print('      Power = (PT_MAX - mean(pt_dbm)) / (PT_MAX - PT_MIN) * 100')
    print('    and env_6g_v5.py:_compute_sinr(d_m, is_los, velocity_ms) takes NO rb_count '
          'argument,')
    print('    while RLF fires on rsrp_dbm < -92 dBm, which is also independent of rb_count.')
    print('    So the resource-block channel and the reliability channel are MECHANICALLY')
    print('    INDEPENDENT in this model. An earlier draft of this block asserted that holding')
    print('    fewer RBs concentrates power and thereby lowers RLF. That mechanism DOES NOT')
    print('    EXIST here and the claim is withdrawn.')

    dspec = np.array([store['flat']['Spec']['seed_means'][s] - PUBLISHED_TABLE_IV[s]['Spec']
                      for s in range(5)])
    dtput = np.array([store['flat']['Tput']['seed_means'][s] - PUBLISHED_TABLE_IV[s]['Tput']
                      for s in range(5)])
    RB_MAX_, RB_MIN_ = 66.0, 10.0
    span = RB_MAX_ - RB_MIN_
    rb_pub = RB_MAX_ - (span / 100.0) * np.array([PUBLISHED_TABLE_IV[s]['Spec'] for s in range(5)])
    rb_new = RB_MAX_ - (span / 100.0) * store['flat']['Spec']['seed_means']
    tput_pub = np.array([PUBLISHED_TABLE_IV[s]['Tput'] for s in range(5)])
    # Tput is LINEAR in rb_count at fixed SINR, so the RB change predicts a proportional change.
    pred_dtput = tput_pub * (rb_new / rb_pub - 1.0)
    print(f'\n    {"seed":>4s} {"mean RBs pub":>13s} {"mean RBs new":>13s} {"dTput predicted":>16s} '
          f'{"dTput observed":>15s} {"residual":>10s}')
    for s in range(5):
        print(f'    s{s:<3d} {rb_pub[s]:13.3f} {rb_new[s]:13.3f} {pred_dtput[s]:+16.3f} '
              f'{dtput[s]:+15.3f} {dtput[s]-pred_dtput[s]:+10.3f}')
    share = 100.0 * pred_dtput.sum() / dtput.sum() if dtput.sum() else float('nan')
    print(f'    Mean predicted {pred_dtput.mean():+.3f} vs observed {dtput.mean():+.3f} '
          f'-> the RB change accounts for {share:.0f}% of the throughput drop.')
    print('    (Approximate: Tput averages a PRODUCT over steps, so mean(rb)*mean(log2 term) is')
    print('     not exactly mean(rb*log2 term). It is indicative, not an identity.)')
    print('    The residual has to come from the log2(1 + SINR) factor. SINR here is a function')
    print('    of distance, LOS state and velocity only -- no action the policy can take moves')
    print('    it. So a shifted SINR distribution means the two protocols EVALUATED DIFFERENT')
    print('    GEOMETRY: a different draw of trajectory segments, not a different policy.')
    print('    That single difference would also move RLF, because RLF fires on an RSRP')
    print('    threshold that geometry drives directly -- and it would move it most on the')
    print('    seeds already sitting nearest -92 dBm, which are the power savers. Seed 4 saves')
    print('    0.3% power, sits far from the threshold, and does not move (+0.047, +0.2 SE).')
    print('    STATUS: hypothesis, not result. It is consistent with every sign in the table')
    print('    above and it explains why one seed is exempt, but it is NOT confirmed, because')
    print('    the original evaluation ledger has not been located and the old protocol\'s')
    print('    episode draw is therefore unknown. The manuscript states the shift and states')
    print('    that its cause is not established. Nothing here may be written as a finding.')

    # ---------------- the replacement Table VII ----------------
    print('\n' + '=' * 118)
    print('  THE REPLACEMENT TABLE VII (mean +- sigma_train; "<x" = not resolvable, bounded above)')
    print('=' * 118)
    print('  A cell renders as "mean +- sigma_train" only when F(4,28) rejects sigma_train^2 = 0')
    print('  at p < 0.05. Otherwise it renders as "mean (sigma_train < x)", where x is the UPPER')
    print('  ENDPOINT OF THE TWO-SIDED 95% INTERVAL (equivalently the 97.5% one-sided upper bound)')
    print('  rounded UP -- the same bound the headline ratio uses, so the paper quotes one number')
    print('  for one quantity throughout.')
    print('  | Controller | HO (%) | Power (%) | Spec (%) | RLF (%) | Tput (Mbps) |')
    for arm in ARMS:
        cells = []
        for m in METRICS:
            d = store[arm][m]
            dp = 1 if m == 'Tput' else 2
            if d['resolvable']:
                cells.append(f'{d["grand"]:.{dp}f} +- {d["sd_train"]:.{dp}f}')
            else:
                cells.append(f'{d["grand"]:.{dp}f} (sigma_train < {ru(d["sd_train_hi"], 2):.2f})')
        print(f'  | {LABEL[arm]:<42s} | ' + ' | '.join(cells) + ' |')
    print('\n  ...and, for comparison, the same table with sigma = population sd of the five seed')
    print('  means (the manuscript\'s existing convention, still carrying sigma_e/sqrt(8)):')
    for arm in ARMS:
        cells = []
        for m in METRICS:
            d = store[arm][m]
            dp = 1 if m == 'Tput' else 2
            cells.append(f'{d["grand"]:.{dp}f} +- {d["sd_seedmeans_pop"]:.{dp}f}')
        print(f'  | {LABEL[arm]:<42s} | ' + ' | '.join(cells) + ' |')

    # ---------------- Findings 1-4 recomputation ----------------
    print('\n' + '=' * 118)
    print('  FINDINGS 1-4 -- every number that must change')
    print('=' * 118)

    gd = {m: store['guarded'][m] for m in METRICS}
    print('\n  FINDING 1 (A3-class reliability while saving most of the resources)')
    print(f'    RLF            4.61 +- 0.92   ->  {gd["RLF"]["grand"]:.2f} '
          f'(sigma_train not resolvable, < {ru(gd["RLF"]["sd_train_hi"],2):.2f})')
    print(f'    Power saving  63.61 +- 4.38   ->  {gd["Power"]["grand"]:.2f} +- '
          f'{gd["Power"]["sd_train"]:.2f}')
    print(f'    Spectrum      80.63           ->  {gd["Spec"]["grand"]:.2f} +- '
          f'{gd["Spec"]["sd_train"]:.2f}')
    print(f'    Throughput   505.9  +- 223.5  ->  {gd["Tput"]["grand"]:.1f} +- '
          f'{gd["Tput"]["sd_train"]:.1f}')
    frac = 100.0 * gd['Power']['grand'] / ORACLE_CEILING
    print(f'    fraction of the {ORACLE_CEILING}% oracle ceiling: 75.3%  ->  {frac:.1f}%')
    print(f'    A3 row (3GPP reference, deterministic rule, evaluation-seed spread) is NOT in this')
    print(f'    grid and keeps its published values: HO 97.14 +- 0.79, RLF 3.73 +- 1.44, Tput 1902.')
    print(f'    Its +- is evaluation-seed spread BY CONSTRUCTION -- a deterministic rule has no')
    print(f'    training seed -- so it must be visually separated and excluded from every test.')

    print('\n  FINDING 2 (the lottery collapses on the guarded axes) -- flat vs guarded')
    print(f'    sigma collapse, on sigma_train (evaluation noise removed):')
    for m, old_f, old_g in (('RLF', 4.18, 0.92), ('Power', 18.28, 4.38), ('HO', 2.02, 0.45)):
        df_, dg_ = store['flat'][m], store['guarded'][m]
        gs = f'{dg_["sd_train"]:.2f}' if dg_['resolvable'] else f'<{ru(dg_["sd_train_hi"],2):.2f}'
        print(f'      {m:<6s}  published {old_f:6.2f} -> {old_g:5.2f}   '
              f'measured {df_["sd_train"]:6.2f} -> {gs}')
    print('    Welch tests on the k = 5 seed means (each now an average of R = 8 CRN repeats):')
    print(f'    {"pair":<28s} {"mean_a":>8s} {"mean_b":>8s} {"t":>7s} {"p":>8s} {"d":>7s}'
          f' | {"t_crn":>7s} {"p_crn":>8s} {"df":>5s} | published')
    for (a, b, m), pub in PUBLISHED_TESTS.items():
        w = welch_pair(data, store, a, b, m)
        print(f'    {a+" vs "+b+" ["+m+"]":<28s} {w["mean_a"]:8.2f} {w["mean_b"]:8.2f} '
              f'{w["t"]:+7.2f} {w["p"]:8.4f} {w["d"]:+7.2f}'
              f' | {w["t_corr"]:+7.2f} {w["p_corr"]:8.4f} {w["df_corr"]:5.1f}'
              f' | t={pub[0]:+.2f} p={pub[1]:.4f} d={pub[2]:+.2f}')

    print('\n  FINDING 3 (both ingredients are necessary)')
    b_, g_ = store['binary'], store['guarded']
    i_ = store['incremental']
    print(f'    binary unguarded RLF   18.59 +- 6.06  ->  {b_["RLF"]["grand"]:.2f} +- '
          f'{b_["RLF"]["sd_train"]:.2f}')
    print(f'    BDQ RLF                 9.04 +- 4.23  ->  {i_["RLF"]["grand"]:.2f} +- '
          f'{i_["RLF"]["sd_train"]:.2f}   (regenerated seeds 2-4)')
    print(f'    BDQ spectrum sigma          29.94     ->  {i_["Spec"]["sd_train"]:.2f}')
    print(f'    proposed spectrum sigma     17.02     ->  {g_["Spec"]["sd_train"]:.2f}')
    print(f'    power: proposed 63.61 +- 4.38 vs BDQ 26.88 +- 15.20  ->  '
          f'{g_["Power"]["grand"]:.2f} +- {g_["Power"]["sd_train"]:.2f} vs '
          f'{i_["Power"]["grand"]:.2f} +- {i_["Power"]["sd_train"]:.2f}')

    print('\n  FINDING 4 (spectrum saving keeps a wide spread in the proposed arm)')
    sp = store['guarded']['Spec']['seed_means']
    tp = store['guarded']['Tput']['seed_means']
    lo = int(np.argmin(sp))
    print(f'    published: "one seed lands at 47.26% with throughput 931.8 Mbps"')
    print(f'    measured : per-seed spectrum saving  ' + '  '.join(f'{v:.2f}' for v in sp))
    print(f'               per-seed throughput       ' + '  '.join(f'{v:.1f}' for v in tp))
    print(f'    lowest-spectrum seed = s{lo}: {sp[lo]:.2f}% spectrum, {tp[lo]:.1f} Mbps')

    # ---------------- what the published +- actually was ----------------
    print('\n' + '=' * 118)
    print('  WHY THE PUBLISHED +- WAS THE WRONG QUANTITY (the AE criticism, quantified)')
    print('=' * 118)
    print('  Each published seed contributed ONE unseeded evaluation, so the published sigma')
    print('  estimated sqrt(sigma_train^2 + sigma_eval_single^2), not sigma_train.')
    print(f'  {"arm":<12s} {"metric":<7s} {"pub sd":>8s} {"sigma_train":>12s} '
          f'{"sigma_eval_1draw":>17s} {"combined":>9s}  verdict')
    for arm in ARMS:
        for m in METRICS:
            d = store[arm][m]
            _, ps = PUBLISHED[arm][m]
            sd1 = np.sqrt(max(d['v_block'], 0.0) + d['ms_e'])      # a single unseeded draw
            st = d['sd_train'] if d['resolvable'] else 0.0
            comb = np.sqrt(st ** 2 + sd1 ** 2)
            if not d['resolvable']:
                verdict = 'sigma_train NOT resolvable -- published sd is eval noise'
            elif sd1 > 0.5 * st:
                verdict = 'eval noise material'
            else:
                verdict = 'sigma_train dominates -- published sd approx. right'
            sts = f'{st:.3f}' if d['resolvable'] else 'n/r'
            print(f'  {arm:<12s} {m:<7s} {ps:8.2f} {sts:>12s} {sd1:17.3f} {comb:9.3f}  {verdict}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
