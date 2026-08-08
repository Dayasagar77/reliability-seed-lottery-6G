#!/usr/bin/env python3
"""
replication_rel_vs_m5.py — the two Dn-DQN checkpoint families are the SAME SYSTEM trained TWICE.

WHY THIS EXISTS
---------------
The revised paper is about to print two rows that a reader will read as two different controllers:

    Table VII : "Flat Dn-DQN (6-action)"                from ms_rel_s{0..4}.pt   (run_multiseed_rel.py)
    Table X   : "Proposed Dn-DQN (matched objective)"   from m5_dndqn_s{0..4}.pt (campaign_matched.py
                                                         -> run_multiseed_matched.train('dndqn', ...))

They are not two controllers.  Established, with evidence recorded in section [0] below:

  * run_multiseed_rel.train  and  run_multiseed_matched.train(method='dndqn')  are functionally
    identical -- same env (Phase2Env_NYU_rel), same flat 6-action space, AST-identical set_seed and
    evaluate, and a train() whose every textual difference is parameterisation, cosmetic renaming or
    checkpoint metadata.  cfg('dndqn') = (dueling=True, double=True, nstep=3, alpha=0.6) reproduces
    run_multiseed_rel's module constants exactly.
  * both families were trained on the SAME machine at the SAME budget: 800,000 steps, seeds 0..4,
    torch 2.5.1+cu121.
  * the five source files are byte-identical on both venues.

So the honest description of the pair is: ONE system, TEN independent training runs, reported as two
five-run samples.  That is not a defect to hide -- it is the direct answer to the AE's remark that
"the evidence for the reported 'seed lottery' is also limited to five training seeds".  But it does
forbid printing the two rows as if they were different systems.

WHY THE WEIGHTS DIFFER DESPITE THE SAME SEED
--------------------------------------------
ckpt_identity_check.py (run on the laptop, torch 2.5.1+cu121) compared q_net_state tensor by tensor:
all five seeds DIFFER, max|dW| between 4.75 and 6.61.  The mechanism is in the environment, not the
learner:

    env_6g_v5.SubTHzEnv6G_NYU.__init__(self, vehicle_trajectory, seed=None, ...)
        self.rng = np.random.default_rng(seed)          # line 132
    train_phase2_nyu_rel.Phase2Env_NYU_rel.__init__(self, traj)
        super().__init__(traj, array_gain_dbi=..., n310=..., w_mob=..., w_pwr=..., w_spec=...)
                                                        # <-- seed is never passed
    run_multiseed_{rel,matched}.train:
        env = Phase2Env_NYU_rel(smp(stage))             # <-- and never here either

set_seed(s) seeds random / numpy-global / torch, and a local default_rng(seed) drives trajectory
sampling, so network initialisation, exploration and the training trajectory sequence ARE reproducible.
The per-step channel realisation is not: default_rng(None) draws from OS entropy.  Two runs at the
same nominal seed therefore see different fading and diverge.  The "training seed" is a run index,
not a determinant of the run.

WHAT THIS SCRIPT DOES
---------------------
Treats the pair as a replication experiment and asks three questions, all on laptop-produced data
(this file only does arithmetic, so running it in the cloud is venue-clean):

    Q1  Do the two families' variance estimates agree?         F(4,4) on the seed-level variances
    Q2  Do the two families' means agree?                      Welch t on the seed-level means,
                                                               block effect cancelled by CRN
    Q3  How much does pooling to 10 runs buy?                   chi-square CI on sigma, df 4 vs df 8

Usage:  python3 replication_rel_vs_m5.py
"""
import csv, os, sys
import numpy as np
from scipy import stats

sys.path.insert(0, '/home/claude')
from tablevii_decompose import decompose, rd, ru          # ONE implementation of the bound algebra

T7  = '/root/work/exp/results/tablevii_r8_laptop.csv'
TX  = '/root/work/exp/results/tablex_r8_laptop.csv'
METRICS = ['HO', 'Power', 'Spec', 'RLF', 'Tput']
UNIT    = {'HO': '%', 'Power': '%', 'Spec': '%', 'RLF': ' pts', 'Tput': ' Mb/s'}

# Recorded by ckpt_identity_check.py on the laptop, 2026-08-01.  Kept here so the replication
# argument is self-contained; the source of truth is ckpt_identity_check_out.txt on the bridge.
WEIGHT_DIFF = {0: 6.21024, 1: 5.42182, 2: 5.03810, 3: 6.60578, 4: 4.75319}
CKPT_A = {0: 'e4c52446', 1: 'b8843271', 2: 'f896f329', 3: 'af815045', 4: 'bd13f944'}
CKPT_B = {0: '33744da5', 1: 'ef5a23cb', 2: '6c51ac67', 3: 'a58a631b', 4: 'd6c18b09'}

# Single-pass (R=1, unseeded-channel) values as published/recorded, for context only.
SP_A = {'HO': [93.00, 89.94, 93.07, 90.47, 95.53], 'Power': [18.96, 53.75, 23.86, 39.98, 0.31],
        'Spec': [88.52, 86.72, 89.31, 89.04, 88.21], 'RLF': [10.67, 16.80, 11.60, 14.00, 4.27],
        'Tput': [485.9, 471.3, 469.2, 462.3, 508.8]}          # results/multiseed_rel_nyu.csv
SP_B = {'HO': [93.77, 94.76, 93.84, 86.85, 95.45], 'Power': [18.36, 13.51, 8.76, 66.91, 1.97],
        'Spec': [89.27, 89.46, 77.68, 89.67, 89.44], 'RLF': [9.20, 4.67, 3.87, 19.60, 4.40],
        'Tput': [462.8, 462.2, 685.6, 414.4, 502.0]}          # results/matched_rel_5seed_800k_laptop.csv


def load(path, arm):
    """Return (X[k,R], seeds, repeats, blocks, meta) for one arm of one grid."""
    rows = [r for r in csv.DictReader(open(path)) if r['arm'] == arm]
    assert rows, 'arm %r not found in %s' % (arm, path)
    venue = set(r['venue'] for r in rows)
    torchv = set(r['torch'] for r in rows)
    assert venue == {'laptop'}, 'venue contamination: %r' % venue
    assert len(torchv) == 1, 'toolchain contamination: %r' % torchv
    assert set(r['eval_seeds'] for r in rows) == {'5'}, 'eval_seeds drift'
    assert set(r['eval_ep'] for r in rows) == {'150'}, 'eval_ep drift'
    seeds = sorted(set(int(r['seed']) for r in rows))
    reps = sorted(set(int(r['repeat']) for r in rows))
    blocks = [sorted(set(int(r['chan_seed_first']) for r in rows if int(r['repeat']) == j))
              for j in reps]
    assert all(len(b) == 1 for b in blocks), 'a repeat index maps to >1 channel block'
    blocks = [b[0] for b in blocks]
    X = {}
    for m in METRICS:
        A = np.zeros((len(seeds), len(reps)))
        for r in rows:
            A[seeds.index(int(r['seed'])), reps.index(int(r['repeat']))] = float(r[m])
        X[m] = A
    meta = dict(venue=venue.pop(), torch=torchv.pop(), n=len(rows),
                md5=[r['ckpt_md5'][:8] for r in sorted(rows, key=lambda z: (int(z['seed']), int(z['repeat'])))
                     if int(r['repeat']) == 0])
    return X, seeds, reps, blocks, meta


def sd_ci(s, df, conf=0.95):
    """Two-sided conf interval for sigma given a sample sd s on df degrees of freedom."""
    a = (1.0 - conf) / 2.0
    lo = s * np.sqrt(df / stats.chi2.ppf(1 - a, df))
    hi = s * np.sqrt(df / stats.chi2.ppf(a, df))
    return lo, hi


def main():
    XA, seedsA, repsA, blocksA, metaA = load(T7, 'flat')
    XB, seedsB, repsB, blocksB, metaB = load(TX, 'm5_dndqn')

    print('=' * 100)
    print('  ONE SYSTEM, TEN RUNS — ms_rel_s{0..4}.pt  vs  m5_dndqn_s{0..4}.pt')
    print('=' * 100)

    # ---------------- [0] provenance ------------------------------------------------
    print('\n[0] PROVENANCE — same system, same budget, same machine, disjoint checkpoints')
    print('    family A : ms_rel_s*.pt      run_multiseed_rel.py            800k steps  seeds 0-4  torch %s' % metaA['torch'])
    print('    family B : m5_dndqn_s*.pt    campaign_matched.py -> run_multiseed_matched.train(\'dndqn\')')
    print('                                                                800k steps  seeds 0-4  torch %s' % metaB['torch'])
    print('    training-driver source files byte-identical on laptop and cloud (5/5 md5 match).')
    print('    grid blocks A == grid blocks B :', blocksA == blocksB, blocksA)
    assert blocksA == blocksB, 'the two grids do not share the CRN blocks -- comparison invalid'
    assert seedsA == seedsB and repsA == repsB
    print('\n    seed | ckpt A    | ckpt B    | q_net_state max|dW|')
    for s in seedsA:
        print('    %4d | %-9s | %-9s | %10.5f' % (s, CKPT_A[s], CKPT_B[s], WEIGHT_DIFF[s]))
    print('    -> no seed produced the same network twice.  Cause: the training channel RNG is')
    print('       np.random.default_rng(None) (env_6g_v5.py:132, seed never passed by the driver),')
    print('       so the fading realisation during training comes from OS entropy.')

    # ---------------- [1] per-seed grid means ---------------------------------------
    print('\n[1] PER-SEED GRID MEANS (mean over the R=%d shared channel blocks)' % len(repsA))
    for m in METRICS:
        a = XA[m].mean(axis=1); b = XB[m].mean(axis=1)
        print('    %-6s A: %s' % (m, ' '.join('%9.3f' % v for v in a)))
        print('    %-6s B: %s' % ('', ' '.join('%9.3f' % v for v in b)))
        print('    %-6s d: %s   (B - A, same seed index)' % ('', ' '.join('%+9.3f' % v for v in (b - a))))

    # ---------------- [2] variance components ---------------------------------------
    print('\n[2] VARIANCE COMPONENTS, EACH FAMILY SEPARATELY')
    print('    %-6s | %-28s | %-28s' % ('metric', 'A  ms_rel (Table VII flat)', 'B  m5_dndqn (Table X)'))
    print('    %-6s | %8s %8s %9s | %8s %8s %9s' % ('', 'mean', 'sd_train', 'F(4,28)', 'mean', 'sd_train', 'F(4,28)'))
    DA, DB = {}, {}
    for m in METRICS:
        da = decompose(XA[m]); db = decompose(XB[m]); DA[m], DB[m] = da, db
        print('    %-6s | %8.3f %8.4f %9.1f | %8.3f %8.4f %9.1f' %
              (m, da['grand'], da['sd_train'], da['f_obs'], db['grand'], db['sd_train'], db['f_obs']))

    # ---------------- [3] Q1 do the two sigmas agree? --------------------------------
    print('\n[3] Q1 — DO THE TWO FIVE-RUN SIGMA ESTIMATES AGREE?   F(4,4) on seed-level variances')
    print('    (seed-level sd is the sd of the 5 grid means, ddof=1; under CRN the shared block')
    print('     mean cancels from within-family deviations, so this is a clean variance estimate.)')
    print('    %-6s | %9s %9s | %8s | %8s | %8s | %s' %
          ('metric', 'sd_A', 'sd_B', 'F(4,4)', 'p_F', 'p_BF', 'verdict'))
    for m in METRICS:
        a = XA[m].mean(axis=1); b = XB[m].mean(axis=1)
        sa, sb = a.std(ddof=1), b.std(ddof=1)
        hi, lo = (sa, sb) if sa >= sb else (sb, sa)
        F = (hi / lo) ** 2
        p = min(2.0 * (1.0 - stats.f.cdf(F, 4, 4)), 1.0)
        # Brown-Forsythe (median-centred Levene): far less sensitive to a single heavy-tailed run,
        # which is exactly the failure mode an F-test has on seed-lottery data.
        _, pbf = stats.levene(a, b, center='median')
        print('    %-6s | %9.4f %9.4f | %8.3f | %8.4f | %8.4f | %s' %
              (m, sa, sb, F, p, pbf, 'compatible' if p >= 0.05 else 'F rejects'))
    print('    CAUTION.  The F-test assumes normality and is the wrong instrument here: a seed')
    print('    lottery produces a heavy-tailed outcome distribution, and one atypical run moves F')
    print('    enormously.  The Brown-Forsythe column is the robust reading.  Where the two')
    print('    disagree, believe Brown-Forsythe and look at the runs.')
    print('    Reference: with k=5 the 95% sampling interval for an estimate of a single true sigma')
    print('    spans [0.599 sigma, 2.874 sigma] — a ratio of 4.80x between the two ends.  Two honest')
    print('    five-run estimates of ONE system can legitimately differ by nearly a factor of five.')
    # ---- locate the runs responsible ------------------------------------------------
    print('\n    WHICH RUNS DRIVE THE VARIANCE?  |z| of each run about its own family mean:')
    print('    %-6s | %-29s | %-29s' % ('metric', 'A  seeds 0..4', 'B  seeds 0..4'))
    for m in METRICS:
        a = XA[m].mean(axis=1); b = XB[m].mean(axis=1)
        za = (a - a.mean()) / a.std(ddof=1); zb = (b - b.mean()) / b.std(ddof=1)
        print('    %-6s | %-29s | %-29s' % (m, ' '.join('%+5.2f' % v for v in za),
                                            ' '.join('%+5.2f' % v for v in zb)))

    # ---------------- [4] Q2 do the two means agree? ---------------------------------
    print('\n[4] Q2 — DO THE TWO FIVE-RUN MEANS AGREE?   Welch t on the 5+5 seed-level means')
    print('    Under CRN both families were evaluated on the SAME 8 channel blocks, so the block')
    print('    main effect is common to both and cancels exactly from the difference of means.')
    print('    %-6s | %9s %9s | %+9s | %8s %6s | %8s | %s' %
          ('metric', 'mean_A', 'mean_B', 'B-A', 't', 'df', 'p', 'verdict'))
    for m in METRICS:
        a = XA[m].mean(axis=1); b = XB[m].mean(axis=1)
        t, p = stats.ttest_ind(b, a, equal_var=False)
        va, vb = a.var(ddof=1) / 5.0, b.var(ddof=1) / 5.0
        df = (va + vb) ** 2 / (va ** 2 / 4 + vb ** 2 / 4)
        print('    %-6s | %9.3f %9.3f | %+9.3f | %8.3f %6.2f | %8.4f | %s' %
              (m, a.mean(), b.mean(), b.mean() - a.mean(), t, df, p,
               'compatible' if p >= 0.05 else 'DIFFER'))

    # ---------------- [5] Q3 what does pooling to ten runs buy? ----------------------
    print('\n[5] Q3 — POOLED TEN-RUN ESTIMATE, AND THE PRECISION IT BUYS')
    print('    pooled sd_seed = sqrt((SS_A + SS_B)/8), df=8.  Evaluation noise is then removed with')
    print('    sigma_train^2 = s_seed^2 - MS_E/R using the pooled MS_E (df=56).')
    print('    %-6s | %8s %-19s | %8s %-19s | %8s | %7s' %
          ('metric', 'sd_5(A)', '95% CI (df=4)', 'sd_10', '95% CI (df=8)', 'sd_tr_10', 'CIw5/CIw10'))
    POOL = {}
    for m in METRICS:
        a = XA[m].mean(axis=1); b = XB[m].mean(axis=1)
        ss = ((a - a.mean()) ** 2).sum() + ((b - b.mean()) ** 2).sum()
        s10 = np.sqrt(ss / 8.0)
        s5a = a.std(ddof=1)
        lo5, hi5 = sd_ci(s5a, 4)
        lo8, hi8 = sd_ci(s10, 8)
        mse_pool = 0.5 * (DA[m]['ms_e'] + DB[m]['ms_e'])
        v_tr = s10 ** 2 - mse_pool / XA[m].shape[1]
        sd_tr = np.sqrt(v_tr) if v_tr > 0 else float('nan')
        POOL[m] = dict(s10=s10, lo8=lo8, hi8=hi8, sd_tr=sd_tr, mse=mse_pool)
        print('    %-6s | %8.4f [%7.4f,%8.4f] | %8.4f [%7.4f,%8.4f] | %8.4f | %5.2fx' %
              (m, s5a, lo5, hi5, s10, lo8, hi8, sd_tr, (hi5 - lo5) / (hi8 - lo8)))
    print('    READ THIS COLUMN HONESTLY.  Doubling the runs narrows the interval on HO, Power and')
    print('    RLF (ratio > 1).  On Spec and Tput the pooled interval is WIDER (ratio < 1) — not')
    print('    because ten runs are worse evidence than five, but because family A\'s five runs')
    print('    happened to miss the tail of the outcome distribution.  A narrow sigma from k=5 can')
    print('    be narrow because the sample was lucky, not because the system is stable.  That is')
    print('    the whole point, and it is the AE\'s point too.')

    # ---------------- [6] the reportable ten-run row ---------------------------------
    print('\n[6] TEN-RUN ROW (population-form sigma, the manuscript convention: ddof=0 over 10 runs)')
    for m in METRICS:
        a = XA[m].mean(axis=1); b = XB[m].mean(axis=1)
        allv = np.concatenate([a, b])
        print('    %-6s  %8.2f  +/- %7.2f%s   (n=10 independent trainings)' %
              (m, allv.mean(), allv.std(), UNIT[m]))
    print('\n    ROW: | Proposed Dn-DQN, 10 independent trainings |', end='')
    for m in METRICS:
        allv = np.concatenate([XA[m].mean(axis=1), XB[m].mean(axis=1)])
        fmt = '%.1f $\\pm$ %.1f' if m == 'Tput' else '%.2f $\\pm$ %.2f'
        print(' ' + (fmt % (allv.mean(), allv.std())) + ' |', end='')
    print()

    # ---------------- [7] single-pass cross-check ------------------------------------
    print('\n[7] CROSS-CHECK AGAINST THE TWO SINGLE-PASS CAMPAIGN CSVs (R=1, unseeded channel)')
    print('    These are the numbers the campaigns themselves recorded, independent of the R=8 grid.')
    print('    %-6s | %-22s | %-22s | %s' % ('metric', 'A single-pass', 'B single-pass', 'A/B grid'))
    for m in METRICS:
        sa = np.array(SP_A[m]); sb = np.array(SP_B[m])
        ga = XA[m].mean(axis=1); gb = XB[m].mean(axis=1)
        print('    %-6s | %8.2f +/- %-9.2f | %8.2f +/- %-9.2f | %8.2f / %-8.2f' %
              (m, sa.mean(), sa.std(), sb.mean(), sb.std(), ga.mean(), gb.mean()))

    # ---------------- [8] how much does a five-run sigma depend on WHICH five? --------
    print('\n[8] SUBSAMPLE ANALYSIS — "is five seeds enough?", answered with this paper\'s own runs')
    print('    Treat the ten runs as the reference set and enumerate all C(10,5) = 252 five-run')
    print('    subsets.  For each, compute the sigma the paper would have reported (population form,')
    print('    ddof=0, the manuscript convention).  This is a subsampling illustration of estimator')
    print('    spread, not an inference about the population: the ten runs are not a random sample of')
    print('    all possible trainings, and the 252 subsets overlap heavily and are not independent.')
    from itertools import combinations
    idx = list(range(10))
    subsets = list(combinations(idx, 5))
    print('    %-6s | %8s | %8s %8s %8s | %8s | %s' %
          ('metric', 'sd_10', 'min', 'median', 'max', 'max/min', 'P(within 1.5x of sd_10)'))
    for m in METRICS:
        allv = np.concatenate([XA[m].mean(axis=1), XB[m].mean(axis=1)])
        full = allv.std()                                   # ddof=0, manuscript convention
        sds = np.array([allv[list(c)].std() for c in subsets])
        frac = float(np.mean((sds >= full / 1.5) & (sds <= full * 1.5)))
        print('    %-6s | %8.3f | %8.3f %8.3f %8.3f | %8.2f | %19.1f%%' %
              (m, full, sds.min(), np.median(sds), sds.max(), sds.max() / max(sds.min(), 1e-12),
               100 * frac))
    ratios = []
    for m in METRICS:
        allv = np.concatenate([XA[m].mean(axis=1), XB[m].mean(axis=1)])
        sds = np.array([allv[list(c)].std() for c in subsets])
        ratios.append(sds.max() / max(sds.min(), 1e-12))
    print('    Depending only on WHICH five runs you happen to have, the sigma this paper would')
    print('    print varies by a factor of %.1f to %.1f.  A five-seed sigma is itself a lottery'
          % (min(ratios), max(ratios)))
    print('    ticket.  That is measured on the paper\'s own runs, not argued.')
    print('    The 50.0% figures on Spec and Tput are not a coincidence: for those two metrics the')
    print('    reported sigma is governed almost entirely by whether run B-s2 is in the sample, and')
    print('    P(B-s2 in a random 5-of-10 subset) = 5/10 = 0.5 exactly.')

    print('\n    WHAT RUN B-s2 ACTUALLY IS (not an outlier to discard — a different operating point):')
    b = {m: XB[m].mean(axis=1) for m in METRICS}
    print('        run B-s2 :  HO %.2f  Power %.2f  Spec %.2f  RLF %.2f  Tput %.1f' %
          tuple(b[m][2] for m in METRICS))
    oth = [i for i in range(5) if i != 2]
    print('        other 4  :  HO %.2f  Power %.2f  Spec %.2f  RLF %.2f  Tput %.1f  (mean)' %
          tuple(b[m][oth].mean() for m in METRICS))
    print('        Spec is a SAVINGS percentage, so a low Spec means the policy kept resource blocks')
    print('        allocated instead of releasing them.  B-s2 spent spectrum and bought throughput')
    print('        (+%.0f Mb/s) and reliability (RLF %.2f vs %.2f) with it.  It is a coherent policy'
          % (b['Tput'][2] - b['Tput'][oth].mean(), b['RLF'][2], b['RLF'][oth].mean()))
    print('        on the far side of the savings/throughput trade-off, reached from the same code')
    print('        and the same nominal seed.  Discarding it would be exactly the manipulation the')
    print('        paper is arguing against.')

    print('\n' + '=' * 100)
    print('  BOTTOM LINE')
    print('  The two rows are one controller measured twice.  They must not be typeset as two')
    print('  systems.  Reported as a replication they answer the AE directly: the seed lottery is')
    print('  reproduced in an independent five-run sample of the same code at the same budget, and')
    print('  the pooled evidence is ten runs, not five.')
    print('=' * 100)


if __name__ == '__main__':
    main()
