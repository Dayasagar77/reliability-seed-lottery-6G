#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
q3_analyse.py - the pre-registered decision rules for the 2x2 assignment ablation, applied to
                whatever units exist. It decides; it does not argue.

Hypotheses, fixed on 08 Aug 2026 BEFORE any of these arms was trained:

  H1  among the four guarded cells, g_bi (measurement-derived) attains the LOWEST sigma_train(RLF)
  H2  sigma_train(RLF) for g_ib (inverted) is STRICTLY GREATER than for g_bi, with the lower bound
      of g_ib exceeding the upper bound of g_bi at >= 95% joint coverage        <- THE FALSIFIER
  H3  g_bb does not attain lower sigma_train(RLF) than g_bi

  H2 holds        -> the measurement-derived-assignment claim is supported and may be stated, in
                     the narrowed form of section 8 of the pre-registration
  H2 fails        -> the claim is FALSE. Report as a pre-registered null; the manuscript reverts
  H2 inconclusive -> reported as a bound. The claim is NOT made
  H3 fails        -> the mechanism is binarisation, not assignment; reframe and withdraw

The variance component and its bounds use the same estimators as the published analysis:
  two-way crossed random effects over trainings x CRN blocks,
      sigma_train = sqrt(max(MS_A - MS_E, 0) / R)
  one-sided bounds via the exact pivot F_obs / (1 + R*gamma) ~ F(dfA, dfE),
      gamma = sigma_train^2 / sigma_e^2
identical to ratio174_provenance.decompose/bounds, so a bound printed here is the same kind of
object as the 0.17 in Table 1.

All five metrics are printed for every arm, whatever the verdict, as section 9 requires.
"""
import os, sys, csv, glob, math, json

METRICS = ['HO', 'Power', 'Spec', 'RLF', 'Tput']
EVAL_DIR = 'results/q3_assignment_eval'
PUBLISHED_GBI = ('results/tablevii_r8', 'unit_guarded_s*_*.csv')

ARM_ORDER = ['g_ii', 'g_bi', 'g_bb', 'g_ib']
LABEL = {
    'g_ii': 'incremental power + incremental spectrum',
    'g_bi': 'binary power + incremental spectrum  (PROPOSED)',
    'g_bb': 'binary power + binary spectrum',
    'g_ib': 'incremental power + binary spectrum  (INVERTED)',
}

try:
    from scipy import stats
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


def load(patterns):
    by = {}
    for d, pat in patterns:
        for p in sorted(glob.glob(os.path.join(d, pat))):
            for row in csv.DictReader(open(p, newline='', encoding='utf-8-sig')):
                try:
                    s = int(row['seed'])
                except (KeyError, ValueError):
                    continue
                by.setdefault(s, []).append({m: float(row[m]) for m in METRICS})
    return by


def arm_files(arm):
    if arm == 'g_bi':
        if glob.glob(os.path.join(EVAL_DIR, 'unit_g_bi_s*.csv')):
            return [(EVAL_DIR, 'unit_g_bi_s*.csv')]
        return [PUBLISHED_GBI]
    return [(EVAL_DIR, 'unit_%s_s*.csv' % arm)]


def decompose(by, metric):
    seeds = sorted(by)
    if len(seeds) < 2:
        return None
    R = min(len(by[s]) for s in seeds)
    if R < 2:
        return None
    Y = [[by[s][r][metric] for r in range(R)] for s in seeds]
    k = len(Y)
    gm = sum(sum(r) for r in Y) / (k * R)
    rm = [sum(r) / R for r in Y]
    cm = [sum(Y[i][j] for i in range(k)) / k for j in range(R)]
    ms_a = R * sum((m - gm) ** 2 for m in rm) / (k - 1)
    ss_e = sum((Y[i][j] - rm[i] - cm[j] + gm) ** 2 for i in range(k) for j in range(R))
    ms_e = ss_e / ((k - 1) * (R - 1))
    var = (ms_a - ms_e) / R
    return dict(k=k, R=R, mean=gm, per_seed=rm, ms_a=ms_a, ms_e=ms_e,
                F=(ms_a / ms_e) if ms_e > 0 else float('inf'),
                dfA=k - 1, dfE=(k - 1) * (R - 1),
                sigma=math.sqrt(var) if var > 0 else 0.0, var=var)


def bounds(d, alpha=0.025):
    """One-sided (1-alpha) bounds on sigma_train. alpha=0.025 gives 95% joint coverage
    when one arm's lower bound is compared with another arm's upper bound."""
    if not HAVE_SCIPY or d is None or not math.isfinite(d['F']):
        return None, None
    q_lo = stats.f.ppf(alpha, d['dfA'], d['dfE'])
    q_hi = stats.f.ppf(1 - alpha, d['dfA'], d['dfE'])
    g_u = (d['F'] / q_lo - 1) / d['R']
    g_l = (d['F'] / q_hi - 1) / d['R']
    se2 = d['ms_e']
    return math.sqrt(max(g_l, 0) * se2), math.sqrt(max(g_u, 0) * se2)


def main():
    print('=' * 104)
    print('  2x2 ASSIGNMENT ABLATION - pre-registered decision rules (08 Aug 2026)')
    print('  guard held constant at 9 dB in every cell; only the actuator granularity varies')
    print('=' * 104)
    if not HAVE_SCIPY:
        print('  scipy is not importable - point estimates only, no bounds, so H2 cannot be '
              'decided. Install scipy and rerun.')

    res, incomplete = {}, []
    for arm in ARM_ORDER:
        by = load(arm_files(arm))
        if len(by) < 5:
            incomplete.append('%s (seeds %s)' % (arm, sorted(by)))
            continue
        d = decompose(by, 'RLF')
        if d is None:
            incomplete.append('%s (only %d repeat(s))' % (arm, min(len(v) for v in by.values())))
            continue
        lo, hi = bounds(d)
        res[arm] = dict(rlf=d, lo=lo, hi=hi,
                        all={m: decompose(by, m) for m in METRICS})

    print('\n  SUSTAINED RLF - the primary outcome')
    print('  %-5s %-46s %2s %2s %9s %11s %11s %11s' %
          ('arm', 'assignment', 'k', 'R', 'mean', 'sigma_train', '97.5% low', '97.5% high'))
    print('  ' + '-' * 100)
    for arm in ARM_ORDER:
        if arm not in res:
            continue
        d, lo, hi = res[arm]['rlf'], res[arm]['lo'], res[arm]['hi']
        print('  %-5s %-46s %2d %2d %9.3f %11.3f %11s %11s'
              % (arm, LABEL[arm], d['k'], d['R'], d['mean'], d['sigma'],
                 '%.3f' % lo if lo is not None else 'n/a',
                 '%.3f' % hi if hi is not None else 'n/a'))
        print('        per-training: ' + '  '.join('%6.2f' % v for v in d['per_seed']))

    for m in METRICS:
        if m == 'RLF':
            continue
        print('\n  %s' % m)
        for arm in ARM_ORDER:
            if arm not in res:
                continue
            d = res[arm]['all'][m]
            print('    %-5s mean %9.3f   sigma_train %9.3f   per-training %s'
                  % (arm, d['mean'], d['sigma'], ' '.join('%7.2f' % v for v in d['per_seed'])))

    if incomplete:
        print('\n  INCOMPLETE: ' + '; '.join(incomplete))
        print('  No hypothesis is decided until all four cells have 5 seeds x R repeats.')
        return 1

    print('\n' + '=' * 104)
    print('  VERDICT')
    print('=' * 104)
    s = {a: res[a]['rlf']['sigma'] for a in ARM_ORDER}
    order = sorted(ARM_ORDER, key=lambda a: s[a])
    print('  sigma_train(RLF) ranking, lowest first: ' +
          '  '.join('%s %.3f' % (a, s[a]) for a in order))

    h1 = order[0] == 'g_bi'
    print('\n  H1  g_bi has the lowest sigma_train         -> %s' % ('HOLDS' if h1 else 'FAILS'))

    lo_ib, hi_bi = res['g_ib']['lo'], res['g_bi']['hi']
    if lo_ib is None or hi_bi is None:
        h2 = None
        print('  H2  bounds unavailable                     -> CANNOT DECIDE')
    elif lo_ib > hi_bi:
        h2 = True
        print('  H2  g_ib lower bound %.3f > g_bi upper bound %.3f  -> HOLDS (bounds separated)'
              % (lo_ib, hi_bi))
    elif s['g_ib'] > s['g_bi']:
        h2 = None
        print('  H2  g_ib sigma %.3f > g_bi %.3f but bounds overlap (%.3f vs %.3f) -> INCONCLUSIVE'
              % (s['g_ib'], s['g_bi'], lo_ib, hi_bi))
    else:
        h2 = False
        print('  H2  g_ib sigma %.3f is NOT greater than g_bi %.3f  -> FAILS'
              % (s['g_ib'], s['g_bi']))

    h3 = s['g_bb'] >= s['g_bi']
    print('  H3  g_bb does not beat g_bi                -> %s' % ('HOLDS' if h3 else 'FAILS'))

    print('\n  CONSEQUENCE, from section 7 of the pre-registration:')
    if h2 is True and h3:
        print('    The measurement-derived-assignment claim is SUPPORTED.')
        print('    It may be stated only in the narrowed form of section 8: domain-specific,')
        print('    ablation-bound, and positioned against Seyde 2021/2024, Kanervisto 2020,')
        print('    Xu 2020 and Cheng 2019.')
    elif h2 is False:
        print('    The claim is FALSE. Report as a pre-registered null, as the measurement-shielded')
        print('    Bellman backup was. The manuscript reverts to its current position: no new')
        print('    learning algorithm is offered.')
    elif h2 is None:
        print('    The claim is NOT made. Report the bound and the minimum detectable effect so a')
        print('    reader can see what this design could not resolve.')
    if not h3:
        print('    H3 failed: the mechanism is binarisation, not assignment. The assignment framing')
        print('    is withdrawn and the result is reframed accordingly.')

    out = 'results/q3_assignment_eval/q3_verdict.json'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({'sigma_rlf': s, 'H1': h1, 'H2': h2, 'H3': h3,
               'bounds': {a: {'lo': res[a]['lo'], 'hi': res[a]['hi']} for a in ARM_ORDER},
               'metrics': {a: {m: {'mean': res[a]['all'][m]['mean'],
                                   'sigma': res[a]['all'][m]['sigma'],
                                   'per_seed': res[a]['all'][m]['per_seed']}
                               for m in METRICS} for a in ARM_ORDER}},
              open(out, 'w'), indent=2)
    print('\n  wrote %s' % out)
    print('=' * 104)
    return 0


if __name__ == '__main__':
    sys.exit(main())
