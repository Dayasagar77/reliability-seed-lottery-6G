#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
q4_analyse.py - the pre-registered decision rules for the risk-asymmetry campaign.

  H1  sigma_train(RLF) for qr_asym < qr_sym                       asymmetry beats symmetry
  H2  qr_asym_u (NO GUARD) reaches mean RLF within 2 points of the published guarded arm's 4.91%
      AND sigma_train below 1.0, against the published unguarded binary arm's 18.29 +- 6.22
                                                                  THE HEADLINE
  H3  qr_sym does not attain a lower sigma_train(RLF) than the published guarded arm
  H4  latency stays inside the Near-RT budget                     (measured by q4_latency.py)

sigma_train and its one-sided 97.5% bounds use the same estimators as the published analysis:
two-way crossed random effects over trainings x CRN blocks, and the exact F pivot
F_obs/(1 + R*gamma) ~ F(dfA, dfE) - identical to ratio174_provenance.py, which produced the 0.17
in Table 1. All five metrics are printed for every arm whatever the verdict.
"""
import os, sys, csv, glob, math, json

METRICS = ['HO', 'Power', 'Spec', 'RLF', 'Tput']
EVAL_DIR = 'results/q4_riskasym_eval'
PUB = {
    'pub_g': ('Published guarded (scalar Q, expectation)', 'results/tablevii_r8', 'unit_guarded_s*_*.csv'),
    'pub_u': ('Published unguarded binary (scalar Q)',      'results/tablevii_r8', 'unit_binary_s*_*.csv'),
}
NEW = {
    'qr_sym':    'Quantile, symmetric mean, guarded',
    'qr_asym':   'Quantile, reliability CVaR + spectrum mean, guarded',
    'qr_asym_u': 'Quantile, reliability CVaR, NO GUARD',
    'qr_inv':    'Quantile, inverted risk assignment, guarded',
}
ORDER = ['pub_u', 'pub_g', 'qr_sym', 'qr_asym', 'qr_inv', 'qr_asym_u']

# The hypotheses of section 4 of the pre-registration reference ONLY these arms. qr_inv is the
# optional falsifier of section 3 - "Optional 4th arm, 5 more trainings, run only if the schedule
# allows" - and appears in the verdict as an informational line when it exists. Requiring it to
# decide H1/H2/H3 was a defect in this script: it blocked a verdict the design does not need.
REQUIRED = ['pub_u', 'pub_g', 'qr_sym', 'qr_asym', 'qr_asym_u']

# pre-registered thresholds
H2_MEAN_TOL = 2.0
H2_SIGMA_TOL = 1.0
PUB_G_MEAN = 4.91

try:
    from scipy import stats
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


def load(d, pat):
    by = {}
    for p in sorted(glob.glob(os.path.join(d, pat))):
        for row in csv.DictReader(open(p, newline='', encoding='utf-8-sig')):
            try:
                s = int(row['seed'])
            except (KeyError, ValueError):
                continue
            by.setdefault(s, []).append({m: float(row[m]) for m in METRICS})
    return by


def files_for(arm):
    if arm in PUB:
        return PUB[arm][1], PUB[arm][2]
    return EVAL_DIR, 'unit_%s_s*.csv' % arm


def label_for(arm):
    return PUB[arm][0] if arm in PUB else NEW.get(arm, arm)


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
                sigma=math.sqrt(var) if var > 0 else 0.0)


def bounds(d, alpha=0.025):
    if not HAVE_SCIPY or d is None or not math.isfinite(d['F']):
        return None, None
    q_lo = stats.f.ppf(alpha, d['dfA'], d['dfE'])
    q_hi = stats.f.ppf(1 - alpha, d['dfA'], d['dfE'])
    g_u = (d['F'] / q_lo - 1) / d['R']
    g_l = (d['F'] / q_hi - 1) / d['R']
    return math.sqrt(max(g_l, 0) * d['ms_e']), math.sqrt(max(g_u, 0) * d['ms_e'])


def main():
    print('=' * 110)
    print('  RISK-ASYMMETRY CAMPAIGN - pre-registered decision rules (10 Aug 2026)')
    print('  does a change to the LEARNING RULE do what the non-learned guard does?')
    print('=' * 110)
    if not HAVE_SCIPY:
        print('  scipy missing - point estimates only, no bounds')

    res, missing = {}, []
    for arm in ORDER:
        d, pat = files_for(arm)
        by = load(d, pat)
        if len(by) < 5:
            missing.append('%s (seeds %s)' % (arm, sorted(by)))
            continue
        r = decompose(by, 'RLF')
        if r is None:
            missing.append('%s (repeats < 2)' % arm)
            continue
        lo, hi = bounds(r)
        res[arm] = dict(rlf=r, lo=lo, hi=hi, all={m: decompose(by, m) for m in METRICS})

    print('\n  SUSTAINED RLF - the primary outcome')
    print('  %-10s %-52s %9s %11s %11s %11s' %
          ('arm', 'what it is', 'mean', 'sigma_train', '97.5% low', '97.5% high'))
    print('  ' + '-' * 106)
    for arm in ORDER:
        if arm not in res:
            continue
        r, lo, hi = res[arm]['rlf'], res[arm]['lo'], res[arm]['hi']
        print('  %-10s %-52s %9.3f %11.3f %11s %11s'
              % (arm, label_for(arm)[:52], r['mean'], r['sigma'],
                 '%.3f' % lo if lo is not None else 'n/a',
                 '%.3f' % hi if hi is not None else 'n/a'))
        print('             per-training: ' + '  '.join('%6.2f' % v for v in r['per_seed']))

    for m in METRICS:
        if m == 'RLF':
            continue
        print('\n  %s' % m)
        for arm in ORDER:
            if arm not in res:
                continue
            d = res[arm]['all'][m]
            print('    %-10s mean %9.3f   sigma_train %9.3f   per-training %s'
                  % (arm, d['mean'], d['sigma'], ' '.join('%7.2f' % v for v in d['per_seed'])))

    blocking = [m for m in missing if m.split(' ')[0] in REQUIRED]
    optional = [m for m in missing if m.split(' ')[0] not in REQUIRED]
    if optional:
        print('\n  OPTIONAL ARM NOT RUN: ' + '; '.join(optional))
        print('  qr_inv is the optional falsifier of section 3 ("run only if the schedule allows").')
        print('  No hypothesis in section 4 references it, so it does not block the verdict.')
        print('  Section 5 forbids adding it now: no arm may be run after results are seen.')
    if blocking:
        print('\n  INCOMPLETE: ' + '; '.join(blocking))
        print('  No hypothesis is decided until every arm the hypotheses reference has 5 seeds x R.')
        return 1

    print('\n' + '=' * 110)
    print('  VERDICT')
    print('=' * 110)
    S = {a: res[a]['rlf']['sigma'] for a in res}
    M = {a: res[a]['rlf']['mean'] for a in res}

    h1 = S['qr_asym'] < S['qr_sym']
    print('  H1  asymmetry beats symmetry: sigma qr_asym %.3f vs qr_sym %.3f  -> %s'
          % (S['qr_asym'], S['qr_sym'], 'HOLDS' if h1 else 'FAILS'))

    dm = abs(M['qr_asym_u'] - PUB_G_MEAN)
    h2 = dm <= H2_MEAN_TOL and S['qr_asym_u'] <= H2_SIGMA_TOL
    print('  H2  THE HEADLINE - the algorithm without the guard:')
    print('        qr_asym_u  mean %.3f (guarded published %.2f, gap %.3f, tol %.1f)'
          % (M['qr_asym_u'], PUB_G_MEAN, dm, H2_MEAN_TOL))
    print('        qr_asym_u  sigma_train %.3f (tol %.1f)   against published unguarded %.3f / %.3f'
          % (S['qr_asym_u'], H2_SIGMA_TOL, M.get('pub_u', float('nan')), S.get('pub_u', float('nan'))))
    print('        -> %s' % ('HOLDS' if h2 else 'FAILS'))

    h3 = S['qr_sym'] >= S.get('pub_g', 0.0)
    print('  H3  distributional alone is not the mechanism: sigma qr_sym %.3f vs published guarded '
          '%.3f  -> %s' % (S['qr_sym'], S.get('pub_g', float('nan')), 'HOLDS' if h3 else 'FAILS'))

    if 'qr_inv' in S:
        inv_lo = res['qr_inv']['lo']
        asym_hi = res['qr_asym']['hi']
        sep = inv_lo is not None and asym_hi is not None and inv_lo > asym_hi
        print('  --  inverted falsifier: sigma qr_inv %.3f vs qr_asym %.3f, bounds %s'
              % (S['qr_inv'], S['qr_asym'], 'separated' if sep else 'overlapping'))

    print('\n  CONSEQUENCE, from section 5 of the pre-registration:')
    if h2 and h1:
        print('    The paper gains an ALGORITHMIC CONTRIBUTION: measurement-keyed risk asymmetry')
        print('    removes the between-training reliability lottery WITHOUT the non-learned guard.')
        print('    Stated only in the narrowed form of section 7, after the prior-art gate.')
    elif h1 and not h2:
        print('    The asymmetry helps a guarded controller but does NOT replace the guard.')
        print('    Secondary result; the guard remains the reliability mechanism; the paper keeps')
        print('    its current claim.')
    else:
        print('    H1 failed: the contribution is not ours. Report as a pre-registered null and')
        print('    drop it, as the measurement-shielded Bellman backup was dropped.')
    if not h3:
        print('    H3 failed: plain distributional learning is the mechanism. The asymmetry framing')
        print('    is withdrawn.')

    out = os.path.join(EVAL_DIR, 'q4_verdict.json')
    os.makedirs(EVAL_DIR, exist_ok=True)
    json.dump({'sigma_rlf': S, 'mean_rlf': M, 'H1': h1, 'H2': h2, 'H3': h3,
               'bounds': {a: {'lo': res[a]['lo'], 'hi': res[a]['hi']} for a in res},
               'metrics': {a: {m: {'mean': res[a]['all'][m]['mean'],
                                   'sigma': res[a]['all'][m]['sigma'],
                                   'per_seed': res[a]['all'][m]['per_seed']}
                               for m in METRICS} for a in res}},
              open(out, 'w'), indent=2)
    print('\n  wrote %s' % out)
    print('=' * 110)
    return 0


if __name__ == '__main__':
    sys.exit(main())
