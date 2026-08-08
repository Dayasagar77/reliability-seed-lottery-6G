#!/usr/bin/env python3
"""verify_tablexvii.py — independent re-derivation of the rigor manuscript's Table XVII
from results/srdndqn_*.csv, with the explicit question: WHICH dispersion denominator
was used, and is it the one the LOCKED reporting basis requires (unbiased, k-1)?

Zero cells copied from the manuscript except the PUBLISHED[] literals below, which are
transcribed solely as comparison targets.
"""
import csv, math, itertools
from statistics import mean

R = 'results/'
ARMS = {
    'postguard':    'Measured state guard (proposed, VI-B)',
    'shielded':     'Shielded backup only (guard removed)',
    'shield_guard': 'Shielded backup + guard',
    'lagrangian':   'Lagrangian constraint + guard',
}
METRICS = ['RLF', 'Override', 'Power', 'Spec', 'HO']

# Transcribed from DnDQN_manuscript_rigor.docx Table XVII, for comparison only.
PUBLISHED = {
    'postguard':    {'RLF': (4.08, 0.42), 'Override': (5.12, 0.50), 'Power': (60.84, 10.33),
                     'Spec': (58.39, 18.42), 'HO': (94.55, 0.67)},
    'shielded':     {'RLF': (21.87, 1.70), 'Override': None,        'Power': (83.26, 6.21),
                     'Spec': (56.63, 31.23), 'HO': (87.42, 1.74)},
    'shield_guard': {'RLF': (4.99, 0.93), 'Override': (5.06, 0.90), 'Power': (58.69, 11.82),
                     'Spec': (80.19, 7.82), 'HO': (94.09, 0.65)},
    'lagrangian':   {'RLF': (5.20, 0.81), 'Override': (5.43, 0.26), 'Power': (65.74, 1.86),
                     'Spec': (54.37, 25.37), 'HO': (94.29, 0.65)},
}


def load(arm):
    rows = {}
    with open(R + 'srdndqn_%s.csv' % arm) as f:
        for r in csv.DictReader(f):
            rows[int(r['seed'])] = {k: (float(v) if v not in ('', None) else None)
                                    for k, v in r.items() if k != 'seed'}
    return rows


def sd(xs, ddof):
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - ddof))


data = {a: load(a) for a in ARMS}

print('=' * 92)
print('1. WHICH DENOMINATOR?  recomputing every published cell both ways')
print('=' * 92)
print('%-14s %-9s %8s | %8s %8s | %8s %8s | verdict' %
      ('arm', 'metric', 'pub sd', 'sd(n)', 'd(n)', 'sd(n-1)', 'd(n-1)'))
tally = {'n': 0, 'n-1': 0, 'neither': 0}
for arm in ARMS:
    seeds = sorted(data[arm])
    for met in METRICS:
        pub = PUBLISHED[arm][met]
        if pub is None:
            continue
        xs = [data[arm][s][met] for s in seeds]
        if any(x is None for x in xs):
            print('%-14s %-9s   (blank in CSV — arm has no guard, matches published em-dash)'
                  % (arm, met))
            continue
        m = mean(xs)
        s_n, s_n1 = sd(xs, 0), sd(xs, 1)
        pm, ps = pub
        dn, dn1 = abs(s_n - ps), abs(s_n1 - ps)
        v = 'n' if dn < 0.005 else ('n-1' if dn1 < 0.005 else 'NEITHER')
        tally['n' if v == 'n' else ('n-1' if v == 'n-1' else 'neither')] += 1
        mflag = '' if abs(m - pm) < 0.006 else '  <-- MEAN MISMATCH pub=%.2f got=%.4f' % (pm, m)
        print('%-14s %-9s %8.2f | %8.4f %8.4f | %8.4f %8.4f | %s%s'
              % (arm, met, ps, s_n, dn, s_n1, dn1, v, mflag))

print()
print('TALLY  matches n-denominator: %d   matches (n-1)-denominator: %d   neither: %d'
      % (tally['n'], tally['n-1'], tally['neither']))

print()
print('=' * 92)
print('2. WHAT THE PUBLISHED sd WOULD BECOME UNDER THE LOCKED (k-1) RULE')
print('=' * 92)
print('%-14s %-9s %10s %10s %10s' % ('arm', 'metric', 'mean', 'sd published', 'sd (k-1)'))
for arm in ARMS:
    seeds = sorted(data[arm])
    for met in METRICS:
        pub = PUBLISHED[arm][met]
        if pub is None:
            continue
        xs = [data[arm][s][met] for s in seeds]
        if any(x is None for x in xs):
            continue
        print('%-14s %-9s %10.2f %10.2f %10.2f'
              % (arm, met, mean(xs), pub[1], sd(xs, 1)))

print()
print('=' * 92)
print('3. PAIRED TESTS RE-DERIVED (seed-by-seed, ddof=1 as a paired t requires)')
print('=' * 92)


def paired(a, b, met):
    seeds = sorted(set(data[a]) & set(data[b]))
    d = [data[a][s][met] - data[b][s][met] for s in seeds]
    n = len(d)
    md = mean(d)
    sdd = sd(d, 1)
    t = md / (sdd / math.sqrt(n))
    dz = md / sdd
    return n, md, sdd, t, dz


def perm_p(a, b, met):
    """exact two-sided sign-flip permutation p — no normality assumption."""
    seeds = sorted(set(data[a]) & set(data[b]))
    d = [data[a][s][met] - data[b][s][met] for s in seeds]
    obs = abs(mean(d))
    hits = 0
    tot = 0
    for signs in itertools.product([1, -1], repeat=len(d)):
        tot += 1
        if abs(mean([s * x for s, x in zip(signs, d)])) >= obs - 1e-12:
            hits += 1
    return hits / tot


CMP = [
    ('NECESSITY  guard removed vs proposed', 'shielded', 'postguard', 'RLF'),
    ('NECESSITY  guard removed vs proposed', 'shielded', 'postguard', 'HO'),
    ('SUFFICIENCY shield+guard vs proposed', 'shield_guard', 'postguard', 'RLF'),
    ('SUFFICIENCY shield+guard vs proposed', 'shield_guard', 'postguard', 'Override'),
    ('lagrangian vs proposed', 'lagrangian', 'postguard', 'RLF'),
]
print('%-38s %-9s %3s %9s %9s %9s %8s %10s'
      % ('comparison', 'metric', 'n', 'mean d', 'sd d', 't', 'd_z', 'exact p'))
for label, a, b, met in CMP:
    n, md, sdd, t, dz = paired(a, b, met)
    p = perm_p(a, b, met)
    print('%-38s %-9s %3d %9.4f %9.4f %9.4f %8.4f %10.5f'
          % (label, met, n, md, sdd, t, dz, p))

print()
print('NOTE  exact p is the sign-flip permutation p over all 2^5 = 32 sign assignments.')
print('      Its finest attainable resolution at n=5 is 2/32 = 0.0625, so NO paired test')
print('      on 5 seeds can produce a distribution-free p below 0.0625. Any p < 0.0001')
print('      quoted for these arms is a normal-theory t p-value, not an exact one.')
