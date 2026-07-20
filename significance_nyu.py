#!/usr/bin/env python3
"""significance_nyu.py — statistical significance on the NYU channel (channel-consistency fix;
replaces the v6 bootstrap_ci for the paper). Uses the matched-objective per-seed results
(results/matched_rel_nyu.csv: 3 training seeds x {Proposed, DDQN+PER, Standard}) and tests
Proposed vs each baseline on RLF / Spec / Power with Welch's t-test + Cohen's d effect size.
-> results/significance_nyu.csv
Honesty note: n=3 training seeds -> effect sizes (d) are the reliable signal; t-test p-values are
low-power at n=3. A per-episode bootstrap (like the old v6 analysis) can tighten p if needed."""
import csv, numpy as np
try:
    from scipy import stats; HAVE = True
except Exception:
    HAVE = False

rows = list(csv.DictReader(open('results/matched_rel_nyu.csv')))
M = {}
for r in rows:
    d = M.setdefault(r['method'], {'RLF': [], 'Spec': [], 'Power': [], 'HO': []})
    for k in ['RLF', 'Spec', 'Power', 'HO']: d[k].append(float(r[k]))

def cohend(a, b):
    a, b = np.array(a), np.array(b); na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float('inf')

def welch(a, b):
    a, b = np.array(a), np.array(b)
    if HAVE:
        t, p = stats.ttest_ind(a, b, equal_var=False); return float(t), float(p)
    t = (a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float(t), float('nan')

prop = 'Proposed Dn-DQN'; out = []
print(f"scipy={'yes' if HAVE else 'no'}\n{'comparison':26s} {'metric':6s} {'Proposed':>14s} {'baseline':>14s} {'d':>7s} {'t':>7s} {'p':>8s}")
for base in [m for m in M if m != prop]:
    for metric in ['RLF', 'Spec', 'Power']:
        pa, ba = M[prop][metric], M[base][metric]
        d = cohend(pa, ba); t, p = welch(pa, ba)
        pm = f"{np.mean(pa):.2f}±{np.std(pa, ddof=1):.2f}"; bm = f"{np.mean(ba):.2f}±{np.std(ba, ddof=1):.2f}"
        print(f"{'Proposed vs ' + base:26s} {metric:6s} {pm:>14s} {bm:>14s} {d:7.2f} {t:7.2f} {p if p==p else float('nan'):8.4f}")
        out.append({'comparison': f'Proposed vs {base}', 'metric': metric, 'proposed': pm, 'baseline': bm,
                    'cohens_d': f"{d:.2f}", 'welch_t': f"{t:.2f}", 'p_value': (f"{p:.4f}" if p == p else 'n/a'), 'n_seeds': len(pa)})
with open('results/significance_nyu.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['comparison', 'metric', 'proposed', 'baseline', 'cohens_d', 'welch_t', 'p_value', 'n_seeds'])
    w.writeheader(); w.writerows(out)
print("  -> results/significance_nyu.csv  (n=3 training seeds; Cohen's d robust, p low-power at n=3)")
