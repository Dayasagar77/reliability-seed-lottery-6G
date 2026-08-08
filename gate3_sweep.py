#!/usr/bin/env python3
"""gate3_sweep.py -- READ-ONLY. Run the quantisation gate over EVERY R = 8 grid and every
printed dispersion that can be sourced from a machine-readable artifact on disk.

Why this exists.  Task #208 found that Sustained RLF is a count out of 750 and therefore
moves in steps of exactly 2/15 = 0.1333 pp, and that one Table XVII bound (0.10) was printed
finer than that step.  The obvious next question is whether the SAME defect is sitting in the
tables that were locked earlier -- Table IV, Table VII, Table X.  "OD-1's 0.17 and OD-2's 3.89
are both above 0.1333, so they're fine" is an answer arrived at by eye, and a site list
assembled by eye is not a blast radius.  So every grid is re-scanned for its own quanta and
every printed dispersion available on disk is checked against them mechanically.

The gate can only WIDEN a bound, never narrow one, so firing is always the conservative
direction.  Cells that cannot be sourced from disk are listed explicitly as UNCHECKED rather
than assumed clean.

Writes only results/gate3_sweep.json.
"""
import csv, hashlib, json, math, os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results')

GRIDS = {
    'tableVII_tableIV': 'tablevii_r8_laptop.csv',    # arms flat / incremental / binary / guarded
    'tableIX_tableX':   'tablex_r8_laptop.csv',      # arms m5_dndqn / m5_ddqn / m5_std / sd5_safe
    'tableXVII':        'srdndqn_r8_cloud.csv',      # arms postguard / shielded / shield_guard / lagrangian
}
METRICS = ['HO', 'Power', 'Spec', 'RLF', 'Tput', 'Override']


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def quantum(vals, tol=1e-7):
    """Least-squares quantum detector.  Identical logic to tablexvii_render.quantum().

    The naive version (q = smallest observed gap, then demand every value be an integer
    multiple to 1e-4) CANNOT pass on 4-dp-stored data: q is itself a difference of two
    rounded values, so the error in n*q grows with n.  It rejected RLF's true 2/15 step and
    declared the metric continuous -- a gate passing because its detector never fired.
    Here q is refined by least squares against integer indices and the fit is judged by
    residual RELATIVE to q.  This makes the gate fire MORE often, which is the only
    permitted direction to move a gate.
    """
    v = np.array(sorted({round(float(x), 9) for x in vals}))
    if len(v) < 3:
        return None
    gaps = np.diff(v)
    gaps = gaps[gaps > tol]
    if not len(gaps):
        return None
    q0 = float(gaps.min())
    n = np.round(v / q0)
    if n.min() < 1 or (n[1:] - n[:-1]).min() < 1:
        return None
    q = float((n * v).sum() / (n * n).sum())
    resid = float(np.abs(v - n * q).max())
    return {'q': q, 'resid': resid, 'resid_over_q': resid / q, 'n_distinct': int(len(v))} \
        if resid < 0.01 * q else None


print('=' * 118)
print('  PART 1 -- quantum of every metric on every R = 8 grid (pooled over all 160 evaluations)')
print('=' * 118)
QUANT = {}
for name, fn in GRIDS.items():
    path = os.path.join(R, fn)
    rows = list(csv.DictReader(open(path)))
    print(f'\n  {name}   {fn}   md5 {md5(path)}   {len(rows)} rows')
    QUANT[name] = {}
    for m in METRICS:
        vals = [float(r[m]) for r in rows if r.get(m) not in (None, '')]
        if not vals:
            continue
        q = quantum(vals)
        QUANT[name][m] = q
        if q:
            frac, qv = '', q['q']
            for den in range(1, 4001):
                num = round(qv * den)
                if num >= 1 and abs(qv * den - num) < 1e-4 * max(1, num):
                    frac = '  = %d/%d' % (num, den)
                    break
            print(f'    {m:9s} {len(set(vals)):4d} distinct  quantum {q["q"]:.7f}{frac}   '
                  f'resid {q["resid"]:.2e} = {q["resid_over_q"]:.5f} of q')
        else:
            print(f'    {m:9s} {len(set(vals)):4d} distinct  no common quantum (continuous)')

print('\n' + '=' * 118)
print('  PART 2 -- every printed dispersion sourced from disk, checked against its own quantum')
print('=' * 118)

CHECKS = []

# --- Table IV summary row + Table VII row 1 (OD-2), from results/od2_tableiv.json -----------
p = os.path.join(R, 'od2_tableiv.json')
od2 = json.load(open(p))
print(f'\n  source: od2_tableiv.json  md5 {md5(p)}   (arm "{od2["arm"]}", grid {os.path.basename(od2["csv"])})')
for m in ('HO', 'Power', 'Spec', 'RLF', 'Tput'):
    g = od2['gate3'][m]
    printed = g.get('locked_sd_train')
    if not g.get('resolvable'):
        printed = None
    CHECKS.append({'table': 'Table IV r6 / Table VII r1', 'arm': od2['arm'], 'metric': m,
                   'kind': 'sd_train (+-)', 'printed': printed,
                   'grid': 'tableVII_tableIV', 'resolvable': bool(g.get('resolvable'))})

# --- Table IV per-seed rows (point values, NOT dispersion) -- recorded, not gated ----------
# A mean of 8 CRN repeats lives on a grid of q/8, so a per-seed MEAN printed at 2 dp is not a
# resolution claim.  The gate applies to dispersion and to bounds, which are stated in the
# metric's own units and read as "the instrument can tell these apart".
print('  (Table IV per-seed cells are means of 8 repeats, not dispersions -- not gated; see note)')

# --- OD-1 LOCKED sec.6 line 88, guarded arm.  Transcribed values, gated here. --------------
LOCKED_GUARD = [('HO', 0.29, True), ('Power', 5.02, True), ('Spec', 19.17, True),
                ('RLF', 0.17, False), ('Tput', 250.3, True)]
print('\n  source: LOCKED sec.6 line 88 (guarded arm), transcribed -- '
      '94.71 +- 0.29 | 63.48 +- 5.02 | 80.72 +- 19.17 | 4.91 (sigma_train < 0.17) | 504.0 +- 250.3')
for m, val, resolvable in LOCKED_GUARD:
    CHECKS.append({'table': 'Table VII r4 (guarded)', 'arm': 'guarded', 'metric': m,
                   'kind': 'sd_train (+-)' if resolvable else 'bound (<)',
                   'printed': val, 'grid': 'tableVII_tableIV', 'resolvable': resolvable})

# --- Table XVII, regenerated this task ----------------------------------------------------
p = os.path.join(R, 'tablexvii_r8.json')
x17 = json.load(open(p))
print(f'\n  source: tablexvii_r8.json  md5 {md5(p)}')
for arm, a in x17['arms'].items():
    for m, c in a['cells'].items():
        if c.get('render') in (None, '---') or c.get('structural_zero'):
            continue
        if c.get('sd_train') is not None:
            CHECKS.append({'table': 'Table XVII', 'arm': arm, 'metric': m, 'kind': 'sd_train (+-)',
                           'printed': round(c['sd_train'], 2), 'grid': 'tableXVII', 'resolvable': True})
        elif c.get('sd_train_hi_975') is not None:
            CHECKS.append({'table': 'Table XVII', 'arm': arm, 'metric': m, 'kind': 'bound (<)',
                           'printed': math.ceil(c['sd_train_hi_975'] * 100 - 1e-12) / 100,
                           'grid': 'tableXVII', 'resolvable': False})

print('\n  ' + '-' * 114)
print(f'  {"table":28s} {"arm":13s} {"metric":8s} {"kind":15s} {"printed":>9s} {"quantum":>9s}  verdict')
print('  ' + '-' * 114)
fired = []
for c in CHECKS:
    q = QUANT[c['grid']].get(c['metric'])
    qv = q['q'] if q else None
    if c['printed'] is None:
        v = 'n/a (not printed as a dispersion)'
    elif qv is None:
        v = 'continuous metric -- gate not applicable'
    elif c['printed'] < qv:
        v = f'FIRES -- widen to {math.ceil(qv * 100 - 1e-12) / 100:.2f}'
        fired.append(c)
    else:
        v = 'clear (%.2f quanta)' % (c['printed'] / qv)
    c['quantum'] = qv
    c['verdict'] = v
    ps = '--' if c['printed'] is None else ('%.2f' % c['printed'])
    qs = '--' if qv is None else ('%.4f' % qv)
    print('  %-28s %-13s %-8s %-15s %9s %9s  %s'
          % (c['table'], c['arm'], c['metric'], c['kind'], ps, qs, v))

print('\n  ' + '=' * 114)
print(f'  CELLS GATED: {len([c for c in CHECKS if c["printed"] is not None])}   '
      f'GATE FIRED ON: {len(fired)}  -> {[(c["table"], c["arm"], c["metric"], c["printed"]) for c in fired]}')
print('  UNCHECKED (no machine-readable source on disk): Table X printed cells, Table IX t/p/d '
      '(not dispersions in metric units), Table XI (k = 3, separate spec).')
print('  ' + '=' * 114)

out = os.path.join(R, 'gate3_sweep.json')
json.dump({'grids': {k: md5(os.path.join(R, v)) for k, v in GRIDS.items()},
           'quantum': {k: {m: (q and q['q']) for m, q in v.items()} for k, v in QUANT.items()},
           'quantum_detail': QUANT, 'checks': CHECKS,
           'fired': fired,
           'unchecked': ['Table X printed cells (no machine-readable locked spec on disk)',
                         'Table IX t/p/d (not dispersions in metric units)',
                         'Table XI (k = 3, separate spec)']},
          open(out, 'w', encoding='utf-8'), indent=2, default=float)
print(f'\nwritten {out}')
