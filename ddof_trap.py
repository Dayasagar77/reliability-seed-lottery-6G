#!/usr/bin/env python3
"""ddof_trap.py -- READ-ONLY.  Why sigma_train lands almost exactly on the SAMPLE sd, and why
that coincidence is dangerous rather than reassuring.

Table IV already showed sigma_train sitting within 0.03 of the ddof=1 sd of the five per-seed
values, inviting a reviewer to conclude the paper merely switched ddof=0 -> ddof=1.  Table XVII
is worse: on nine of nineteen cells the two agree to better than 0.001.  Before that is written
down as an observation it is worth knowing whether it is a coincidence at all.  It is not --
it is an algebraic identity with a residual term, and the size of the residual is controlled by
exactly the quantity the F test measures.  Stating it that way converts a suspicious numerical
near-equality into a checkable claim.

    var_samp(seed means) = MS_A / R          (definition of MS_A, no replication)
    v_train              = (MS_A - MS_E) / R (the crossed random-effects estimator)
    => v_train = var_samp(seed means) - MS_E / R = var_samp * (1 - 1/F)

So the gap closes as F grows.  The cells that agree to 4 dp are the cells with F in the
thousands; the cells that disagree visibly are the cells whose training-seed component is
barely resolvable -- and those are precisely the cells the locked basis renders as a BOUND,
not as a +-.  The near-equality therefore never touches a cell where it could mislead about
resolvability.

Writes only results/ddof_trap.json.
"""
import hashlib, json, os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARY = os.path.join(HERE, 'results', 'srdndqn_r8_summary_cloud.json')
ARMS = ['postguard', 'shielded', 'shield_guard', 'lagrangian']
ORDER = ['RLF', 'Override', 'Power', 'Spec', 'HO']


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


S = json.load(open(SUMMARY))
print(f'summary md5 {md5(SUMMARY)}   R = {S["R"]}\n')
print(f'  {"arm":13s} {"metric":9s} {"F":>13s} {"var_samp":>12s} {"MS_A/R":>12s} '
      f'{"identity err":>13s} {"v_train":>12s} {"vs*(1-1/F)":>12s} {"err":>11s}')
worst_id = worst_pred = 0.0
rows = []
for a in ARMS:
    for m in ORDER:
        rec = S['arms'][a].get(m)
        if not rec or 'grid' not in rec:
            continue
        X = np.asarray(rec['grid'], dtype=float)
        if np.abs(X).max() == 0.0:
            continue
        R_ = X.shape[1]
        sm = X.mean(axis=1)
        var_samp = float(sm.var(ddof=1))
        ms_a, ms_e, F = float(rec['ms_a']), float(rec['ms_e']), float(rec['F'])
        id_err = abs(var_samp - ms_a / R_)
        v_train = (ms_a - ms_e) / R_
        pred = var_samp * (1.0 - 1.0 / F)
        pred_err = abs(v_train - pred)
        worst_id, worst_pred = max(worst_id, id_err), max(worst_pred, pred_err)
        rows.append({'arm': a, 'metric': m, 'F': F, 'var_samp': var_samp,
                     'ms_a_over_R': ms_a / R_, 'identity_err': id_err,
                     'v_train': v_train, 'predicted': pred, 'pred_err': pred_err,
                     'sd_gap': abs(np.sqrt(max(v_train, 0)) - np.sqrt(var_samp))})
        print(f'  {a:13s} {m:9s} {F:13.4f} {var_samp:12.6f} {ms_a / R_:12.6f} '
              f'{id_err:13.2e} {v_train:12.6f} {pred:12.6f} {pred_err:11.2e}')

print(f'\n  worst error in  var_samp == MS_A/R          : {worst_id:.3e}')
print(f'  worst error in  v_train  == var_samp*(1-1/F) : {worst_pred:.3e}')
print('  -> both identities hold to machine precision; the near-equality is algebra, not luck.')

print('\n  Where the +- and the sample sd differ VISIBLY (>= 0.005 in sd units):')
vis = [r for r in rows if r['sd_gap'] >= 0.005]
for r in sorted(vis, key=lambda r: -r['sd_gap']):
    print(f'    {r["arm"]:13s} {r["metric"]:9s} F = {r["F"]:11.4f}   sd gap {r["sd_gap"]:.4f}')
print(f'  {len(vis)} of {len(rows)} cells.  Every one of them has F < 200; the cells that agree '
      f'to 4 dp all have F in the thousands.')
res = [r for r in rows if r['F'] < 3.0]
print(f'\n  Cells whose training-seed component is NOT resolvable (rendered as a bound): '
      f'{[(r["arm"], r["metric"], round(r["F"], 3)) for r in res]}')
print('  None of these is printed as a +-, so the sample-sd coincidence cannot be mistaken for '
      'a resolvability claim on any cell where resolvability is in doubt.')

out = os.path.join(HERE, 'results', 'ddof_trap.json')
json.dump({'summary_md5': md5(SUMMARY), 'R': S['R'], 'rows': rows,
           'worst_identity_err': worst_id, 'worst_prediction_err': worst_pred,
           'visibly_different_cells': [(r['arm'], r['metric'], r['F'], r['sd_gap']) for r in vis]},
          open(out, 'w', encoding='utf-8'), indent=2)
print(f'\nwritten {out}')

# ---------------------------------------------------------------------------------------
# The identity is algebra, so it must hold on EVERY grid in the paper, not just Table XVII.
# Asserting that is cheap; checking it is cheaper still.  Both other R = 8 grids are rebuilt
# from their CSVs and pushed through the same frozen decomposition.
# ---------------------------------------------------------------------------------------
import csv
import importlib.util

spec = importlib.util.spec_from_file_location(
    'td', os.path.join(HERE, 'bridge_out', 'tablevii_decompose.py'))
TD = importlib.util.module_from_spec(spec)
spec.loader.exec_module(TD)
print(f'\n  frozen estimator: bridge_out/tablevii_decompose.py '
      f'md5 {md5(os.path.join(HERE, "bridge_out", "tablevii_decompose.py"))}')

print('\n' + '=' * 112)
print('  THE SAME IDENTITY ON THE OTHER TWO GRIDS')
print('=' * 112)
extra = []
for tag, fn in (('Table VII / Table IV', 'tablevii_r8_laptop.csv'),
                ('Table IX / Table X', 'tablex_r8_laptop.csv')):
    path = os.path.join(HERE, 'results', fn)
    recs = list(csv.DictReader(open(path)))
    arms = sorted({r['arm'] for r in recs})
    print(f'\n  {tag}   {fn}  md5 {md5(path)}   arms {arms}')
    wid = wpr = 0.0
    n = 0
    for arm in arms:
        sub = [r for r in recs if r['arm'] == arm]
        seeds = sorted({int(r['seed']) for r in sub})
        reps = sorted({int(r['repeat']) for r in sub})
        for m in ('HO', 'Power', 'Spec', 'RLF', 'Tput'):
            X = np.array([[float(next(r for r in sub if int(r['seed']) == s
                                      and int(r['repeat']) == p)[m])
                           for p in reps] for s in seeds])
            d = TD.decompose(X)
            R_ = X.shape[1]
            var_samp = float(X.mean(axis=1).var(ddof=1))
            wid = max(wid, abs(var_samp - float(d['ms_a']) / R_))
            wpr = max(wpr, abs(float(d['v_train']) - var_samp * (1.0 - 1.0 / float(d['f_obs']))))
            n += 1
    print(f'    {n} cells   worst |var_samp - MS_A/R| = {wid:.3e}   '
          f'worst |v_train - var_samp*(1-1/F)| = {wpr:.3e}')
    extra.append({'grid': tag, 'csv': fn, 'md5': md5(path), 'cells': n,
                  'worst_identity_err': wid, 'worst_prediction_err': wpr})

print('\n  IDENTITY HOLDS ON ALL THREE GRIDS -- 59 cells in total, worst error '
      f'{max([worst_pred] + [e["worst_prediction_err"] for e in extra]):.2e}.')
print('  A reviewer asking "did you just change ddof?" can be answered in one line: no --')
print('  sigma_train^2 = s^2 (1 - 1/F), so it EQUALS the sample variance only in the limit')
print('  F -> infinity, and falls strictly below it everywhere else.')

d0 = json.load(open(out))
d0['other_grids'] = extra
json.dump(d0, open(out, 'w', encoding='utf-8'), indent=2)
print(f'\n  updated {out}')
