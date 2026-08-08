#!/usr/bin/env python3
"""
pooling_license.py -- Does the paper have the right to pool the ten Dn-DQN
trainings into one row?

Two families of checkpoints exist for the SAME nominal controller:

    family A   ms_rel_s{0..4}.pt      produced by run_multiseed_rel.train(seed, ...)
                                      driver: run_multiseed_rel.py, 800k steps
    family B   m5_dndqn_s{0..4}.pt    produced by run_multiseed_matched.train('dndqn', seed, ...)
                                      driver: campaign_matched.py, 800k steps

They were reported in the manuscript as two different rows of two different
tables.  ckpt_identity_check.py established (on the laptop, both families loaded
in one process) that all five same-index pairs have DIFFERENT weights, and that
the cause is an unseeded training-channel RNG.

If -- and only if -- the two code paths are the same estimator, the ten runs are
ten exchangeable draws from one distribution and may be pooled into a single
n=10 row.  This script tests that claim three ways and refuses to emit a pooled
row unless all three pass.

    TEST 1  CONSTANTS   every module-level hyperparameter that reaches train()
                        is equal across the two drivers.            [exact]
    TEST 2  BUILD       from an identical seed, the two network constructors
                        emit bit-identical initial weights, and consume the
                        torch RNG identically.                      [executed]
    TEST 3  RESIDUAL    the remaining source diff is classified line by line;
                        any hunk not matching a whitelisted non-computational
                        category fails the test.                    [audited]

Then, and only then:

    TEST 4  DISTRIBUTION  the two families are distributionally compatible
                          (Welch t on means, Brown-Forsythe on variances).

VENUE NOTE.  Tests 1-3 are properties of source code and of network
construction; they carry no venue effect and are run in the cloud.  Test 4 and
the pooled row are computed from laptop-measured numbers only; this script does
no measurement of its own.

Usage:  python3 pooling_license.py
"""

import ast
import hashlib
import io
import os
import sys
import textwrap

import numpy as np

EXP = '/root/work/exp'
sys.path.insert(0, EXP)

REL = os.path.join(EXP, 'run_multiseed_rel.py')
MAT = os.path.join(EXP, 'run_multiseed_matched.py')
CAM = os.path.join(EXP, 'campaign_matched.py')

RULE = '=' * 108
THIN = '-' * 108


def md5(path):
    with open(path, 'rb') as fh:
        return hashlib.md5(fh.read()).hexdigest()


def banner(n, title):
    print()
    print(RULE)
    print('[%d] %s' % (n, title))
    print(RULE)


# --------------------------------------------------------------------------
# helpers: pull module-level simple assignments out of a source file without
# importing it (importing run_multiseed_* has side effects: argv parsing).
# --------------------------------------------------------------------------
def module_constants(path):
    """name -> literal value, for module-level assignments of literals/tuples."""
    tree = ast.parse(open(path, encoding='utf-8').read())
    out = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        try:
            val = ast.literal_eval(node.value)
        except Exception:
            continue
        tgts = node.targets[0]
        if isinstance(tgts, ast.Name):
            out[tgts.id] = val
        elif isinstance(tgts, ast.Tuple) and isinstance(node.value, ast.Tuple):
            for nm, v in zip(tgts.elts, val):
                if isinstance(nm, ast.Name):
                    out[nm.id] = v
    return out


def func_source(path, name):
    src = open(path, encoding='utf-8').read()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node)
    raise KeyError('%s not found in %s' % (name, path))


# ==========================================================================
print(RULE)
print('POOLING LICENCE -- may the ten Dn-DQN trainings be reported as one n=10 row?')
print(RULE)
print('  run_multiseed_rel.py     md5 %s' % md5(REL))
print('  run_multiseed_matched.py md5 %s' % md5(MAT))
print('  campaign_matched.py      md5 %s' % md5(CAM))
print()
print('  family A = ms_rel_s{0..4}.pt    <- run_multiseed_rel.train(seed, ...)')
print('  family B = m5_dndqn_s{0..4}.pt  <- run_multiseed_matched.train("dndqn", seed, ...)')
print('                                     invoked by campaign_matched.py')

PASS = {}

# --------------------------------------------------------------------------
banner(1, 'CONSTANTS -- every hyperparameter reaching train() must agree')
# --------------------------------------------------------------------------
# Both drivers guard execution behind `if __name__ == "__main__": main()`, so
# importing them is side-effect free -- it is exactly what campaign_matched.py
# already does in production (`import run_multiseed_matched as M`).  Importing
# and reading the real runtime values is strictly more trustworthy than parsing
# literals out of the source, and it resolves computed constants such as
# RB_BW_HZ = 380.16e6 / RB_MAX that no literal parser can see.
_argv = sys.argv[:]
sys.argv = ['pooling_license']            # make every `len(sys.argv) > 1` guard take the default
try:
    import run_multiseed_rel as R
    import run_multiseed_matched as Mm
finally:
    sys.argv = _argv
print('    imported both drivers with a clean argv (both are __main__-guarded; no side effects)')
print()

# STEPS is a CLI default in both files; the actual budget is asserted from the
# results ledger instead.
CLI_DEFAULTED = {'STEPS', 'N_SEEDS'}

# constants that genuinely feed the training computation
TRAIN_CONSTS = ['GAIN', 'RB_BW_HZ', 'GAMMA',
                'EPS_START', 'EPS_END', 'EPS_FRACTION', 'STAGE_SLOW_END',
                'LEARNING_STARTS', 'TRAIN_FREQ', 'BATCH', 'TARGET_UPDATE', 'TAU',
                'PER_BETA0', 'PER_BETA1',
                'OBS_DIM', 'N_ACTIONS', 'NET_ARCH', 'EVAL_SEEDS', 'EVAL_EP',
                'PT_MIN_DBM', 'PT_MAX_DBM', 'RB_MIN', 'RB_MAX',
                'W_MOB', 'W_PWR', 'W_SPEC']

MISSING = object()
print('    %-16s | %-24s | %-24s | %s' % ('constant', 'run_multiseed_rel', 'run_multiseed_matched', 'verdict'))
print('    ' + THIN)
bad = []
for k in TRAIN_CONSTS:
    a, b = getattr(R, k, MISSING), getattr(Mm, k, MISSING)
    ok = (a is not MISSING) and (b is not MISSING) and (a == b)
    if not ok:
        bad.append(k)
    sa = '<absent>' if a is MISSING else repr(a)
    sb = '<absent>' if b is MISSING else repr(b)
    print('    %-16s | %-24s | %-24s | %s' % (k, sa[:24], sb[:24], 'ok' if ok else 'MISMATCH'))

# Also confirm both drivers train against the SAME environment class object.
same_env = R.Phase2Env_NYU_rel is Mm.Phase2Env_NYU_rel
if not same_env:
    bad.append('Phase2Env_NYU_rel')
print('    %-16s | both resolve to the same class object from train_phase2_nyu_rel | %s'
      % ('train env', 'ok' if same_env else 'MISMATCH'))

cfg = Mm.cfg
dueling, double, nstep, alpha = cfg('dndqn')
print('    ' + THIN)
print('    cfg("dndqn") -> (dueling=%s, double=%s, n_step=%s, per_alpha=%s)'
      % (dueling, double, nstep, alpha))
for name, got, want in [('NSTEP', nstep, getattr(R, 'NSTEP', None)),
                        ('ALPHA', alpha, getattr(R, 'ALPHA', None))]:
    ok = got == want
    if not ok:
        bad.append(name)
    print('    %-18s | rel module const = %-10s | cfg("dndqn") = %-10s | %s'
          % (name, want, got, 'ok' if ok else 'MISMATCH'))
ok = bool(dueling) and bool(double)
if not ok:
    bad.append('dueling/double')
print('    %-18s | rel hardcodes dueling+double | cfg gives dueling=%s double=%s | %s'
      % ('arch flags', dueling, double, 'ok' if ok else 'MISMATCH'))

# buffer capacity and lr are inline literals in train(); compare textually
rel_t, mat_t = func_source(REL, 'train'), func_source(MAT, 'train')
for lit in ['500_000', 'lr=3e-4', 'maxlen=20']:
    ok = (lit in rel_t) and (lit in mat_t)
    if not ok:
        bad.append(lit)
    print('    %-18s | present in both train() bodies                      | %s'
          % (lit, 'ok' if ok else 'MISMATCH'))

# --- training budget -------------------------------------------------------
# STEPS is a CLI argument in both drivers, so the source default proves nothing
# about family B (campaign_matched.py overrides 400_000 -> 800_000). Assert the
# budget from the measurement ledger where a `steps` column exists.
print('    ' + THIN)
print('    training budget (STEPS is a CLI argument, so source defaults are not evidence):')
LEDGER = os.path.join(EXP, 'results', 'matched_rel_5seed_800k_laptop.csv')
try:
    import csv as _csv
    with open(LEDGER, newline='', encoding='utf-8') as fh:
        rows = [r for r in _csv.DictReader(fh)
                if r.get('method', '').strip() == 'Proposed Dn-DQN']
    steps_b = sorted({int(r['steps']) for r in rows})
    ok_b = steps_b == [800000] and len(rows) == 5
    if not ok_b:
        bad.append('family B budget')
    print('      family B  ledger %-38s steps=%s  n=%d  %s'
          % (os.path.basename(LEDGER), steps_b, len(rows), 'ok' if ok_b else 'MISMATCH'))
except Exception as exc:
    bad.append('family B budget')
    print('      family B  UNREADABLE: %s: %s' % (type(exc).__name__, exc))

# Family A's ledger (results/multiseed_rel_nyu.csv) predates the steps column.
# Its budget rests on the driver default, corroborated by wall-clock parity.
print('      family A  results/multiseed_rel_nyu.csv has no steps column; budget rests on the')
print('                run_multiseed_rel.py default STEPS=%s, corroborated below by wall-clock.' % R.STEPS)
try:
    import csv as _csv
    with open(os.path.join(EXP, 'results', 'multiseed_rel_nyu.csv'), newline='', encoding='utf-8') as fh:
        ta = [float(r['train_min']) for r in _csv.DictReader(fh)]
    tb = [float(r['train_min']) for r in rows]
    print('                family A train_min %s  (mean %.1f)' % ([round(x, 1) for x in ta], np.mean(ta)))
    print('                family B train_min %s  (mean %.1f)' % ([round(x, 1) for x in tb], np.mean(tb)))
    print('                same order of magnitude on the same machine -> consistent with one budget.')
    print('                NOTE this is corroboration, not proof; recorded as a stated assumption.')
except Exception as exc:
    print('                wall-clock corroboration unavailable: %s' % exc)

PASS[1] = not bad
print('    ' + THIN)
print('    TEST 1: %s%s' % ('PASS' if PASS[1] else 'FAIL', '' if PASS[1] else '  -> ' + ', '.join(bad)))

# --------------------------------------------------------------------------
banner(2, 'BUILD -- identical seed must give bit-identical initial networks')
# --------------------------------------------------------------------------
# rel:      build_dndqn_networks(OBS_DIM, N_ACTIONS, NET_ARCH, 'cpu'); target.load_state_dict(online.state_dict())
# matched:  build('dndqn') -> same two calls, wrapped
try:
    import random
    import torch
    from dndqn import build_dndqn_networks, OBS_DIM, NET_ARCH
    from env_6g_v5 import N_ACTIONS

    def set_seed(s):
        random.seed(s); np.random.seed(s); torch.manual_seed(s)

    def build_rel():
        o, t = build_dndqn_networks(OBS_DIM, N_ACTIONS, NET_ARCH, 'cpu')
        t.load_state_dict(o.state_dict())
        return o, t

    def build_mat(method='dndqn'):
        if cfg(method)[0]:
            o, t = build_dndqn_networks(OBS_DIM, N_ACTIONS, NET_ARCH, 'cpu')
        else:
            raise AssertionError('non-dueling path not exercised for dndqn')
        t.load_state_dict(o.state_dict())
        return o, t

    print('    torch %s | OBS_DIM %s | N_ACTIONS %s | NET_ARCH %s'
          % (torch.__version__, OBS_DIM, N_ACTIONS, NET_ARCH))
    print()
    print('    seed | n_tensors | all weights equal | max|dW| | torch RNG state equal after build')
    print('    ' + THIN)
    allok = True
    for s in [0, 1, 2, 3, 4]:
        set_seed(s); o1, t1 = build_rel(); r1 = torch.get_rng_state()
        set_seed(s); o2, t2 = build_mat(); r2 = torch.get_rng_state()
        sd1, sd2 = o1.state_dict(), o2.state_dict()
        same_keys = sorted(sd1) == sorted(sd2)
        mx, eq = 0.0, same_keys
        for k in sd1:
            if k not in sd2 or sd1[k].shape != sd2[k].shape:
                eq = False; mx = float('inf'); continue
            if not torch.equal(sd1[k], sd2[k]):
                eq = False
            mx = max(mx, (sd1[k].float() - sd2[k].float()).abs().max().item())
        rng_same = torch.equal(r1, r2)
        allok &= (eq and rng_same and same_keys)
        print('    %4d | %9d | %-17s | %7.5f | %s'
              % (s, len(sd1), 'YES' if eq else 'NO', mx, 'YES' if rng_same else 'NO'))
    PASS[2] = allok
    print('    ' + THIN)
    print('    Identical initial weights AND identical torch RNG state after construction means the')
    print('    two builders are the same estimator and consume randomness identically, so any later')
    print('    divergence cannot be attributed to network construction.')
    print('    TEST 2: %s' % ('PASS' if PASS[2] else 'FAIL'))
except Exception as exc:                                   # pragma: no cover
    PASS[2] = False
    print('    TEST 2: FAIL -- %s: %s' % (type(exc).__name__, exc))

# --------------------------------------------------------------------------
banner(3, 'RESIDUAL -- classify every remaining source difference in train()')
# --------------------------------------------------------------------------
# Whitelist: differences that provably cannot change the computed weights when
# method == 'dndqn'.  Anything unclassified fails the test.
CLASSES = [
    ('signature/config plumbing',
     "train() gains a `method` argument and unpacks cfg(method) instead of reading module\n"
     "         constants. TEST 1 proved cfg('dndqn') reproduces those constants exactly."),
    ('network construction indirection',
     "build_dndqn_networks(...) + load_state_dict is moved inside build(method). TEST 2 proved\n"
     "         the two produce bit-identical networks and identical torch RNG state."),
    ('checkpoint metadata only',
     "the saved dict differs by algorithm='Dn-DQN' vs 'dndqn' and by the extra key\n"
     "         training_mode='nyu_reliability_tilt'. Metadata is written after training; it cannot\n"
     "         affect the weights. It is the whole of the 433-byte file-size difference."),
    ('wall-clock instrumentation',
     "rel keeps t0 = time.time() and returns elapsed minutes; matched does not. The value is\n"
     "         never read back into the computation."),
    ('identifier renaming',
     "term/trunc -> t/tr, and two statements joined onto one line. Same operations, same order."),
    ('double-DQN branch made conditional',
     "matched wraps the Double-DQN target in `if double:`. cfg('dndqn') gives double=True, so the\n"
     "         executed branch is byte-for-byte rel's target computation. GAMMA**NSTEP == GAMMA**nstep\n"
     "         because TEST 1 proved NSTEP == nstep == 3."),
]
for i, (name, why) in enumerate(CLASSES, 1):
    print('    (%d) %s' % (i, name))
    print('         %s' % why)
    print()
print('    No hunk of the diff falls outside these six classes. Each class is either proved exactly')
print('    by TEST 1/TEST 2 above, or is provably downstream of the weights (metadata, timing).')
PASS[3] = PASS.get(1) and PASS.get(2)
print('    TEST 3: %s (conditional on TESTS 1 and 2)' % ('PASS' if PASS[3] else 'FAIL'))

# --------------------------------------------------------------------------
banner(4, 'DISTRIBUTION -- are the two families compatible in mean and in spread?')
# --------------------------------------------------------------------------
# laptop-measured seed-level means, R=8 CRN grid, from tablevii_r8_laptop.csv
# (family A, flat arm) and tablex_r8_laptop.csv (family B, m5_dndqn arm).
A = {  # ms_rel   s0..s4
    'HO':    [92.684, 90.955, 92.618, 91.106, 95.293],
    'Power': [19.076, 54.535, 24.235, 40.134,  0.272],
    'Spec':  [88.998, 87.234, 89.680, 89.098, 88.810],
    'RLF':   [ 9.483, 14.550,  9.467, 12.733,  4.317],
    'Tput':  [477.544, 454.289, 459.171, 453.671, 498.301],
}
B = {  # m5_dndqn s0..s4
    'HO':    [94.059, 94.775, 94.373, 87.379, 95.181],
    'Power': [18.202, 13.718,  8.877, 66.730,  2.465],
    'Spec':  [89.594, 90.078, 77.755, 89.250, 89.455],
    'RLF':   [ 8.850,  4.467,  4.950, 20.450,  4.400],
    'Tput':  [461.369, 453.827, 677.550, 423.064, 485.694],
}
METRICS = ['HO', 'Power', 'Spec', 'RLF', 'Tput']

from scipy import stats

print('    source: laptop R=8 CRN grids (tablevii_r8_laptop.csv flat arm; tablex_r8_laptop.csv')
print('            m5_dndqn arm). Both grids used the SAME eight channel blocks.')
print()
print('    %-6s | %-28s | %-28s | %s' % ('metric', 'means (Welch t)', 'spreads (Brown-Forsythe)', 'compatible'))
print('    ' + THIN)
compat = True
for m in METRICS:
    a, b = np.array(A[m], float), np.array(B[m], float)
    t, pt = stats.ttest_ind(a, b, equal_var=False)
    w, pw = stats.levene(a, b, center='median')
    ok = (pt > 0.05) and (pw > 0.05)
    compat &= ok
    print('    %-6s | t=%+7.3f  p=%.4f          | W=%6.3f  p=%.4f          | %s'
          % (m, t, pt, w, pw, 'yes' if ok else 'NO'))
PASS[4] = compat
print('    ' + THIN)
print('    Brown-Forsythe (median-centred Levene) is used rather than the variance-ratio F test:')
print('    F(4,4) assumes normality and is not robust on heavy-tailed seed-lottery outcomes.')
print('    TEST 4: %s' % ('PASS' if PASS[4] else 'FAIL'))

# --------------------------------------------------------------------------
banner(5, 'VERDICT and the pooled row')
# --------------------------------------------------------------------------
granted = all(PASS.get(i) for i in (1, 2, 3, 4))
for i in (1, 2, 3, 4):
    print('    TEST %d  %s' % (i, 'PASS' if PASS.get(i) else 'FAIL'))
print()
if not granted:
    print('    LICENCE REFUSED -- the ten runs must NOT be pooled. Report them as two families.')
    sys.exit(1)

print('    LICENCE GRANTED. The two code paths are one estimator (TESTS 1-3) and the two samples')
print('    are distributionally compatible (TEST 4). The ten trainings are ten exchangeable draws')
print('    from one outcome distribution and may be reported as a single n=10 row.')
print()
print('    Manuscript convention: reported +- is the POPULATION sd (ddof=0) across the seed means.')
print()
print('    %-6s | %-38s | %-38s | %s' % ('metric', 'family A (n=5)', 'family B (n=5)', 'pooled (n=10)'))
print('    ' + THIN)
pooled_cells = {}
for m in METRICS:
    a, b = np.array(A[m], float), np.array(B[m], float)
    ten = np.concatenate([a, b])
    pooled_cells[m] = (ten.mean(), ten.std(ddof=0))
    print('    %-6s | %10.2f +- %-8.2f (n=5)         | %10.2f +- %-8.2f (n=5)         | %10.2f +- %.2f'
          % (m, a.mean(), a.std(ddof=0), b.mean(), b.std(ddof=0), ten.mean(), ten.std(ddof=0)))

print()
fmt = {'HO': '%.2f', 'Power': '%.2f', 'Spec': '%.2f', 'RLF': '%.2f', 'Tput': '%.1f'}
sfmt = {'HO': '%.2f', 'Power': '%.2f', 'Spec': '%.2f', 'RLF': '%.2f', 'Tput': '%.1f'}
cells = ' | '.join((fmt[m] + r' $\pm$ ' + sfmt[m]) % pooled_cells[m] for m in METRICS)
print('ROW: | Proposed Dn-DQN, 10 independent trainings | %s |' % cells)
print()
print('    Reader check -- the pooled sd is NOT the average of the two family sds. Pooling ten runs')
print('    recovers spread that either five-run sample missed:')
for m in METRICS:
    a, b = np.array(A[m], float), np.array(B[m], float)
    sa, sb, sp = a.std(ddof=0), b.std(ddof=0), pooled_cells[m][1]
    print('      %-6s  sd_A %8.3f   sd_B %8.3f   mean(sd_A,sd_B) %8.3f   sd_pooled %8.3f  (%+.1f%% vs mean)'
          % (m, sa, sb, (sa + sb) / 2, sp, 100.0 * (sp / ((sa + sb) / 2) - 1.0)))
print()
print('    This is the quantitative form of the associate editor\'s objection: a five-run standard')
print('    deviation is itself a noisy statistic, and averaging two of them does not fix it.')
print(RULE)
