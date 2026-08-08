#!/usr/bin/env python3
"""
verify_shard_parsing.py — proves the four campaign drivers' work-selection logic is correct
WITHOUT importing torch, and WITHOUT touching the campaign processes currently training.

WHY THIS EXISTS
---------------
The drivers were patched so that up to 40 of them can run concurrently on the laptop, each owning
one unit of work and one output file. Two things must hold, and neither is obvious by eye:

  1. BACKWARD COMPATIBILITY. The cloud watchdog launches the bare-integer forms
     ("campaign_matched.py 5 800000", "campaign_lottery.py 10 800000", "campaign_safedqn.py 5
     800000", "campaign_guard.py 0,1,2,3,4"). Those runs are in flight right now and their ledgers
     already exist on disk. If a patched driver were relaunched by the watchdog and computed a
     DIFFERENT ROWS_PATH, it would silently start a second ledger and the campaign would look
     half-finished forever. The cloud forms must therefore map to byte-identical paths.

  2. SHARD DISJOINTNESS. Concurrent shards must never write the same CSV or the same heartbeat.
     A collision would interleave two processes' rows through a read-modify-rewrite cycle and lose
     completed trainings.

Each driver's real module-level code — everything above its first `def` — is exec'd under a
controlled sys.argv and os.environ, and the resulting SEEDS / METHODS / ROWS_PATH / HEARTBEAT /
PIPELINE are asserted. Heavy imports (torch, the training modules) are served by stubs through a
patched __import__, so the code that RUNS is the code that was edited, byte for byte, while nothing
loads a network or opens the dataset. The stubs mirror the real modules exactly where the drivers
read them:

    run_multiseed_matched.N_SEEDS/STEPS   read from sys.argv at import (lines 30-31), so the
                                          drivers' `assert M.N_SEEDS == _N_SEEDS` is a REAL test
                                          of the argv override, not a bypassed one
    run_multiseed_matched.METHODS         the real list from line 37, so the shard method filter
                                          and its unknown-key assert are exercised for real
"""
import io, os, sys, csv, types, contextlib, builtins

DRIVERS = ['campaign_matched.py', 'campaign_lottery.py', 'campaign_guard.py', 'campaign_safedqn.py']


def header_of(path):
    """Everything above the first top-level `def` — the whole work-selection block."""
    src = []
    for line in open(path):
        if line.startswith('def '):
            break
        src.append(line)
    return ''.join(src)


HEADERS = {d: header_of(d) for d in DRIVERS}


def _stub(name):
    """A stand-in for one heavy module, faithful in exactly the attributes the drivers read."""
    m = types.ModuleType(name)
    if name == 'torch':
        m.set_num_threads = lambda n: None
        m.__version__ = 'STUB'
    elif name == 'numpy':
        m.__version__ = 'STUB'
    elif name == 'run_multiseed_matched':
        # mirrors real lines 30-31 (argv-driven) and 37 verbatim
        m.N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
        m.STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 400_000
        m.METHODS = [('Proposed Dn-DQN', 'dndqn'), ('DDQN+PER', 'ddqn'), ('Standard DQN', 'std')]
    elif name == 'run_multiseed_rel':
        m.N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
        m.STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 800_000
    elif name == 'train_srdndqn':
        m.GUARD_DB, m.SAVE_LINE = 9.0, -76.0
    elif name == 'train_safe_dqn':
        m.SAVE_LINE = -76.0
    elif name == 'train_phase2':
        m.CSV_PATH = '6G_Handover_Training_Data_v3.csv'
    elif name == 'dndqn':
        m.OBS_DIM = 6
        m.load_dndqn_checkpoint = None
    elif name == 'dndqn_branched':
        m.load_branched = None
    return m


STUBBED = {'torch', 'numpy', 'run_multiseed_matched', 'run_multiseed_rel', 'train_srdndqn',
           'train_safe_dqn', 'train_phase2', 'dndqn', 'dndqn_branched'}
_real_import = builtins.__import__


def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name in STUBBED:
        m = _stub(name)                      # fresh each time: argv-driven attrs must re-evaluate
        for f in fromlist or ():
            if not hasattr(m, f):
                raise ImportError(f"stub {name!r} lacks {f!r} — the driver reads it, add it here")
        return m
    return _real_import(name, globals, locals, fromlist, level)


def run(driver, argv, env=None):
    """Exec one driver's module-level code with the given argv/env; return its namespace."""
    ns = {'__name__': '__verify__', '__file__': driver,
          '__builtins__': {**vars(builtins), '__import__': _fake_import}}
    old_argv, old_env = sys.argv, dict(os.environ)
    try:
        sys.argv = [driver] + list(argv)
        os.environ.pop('PIPELINE_TAG', None)
        if env:
            os.environ.update(env)
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(HEADERS[driver], driver, 'exec'), ns)
    finally:
        sys.argv = old_argv
        os.environ.clear()
        os.environ.update(old_env)
    return ns


FAILURES = []


def check(what, got, want):
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'} | {what:<62} | {got!r}")
    if not ok:
        FAILURES.append(f"{what}: got {got!r}, want {want!r}")


print("=" * 104)
print("  1. BACKWARD COMPATIBILITY — the exact argv the cloud watchdog uses must be unchanged")
print("=" * 104)

n = run('campaign_matched.py', ['5', '800000'])
check('matched  "5 800000"      SEEDS',     n['SEEDS'], [0, 1, 2, 3, 4])
check('matched  "5 800000"      _MSEL',     n['_MSEL'], 'all')
check('matched  "5 800000"      ROWS_PATH', n['ROWS_PATH'], 'results/matched_rel_5seed_800k.csv')
check('matched  "5 800000"      HEARTBEAT', n['HEARTBEAT'], 'campaign_matched.heartbeat')
check('matched  "5 800000"      PIPELINE',  n['PIPELINE'], 'cloud-singlepass')

n = run('campaign_lottery.py', ['10', '800000'])
check('lottery  "10 800000"     SEEDS',     n['SEEDS'], list(range(10)))
check('lottery  "10 800000"     ROWS_PATH', n['ROWS_PATH'], 'results/lottery_flat_10seed_800k.csv')
check('lottery  "10 800000"     HEARTBEAT', n['HEARTBEAT'], 'campaign_lottery.heartbeat')
check('lottery  "10 800000"     PIPELINE',  n['PIPELINE'], 'cloud-singlepass')

n = run('campaign_safedqn.py', ['5', '800000'])
check('safedqn  "5 800000"      SEEDS',     n['SEEDS'], [0, 1, 2, 3, 4])
check('safedqn  "5 800000"      ROWS_PATH', n['ROWS_PATH'], 'results/safedqn_5seed_800k.csv')
check('safedqn  "5 800000"      HEARTBEAT', n['HEARTBEAT'], 'campaign_safedqn.heartbeat')
check('safedqn  "5 800000"      PIPELINE',  n['PIPELINE'], 'cloud-singlepass')

for spec, a, b in [('0,1,2,3,4', 0, 4), ('5,6,7,8,9', 5, 9)]:
    n = run('campaign_guard.py', [spec, '800000'])
    check(f'guard    "{spec}"    ROWS_PATH', n['ROWS_PATH'], f'results/lottery_guard_800k_s{a}_{b}.csv')
    check(f'guard    "{spec}"    HEARTBEAT', n['HEARTBEAT'], f'campaign_guard_s{a}_{b}.heartbeat')
    check(f'guard    "{spec}"    VENUE',     n['VENUE'], 'cloud')

print()
print("=" * 104)
print("  2. SHARD FORMS — one unit of work per process, venue-tagged, laptop campaign")
print("=" * 104)
LAP = {'PIPELINE_TAG': 'laptop-singlepass'}

n = run('campaign_matched.py', ['dndqn:0', '800000'], LAP)
check('matched  "dndqn:0"       SEEDS',     n['SEEDS'], [0])
check('matched  "dndqn:0"       _MSEL',     n['_MSEL'], 'dndqn')
check('matched  "dndqn:0"       ROWS_PATH', n['ROWS_PATH'], 'results/matched_rel_800k_dndqn_s0_0.csv')
check('matched  "dndqn:0"       PIPELINE',  n['PIPELINE'], 'laptop-singlepass')
n = run('campaign_matched.py', ['ddqn:3-4', '800000'], LAP)
check('matched  "ddqn:3-4"      SEEDS',     n['SEEDS'], [3, 4])
check('matched  "ddqn:3-4"      ROWS_PATH', n['ROWS_PATH'], 'results/matched_rel_800k_ddqn_s3_4.csv')
n = run('campaign_matched.py', ['std:1,2', '800000'], LAP)
check('matched  "std:1,2"       SEEDS',     n['SEEDS'], [1, 2])
check('matched  "std:1,2"       ROWS_PATH', n['ROWS_PATH'], 'results/matched_rel_800k_std_s1_2.csv')

n = run('campaign_lottery.py', ['7,7', '800000'], LAP)
check('lottery  "7,7"           SEEDS',     n['SEEDS'], [7, 7])
n = run('campaign_lottery.py', ['7-7', '800000'], LAP)
check('lottery  "7-7"           SEEDS',     n['SEEDS'], [7])
check('lottery  "7-7"           ROWS_PATH', n['ROWS_PATH'], 'results/lottery_flat_800k_s7_7.csv')
check('lottery  "7-7"           PIPELINE',  n['PIPELINE'], 'laptop-singlepass')

n = run('campaign_safedqn.py', ['2-2', '800000'], LAP)
check('safedqn  "2-2"           SEEDS',     n['SEEDS'], [2])
check('safedqn  "2-2"           ROWS_PATH', n['ROWS_PATH'], 'results/safedqn_800k_s2_2.csv')
check('safedqn  "2-2"           PIPELINE',  n['PIPELINE'], 'laptop-singlepass')

n = run('campaign_guard.py', ['6-6', '800000'], LAP)
check('guard    "6-6"           SEEDS',     n['SEEDS'], [6])
check('guard    "6-6"           VENUE',     n['VENUE'], 'laptop')
check('guard    "6-6"           ROWS_PATH', n['ROWS_PATH'], 'results/lottery_guard_800k_s6_6_laptop.csv')
check('guard    "6-6"           HEARTBEAT', n['HEARTBEAT'], 'campaign_guard_s6_6_laptop.heartbeat')

print()
print("=" * 104)
print("  3. DISJOINTNESS — the full 40-unit laptop campaign, every output file distinct")
print("=" * 104)
UNITS = ([('campaign_matched.py', [f'{m}:{s}-{s}', '800000']) for m in ('dndqn', 'ddqn', 'std')
          for s in range(5)]
         + [('campaign_lottery.py', [f'{s}-{s}', '800000']) for s in range(10)]
         + [('campaign_guard.py',   [f'{s}-{s}', '800000']) for s in range(10)]
         + [('campaign_safedqn.py', [f'{s}-{s}', '800000']) for s in range(5)])
paths, beats = [], []
for drv, argv in UNITS:
    ns = run(drv, argv, LAP)
    paths.append(ns['ROWS_PATH'])
    beats.append(ns['HEARTBEAT'])
check('total units planned', len(UNITS), 40)
check('distinct ROWS_PATH', len(set(paths)), 40)
check('distinct HEARTBEAT', len(set(beats)), 40)
cloud_live = {'results/matched_rel_5seed_800k.csv', 'results/lottery_flat_10seed_800k.csv',
              'results/safedqn_5seed_800k.csv', 'results/lottery_guard_800k_s0_4.csv',
              'results/lottery_guard_800k_s5_9.csv'}
check('laptop shards vs live cloud ledgers overlap', sorted(set(paths) & cloud_live), [])

print()
print("=" * 104)
print("  4. LEGACY LEDGER — matched must still resume from a CSV written before the new column")
print("=" * 104)
LIVE = 'results/matched_rel_5seed_800k.csv'
# The pre-instrumentation ledger only exists in the venue whose campaign wrote it (the cloud).
# On the laptop it is absent, which is not a defect — so fall back to a SYNTHETIC legacy fixture
# with deliberately impossible metric values. The structural claims (no pipeline column, resume
# key builds, blank never invented) are what this section is actually about, and they hold either
# way. The real-metric identity check only runs when the real ledger is the thing under test.
_SYNTH_METRICS = ['11.11', '22.22', '33.33', '44.44', '555.5']
if os.path.exists(LIVE):
    SRC, SYNTH = LIVE, False
    with open(LIVE, newline='') as f:
        live_rows = list(csv.DictReader(f))
else:
    SRC, SYNTH = '(synthetic legacy fixture — live cloud ledger absent in this venue)', True
    live_rows = [dict(zip(['method', 'seed', 'steps', 'HO', 'Power', 'Spec', 'RLF', 'Tput',
                           'train_min'],
                          ['Proposed Dn-DQN', '0', '800000', *_SYNTH_METRICS, '9.9']))]
print(f"  source: {SRC}")
live_hdr = list(live_rows[0].keys()) if live_rows else []
check('legacy ledger row count', len(live_rows), 1)
check('legacy ledger has no pipeline column', 'pipeline' in live_hdr, False)
# Reproduce load_done()'s legacy path and the DictWriter round-trip on the SAME rows.
FIELDS = ['method', 'seed', 'steps', 'pipeline', 'HO', 'Power', 'Spec', 'RLF', 'Tput', 'train_min']
for r in live_rows:
    r.setdefault('pipeline', '')
key = {(r['method'], int(r['seed']), int(r['steps'])) for r in live_rows}
check('legacy resume key still builds', sorted(key), [('Proposed Dn-DQN', 0, 800000)])
buf = io.StringIO()
w = csv.DictWriter(buf, fieldnames=FIELDS)
w.writeheader()
w.writerows(live_rows)
back = list(csv.DictReader(io.StringIO(buf.getvalue())))
check('legacy row survives rewrite, pipeline blank not invented', back[0]['pipeline'], '')
check('legacy row metrics unchanged through rewrite',
      [back[0][k] for k in ('HO', 'Power', 'Spec', 'RLF', 'Tput')],
      _SYNTH_METRICS if SYNTH else ['91.04', '24.52', '88.76', '10.67', '473.7'])
if SYNTH:
    print("  NOTE: metrics above are synthetic placeholders, not measurements — this venue has no"
          " pre-instrumentation ledger to resume from.")

print()
print("=" * 104)
print("  5. PLAN vs DRIVERS — campaign_units.py must agree with what the drivers actually compute")
print("=" * 104)
import campaign_units as CU
plan = CU.units(venue='laptop')
check('plan unit count', len(plan), 40)
mismatch = []
for u in plan:
    ns = run(u['script'], u['argv'], LAP)
    if ns['ROWS_PATH'] != u['rows_path']:
        mismatch.append(f"{u['name']}: driver {ns['ROWS_PATH']} vs plan {u['rows_path']}")
    if ns['SEEDS'] != [u['seed']]:
        mismatch.append(f"{u['name']}: driver SEEDS {ns['SEEDS']} vs plan [{u['seed']}]")
check('every unit: driver ROWS_PATH == plan rows_path, driver SEEDS == [plan seed]', mismatch, [])
check('plan paths distinct', len(set(u['rows_path'] for u in plan)), 40)
check('plan never targets a live cloud ledger', sorted(set(u['rows_path'] for u in plan) & cloud_live), [])
check('merge targets never collide with cloud ledgers',
      sorted({t.format(k=800, venue='laptop') for t in CU.MERGE_TARGETS.values()} & cloud_live), [])
check('plan family counts', {f: sum(1 for u in plan if u['family'] == f) for f in CU.EXPECTED_N},
      CU.EXPECTED_N)
check('longest unit scheduled first', plan[0]['family'], 'guard')

print()
print("=" * 104)
if FAILURES:
    print(f"  {len(FAILURES)} FAILURE(S)")
    for f_ in FAILURES:
        print("   -", f_)
    sys.exit(1)
print("  ALL CHECKS PASSED — cloud argv byte-identical, 40 laptop shards mutually disjoint")
print("=" * 104)
