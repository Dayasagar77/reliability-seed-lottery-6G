#!/usr/bin/env python3
"""
test_merge_shards.py — proves merge_shards.py REFUSES the four ways a merge can corrupt a result.

The merge is the last point at which a mistake is still cheap. After it, a wrong number is in a
table, and a table is what gets published. So the refusals matter more than the happy path, and
they are tested first.

SAFETY: every fixture is synthetic and lives in a throwaway temp directory with its own results/
folder. Nothing here reads or writes /root/work/exp/results. The synthetic metric values are
deliberately impossible (HO 11.11, 22.22, ...) so that if a fixture ever escaped the sandbox it
would be instantly recognisable as not a measurement.
"""
import csv, os, shutil, subprocess, sys, tempfile

import campaign_units as CU

HERE = os.path.dirname(os.path.abspath(__file__))
FAKE = {'HO': '11.11', 'Power': '22.22', 'Spec': '33.33', 'RLF': '44.44', 'Tput': '555.5',
        'train_min': '9.9'}
FAILURES = []


def shard(ws, family, method, seed, venue='laptop', **over):
    """Write one synthetic shard exactly where the plan says that unit's file goes."""
    p = os.path.join(ws, CU.rows_path(family, method, seed, venue))
    row = {'seed': seed, 'steps': CU.STEPS, 'pipeline': f'{venue}-singlepass', **FAKE}
    if family == 'matched':
        row['method'] = {'dndqn': 'Proposed Dn-DQN', 'ddqn': 'DDQN+PER', 'std': 'Standard DQN'}[method]
    if family == 'guard':
        row['arm'], row['Override'] = 'postguard', '1.23'
    row.update(over)
    order = (['method'] if family == 'matched' else []) + ['seed', 'steps'] + \
            (['arm'] if family == 'guard' else []) + ['pipeline'] + \
            ['HO', 'Power', 'Spec'] + (['Override'] if family == 'guard' else []) + \
            ['RLF', 'Tput', 'train_min']
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=order)
        w.writeheader()
        w.writerow({k: row[k] for k in order})
    return p


def workspace(drop=(), **kw):
    ws = tempfile.mkdtemp(prefix='mergetest_')
    for f in ('merge_shards.py', 'campaign_units.py'):
        shutil.copy(os.path.join(HERE, f), ws)
    os.makedirs(os.path.join(ws, 'results'), exist_ok=True)
    for u in CU.units(venue='laptop'):
        if u['name'] in drop:
            continue
        shard(ws, u['family'], u['method'], u['seed'], **kw.get(u['name'], {}))
    return ws


def run_merge(ws, *args):
    r = subprocess.run([sys.executable, '-u', 'merge_shards.py', '--venue', 'laptop', *args],
                       cwd=ws, capture_output=True, text=True)
    return r.stdout + r.stderr


def check(what, cond, detail=''):
    print(f"  {'PASS' if cond else 'FAIL'} | {what}{(' | ' + detail) if detail and not cond else ''}")
    if not cond:
        FAILURES.append(what)


print("=" * 96)
print("  merge_shards.py — refusal behaviour (synthetic fixtures, sandboxed)")
print("=" * 96)

# 1. complete and consistent -> writes all four
ws = workspace()
out = run_merge(ws)
check('complete campaign merges all 4 families', '4/4 families merged' in out, out[-600:])
for fam, n in CU.EXPECTED_N.items():
    tgt = os.path.join(ws, CU.MERGE_TARGETS[fam].format(k=800, venue='laptop'))
    got = len(list(csv.DictReader(open(tgt, newline='')))) if os.path.exists(tgt) else 0
    check(f'{fam}: merged ledger holds {n} rows', got == n, f'got {got}')
matched = list(csv.DictReader(open(os.path.join(ws, 'results/matched_rel_5seed_800k_laptop.csv'))))
check('matched keeps all three methods', len({r['method'] for r in matched}) == 3)
check('merge never emits a cloud ledger name',
      not any(os.path.exists(os.path.join(ws, p)) for p in
              ('results/matched_rel_5seed_800k.csv', 'results/lottery_flat_10seed_800k.csv',
               'results/safedqn_5seed_800k.csv', 'results/lottery_guard_800k_s0_4.csv')))
shutil.rmtree(ws)

# 2. one seed missing -> that family refuses, the others still merge
ws = workspace(drop={'lottery_s7'})
out = run_merge(ws)
check('missing seed is named in the report', 'lottery_s7' in out)
check('incomplete family is not written',
      not os.path.exists(os.path.join(ws, 'results/lottery_flat_10seed_800k_laptop.csv')))
check('incomplete family refuses explicitly', 'NOT WRITTEN' in out)
check('other families still merge', '3/4 families merged' in out, out[-400:])
shutil.rmtree(ws)

# 3. a cloud-tagged row inside a laptop merge -> mixed venues, refuse
ws = workspace(**{'guard_s3': {'venue': 'laptop', 'pipeline': 'cloud-singlepass'}})
out = run_merge(ws)
check('mixed venues detected', 'MIXED VENUES' in out, out[-400:])
check('mixed-venue family is not written',
      not os.path.exists(os.path.join(ws, 'results/lottery_guard_10seed_800k_laptop.csv')))
shutil.rmtree(ws)

# 4. two different results claiming the same unit -> conflict, refuse
ws = workspace()
p = os.path.join(ws, CU.rows_path('safedqn', None, 2, 'laptop'))
rows = list(csv.DictReader(open(p, newline='')))
dup = dict(rows[0]); dup['HO'] = '99.99'
with open(p, 'a', newline='') as f:
    csv.DictWriter(f, fieldnames=list(rows[0].keys())).writerow(dup)
out = run_merge(ws)
check('conflicting duplicate detected', 'CONFLICT' in out, out[-400:])
check('shard holding 2 rows is flagged', 'must hold exactly 1' in out)
check('conflicted family is not written',
      not os.path.exists(os.path.join(ws, 'results/safedqn_5seed_800k_laptop.csv')))
shutil.rmtree(ws)

# 5. untagged provenance -> refuse (a row whose venue is unknown cannot be reported)
ws = workspace(**{'matched_ddqn_s1': {'pipeline': ''}})
out = run_merge(ws)
check('untagged provenance refused', 'provenance unknown' in out, out[-400:])
shutil.rmtree(ws)

# 6. --allow-partial reports but still writes nothing
ws = workspace(drop={'guard_s9'})
out = run_merge(ws, '--allow-partial')
check('--allow-partial writes nothing for the broken family',
      not os.path.exists(os.path.join(ws, 'results/lottery_guard_10seed_800k_laptop.csv')))
check('--allow-partial still prints the partial summary', 'n= 9' in out or '(n= 9)' in out, out[-500:])
shutil.rmtree(ws)

print("=" * 96)
if FAILURES:
    print(f"  {len(FAILURES)} FAILURE(S): {FAILURES}")
    sys.exit(1)
print("  ALL REFUSALS WORK — merge cannot silently publish an incomplete or mixed-venue sigma")
print("=" * 96)
