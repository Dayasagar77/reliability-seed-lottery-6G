#!/usr/bin/env python3
"""
test_analyze_variance.py — proves analyze_variance.py tests ONE pipeline and refuses to mix.

WHY THIS TEST EXISTS
--------------------
Until 2026-08-01 analyze_variance.py hardcoded the CLOUD result filenames and printed
"pipeline = cloud single-pass" over whatever it found. That was correct when the cloud was the only
campaign; it became a live hazard the moment the laptop became the reporting pipeline, because the
script would then have reported torch-2.13 numbers as the paper's torch-2.5.1 result. That is a
quieter version of the exact defect the Results in Engineering AE rejected the paper for.

The subtle part is not the filenames. It is that ONLY `guard` shards carry a venue suffix in their
names -- a lottery shard is called `lottery_flat_800k_s3_3.csv` in BOTH venues. So a fix that only
swapped filenames would still silently accept a cloud row in a laptop run. Provenance therefore has
to come from the `pipeline` COLUMN inside the row, and that is what these tests pin down.

SAFETY: every fixture is synthetic, in a throwaway temp directory with its own results/ folder.
Nothing here reads or writes /root/work/exp/results. RLF values are drawn from a fixed list of
deliberately implausible numbers so that a fixture escaping the sandbox would be obvious on sight.
"""
import csv, json, os, shutil, subprocess, sys, tempfile

import numpy as np
from scipy import stats

import campaign_units as CU

HERE = os.path.dirname(os.path.abspath(__file__))
FAILURES = []

# Deliberately implausible RLF values: a real sustained-RLF rate never looks like this.
FLAT_RLF = [41.1, 42.2, 43.3, 44.4, 45.5, 46.6, 47.7, 48.8, 49.9, 40.0]     # wide on purpose
GUARD_RLF = [31.1, 31.2, 31.3, 31.4, 31.5, 31.6, 31.7, 31.8, 31.9, 31.0]    # narrow on purpose


def row(family, seed, rlf, venue='laptop', pipeline=None):
    r = {'seed': str(seed), 'steps': str(CU.STEPS),
         'pipeline': f'{venue}-singlepass' if pipeline is None else pipeline,
         'HO': '11.11', 'Power': '22.22', 'Spec': '33.33',
         'RLF': f'{rlf}', 'Tput': '555.5', 'train_min': '9.9'}
    if family == 'guard':
        r['arm'], r['Override'] = 'postguard', '1.23'
    return r


def write_csv(path, rows, drop_pipeline=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = [k for k in rows[0] if not (drop_pipeline and k == 'pipeline')]
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def workspace():
    ws = tempfile.mkdtemp(prefix='varitest_')
    for f in ('analyze_variance.py', 'campaign_units.py'):
        shutil.copy(os.path.join(HERE, f), ws)
    os.makedirs(os.path.join(ws, 'results'), exist_ok=True)
    return ws


def run(ws, *args):
    r = subprocess.run([sys.executable, '-u', 'analyze_variance.py', *args],
                       cwd=ws, capture_output=True, text=True)
    return r.stdout + r.stderr


def load_json(ws, venue):
    p = os.path.join(ws, f'results/variance_tests_{venue}.json')
    return json.load(open(p)) if os.path.exists(p) else None


def check(what, cond, detail=''):
    print(f"  {'PASS' if cond else 'FAIL'} | {what}{(' | ' + detail) if detail and not cond else ''}")
    if not cond:
        FAILURES.append(what)


print("=" * 96)
print("  analyze_variance.py — pipeline isolation (synthetic fixtures, sandboxed)")
print("=" * 96)

# ---------------------------------------------------------------------------------------------
# 1. The happy path: a complete laptop merge is the tested sample, and it is labelled laptop.
# ---------------------------------------------------------------------------------------------
print("\n1. complete laptop merge")
ws = workspace()
write_csv(os.path.join(ws, CU.MERGE_TARGETS['lottery'].format(k=800, venue='laptop')),
          [row('lottery', s, FLAT_RLF[s]) for s in range(10)])
write_csv(os.path.join(ws, CU.MERGE_TARGETS['guard'].format(k=800, venue='laptop')),
          [row('guard', s, GUARD_RLF[s]) for s in range(10)])
out = run(ws)
j = load_json(ws, 'laptop')
check('defaults to the laptop (reporting) pipeline', 'venue = laptop' in out, out[:400])
check('says laptop is the reporting pipeline', 'REPORTING pipeline' in out)
check('does not print the old cloud banner', 'pipeline = cloud single-pass' not in out)
check('tested sample is n=10 vs n=10', 'n=10 vs n=10' in out, out[-900:])
check('sources the merged ledger', 'merged ledger' in out)
check('json written under the venue name', j is not None)
check('json records venue=laptop', j and j['venue'] == 'laptop')
check('json marks the sample complete', j and j['complete'] is True)
check('json records only laptop tags', j and j['pipeline_tags'] == ['laptop-singlepass'],
      str(j and j['pipeline_tags']))
check('no un-suffixed variance_tests.json is written',
      not os.path.exists(os.path.join(ws, 'results/variance_tests.json')))
check('no cloud json is written', load_json(ws, 'cloud') is None)

# Numeric identity: the statistics must be exactly what scipy gives on the same two vectors.
# The revision changed sourcing and labelling only; nothing statistical was touched.
bf = stats.levene(FLAT_RLF, GUARD_RLF, center='median')
check('Brown-Forsythe matches scipy on the same vectors',
      j and abs(j['tests']['brown_forsythe']['p'] - float(bf.pvalue)) < 1e-12,
      f"script {j and j['tests']['brown_forsythe']['p']} vs scipy {float(bf.pvalue)}")
check('sigma is population form (ddof=0)',
      j and abs(j['tests']['sigma']['flat'] - float(np.std(FLAT_RLF))) < 1e-9)
check('bootstrap seed still 20260801', j and j['tests']['sigma']['boot_seed'] == 20260801)
shutil.rmtree(ws)

# ---------------------------------------------------------------------------------------------
# 2. THE ONE THAT MATTERS: a cloud-tagged row sitting in a laptop-named file is dropped.
#    Lottery shards have the SAME filename in both venues, so this cannot be caught by name.
# ---------------------------------------------------------------------------------------------
print("\n2. cloud row inside a laptop ledger")
ws = workspace()
rows = [row('lottery', s, FLAT_RLF[s]) for s in range(10)]
rows[4]['pipeline'] = 'cloud-singlepass'          # one seed retrained on the wrong machine
write_csv(os.path.join(ws, CU.MERGE_TARGETS['lottery'].format(k=800, venue='laptop')), rows)
write_csv(os.path.join(ws, CU.MERGE_TARGETS['guard'].format(k=800, venue='laptop')),
          [row('guard', s, GUARD_RLF[s]) for s in range(10)])
out = run(ws)
j = load_json(ws, 'laptop')
check('foreign-venue row is dropped, not pooled', "DROPPED 1 row(s) tagged 'cloud-singlepass'" in out,
      out[:1200])
check('dropped row is excluded from n', 'n=9 vs n=10' in out, out[-900:])
check('the dropped value is absent from the tested vector',
      j and FLAT_RLF[4] not in j['flat_rlf'], str(j and j['flat_rlf']))
check('incomplete arm is called out', 'INCOMPLETE: flat has 9/10 seeds' in out)
check('json marks the sample incomplete', j and j['complete'] is False)
shutil.rmtree(ws)

# ---------------------------------------------------------------------------------------------
# 3. An untagged row cannot be reported at all — provenance unknown.
# ---------------------------------------------------------------------------------------------
print("\n3. untagged provenance")
ws = workspace()
rows = [row('lottery', s, FLAT_RLF[s]) for s in range(10)]
rows[7]['pipeline'] = ''
write_csv(os.path.join(ws, CU.MERGE_TARGETS['lottery'].format(k=800, venue='laptop')), rows)
write_csv(os.path.join(ws, CU.MERGE_TARGETS['guard'].format(k=800, venue='laptop')),
          [row('guard', s, GUARD_RLF[s]) for s in range(10)])
out = run(ws)
j = load_json(ws, 'laptop')
check('untagged row refused', 'provenance unknown, do not report' in out, out[:1200])
check('untagged row excluded from n', j and j['n_flat'] == 9, str(j and j['n_flat']))
shutil.rmtree(ws)

# ---------------------------------------------------------------------------------------------
# 4. A resumed run is the same toolchain, so it counts — but it must be disclosed.
# ---------------------------------------------------------------------------------------------
print("\n4. resumed (restarted) run")
ws = workspace()
rows = [row('lottery', s, FLAT_RLF[s]) for s in range(10)]
rows[2]['pipeline'] = 'laptop-resumed'
write_csv(os.path.join(ws, CU.MERGE_TARGETS['lottery'].format(k=800, venue='laptop')), rows)
write_csv(os.path.join(ws, CU.MERGE_TARGETS['guard'].format(k=800, venue='laptop')),
          [row('guard', s, GUARD_RLF[s]) for s in range(10)])
out = run(ws)
j = load_json(ws, 'laptop')
check('resumed row is KEPT (same toolchain)', j and j['n_flat'] == 10, str(j and j['n_flat']))
check('resumed row is flagged for disclosure', 'DISCLOSE' in out, out[:1200])
check('json carries both tags', j and j['pipeline_tags'] == ['laptop-resumed', 'laptop-singlepass'],
      str(j and j['pipeline_tags']))
shutil.rmtree(ws)

# ---------------------------------------------------------------------------------------------
# 5. Mid-campaign: per-seed shards, no merge yet. Usable as a preview, never as a result.
# ---------------------------------------------------------------------------------------------
print("\n5. shards present, merge not yet run")
ws = workspace()
for s in range(10):
    write_csv(os.path.join(ws, CU.rows_path('lottery', None, s, 'laptop')),
              [row('lottery', s, FLAT_RLF[s])])
for s in range(6):
    write_csv(os.path.join(ws, CU.rows_path('guard', None, s, 'laptop')),
              [row('guard', s, GUARD_RLF[s])])
out = run(ws)
j = load_json(ws, 'laptop')
check('falls back to per-seed shards', 'per-seed shards (merge not yet run)' in out, out[:1400])
check('in-flight preview is labelled as such', 'CAMPAIGN IN FLIGHT' in out)
check('partial guard arm counted honestly', 'n=10 vs n=6' in out, out[-900:])
check('preview json marked incomplete', j and j['complete'] is False)
check('preview json names its shard sources', j and len(j['sources']['guard']) == 6)
shutil.rmtree(ws)

# ---------------------------------------------------------------------------------------------
# 6. The same shard filenames, but written by the cloud: a laptop run must find NOTHING.
#    This is the filename collision that a name-only fix would have walked straight into.
# ---------------------------------------------------------------------------------------------
print("\n6. cloud-written shards, laptop run")
ws = workspace()
for s in range(10):
    write_csv(os.path.join(ws, CU.rows_path('lottery', None, s, 'laptop')),
              [row('lottery', s, FLAT_RLF[s], venue='cloud')])
out = run(ws)
check('cloud shards rejected by a laptop run', 'DROPPED 10 row(s) tagged' in out, out[:1400])
check('laptop tested sample is empty, not borrowed', 'flat reliability-tilted Dn-DQN     n=0' in out,
      out[-900:])
check('nothing written when the sample is empty', load_json(ws, 'laptop') is None)
shutil.rmtree(ws)

# ---------------------------------------------------------------------------------------------
# 7. --venue cloud still reads the legacy pre-instrumentation ledgers (no pipeline column).
# ---------------------------------------------------------------------------------------------
print("\n7. cloud venue, legacy untagged ledgers")
ws = workspace()
write_csv(os.path.join(ws, 'results/lottery_flat_10seed_800k.csv'),
          [row('lottery', s, FLAT_RLF[s]) for s in range(10)], drop_pipeline=True)
write_csv(os.path.join(ws, 'results/lottery_guard_800k_s0_4.csv'),
          [row('guard', s, GUARD_RLF[s]) for s in range(5)], drop_pipeline=True)
write_csv(os.path.join(ws, 'results/lottery_guard_800k_s5_9.csv'),
          [row('guard', s, GUARD_RLF[s]) for s in range(5, 10)], drop_pipeline=True)
out = run(ws, '--venue', 'cloud')
j = load_json(ws, 'cloud')
check('legacy cloud ledgers are found', 'legacy pre-instrumentation cloud ledger' in out, out[:1400])
check('legacy provenance caveat is printed', 'NO pipeline column' in out)
check('cloud arm labelled as replication, not the paper', 'REPLICATION arm' in out)
check('both legacy guard shards read', j and j['n_guard'] == 10, str(j and j['n_guard']))
check('cloud json is venue-suffixed', j is not None and load_json(ws, 'laptop') is None)
shutil.rmtree(ws)

# ---------------------------------------------------------------------------------------------
# 8. Two rows claiming the same seed = two trainings, one lying. Refuse outright.
# ---------------------------------------------------------------------------------------------
print("\n8. duplicate seed")
ws = workspace()
rows = [row('lottery', s, FLAT_RLF[s]) for s in range(10)] + [row('lottery', 3, 99.9)]
write_csv(os.path.join(ws, CU.MERGE_TARGETS['lottery'].format(k=800, venue='laptop')), rows)
out = run(ws)
check('duplicate seed is fatal', 'FATAL: duplicate seed(s) [3]' in out, out[-600:])
check('nothing written on a fatal', load_json(ws, 'laptop') is None)
shutil.rmtree(ws)

# ---------------------------------------------------------------------------------------------
# 9. Empty workspace: say what was looked for, do not imply a defect.
# ---------------------------------------------------------------------------------------------
print("\n9. nothing on disk yet")
ws = workspace()
out = run(ws)
check('empty state is explained, not silent', 'NO DATA — none of these exist yet' in out, out[:900])
check('the expected merge target is named',
      'results/lottery_flat_10seed_800k_laptop.csv' in out)
check('no json written from nothing', load_json(ws, 'laptop') is None)
shutil.rmtree(ws)

print()
print("=" * 96)
if FAILURES:
    print(f"  {len(FAILURES)} FAILURE(S): {FAILURES}")
    sys.exit(1)
print("  PIPELINE ISOLATION HOLDS — a variance test cannot silently mix two toolchains")
print("=" * 96)
