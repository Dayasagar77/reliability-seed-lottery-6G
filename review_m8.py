#!/usr/bin/env python3
"""
review_m8.py -- is the re-derived blockage pool's blockage picture the same as the original
pool's would have been?

The reviewer's point: the blockage stress test of Section IX-I rebuilds its trajectory pool from
the raw HighD recordings because the original preprocessing script was not preserved, and the
rebuilt pool is LARGER (5,048 against 3,251). If the extra trajectories carry a different amount
of blockage, the stress test is run on a friendlier or harsher population than the paper's other
results and the comparison is not clean.

WHAT ACTUALLY DIFFERS. Both pipelines apply the SAME length filter -- train_phase2.load_trajectories
skips len(traj) < HO_MIN_LEAD + 10, rederive_blockage skips len(tg) < 35, and HO_MIN_LEAD = 25.
What the re-derivation does NOT do is the SPEED BANDING: the working pool of every other result
is slow + med only, with the fast band held out as the generalization set of Section IX-J. So the
"looser filter" is the absence of the speed band, not a different length rule.

THE TEST. Recompute the blockage statistics on the subset of the re-derived pool that WOULD have
passed the original speed filter, and compare against the full re-derived pool. If they agree,
the extra trajectories did not change the blockage picture and the stress test is comparable.
Read-only.
"""
import csv
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, '/home/claude/work/sim')
VEL_SLOW_MAX = None
VEL_MED_MAX = None
exec(open('/home/claude/work/sim/train_phase2.py').read().split('def ')[0].split('class ')[0]
     .replace('import ', '#import '), globals()) if False else None
import re
src = open('/home/claude/work/sim/train_phase2.py').read()
for name in ('VEL_SLOW_MAX', 'VEL_MED_MAX', 'HO_MIN_LEAD'):
    m = re.search(r'^%s\s*=\s*([0-9.]+)' % name, src, re.M)
    globals()[name] = float(m.group(1)) if m else None
    print('  %-14s = %s' % (name, globals()[name]))

rows = []
for rec in (35, 40, 50):
    with open('/home/claude/work/sim/rederived_blockage_%d.csv' % rec) as f:
        for r in csv.DictReader(f):
            rows.append((int(r['Recording']), int(r['Car_ID']), float(r['v_abs']),
                         int(r['blocked'])))
print('\n  rows read: %s' % format(len(rows), ','))

traj = defaultdict(list)
for rec, cid, v, b in rows:
    traj[(rec, cid)].append((v, b))

BAR = '=' * 96
print('\n' + BAR)
print('m8 -- BLOCKAGE STATISTICS, FULL RE-DERIVED POOL vs THE ORIGINAL-FILTER SUBSET')
print(BAR)


def stats(keys, name):
    n = len(keys)
    steps = sum(len(traj[k]) for k in keys)
    blocked = sum(b for k in keys for _, b in traj[k])
    with_ev = sum(1 for k in keys if any(b for _, b in traj[k]))
    per = [100.0 * sum(b for _, b in traj[k]) / len(traj[k]) for k in keys]
    print('  %-34s trajectories %5d | steps %8s | steps blocked %6.2f%% | '
          'trajectories with >=1 event %5.1f%%'
          % (name, n, format(steps, ','), 100.0 * blocked / steps if steps else 0,
             100.0 * with_ev / n if n else 0))
    return dict(n=n, steps=steps, pct_steps=100.0 * blocked / steps if steps else 0,
                pct_traj=100.0 * with_ev / n if n else 0, per=np.array(per))


allk = sorted(traj)
full = stats(allk, 'FULL re-derived pool')

speed = {k: float(np.mean([v for v, _ in traj[k]])) for k in allk}
sub = [k for k in allk if speed[k] <= VEL_MED_MAX]
subset = stats(sub, 'original-filter subset (slow+med)')
fast = [k for k in allk if speed[k] > VEL_MED_MAX]
fk = stats(fast, 'excluded fast band, for contrast')

print('\n  DIFFERENCE the looser filter makes:')
print('    steps blocked            : %.2f%% -> %.2f%%   (delta %+.2f pp)'
      % (subset['pct_steps'], full['pct_steps'], full['pct_steps'] - subset['pct_steps']))
print('    trajectories with event  : %.1f%%  -> %.1f%%    (delta %+.1f pp)'
      % (subset['pct_traj'], full['pct_traj'], full['pct_traj'] - subset['pct_traj']))
from scipy import stats as st
t, p = st.ttest_ind(subset['per'], fk['per'], equal_var=False)
print('    per-trajectory blocked fraction, slow+med vs fast: %.2f%% vs %.2f%%, '
      'Welch t = %.2f, p = %.4f' % (subset['per'].mean(), fk['per'].mean(), t, p))
print('\n  paper reports 10-12%% of steps blocked and 54-59%% of trajectories with >=1 event.')
print(BAR)

# ---------------------------------------------------------------- per-recording
print('\n  PER-RECORDING (the paper quotes ranges, so these should be the endpoints):')
for rec in (35, 40, 50):
    ks = [k for k in sub if k[0] == rec]
    steps = sum(len(traj[k]) for k in ks)
    blocked = sum(b for k in ks for _, b in traj[k])
    withev = sum(1 for k in ks if any(b for _, b in traj[k]))
    print('    rec %d : %4d trajectories | %7s steps | %5.2f%% steps blocked | '
          '%5.1f%% trajectories with >=1 event'
          % (rec, len(ks), format(steps, ','), 100.0 * blocked / steps, 100.0 * withev / len(ks)))
print('\n  reconstruction check: the speed-filtered subset has %d trajectories; the manuscript '
      'reports 5,048.' % len(sub))
print('  %s' % ('MATCH — the reported pool is already the slow+med subset, so the speed filter '
                'is NOT looser' if len(sub) == 5048 else 'NO MATCH — reconsider the filter'))
