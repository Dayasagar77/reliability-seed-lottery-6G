"""q2_eval_fix.py -- re-evaluate the Q2 checkpoints in the THREE-LEVEL environment.

THE BUG. q2_campaign.evaluate() called train_branched_nyu.evaluate_branched(). That function looks
up BranchedEnv6G in ITS OWN module globals -- the published 3-action class -- so a 4-action net was
evaluated in a 3-action world. In the published step(), a_rel == 3 matches neither the PROTECT nor
the SAVE branch and is not R_HANDOVER, so it silently becomes "hold power, hold cell": the MID
press does nothing at all. Seed 1, which presses MID 62.3% of the time and SAVE never, therefore
recorded 0.00% power saving -- pt_dbm simply never left PT_MAX. Every metric in
results/q2_3level/q2_3level.csv is measured in the wrong environment.

The action-intent probe is NOT affected: it constructs the three-level env explicitly.

THE FIX. Rebuild evaluate_branched as a function object whose globals carry the three-level class,
leaving the published module untouched. Same code, same CRN harness, one name rebound.

THE GATE. A fix you cannot see working is not a fix. This proves, on a real checkpoint, that
  (a) the published evaluator NEVER produces pt_dbm = 26.5, and
  (b) the corrected evaluator DOES,
before any number is rewritten. The old CSV is moved aside, not overwritten.
"""
import csv
import hashlib
import os
import sys
import time
import types

R = 8
SEEDS = [0, 1, 2, 3, 4]
GUARD_DB, GUARD_MODE = 9.0, 'state'
METRICS = ['HO', 'Power', 'Spec', 'RLF', 'Tput']
OUT = os.path.join('results', 'q2_3level')
VENUE = 'laptop'
PT_MID = 26.5


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def log(m):
    print(time.strftime('%H:%M:%S') + ' | ' + m, flush=True)


def main():
    sys.argv = sys.argv[:1]
    import torch
    torch.set_num_threads(1)
    import numpy as np
    import env_6g_v5
    import train_phase2 as tp
    import train_branched_nyu as B
    import train_branched_3l as B3
    from dndqn_branched import load_branched
    from dndqn import OBS_DIM
    import tablevii_r8 as T7
    T7.install_patch(env_6g_v5)

    NEW = B3.BranchedEnv6G

    # ---- corrected evaluator: same code object, one global rebound -------------------------
    g = dict(B.evaluate_branched.__globals__)
    g['BranchedEnv6G'] = NEW
    eval_3l = types.FunctionType(B.evaluate_branched.__code__, g,
                                 'evaluate_branched_3l',
                                 B.evaluate_branched.__defaults__,
                                 B.evaluate_branched.__closure__)

    # ---- instrumentation: what power levels actually occur during an evaluation ------------
    seen = {'pub': set(), 'new': set()}
    for tag, cls in (('pub', B.BranchedEnv6G), ('new', NEW)):
        orig = cls.step
        if getattr(cls, '_probe_installed', None) != tag:
            def mk(o, t):
                def s(self, action):
                    out = o(self, action)
                    seen[t].add(round(float(out[4]['pt_dbm']), 2))
                    return out
                return s
            cls.step = mk(orig, tag)
            cls._probe_installed = tag

    slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH)
    pool = slow + med
    net = load_branched('q2_3l_s1.pt', OBS_DIM)[0]      # s1: 62.3% MID, the clearest case

    print('=' * 92)
    print('GATE -- does the corrected evaluator actually reach the 26.5 dBm level?')
    print('=' * 92)
    with T7.seeded(0):
        m_pub = B.evaluate_branched(net, pool, True, guard_db=GUARD_DB, guard_mode=GUARD_MODE)
    with T7.seeded(0):
        m_new = eval_3l(net, pool, True, guard_db=GUARD_DB, guard_mode=GUARD_MODE)
    print('  published evaluator, pt_dbm levels seen : %s' % sorted(seen['pub']))
    print('  corrected evaluator, pt_dbm levels seen : %s' % sorted(seen['new']))
    ok = True
    if PT_MID in seen['pub']:
        print('  FAIL: the published evaluator reached %.1f dBm -- the diagnosis is wrong.' % PT_MID)
        ok = False
    else:
        print('  [ok] the published evaluator never reaches %.1f dBm  <-- this is the bug' % PT_MID)
    if PT_MID not in seen['new']:
        print('  FAIL: the corrected evaluator still never reaches %.1f dBm.' % PT_MID)
        ok = False
    else:
        print('  [ok] the corrected evaluator does reach %.1f dBm  <-- the fix is live' % PT_MID)
    print('  s1 Power saving: published %.2f%%  ->  corrected %.2f%%'
          % (m_pub['Power'], m_new['Power']))
    print('  s1 RLF         : published %.2f%%  ->  corrected %.2f%%' % (m_pub['RLF'], m_new['RLF']))
    if abs(m_pub['Power'] - m_new['Power']) < 1e-9:
        print('  FAIL: the two evaluators agree exactly -- the rebind did not take.')
        ok = False
    else:
        print('  [ok] the two evaluators disagree, as they must')
    print('=' * 92)
    if not ok:
        print('  VERDICT: FAIL -- not rewriting anything.')
        return 2
    print('  VERDICT: PASS -- re-evaluating all five checkpoints in the three-level environment')
    print('=' * 92)

    # ---- quarantine the wrong file, then re-measure ----------------------------------------
    bad = os.path.join(OUT, 'q2_3level.csv')
    if os.path.exists(bad):
        q = os.path.join(OUT, 'QUARANTINE_q2_3level_MEASURED_IN_PUBLISHED_3ACTION_ENV.csv')
        os.replace(bad, q)
        log('quarantined the wrong grid -> %s' % q)

    rows = []
    for s in SEEDS:
        ckp = 'q2_3l_s%d.pt' % s
        net = load_branched(ckp, OBS_DIM)[0]
        for r in range(R):
            with T7.seeded(r):
                m = eval_3l(net, pool, True, guard_db=GUARD_DB, guard_mode=GUARD_MODE)
            row = {'seed': s, 'ckpt': ckp, 'ckpt_md5': md5(ckp), 'repeat': r,
                   'chan_seed_first': T7.BASE + 100000 * r, 'steps': 800000,
                   'branch_sizes': '4x3', 'guard_db': GUARD_DB, 'guard_mode': GUARD_MODE,
                   'eval_env': 'three-level', 'venue': VENUE, 'torch': torch.__version__,
                   'numpy': np.__version__,
                   'measured_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
            for k in METRICS:
                row[k] = '%.4f' % float(m[k])
            rows.append(row)
        log('  s%d re-evaluated' % s)
    with open(bad, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    log('wrote %s (%d cells, eval_env=three-level)' % (bad, len(rows)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
