#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
q3_smoke.py - prove the 2x2 rig is correct BEFORE spending 15 trainings on it.

Nothing here is a result. Every check is a statement about the machinery, and each one is
designed to fail loudly if the campaign would silently measure the wrong thing.

  1  RB_STEP is read in exactly two places in env_6g_branched.py, both in the spectrum branch,
     and the C5 reward constant does not involve it. This is what makes the rebind minimal.
  2  the binary spectrum actuator really is binary: from any allocation, one S_DOWN lands on
     RB_MIN and one S_UP lands on RB_MAX, and the incremental one still moves by 8
  3  the binary POWER actuator is unchanged and the incremental one still steps by 3 dBm
  4  the guard fires identically in all four cells - the actuator swap must not touch it
  5  the reward is byte-identical across the four cells for the same (state, action) trace
  6  the published arm's action space is untouched when the campaign is not running: importing
     q3_actuator and configuring g_bi leaves RB_STEP at its published value
  7  end to end: a short training of the falsifier arm writes a checkpoint, and the published
     evaluator loads and scores it through the CRN protocol

Usage:  python -u q3_smoke.py [--steps 4000]
"""
import os, sys, re, argparse, shutil, time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

FAIL = []


def ck(cond, msg, detail=''):
    print(('  PASS  ' if cond else '  FAIL  ') + msg + (('   ' + detail) if detail else ''))
    if not cond:
        FAIL.append(msg)
    return cond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--skip-train', action='store_true')
    a = ap.parse_args()

    print('=' * 100)
    print('  q3 SMOKE TEST - the 2x2 assignment rig')
    print('=' * 100)

    import torch
    torch.set_num_threads(1)
    import numpy as np
    print('  torch %s | numpy %s | python %s' % (torch.__version__, np.__version__,
                                                 sys.version.split()[0]))

    # ---------------------------------------------------------------- 1 source inspection
    print('\n1. the rebind is minimal')
    src = open('env_6g_branched.py', encoding='utf-8').read().splitlines()
    uses = [(i + 1, l.strip()) for i, l in enumerate(src)
            if re.search(r'\bRB_STEP\b', l) and not l.strip().startswith('from')]
    ck(len(uses) == 2, 'RB_STEP is read in exactly 2 places (found %d)' % len(uses),
       str([u[0] for u in uses]))
    ck(all('rb_count' in l for _, l in uses), 'both uses are in the spectrum branch')
    c5 = [l for l in src if '_RB_STEP_REW =' in l]
    ck(len(c5) == 1 and 'RB_STEP' not in c5[0].split('=', 1)[1],
       'the C5 reward constant does not involve RB_STEP', c5[0].strip() if c5 else '')

    # ---------------------------------------------------------------- 2/3 actuator semantics
    print('\n2-3. actuator semantics')
    import q3_actuator as A
    import env_6g_branched as E
    from env_6g_v5 import RB_MIN, RB_MAX, PT_MIN_DBM, PT_MAX_DBM, PT_STEP_DBM
    from dndqn_branched import R_STAY_PROTECT, R_STAY_SAVE, S_HOLD, S_DOWN, S_UP
    import train_phase2 as tp
    slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH)
    pool = slow + med
    traj = pool[0]

    def probe(binary_power, binary_spectrum, start_rb, start_pt, a_rel, a_spec):
        A.set_spectrum(binary_spectrum)
        env = E.BranchedEnv6G(traj, binary_power=binary_power,
                              guard_db=A.GUARD_DB, guard_mode=A.GUARD_MODE)
        env.reset(seed=12345)
        env.rb_count, env.pt_dbm = start_rb, start_pt
        env._prev_rsrp = -60.0                       # well above the guard line: guard idle
        env.step((a_rel, a_spec))
        return env.rb_count, env.pt_dbm

    rb, _ = probe(True, True, 34, 30.0, R_STAY_PROTECT, S_DOWN)
    ck(rb == RB_MIN, 'binary spectrum: S_DOWN from 34 lands on RB_MIN=%d' % RB_MIN, 'got %d' % rb)
    rb, _ = probe(True, True, 34, 30.0, R_STAY_PROTECT, S_UP)
    ck(rb == RB_MAX, 'binary spectrum: S_UP from 34 lands on RB_MAX=%d' % RB_MAX, 'got %d' % rb)
    rb, _ = probe(True, False, 34, 30.0, R_STAY_PROTECT, S_DOWN)
    ck(rb == 26, 'incremental spectrum: S_DOWN from 34 moves to 26', 'got %d' % rb)
    rb, _ = probe(True, False, 34, 30.0, R_STAY_PROTECT, S_UP)
    ck(rb == 42, 'incremental spectrum: S_UP from 34 moves to 42', 'got %d' % rb)

    _, pt = probe(True, False, 34, 27.0, R_STAY_SAVE, S_HOLD)
    ck(pt == PT_MIN_DBM, 'binary power: SAVE from 27 lands on PT_MIN=%g' % PT_MIN_DBM, 'got %g' % pt)
    _, pt = probe(False, False, 34, 27.0, R_STAY_SAVE, S_HOLD)
    ck(pt == 27.0 - PT_STEP_DBM, 'incremental power: SAVE from 27 steps to %g'
       % (27.0 - PT_STEP_DBM), 'got %g' % pt)
    _, pt = probe(False, True, 34, 27.0, R_STAY_PROTECT, S_HOLD)
    ck(pt == 27.0 + PT_STEP_DBM, 'incremental power is unaffected by the spectrum rebind',
       'got %g' % pt)

    # ---------------------------------------------------------------- 4 the guard is untouched
    print('\n4. the guard fires identically in all four cells')
    fired = {}
    for arm in A.ARM_ORDER:
        label, bp, bs, _ = A.ARMS[arm]
        A.set_spectrum(bs)
        env = E.BranchedEnv6G(traj, binary_power=bp, guard_db=A.GUARD_DB,
                              guard_mode=A.GUARD_MODE)
        env.reset(seed=999)
        env.pt_dbm, env.rb_count = PT_MIN_DBM, 34
        env._prev_rsrp = -80.0                       # below the -76 dBm save line
        env.step((R_STAY_SAVE, S_HOLD))
        fired[arm] = env.pt_dbm
    ck(all(v == PT_MAX_DBM for v in fired.values()),
       'guard restores PT_MAX below the save line in every cell', str(fired))
    notfired = {}
    for arm in A.ARM_ORDER:
        label, bp, bs, _ = A.ARMS[arm]
        A.set_spectrum(bs)
        env = E.BranchedEnv6G(traj, binary_power=bp, guard_db=A.GUARD_DB,
                              guard_mode=A.GUARD_MODE)
        env.reset(seed=999)
        env.pt_dbm, env.rb_count = PT_MIN_DBM, 34
        env._prev_rsrp = -70.0                       # above the save line
        env.step((R_STAY_SAVE, S_HOLD))
        notfired[arm] = env.pt_dbm
    ck(all(v == PT_MIN_DBM for v in notfired.values()),
       'guard stays idle above the save line in every cell', str(notfired))

    # ---------------------------------------------------------------- 5 reward map identity
    print('\n5. the reward map is identical across the four cells')
    # NOTE. Two different claims must not be confused. Total episode reward WILL differ between
    # cells, because a binary actuator lands the agent in different states than an incremental one
    # does - that divergence IS the treatment. What must be identical is the reward FUNCTION: the
    # constants and the mapping from outcome to reward. Those are module-level in env_6g_branched
    # and the rebind must not touch any of them.
    CONSTS = ['W_MOB_REL', 'W_PWR_REL', 'W_SPEC_REL', 'SAFE_RSRP_THRESH', 'PWR_DOWN_UNSAFE_PEN',
              '_PWR_STEP_REW', '_RB_STEP_REW', 'RLF_TERMINAL_PENALTY']
    A.set_spectrum(False)
    before = {c: getattr(E, c) for c in CONSTS}
    A.set_spectrum(True)
    after_bin = {c: getattr(E, c) for c in CONSTS}
    A.set_spectrum(False)
    after_inc = {c: getattr(E, c) for c in CONSTS}
    ck(before == after_bin == after_inc,
       'every reward constant survives the rebind unchanged',
       ' '.join('%s=%s' % (c, before[c]) for c in CONSTS))
    want_c5 = E.W_SPEC_REL * (PT_STEP_DBM / (PT_MAX_DBM - PT_MIN_DBM)) * 10.0
    ck(abs(E._RB_STEP_REW - want_c5) < 1e-12,
       'C5 constant equals its published formula, which does not involve RB_STEP',
       '%.9f vs %.9f' % (E._RB_STEP_REW, want_c5))
    # and for an IDENTICAL resulting state the reward is identical in every cell
    got = {}
    for arm in A.ARM_ORDER:
        label, bp, bs, _ = A.ARMS[arm]
        A.set_spectrum(bs)
        env = E.BranchedEnv6G(traj, binary_power=bp, guard_db=A.GUARD_DB,
                              guard_mode=A.GUARD_MODE)
        env.reset(seed=7)
        env.pt_dbm, env.rb_count = 30.0, RB_MAX
        env._prev_rsrp = -70.0
        o, r, t, tr, info = env.step((R_STAY_PROTECT, S_HOLD))   # HOLD: no actuator involved
        got[arm] = (round(r, 9), env.pt_dbm, env.rb_count)
    ck(len(set(got.values())) == 1,
       'a HOLD step from an identical state gives an identical reward in every cell', str(got))

    # ---------------------------------------------------------------- 6 published arm untouched
    print('\n6. the published arm is untouched')
    A.configure('g_bi')
    ck(E.RB_STEP == A.RB_STEP_PUBLISHED,
       'configuring g_bi leaves RB_STEP at the published %d' % A.RB_STEP_PUBLISHED,
       'got %d' % E.RB_STEP)
    import hashlib
    for f in ('env_6g_branched.py', 'env_6g_v5.py', 'train_branched_nyu.py', 'dndqn_branched.py',
              'tablevii_r8.py'):
        h = hashlib.md5(open(f, 'rb').read()).hexdigest()
        print('  note  %-24s md5 %s' % (f, h))

    # ---------------------------------------------------------------- 7 end to end
    if not a.skip_train:
        print('\n7. end to end on the falsifier arm (short training, then the published evaluator)')
        os.makedirs('results/q3_assignment', exist_ok=True)
        ck_path = 'results/q3_assignment/q3_g_ib_s0.pt'
        keep = None
        if os.path.exists(ck_path):
            keep = ck_path + '.smokebak'
            shutil.move(ck_path, keep)
            print('  note  moved an existing %s aside to %s' % (ck_path, os.path.basename(keep)))
        t0 = time.time()
        rc = os.system('"%s" -u q3_campaign.py --unit g_ib 0 --steps %d' % (sys.executable, a.steps))
        ck(rc == 0 and os.path.exists(ck_path),
           'short training wrote a checkpoint (%d steps, %.1f min)' % (a.steps, (time.time() - t0) / 60))
        if os.path.exists(ck_path):
            rc = os.system('"%s" -u q3_eval_unit.py --unit g_ib 0 --smoke' % sys.executable)
            unit = 'results/q3_smoke_eval/unit_g_ib_s0_laptop.csv'
            alt = [p for p in os.listdir('results/q3_smoke_eval')] if os.path.isdir('results/q3_smoke_eval') else []
            ck(rc == 0 and (os.path.exists(unit) or any(x.startswith('unit_g_ib_s0') for x in alt)),
               'published evaluator scored it through the CRN protocol', str(alt))
        os.remove(ck_path) if os.path.exists(ck_path) else None
        if keep:
            shutil.move(keep, ck_path)
            print('  note  restored the original checkpoint')
        print('  note  the smoke checkpoint has been deleted; results/q3_smoke_eval/ is scratch')

    # ---------------------------------------------------------------- 8 live inside the evaluator
    print('\n8. the actuator is live inside the PUBLISHED evaluator, not just in step()')
    saved_argv = sys.argv[:]
    sys.argv = [saved_argv[0]]
    try:
        import tablevii_r8 as T
    finally:
        sys.argv = saved_argv

    class AlwaysDown:
        """parameter-free: press SAVE-side spectrum every step, never move power"""
        def act(self, obs):
            return [R_STAY_PROTECT, S_DOWN]

    St = T.import_stack()
    T.install_patch(St['env_6g_v5'])
    B = St['B']
    keep_ep = B.EVAL_EP
    B.EVAL_EP = 6
    spec = {}
    for binary in (False, True):
        A.set_spectrum(binary)
        A.assert_spectrum(binary)
        with T.seeded(0):
            m = B.evaluate_branched(AlwaysDown(), pool, True,
                                    guard_db=A.GUARD_DB, guard_mode=A.GUARD_MODE)
        spec['binary' if binary else 'incremental'] = round(float(m['Spec']), 3)
    B.EVAL_EP = keep_ep
    ck(spec['binary'] > spec['incremental'] + 5.0 and spec['binary'] > 90.0,
       'always-S_DOWN returns the whole allocation in one press under the binary actuator',
       'binary %.2f%% vs incremental %.2f%%' % (spec['binary'], spec['incremental']))

    print('\n' + '=' * 100)
    print('  SMOKE: ALL PASSED' if not FAIL else '  SMOKE: %d FAILURE(S)' % len(FAIL))
    for f in FAIL:
        print('   -', f)
    print('=' * 100)
    if not FAIL:
        print('  the rig is sound. To run the campaign:')
        print('      python -u q3_campaign.py --plan')
        print('      python -u q3_campaign.py --launch --slots 8')
        print('      python -u q3_eval_unit.py --all --slots 8 --use-published-gbi')
        print('      python -u q3_analyse.py')
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
