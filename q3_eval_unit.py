#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
q3_eval_unit.py - evaluates the 2x2 assignment arms through the PUBLISHED evaluator.

The pre-registration requires that every arm meets the same channel realisations as the published
arms: R = 8 common-random-number blocks, 5 evaluation seeds x 150 episodes, block seed
BASE + 100000 * repeat + episode index with BASE = 20260801. All of that lives in tablevii_r8.py
and is used here without modification.

What this script overrides on the imported module, and nothing else:
    T.ARMS      - so the new arm keys resolve to (label, binary_power, guard, mode, 'branched')
    T.ckpt_for  - so a unit reads results/q3_assignment/q3_<arm>_s<seed>.pt
    T.OUTDIR    - so results land in results/q3_assignment_eval/ and cannot overwrite Table VII

and, through q3_actuator, env_6g_branched.RB_STEP, which is the spectrum actuator. The evaluation
loop, the CRN patch, the eval seeds, the episode count and the five metric definitions are the
published objects, called unmodified.

The published arm g_bi is evaluated from its own published checkpoints (br_gst_s{seed}.pt), so the
2x2 is compared on one protocol without retraining or re-evaluating anything the manuscript
already reports - the existing results/tablevii_r8/unit_guarded_s*.csv files are the same
measurement and may be used instead; --use-published-gbi does exactly that.

USAGE
    python -u q3_eval_unit.py --unit g_ib 0
    python -u q3_eval_unit.py --all --slots 8
    python -u q3_eval_unit.py --all --slots 8 --smoke        # R=2, 6 episodes, tagged outputs
"""
import os, sys, argparse, time, subprocess

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

SEEDS = [0, 1, 2, 3, 4]
CKPT_DIR = 'results/q3_assignment'
OUTDIR_REAL = 'results/q3_assignment_eval'
OUTDIR_SMOKE = 'results/q3_smoke_eval'


def build(smoke):
    """Import tablevii_r8 with argv neutralised and patch the three module globals."""
    if smoke:
        os.environ['T7_R'] = '2'
        os.environ['T7_EVAL_EP'] = '6'
    saved = sys.argv[:]
    sys.argv = [saved[0]]
    try:
        import tablevii_r8 as T
    finally:
        sys.argv = saved
    import q3_actuator as A

    T.ARMS = dict(T.ARMS)
    for arm in A.ARM_ORDER:
        label, bp, bs, _ = A.ARMS[arm]
        T.ARMS[arm] = (label, bp, A.GUARD_DB, A.GUARD_MODE, 'branched')

    def ckpt_for(arm, seed):
        if arm == 'g_bi':
            return 'br_gst_s%d.pt' % seed, 'published-20260712'
        return os.path.join(CKPT_DIR, 'q3_%s_s%d.pt' % (arm, seed)), 'q3-assignment-campaign'

    T.ckpt_for = ckpt_for
    T.OUTDIR = OUTDIR_SMOKE if smoke else OUTDIR_REAL
    os.makedirs(T.OUTDIR, exist_ok=True)
    return T, A


def run_unit(arm, seed, smoke):
    T, A = build(smoke)
    label, bp, bs = A.configure(arm)          # rebinds env_6g_branched.RB_STEP
    A.assert_spectrum(bs)
    T.log('[q3] %s s%d | %s | R=%d | OUTDIR=%s' % (arm, seed, A.describe(arm), T.R, T.OUTDIR))
    rc = T.run_unit(arm, seed)
    A.assert_spectrum(bs)                     # evaluation must not have reset it
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--unit', nargs=2, metavar=('ARM', 'SEED'))
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--arms', default='')
    ap.add_argument('--seeds', default='0-4')
    ap.add_argument('--slots', type=int, default=8)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--use-published-gbi', action='store_true',
                    help='skip g_bi and read results/tablevii_r8/unit_guarded_s*.csv in the '
                         'analysis instead; identical measurement, zero new compute')
    a = ap.parse_args()

    if a.unit:
        return run_unit(a.unit[0], int(a.unit[1]), a.smoke)

    import q3_actuator as A
    arms = [x.strip() for x in a.arms.split(',') if x.strip()] or list(A.ARM_ORDER)
    if a.use_published_gbi and 'g_bi' in arms:
        arms.remove('g_bi')
    if '-' in a.seeds:
        lo, hi = a.seeds.split('-')
        seeds = list(range(int(lo), int(hi) + 1))
    else:
        seeds = [int(x) for x in a.seeds.split(',')]
    units = [(arm, s) for arm in arms for s in seeds]

    outdir = OUTDIR_SMOKE if a.smoke else OUTDIR_REAL
    print('q3 evaluation | %d units | slots=%d | outdir=%s%s'
          % (len(units), a.slots, outdir, '  [SMOKE: R=2, 6 episodes]' if a.smoke else ''))
    if not a.all:
        for u in units:
            print('   would run', u)
        print('add --all to launch')
        return 0

    env = dict(os.environ)
    env['OMP_NUM_THREADS'] = env['MKL_NUM_THREADS'] = '1'
    env['PYTHONPATH'] = os.getcwd() + os.pathsep + env.get('PYTHONPATH', '')
    os.makedirs('logs', exist_ok=True)
    queue, running, failed, done = list(units), {}, [], 0
    t0 = time.time()
    while queue or running:
        for name in list(running):
            p, started, fh = running[name]
            if p.poll() is None:
                continue
            fh.close()
            if p.returncode == 0:
                done += 1
                print('DONE  %s (%d/%d) %.1fm' % (name, done, len(units), (time.time() - started) / 60))
            else:
                failed.append(name)
                print('FAIL  %s rc=%s - see logs/%s.log' % (name, p.returncode, name))
            del running[name]
        while queue and len(running) < a.slots:
            arm, seed = queue.pop(0)
            name = 'q3eval_%s_s%d' % (arm, seed)
            fh = open(os.path.join('logs', name + '.log'), 'a', encoding='utf-8')
            cmd = [sys.executable, '-u', os.path.abspath(__file__), '--unit', arm, str(seed)]
            if a.smoke:
                cmd.append('--smoke')
            p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=os.getcwd())
            running[name] = (p, time.time(), fh)
            print('START', name, 'pid', p.pid)
        if queue or running:
            time.sleep(10)
    print('evaluation finished in %.1f min | %d done, %d failed' % ((time.time() - t0) / 60, done, len(failed)))
    if failed:
        return 1
    print('NEXT: python -u q3_analyse.py')
    return 0


if __name__ == '__main__':
    sys.exit(main())
