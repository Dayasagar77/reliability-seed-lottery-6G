#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
q4_eval_unit.py - evaluates the risk-asymmetry arms through the PUBLISHED evaluator.

Same protocol as every arm in the paper: R = 8 common-random-number blocks, 5 evaluation seeds x
150 episodes, block seed BASE + 100000 * repeat + episode index with BASE = 20260801. That machinery
lives in tablevii_r8.py and is used unmodified.

Overridden on the imported module, and nothing else:
    T.ARMS         so the new keys resolve to (label, binary_power, guard, mode, 'branched')
    T.ckpt_for     so a unit reads results/q4_riskasym/q4_<arm>_s<seed>.pt
    T.OUTDIR       so results land in results/q4_riskasym_eval/
    T.import_stack wrapped so load_branched returns the QUANTILE network

The evaluator only ever calls net.act(obs), and BranchedQRNet.act has the published signature, so
the evaluation loop, the metric definitions and the CRN patch are untouched.

USAGE
    python -u q4_eval_unit.py --unit qr_asym_u 0
    python -u q4_eval_unit.py --all --slots 8
    python -u q4_eval_unit.py --unit qr_asym 0 --smoke        # R=2, 6 episodes, scratch outdir
"""
import os, sys, argparse, time, subprocess

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

CKPT_DIR = 'results/q4_riskasym'
OUTDIR_REAL = 'results/q4_riskasym_eval'
OUTDIR_SMOKE = 'results/q4_smoke_eval'


def build(smoke):
    if smoke:
        os.environ['T7_R'] = '2'
        os.environ['T7_EVAL_EP'] = '6'
    saved = sys.argv[:]
    sys.argv = [saved[0]]
    try:
        import tablevii_r8 as T
    finally:
        sys.argv = saved
    import q4_campaign as C
    import q4_qr_branched as Q

    T.ARMS = dict(T.ARMS)
    for arm, (label, tag, guard, bp, _) in C.ARMS.items():
        T.ARMS[arm] = (label, bp, guard, C.GUARD_MODE, 'branched')

    def ckpt_for(arm, seed):
        return os.path.join(CKPT_DIR, 'q4_%s_s%d.pt' % (arm, seed)), 'q4-riskasym-campaign'

    T.ckpt_for = ckpt_for
    T.OUTDIR = OUTDIR_SMOKE if smoke else OUTDIR_REAL
    os.makedirs(T.OUTDIR, exist_ok=True)

    orig_stack = T.import_stack

    def stack():
        S = orig_stack()
        S = dict(S)
        S['load_branched'] = lambda path, obs_dim, device='cpu': Q.load_qr(path, obs_dim, device)
        return S

    T.import_stack = stack
    return T


def run_unit(arm, seed, smoke):
    T = build(smoke)
    import q4_campaign as C
    label, tag, guard, bp, _ = C.ARMS[arm]
    T.log('[q4] %s s%d | %s | risk=%s guard=%s | R=%d | OUTDIR=%s'
          % (arm, seed, label, C.risk_modes(tag), guard, T.R, T.OUTDIR))
    return T.run_unit(arm, seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--unit', nargs=2, metavar=('ARM', 'SEED'))
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--arms', default='')
    ap.add_argument('--seeds', default='0-4')
    ap.add_argument('--slots', type=int, default=8)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()

    if a.unit:
        return run_unit(a.unit[0], int(a.unit[1]), a.smoke)

    import q4_campaign as C
    arms = [x.strip() for x in a.arms.split(',') if x.strip()] or list(C.DEFAULT_ARMS)
    if '-' in a.seeds:
        lo, hi = a.seeds.split('-')
        seeds = list(range(int(lo), int(hi) + 1))
    else:
        seeds = [int(x) for x in a.seeds.split(',')]
    units = [(arm, s) for arm in arms for s in seeds
             if os.path.exists(os.path.join(CKPT_DIR, 'q4_%s_s%d.pt' % (arm, s)))]
    missing = [(arm, s) for arm in arms for s in seeds
               if not os.path.exists(os.path.join(CKPT_DIR, 'q4_%s_s%d.pt' % (arm, s)))]
    print('q4 evaluation | %d units | slots=%d%s'
          % (len(units), a.slots, '  [SMOKE]' if a.smoke else ''))
    if missing:
        print('  MISSING checkpoints, not evaluated:', missing)
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
            name = 'q4eval_%s_s%d' % (arm, seed)
            fh = open(os.path.join('logs', name + '.log'), 'a', encoding='utf-8')
            cmd = [sys.executable, '-u', os.path.abspath(__file__), '--unit', arm, str(seed)]
            if a.smoke:
                cmd.append('--smoke')
            p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=os.getcwd())
            running[name] = (p, time.time(), fh)
            print('START', name, 'pid', p.pid)
        if queue or running:
            time.sleep(10)
    print('evaluation finished in %.1f min | %d done, %d failed'
          % ((time.time() - t0) / 60, done, len(failed)))
    if failed:
        return 1
    print('NEXT: python -u q4_analyse.py')
    return 0


if __name__ == '__main__':
    sys.exit(main())
