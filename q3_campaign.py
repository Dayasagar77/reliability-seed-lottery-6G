#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
q3_campaign.py - trains the three new cells of the pre-registered 2x2 assignment ablation.

    g_ii   incremental power + incremental spectrum
    g_bb   binary power      + binary spectrum
    g_ib   incremental power + binary spectrum        <- H2, the falsifier

The fourth cell, g_bi (binary power + incremental spectrum), is the published proposed arm. It is
NOT retrained: its five checkpoints and its R = 8 evaluation already exist, and retraining it
would replace a published number with a fresh draw of the very lottery this paper measures.

PROTOCOL, fixed by the pre-registration of 08 Aug 2026 and not adjustable from the command line
except for --steps, which exists only so the smoke test can run a short one:
    5 independent trainings per arm, seeds 0-4, random initialisation
    8 x 10^5 steps, the published budget
    identical reward W = (0.80, 0.10, 0.10), RLF terminal penalty -300
    identical channel (NYU 142 GHz CI) and HighD slow+medium pool
    guard held constant at G = 9 dB, save line -76 dBm, mode 'state', in every cell
    best-checkpoint selection, the published rule

PROVENANCE. No published source file is edited. train_branched_nyu.train() is imported and called
unmodified; the only thing this driver changes is env_6g_branched.RB_STEP, via q3_actuator, which
is the spectrum actuator and nothing else (see that module's docstring, and q3_smoke.py).

USAGE (PowerShell, from the folder holding the drivers)
    python -u q3_campaign.py --plan                     # print what would run, launch nothing
    python -u q3_campaign.py --launch --slots 8         # the real campaign, 15 trainings
    python -u q3_campaign.py --unit g_ib 0              # one training, in this process
    python -u q3_campaign.py --launch --arms g_ib       # one arm only
Resume: rerun the same command. A unit is done when its .pt and its manifest row exist.
"""
import os, sys, csv, time, hashlib, argparse, subprocess

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

OUTDIR = 'results/q3_assignment'
MANIFEST = os.path.join(OUTDIR, 'q3_manifest.csv')
LOGDIR = 'logs'
SEEDS = [0, 1, 2, 3, 4]
STEPS_REAL = 800_000


def say(msg, logname='q3_campaign'):
    line = '%s | %s' % (time.strftime('%Y-%m-%d %H:%M:%S'), msg)
    print(line, flush=True)
    try:
        os.makedirs(LOGDIR, exist_ok=True)
        with open(os.path.join(LOGDIR, logname + '.log'), 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    except OSError:
        pass


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def ckpt_path(arm, seed):
    return os.path.join(OUTDIR, 'q3_%s_s%d.pt' % (arm, seed))


def unit_done(arm, seed):
    return os.path.exists(ckpt_path(arm, seed)) and os.path.getsize(ckpt_path(arm, seed)) > 0


# ---------------------------------------------------------------- one training
def run_unit(arm, seed, steps):
    import torch
    torch.set_num_threads(1)
    import numpy as np
    import q3_actuator as A

    label, bp, bs = A.configure(arm)          # rebinds env_6g_branched.RB_STEP
    A.assert_spectrum(bs)

    saved = sys.argv[:]
    sys.argv = ['train_branched_nyu.py', '1', str(steps)]
    try:
        import train_branched_nyu as B
    finally:
        sys.argv = saved
    import train_phase2 as tp

    A.assert_spectrum(bs)                     # importing must not have reset it
    os.makedirs(OUTDIR, exist_ok=True)
    out = ckpt_path(arm, seed)
    slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH)

    say('train START %s s%d | %s | steps=%s | pool=%d'
        % (arm, seed, A.describe(arm), '{:,}'.format(steps), len(slow) + len(med)),
        'q3_%s_s%d' % (arm, seed))
    t0 = time.time()
    B.train(seed, bp, out, slow, med, steps=steps, guard_db=A.GUARD_DB, guard_mode=A.GUARD_MODE)
    mins = (time.time() - t0) / 60.0
    A.assert_spectrum(bs)                     # and training must not have reset it either

    if not os.path.exists(out):
        say('train FAILED %s s%d - no checkpoint written' % (arm, seed), 'q3_%s_s%d' % (arm, seed))
        return 1

    new = not os.path.exists(MANIFEST)
    with open(MANIFEST, 'a', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(['arm', 'label', 'seed', 'steps', 'binary_power', 'binary_spectrum',
                        'rb_step', 'guard_db', 'guard_mode', 'ckpt', 'md5', 'train_min', 'utc',
                        'torch', 'numpy'])
        w.writerow([arm, label, seed, steps, bp, bs,
                    A.RB_STEP_BINARY if bs else A.RB_STEP_PUBLISHED, A.GUARD_DB, A.GUARD_MODE,
                    os.path.basename(out), md5(out), '%.1f' % mins,
                    time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    torch.__version__, np.__version__])
    say('train DONE  %s s%d in %.1fm -> %s' % (arm, seed, mins, os.path.basename(out)),
        'q3_%s_s%d' % (arm, seed))
    return 0


# ---------------------------------------------------------------- launcher
def launch(units, slots, steps, floor_mb):
    env = dict(os.environ)
    env['OMP_NUM_THREADS'] = env['MKL_NUM_THREADS'] = '1'
    env['PYTHONPATH'] = os.getcwd() + os.pathsep + env.get('PYTHONPATH', '')
    os.makedirs(LOGDIR, exist_ok=True)
    queue, running, failed, done = list(units), {}, [], 0
    total, t0 = len(queue), time.time()
    while queue or running:
        for name in list(running):
            p, started, fh, (arm, seed) = running[name]
            if p.poll() is None:
                continue
            fh.close()
            mins = (time.time() - started) / 60
            if p.returncode == 0 and unit_done(arm, seed):
                done += 1
                say('DONE  %s in %.1fm (%d/%d)' % (name, mins, done, total))
            else:
                failed.append((arm, seed))
                say('FAIL  %s after %.1fm rc=%s - see logs/%s.log' % (name, mins, p.returncode, name))
            del running[name]
        while queue and len(running) < slots:
            free = free_mb()
            if free < floor_mb:
                say('HOLD  %d MB free < floor %d MB (%d running, %d queued)'
                    % (free, floor_mb, len(running), len(queue)))
                break
            arm, seed = queue.pop(0)
            name = 'q3_%s_s%d' % (arm, seed)
            fh = open(os.path.join(LOGDIR, name + '.log'), 'a', encoding='utf-8')
            cmd = [sys.executable, '-u', os.path.abspath(__file__), '--unit', arm, str(seed),
                   '--steps', str(steps)]
            p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=os.getcwd())
            running[name] = (p, time.time(), fh, (arm, seed))
            say('START %s pid %d (%d MB free)' % (name, p.pid, free))
        status(done, total, running, t0)
        if queue or running:
            time.sleep(20)
    say('campaign finished in %.1f min | %d done, %d failed' % ((time.time() - t0) / 60, done, len(failed)))
    return failed


def free_mb():
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes

        class MS(ctypes.Structure):
            _fields_ = [('dwLength', wintypes.DWORD), ('dwMemoryLoad', wintypes.DWORD),
                        ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                        ('ullTotalPageFile', ctypes.c_ulonglong),
                        ('ullAvailPageFile', ctypes.c_ulonglong),
                        ('ullTotalVirtual', ctypes.c_ulonglong),
                        ('ullAvailVirtual', ctypes.c_ulonglong),
                        ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
        st = MS()
        st.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        return st.ullAvailPhys // (1024 * 1024)
    try:
        for line in open('/proc/meminfo'):
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 1 << 20


def status(done, total, running, t0):
    txt = ('%s phase=train done=%d/%d running=%d free_mb=%d elapsed_min=%.1f'
           % (time.strftime('%Y-%m-%d %H:%M:%S'), done, total, len(running), free_mb(),
              (time.time() - t0) / 60))
    for name, rec in running.items():
        txt += '\n  %s pid %d %.1fm' % (name, rec[0].pid, (time.time() - rec[1]) / 60)
    try:
        with open('q3_campaign.status', 'w', encoding='utf-8') as f:
            f.write(txt + '\n')
    except OSError:
        pass


def main():
    import q3_actuator as A
    ap = argparse.ArgumentParser()
    ap.add_argument('--unit', nargs=2, metavar=('ARM', 'SEED'))
    ap.add_argument('--launch', action='store_true')
    ap.add_argument('--plan', action='store_true')
    ap.add_argument('--arms', default='')
    ap.add_argument('--seeds', default='0-4')
    ap.add_argument('--slots', type=int, default=8)
    ap.add_argument('--floor-mb', type=int, default=1500)
    ap.add_argument('--steps', type=int, default=STEPS_REAL)
    a = ap.parse_args()

    if a.steps != STEPS_REAL:
        say('*** SHORT RUN *** steps=%s is not the published budget %s - this is a rehearsal, '
            'not a result' % ('{:,}'.format(a.steps), '{:,}'.format(STEPS_REAL)))

    if a.unit:
        return run_unit(a.unit[0], int(a.unit[1]), a.steps)

    arms = A.TRAIN_ARMS
    if a.arms:
        want = [x.strip() for x in a.arms.split(',') if x.strip()]
        bad = [x for x in want if x not in A.ARMS]
        if bad:
            say('unknown arm(s) %s; expected a subset of %s' % (bad, A.ARM_ORDER))
            return 2
        never = [x for x in want if not A.ARMS[x][3]]
        if never:
            say('REFUSING: %s is the published arm and is not retrained by this campaign' % never)
            return 2
        arms = want
    if '-' in a.seeds:
        lo, hi = a.seeds.split('-')
        seeds = list(range(int(lo), int(hi) + 1))
    else:
        seeds = [int(x) for x in a.seeds.split(',')]

    units = [(arm, s) for arm in arms for s in seeds]
    todo = [u for u in units if not unit_done(*u)]
    say('=' * 96)
    say('2x2 ASSIGNMENT ABLATION | %d arms x %d seeds = %d trainings | steps=%s'
        % (len(arms), len(seeds), len(units), '{:,}'.format(a.steps)))
    for arm in arms:
        say('  %-5s %s' % (arm, A.describe(arm)))
    say('  g_bi  %s' % A.describe('g_bi'))
    say('       ^ published arm: NOT retrained. Its five checkpoints already exist.')
    say('%d to run, %d already complete | slots=%d floor=%d MB'
        % (len(todo), len(units) - len(todo), a.slots, a.floor_mb))
    if a.plan or not a.launch:
        say('PLAN ONLY - nothing launched. Add --launch to run.')
        return 0
    if not todo:
        say('nothing to do')
        return 0
    failed = launch(todo, a.slots, a.steps, a.floor_mb)
    if failed:
        say('FAILED units: %s' % failed)
        return 1
    say('NEXT: python -u q3_eval_unit.py --all --slots %d' % a.slots)
    return 0


if __name__ == '__main__':
    sys.exit(main())
