#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
q4_campaign.py - trains the three new arms of the pre-registered risk-asymmetry campaign.

    qr_sym      quantile trunk, both branches act on the mean, guard 9 dB      (baseline)
    qr_asym     quantile trunk, reliability CVaR(0.25), spectrum mean, guard   (proposed)
    qr_asym_u   the same, GUARD DISABLED                                       (the headline)

optional:
    qr_inv      quantile trunk, reliability mean, spectrum CVaR(0.25), guard   (falsifier)

The two published comparators - the guarded and the unguarded binary arms - are NOT retrained.
Their checkpoints and their R = 8 evaluations already exist.

PROVENANCE. The training loop is the PUBLISHED train_branched_nyu.train(), reached by rebinding
three module-level names before it is called:

    B.build_branched_networks -> q4 quantile builder
    B.branched_td_loss        -> q4 quantile Huber loss
    B.save_branched           -> q4 quantile save

Everything else - epsilon schedule, replay buffer, n-step returns, PER, batch size, target update,
curriculum stage, episode handling, the best-checkpoint rule, Adam at 3e-4, the 8 x 10^5 budget -
is the published object, unedited. No file on disk is modified.

USAGE (PowerShell, from the folder holding the drivers)
    python -u q4_campaign.py --plan
    python -u q4_campaign.py --launch --slots 8
    python -u q4_campaign.py --unit qr_asym_u 0
    python -u q4_campaign.py --launch --arms qr_inv          # the optional falsifier
"""
import os, sys, csv, json, glob, time, shutil, hashlib, argparse, subprocess

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

OUTDIR = 'results/q4_riskasym'
MANIFEST = os.path.join(OUTDIR, 'q4_manifest.csv')
SUPERSEDED = os.path.join(OUTDIR, '_superseded')
SNAP_EVERY = 25_000            # steps between crash-recovery snapshots
MANIFEST_COLS = ['arm', 'label', 'seed', 'steps', 'steps_executed', 'resumed',
                 'risk_reliability', 'risk_spectrum', 'alpha', 'n_quantiles',
                 'guard_db', 'guard_mode', 'binary_power', 'ckpt', 'md5',
                 'train_min', 'utc', 'torch', 'numpy']
LOGDIR = 'logs'
SEEDS = [0, 1, 2, 3, 4]
STEPS_REAL = 800_000
GUARD_DB, GUARD_MODE = 9.0, 'state'

# key -> (label, risk modes, guard_db, binary_power, in the default launch set)
ARMS = {
    'qr_sym':    ('Quantile trunk, symmetric mean (baseline)',            'sym',  GUARD_DB, True, True),
    'qr_asym':   ('Quantile trunk, reliability CVaR + spectrum mean',     'asym', GUARD_DB, True, True),
    'qr_asym_u': ('Quantile trunk, reliability CVaR, GUARD DISABLED',     'asym', None,     True, True),
    'qr_inv':    ('Quantile trunk, INVERTED risk assignment (falsifier)', 'inv',  GUARD_DB, True, False),
}
ARM_ORDER = ['qr_sym', 'qr_asym', 'qr_asym_u', 'qr_inv']
DEFAULT_ARMS = [a for a in ARM_ORDER if ARMS[a][4]]


def say(msg, logname='q4_campaign'):
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
    return os.path.join(OUTDIR, 'q4_%s_s%d.pt' % (arm, seed))


def sidecar_path(arm, seed):
    """The per-unit completion witness. One writer per file, so concurrent units cannot race."""
    return os.path.join(OUTDIR, 'q4_%s_s%d.done.json' % (arm, seed))


def snap_path(arm, seed):
    """Where the resumable loop parks its crash-recovery state for this unit."""
    return os.path.join(OUTDIR, 'q4_%s_s%d.snapshot' % (arm, seed))


def unit_done(arm, seed, steps):
    """True iff this unit RAN TO COMPLETION at the requested step budget.

    NOT "does the checkpoint exist". train_branched_nyu.train() overwrites the checkpoint on every
    improvement from frac > EPS_FRACTION (= 0.30) onward, so the file is on disk from roughly step
    240,001 of 800,000 and its presence proves only that the run got past 30 per cent. The sidecar
    is written after training returns and carries the md5 and the step budget, so neither a
    shorter rehearsal nor a checkpoint replaced by a later run can pass as complete.
    """
    sc, ck = sidecar_path(arm, seed), ckpt_path(arm, seed)
    if not (os.path.exists(sc) and os.path.exists(ck)):
        return False
    try:
        rec = json.load(open(sc, encoding='utf-8'))
    except (OSError, ValueError):
        return False
    if int(rec.get('steps', -1)) != int(steps):
        return False
    return rec.get('md5') == md5(ck)


def write_sidecar(rec):
    """Atomic, single-writer completion witness: temp file then os.replace."""
    os.makedirs(OUTDIR, exist_ok=True)
    sc = sidecar_path(rec['arm'], rec['seed'])
    tmp = sc + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(rec, fh, indent=2, sort_keys=True)
    os.replace(tmp, sc)
    return sc


def quarantine():
    """Move every prior artefact of this campaign aside so it restarts from zero. No deletes."""
    items = sorted(glob.glob(os.path.join(OUTDIR, 'q4_*.pt'))
                   + glob.glob(os.path.join(OUTDIR, 'q4_*.done.json'))
                   + glob.glob(os.path.join(OUTDIR, 'q4_*.snapshot')))
    if os.path.exists(MANIFEST):
        items.append(MANIFEST)
    if not items:
        return 0, None
    dst = os.path.join(SUPERSEDED, time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()))
    os.makedirs(dst, exist_ok=True)
    for p in items:
        shutil.move(p, os.path.join(dst, os.path.basename(p)))
    return len(items), dst


def rebuild_manifest():
    """Regenerate the manifest CSV from the race-free sidecars."""
    os.makedirs(OUTDIR, exist_ok=True)
    recs = []
    for p in sorted(glob.glob(os.path.join(OUTDIR, 'q4_*.done.json'))):
        try:
            recs.append(json.load(open(p, encoding='utf-8')))
        except (OSError, ValueError):
            say('unreadable sidecar %s - skipped' % p)
    recs.sort(key=lambda r: (str(r.get('arm', '')), int(r.get('seed', 0))))
    with open(MANIFEST, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(MANIFEST_COLS)
        for r in recs:
            w.writerow([r.get(c, '') for c in MANIFEST_COLS])
    return len(recs)


def risk_modes(tag):
    import q4_qr_branched as Q
    return {'sym': Q.RISK_SYM, 'asym': Q.RISK_ASYM, 'inv': Q.RISK_INV}[tag]


def run_unit(arm, seed, steps):
    import torch
    torch.set_num_threads(1)
    import numpy as np
    import q4_qr_branched as Q

    if arm not in ARMS:
        say('unknown arm %r' % arm)
        return 2
    label, tag, guard, binary_power, _ = ARMS[arm]
    risk = risk_modes(tag)

    saved = sys.argv[:]
    sys.argv = ['train_branched_nyu.py', '1', str(steps)]
    try:
        import train_branched_nyu as B
    finally:
        sys.argv = saved
    import train_phase2 as tp

    # ---- the only three rebinds. The trainer itself is the published object.
    orig = (B.build_branched_networks, B.branched_td_loss, B.save_branched)

    def builder(obs_dim, *a, **kw):
        return Q.build_qr_networks(obs_dim, risk=risk)

    def saver(net, path, extra=None):
        e = dict(extra or {})
        e.update({'arm': arm, 'risk': risk, 'alpha': Q.CVAR_ALPHA,
                  'n_quantiles': Q.N_QUANTILES, 'guard_db': guard,
                  'target_steps': steps})
        return Q.save_qr(net, path, e)

    B.build_branched_networks = builder
    B.branched_td_loss = Q.qr_branched_td_loss
    B.save_branched = saver

    os.makedirs(OUTDIR, exist_ok=True)
    out = ckpt_path(arm, seed)
    slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH)
    say('train START %s s%d | %s | risk=%s alpha=%.2f N=%d guard=%s | steps=%s | pool=%d'
        % (arm, seed, label, risk, Q.CVAR_ALPHA, Q.N_QUANTILES, guard,
           '{:,}'.format(steps), len(slow) + len(med)), 'q4_%s_s%d' % (arm, seed))
    import q4_resumable as R
    snap = snap_path(arm, seed)
    t0 = time.time()
    try:
        executed = R.train_resumable(B, seed, binary_power, out, slow, med, steps=steps,
                                     guard_db=guard, guard_mode=GUARD_MODE,
                                     snap_path=snap, snap_every=SNAP_EVERY,
                                     log=lambda m: say(m, 'q4_%s_s%d' % (arm, seed)))
    finally:
        B.build_branched_networks, B.branched_td_loss, B.save_branched = orig
    mins = (time.time() - t0) / 60.0

    if not os.path.exists(out):
        say('train FAILED %s s%d - no checkpoint' % (arm, seed), 'q4_%s_s%d' % (arm, seed))
        return 1
    rec = {'arm': arm, 'label': label, 'seed': seed, 'steps': steps,
           'steps_executed': executed, 'resumed': bool(executed < steps),
           'risk_reliability': risk[0], 'risk_spectrum': risk[1], 'alpha': Q.CVAR_ALPHA,
           'n_quantiles': Q.N_QUANTILES, 'guard_db': guard, 'guard_mode': GUARD_MODE,
           'binary_power': binary_power, 'ckpt': os.path.basename(out), 'md5': md5(out),
           'train_min': '%.1f' % mins,
           'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'torch': torch.__version__, 'numpy': np.__version__}

    # the completion witness is written FIRST, atomically, by this unit alone. A process that
    # dies before this point leaves no sidecar, so the unit is correctly seen as incomplete -
    # and its snapshot is still there, so it resumes rather than restarting.
    write_sidecar(rec)
    R.clear_snapshot(snap)          # retired, not deleted: moved to _spent_snapshots/

    new = not os.path.exists(MANIFEST)
    with open(MANIFEST, 'a', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(MANIFEST_COLS)
        w.writerow([rec[c] for c in MANIFEST_COLS])
    say('train DONE  %s s%d in %.1fm -> %s' % (arm, seed, mins, os.path.basename(out)),
        'q4_%s_s%d' % (arm, seed))
    return 0


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


def busy():
    """Refuse to start while the 2x2 campaign is still running - one campaign at a time."""
    p = 'q3_campaign.status'
    if not os.path.exists(p):
        return False
    try:
        head = open(p, encoding='utf-8').readline()
    except OSError:
        return False
    if 'running=0' in head or 'done=15/15' in head:
        return False
    stale_min = (time.time() - os.path.getmtime(p)) / 60
    return stale_min < 10


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
            if p.returncode == 0 and unit_done(arm, seed, steps):
                done += 1
                say('DONE  %s in %.1fm (%d/%d)' % (name, mins, done, total))
            else:
                failed.append((arm, seed))
                say('FAIL  %s after %.1fm rc=%s - see logs/%s.log' % (name, mins, p.returncode, name))
            del running[name]
        while queue and len(running) < slots:
            free = free_mb()
            if free < floor_mb:
                say('HOLD  %d MB free < floor %d MB' % (free, floor_mb))
                break
            arm, seed = queue.pop(0)
            name = 'q4_%s_s%d' % (arm, seed)
            fh = open(os.path.join(LOGDIR, name + '.log'), 'a', encoding='utf-8')
            cmd = [sys.executable, '-u', os.path.abspath(__file__), '--unit', arm, str(seed),
                   '--steps', str(steps)]
            p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=os.getcwd())
            running[name] = (p, time.time(), fh, (arm, seed))
            say('START %s pid %d (%d MB free)' % (name, p.pid, free))
        txt = ('%s phase=train done=%d/%d running=%d free_mb=%d elapsed_min=%.1f'
               % (time.strftime('%Y-%m-%d %H:%M:%S'), done, total, len(running), free_mb(),
                  (time.time() - t0) / 60))
        for name, rec in running.items():
            txt += '\n  %s pid %d %.1fm' % (name, rec[0].pid, (time.time() - rec[1]) / 60)
        try:
            open('q4_campaign.status', 'w', encoding='utf-8').write(txt + '\n')
        except OSError:
            pass
        if queue or running:
            time.sleep(20)
    say('campaign finished in %.1f min | %d done, %d failed' % ((time.time() - t0) / 60, done, len(failed)))
    return failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--unit', nargs=2, metavar=('ARM', 'SEED'))
    ap.add_argument('--launch', action='store_true')
    ap.add_argument('--plan', action='store_true')
    ap.add_argument('--arms', default='')
    ap.add_argument('--seeds', default='0-4')
    ap.add_argument('--slots', type=int, default=8)
    ap.add_argument('--floor-mb', type=int, default=1500)
    ap.add_argument('--steps', type=int, default=STEPS_REAL)
    ap.add_argument('--allow-concurrent', action='store_true')
    ap.add_argument('--fresh', action='store_true',
                    help='move prior checkpoints, sidecars, snapshots and the manifest '
                         'into results/q4_riskasym/_superseded/<utc>/ so the campaign '
                         'starts from zero. Nothing is deleted.')
    ap.add_argument('--rebuild-manifest', action='store_true',
                    help='regenerate the manifest CSV from the per-unit sidecars')
    a = ap.parse_args()

    if a.rebuild_manifest:
        n = rebuild_manifest()
        say('manifest rebuilt from %d sidecar(s) -> %s' % (n, MANIFEST))
        return 0

    if a.unit:
        return run_unit(a.unit[0], int(a.unit[1]), a.steps)

    if a.steps != STEPS_REAL:
        say('*** SHORT RUN *** steps=%s is a rehearsal, not a result' % '{:,}'.format(a.steps))

    arms = DEFAULT_ARMS
    if a.arms:
        want = [x.strip() for x in a.arms.split(',') if x.strip()]
        bad = [x for x in want if x not in ARMS]
        if bad:
            say('unknown arm(s) %s' % bad)
            return 2
        arms = want
    if '-' in a.seeds:
        lo, hi = a.seeds.split('-')
        seeds = list(range(int(lo), int(hi) + 1))
    else:
        seeds = [int(x) for x in a.seeds.split(',')]
    units = [(arm, s) for arm in arms for s in seeds]
    if a.fresh:
        n, dst = quarantine()
        say('--fresh: %d prior artefact(s) moved to %s' % (n, dst) if n
            else '--fresh: nothing to quarantine, the campaign is already clean')
    todo = [u for u in units if not unit_done(u[0], u[1], a.steps)]

    say('=' * 96)
    say('RISK-ASYMMETRY CAMPAIGN | %d arms x %d seeds = %d trainings | steps=%s'
        % (len(arms), len(seeds), len(units), '{:,}'.format(a.steps)))
    import q4_qr_branched as Q
    for arm in arms:
        label, tag, guard, bp, _ = ARMS[arm]
        say('  %-10s %s | risk=%s guard=%s' % (arm, label, risk_modes(tag), guard))
    say('  N = %d quantiles, CVaR alpha = %.2f, both fixed by the pre-registration'
        % (Q.N_QUANTILES, Q.CVAR_ALPHA))
    say('  published comparators (guarded and unguarded binary) are read, never retrained')
    say('%d to run, %d already complete | slots=%d floor=%d MB'
        % (len(todo), len(units) - len(todo), a.slots, a.floor_mb))

    if busy() and not a.allow_concurrent:
        say('REFUSING: the 2x2 campaign is still running (q3_campaign.status is live). One '
            'campaign at a time - the laptop cannot hold both. Wait for it, or pass '
            '--allow-concurrent if you know it has finished.')
        return 1
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
    say('NEXT: python -u q4_eval_unit.py --all --slots %d' % a.slots)
    return 0


if __name__ == '__main__':
    sys.exit(main())
