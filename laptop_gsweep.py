#!/usr/bin/env python3
"""
laptop_gsweep.py -- M4 strong form, resumable. Run it on the laptop; stop it whenever.

    python -u laptop_gsweep.py

Ctrl-C, close the lid, reboot -- then run the same command again. It picks up inside the
training it was in the middle of, not at the start of it. Nothing already finished is redone.

WHAT IT PRODUCES
    6 guard constants {6,7,8,9,10,12} dB x 5 training seeds = 30 controllers at 8e5 steps,
    each then evaluated over the R = 8 common-random-number grid at the published channel seeds.
    That is the curve needed to say whether G = 9 dB was the right CHOICE, as opposed to how a
    G=9 policy behaves when the guard is moved under it (which is the cheap sweep, already run).

WHY G = 9 IS RETRAINED HERE TOO, rather than reusing br_gst_s*.pt: training is not reproducible
across machines, so a curve with one point from another venue would confound the guard with the
hardware. Every point on this curve comes from this machine.

RESUME GUARANTEE. Snapshots every SNAP_EVERY steps at an episode boundary, written atomically.
resumable_selftest2.py proves a killed-and-resumed run is BIT-IDENTICAL to an uninterrupted one.
A checkpoint is only accepted as finished if its .done marker exists -- a bare .pt file may be a
mid-run best-so-far save, and accepting one silently would put an undertrained controller in the
table. That exact failure was seen once and is what the marker prevents.

TIME. ~50 min per training on one core; 30 trainings. Two at a time on a laptop is comfortable.
Set SLOTS below. Evaluation adds ~2.5 min per controller at the end and is separately resumable.
"""
import csv
import hashlib
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

G_GRID = [6.0, 7.0, 8.0, 9.0, 10.0, 12.0]
SEEDS = [0, 1, 2, 3, 4]
STEPS = int(os.environ.get('GS_STEPS', 800_000))
SNAP_EVERY = int(os.environ.get('GS_SNAP', 20_000))
SLOTS = int(os.environ.get('GS_SLOTS', 2))
R = 8
OUT = os.path.join('results', 'gtrain')
METRICS = ['HO', 'Power', 'Spec', 'RLF', 'Tput']


def log(m):
    print(time.strftime('%H:%M:%S') + ' | ' + m, flush=True)


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def ck(G, s):
    return 'gt_G%s_s%d.pt' % (('%g' % G).replace('.', 'p'), s)


def snap(G, s):
    return ck(G, s) + '.snap'


def done(G, s):
    return os.path.exists(ck(G, s)) and os.path.exists(ck(G, s) + '.done')


def train_one(G, s):
    sys.argv = [sys.argv[0]]
    import torch
    torch.set_num_threads(1)
    import train_branched_nyu as B
    import train_phase2 as tp
    import resumable_train as RT
    slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH)
    log('[G=%g s%d] training %s steps (snapshot every %s)'
        % (G, s, format(STEPS, ','), format(SNAP_EVERY, ',')))
    t0 = time.time()
    RT.train_resumable(B, s, True, ck(G, s), slow, med, steps=STEPS, guard_db=G,
                       guard_mode='state', snapshot=snap(G, s), snap_every=SNAP_EVERY)
    with open(ck(G, s) + '.done', 'w') as f:
        f.write('%s %d %s\n' % (md5(ck(G, s)), STEPS,
                                time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    log('[G=%g s%d] DONE in %.1f min' % (G, s, (time.time() - t0) / 60))
    return 0


def evaluate():
    sys.argv = [sys.argv[0]]
    import torch
    torch.set_num_threads(1)
    import numpy as np
    import env_6g_v5
    import train_phase2 as tp
    import train_branched_nyu as B
    from dndqn_branched import load_branched
    from dndqn import OBS_DIM
    import tablevii_r8 as T7
    T7.install_patch(env_6g_v5)
    slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH)
    pool = slow + med
    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    path = os.path.join(OUT, 'gtrain.csv')
    rows = list(csv.DictReader(open(path))) if os.path.exists(path) else []
    have = {(float(r['guard_db']), int(r['seed']), int(r['repeat'])) for r in rows}
    for G in G_GRID:
        for s in SEEDS:
            if not done(G, s):
                log('[eval] G=%g s%d not trained yet — skipping' % (G, s))
                continue
            if all((G, s, r) in have for r in range(R)):
                continue
            net = load_branched(ck(G, s), OBS_DIM)[0]
            for r in range(R):
                if (G, s, r) in have:
                    continue
                with T7.seeded(r):
                    m = B.evaluate_branched(net, pool, True, guard_db=G, guard_mode='state')
                row = {'guard_db': G, 'seed': s, 'ckpt': ck(G, s), 'ckpt_md5': md5(ck(G, s)),
                       'repeat': r, 'chan_seed_first': T7.BASE + 100000 * r, 'steps': STEPS,
                       'trained_at_G': G, 'venue': 'laptop', 'torch': torch.__version__,
                       'numpy': np.__version__,
                       'measured_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
                for k in METRICS:
                    row[k] = '%.4f' % float(m[k])
                rows.append(row)
                with open(path + '.tmp', 'w', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)
                os.replace(path + '.tmp', path)
            log('[eval] G=%g s%d done' % (G, s))
    log('evaluation complete -> %s' % path)
    return 0


def main():
    if len(sys.argv) > 3 and sys.argv[1] == '--train-one':
        return train_one(float(sys.argv[2]), int(sys.argv[3]))
    if len(sys.argv) > 1 and sys.argv[1] == '--eval':
        return evaluate()

    import subprocess
    todo = [(G, s) for G in G_GRID for s in SEEDS if not done(G, s)]
    fin = len(G_GRID) * len(SEEDS) - len(todo)
    resumable = [(G, s) for G, s in todo if os.path.exists(snap(G, s))]
    log('=' * 84)
    log('M4 STRONG FORM | %d of %d trainings finished | %d to go (%d mid-run, will resume)'
        % (fin, len(G_GRID) * len(SEEDS), len(todo), len(resumable)))
    log('~%.1f h remaining at %d slot(s). Ctrl-C is safe: rerun to continue.'
        % (len(todo) * 50.0 / 60.0 / SLOTS, SLOTS))
    log('=' * 84)
    running = []
    try:
        while todo or running:
            while todo and len(running) < SLOTS:
                G, s = todo.pop(0)
                lf = open('gt_G%s_s%d.log' % (('%g' % G).replace('.', 'p'), s), 'a')
                p = subprocess.Popen([sys.executable, '-u', __file__, '--train-one',
                                      '%g' % G, str(s)], stdout=lf, stderr=subprocess.STDOUT)
                running.append((G, s, p, lf))
                log('  started G=%g s%d (pid %d), %d queued' % (G, s, p.pid, len(todo)))
            time.sleep(15)
            for it in list(running):
                G, s, p, lf = it
                if p.poll() is not None:
                    lf.close()
                    running.remove(it)
                    log('  G=%g s%d exited rc=%d %s'
                        % (G, s, p.returncode, 'OK' if done(G, s) else '(will resume next run)'))
    except KeyboardInterrupt:
        log('interrupted — snapshots are on disk; rerun this script to continue')
        for G, s, p, lf in running:
            p.terminate()
        return 130
    log('all 30 trainings complete — evaluating')
    return evaluate()


if __name__ == '__main__':
    sys.exit(main())
