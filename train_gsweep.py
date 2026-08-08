#!/usr/bin/env python3
"""
train_gsweep.py -- M4, the strong form: TRAIN a controller at each guard constant.

WHY THE CHEAP VERSION IS NOT ENOUGH. review_M4_gsweep.py evaluates the five G=9 checkpoints at
other G. That measures how a policy behaves when the guard it learned to lean on is changed
underneath it -- guard ROBUSTNESS. It cannot answer "was 9 dB the right choice", because a
controller trained at G = 6 would have learned a different policy: the guard is part of the
environment transition (the applied power, not the intent, determines the next state), so
changing G changes what the agent is optimising against, not merely what happens at test time.

Answering guard SELECTION needs a controller trained at each G. That is what this does.

    6 guard constants x 5 training seeds x 8e5 steps = 30 trainings
    then R = 8 CRN evaluation of each, at the SAME G it was trained at

VENUE. All 30 trainings and their evaluations happen in this container, so the curve is
internally one venue and one protocol. It is NOT mixed with the laptop-trained Table VII row --
the G = 9 point of this curve is its own in-venue reference, which is why G = 9 is retrained here
rather than reusing br_gst_s*.pt. Training is not reproducible across venues (or within one), so
a curve assembled from two venues would confound the guard with the machine.

Resumable at checkpoint granularity and at evaluation-row granularity: a restart re-does nothing
that finished.
"""
import csv
import hashlib
import os
import subprocess
import sys
import time

G_GRID = [6.0, 7.0, 8.0, 9.0, 10.0, 12.0]
SEEDS = [0, 1, 2, 3, 4]
STEPS = int(os.environ.get('GS_STEPS', 800_000))
R = 8
SLOTS = 2
OUT = 'results/gtrain'
METRICS = ['HO', 'Power', 'Spec', 'RLF', 'Tput']


def log(m):
    print(time.strftime('%Y-%m-%d %H:%M:%S') + ' | ' + m, flush=True)


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def tag(G, s):
    return 'g%02d_s%d' % (int(round(G * 10)) // 10 if G == int(G) else int(G * 10), s)


def ck_for(G, s):
    return 'gt_G%s_s%d.pt' % (('%g' % G).replace('.', 'p'), s)


def train_one(G, s):
    saved = sys.argv
    sys.argv = [saved[0]]
    import torch
    torch.set_num_threads(1)
    import train_branched_nyu as B
    import train_phase2 as tp
    sys.argv = saved
    slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH)
    out = ck_for(G, s)
    log('[train G=%g s%d] %s steps -> %s' % (G, s, format(STEPS, ','), out))
    t0 = time.time()
    B.train(s, True, out, slow, med, steps=STEPS, guard_db=G, guard_mode='state')
    log('[train G=%g s%d] done in %.1f min | md5 %s' % (G, s, (time.time() - t0) / 60, md5(out)))
    return 0


def eval_all():
    saved = sys.argv
    sys.argv = [saved[0]]
    import torch
    torch.set_num_threads(1)
    import numpy as np
    import env_6g_v5
    import train_phase2 as tp
    import train_branched_nyu as B
    from dndqn_branched import load_branched
    from dndqn import OBS_DIM
    import tablevii_r8 as T7
    sys.argv = saved
    T7.install_patch(env_6g_v5)
    slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH)
    pool = slow + med
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, 'gtrain.csv')
    rows = list(csv.DictReader(open(path))) if os.path.exists(path) else []
    done = {(float(r['guard_db']), int(r['seed']), int(r['repeat'])) for r in rows}
    for G in G_GRID:
        for s in SEEDS:
            ck = ck_for(G, s)
            if not os.path.exists(ck):
                log('[eval] MISSING %s -- skipping' % ck)
                continue
            if all((G, s, r) in done for r in range(R)):
                continue
            net = load_branched(ck, OBS_DIM)[0]
            for r in range(R):
                if (G, s, r) in done:
                    continue
                t0 = time.time()
                with T7.seeded(r):
                    m = B.evaluate_branched(net, pool, True, guard_db=G, guard_mode='state')
                row = {'guard_db': G, 'seed': s, 'ckpt': ck, 'ckpt_md5': md5(ck), 'repeat': r,
                       'chan_seed_first': T7.BASE + 100000 * r, 'steps': STEPS,
                       'secs': round(time.time() - t0, 1),
                       'venue': 'cloud-hw-torch2.5.1', 'torch': torch.__version__,
                       'numpy': np.__version__, 'trained_at_G': G,
                       'env_v5_md5': md5('env_6g_v5.py'),
                       'env_branched_md5': md5('env_6g_branched.py'),
                       'driver_md5': md5('train_branched_nyu.py'),
                       'measured_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
                for k in METRICS:
                    row[k] = '%.4f' % float(m[k])
                rows.append(row)
                with open(path + '.tmp', 'w', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)
                os.replace(path + '.tmp', path)
            log('[eval G=%g s%d] done' % (G, s))
    return 0


def main():
    if len(sys.argv) > 3 and sys.argv[1] == '--train-one':
        return train_one(float(sys.argv[2]), int(sys.argv[3]))
    if len(sys.argv) > 1 and sys.argv[1] == '--eval':
        return eval_all()

    todo = [(G, s) for G in G_GRID for s in SEEDS if not os.path.exists(ck_for(G, s))]
    log('=' * 92)
    log('M4 STRONG FORM | %d guard constants x %d seeds | %s steps | %d already trained'
        % (len(G_GRID), len(SEEDS), format(STEPS, ','), len(G_GRID) * len(SEEDS) - len(todo)))
    log('estimated %.1f h at %d slots' % (len(todo) * 50.0 / 60.0 / SLOTS, SLOTS))
    log('=' * 92)
    running = []
    while todo or running:
        while todo and len(running) < SLOTS:
            G, s = todo.pop(0)
            f = open('gt_train_G%s_s%d.log' % (('%g' % G).replace('.', 'p'), s), 'w')
            p = subprocess.Popen([sys.executable, '-u', __file__, '--train-one', '%g' % G, str(s)],
                                 stdout=f, stderr=subprocess.STDOUT)
            running.append((G, s, p, f))
            log('  launched G=%g s%d (pid %d) | %d left in queue' % (G, s, p.pid, len(todo)))
        time.sleep(15)
        for item in list(running):
            G, s, p, f = item
            if p.poll() is not None:
                f.close()
                running.remove(item)
                ok = os.path.exists(ck_for(G, s))
                log('  G=%g s%d exited rc=%d %s' % (G, s, p.returncode,
                                                    'md5 ' + md5(ck_for(G, s))[:8] if ok
                                                    else '*** NO CHECKPOINT ***'))
    log('all trainings complete -- evaluating')
    return subprocess.call([sys.executable, '-u', __file__, '--eval'])


if __name__ == '__main__':
    sys.exit(main())
