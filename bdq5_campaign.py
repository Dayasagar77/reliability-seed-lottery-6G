#!/usr/bin/env python3
"""
bdq5_campaign.py -- C4: retrain ALL FIVE branched-incremental (BDQ-style) seeds under one protocol.

    python -u bdq5_campaign.py            # 5 trainings x 8e5 steps, then the R=8 CRN grid
    python -u bdq5_campaign.py --eval     # evaluation only, for checkpoints already finished

WHY. Reviewer 2, C4: three of the five branched-incremental checkpoints were regenerated on
2026-08-01 after the originals were lost, while seeds 0 and 1 date from 2026-07-12. For that arm
alone a change of protocol and a change of weights cannot be separated -- and it is the arm
carrying the claim that fine-grained branching without binarisation is worse. The manuscript
discloses this honestly. Retraining all five under today's code REMOVES the confound instead of
disclosing it, which is the only item in that review that upgrades a limitation into a result.

The existing checkpoints are NOT touched: new weights are written as bdq5_s{seed}.pt, so the
published grid stays reproducible and the two can be compared.

ONE-ORIGIN GATE. The confound was heterogeneous provenance, so provenance is what gets checked.
Each .done marker records the training-module hash, the step budget and the finish time, and the
evaluator REFUSES to build a grid unless all five agree on module hash and budget. A retrain that
silently reintroduced mixed provenance would look exactly like a clean one in the numbers.

ARM DEFINITION, taken from train_branched_nyu.main(): the incremental arm is
('branched-incremental (BDQ)', binary_power=False, tag='inc', guard=None) -- fine-grained power
dial, no binarisation, NO guard. Evaluation must match: binary_power=False, guard_db=None.
"""
import csv
import hashlib
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

SEEDS = [0, 1, 2, 3, 4]
STEPS = int(os.environ.get('BDQ_STEPS', 800_000))
SNAP_EVERY = int(os.environ.get('BDQ_SNAP', 20_000))
SLOTS = int(os.environ.get('BDQ_SLOTS', 2))
R = 8
VENUE = 'laptop'
OUT = os.path.join('results', 'bdq5')
TRAINER = 'train_branched_nyu.py'
METRICS = ['HO', 'Power', 'Spec', 'RLF', 'Tput']


def log(m):
    print(time.strftime('%H:%M:%S') + ' | ' + m, flush=True)


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def ck(s):
    return 'bdq5_s%d.pt' % s


def done(s):
    return os.path.exists(ck(s)) and os.path.exists(ck(s) + '.done')


def _imports():
    sys.argv = [sys.argv[0]]
    import torch
    torch.set_num_threads(1)
    import train_phase2 as tp
    import train_branched_nyu as B
    import resumable_train as RT
    return torch, tp, B, RT


def train_one(s, steps=None):
    steps = steps or STEPS
    torch, tp, B, RT = _imports()
    slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH)
    log('[bdq s%d] training %s steps (incremental power dial, NO binarisation, NO guard)'
        % (s, format(steps, ',')))
    t0 = time.time()
    RT.train_resumable(B, s, False, ck(s), slow, med, steps=steps, guard_db=None,
                       guard_mode='state', snapshot=ck(s) + '.snap', snap_every=SNAP_EVERY)
    with open(ck(s) + '.done', 'w') as f:
        f.write('%s %d %s %s\n' % (md5(ck(s)), steps, md5(TRAINER),
                                   time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    log('[bdq s%d] DONE in %.1f min' % (s, (time.time() - t0) / 60))
    return 0


def origin_gate():
    """All five checkpoints must come from one protocol. That is the whole point of C4."""
    marks = {}
    for s in SEEDS:
        p = ck(s) + '.done'
        if not os.path.exists(p):
            return None, 's%d has no .done marker' % s
        parts = open(p).read().split()
        if len(parts) < 4:
            return None, 's%d .done marker is malformed' % s
        marks[s] = dict(ckpt_md5=parts[0], steps=int(parts[1]), trainer_md5=parts[2],
                        finished=parts[3])
    tr = {m['trainer_md5'] for m in marks.values()}
    st = {m['steps'] for m in marks.values()}
    if len(tr) != 1:
        return None, 'mixed trainer hashes across seeds: %s' % sorted(tr)
    if len(st) != 1:
        return None, 'mixed step budgets across seeds: %s' % sorted(st)
    return marks, None


def evaluate():
    torch, tp, B, RT = _imports()
    import numpy as np
    import env_6g_v5
    from train_branched_nyu import evaluate_branched
    from dndqn_branched import load_branched
    from dndqn import OBS_DIM
    import tablevii_r8 as T7
    T7.install_patch(env_6g_v5)

    marks, err = origin_gate()
    if err:
        log('*** ONE-ORIGIN GATE FAILED: %s ***' % err)
        log('    C4 asks for a single protocol across all five seeds. Refusing to write a grid '
            'that would carry the same confound it is meant to remove.')
        return 3
    log('one-origin gate PASS: trainer %s, %s steps, all five seeds'
        % (list({m['trainer_md5'] for m in marks.values()})[0][:12],
           format(list({m['steps'] for m in marks.values()})[0], ',')))

    slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH)
    pool = slow + med
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, 'bdq5.csv')
    rows = list(csv.DictReader(open(path))) if os.path.exists(path) else []
    have = {(int(r['seed']), int(r['repeat'])) for r in rows}

    for s in SEEDS:
        net, meta = load_branched(ck(s), OBS_DIM)
        if meta.get('binary_power') is not False:
            log('*** s%d checkpoint says binary_power=%r, expected False. REFUSING: this is not '
                'the incremental arm. ***' % (s, meta.get('binary_power')))
            return 4
        for r in range(R):
            if (s, r) in have:
                continue
            with T7.seeded(r):
                m = evaluate_branched(net, pool, False, guard_db=None, guard_mode='state')
            row = {'seed': s, 'ckpt': ck(s), 'ckpt_md5': md5(ck(s)), 'repeat': r,
                   'chan_seed_first': T7.BASE + 100000 * r, 'steps': STEPS,
                   'arm': 'incremental_retrained', 'binary_power': 'False', 'guard_db': '',
                   'trainer_md5': marks[s]['trainer_md5'], 'origin': 'retrained-20260803',
                   'venue': VENUE, 'torch': torch.__version__, 'numpy': np.__version__,
                   'measured_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
            for k in METRICS:
                row[k] = '%.4f' % float(m[k])
            rows.append(row)
        log('  s%d evaluated' % s)
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    log('wrote %s (%d cells)' % (path, len(rows)))

    # ---- summary against the published mixed-provenance arm ---------------------------------
    import tablevii_decompose as D
    D.CSV = 'results/tablevii_r8_laptop.csv'

    def grid(cells):
        out = {}
        for m in METRICS:
            X = np.array([[float(cells[(s, r)][m]) for r in range(R)] for s in SEEDS])
            d = D.decompose(X)
            sd = ('%.2f' % d['sd_train']) if d['resolvable'] else '<%.2f' % d['sd_train_hi']
            out[m] = (d['grand'], sd)
        return out

    new = grid({(int(r['seed']), int(r['repeat'])): r for r in rows})
    old_rows = [r for r in csv.DictReader(open('results/tablevii_r8_laptop.csv'))
                if r['arm'] == 'incremental']
    print()
    print('=' * 96)
    print('C4 -- BRANCHED-INCREMENTAL (BDQ) RETRAINED, ALL FIVE SEEDS, ONE PROTOCOL')
    print('=' * 96)
    if old_rows:
        old = grid({(int(r['seed']), int(r['repeat'])): r for r in old_rows})
        print('  %-8s %28s %28s' % ('metric', 'published (mixed origin)', 'retrained (one origin)'))
        for m in METRICS:
            print('  %-8s %18.2f +- %-7s %18.2f +- %-7s'
                  % (m, old[m][0], old[m][1], new[m][0], new[m][1]))
    else:
        print('  published incremental arm not found in tablevii_r8_laptop.csv; new arm only')
        for m in METRICS:
            print('  %-8s %18.2f +- %-7s' % (m, new[m][0], new[m][1]))
    print('=' * 96)
    return 0


def main():
    if '--train-one' in sys.argv:
        i = sys.argv.index('--train-one')
        return train_one(int(sys.argv[i + 1]))
    if '--eval' in sys.argv:
        return evaluate()
    todo = [s for s in SEEDS if not done(s)]
    resumable = [s for s in todo if os.path.exists(ck(s) + '.snap')]
    log('=' * 84)
    log('C4 -- retraining the branched-incremental arm, all five seeds, one protocol')
    log('  todo %s   resumable %s   slots %d' % (todo or 'none', resumable or 'none', SLOTS))
    log('  ~%.1f h remaining. Ctrl-C is safe: rerun to continue.'
        % (len(todo) * 40.0 / 60.0 / SLOTS))
    log('=' * 84)
    running = []
    try:
        while todo or running:
            while todo and len(running) < SLOTS:
                s = todo.pop(0)
                p = subprocess.Popen([sys.executable, '-u', __file__, '--train-one', str(s)],
                                     stdout=sys.stdout, stderr=sys.stderr)
                running.append((s, p))
                log('  started s%d (pid %d), %d queued' % (s, p.pid, len(todo)))
            time.sleep(5)
            for s, p in list(running):
                if p.poll() is not None:
                    running.remove((s, p))
                    log('  s%d exited rc=%s %s' % (s, p.returncode,
                                                   'OK' if p.returncode == 0 else '*** FAILED ***'))
    except KeyboardInterrupt:
        log('interrupted - snapshots are on disk; rerun this script to continue')
        return 1
    log('all five trainings complete - evaluating')
    return evaluate()


if __name__ == '__main__':
    sys.exit(main())
