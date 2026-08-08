#!/usr/bin/env python3
"""
q5_campaign.py -- Q5: train and evaluate the throughput-floor arm, resumably.

    python -u q5_campaign.py --smoke     # ~3 min: does it train, does the term actually fire?
    python -u q5_campaign.py             # 5 trainings x 8e5 steps, then the R=8 CRN grid
    python -u q5_campaign.py --eval      # evaluation only, for checkpoints already finished

WHAT THIS ANSWERS. Q5 asks whether a simple throughput-floor term anchors the spectrum axis, and
says a negative result would be informative. So the campaign measures three things, not one:

  1. the R=8 CRN grid -- RLF, Power, Spec, Tput and their between-training components, directly
     comparable to the published guarded arm banked in results/tablevii_r8_laptop.csv;
  2. whether the DISPERSION on the spectrum/throughput axes actually fell (that is the anchoring
     claim, and it is the whole point of the term);
  3. whether reliability sensitivity came BACK -- the specific negative outcome the reviewer named.

  It also re-measures the per-step throughput distribution of the trained arm, using the same probe
  as q5_floor_probe.py, so "the floor moved the distribution" is a measurement rather than an
  inference from the mean.

RESUME. Snapshots at episode boundaries via resumable_train; a checkpoint counts as finished only
when its .done marker exists. A bare .pt is a mid-run best-so-far save.
"""
import csv
import hashlib
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

SEEDS = [0, 1, 2, 3, 4]
STEPS = int(os.environ.get('Q5_STEPS', 800_000))
SNAP_EVERY = int(os.environ.get('Q5_SNAP', 20_000))
SLOTS = int(os.environ.get('Q5_SLOTS', 2))
GUARD_DB, GUARD_MODE = 9.0, 'state'
R = 8
OUT = os.path.join('results', 'q5floor')
METRICS = ['HO', 'Power', 'Spec', 'RLF', 'Tput']
PCTS = [1, 5, 10, 15, 20, 25, 33, 50, 66, 75, 90, 95, 99]
VENUE = os.environ.get('Q5_VENUE', 'laptop')


def log(m):
    print(time.strftime('%H:%M:%S') + ' | ' + m, flush=True)


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def ck(s):
    return 'q5_fl_s%d.pt' % s


def done(s):
    return os.path.exists(ck(s)) and os.path.exists(ck(s) + '.done')


def _imports():
    sys.argv = [sys.argv[0]]
    import torch
    torch.set_num_threads(1)
    import train_phase2 as tp
    import train_branched_q5 as B5
    import resumable_train as RT
    return torch, tp, B5, RT


def train_one(s, steps=None):
    steps = steps or STEPS
    torch, tp, B5, RT = _imports()
    from q5_floor import T_FLOOR
    slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH)
    log('[q5 s%d] training %s steps (floor %.2f Mbps, guard %.1f dB, %s mode)'
        % (s, format(steps, ','), T_FLOOR, GUARD_DB, GUARD_MODE))
    t0 = time.time()
    RT.train_resumable(B5, s, True, ck(s), slow, med, steps=steps, guard_db=GUARD_DB,
                       guard_mode=GUARD_MODE, snapshot=ck(s) + '.snap', snap_every=SNAP_EVERY)
    with open(ck(s) + '.done', 'w') as f:
        f.write('%s %d %.4f %s\n' % (md5(ck(s)), steps, T_FLOOR,
                                     time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    log('[q5 s%d] DONE in %.1f min' % (s, (time.time() - t0) / 60))
    return 0


def tput_percentiles(net, pool, eval_seeds, eval_ep):
    """Per-step throughput distribution of a trained policy, published formula, instrumentation
    only. Mirrors q5_floor_probe.py so the two are directly comparable."""
    import numpy as np
    import train_branched_q5 as B5
    from train_branched_nyu import GAIN, RB_BW_HZ
    from env_6g_v5 import W_MOB, W_PWR, W_SPEC
    rbm = RB_BW_HZ / 1e6
    vals = []
    for seed in eval_seeds:
        rng = np.random.default_rng(seed)
        for _ in range(eval_ep):
            traj = pool[rng.integers(len(pool))]
            env = B5.BranchedEnv6G(traj, binary_power=True, guard_db=GUARD_DB,
                                   guard_mode=GUARD_MODE, array_gain_dbi=GAIN,
                                   w_mob=W_MOB, w_pwr=W_PWR, w_spec=W_SPEC)
            obs, _ = env.reset()
            d = False
            while not d:
                a = net.act(obs)
                obs, _, t, tr, info = env.step(a)
                d = t or tr
                sdb = float(obs[4]) * 100.0
                vals.append(float(info['rb_count']) * rbm
                            * math.log2(1.0 + 10.0 ** (sdb / 10.0)))
    return np.array(vals)


def evaluate(only=None):
    torch, tp, B5, RT = _imports()
    import numpy as np
    import env_6g_v5
    from train_branched_nyu import evaluate_branched, EVAL_SEEDS, EVAL_EP
    from dndqn_branched import load_branched
    from dndqn import OBS_DIM
    from q5_floor import T_FLOOR
    import tablevii_r8 as T7
    T7.install_patch(env_6g_v5)
    slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH)
    pool = slow + med
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, 'q5_campaign.csv')
    rows = list(csv.DictReader(open(path))) if os.path.exists(path) else []
    have = {(int(r['seed']), int(r['repeat'])) for r in rows}
    dpath = os.path.join(OUT, 'q5_trained_tput.csv')
    drows = list(csv.DictReader(open(dpath))) if os.path.exists(dpath) else []
    dhave = {int(r['seed']) for r in drows}

    for s in (only if only is not None else SEEDS):
        if not done(s):
            log('[eval] s%d not trained yet - skipping' % s)
            continue
        net = load_branched(ck(s), OBS_DIM)[0]
        if not all((s, r) in have for r in range(R)):
            for r in range(R):
                if (s, r) in have:
                    continue
                with T7.seeded(r):
                    m = evaluate_branched(net, pool, True, guard_db=GUARD_DB,
                                          guard_mode=GUARD_MODE)
                row = {'seed': s, 'ckpt': ck(s), 'ckpt_md5': md5(ck(s)), 'repeat': r,
                       'chan_seed_first': T7.BASE + 100000 * r, 'steps': STEPS,
                       't_floor_mbps': '%.4f' % T_FLOOR, 'guard_db': GUARD_DB,
                       'guard_mode': GUARD_MODE, 'venue': VENUE, 'torch': torch.__version__,
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
            log('[eval] s%d metrics done' % s)
        if s not in dhave:
            with T7.seeded(0):
                v = tput_percentiles(net, pool, EVAL_SEEDS, EVAL_EP)
            drow = {'seed': s, 'ckpt': ck(s), 'steps': len(v),
                    't_floor_mbps': '%.4f' % T_FLOOR,
                    'below_floor_pct': '%.4f' % (100.0 * (v < T_FLOOR).mean()),
                    'mean_mbps': '%.4f' % v.mean(), 'std_mbps': '%.4f' % v.std(),
                    'venue': VENUE,
                    'measured_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
            for p, q in zip(PCTS, np.percentile(v, PCTS)):
                drow['p%d' % p] = '%.4f' % q
            drows.append(drow)
            with open(dpath, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(drows[0].keys()))
                w.writeheader()
                w.writerows(drows)
            log('[eval] s%d tput: mean %.1f Mbps | %.1f%% of steps below the floor'
                % (s, v.mean(), 100.0 * (v < T_FLOOR).mean()))
    log('evaluation complete -> %s and %s' % (path, dpath))
    return 0


def smoke():
    # train_branched_nyu parses sys.argv[1] as an int at import time; clear argv before any import.
    sys.argv = sys.argv[:1]
    steps = int(os.environ.get('Q5_SMOKE_STEPS', 12_000))
    s = 99
    for suffix in ('', '.done', '.snap'):
        p = ck(s) + suffix
        if os.path.exists(p):
            os.remove(p)
    print('=' * 92)
    print('Q5 SMOKE | %s steps, seed %d -- does it train, and does the floor term FIRE in training?'
          % (format(steps, ','), s))
    print('=' * 92)

    # The term is only meaningful if it actually fires during training. If T_FLOOR were set below
    # everything the policy produces, the run would complete, the reward would be the published
    # reward, and Q5 would report "no effect" -- a result about the constant, not the design.
    import q5_floor
    import train_branched_q5 as _B5
    REC = []
    from train_branched_nyu import RB_BW_HZ
    _B5.BranchedEnv6G.step.__globals__['_floor_term'] = q5_floor.make_floor_term(
        _B5.BranchedEnv6G.W_FLOOR, q5_floor.T_FLOOR, RB_BW_HZ / 1e6, REC)
    train_one(s, steps=steps)

    ok = True
    if not REC:
        print('  FAIL: the floor term was never evaluated during training.')
        ok = False
    else:
        import numpy as np
        ts = np.array([x[0] for x in REC])
        ps = np.array([x[1] for x in REC])
        fired = int((ps < 0).sum())
        print('  floor term over %s training steps:' % format(len(ps), ','))
        print('     fired (below %.1f Mbps) : %s  (%.2f%%)'
              % (q5_floor.T_FLOOR, format(fired, ','), 100.0 * fired / len(ps)))
        print('     mean penalty when fired : %.6f' % (ps[ps < 0].mean() if fired else 0.0))
        print('     worst penalty           : %.6f' % ps.min())
        print('     per-step tput mean      : %.1f Mbps' % ts.mean())
        if fired == 0:
            print('  FAIL: the term never fired -- Q5 would report "no effect", which would be a '
                  'result\n        about T_FLOOR being too low, not about the design.')
            ok = False
        elif fired == len(ps):
            print('  FAIL: the term fired on EVERY step -- it has degenerated into a constant '
                  'offset\n        plus a standing push to allocate RBs.')
            ok = False
        else:
            print('  [ok] the term fires on some but not all steps -- it is an anchor, not a '
                  'constant')

    torch, tp, B5, RT = _imports()
    from dndqn_branched import load_branched
    from dndqn import OBS_DIM
    import env_6g_v5
    import tablevii_r8 as T7
    T7.install_patch(env_6g_v5)
    net, meta = load_branched(ck(s), OBS_DIM)
    if list(meta.get('branch_sizes', [])) != [3, 3]:
        print('  FAIL: branch_sizes is %s; Q5 must NOT change the action space'
              % meta.get('branch_sizes'))
        ok = False
    else:
        print('  [ok] action space unchanged at [3, 3] -- Q5 changes the reward, not the actuator')
    slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH)
    from train_branched_nyu import evaluate_branched
    with T7.seeded(0):
        m = evaluate_branched(net, slow + med, True, guard_db=GUARD_DB, guard_mode=GUARD_MODE)
    print('  [ok] evaluation runs: RLF %.2f%% Spec %.2f%% Tput %.1f Mbps'
          % (m['RLF'], m['Spec'], m['Tput']))
    print('=' * 92)
    print('  VERDICT: %s' % ('PASS - safe to launch the campaign' if ok else '*** FAIL ***'))
    print('=' * 92)
    return 0 if ok else 1


def main():
    if '--train-one' in sys.argv:
        i = sys.argv.index('--train-one')
        return train_one(int(sys.argv[i + 1]))
    if '--smoke' in sys.argv:
        return smoke()
    if '--eval' in sys.argv:
        return evaluate()

    import subprocess
    todo = [s for s in SEEDS if not done(s)]
    resumable = [s for s in todo if os.path.exists(ck(s) + '.snap')]
    log('=' * 84)
    log('Q5 THROUGHPUT FLOOR | %d of %d trainings finished | %d to go (%d mid-run, will resume)'
        % (len(SEEDS) - len(todo), len(SEEDS), len(todo), len(resumable)))
    log('~%.1f h remaining at %d slot(s). Ctrl-C is safe: rerun to continue.'
        % (len(todo) * 50.0 / 60.0 / SLOTS, SLOTS))
    log('=' * 84)
    running = []
    try:
        while todo or running:
            while todo and len(running) < SLOTS:
                s = todo.pop(0)
                lf = open('q5_fl_s%d.log' % s, 'a')
                p = subprocess.Popen([sys.executable, '-u', __file__, '--train-one', str(s)],
                                     stdout=lf, stderr=subprocess.STDOUT)
                running.append((s, p, lf))
                log('  started s%d (pid %d), %d queued' % (s, p.pid, len(todo)))
            time.sleep(15)
            for it in list(running):
                s, p, lf = it
                if p.poll() is not None:
                    lf.close()
                    running.remove(it)
                    log('  s%d exited rc=%d %s'
                        % (s, p.returncode, 'OK' if done(s) else '(will resume next run)'))
    except KeyboardInterrupt:
        log('interrupted - snapshots are on disk; rerun this script to continue')
        for s, p, lf in running:
            p.terminate()
        return 130
    log('all %d trainings complete - evaluating' % len(SEEDS))
    return evaluate()


if __name__ == '__main__':
    sys.exit(main())
