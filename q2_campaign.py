#!/usr/bin/env python3
"""
q2_campaign.py -- Q2: train and evaluate the three-level actuator, resumably.

    python -u q2_campaign.py --smoke     # ~4 min: does it train, does MID get used at all?
    python -u q2_campaign.py             # 5 trainings x 8e5 steps, then the R=8 CRN grid
    python -u q2_campaign.py --eval      # evaluation only, for checkpoints already finished

WHAT THIS ANSWERS. The reviewer asks whether the binary result is robust to a small intermediate
option, "or whether the cliff truly makes intermediate powers 'poor value'". The aggregate metrics
answer the first half. The second half is answered directly, by the ACTION DISTRIBUTION: if trained
policies are given a 26.5 dBm option, on a neutral bonus schedule, and decline to use it, that is
the cliff speaking. So every evaluation here also records how often each reliability action is
chosen. A table of RLF alone would leave the reviewer's actual question unanswered.

COMPARATOR. The published binary guarded arm is not retrained. Its R=8 CRN grid is already banked
on this machine in results/tablevii_r8_laptop.csv (arm='guarded', venue='laptop'), measured under
the same harness at the same channel seeds, so the comparison is within-venue. The provenance
difference that remains -- those checkpoints were trained earlier, on an earlier toolchain -- is
disclosed in the result document; it is not resolvable by re-running, because training is not
reproducible run-to-run, which is this revision's main finding.

RESUME. Same discipline as laptop_gsweep.py: snapshots at episode boundaries via resumable_train,
and a checkpoint counts as finished only when its .done marker exists. A bare .pt is a mid-run
best-so-far save; accepting one would put an undertrained controller in the table with nothing
looking wrong.
"""
import csv
import hashlib
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

SEEDS = [0, 1, 2, 3, 4]
STEPS = int(os.environ.get('Q2_STEPS', 800_000))
SNAP_EVERY = int(os.environ.get('Q2_SNAP', 20_000))
SLOTS = int(os.environ.get('Q2_SLOTS', 2))
GUARD_DB, GUARD_MODE = 9.0, 'state'
R = 8
OUT = os.path.join('results', 'q2_3level')
METRICS = ['HO', 'Power', 'Spec', 'RLF', 'Tput']
VENUE = os.environ.get('Q2_VENUE', 'laptop')


def log(m):
    print(time.strftime('%H:%M:%S') + ' | ' + m, flush=True)


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def ck(s):
    return 'q2_3l_s%d.pt' % s


def done(s):
    return os.path.exists(ck(s)) and os.path.exists(ck(s) + '.done')


def _imports():
    sys.argv = [sys.argv[0]]
    import torch
    torch.set_num_threads(1)
    import train_phase2 as tp
    import train_branched_3l as B3
    import resumable_train as RT
    return torch, tp, B3, RT


def train_one(s, steps=None):
    steps = steps or STEPS
    torch, tp, B3, RT = _imports()
    slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH)
    log('[q2 s%d] training %s steps (three-level actuator, guard %.1f dB, %s mode)'
        % (s, format(steps, ','), GUARD_DB, GUARD_MODE))
    t0 = time.time()
    RT.train_resumable(B3, s, True, ck(s), slow, med, steps=steps, guard_db=GUARD_DB,
                       guard_mode=GUARD_MODE, snapshot=ck(s) + '.snap', snap_every=SNAP_EVERY)
    with open(ck(s) + '.done', 'w') as f:
        f.write('%s %d %s\n' % (md5(ck(s)), steps,
                                time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    log('[q2 s%d] DONE in %.1f min' % (s, (time.time() - t0) / 60))
    return 0


# --------------------------------------------------------------------- action-distribution probe
def action_counts(net, pool, eval_seeds, eval_ep, n_actions):
    """Count reliability-branch choices of the GREEDY policy over the evaluation pool.

    This is instrumentation, not a second evaluation: it drives the same env with the same net and
    counts what the net asks for. It records the INTENT (what the policy pressed), which is what
    the reviewer's question is about -- the state guard may override the power afterwards, and how
    often it does so is recorded separately."""
    import numpy as np
    import train_branched_3l as B3
    from train_branched_nyu import GAIN
    from env_6g_v5 import W_MOB, W_PWR, W_SPEC
    cnt = np.zeros(n_actions, dtype=np.int64)
    overridden = 0
    total = 0
    for seed in eval_seeds:
        rng = np.random.default_rng(seed)
        for _ in range(eval_ep):
            traj = pool[rng.integers(len(pool))]
            env = B3.BranchedEnv6G(traj, binary_power=True, guard_db=GUARD_DB,
                                   guard_mode=GUARD_MODE, array_gain_dbi=GAIN,
                                   w_mob=W_MOB, w_pwr=W_PWR, w_spec=W_SPEC)
            obs, _ = env.reset()
            done_ = False
            while not done_:
                a = net.act(obs)
                intent = int(a[0])
                obs, _, t, tr, info = env.step(a)
                done_ = t or tr
                cnt[intent] += 1
                total += 1
                # the guard restores PT_MAX regardless of what was pressed
                from dndqn_branched import R_STAY_PROTECT
                if intent != R_STAY_PROTECT and float(info['pt_dbm']) >= 29.999:
                    overridden += 1
    return cnt, overridden, total


def evaluate(only=None):
    torch, tp, B3, RT = _imports()
    import numpy as np
    import env_6g_v5
    import train_phase2 as tp2
    from train_branched_nyu import evaluate_branched, EVAL_SEEDS, EVAL_EP
    from dndqn_branched import load_branched
    from dndqn import OBS_DIM
    import tablevii_r8 as T7
    from q2_threelevel import BRANCH_SIZES_3L
    T7.install_patch(env_6g_v5)
    slow, med, fast, skip = tp2.load_trajectories(tp2.CSV_PATH)
    pool = slow + med
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, 'q2_3level.csv')
    rows = list(csv.DictReader(open(path))) if os.path.exists(path) else []
    have = {(int(r['seed']), int(r['repeat'])) for r in rows}
    apath = os.path.join(OUT, 'q2_actions.csv')
    arows = list(csv.DictReader(open(apath))) if os.path.exists(apath) else []
    ahave = {int(r['seed']) for r in arows}

    for s in (only if only is not None else SEEDS):
        if not done(s):
            log('[eval] s%d not trained yet - skipping' % s)
            continue
        net, meta = load_branched(ck(s), OBS_DIM)
        bs = list(meta.get('branch_sizes', []))
        if bs != BRANCH_SIZES_3L:
            log('*** s%d checkpoint carries branch_sizes=%s, expected %s. REFUSING to evaluate a '
                'checkpoint that is not the three-level controller. ***' % (s, bs, BRANCH_SIZES_3L))
            return 3
        if not all((s, r) in have for r in range(R)):
            for r in range(R):
                if (s, r) in have:
                    continue
                with T7.seeded(r):
                    m = evaluate_branched(net, pool, True, guard_db=GUARD_DB,
                                          guard_mode=GUARD_MODE)
                row = {'seed': s, 'ckpt': ck(s), 'ckpt_md5': md5(ck(s)), 'repeat': r,
                       'chan_seed_first': T7.BASE + 100000 * r, 'steps': STEPS,
                       'branch_sizes': 'x'.join(map(str, bs)), 'guard_db': GUARD_DB,
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
        if s not in ahave:
            with T7.seeded(0):
                cnt, ovr, tot = action_counts(net, pool, EVAL_SEEDS, EVAL_EP, bs[0])
            arow = {'seed': s, 'ckpt': ck(s), 'steps_counted': tot,
                    'guard_overrode_intent': ovr, 'venue': VENUE,
                    'measured_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
            for i, c in enumerate(cnt):
                arow['a%d' % i] = int(c)
            arows.append(arow)
            with open(apath, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(arows[0].keys()))
                w.writeheader()
                w.writerows(arows)
            log('[eval] s%d action intent: PROTECT %d SAVE %d HANDOVER %d MID %d (of %s)'
                % (s, cnt[0], cnt[1], cnt[2], cnt[3] if len(cnt) > 3 else -1, format(tot, ',')))
    log('evaluation complete -> %s and %s' % (path, apath))
    return 0


# --------------------------------------------------------------------------------- smoke test
def smoke():
    # train_branched_nyu parses sys.argv[1] as an int AT IMPORT TIME (N_SEEDS), so any flag left
    # on argv turns the first import of it into a ValueError. Every other entry point here imports
    # via _imports(), which clears argv first; this one imports earlier, so it must clear it too.
    sys.argv = sys.argv[:1]
    steps = int(os.environ.get('Q2_SMOKE_STEPS', 15_000))
    s = 99
    for suffix in ('', '.done', '.snap'):
        p = ck(s) + suffix
        if os.path.exists(p):
            os.remove(p)
    print('=' * 92)
    print('Q2 SMOKE | %s steps, seed %d -- does it train, and is MID actually EXPLORED?'
          % (format(steps, ','), s))
    print('=' * 92)

    # ---- the check that matters ------------------------------------------------------------
    # The published trainer draws its epsilon-greedy action with a hard-coded bound:
    #     a = [random.randint(0, 2), random.randint(0, 2)]
    # Enlarge the network to four outputs without enlarging that draw and MID is NEVER explored.
    # Training would complete, the agent would not use MID, and the paper would report that the
    # cliff makes intermediate powers poor value -- an artefact of an exploration bound, with
    # nothing in the output looking wrong. So count what the trainer actually presses.
    import train_branched_3l as _B3
    _cls = _B3.BranchedEnv6G
    _orig_step = _cls.step
    TRAIN_COUNT = [0, 0, 0, 0]

    def _counting_step(self, action):
        TRAIN_COUNT[int(action[0])] += 1
        return _orig_step(self, action)
    _cls.step = _counting_step
    try:
        train_one(s, steps=steps)
    finally:
        _cls.step = _orig_step

    _tot = sum(TRAIN_COUNT)
    lbl0 = ['PROTECT', 'SAVE', 'HANDOVER', 'MID']
    print('  reliability actions PRESSED DURING TRAINING (%s steps):' % format(_tot, ','))
    for i, c in enumerate(TRAIN_COUNT):
        print('     %-9s %8s  %6.2f%%' % (lbl0[i], format(c, ','), 100.0 * c / max(_tot, 1)))
    ok0 = True
    if TRAIN_COUNT[3] == 0:
        print('  FAIL: MID was pressed ZERO times in training. The epsilon-greedy draw is not '
              'covering\n        the enlarged branch -- the campaign would be vacuous.')
        ok0 = False
    else:
        print('  [ok] MID is explored during training (%s presses) -- the enlarged epsilon-greedy '
              'draw works' % format(TRAIN_COUNT[3], ','))

    torch, tp, B3, RT = _imports()
    from dndqn_branched import load_branched
    from dndqn import OBS_DIM
    from q2_threelevel import BRANCH_SIZES_3L
    import env_6g_v5
    import tablevii_r8 as T7
    T7.install_patch(env_6g_v5)
    net, meta = load_branched(ck(s), OBS_DIM)
    bs = list(meta.get('branch_sizes', []))
    ok = ok0
    if bs != BRANCH_SIZES_3L:
        print('  FAIL: checkpoint carries branch_sizes=%s, expected %s' % (bs, BRANCH_SIZES_3L))
        ok = False
    else:
        print('  [ok] checkpoint carries branch_sizes=%s' % bs)
    qs = net(torch.randn(2, OBS_DIM))
    if [tuple(q.shape) for q in qs] != [(2, 4), (2, 3)]:
        print('  FAIL: head shapes %s' % [tuple(q.shape) for q in qs]); ok = False
    else:
        print('  [ok] head shapes are [(2, 4), (2, 3)] -- the reliability head has four outputs')

    slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH)
    pool = slow + med
    from train_branched_nyu import evaluate_branched
    with T7.seeded(0):
        m = evaluate_branched(net, pool, True, guard_db=GUARD_DB, guard_mode=GUARD_MODE)
    print('  [ok] evaluation runs on the 4-action net: RLF %.2f%% Spec %.2f%% Pwr %.2f%%'
          % (m['RLF'], m['Spec'], m['Power']))

    with T7.seeded(0):
        cnt, ovr, tot = action_counts(net, pool, [0], 20, bs[0])
    lbl = ['PROTECT', 'SAVE', 'HANDOVER', 'MID']
    print('  GREEDY action intent over %s steps (informational only):' % format(tot, ','))
    for i, c in enumerate(cnt):
        print('     %-9s %8s  %6.2f%%' % (lbl[i], format(int(c), ','), 100.0 * c / max(tot, 1)))
    print('     guard overrode a non-PROTECT intent on %s steps (%.2f%%)'
          % (format(ovr, ','), 100.0 * ovr / max(tot, 1)))
    print('  NOTE: at %s steps the greedy policy is degenerate and usually collapses onto one\n'
          '        action. That says nothing about MID and is NOT a pass criterion here -- the\n'
          '        criterion is the training-time exploration count above.' % format(steps, ','))
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
    log('Q2 THREE-LEVEL | %d of %d trainings finished | %d to go (%d mid-run, will resume)'
        % (len(SEEDS) - len(todo), len(SEEDS), len(todo), len(resumable)))
    log('~%.1f h remaining at %d slot(s). Ctrl-C is safe: rerun to continue.'
        % (len(todo) * 50.0 / 60.0 / SLOTS, SLOTS))
    log('=' * 84)
    running = []
    try:
        while todo or running:
            while todo and len(running) < SLOTS:
                s = todo.pop(0)
                lf = open('q2_3l_s%d.log' % s, 'a')
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
