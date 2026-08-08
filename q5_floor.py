#!/usr/bin/env python3
"""
q5_floor.py -- Q5: a simple throughput-floor reward term, to test whether the spectrum axis can be
anchored.

THE QUESTION, VERBATIM. "The spectrum axis is described as 'unanchored' and its anchoring as future
work. Has a simple throughput-floor reward term been piloted? Even a negative result (e.g., it
re-introduces reliability sensitivity) would be informative and would bound the design space."

THE TERM. A one-sided hinge on the per-step throughput -- zero above the floor, growing linearly
with the proportional shortfall below it:

    r_floor = -W_FLOOR * max(0, (T_FLOOR - T_step) / T_FLOOR)

It is a PENALTY ONLY. It never pays for throughput above the floor, because a two-sided term would
be a different mechanism -- a throughput objective rather than an anchor -- and the paper is
explicit that throughput is a consequence of RB allocation, not something the controller maximises
(the reviewer makes the same point in m5).

WHERE THE TWO CONSTANTS COME FROM. Both are measured or matched, not chosen for convenience.

  T_FLOOR = 239.9026 Mbps. This is the p33 of the pooled per-step throughput of the PUBLISHED
  guarded arm, measured by q5_floor_probe.py over 906,868 steps (5 seeds x R=4) and banked in
  results/q5floor/q5_floor_choice.csv. p25 was the first choice and was rejected on the evidence:
  the pooled CDF jumps from 81.2 Mbps at p20 to 203.8 at p25, because that is exactly where the
  seed mixture crosses over -- seed 1 runs at ~900 Mbps mean while seeds 0 and 2 run at ~300. A
  floor sitting on that discontinuity would move substantially with a different seed pool. p33
  sits in a smooth region (239.9 -> 311.4 at p50) and still fires on about a third of steps.

  W_FLOOR = _RB_STEP_REW = 0.428571. Matched to the existing C5 RB-down bonus, so the new term is
  the same order as the shaping already in the reward -- neither ignorable nor dominant. At the
  1st percentile of observed throughput (19.1 Mbps) the penalty is 0.394; at the floor it is 0.

THROUGHPUT IS THE PUBLISHED QUANTITY. train_branched_nyu.py:76-77 computes it as
    sdb = obs[4] * 100.0 ;  rb_count * (RB_BW_HZ/1e6) * log2(1 + 10**(sdb/10))
and that expression is what the term uses, so the floor is a floor on exactly the Tput the paper
reports -- not on a lookalike.

WHAT IS NOT REIMPLEMENTED. BranchedEnv6G.step is transformed by two exact string replacements,
each asserted to match exactly once. The action space is UNCHANGED, so unlike Q2 the trainer needs
no edit at all -- only its BranchedEnv6G binding is swapped.

    python q5_floor.py --gate            # prove the fixed point at W_FLOOR = 0, ~40 s
    python q5_floor.py --emit-trainer    # write train_branched_q5.py
"""
import inspect
import math
import os
import sys
import textwrap

T_FLOOR = float(os.environ.get('Q5_T_FLOOR', 239.9026))     # Mbps, p33, measured
FLOOR_BASIS = 'p33 of pooled per-step throughput, published guarded arm, 5 seeds x R=4, ' \
              '906,868 steps (results/q5floor/q5_floor_choice.csv)'

_EDITS = [
    ("throughput-floor term added after C3",
     "        if terminated and info.get('sustained_rlf'):   # C3 terminal penalty\n"
     "            reward += RLF_TERMINAL_PENALTY\n"
     "        return obs, reward, terminated, truncated, info\n",
     "        if terminated and info.get('sustained_rlf'):   # C3 terminal penalty\n"
     "            reward += RLF_TERMINAL_PENALTY\n"
     "        reward += _floor_term(obs, info)               # Q5: throughput-floor anchor\n"
     "        return obs, reward, terminated, truncated, info\n"),

    ("explicit base call (exec has no __class__ cell)",
     "super().step(flat_action)",
     "SubTHzEnv6G_NYU.step(self, flat_action)"),
]


def make_floor_term(w_floor, t_floor, rb_bw_mhz, record=None):
    """Return the hinge. Penalty only: the value is always <= 0."""
    def _floor_term(obs, info):
        sdb = float(obs[4]) * 100.0
        t = float(info['rb_count']) * rb_bw_mhz * math.log2(1.0 + 10.0 ** (sdb / 10.0))
        pen = -w_floor * max(0.0, (t_floor - t) / t_floor)
        if record is not None:
            record.append((t, pen))
        return pen
    return _floor_term


def build_env_class(w_floor=None, record=None):
    """Return (BranchedEnv6G_Q5, published_class, w_floor). Published class untouched.

    w_floor=0.0 makes the term identically zero, which is the fixed point the gate exercises."""
    import env_6g_branched as EB
    from train_branched_nyu import RB_BW_HZ
    w = EB._RB_STEP_REW if w_floor is None else float(w_floor)

    src = inspect.getsource(EB.BranchedEnv6G.step)
    for desc, old, new in _EDITS:
        n = src.count(old)
        if n != 1:
            raise SystemExit('Q5 transform: "%s" matched %d times, expected exactly 1. '
                             'Refusing to transform source I do not understand.' % (desc, n))
        src = src.replace(old, new, 1)
    src = textwrap.dedent(src)

    g = dict(vars(EB))
    g['_floor_term'] = make_floor_term(w, T_FLOOR, RB_BW_HZ / 1e6, record)
    ns = {}
    exec(compile(src, '<q5_floor:step>', 'exec'), g, ns)

    cls = type('BranchedEnv6G_Q5', (EB.BranchedEnv6G,), {
        'step': ns['step'],
        '__doc__': 'Published BranchedEnv6G plus a one-sided throughput-floor penalty at '
                   '%.4f Mbps, weight %.6f.' % (T_FLOOR, w),
    })
    cls.T_FLOOR, cls.W_FLOOR = T_FLOOR, w
    return cls, EB.BranchedEnv6G, w


# --------------------------------------------------------------------------- the gate
def _rollout(cls, seed, actions, traj, **kw):
    import numpy as np
    env = cls(traj, **kw)
    obs, _ = env.reset(seed=seed)
    rec = []
    for a in actions:
        obs, r, term, trunc, info = env.step(a)
        rec.append((np.asarray(obs, dtype=float).tobytes(), float(r), bool(term), bool(trunc),
                    float(info['pt_dbm']), int(info['rb_count'])))
        if term or trunc:
            break
    return rec


def _sweep(cls, seed, actions, trajs, **kw):
    return [_rollout(cls, seed + i, actions, t, **kw) for i, t in enumerate(trajs)]


def gate():
    import random
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import train_phase2 as tp
    import env_6g_branched as EB
    from train_branched_nyu import RB_BW_HZ

    slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH)
    pool = slow + med
    N, SEED = 400, 20260803
    N_TRAJ = int(os.environ.get('Q5_GATE_TRAJ', 25))
    rng = random.Random(12345)
    acts = [(rng.randint(0, 2), rng.randint(0, 2)) for _ in range(N)]
    trajs = [pool[(i * 137) % len(pool)] for i in range(N_TRAJ)]
    ok = True

    print('=' * 92)
    print('Q5 FIXED-POINT GATE | at W_FLOOR = 0 the term must vanish and leave the published env')
    print('=' * 92)
    print('  T_FLOOR = %.4f Mbps   (%s)' % (T_FLOOR, FLOOR_BASIS))
    print('  W_FLOOR = %.6f      (matched to the C5 RB-down bonus _RB_STEP_REW)'
          % EB._RB_STEP_REW)

    ZERO, PUB, _ = build_env_class(w_floor=0.0)
    LIVE_REC = []
    LIVE, _, w = build_env_class(record=LIVE_REC)

    # --- 0. determinism, or nothing below means anything ------------------------------------
    a1 = _sweep(PUB, SEED, acts, trajs, binary_power=True, guard_db=9.0, guard_mode='state')
    a2 = _sweep(PUB, SEED, acts, trajs, binary_power=True, guard_db=9.0, guard_mode='state')
    if a1 != a2:
        print('  FAIL: the published env is not reproducible at a fixed reset seed.')
        return 2
    print('  [ok] published env reproducible (%d episodes, %s steps)'
          % (len(a1), format(sum(len(e) for e in a1), ',')))

    # --- 1. fixed point at W_FLOOR = 0 -------------------------------------------------------
    total = 0
    for bp in (True, False):
        for gd, gm in ((9.0, 'state'), (9.0, 'action'), (None, 'state')):
            p = _sweep(PUB, SEED, acts, trajs, binary_power=bp, guard_db=gd, guard_mode=gm)
            q = _sweep(ZERO, SEED, acts, trajs, binary_power=bp, guard_db=gd, guard_mode=gm)
            ns = sum(len(e) for e in p)
            total += ns
            tag = 'binary=%-5s guard=%-4s mode=%-6s' % (bp, gd, gm)
            if p != q:
                print('  FAIL: %s diverges at W_FLOOR = 0' % tag)
                ok = False
            else:
                print('  [ok] %s identical over %2d episodes / %s steps'
                      % (tag, len(p), format(ns, ',')))
    print('  ---- %s step-comparisons at W_FLOOR = 0, all identical' % format(total, ','))

    # --- 2. the gate must be shown FIRING ----------------------------------------------------
    LIVE_REC.clear()
    live = _sweep(LIVE, SEED, acts, trajs, binary_power=True, guard_db=9.0, guard_mode='state')
    base = _sweep(ZERO, SEED, acts, trajs, binary_power=True, guard_db=9.0, guard_mode='state')
    if live == base:
        print('  FAIL: at W_FLOOR = %.6f the environment is STILL identical to the published one '
              '-- the term is not reaching the reward and every Q5 result would be vacuous.' % w)
        ok = False
    else:
        print('  [ok] gate fires: at W_FLOOR = %.6f the reward differs from the published env' % w)

    # --- 3. the term itself, recomputed independently ----------------------------------------
    if not LIVE_REC:
        print('  FAIL: the term was never evaluated.')
        ok = False
    else:
        ts = np.array([x[0] for x in LIVE_REC])
        ps = np.array([x[1] for x in LIVE_REC])
        want = -w * np.maximum(0.0, (T_FLOOR - ts) / T_FLOOR)
        if np.max(np.abs(ps - want)) > 1e-12:
            print('  FAIL: the penalty does not match the hinge formula (max err %.3g)'
                  % np.max(np.abs(ps - want)))
            ok = False
        else:
            print('  [ok] penalty matches -W*max(0,(T_floor-T)/T_floor) on all %s evaluations'
                  % format(len(ps), ','))
        if ps.max() > 0.0:
            print('  FAIL: the term paid a POSITIVE reward on some step -- it must be a penalty '
                  'only, never a throughput bonus.')
            ok = False
        else:
            print('  [ok] the term is penalty-only (max %.6f, min %.6f)' % (ps.max(), ps.min()))
        fired = float((ts < T_FLOOR).mean())
        print('  [i]  fired on %.1f%% of these random-action steps (the design target was ~33%% '
              'under the TRAINED policy; random actions are not the trained policy)' % (100 * fired))

    print('=' * 92)
    print('  VERDICT: %s' % ('PASS -- at W_FLOOR = 0 this IS the published environment, and the '
                             'term is live and correct when switched on'
                             if ok else '*** FAIL -- do not run the campaign ***'))
    print('=' * 92)
    return 0 if ok else 1


# --------------------------------------------------------------------------- trainer emission
TRAINER = '''#!/usr/bin/env python3
"""train_branched_q5.py -- GENERATED by q5_floor.py. Do not edit by hand.

The published train(), VERBATIM -- Q5 changes the reward, not the action space, so unlike Q2 there
is no trainer edit at all. The only difference is the binding: BranchedEnv6G resolves to the
throughput-floor subclass, so the initial construction, the episode-boundary rebuild inside
train(), and the rebuild resumable_train.py inserts into its restore block all build the Q5
environment.
"""
from train_branched_nyu import *                      # noqa: F401,F403  (published trainer)
import train_branched_nyu as _PUB
from q5_floor import build_env_class, T_FLOOR

BranchedEnv6G = build_env_class()[0]                  # override AFTER the star import

'''


def emit_trainer(path='train_branched_q5.py'):
    import train_branched_nyu as B
    src = textwrap.dedent(inspect.getsource(B.train))
    pub = src
    for anchor in ('    for step in range(1, steps + 1):',
                   "            epr.append(er); er = 0.0; env = BranchedEnv6G(smp(stage), "
                   "binary_power=binary_power, guard_db=guard_db, guard_mode=guard_mode); "
                   "obs, _ = env.reset()"):
        if src.count(anchor) != 1:
            raise SystemExit('resumable_train anchor does not match exactly once.')
    print('  [ok] %-48s intact' % 'resumable_train anchors')
    if src != pub:
        raise SystemExit('the trainer source was modified; Q5 must not modify it.')
    print('  [ok] %-48s verbatim' % 'published train() copied unmodified')
    out = TRAINER + src
    compile(out, path, 'exec')
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write(out)
    print('written  : %s (%d bytes)' % (path, len(out.encode('utf-8'))))
    return 0


if __name__ == '__main__':
    if '--emit-trainer' in sys.argv:
        sys.argv = sys.argv[:1]
        sys.exit(emit_trainer())
    if '--gate' in sys.argv:
        sys.argv = sys.argv[:1]
        sys.exit(gate())
    print(__doc__)
    sys.exit(0)
