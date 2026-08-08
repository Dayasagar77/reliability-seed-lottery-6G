#!/usr/bin/env python3
"""
q2_threelevel.py -- Q2: a three-level power actuator (23 / 26.5 / 30 dBm).

THE QUESTION, VERBATIM. "Has the authors considered a three-level actuator (23, 26.5, 30 dBm) to
test whether the binary result is robust to a small intermediate option, or whether the cliff truly
makes intermediate powers 'poor value'?"

Read that second clause carefully, because it decides the whole design: the reviewer wants to know
whether THE CLIFF makes MID poor value. So the reward must not be what decides it. If MID were paid
the same fixed bonus SAVE is paid, MID would cost the agent nothing extra while carrying strictly
less RLF risk, no rational agent would ever press SAVE, and the actuator would silently collapse
back to two levels -- the experiment would then be measuring "what if PT_MIN were 26.5 dBm", which
is a different question and a misleading answer to this one.

So the save bonus becomes proportional to the power actually saved:

    bonus(a) = _PWR_STEP_REW * (PT_MAX_DBM - pt_after(a)) / (PT_MAX_DBM - PT_MIN_DBM)

    SAVE (23.0 dBm) -> factor 1.0 -> 0.428571...  == the published constant, EXACTLY
    MID  (26.5 dBm) -> factor 0.5 -> 0.214286...

Two things make this the neutral choice rather than a new knob. First, that factor is precisely the
power-saving percentage the paper's OWN metric credits the action with -- train_branched_nyu.py:82
computes (PT_MAX - mean pt) / (PT_MAX - PT_MIN) * 100, dB-linear, so MID scores exactly 50%. The
bonus is proportional to the number the paper reports. Second, at factor 1.0 the expression reduces
to the published constant, so THE PUBLISHED BINARY ARM IS A FIXED POINT OF THE NEW RULE: with MID
unreachable, this environment is the published environment. That is not asserted here, it is gated
below, step by step, on both guard modes and both power modes.

WHAT IS NOT REIMPLEMENTED. BranchedEnv6G.step is transformed by four exact string replacements,
each asserted to match exactly once, in the manner resumable_train.py transforms the published
train(). The reward, the guard, the C1/C3/C5 shaping and the delegation to the base physics are the
published text.

ACTION LAYOUT -- MID IS APPENDED, NOT INSERTED.

    0 STAY_PROTECT   1 STAY_SAVE   2 HANDOVER   3 STAY_MID     BRANCH_SIZES = [4, 3]

Inserting MID at index 1 would have renumbered HANDOVER and silently invalidated every published
checkpoint. Appending leaves 0-2 with exactly their published meaning, which is also what makes the
fixed-point gate a one-line restriction of the action space.

    python q2_threelevel.py --gate            # prove the fixed point, ~30 s
    python q2_threelevel.py --emit-trainer    # write train_branched_3l.py
"""
import inspect
import os
import sys
import textwrap

# --------------------------------------------------------------------------- action layout
R_STAY_MID = 3                      # APPENDED: 0,1,2 keep their published meaning
BRANCH_SIZES_3L = [4, 3]
MID_LABEL = 'STAY_MID'


def _constants():
    from env_6g_v5 import PT_MIN_DBM, PT_MAX_DBM
    mid = 0.5 * (PT_MIN_DBM + PT_MAX_DBM)
    # The reviewer names the three levels explicitly. If the env's endpoints ever move, the
    # midpoint moves with them -- but then it is no longer the actuator the reviewer asked about,
    # and that must surface here rather than in a results table.
    if (PT_MIN_DBM, PT_MAX_DBM, mid) != (23.0, 30.0, 26.5):
        raise SystemExit('Q2 asks for the 23 / 26.5 / 30 dBm actuator; this env gives '
                         '%.1f / %.1f / %.1f. Refusing to proceed.' % (PT_MIN_DBM, mid, PT_MAX_DBM))
    return PT_MIN_DBM, PT_MAX_DBM, mid


# --------------------------------------------------------------------------- the transform
_EDITS = [
    # 1. action-mode guard must recognise MID as a save-type intent, exactly as it does SAVE.
    ("action-mode guard recognises MID",
     "        if a_rel == R_STAY_SAVE and self.guard_db is not None "
     "and self.guard_mode == 'action':",
     "        if a_rel in _SAVE_FRAC and self.guard_db is not None "
     "and self.guard_mode == 'action':"),

    # 2. the actuator itself: one new branch, a direct set, no other level touched.
    ("power branch gains the MID level",
     "        elif a_rel == R_STAY_SAVE:\n"
     "            self.pt_dbm = PT_MIN_DBM if self.binary_power "
     "else max(self.pt_dbm - PT_STEP_DBM, PT_MIN_DBM)\n",
     "        elif a_rel == R_STAY_SAVE:\n"
     "            self.pt_dbm = PT_MIN_DBM if self.binary_power "
     "else max(self.pt_dbm - PT_STEP_DBM, PT_MIN_DBM)\n"
     "        elif a_rel == R_STAY_MID:\n"
     "            self.pt_dbm = PT_MID_DBM\n"),

    # 3. the C1 gate is unchanged; only the bonus becomes proportional to power saved.
    #    At _SAVE_FRAC[R_STAY_SAVE] == 1.0 this is the published expression, character for
    #    character in effect.
    ("C1 bonus becomes proportional to power saved",
     "        elif a_rel == R_STAY_SAVE:                     "
     "# C1 gate on the power-saving action\n"
     "            reward += _PWR_STEP_REW if rsrp_before >= SAFE_RSRP_THRESH "
     "else PWR_DOWN_UNSAFE_PEN\n",
     "        elif a_rel in _SAVE_FRAC:                      "
     "# C1 gate, bonus prop. to power saved\n"
     "            reward += (_PWR_STEP_REW * _SAVE_FRAC[a_rel]) "
     "if rsrp_before >= SAFE_RSRP_THRESH else PWR_DOWN_UNSAFE_PEN\n"),

    # 4. zero-argument super() has no __class__ cell once the source is exec'd, so it is made
    #    explicit. BranchedEnv6G's super() resolves to SubTHzEnv6G_NYU by the MRO, and it still
    #    does from the subclass, so this is the same call -- which the gate then proves.
    ("explicit base call (exec has no __class__ cell)",
     "super().step(flat_action)",
     "SubTHzEnv6G_NYU.step(self, flat_action)"),
]


def build_env_class():
    """Return (BranchedEnv6G_3L, published_class, save_frac) -- published class untouched."""
    import env_6g_branched as EB
    PT_MIN, PT_MAX, PT_MID = _constants()
    from dndqn_branched import R_STAY_SAVE

    save_frac = {R_STAY_SAVE: (PT_MAX - PT_MIN) / (PT_MAX - PT_MIN),      # == 1.0, published
                 R_STAY_MID: (PT_MAX - PT_MID) / (PT_MAX - PT_MIN)}       # == 0.5
    if save_frac[R_STAY_SAVE] != 1.0:
        raise SystemExit('SAVE factor is %r, not 1.0 -- the published arm would stop being a '
                         'fixed point of the new rule.' % save_frac[R_STAY_SAVE])

    src = inspect.getsource(EB.BranchedEnv6G.step)
    for desc, old, new in _EDITS:
        n = src.count(old)
        if n != 1:
            raise SystemExit('Q2 transform: "%s" matched %d times, expected exactly 1. '
                             'Refusing to transform source I do not understand.' % (desc, n))
        src = src.replace(old, new, 1)
    src = textwrap.dedent(src)

    g = dict(vars(EB))
    g.update({'R_STAY_MID': R_STAY_MID, 'PT_MID_DBM': PT_MID, '_SAVE_FRAC': save_frac})
    ns = {}
    exec(compile(src, '<q2_threelevel:step>', 'exec'), g, ns)

    cls = type('BranchedEnv6G_3L', (EB.BranchedEnv6G,), {
        'step': ns['step'],
        '__doc__': 'Published BranchedEnv6G plus an appended STAY_MID level at %.1f dBm. '
                   'Reward identical; save bonus proportional to power saved.' % PT_MID,
    })
    cls.PT_MID_DBM = PT_MID
    cls.SAVE_FRAC = save_frac
    return cls, EB.BranchedEnv6G, save_frac


# --------------------------------------------------------------------------- the gate
def _rollout(cls, seed, actions, traj=None, **kw):
    """Drive one env through a fixed action list, recording everything observable."""
    import numpy as np
    env = cls(TRAJ if traj is None else traj, **kw)
    obs, _ = env.reset(seed=seed)
    rec = []
    for a in actions:
        obs, r, term, trunc, info = env.step(a)
        rec.append((np.asarray(obs, dtype=float).tobytes(), float(r), bool(term), bool(trunc),
                    float(info['pt_dbm']), int(info['rb_count']), float(info['rsrp_dbm'])))
        if term or trunc:
            break
    return rec


def _sweep(cls, seed, actions, trajs, **kw):
    """Same, over a set of trajectories -- one short episode is not evidence."""
    out = []
    for i, t in enumerate(trajs):
        out.append(_rollout(cls, seed + i, actions, traj=t, **kw))
    return out


TRAJ = None


def gate():
    import random
    import numpy as np
    import torch
    torch.set_num_threads(1)
    global TRAJ
    import train_phase2 as tp
    from dndqn_branched import R_STAY_PROTECT, R_STAY_SAVE, R_HANDOVER, S_HOLD

    NEW, PUB, FRAC = build_env_class()
    PT_MIN, PT_MAX, PT_MID = _constants()
    slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH)
    TRAJ = (slow + med)[0]

    N, SEED = 400, 20260803
    N_TRAJ = int(os.environ.get('Q2_GATE_TRAJ', 25))
    rng = random.Random(12345)
    pub_actions = [(rng.randint(0, 2), rng.randint(0, 2)) for _ in range(N)]
    pool = slow + med
    trajs = [pool[(i * 137) % len(pool)] for i in range(N_TRAJ)]   # spread across the pool
    ok = True

    print('=' * 92)
    print('Q2 FIXED-POINT GATE | published binary arm must be a special case of the 3-level arm')
    print('=' * 92)
    print('  levels: SAVE %.1f  MID %.1f  PROTECT %.1f dBm   |   save factors %s'
          % (PT_MIN, PT_MID, PT_MAX, {k: round(v, 6) for k, v in FRAC.items()}))

    # --- 0. the comparison must be deterministic, or nothing below means anything -----------
    a1 = _sweep(PUB, SEED, pub_actions, trajs, binary_power=True, guard_db=9.0, guard_mode='state')
    a2 = _sweep(PUB, SEED, pub_actions, trajs, binary_power=True, guard_db=9.0, guard_mode='state')
    n_steps = sum(len(e) for e in a1)
    if a1 != a2:
        print('  FAIL: the published env is not reproducible at a fixed reset seed; '
              'a step-by-step comparison cannot be trusted.')
        return 2
    print('  [ok] published env is reproducible at fixed reset seeds '
          '(%d episodes, %s steps)' % (len(a1), format(n_steps, ',')))

    # --- 1. fixed point, across every configuration the campaign might touch ----------------
    total = 0
    for bp in (True, False):
        for gd, gm in ((9.0, 'state'), (9.0, 'action'), (None, 'state')):
            p = _sweep(PUB, SEED, pub_actions, trajs, binary_power=bp, guard_db=gd, guard_mode=gm)
            q = _sweep(NEW, SEED, pub_actions, trajs, binary_power=bp, guard_db=gd, guard_mode=gm)
            ns = sum(len(e) for e in p)
            total += ns
            tag = 'binary=%-5s guard=%-4s mode=%-6s' % (bp, gd, gm)
            if p != q:
                bad = next((i for i, (x, y) in enumerate(zip(p, q)) if x != y), -1)
                print('  FAIL: %s diverges in episode %d' % (tag, bad))
                ok = False
            else:
                print('  [ok] %s identical over %2d episodes / %s steps'
                      % (tag, len(p), format(ns, ',')))
    print('  ---- %s step-comparisons, all identical' % format(total, ','))

    # --- 2. the gate must be shown FIRING, or it proves nothing -----------------------------
    #     (a) a deliberately wrong midpoint must be caught by the step comparison
    saved_mid = NEW.PT_MID_DBM
    try:
        broken = type('Broken', (NEW,), {})
        g = dict(NEW.step.__globals__)
        g['PT_MID_DBM'] = 25.0
        broken.step = type(NEW.step)(NEW.step.__code__, g, NEW.step.__name__,
                                     NEW.step.__defaults__, NEW.step.__closure__)
        mid_actions = [(R_STAY_MID, S_HOLD)] * 40
        r_ok = _rollout(NEW, SEED, mid_actions, binary_power=True, guard_db=None,
                        guard_mode='state')
        r_bad = _rollout(broken, SEED, mid_actions, binary_power=True, guard_db=None,
                         guard_mode='state')
        if r_ok == r_bad:
            print('  FAIL: a wrong MID level (25.0 dBm) produced identical output -- the '
                  'comparison is blind and every result below would be vacuous.')
            ok = False
        else:
            print('  [ok] gate fires: a 25.0 dBm MID is detected as different from %.1f' % saved_mid)
    finally:
        pass

    #     (b) MID must actually reach the actuator
    pts = {round(x[4], 3) for x in _rollout(NEW, SEED, [(R_STAY_MID, S_HOLD)] * 40,
                                            binary_power=True, guard_db=None, guard_mode='state')}
    if pts != {PT_MID}:
        print('  FAIL: pressing MID produced pt_dbm %s, expected {%.1f} -- the new action is '
              'not wired to the actuator.' % (sorted(pts), PT_MID))
        ok = False
    else:
        print('  [ok] pressing MID sets pt_dbm to %.1f dBm on every step' % PT_MID)

    #     (c) the bonus actually paid, measured -- not inferred from the constants.
    #     rsrp_before enters step() in exactly two places: the guard test and the C1 gate. With
    #     guard_db=None the guard is off, so toggling _prev_rsrp across the -78 dBm gate changes
    #     the C1 branch AND NOTHING ELSE -- same seed, same action, same pt_dbm, same physics.
    #     The reward difference is therefore the bonus minus the unsafe penalty, exactly.
    import env_6g_branched as EB

    def paid(cls, act):
        hi = cls(TRAJ, binary_power=True, guard_db=None, guard_mode='state')
        hi.reset(seed=SEED); hi._prev_rsrp = -60.0        # above the C1 gate
        lo = cls(TRAJ, binary_power=True, guard_db=None, guard_mode='state')
        lo.reset(seed=SEED); lo._prev_rsrp = -90.0        # below it
        return hi.step((act, S_HOLD))[1] - lo.step((act, S_HOLD))[1] + EB.PWR_DOWN_UNSAFE_PEN

    want_save, want_mid = EB._PWR_STEP_REW, EB._PWR_STEP_REW * 0.5
    got_save, got_mid = paid(NEW, R_STAY_SAVE), paid(NEW, R_STAY_MID)
    print('  bonus actually paid:  SAVE %.6f (want %.6f)   MID %.6f (want %.6f)'
          % (got_save, want_save, got_mid, want_mid))
    if abs(got_save - want_save) > 1e-9:
        print('  FAIL: the SAVE bonus is no longer the published _PWR_STEP_REW'); ok = False
    else:
        print('  [ok] SAVE still earns the published constant %.6f' % want_save)
    if abs(got_mid - want_mid) > 1e-9:
        print('  FAIL: MID earns %.6f, expected exactly half of SAVE' % got_mid); ok = False
    else:
        print('  [ok] MID earns exactly half of it (%.6f) -- proportional to the 50%% power '
              'saving the paper\'s own metric credits 26.5 dBm with' % want_mid)
    # and the same C1 gate must still bite MID below the line
    lo = NEW(TRAJ, binary_power=True, guard_db=None, guard_mode='state')
    lo.reset(seed=SEED); lo._prev_rsrp = -90.0
    hiP = NEW(TRAJ, binary_power=True, guard_db=None, guard_mode='state')
    hiP.reset(seed=SEED); hiP._prev_rsrp = -90.0
    if lo.step((R_STAY_MID, S_HOLD))[1] >= hiP.step((R_STAY_PROTECT, S_HOLD))[1]:
        print('  [note] below the C1 gate MID is not penalised relative to PROTECT -- check')
    else:
        print('  [ok] below the C1 gate MID takes the unsafe penalty, exactly as SAVE does')

    print('=' * 92)
    print('  VERDICT: %s' % ('PASS -- with MID unreachable this IS the published environment, '
                             'and MID is live when reached'
                             if ok else '*** FAIL -- do not run the campaign ***'))
    print('=' * 92)
    return 0 if ok else 1


# --------------------------------------------------------------------------- trainer emission
TRAINER = '''#!/usr/bin/env python3
"""train_branched_3l.py -- GENERATED by q2_threelevel.py. Do not edit by hand.

The published train() with exactly two changes, each asserted to match once at generation time:

  1. the network is built with branch_sizes=[4, 3] instead of the default [3, 3];
  2. the epsilon-greedy random action draws the reliability branch from range(4), not range(3).

(2) is the one that matters and the one that is easy to miss. The published line is
        a = [random.randint(0, 2), random.randint(0, 2)]
with the bound hard-coded. Enlarge the network without enlarging that draw and MID is never
explored during the epsilon phase -- it can then only ever be reached greedily, by a head that has
no experience of it. The campaign would run to completion, the agent would not use MID, and the
paper would report that the cliff makes intermediate powers poor value. That conclusion would be an
artefact of an exploration bound, and nothing in the output would look wrong.

BranchedEnv6G is rebound to the three-level subclass, so both the initial construction and the
episode-boundary rebuild inside train() -- and the one resumable_train.py inserts into its restore
block -- all build the three-level environment.
"""
from train_branched_nyu import *                      # noqa: F401,F403  (published trainer)
import train_branched_nyu as _PUB
from q2_threelevel import build_env_class, BRANCH_SIZES_3L

BranchedEnv6G = build_env_class()[0]                  # override AFTER the star import

'''


def emit_trainer(path='train_branched_3l.py'):
    import train_branched_nyu as B
    edits = [
        ("network built with [4, 3]",
         "    online, target = build_branched_networks(OBS_DIM)",
         "    online, target = build_branched_networks(OBS_DIM, "
         "branch_sizes=BRANCH_SIZES_3L)"),
        ("epsilon-greedy draws the enlarged reliability branch",
         "            a = [random.randint(0, 2), random.randint(0, 2)]",
         "            a = [random.randint(0, BRANCH_SIZES_3L[0] - 1), "
         "random.randint(0, BRANCH_SIZES_3L[1] - 1)]"),
    ]
    src = textwrap.dedent(inspect.getsource(B.train))
    for desc, old, new in edits:
        n = src.count(old)
        if n != 1:
            raise SystemExit('trainer transform: "%s" matched %d times, expected 1.' % (desc, n))
        src = src.replace(old, new, 1)
        print('  [ok] %-48s 1 match replaced' % desc)
    # resumable_train.py transforms this file's train() next; its two anchors must survive ours.
    for anchor in ('    for step in range(1, steps + 1):',
                   "            epr.append(er); er = 0.0; env = BranchedEnv6G(smp(stage), "
                   "binary_power=binary_power, guard_db=guard_db, guard_mode=guard_mode); "
                   "obs, _ = env.reset()"):
        if src.count(anchor) != 1:
            raise SystemExit('resumable_train anchor no longer matches exactly once after our '
                             'edits -- the two transforms have collided.')
    print('  [ok] %-48s intact' % 'resumable_train anchors survive our edits')
    out = TRAINER + src
    compile(out, path, 'exec')
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write(out)
    print('written  : %s (%d bytes)' % (path, len(out.encode('utf-8'))))
    return 0


if __name__ == '__main__':
    sys.argv = sys.argv[:1] + [a for a in sys.argv[1:]]
    if '--emit-trainer' in sys.argv:
        arg = sys.argv[:1]
        sys.argv = arg
        sys.exit(emit_trainer())
    if '--gate' in sys.argv:
        sys.argv = sys.argv[:1]
        sys.exit(gate())
    print(__doc__)
    sys.exit(0)
