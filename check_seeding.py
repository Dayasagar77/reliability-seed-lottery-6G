#!/usr/bin/env python3
"""
check_seeding.py — proves, natively, that the evaluation channel is deterministic on both
paths policy_fig_data.py uses, so that script needs no constructor seed.

Why this exists: the 20260804 fixer inserted `seed=` into the env CONSTRUCTOR of the other
scripts. policy_fig_data.py cannot take that treatment and does not need it:

  * it seeds through reset() instead — env_6g_v5.py:304 rebuilds self.rng from the reset seed,
    and every fading / shadow / start-position draw happens at or after that point
    (env_6g_v5.py:212, :319 and env_6g_nyu.py:59, :70). __init__ only *creates* the generator.
  * BranchedEnv6G.__init__ (env_6g_branched.py:35) takes no `seed` argument and has no **kw,
    so BranchedEnv6G(..., seed=...) would raise TypeError.

    python check_seeding.py        # expects: ALL PATHS DETERMINISTIC
"""
import numpy as np
from env_6g_nyu import SubTHzEnv6G_NYU
from env_6g_branched import BranchedEnv6G
from env_6g_v5 import W_MOB, W_PWR, W_SPEC
import train_phase2 as tp

GAIN = 15.0
slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH)
POOL = slow + med
traj = POOL[0]


def rollout(make_env, seed, action):
    env = make_env()
    obs, _ = env.reset(seed=seed)
    trace, done = [], False
    for _ in range(60):
        if done:
            break
        obs, r, t, tr, info = env.step(action)
        done = t or tr
        trace.append((round(float(obs[0]), 12), round(float(info['rsrp_dbm']), 12), round(float(r), 12)))
    return trace


CASES = [
    ("SubTHzEnv6G_NYU  (run_flat path)",
     lambda: SubTHzEnv6G_NYU(traj, array_gain_dbi=GAIN, w_mob=W_MOB, w_pwr=W_PWR, w_spec=W_SPEC), 0),
    ("BranchedEnv6G    (run_branched / greedy path)",
     lambda: BranchedEnv6G(traj, binary_power=True, guard_db=9.0, guard_mode='state',
                           array_gain_dbi=GAIN, w_mob=W_MOB, w_pwr=W_PWR, w_spec=W_SPEC), (0, 0)),
]

ok = True
for name, mk, act in CASES:
    a = rollout(mk, 4242, act)
    b = rollout(mk, 4242, act)          # same seed -> must be identical
    c = rollout(mk, 9999, act)          # different seed -> must differ (proves the seed is live)
    same, differs = (a == b), (a != c)
    ok &= same and differs
    print(f"{name:46s} repeatable={same}  seed-sensitive={differs}  steps={len(a)}")

print()
print("ALL PATHS DETERMINISTIC" if ok else "*** NOT DETERMINISTIC — do not publish these figures ***")

# and the negative control: the constructor kwarg the other 13 scripts use is not available here
try:
    BranchedEnv6G(traj, binary_power=True, array_gain_dbi=GAIN, seed=0)
    print("NOTE: BranchedEnv6G accepted seed= — the note in policy_fig_data.py needs revisiting")
except TypeError as e:
    print(f"confirmed: BranchedEnv6G(..., seed=...) -> TypeError ({e})")
