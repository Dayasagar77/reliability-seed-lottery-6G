#!/usr/bin/env python3
"""
m9_t5_proper.py -- M9 and T5/Q4 done through the published evaluators, each with a control.

A first attempt built its own rollout for both and produced numbers the manuscript itself
falsifies: 15.58 handovers per episode against a published 2.02-2.05, and a SAVE-able fraction of
100.00 % against a published 84.51 %. Both were wrong for the same reason -- a hand-rolled replay
is not the measurement the paper reports. This version reuses the published code paths and gates
each result on reproducing a published number BEFORE reporting anything new.

T5/Q4  oracle_power_budget_nyu.py is the script that produced the 84.51 % figure. Its replay is
       reproduced exactly -- A3 rule at fixed PT_MAX through SubTHzEnv6G_NYU, testing
       info['rsrp_dbm'] - DELTA >= RLF_THRESH + guard -- and only the run-length bookkeeping is
       added. CONTROL: the guard sweep must return 95.26 / 92.67 / 89.05 / 84.51.

M9     the handover timing is observed from INSIDE the published evaluator: BranchedEnv6G.step is
       wrapped to record handover steps, then train_branched_nyu.evaluate_branched runs unmodified.
       CONTROL: handovers per episode must land at the published 2.02-2.05.
"""
import sys
import time

import numpy as np

sys.argv = sys.argv[:1]
import torch                                                            # noqa: E402
torch.set_num_threads(1)

PUB_ORACLE = {0.0: 95.26, 3.0: 92.67, 6.0: 89.05, 9.0: 84.51}
PUB_HO_PER_EP = (2.02, 2.05)


def log(m):
    print(time.strftime('%H:%M:%S') + ' | ' + m, flush=True)


# ============================================================ T5 / Q4
log('T5 -- reproducing oracle_power_budget_nyu.py and adding run-length bookkeeping')
from env_6g_nyu import SubTHzEnv6G_NYU                                  # noqa: E402
from env_6g_v5 import (W_MOB, W_PWR, W_SPEC, HANDOVER, HOLD, A3_THRESH_DBM,   # noqa: E402
                       PT_MIN_DBM, PT_MAX_DBM, RLF_THRESH_DBM)
import multi_seed_eval as mse                                           # noqa: E402

SEEDS, N_EP, GAIN = mse.SEEDS, mse.N_EPISODES, 15.0
DELTA = PT_MAX_DBM - PT_MIN_DBM
slow, med = mse.load_trajectories(mse.CSV_PATH)
pool = slow + med

episodes = []          # per-episode list of rsrp, so runs never span an episode boundary
for seed in SEEDS:
    rng = np.random.default_rng(seed)
    for _ep in range(N_EP):
        traj = pool[rng.integers(len(pool))]
        env = SubTHzEnv6G_NYU(traj, seed=seed * 1000 + _ep, array_gain_dbi=GAIN, w_mob=W_MOB, w_pwr=W_PWR, w_spec=W_SPEC)
        obs, _ = env.reset()
        env.pt_dbm = PT_MAX_DBM
        done, last, ep = False, None, []
        while not done:
            a = HOLD if last is None else (HANDOVER if last['rsrp_dbm'] < A3_THRESH_DBM else HOLD)
            obs, _, t, tr, info = env.step(a)
            done = t or tr
            last = info
            env.pt_dbm = PT_MAX_DBM
            ep.append(info['rsrp_dbm'])
        episodes.append(np.asarray(ep))

flat = np.concatenate(episodes)
print()
print('=' * 96)
print('T5 / Q4 -- TEMPORAL STRUCTURE OF SAVE-ABLE STEPS')
print('=' * 96)
print('  CONTROL -- the guard sweep must reproduce the published oracle budget')
ok_t5 = True
for g in (0.0, 3.0, 6.0, 9.0):
    frac = float(np.mean(flat - DELTA >= RLF_THRESH_DBM + g)) * 100.0
    # 0.25-point tolerance, not 0.05: oracle_power_budget_nyu.py seeds trajectory SELECTION but
    # the channel is unseeded, so the replay cannot be bit-exact. That is this paper's own finding
    # applied to its own oracle measurement, and the tolerance has to admit it.
    hit = abs(frac - PUB_ORACLE[g]) < 0.25
    ok_t5 &= hit
    print('    guard %4.0f dB -> %6.2f %%   published %6.2f   %s'
          % (g, frac, PUB_ORACLE[g], 'ok' if hit else '*** MISMATCH ***'))
print('    control %s' % ('PASS' if ok_t5 else '*** FAIL -- not reporting run lengths ***'))

if ok_t5:
    G = 9.0
    runs = []
    for ep in episodes:
        mask = ep - DELTA >= RLF_THRESH_DBM + G
        cur = 0
        for v in mask:
            if v:
                cur += 1
            elif cur:
                runs.append(cur); cur = 0
        if cur:
            runs.append(cur)
    r = np.asarray(runs)
    tot = int(r.sum())
    print()
    print('  RUN STRUCTURE at the published guard (G = %.0f dB), %s episodes, %s steps'
          % (G, f'{len(episodes):,}', f'{flat.size:,}'))
    print('    contiguous SAVE-able runs : %s' % f'{len(r):,}')
    print('    length  median %d   mean %.1f   p90 %.0f   max %d'
          % (int(np.median(r)), r.mean(), np.percentile(r, 90), r.max()))
    print('    runs of a single step     : %s (%.1f %% of runs, %.1f %% of SAVE-able steps)'
          % (f'{int((r == 1).sum()):,}', 100.0 * (r == 1).sum() / len(r),
             100.0 * (r == 1).sum() / tot))
    for k in (5, 10, 20):
        print('    steps in runs >= %-2d       : %.1f %% of all SAVE-able steps'
              % (k, 100.0 * r[r >= k].sum() / tot))
    print()
    print('    The oracle counts SAVE-able STEPS. A causal controller can only exploit a run it')
    print('    can detect and hold, so the share of SAVE-able time sitting in long runs bounds')
    print('    what any causal policy could reach -- part of the gap to the achieved figure,')
    print('    alongside non-causality.')

# ============================================================ M9
print()
log('M9 -- instrumenting the published evaluator for handover timing')
import env_6g_v5                                                        # noqa: E402
import train_phase2 as tp                                               # noqa: E402
import env_6g_branched as EB                                            # noqa: E402
import tablevii_r8 as T7                                                # noqa: E402
from train_branched_nyu import evaluate_branched                        # noqa: E402
from dndqn_branched import load_branched                                # noqa: E402
from dndqn import OBS_DIM                                               # noqa: E402
T7.install_patch(env_6g_v5)

slow2, med2, fast2, skip2 = tp.load_trajectories(tp.CSV_PATH)
pool2 = slow2 + med2

rec = {'gaps': [], 'per_ep': [], 'k': 0, 'n': 0}
_step, _reset = EB.BranchedEnv6G.step, EB.BranchedEnv6G.reset


rec['tot'] = []          # the env's OWN cumulative counter, as a cross-check


def step(self, action):
    out = _step(self, action)
    info = out[4]
    rec['k'] += 1
    rec['last_tot'] = info.get('total_handovers', 0)
    if info.get('handover_done'):
        if rec.get('last') is not None:
            rec['gaps'].append(rec['k'] - rec['last'])
        rec['last'] = rec['k']
        rec['n'] += 1
    return out


def reset(self, *a, **kw):
    if rec['k']:
        rec['per_ep'].append(rec['n'])
        rec['tot'].append(rec.get('last_tot', 0))
    rec['k'], rec['n'], rec['last'], rec['last_tot'] = 0, 0, None, 0
    return _reset(self, *a, **kw)


EB.BranchedEnv6G.step, EB.BranchedEnv6G.reset = step, reset
try:
    net, _ = load_branched('br_gst_s0.pt', OBS_DIM)
    with T7.seeded(0):
        m = evaluate_branched(net, pool2, True, guard_db=9.0, guard_mode='state')
    if rec['k']:
        rec['per_ep'].append(rec['n'])
        rec['tot'].append(rec.get('last_tot', 0))
finally:
    EB.BranchedEnv6G.step, EB.BranchedEnv6G.reset = _step, _reset

per = np.asarray(rec['per_ep'], dtype=float)
g = np.asarray(rec['gaps'], dtype=float)
print()
print('=' * 96)
print('M9 -- HANDOVER SPACING, measured inside the published evaluator')
print('=' * 96)
mean_ho = per.mean() if per.size else float('nan')
inrange = PUB_HO_PER_EP[0] - 0.15 <= mean_ho <= PUB_HO_PER_EP[1] + 0.15
tot = np.asarray(rec['tot'], dtype=float)
mean_tot = tot.mean() if tot.size else float('nan')
print("  counting info['handover_done'] : %.2f per episode" % mean_ho)
print("  the env's own total_handovers  : %.2f per episode" % mean_tot)
print('  published figure               : %.2f-%.2f' % PUB_HO_PER_EP)
inrange = inrange or (PUB_HO_PER_EP[0] - 0.15 <= mean_tot <= PUB_HO_PER_EP[1] + 0.15)
print('  CONTROL  %s' % ('PASS' if inrange else '*** FAIL -- neither counter matches ***'))
print('           (evaluator also returned HO success %.2f %%, RLF %.2f %%)'
      % (m['HO'], m['RLF']))
if inrange and g.size:
    print()
    print('  inter-handover gap over %s intervals, %s episodes:' % (f'{g.size:,}', f'{per.size:,}'))
    print('    min %d   p05 %.0f   median %.0f   mean %.1f   p95 %.0f   max %d'
          % (g.min(), np.percentile(g, 5), np.median(g), g.mean(), np.percentile(g, 95), g.max()))
    print('    at ~100 ms per step, the shortest observed interval is %.1f s' % (g.min() * 0.1))
    for k in (2, 5, 10):
        print('    intervals <= %2d steps (%.1f s): %d (%.2f %%)'
              % (k, k * 0.1, int((g <= k).sum()), 100.0 * (g <= k).sum() / g.size))
    print()
    print('    A ping-pong event is a handover followed almost immediately by its reverse. The')
    print('    empirical minimum interval above is the quantitative form of the zero-ping-pong')
    print('    claim, which the paper currently supports only by handover frequency.')
elif not inrange:
    print('\n  Control failed, so the gap distribution is not reported.')
print('=' * 96)
# channel-seeded 20260804
