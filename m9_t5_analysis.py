#!/usr/bin/env python3
"""
m9_t5_analysis.py -- the last two cheap items from review 2.

M9. "zero ping-pong events are observed across all evaluated trajectories" is a strong claim; the
reviewer asks for a quantitative confirmation, e.g. the distribution of time between handovers.
The paper already reports 2.02-2.05 handovers per episode; this adds the distribution.

T5 / Q4. The oracle budget counts SAVE-able STEPS. The reviewer points out that if those steps
cluster, the achievable average power saving under a causal policy differs from the step fraction,
so the 84.51 % figure and the achieved 63.48 % are not directly comparable. This measures the
run-length structure the oracle figure conceals.

Both are replays of frozen behaviour -- no training, no new claims about the controller.
"""
import sys
import time
from collections import Counter

import numpy as np

sys.argv = sys.argv[:1]
import torch                                                            # noqa: E402
torch.set_num_threads(1)

import env_6g_v5                                                        # noqa: E402
import train_phase2 as tp                                               # noqa: E402
import tablevii_r8 as T7                                                # noqa: E402
from dndqn_branched import load_branched                                # noqa: E402
from dndqn import OBS_DIM                                               # noqa: E402
T7.install_patch(env_6g_v5)

import env_6g_branched as EB                                            # noqa: E402

PT_MAX, PT_MIN = 30.0, 23.0
RLF_THRESH = -92.0
G = 9.0
EPISODES = 300


def log(m):
    print(time.strftime('%H:%M:%S') + ' | ' + m, flush=True)


slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH)
pool = slow + med

# ---------------------------------------------------------------- M9: handover spacing
log('M9 -- replaying the guarded controller for handover timing')
net, _ = load_branched('br_gst_s0.pt', OBS_DIM)
net.eval()
gaps, per_ep, ho_total, steps_total = [], [], 0, 0
with T7.seeded(0):
    rng = np.random.default_rng(0)
    for _ in range(EPISODES):
        traj = pool[rng.integers(len(pool))]
        env = EB.BranchedEnv6G(traj, binary_power=True, guard_db=G, guard_mode='state')
        obs, _ = env.reset()
        done, k, last_ho, n_ho = False, 0, None, 0
        while not done:
            with torch.no_grad():
                a = [int(q.argmax(dim=-1).item())
                     for q in net(torch.FloatTensor(obs).unsqueeze(0))]
            obs, _, t, tr, info = env.step(a)
            done = t or tr
            k += 1
            if info.get('handover_done'):
                n_ho += 1
                if last_ho is not None:
                    gaps.append(k - last_ho)
                last_ho = k
        per_ep.append(n_ho)
        ho_total += n_ho
        steps_total += k

g = np.array(gaps) if gaps else np.array([0])
print()
print('=' * 92)
print('M9 -- HANDOVER SPACING (guarded controller, %d episodes, %s steps)'
      % (EPISODES, f'{steps_total:,}'))
print('=' * 92)
print('  handovers per episode : mean %.2f   distribution %s'
      % (np.mean(per_ep), dict(sorted(Counter(per_ep).items()))))
print('  inter-handover gap    : n = %d   min %d   p05 %.0f   median %.0f   mean %.1f   max %d'
      % (len(g), g.min(), np.percentile(g, 5), np.median(g), g.mean(), g.max()))
print('  at ~100 ms per step, the minimum observed gap is %.1f s' % (g.min() * 0.1))
short = int((g <= 10).sum())
print('  gaps <= 1 s (10 steps): %d of %d (%.2f%%)' % (short, len(g), 100.0 * short / max(len(g), 1)))
print('  A ping-pong event would appear as a gap of a few steps. The empirical minimum above is')
print('  the quantitative form of the "zero ping-pong" claim.')

# ---------------------------------------------------------------- T5: SAVE-able run lengths
log('T5 -- replaying the protective baseline for SAVE-able run structure')
runs, saveable, total = [], 0, 0
with T7.seeded(0):
    rng = np.random.default_rng(0)
    for _ in range(EPISODES):
        traj = pool[rng.integers(len(pool))]
        env = EB.BranchedEnv6G(traj, binary_power=True, guard_db=G, guard_mode='state')
        obs, _ = env.reset()
        done, cur = False, 0
        while not done:
            obs, _, t, tr, info = env.step([0, 1])       # PROTECT, hold RB -> always Pmax
            done = t or tr
            rsrp = float(obs[4]) * 100.0
            ok = rsrp - (PT_MAX - PT_MIN) - G >= RLF_THRESH
            total += 1
            if ok:
                saveable += 1
                cur += 1
            else:
                if cur:
                    runs.append(cur)
                cur = 0
        if cur:
            runs.append(cur)

r = np.array(runs) if runs else np.array([0])
print()
print('=' * 92)
print('T5 / Q4 -- TEMPORAL STRUCTURE OF SAVE-ABLE STEPS (G = %.0f dB)' % G)
print('=' * 92)
print('  SAVE-able fraction of steps : %.2f%%  (%s of %s)'
      % (100.0 * saveable / max(total, 1), f'{saveable:,}', f'{total:,}'))
print('  contiguous runs             : n = %d   median %d   mean %.1f   p90 %.0f   max %d'
      % (len(r), int(np.median(r)), r.mean(), np.percentile(r, 90), r.max()))
print('  runs of 1 step only         : %d (%.1f%%)'
      % (int((r == 1).sum()), 100.0 * (r == 1).sum() / max(len(r), 1)))
print('  share of SAVE-able steps in runs >= 10 steps (>= 1 s): %.1f%%'
      % (100.0 * r[r >= 10].sum() / max(r.sum(), 1)))
print()
print('  A causal controller can only exploit a run it can detect and hold. Long runs are')
print('  exploitable; isolated single steps largely are not, which is part of the gap between')
print('  the oracle bound and the achieved figure -- alongside non-causality.')
print('=' * 92)
