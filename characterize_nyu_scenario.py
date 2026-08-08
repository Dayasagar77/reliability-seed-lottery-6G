"""
characterize_nyu_scenario.py — is the NEW channel a COHERENT, non-degenerate scenario to
retrain in? Stress-test coverage under a passive policy (HOLD, max power, no handover) with
the FULL stochastic channel (CI + per-association shadow + alpha-mu small scale), for a few
array-gain values. Reports episode length, coverage vs thresholds, and sustained-RLF rate.
A good scenario: LOS mostly served, NLOS genuinely challenging, but not instant-death.
"""
import numpy as np
import multi_seed_eval as mse
from env_6g_nyu import SubTHzEnv6G_NYU
from env_6g_v5 import HOLD, A3_THRESH_DBM, RLF_THRESH_DBM

slow, med = mse.load_trajectories(mse.CSV_PATH)
pool = slow + med
rng = np.random.default_rng(0)
sample = [pool[i] for i in rng.integers(0, len(pool), size=300)]

for G in [5, 10, 15, 20]:
    ep_len, below_a3, below_rlf, n_steps, rlf_term, n_ep = [], 0, 0, 0, 0, 0
    for traj in sample:
        env = SubTHzEnv6G_NYU(traj, seed=0, array_gain_dbi=G)
        obs, _ = env.reset(seed=int(rng.integers(1e9)))
        done, steps = False, 0
        while not done:
            obs, r, term, trunc, info = env.step(HOLD)   # passive: never handover, max power
            steps += 1
            n_steps += 1
            if info['rsrp_dbm'] < A3_THRESH_DBM:  below_a3  += 1
            if info['rsrp_dbm'] < RLF_THRESH_DBM: below_rlf += 1
            done = term or trunc
            if done and info.get('sustained_rlf'): rlf_term += 1
        ep_len.append(steps); n_ep += 1
    print(f"G={G:>2} dBi | mean ep-len {np.mean(ep_len):5.1f} | "
          f"%steps<A3 {100*below_a3/n_steps:5.1f} | %steps<RLF {100*below_rlf/n_steps:4.1f} | "
          f"sustained-RLF termination {100*rlf_term/n_ep:5.1f}% of episodes")
print("\n(Passive HOLD, max power, no handover — this is the WORST case; a trained agent "
      "that hands over proactively should do much better. Non-degenerate = episodes survive "
      "long enough and handover clearly helps.)")
