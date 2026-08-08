"""Oracle-nearest handover policy under the NEW channel: hand over whenever the nearest gNB
changes, else HOLD. Shows the ACHIEVABLE coverage floor (handover working as intended), to
confirm the scenario is coherent (handover rescues coverage) and to pick the array gain."""
import numpy as np
import multi_seed_eval as mse
from env_6g_nyu import SubTHzEnv6G_NYU
from env_6g_v5 import HOLD, HANDOVER, A3_THRESH_DBM, RLF_THRESH_DBM

slow, med = mse.load_trajectories(mse.CSV_PATH); pool = slow + med
rng = np.random.default_rng(0)
sample = [pool[i] for i in rng.integers(0, len(pool), size=300)]

for G in [10, 15, 20]:
    ep_len, ba3, brlf, ns, rlft, nep = [], 0, 0, 0, 0, 0
    for traj in sample:
        env = SubTHzEnv6G_NYU(traj, seed=0, array_gain_dbi=G)
        env.reset(seed=int(rng.integers(1e9)))
        done, steps = False, 0
        while not done:
            a = HANDOVER if env._get_nearest_gnb() != env.serving_gnb_idx else HOLD
            obs, r, term, trunc, info = env.step(a)
            steps += 1; ns += 1
            if info['rsrp_dbm'] < A3_THRESH_DBM:  ba3  += 1
            if info['rsrp_dbm'] < RLF_THRESH_DBM: brlf += 1
            done = term or trunc
            if done and info.get('sustained_rlf'): rlft += 1
        ep_len.append(steps); nep += 1
    print(f"G={G:>2} dBi | mean ep-len {np.mean(ep_len):5.1f} | %steps<A3 {100*ba3/ns:5.1f} | "
          f"%steps<RLF {100*brlf/ns:4.1f} | sustained-RLF term {100*rlft/nep:5.1f}% of episodes")
print("\nGood scenario = oracle handover keeps most episodes alive (low RLF term) while the "
      "passive-HOLD baseline failed — i.e., handover is both necessary and sufficient.")
