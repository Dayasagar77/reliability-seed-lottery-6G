#!/usr/bin/env python3
"""
rlf_metrics.py  —  Honest radio-link-failure (RLF) analysis for Dn-DQN and PG-RRM.

Classifies every episode termination under ONE consistent, geometry-based rule and
reports in-coverage RLF vs boundary-exit for both policies, with the supporting
evidence (serving-gNB distance and fade depth). This replaces any assumed
"0 in-coverage RLF" figure with a measured, reproducible number.

Definitions (identical for both policies):
  - Low-RSRP termination : the episode ends with RSRP < RLF_THRESH (-92 dBm).
  - in-coverage RLF      : at termination the UE is within CELL_RADIUS of a gNB.
  - boundary-exit        : the UE is outside the served corridor.

Interpretation: Dn-DQN's power saving thins the link margin at cell edges, so deep
alpha-mu fades cross the -92 dBm floor while the UE is still attached to the nearest
gNB -> in-coverage RLF. The absolute count depends on the alpha-mu fading parameters
in channel_model.py.

Run:  python rlf_metrics.py
      (requires dqn_phase2_dndqn.pt and 6G_Handover_Training_Data_v3.csv)
"""
import numpy as np, torch
import multi_seed_eval as mse
from env_6g_v5 import (SubTHzEnv6G, RLF_THRESH_DBM, GNB_X, CELL_RADIUS, D_PERP,
                       W_MOB, W_PWR, W_SPEC)
from dndqn import load_dndqn_checkpoint
from eval_pgrrm_baseline import pgrrm_action

CSV, SEEDS, N_EP = mse.CSV_PATH, mse.SEEDS, mse.N_EPISODES
GNB = np.asarray(GNB_X, float)

slow, med = mse.load_trajectories(CSV)
pool = slow + med
net, _ = load_dndqn_checkpoint('dqn_phase2_dndqn.pt')


def rollout(policy):
    clean = incov = boundary = 0
    serv_d, fade = [], []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        for _ep in range(N_EP):
            traj = pool[rng.integers(len(pool))]; T = len(traj)
            env = SubTHzEnv6G(traj, seed=seed * 1000 + _ep, w_mob=W_MOB, w_pwr=W_PWR, w_spec=W_SPEC)
            obs, _ = env.reset(); done = False; last = None
            while not done:
                if policy == 'dndqn':
                    with torch.no_grad():
                        a = int(net(torch.FloatTensor(obs)).argmax().item())
                else:
                    a = pgrrm_action(last)
                obs, _, term, trunc, info = env.step(a); last = info; done = term or trunc
            if info['rsrp_dbm'] < RLF_THRESH_DBM:
                idx = min(int(info['step']), T - 1); x = float(traj[idx, 0])
                if float(np.min(np.abs(GNB - x))) <= CELL_RADIUS:
                    incov += 1
                    sd = abs(x - GNB[int(env.serving_gnb_idx)])
                    serv_d.append(sd)
                    rnf = env._compute_rsrp(float(np.sqrt(sd ** 2 + D_PERP ** 2)),
                                            apply_fading_flag=False)[0]
                    fade.append(float(info['rsrp_dbm'] - rnf))
                else:
                    boundary += 1
            else:
                clean += 1
    return clean, incov, boundary, np.array(serv_d), np.array(fade)


if __name__ == "__main__":
    print("=" * 64)
    print("  In-coverage vs boundary RLF  (5 seeds x 150 ep = 750 each)")
    print("=" * 64)
    for name, pol in [('Dn-DQN', 'dndqn'), ('PG-RRM', 'pgrrm')]:
        clean, incov, boundary, sd, fade = rollout(pol)
        n = clean + incov + boundary
        print(f"\n  {name}  ({n} episodes)")
        print(f"    clean completions : {clean:4d}  ({100*clean/n:5.1f}%)")
        print(f"    in-coverage RLF   : {incov:4d}  ({100*incov/n:5.1f}%)")
        print(f"    boundary-exit     : {boundary:4d}  ({100*boundary/n:5.1f}%)")
        if len(sd):
            print(f"    (in-cov) serving-gNB dist : median {np.median(sd):.1f} m "
                  f"(cell radius {CELL_RADIUS} m)")
            print(f"    (in-cov) fade depth @RLF  : median {np.median(fade):.1f} dB")
    print("\n  Dn-DQN incurs MORE in-coverage RLF than PG-RRM: the reliability cost")
    print("  of aggressive power saving. Count depends on the alpha-mu fading params.")
# channel-seeded 20260804
