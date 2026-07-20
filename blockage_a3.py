#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""blockage_a3.py — 3GPP A3 reference rule (always P_max, full RBs, reactive HO at -80 dBm)
under the SAME real truck-blockage conditions as blockage_eval.py. Gives the physics floor:
if A3 degrades comparably, the blockage degradation is channel-inherent, not guard-specific.
-> appends A3 rows to results/blockage_a3.csv
"""
import sys, csv
import numpy as np, pandas as pd, torch
torch.set_num_threads(1)
_ARGV = list(sys.argv); sys.argv = [sys.argv[0]]
from dndqn_branched import R_STAY_PROTECT, R_HANDOVER, S_HOLD
from env_6g_v5 import W_MOB, W_PWR, W_SPEC, PT_MIN_DBM, PT_MAX_DBM, RB_MIN, RB_MAX
import multi_seed_eval as mse
from blockage_eval import BlockedBranchedEnv, load_pool
sys.argv = _ARGV

GAIN = 15.0

def a3_action(obs):
    margin = float(obs[1]) * 100.0                 # margin_norm = (rsrp - A3)/100
    return (R_HANDOVER if margin < 0.0 else R_STAY_PROTECT, S_HOLD)

def eval_a3(pool, block_db, nep):
    su, rl = [], []
    for seed in mse.SEEDS:
        r = np.random.default_rng(seed); real = good = n = rlf = 0
        for ep in range(nep):
            traj = pool[r.integers(len(pool))]
            env = BlockedBranchedEnv(traj, block_db=block_db, binary_power=True, guard_db=None,
                                     array_gain_dbi=GAIN, w_mob=W_MOB, w_pwr=W_PWR, w_spec=W_SPEC)
            obs, _ = env.reset(seed=10_000 * seed + ep); done = False; n += 1; info = {}
            while not done:
                a = a3_action(obs); pg = env.serving_gnb_idx
                obs, _, t, tr_, info = env.step(a); done = t or tr_
                if env.serving_gnb_idx != pg:
                    real += 1
                    if not info.get('too_early') and not info.get('too_late'): good += 1
            if info.get('sustained_rlf'): rlf += 1
        su.append(100 * good / max(real, 1)); rl.append(100 * rlf / n)
    return dict(HO=float(np.mean(su)), HOsd=float(np.std(su)), RLF=float(np.mean(rl)), RLFsd=float(np.std(rl)))

if __name__ == '__main__':
    nep = int(sys.argv[1]) if len(sys.argv) > 1 else mse.N_EPISODES
    slow, med, nfast = load_pool(); pool = slow + med
    print(f"A3 under blockage | pool={len(pool)}", flush=True)
    rows = []
    for bdb in (0.0, 10.0, 15.0, 20.0, 25.0, 30.0):
        m = eval_a3(pool, bdb, nep)
        rows.append([bdb, f"{m['RLF']:.3f}", f"{m['RLFsd']:.3f}", f"{m['HO']:.3f}", f"{m['HOsd']:.3f}"])
        print(f"  A3 @ {bdb:4.0f} dB blockage: RLF {m['RLF']:5.2f}±{m['RLFsd']:4.2f}  HO {m['HO']:5.2f}±{m['HOsd']:4.2f}", flush=True)
    with open('results/blockage_a3.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['block_db', 'RLF_mean', 'RLF_std', 'HO_mean', 'HO_std']); w.writerows(rows)
    print("  -> results/blockage_a3.csv", flush=True)
