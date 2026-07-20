#!/usr/bin/env python3
"""oracle_power_budget_nyu.py — POLICY-FREE upper bound on zero-reliability-cost power saving
at 142 GHz (the 'power-saving budget' the physics allows).

Method: drive the A3 rule at FIXED PT_MAX (protective baseline) over the NYU CI channel,
log the per-step served RSRP. Since RSRP scales dB-for-dB with transmit power, the link would
have survived that same step at PT_MIN iff  rsrp - (PT_MAX-PT_MIN) >= RLF_THRESH + guard.
The fraction of such steps = the maximum average power saving achievable by an ORACLE that
toggles PROTECT/SAVE per step with perfect knowledge (normalized savings metric = SAVE-step
fraction). Guard sweep (0/3/6/9 dB) shows robustness to fading uncertainty.
HONESTY: this is an upper bound (oracle foresight, per-step toggling); a learned controller
achieves some fraction of it. Same protocol as headline (5 seeds x 150 ep).
-> results/oracle_power_budget_nyu.csv"""
import csv, os, numpy as np
from env_6g_nyu import SubTHzEnv6G_NYU
from env_6g_v5 import (W_MOB, W_PWR, W_SPEC, HANDOVER, HOLD, A3_THRESH_DBM,
                       PT_MIN_DBM, PT_MAX_DBM, RLF_THRESH_DBM)
import multi_seed_eval as mse

SEEDS, N_EP, GAIN = mse.SEEDS, mse.N_EPISODES, 15.0
DELTA = PT_MAX_DBM - PT_MIN_DBM      # 7 dB
GUARDS = [0.0, 3.0, 6.0, 9.0]
slow, med = mse.load_trajectories(mse.CSV_PATH); pool = slow + med

all_rsrp = []
for seed in SEEDS:
    rng = np.random.default_rng(seed)
    for _ in range(N_EP):
        traj = pool[rng.integers(len(pool))]
        env = SubTHzEnv6G_NYU(traj, array_gain_dbi=GAIN, w_mob=W_MOB, w_pwr=W_PWR, w_spec=W_SPEC)
        obs, _ = env.reset(); env.pt_dbm = PT_MAX_DBM
        done = False; last = None
        while not done:
            a = HOLD if last is None else (HANDOVER if last['rsrp_dbm'] < A3_THRESH_DBM else HOLD)
            obs, _, t, tr, info = env.step(a); done = t or tr; last = info
            env.pt_dbm = PT_MAX_DBM                      # hold protective power fixed
            all_rsrp.append(info['rsrp_dbm'])

r = np.asarray(all_rsrp)
print(f"ORACLE POWER-SAVING BUDGET — NYU CI 142 GHz | {len(r):,} steps | A3 rule @ PT_MAX={PT_MAX_DBM:.0f} dBm")
print(f"  (a step is SAVE-able if rsrp - {DELTA:.0f} dB >= {RLF_THRESH_DBM:.0f} dBm + guard)")
rows = []
for g in GUARDS:
    frac = float(np.mean(r - DELTA >= RLF_THRESH_DBM + g)) * 100.0
    rows.append({'guard_db': g, 'saveable_step_pct': f"{frac:.2f}"})
    print(f"  guard {g:>3.0f} dB -> {frac:6.2f}% of steps SAVE-able  (= max avg power saving at zero reliability cost)")
os.makedirs('results', exist_ok=True)
with open('results/oracle_power_budget_nyu.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['guard_db', 'saveable_step_pct']); w.writeheader(); w.writerows(rows)
print("-> results/oracle_power_budget_nyu.csv")
