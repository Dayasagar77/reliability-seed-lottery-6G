#!/usr/bin/env python3
"""
diag_reliability.py — what actually drives in-coverage RLF?

Tests, on the SAME trajectories/seeds as rlf_metrics.py:
  (A) trained Dn-DQN (baseline)
  (B) trained model but transmit power FORCED to a fixed level (23/26/29/30 dBm)
      -> isolates the RLF floor achievable by power alone.
  (C) deterministic-mean-RSRP guard: force PWR_UP whenever the MEAN (unfaded)
      serving RSRP margin < guard -> proactive, not reactive to a single fade.
Reports in-coverage RLF% + power-saving% for each, so we can see whether
reliability is reachable via power (then a reliability variant works) or is
dominated by the alpha-mu fading params (then calibration is the real lever).
"""
import numpy as np, torch
import multi_seed_eval as mse
from env_6g_v5 import (SubTHzEnv6G, RLF_THRESH_DBM, GNB_X, CELL_RADIUS,
                       W_MOB, W_PWR, W_SPEC, PT_MIN_DBM, PT_MAX_DBM, RB_MIN, RB_MAX,
                       PWR_UP, PWR_DOWN, HOLD)
from dndqn import load_dndqn_checkpoint

CSV, SEEDS, N_EP = mse.CSV_PATH, mse.SEEDS, mse.N_EPISODES
GNB = np.asarray(GNB_X, float)
slow, med = mse.load_trajectories(CSV); pool = slow + med
net, _ = load_dndqn_checkpoint('dqn_phase2_dndqn.pt')

def run(mode, param=None):
    good = tot = clean = incov = 0
    pwr_ep = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        for _ in range(N_EP):
            traj = pool[rng.integers(len(pool))]; T = len(traj)
            env = SubTHzEnv6G(traj, w_mob=W_MOB, w_pwr=W_PWR, w_spec=W_SPEC)
            obs, _ = env.reset(); done = False; info = None; pts = []
            while not done:
                with torch.no_grad():
                    a = int(net(torch.FloatTensor(obs)).argmax().item())
                if mode == 'fixed_pwr':
                    # override so realized power stays at param dBm
                    if env.pt_dbm < param:   a = PWR_UP
                    elif env.pt_dbm > param: a = PWR_DOWN
                    elif a in (PWR_UP, PWR_DOWN): a = HOLD
                elif mode == 'det_guard':
                    d = env._get_distance()
                    det = env._compute_rsrp(d, apply_fading_flag=False)[0]
                    if (det - RLF_THRESH_DBM) < param and env.pt_dbm < PT_MAX_DBM:
                        if a in (PWR_DOWN, HOLD): a = PWR_UP
                obs, _, term, trunc, info = env.step(a); done = term or trunc
                pts.append(env.pt_dbm)
                if info.get('handover_done'):
                    tot += 1
                    if not info.get('too_early') and not info.get('too_late'): good += 1
            pwr_ep.append((PT_MAX_DBM - np.mean(pts))/(PT_MAX_DBM-PT_MIN_DBM)*100)
            if info['rsrp_dbm'] < RLF_THRESH_DBM:
                idx = min(int(info['step']), T-1); x = float(traj[idx,0])
                if float(np.min(np.abs(GNB-x))) <= CELL_RADIUS: incov += 1
            else:
                clean += 1
    n = clean + incov
    return dict(ho=100*good/max(tot,1), pwr=float(np.mean(pwr_ep)),
                incov=incov, pct=100*incov/750)

print("mode                         HO%   Power%   in-covRLF   RLF%")
print("-"*62)
r=run('baseline'); print(f"baseline Dn-DQN           {r['ho']:6.1f} {r['pwr']:7.1f} {r['incov']:9d} {r['pct']:6.1f}")
for P in (30.0, 29.0, 26.0, 23.0):
    r=run('fixed_pwr',P); print(f"fixed power = {P:.0f} dBm         {r['ho']:6.1f} {r['pwr']:7.1f} {r['incov']:9d} {r['pct']:6.1f}")
for M in (15.0, 18.0, 22.0):
    r=run('det_guard',M); print(f"det-margin guard >= {M:.0f} dB     {r['ho']:6.1f} {r['pwr']:7.1f} {r['incov']:9d} {r['pct']:6.1f}")
