#!/usr/bin/env python3
"""
reliability_shield_eval.py — Runtime minimum-RSRP power guard on the trained Dn-DQN.

A deployable O-RAN safety layer (runtime shield): the trained policy is used as-is,
but whenever the (delayed, faded) observed serving RSRP falls below a guard level
RSRP_GUARD, the shield refuses to lower transmit power and instead rebuilds margin
(PWR_UP). HANDOVER and RB actions are left untouched, so mobility and spectrum
behaviour are preserved. Sweeping RSRP_GUARD traces the RLF <-> efficiency frontier.

Metrics per guard (5 seeds x 150 ep = 750, identical trajectory sampling to
rlf_metrics.py / multi_seed_eval.py):
  - HO success (per-HO-event)   : good events / total events
  - Power saving %              : (PT_MAX - mean pt)/(PT_MAX-PT_MIN)*100
  - Spectrum saving %           : (RB_MAX - mean rb)/(RB_MAX-RB_MIN)*100
  - in-coverage RLF             : same geometry rule as rlf_metrics.py

Run: python reliability_shield_eval.py
"""
import numpy as np, torch
import multi_seed_eval as mse
from env_6g_v5 import (SubTHzEnv6G, RLF_THRESH_DBM, GNB_X, CELL_RADIUS,
                       W_MOB, W_PWR, W_SPEC, PT_MIN_DBM, PT_MAX_DBM, RB_MIN, RB_MAX,
                       PWR_UP, PWR_DOWN, HOLD, HANDOVER)
from dndqn import load_dndqn_checkpoint

CSV, SEEDS, N_EP = mse.CSV_PATH, mse.SEEDS, mse.N_EPISODES
GNB = np.asarray(GNB_X, float)
slow, med = mse.load_trajectories(CSV)
pool = slow + med
net, _ = load_dndqn_checkpoint('dqn_phase2_dndqn.pt')

# guard = None  -> no shield (baseline Dn-DQN). Otherwise force margin >= guard.
GUARDS = [None, -88.0, -85.0, -82.0, -80.0, -78.0, -75.0]


def shielded_action(obs, pt_dbm, guard):
    with torch.no_grad():
        a = int(net(torch.FloatTensor(obs)).argmax().item())
    if guard is None:
        return a
    rsrp = float(obs[0]) * 100.0                 # observed (delayed, faded) RSRP dBm
    if rsrp < guard:
        if a == PWR_DOWN or a == HOLD:           # don't thin margin; rebuild it
            a = PWR_UP if pt_dbm < PT_MAX_DBM else (HOLD if a == PWR_DOWN else a)
        # HANDOVER, PWR_UP, RB_UP/RB_DOWN left as chosen (mobility + spectrum intact)
    return a


def rollout(guard):
    good_ho = tot_ho = 0
    clean = incov = boundary = 0
    pwr_ep, spec_ep = [], []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        for _ep in range(N_EP):
            traj = pool[rng.integers(len(pool))]; T = len(traj)
            env = SubTHzEnv6G(traj, seed=seed * 1000 + _ep, w_mob=W_MOB, w_pwr=W_PWR, w_spec=W_SPEC)
            obs, _ = env.reset(); done = False; info = None
            pts, rbs = [], []
            while not done:
                a = shielded_action(obs, env.pt_dbm, guard)
                obs, _, term, trunc, info = env.step(a); done = term or trunc
                pts.append(info['pt_dbm']); rbs.append(info['rb_count'])
                if info.get('handover_done'):
                    tot_ho += 1
                    if not info.get('too_early') and not info.get('too_late'):
                        good_ho += 1
            pwr_ep.append((PT_MAX_DBM - np.mean(pts)) / (PT_MAX_DBM - PT_MIN_DBM) * 100)
            spec_ep.append((RB_MAX - np.mean(rbs)) / (RB_MAX - RB_MIN) * 100)
            if info['rsrp_dbm'] < RLF_THRESH_DBM:
                idx = min(int(info['step']), T - 1); x = float(traj[idx, 0])
                if float(np.min(np.abs(GNB - x))) <= CELL_RADIUS: incov += 1
                else: boundary += 1
            else:
                clean += 1
    n = clean + incov + boundary
    return {
        'ho': 100.0 * good_ho / max(tot_ho, 1),
        'pwr': float(np.mean(pwr_ep)), 'spec': float(np.mean(spec_ep)),
        'incov': incov, 'incov_pct': 100.0 * incov / n,
        'clean': clean, 'boundary': boundary, 'n': n,
    }


if __name__ == "__main__":
    print("=" * 78)
    print("  Reliability shield sweep — minimum-RSRP power guard on trained Dn-DQN")
    print("  (5 seeds x 150 ep = 750 episodes per guard)")
    print("=" * 78)
    print(f"\n  {'guard':>7} | {'HO%':>6} | {'Power%':>7} | {'Spec%':>6} | "
          f"{'in-cov RLF':>10} | {'RLF%':>5} | {'clean%':>6}")
    print("  " + "-" * 74)
    rows = []
    for g in GUARDS:
        r = rollout(g)
        rows.append((g, r))
        gl = "none" if g is None else f"{g:.0f}"
        print(f"  {gl:>7} | {r['ho']:6.1f} | {r['pwr']:7.1f} | {r['spec']:6.1f} | "
              f"{r['incov']:10d} | {r['incov_pct']:5.1f} | "
              f"{100*r['clean']/r['n']:6.1f}")
    print("\n  Baseline (guard=none) vs PG-RRM in-cov RLF (384, 51.2%). As the guard")
    print("  rises, in-coverage RLF falls (reliability up) while power saving falls")
    print("  (efficiency down): the operator-tunable RLF<->efficiency frontier.")
    # save CSV
    import csv
    with open('results/reliability_shield_sweep.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['guard_dbm','HO_pct','Power_pct','Spec_pct','incov_RLF','incov_pct','clean_pct'])
        for g, r in rows:
            w.writerow(['none' if g is None else g, f"{r['ho']:.2f}", f"{r['pwr']:.2f}",
                        f"{r['spec']:.2f}", r['incov'], f"{r['incov_pct']:.2f}",
                        f"{100*r['clean']/r['n']:.2f}"])
    print("\n  -> results/reliability_shield_sweep.csv")
# channel-seeded 20260804
