#!/usr/bin/env python3
"""cliff_sweep_nyu.py — DIRECT, policy-independent measurement of the power-reliability cliff
(the motivating figure for the cliff-aware controller). Under the NYU 142 GHz CI channel with a
plain 3GPP A3 handover rule (no learning, no power/RB actions), transmit power is FIXED at each
level 23..30 dBm and sustained-RLF rate is measured. Same protocol as headline results
(5 eval seeds x 150 ep).  -> results/cliff_power_sweep_nyu.csv

Also sweeps RB count at fixed mid power -> results/cliff_rb_sweep_nyu.csv.
HONESTY NOTE for the paper: in this simulator (consistent with 3GPP), RSRP/RLF is independent of
RB allocation by construction — the RB panel documents that reliability-decoupling as a
standards-consistent modeling property, NOT an empirical discovery. The measured finding is the
POWER cliff: its location, sharpness, and scenario dependence.
"""
import os, csv, numpy as np
from env_6g_nyu import SubTHzEnv6G_NYU
from env_6g_v5 import (W_MOB, W_PWR, W_SPEC, HANDOVER, HOLD, A3_THRESH_DBM,
                       PT_MIN_DBM, PT_MAX_DBM, RB_MIN, RB_MAX)
import multi_seed_eval as mse

SEEDS, N_EP, GAIN = mse.SEEDS, mse.N_EPISODES, 15.0
slow, med = mse.load_trajectories(mse.CSV_PATH); pool = slow + med

def run_episode(env, pt=None, rb=None):
    obs, _ = env.reset()
    if pt is not None: env.pt_dbm = float(pt)
    if rb is not None: env.rb_count = int(rb)
    done = False; last = None; real = good = 0
    while not done:
        a = HOLD if last is None else (HANDOVER if last['rsrp_dbm'] < A3_THRESH_DBM else HOLD)
        pg = env.serving_gnb_idx
        obs, _, t, tr, info = env.step(a); done = t or tr; last = info
        if pt is not None: env.pt_dbm = float(pt)      # hold power fixed (guard)
        if rb is not None: env.rb_count = int(rb)      # hold RBs fixed (guard)
        if env.serving_gnb_idx != pg:
            real += 1
            if not info.get('too_early') and not info.get('too_late'): good += 1
    return bool(info.get('sustained_rlf')), real, good

def sweep(levels, kind):
    rows = []
    for lv in levels:
        per_seed_rlf = []; per_seed_ho = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed); rlf = 0; realT = goodT = 0
            for _ in range(N_EP):
                traj = pool[rng.integers(len(pool))]
                env = SubTHzEnv6G_NYU(traj, array_gain_dbi=GAIN, w_mob=W_MOB, w_pwr=W_PWR, w_spec=W_SPEC)
                s, real, good = run_episode(env, pt=lv if kind == 'pt' else 26.0,
                                            rb=lv if kind == 'rb' else None)
                rlf += int(s); realT += real; goodT += good
            per_seed_rlf.append(100.0 * rlf / N_EP)
            per_seed_ho.append(100.0 * goodT / max(realT, 1))
        m, sd = float(np.mean(per_seed_rlf)), float(np.std(per_seed_rlf))
        ho = float(np.mean(per_seed_ho))
        rows.append({('pt_dbm' if kind == 'pt' else 'rb_count'): lv,
                     'RLF_mean': f"{m:.2f}", 'RLF_std': f"{sd:.2f}", 'HO_succ': f"{ho:.2f}"})
        print(f"  {kind}={lv:>5} -> RLF {m:6.2f} ± {sd:.2f} %   (HOsucc {ho:.1f}%)", flush=True)
    return rows

def main():
    os.makedirs('results', exist_ok=True)
    print(f"DIRECT CLIFF MEASUREMENT — A3 rule, fixed power, NYU CI channel | {len(SEEDS)} seeds x {N_EP} ep", flush=True)
    print("-- power sweep (RLF vs fixed pt_dbm) --", flush=True)
    prows = sweep(list(range(int(PT_MIN_DBM), int(PT_MAX_DBM) + 1)), 'pt')
    with open('results/cliff_power_sweep_nyu.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['pt_dbm', 'RLF_mean', 'RLF_std', 'HO_succ']); w.writeheader(); w.writerows(prows)
    print("-- RB sweep at pt=26 dBm (expected flat BY CONSTRUCTION — see honesty note) --", flush=True)
    rrows = sweep([RB_MIN, 24, 38, 52, RB_MAX], 'rb')
    with open('results/cliff_rb_sweep_nyu.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['rb_count', 'RLF_mean', 'RLF_std', 'HO_succ']); w.writeheader(); w.writerows(rrows)
    rl = [float(r['RLF_mean']) for r in prows]
    print(f"\nCLIFF SUMMARY: RLF at {PT_MIN_DBM:.0f} dBm = {rl[0]:.1f}%  ->  at {PT_MAX_DBM:.0f} dBm = {rl[-1]:.1f}%"
          f" | largest one-dB drop = {max(rl[i]-rl[i+1] for i in range(len(rl)-1)):.1f} pp", flush=True)
    print("-> results/cliff_power_sweep_nyu.csv + cliff_rb_sweep_nyu.csv", flush=True)

if __name__ == "__main__":
    main()
