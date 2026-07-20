#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""policy_fig_data.py — fresh greedy-rollout data for two new manuscript figures. Real data only.

(1) Greedy-policy behaviour (guarded controller): per step, log the reported RSRP the controller
    consumes (previous interval), the reliability-branch intent, and the applied transmit power
    (post-guard). -> results/policy_behavior.csv (binned action shares + mean applied power).
(2) Handover-outcome breakdown per method (A3 rule / flat Dn-DQN / shielded Safe-DQN / guarded
    branched): HO events classified success / too-early / too-late, plus HO events per episode.
    Standard 5 eval seeds x 150 episodes. -> results/ho_outcomes.csv
"""
import csv, sys
import numpy as np, torch
torch.set_num_threads(1)
_A = list(sys.argv); sys.argv = [sys.argv[0]]
import train_phase2 as tp
from dndqn import OBS_DIM, load_dndqn_checkpoint
from dndqn_branched import load_branched, R_STAY_PROTECT, R_STAY_SAVE, R_HANDOVER, S_HOLD
from env_6g_branched import BranchedEnv6G
from env_6g_nyu import SubTHzEnv6G_NYU
from env_6g_v5 import W_MOB, W_PWR, W_SPEC, PT_MAX_DBM
from train_safe_dqn import shielded_action
import multi_seed_eval as mse
sys.argv = _A

GAIN = 15.0
slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH)
POOL = slow + med
print(f'pool={len(POOL)}', flush=True)

# ───────── (1) greedy policy behaviour: guarded branched, 5 models ─────────
BINS = np.arange(-96, -54, 2.0)
cnt = {k: np.zeros(len(BINS) - 1) for k in ('PROTECT', 'SAVE', 'HANDOVER')}
pwr_sum = np.zeros(len(BINS) - 1); pwr_n = np.zeros(len(BINS) - 1)
for mi in range(5):
    net = load_branched(f'br_gst_s{mi}.pt', OBS_DIM)[0]
    r = np.random.default_rng(100 + mi)
    for ep in range(100):
        env = BranchedEnv6G(POOL[r.integers(len(POOL))], binary_power=True, guard_db=9.0,
                            guard_mode='state', array_gain_dbi=GAIN,
                            w_mob=W_MOB, w_pwr=W_PWR, w_spec=W_SPEC)
        obs, _ = env.reset(seed=7000 + 100 * mi + ep); done = False
        while not done:
            rsrp_meas = env._prev_rsrp
            a = net.act(obs)
            obs, _, t, tr_, info = env.step(a); done = t or tr_
            b = np.searchsorted(BINS, rsrp_meas) - 1
            if 0 <= b < len(BINS) - 1:
                name = {R_STAY_PROTECT: 'PROTECT', R_STAY_SAVE: 'SAVE', R_HANDOVER: 'HANDOVER'}[int(a[0])]
                cnt[name][b] += 1
                pwr_sum[b] += info['pt_dbm']; pwr_n[b] += 1
    print(f'  greedy model {mi} done', flush=True)
with open('results/policy_behavior.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['bin_lo', 'bin_hi', 'n', 'PROTECT', 'SAVE', 'HANDOVER', 'mean_applied_pt_dbm'])
    for b in range(len(BINS) - 1):
        n = cnt['PROTECT'][b] + cnt['SAVE'][b] + cnt['HANDOVER'][b]
        row = [BINS[b], BINS[b + 1], int(n)]
        row += [f'{(cnt[k][b] / n if n else 0):.4f}' for k in ('PROTECT', 'SAVE', 'HANDOVER')]
        row += [f'{(pwr_sum[b] / pwr_n[b] if pwr_n[b] else 0):.3f}']
        w.writerow(row)
print('-> results/policy_behavior.csv', flush=True)

# ───────── (2) handover outcomes per method ─────────
def run_branched(policy, label):
    ev = {'success': 0, 'early': 0, 'late': 0, 'ho': 0, 'ep': 0}
    for seed in mse.SEEDS:
        r = np.random.default_rng(seed)
        for ep in range(mse.N_EPISODES):
            env = BranchedEnv6G(POOL[r.integers(len(POOL))], binary_power=True,
                                guard_db=(9.0 if label == 'Guarded branched (proposed)' else None),
                                guard_mode='state', array_gain_dbi=GAIN,
                                w_mob=W_MOB, w_pwr=W_PWR, w_spec=W_SPEC)
            obs, _ = env.reset(seed=10_000 * seed + ep); done = False; ev['ep'] += 1
            while not done:
                a = policy(obs, env); pg = env.serving_gnb_idx
                obs, _, t, tr_, info = env.step(a); done = t or tr_
                if env.serving_gnb_idx != pg:
                    ev['ho'] += 1
                    if info.get('too_early'): ev['early'] += 1
                    elif info.get('too_late'): ev['late'] += 1
                    else: ev['success'] += 1
    return ev

def run_flat(net_or_rule, label):
    ev = {'success': 0, 'early': 0, 'late': 0, 'ho': 0, 'ep': 0}
    for seed in mse.SEEDS:
        r = np.random.default_rng(seed)
        for ep in range(mse.N_EPISODES):
            env = SubTHzEnv6G_NYU(POOL[r.integers(len(POOL))], array_gain_dbi=GAIN,
                                  w_mob=W_MOB, w_pwr=W_PWR, w_spec=W_SPEC)
            obs, _ = env.reset(seed=10_000 * seed + ep); done = False; ev['ep'] += 1
            while not done:
                a = net_or_rule(obs); pg = env.serving_gnb_idx
                obs, _, t, tr_, info = env.step(a); done = t or tr_
                if env.serving_gnb_idx != pg:
                    ev['ho'] += 1
                    if info.get('too_early'): ev['early'] += 1
                    elif info.get('too_late'): ev['late'] += 1
                    else: ev['success'] += 1
    return ev

rows = []
def add(label, ev, n_models=1):
    tot = max(ev['ho'], 1)
    rows.append([label, ev['ep'], ev['ho'], f"{ev['ho']/ev['ep']:.3f}",
                 f"{100*ev['success']/tot:.2f}", f"{100*ev['early']/tot:.2f}", f"{100*ev['late']/tot:.2f}", n_models])
    print(f"  {label:34s} HO/ep {ev['ho']/ev['ep']:.2f}  succ {100*ev['success']/tot:.2f}%  early {100*ev['early']/tot:.2f}%  late {100*ev['late']/tot:.2f}%", flush=True)

# A3 rule (always Pmax, full RBs, reactive HO at −80)
a3 = run_branched(lambda obs, env: (R_HANDOVER if obs[1] * 100.0 < 0 else R_STAY_PROTECT, S_HOLD), '3GPP A3 (rule)')
add('3GPP A3 (rule)', a3)

# flat Dn-DQN (reliability tilt, published single model)
fnet = load_dndqn_checkpoint('dqn_phase2_dndqn_nyu_rel.pt')[0]; fnet.eval()
flat = run_flat(lambda obs: int(fnet(torch.FloatTensor(obs)).argmax().item()), 'Flat Dn-DQN (rel)')
add('Flat Dn-DQN (reliability tilt)', flat)

# shielded Safe-DQN (3 models pooled)
ev = {'success': 0, 'early': 0, 'late': 0, 'ho': 0, 'ep': 0}
for mi in range(3):
    snet = load_dndqn_checkpoint(f'ms_safe_s{mi}.pt')[0]; snet.eval()
    e = run_flat(lambda obs: shielded_action(snet(torch.FloatTensor(obs)), obs), f'safe{mi}')
    for k in ev: ev[k] += e[k]
add('Shielded Safe-DQN (action mask)', ev, 3)

# guarded branched (5 models pooled)
ev = {'success': 0, 'early': 0, 'late': 0, 'ho': 0, 'ep': 0}
for mi in range(5):
    gnet = load_branched(f'br_gst_s{mi}.pt', OBS_DIM)[0]
    e = run_branched(lambda obs, env, n=gnet: n.act(obs), 'Guarded branched (proposed)')
    for k in ev: ev[k] += e[k]
add('Guarded branched (proposed)', ev, 5)

with open('results/ho_outcomes.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['method', 'episodes', 'ho_events', 'ho_per_ep', 'success_pct', 'early_pct', 'late_pct', 'n_models'])
    w.writerows(rows)
print('-> results/ho_outcomes.csv', flush=True)
