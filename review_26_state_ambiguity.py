#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
review_26_state_ambiguity.py — how much does the missing direction-of-travel feature cost?

Reviewer comment 2.6: the six-dimensional state carries no direction of travel, so a vehicle
approaching a gNB and one receding from it can present identical observations while warranting
different decisions. Section IV-A concedes this. This script bounds it.

Method — evaluation only, frozen policies, NO TRAINING and no retraining.
  For each of the five frozen guarded checkpoints, run greedy rollouts and log at every step:
    * the reported RSRP the controller and the guard actually consume (previous interval),
    * the independent state components (RSRP, applied power, RB count, SINR, velocity;
      margin is an affine function of RSRP and carries no extra information),
    * the direction of travel relative to the SERVING cell — approaching iff x < x_gNB,
      the trajectory pool being unidirectional,
    * the decision-relevant outcome: did an out-of-sync sample (RSRP < -92 dBm) occur within
      the next N310 = 3 control intervals — i.e. did protection turn out to be needed.

  Then, over binned observations:
    (1) AMBIGUITY RATE — the share of evaluated steps whose observation bin is visited in both
        directions. A step in a bin only ever seen in one direction is not ambiguous: direction
        is already determined by the other five features.
    (2) The same rate restricted to the guard's operating region (reported RSRP below the
        -76 dBm line), which is where a protection decision is actually at stake.
    (3) OUTCOME DIVERGENCE — within ambiguous bins carrying enough visits in both directions,
        the difference in P(out-of-sync within 3 steps) between approaching and receding.
        If that difference is small, direction carries little decision-relevant information at
        this operating point and the omission is cheap. If it is large, it is not, and the
        figure quantifies what a recurrent or direction-aware policy would stand to recover.

Run from the repository root:   python review_26_state_ambiguity.py
Writes:                         results/state_ambiguity.csv  (per-bin detail)
Prints:                         the three headline figures for Section IV-A.
"""
import csv, os, sys
import numpy as np
import torch
torch.set_num_threads(1)

_A = list(sys.argv); sys.argv = [sys.argv[0]]
import train_phase2 as tp
from dndqn import OBS_DIM
from dndqn_branched import load_branched
from env_6g_branched import BranchedEnv6G
from env_6g_v5 import W_MOB, W_PWR, W_SPEC, GNB_X
sys.argv = _A

GAIN       = 15.0
GUARD_DB   = 9.0
GUARD_LINE = -76.0          # Theta_RLF + dP + G = -92 + 7 + 9
THETA_RLF  = -92.0
N310       = 3
N_MODELS   = 5
N_EPISODES = 100            # same basis and seeding as the Fig. 10 greedy census

# binning: coarse enough that collisions are physically meaningful, fine enough to be honest
def bin_of(rsrp, pt, rb, sinr, vel):
    return (int(np.floor(rsrp / 2.0)),      # 2 dB, as in Fig. 10
            int(round(pt)),                 # binary actuator -> {23, 30} dBm
            int(rb // 8),                   # one RB_UP/RB_DOWN step
            int(np.floor(sinr / 5.0)),      # 5 dB
            int(vel // 5))                  # 5 m/s

def main():
    slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH)
    POOL = slow + med
    print(f'pool={len(POOL)} trajectories', flush=True)

    rows = []                                        # (bin, approaching, needed_protection, rsrp_rep)
    for mi in range(N_MODELS):
        net = load_branched(f'br_gst_s{mi}.pt', OBS_DIM)[0]
        r = np.random.default_rng(100 + mi)
        for ep in range(N_EPISODES):
            env = BranchedEnv6G(POOL[r.integers(len(POOL))], binary_power=True,
                                guard_db=GUARD_DB, guard_mode='state', array_gain_dbi=GAIN,
                                w_mob=W_MOB, w_pwr=W_PWR, w_spec=W_SPEC)
            obs, _ = env.reset(seed=7000 + 100 * mi + ep)
            done, traj = False, []
            while not done:
                rsrp_rep = env._prev_rsrp                     # what the controller/guard consume
                x        = env._veh_x() if hasattr(env, '_veh_x') else float(env.traj[min(env.step_idx, env.T - 1), 0])
                approaching = bool(x < GNB_X[env.serving_gnb_idx])
                a = net.act(obs)
                obs, _, t, tr_, info = env.step(a)
                done = t or tr_
                traj.append((rsrp_rep, approaching, info['pt_dbm'], info['rb_count'],
                             info['sinr_db'], info['xVelocity_ms'], info['rsrp_dbm']))
            # decision-relevant outcome: out-of-sync within the next N310 intervals
            true_rsrp = [s[6] for s in traj]
            for k, s in enumerate(traj):
                fut = true_rsrp[k + 1: k + 1 + N310]
                needed = any(v < THETA_RLF for v in fut)
                rows.append((bin_of(s[0], s[2], s[3], s[4], s[5]), s[1], needed, s[0]))
        print(f'  model {mi} done ({len(rows)} steps so far)', flush=True)

    # ---- aggregate per bin ----
    agg = {}
    for b, appr, needed, rsrp_rep in rows:
        d = agg.setdefault(b, {'nA': 0, 'nR': 0, 'oA': 0, 'oR': 0, 'guard': 0})
        if appr: d['nA'] += 1; d['oA'] += int(needed)
        else:    d['nR'] += 1; d['oR'] += int(needed)
        if rsrp_rep < GUARD_LINE: d['guard'] += 1

    total   = len(rows)
    ambig   = sum(v['nA'] + v['nR'] for v in agg.values() if v['nA'] and v['nR'])
    g_tot   = sum(1 for _, _, _, rr in rows if rr < GUARD_LINE)
    g_ambig = sum(v['guard'] for v in agg.values() if v['nA'] and v['nR'])

    print('\n' + '=' * 78)
    print('(1) ambiguity rate, all evaluated steps      : %7d / %7d = %5.2f%%'
          % (ambig, total, 100.0 * ambig / total))
    print('(2) ambiguity rate, guard region (< %.0f dBm): %7d / %7d = %5.2f%%'
          % (GUARD_LINE, g_ambig, g_tot, 100.0 * g_ambig / max(g_tot, 1)))

    # ---- outcome divergence in well-populated ambiguous bins ----
    MIN = 50
    diffs, wts, detail = [], [], []
    for b, v in sorted(agg.items()):
        if v['nA'] >= MIN and v['nR'] >= MIN:
            pA, pR = v['oA'] / v['nA'], v['oR'] / v['nR']
            diffs.append(abs(pA - pR)); wts.append(v['nA'] + v['nR'])
            detail.append((b, v['nA'], v['nR'], pA, pR, pA - pR))
    if diffs:
        diffs, wts = np.array(diffs), np.array(wts, float)
        print('(3) outcome divergence in %d ambiguous bins with >= %d visits per direction:'
              % (len(diffs), MIN))
        print('       P(out-of-sync within %d steps): mean |difference| = %.4f, '
              'visit-weighted = %.4f, max = %.4f'
              % (N310, diffs.mean(), float((diffs * wts).sum() / wts.sum()), diffs.max()))
    else:
        print('(3) no ambiguous bin carried >= %d visits in both directions.' % MIN)
    print('=' * 78)

    os.makedirs('results', exist_ok=True)
    with open('results/state_ambiguity.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['rsrp_bin_lo_dbm', 'pt_dbm', 'rb_bin', 'sinr_bin_lo_db', 'vel_bin_lo_ms',
                    'n_approaching', 'n_receding', 'p_oos_approaching', 'p_oos_receding', 'delta'])
        for b, nA, nR, pA, pR, d in detail:
            w.writerow([b[0] * 2, b[1], b[2] * 8, b[3] * 5, b[4] * 5,
                        nA, nR, '%.4f' % pA, '%.4f' % pR, '%+.4f' % d])
    print('-> results/state_ambiguity.csv (%d ambiguous bins)' % len(detail))
    print('\nNo checkpoint was modified and no training step was taken.')

if __name__ == '__main__':
    main()
