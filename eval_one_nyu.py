#!/usr/bin/env python3
"""eval_one_nyu.py <checkpoint.pt> [label] — evaluate a single Dn-DQN checkpoint under the NYU
142 GHz CI channel (same metrics as eval_nyu.py). Use to place a new (e.g. reliability-tilted)
model on the reliability-efficiency frontier next to the Table IV points. 5 seeds x 150 ep."""
import sys, math, numpy as np, torch
from env_6g_nyu import SubTHzEnv6G_NYU
from env_6g_v5 import W_MOB, W_PWR, W_SPEC, PT_MIN_DBM, PT_MAX_DBM, RB_MIN, RB_MAX
from dndqn import load_dndqn_checkpoint
import multi_seed_eval as mse

SEEDS, N_EP = mse.SEEDS, mse.N_EPISODES
RB_BW_HZ = 380.16e6 / RB_MAX
GAIN = 15.0
slow, med = mse.load_trajectories(mse.CSV_PATH); pool = slow + med

path  = sys.argv[1]
label = sys.argv[2] if len(sys.argv) > 2 else path
net = load_dndqn_checkpoint(path)[0]; net.eval()

HO, SUC, PWR, SPEC, TP, RLFR = [], [], [], [], [], []
for seed in SEEDS:
    rng = np.random.default_rng(seed)
    real = good = nep = rlf = 0; P = []; S = []; Tp = []
    for _ep in range(N_EP):
        traj = pool[rng.integers(len(pool))]
        env = SubTHzEnv6G_NYU(traj, seed=seed * 1000 + _ep, array_gain_dbi=GAIN, w_mob=W_MOB, w_pwr=W_PWR, w_spec=W_SPEC)
        obs, _ = env.reset(); done = False; nep += 1; pts = []; rbs = []; ep_t = []
        while not done:
            with torch.no_grad(): a = int(net(torch.FloatTensor(obs)).argmax().item())
            pg = env.serving_gnb_idx
            obs, _, t, tr, info = env.step(a); done = t or tr
            pts.append(info['pt_dbm']); rbs.append(info['rb_count'])
            sdb = obs[4] * 100.0
            ep_t.append(info['rb_count'] * (RB_BW_HZ / 1e6) * math.log2(1 + 10 ** (sdb / 10)))
            if env.serving_gnb_idx != pg:
                real += 1
                if not info.get('too_early') and not info.get('too_late'): good += 1
        if info.get('sustained_rlf'): rlf += 1
        P.append((PT_MAX_DBM - np.mean(pts)) / (PT_MAX_DBM - PT_MIN_DBM) * 100)
        S.append((RB_MAX - np.mean(rbs)) / (RB_MAX - RB_MIN) * 100)
        Tp.append(np.mean(ep_t))
    HO.append(real / nep); SUC.append(100 * good / max(real, 1))
    PWR.append(np.mean(P)); SPEC.append(np.mean(S)); TP.append(np.mean(Tp)); RLFR.append(100 * rlf / nep)

print("=" * 92)
print(f"  {label}  under NYU 142 GHz CI channel (gain {GAIN:.0f} dBi) | 5 seeds x {N_EP} ep")
print("=" * 92)
print(f"  realHO/ep {np.mean(HO):.2f} | HOsucc {np.mean(SUC):.1f}% | Power {np.mean(PWR):.1f}% | "
      f"Spec {np.mean(SPEC):.1f}% | Tput {np.mean(TP):.0f} Mbps | RLF {np.mean(RLFR):.1f}%")
print(f"  (vs standard-reward Proposed: 86.1% / 77.1% / 76.1% / 20.9% RLF ; vs A3: 96.2% / 0% / 0% / 5.1% RLF)")
# channel-seeded 20260804
