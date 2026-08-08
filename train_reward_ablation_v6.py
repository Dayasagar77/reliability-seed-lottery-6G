#!/usr/bin/env python3
"""train_reward_ablation_v6.py — reward-component ablation under the corrected channel.
Usage: python train_reward_ablation_v6.py <disable:c1|c3|c5> <out_ckpt> [steps]
Disables ONE reward component and retrains (warm-start from the full corrected model),
to measure each component's contribution.
  C1 = RSRP-gated PWR_DOWN | C3 = sustained-RLF terminal penalty | C5 = symmetric RB_DOWN reward
Note: fine-tuned from the full model, so effects are a conservative lower bound vs from-scratch."""
import sys, time, random
from collections import deque
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from dndqn import build_dndqn_networks, NStepPERBuffer, polyak_update, OBS_DIM, NET_ARCH
from env_6g_v5 import N_ACTIONS, PWR_DOWN, RB_DOWN
from env_6g_v6 import SubTHzEnv6G_v6, N310_DEFAULT
import train_phase2 as tp
from train_phase2_v6 import (W_MOB, W_PWR, W_SPEC, SAFE_RSRP_THRESH, PWR_DOWN_UNSAFE_PEN,
                             _PWR_STEP_REW, _RB_STEP_REW, RLF_TERMINAL_PENALTY)

DISABLE = sys.argv[1].lower(); OUT = sys.argv[2]
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 150000
C1, C3, C5 = DISABLE != 'c1', DISABLE != 'c3', DISABLE != 'c5'
GAMMA, NSTEP, LR, BATCH, BUF, TRAINFREQ, TARGUP, TAU, LSTART = 0.99, 3, 3e-4, 256, 300_000, 4, 500, 0.01, 3_000

class AblatEnv(SubTHzEnv6G_v6):
    def __init__(self, traj):
        super().__init__(traj, n310=N310_DEFAULT, w_mob=W_MOB, w_pwr=W_PWR, w_spec=W_SPEC)
        self._prev = -70.0
    def reset(self, seed=None, options=None):
        o, i = super().reset(seed=seed, options=options); self._prev = -70.0; return o, i
    def step(self, action):
        rb = self._prev
        obs, _, term, trunc, info = super().step(action); self._prev = info['rsrp_dbm']
        reward = W_MOB * info['r_mobility']
        if action == PWR_DOWN:
            reward += (_PWR_STEP_REW if rb >= SAFE_RSRP_THRESH else PWR_DOWN_UNSAFE_PEN) if C1 else W_PWR*info['r_power']
        else:
            reward += W_PWR * info['r_power']
        if action == RB_DOWN:
            reward += _RB_STEP_REW if C5 else W_SPEC*info['r_spectrum']
        else:
            reward += W_SPEC * info['r_spectrum']
        if C3 and term and info.get('sustained_rlf'):
            reward += RLF_TERMINAL_PENALTY
        return obs, reward, term, trunc, info

slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH); pool = slow + med
online, target = build_dndqn_networks(OBS_DIM, N_ACTIONS, NET_ARCH, 'cpu')
online.load_state_dict(torch.load('dqn_phase2_dndqn_v6.pt', map_location='cpu')['q_net_state'])
target.load_state_dict(online.state_dict())
opt = optim.Adam(online.parameters(), lr=LR)
buf = NStepPERBuffer(BUF, n_steps=NSTEP, gamma=GAMMA, alpha=0.6)
rng = np.random.default_rng(1234)
print(f"[reward-ablation] disable {DISABLE.upper()} (C1={C1},C3={C3},C5={C5}) -> {OUT} | {STEPS:,} steps", flush=True)
env = AblatEnv(pool[rng.integers(len(pool))]); obs, _ = env.reset()
epr = deque(maxlen=30); er = 0.0; t0 = time.time()
for i in range(1, STEPS+1):
    eps = max(0.05, 0.10 - 0.05*i/STEPS); beta = min(1.0, 0.4+0.6*i/STEPS)
    if random.random() < eps: a = random.randint(0, N_ACTIONS-1)
    else:
        with torch.no_grad(): a = online(torch.FloatTensor(obs)).argmax().item()
    nobs, r, term, trunc, info = env.step(a); done = term or trunc; er += r
    buf.add(obs, a, r, nobs, term); obs = nobs
    if done:
        epr.append(er); er = 0.0; env = AblatEnv(pool[rng.integers(len(pool))]); obs,_ = env.reset()
    if i >= LSTART and i % TRAINFREQ == 0 and len(buf) >= BATCH:
        ib,wb,ob,ab,rb2,nb,db = buf.sample(BATCH, beta)
        ot=torch.FloatTensor(ob); nt=torch.FloatTensor(nb); at=torch.LongTensor(ab)
        rt=torch.FloatTensor(rb2); dt=torch.FloatTensor(db); wt=torch.FloatTensor(wb)
        with torch.no_grad():
            ba=online(nt).argmax(1,keepdim=True); nq=target(nt).gather(1,ba).squeeze(1)
            tq=rt+(GAMMA**NSTEP)*nq*(1-dt)
        cq=online(ot).gather(1,at.unsqueeze(1)).squeeze(1)
        td=(cq-tq).detach().cpu().numpy()
        loss=(wt*nn.functional.huber_loss(cq,tq,reduction='none')).mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(online.parameters(),10.0); opt.step()
        buf.update_priorities(ib,td)
    if i % TARGUP == 0: polyak_update(online, target, TAU)
    if i % 30000 == 0:
        print(f"  [{i:7d}/{STEPS}] mean_rew={np.mean(epr) if epr else 0:8.2f} ({(time.time()-t0)/60:.1f} min)", flush=True)
torch.save({'q_net_state':online.state_dict(),'obs_dim':OBS_DIM,'n_actions':N_ACTIONS,
            'net_arch':NET_ARCH,'algorithm':f'Dn-DQN-no-{DISABLE.upper()}','n_steps':NSTEP}, OUT)
print(f"[save] {OUT} ({(time.time()-t0)/60:.1f} min)", flush=True)
