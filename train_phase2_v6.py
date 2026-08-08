#!/usr/bin/env python3
"""
train_phase2_v6.py — retrain Dn-DQN under the corrected (sustained-RLF) env_6g_v6.

Same architecture, hyperparameters, reward weights, and Fix1/Fix5 shaping as
train_phase2.py. Two differences, both from env_6g_v6:
  * episodes terminate on SUSTAINED RLF (N310 consecutive out-of-sync), not on a
    single 40 ms sub-threshold sample;
  * the terminal -100 RLF penalty (Fix3) is applied only on a sustained failure.
Random initialisation (no Phase-1 warm start), which the original study showed
reaches performance statistically equivalent to the curriculum (Table V).

SAVES: dqn_phase2_dndqn_v6.pt      RUN: python -u train_phase2_v6.py
"""
import os, sys, time, random
from collections import deque
import numpy as np, torch, torch.nn as nn, torch.optim as optim

from env_6g_v6 import SubTHzEnv6G_v6, N310_DEFAULT
from env_6g_v5 import (PWR_DOWN, RB_DOWN, RLF_THRESH_DBM,
                       PT_MIN_DBM, PT_MAX_DBM, PT_STEP_DBM, RB_MIN, RB_MAX,
                       N_ACTIONS, MAX_STEPS)
from dndqn import (build_dndqn_networks, NStepPERBuffer, polyak_update,
                   epsilon_schedule, beta_schedule, OBS_DIM, NET_ARCH)
import train_phase2 as tp   # reuse load_trajectories + hyperparameters

CSV_PATH        = tp.CSV_PATH
MODEL_SAVE_PATH = "dqn_phase2_dndqn_v6.pt"
W_MOB, W_PWR, W_SPEC = 0.60, 0.20, 0.20
TOTAL_TIMESTEPS = 800_000
LEARNING_RATE, GAMMA = 3e-4, 0.99
BUFFER_SIZE, BATCH_SIZE = 500_000, 256
TARGET_UPDATE_FREQ, TAU, TRAIN_FREQ, LEARNING_STARTS, LOG_INTERVAL = 500, 0.01, 4, 10_000, 5_000
EPS_START, EPS_END, EPS_FRACTION = 1.0, 0.05, 0.30   # random init -> full exploration
NSTEP, PER_ALPHA, PER_BETA_START, PER_BETA_END = 3, 0.6, 0.4, 1.0
STAGE_SLOW_END = 0.08
SAFE_RSRP_THRESH, PWR_DOWN_UNSAFE_PEN = -78.0, -0.20
_PWR_STEP_REW = W_PWR  * (PT_STEP_DBM / (PT_MAX_DBM - PT_MIN_DBM)) * 10.0
_RB_STEP_REW  = W_SPEC * (PT_STEP_DBM / (PT_MAX_DBM - PT_MIN_DBM)) * 10.0
RLF_TERMINAL_PENALTY = -100.0
DEVICE = torch.device("cpu")


class Phase2Env_v6(SubTHzEnv6G_v6):
    """Sustained-RLF training env with Fix1 + Fix5 + sustained Fix3 shaping."""
    def __init__(self, traj):
        super().__init__(traj, n310=N310_DEFAULT, w_mob=W_MOB, w_pwr=W_PWR, w_spec=W_SPEC)
        self._prev_rsrp = -70.0

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._prev_rsrp = -70.0
        return obs, info

    def step(self, action):
        rsrp_before = self._prev_rsrp
        obs, _, terminated, truncated, info = super().step(action)
        self._prev_rsrp = info['rsrp_dbm']
        r_mob  = info.get('r_mobility', 0.0)
        r_pwr  = info.get('r_power', 0.0)
        r_spec = info.get('r_spectrum', 0.0)
        reward = W_MOB * r_mob
        if action == PWR_DOWN:
            reward += _PWR_STEP_REW if rsrp_before >= SAFE_RSRP_THRESH else PWR_DOWN_UNSAFE_PEN
        else:
            reward += W_PWR * r_pwr
        if action == RB_DOWN:
            reward += _RB_STEP_REW
        else:
            reward += W_SPEC * r_spec
        if terminated and info.get('sustained_rlf'):     # Fix3 on SUSTAINED RLF only
            reward += RLF_TERMINAL_PENALTY
        return obs, reward, terminated, truncated, info


def main():
    print("="*68); print("  Phase 2 Dn-DQN — CORRECTED sustained-RLF env (v6)")
    print(f"  N310={N310_DEFAULT} | random init | Fix1+Fix5+sustainedFix3"); print("="*68)
    slow, med, fast, skip = tp.load_trajectories(CSV_PATH)
    pool = slow + med
    print(f"[Data] slow={len(slow)} med={len(med)} pool={len(pool)}")
    online, target = build_dndqn_networks(OBS_DIM, N_ACTIONS, NET_ARCH, 'cpu')
    target.load_state_dict(online.state_dict())
    opt = optim.Adam(online.parameters(), lr=LEARNING_RATE)
    buf = NStepPERBuffer(BUFFER_SIZE, n_steps=NSTEP, gamma=GAMMA, alpha=PER_ALPHA)
    rng = np.random.default_rng(42)
    sample = lambda stg: (slow if stg=="SLOW" else pool)[rng.integers(len(slow if stg=="SLOW" else pool))]
    stage="SLOW"; env=Phase2Env_v6(sample(stage)); obs,_=env.reset()
    epr=deque(maxlen=20); er=0.0; t0=time.time()
    for step in range(1, TOTAL_TIMESTEPS+1):
        frac=step/TOTAL_TIMESTEPS
        stage="SLOW" if frac<STAGE_SLOW_END else "SLOW+MED"
        eps=epsilon_schedule(step, EPS_START, EPS_END, EPS_FRACTION, TOTAL_TIMESTEPS)
        beta=beta_schedule(step, PER_BETA_START, PER_BETA_END, TOTAL_TIMESTEPS)
        if random.random()<eps: a=random.randint(0,N_ACTIONS-1)
        else:
            with torch.no_grad(): a=online(torch.FloatTensor(obs)).argmax().item()
        nobs,r,term,trunc,info=env.step(a); done=term or trunc; er+=r
        buf.add(obs,a,r,nobs,term); obs=nobs
        if done:
            epr.append(er); er=0.0; env=Phase2Env_v6(sample(stage)); obs,_=env.reset()
        if step>=LEARNING_STARTS and step%TRAIN_FREQ==0 and len(buf)>=BATCH_SIZE:
            ib,wb,ob,ab,rb,nb,db=buf.sample(BATCH_SIZE,beta)
            ot=torch.FloatTensor(ob); nt=torch.FloatTensor(nb); at=torch.LongTensor(ab)
            rt=torch.FloatTensor(rb); dt=torch.FloatTensor(db); wt=torch.FloatTensor(wb)
            with torch.no_grad():
                ba=online(nt).argmax(1,keepdim=True); nq=target(nt).gather(1,ba).squeeze(1)
                tq=rt+(GAMMA**NSTEP)*nq*(1-dt)
            cq=online(ot).gather(1,at.unsqueeze(1)).squeeze(1)
            td=(cq-tq).detach().cpu().numpy()
            loss=(wt*nn.functional.huber_loss(cq,tq,reduction='none')).mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(online.parameters(),10.0); opt.step()
            buf.update_priorities(ib,td)
        if step%TARGET_UPDATE_FREQ==0: polyak_update(online,target,TAU)
        if step%LOG_INTERVAL==0:
            mr=float(np.mean(epr)) if epr else 0.0
            print(f"[{step:8d}/{TOTAL_TIMESTEPS}] {stage:<9} mean_rew={mr:9.3f} eps={eps:.3f}", flush=True)
    torch.save({'q_net_state':online.state_dict(),'obs_dim':OBS_DIM,'n_actions':N_ACTIONS,
                'net_arch':NET_ARCH,'algorithm':'Dn-DQN','phase':2,'w_mob':W_MOB,'w_pwr':W_PWR,
                'w_spec':W_SPEC,'n_steps':NSTEP,'per_alpha':PER_ALPHA,
                'training_mode':'random_init_v6_sustained_rlf','n310':N310_DEFAULT},
               MODEL_SAVE_PATH)
    print(f"\n[Save] {MODEL_SAVE_PATH}  ({(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
