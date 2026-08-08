#!/usr/bin/env python3
"""train_reward_scratch_v6.py — FROM-SCRATCH reward-component ablation (unbiased).
Usage: python train_reward_scratch_v6.py <none|c1|c3|c5> [target_steps] [chunk_steps]
Random init (no warm-start), trains under the corrected channel with ONE component
disabled (or all on for 'none'). Resumable: saves dqn_scratch_<x>_v6.pt + a step
sidecar, so you can run it in chunks (or all at once on a laptop: chunk=target).
  On a laptop (no time limit): python -u train_reward_scratch_v6.py none 800000 800000
"""
import os, sys, json, time, random
from collections import deque
import numpy as np, torch, torch.nn as nn, torch.optim as optim
# 1 thread/process by default so 4 parallel runs each pin one core (no oversubscription).
# Override in the cloud with e.g. TORCH_THREADS=4 for a single faster run.
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "1")))
from dndqn import (build_dndqn_networks, NStepPERBuffer, polyak_update,
                   epsilon_schedule, beta_schedule, OBS_DIM, NET_ARCH)
from env_6g_v5 import N_ACTIONS, PWR_DOWN, RB_DOWN
from env_6g_v6 import SubTHzEnv6G_v6, N310_DEFAULT
import train_phase2 as tp
from train_phase2_v6 import (W_MOB, W_PWR, W_SPEC, SAFE_RSRP_THRESH, PWR_DOWN_UNSAFE_PEN,
                             _PWR_STEP_REW, _RB_STEP_REW, RLF_TERMINAL_PENALTY)

DIS = sys.argv[1].lower()
TARGET = int(sys.argv[2]) if len(sys.argv) > 2 else 800_000
CHUNK  = int(sys.argv[3]) if len(sys.argv) > 3 else 150_000
C1, C3, C5 = DIS != 'c1', DIS != 'c3', DIS != 'c5'
OUT = f"dqn_scratch_{DIS}_v6.pt"; SIDE = f"dqn_scratch_{DIS}_v6.json"
GAMMA, NSTEP, LR, BATCH, BUF, TRAINFREQ, TARGUP, TAU, LSTART = 0.99, 3, 3e-4, 500_000, 256, 4, 500, 0.01, 10_000
# (BATCH/BUF positions fixed below)
BATCH, BUF = 256, 500_000

class AblatEnv(SubTHzEnv6G_v6):
    def __init__(self, traj):
        super().__init__(traj, n310=N310_DEFAULT, w_mob=W_MOB, w_pwr=W_PWR, w_spec=W_SPEC); self._p=-70.0
    def reset(self, seed=None, options=None):
        o,i=super().reset(seed=seed, options=options); self._p=-70.0; return o,i
    def step(self, action):
        rb=self._p; obs,_,term,trunc,info=super().step(action); self._p=info['rsrp_dbm']
        reward=W_MOB*info['r_mobility']
        if action==PWR_DOWN:
            reward += (_PWR_STEP_REW if rb>=SAFE_RSRP_THRESH else PWR_DOWN_UNSAFE_PEN) if C1 else W_PWR*info['r_power']
        else: reward += W_PWR*info['r_power']
        if action==RB_DOWN: reward += _RB_STEP_REW if C5 else W_SPEC*info['r_spectrum']
        else: reward += W_SPEC*info['r_spectrum']
        if C3 and term and info.get('sustained_rlf'): reward += RLF_TERMINAL_PENALTY
        return obs,reward,term,trunc,info

slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH); pool = slow + med
online, target = build_dndqn_networks(OBS_DIM, N_ACTIONS, NET_ARCH, 'cpu')
gstep = 0
if os.path.exists(OUT) and os.path.exists(SIDE):
    online.load_state_dict(torch.load(OUT, map_location='cpu')['q_net_state'])
    gstep = json.load(open(SIDE))['global_step']; print(f"[resume] {OUT} @ {gstep}", flush=True)
else:
    print(f"[init] random init | disable {DIS.upper()} (C1={C1},C3={C3},C5={C5})", flush=True)
target.load_state_dict(online.state_dict())
opt = optim.Adam(online.parameters(), lr=LR)
buf = NStepPERBuffer(BUF, n_steps=NSTEP, gamma=GAMMA, alpha=0.6)
rng = np.random.default_rng(42 + gstep)
env = AblatEnv(pool[rng.integers(len(pool))]); obs,_ = env.reset()
epr = deque(maxlen=20); er = 0.0; t0 = time.time()
n_this = min(CHUNK, TARGET - gstep)
print(f"[train] this run {n_this:,} steps (target {TARGET:,})", flush=True)
for i in range(1, n_this+1):
    g = gstep + i
    eps = epsilon_schedule(g, 1.0, 0.05, 0.30, TARGET); beta = beta_schedule(g, 0.4, 1.0, TARGET)
    if random.random() < eps: a = random.randint(0, N_ACTIONS-1)
    else:
        with torch.no_grad(): a = online(torch.FloatTensor(obs)).argmax().item()
    nobs, r, term, trunc, info = env.step(a); done = term or trunc; er += r
    buf.add(obs, a, r, nobs, term); obs = nobs
    if done:
        epr.append(er); er = 0.0; env = AblatEnv(pool[rng.integers(len(pool))]); obs,_=env.reset()
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
    if g % TARGUP == 0: polyak_update(online, target, TAU)
    if i % 30000 == 0:
        print(f"  [{g:7d}/{TARGET}] mean_rew={np.mean(epr) if epr else 0:8.2f} eps={eps:.3f} ({(time.time()-t0)/60:.1f} min)", flush=True)
torch.save({'q_net_state':online.state_dict(),'obs_dim':OBS_DIM,'n_actions':N_ACTIONS,
            'net_arch':NET_ARCH,'algorithm':f'Dn-DQN-scratch-{DIS}','n_steps':NSTEP}, OUT)
json.dump({'global_step': gstep + n_this}, open(SIDE,'w'))
done_flag = (gstep + n_this) >= TARGET
print(f"[save] {OUT} @ {gstep+n_this}/{TARGET} {'(COMPLETE)' if done_flag else '(resume for more)'} "
      f"({(time.time()-t0)/60:.1f} min)", flush=True)
