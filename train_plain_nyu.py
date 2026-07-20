#!/usr/bin/env python3
"""train_plain_nyu.py — baseline (Standard DQN / DDQN+PER) training under the NYU 142 GHz CI
channel. IDENTICAL to train_plain_v6.py except the env is Phase2Env_NYU (NYU channel) instead
of Phase2Env_v6 (free-space). Single-variable change for a fair comparison.
Usage: python train_plain_nyu.py <in_ckpt> <out_ckpt> <mode:std|ddqn> [steps]"""
import sys, time, random
from collections import deque
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from qnet_plain import PlainQNet
from dndqn import NStepPERBuffer, polyak_update, OBS_DIM, NET_ARCH
from env_6g_v5 import N_ACTIONS
import train_phase2 as tp
from train_phase2_nyu import Phase2Env_NYU          # <-- NYU channel env (only change vs v6)

IN, OUT, MODE = sys.argv[1], sys.argv[2], sys.argv[3]
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 150000
DOUBLE = (MODE == 'ddqn')
ALPHA  = 0.6 if MODE == 'ddqn' else 0.0
NSTEP  = 1
GAMMA, LR, BATCH, BUF, TRAINFREQ, TARGUP, TAU, LSTART = 0.99, 3e-4, 256, 300_000, 4, 500, 0.01, 3_000

slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH); pool = slow + med
online, target = PlainQNet(OBS_DIM, N_ACTIONS, NET_ARCH), PlainQNet(OBS_DIM, N_ACTIONS, NET_ARCH)
# The proposed model is a DUELING net (shared/value/advantage) — its state_dict does NOT fit a
# PlainQNet. Baselines are therefore trained from RANDOM init under the NYU channel (a clean,
# standard protocol; the proposed model is likewise random-init). If IN happens to be a genuine
# PlainQNet checkpoint, warm-start from it; otherwise fall back to random init.
try:
    online.load_state_dict(torch.load(IN, map_location='cpu')['q_net_state'])
    print(f"[init] warm-started PlainQNet from {IN}", flush=True)
except Exception as e:
    print(f"[init] {IN} is not a PlainQNet checkpoint ({type(e).__name__}) -> RANDOM init", flush=True)
target.load_state_dict(online.state_dict())
opt = optim.Adam(online.parameters(), lr=LR)
buf = NStepPERBuffer(BUF, n_steps=NSTEP, gamma=GAMMA, alpha=ALPHA)
rng = np.random.default_rng(1234)
print(f"[train-NYU] {MODE.upper()} (double={DOUBLE}, PER-alpha={ALPHA}) {IN} -> {OUT} | {STEPS:,} steps", flush=True)
env = Phase2Env_NYU(pool[rng.integers(len(pool))]); obs, _ = env.reset()
epr = deque(maxlen=30); er = 0.0; t0 = time.time(); best = -1e18
for i in range(1, STEPS+1):
    eps = max(0.05, 0.10 - 0.05*i/STEPS); beta = min(1.0, 0.4+0.6*i/STEPS)
    if random.random() < eps: a = random.randint(0, N_ACTIONS-1)
    else:
        with torch.no_grad(): a = online(torch.FloatTensor(obs)).argmax().item()
    nobs, r, term, trunc, info = env.step(a); done = term or trunc; er += r
    buf.add(obs, a, r, nobs, term); obs = nobs
    if done:
        epr.append(er); er = 0.0; env = Phase2Env_NYU(pool[rng.integers(len(pool))]); obs,_ = env.reset()
        if i > STEPS * 0.4 and len(epr) == epr.maxlen:          # best-checkpoint (post-warmup)
            m = float(np.mean(epr))
            if m > best:
                best = m
                torch.save({'q_net_state':online.state_dict(),'obs_dim':OBS_DIM,'n_actions':N_ACTIONS,
                            'net_arch':NET_ARCH,'algorithm':MODE.upper(),'n_steps':NSTEP,
                            'training_mode':'nyu_ci_channel_bestckpt'}, OUT)
    if i >= LSTART and i % TRAINFREQ == 0 and len(buf) >= BATCH:
        ib,wb,ob,ab,rb,nb,db = buf.sample(BATCH, beta)
        ot=torch.FloatTensor(ob); nt=torch.FloatTensor(nb); at=torch.LongTensor(ab)
        rt=torch.FloatTensor(rb); dt=torch.FloatTensor(db); wt=torch.FloatTensor(wb)
        with torch.no_grad():
            if DOUBLE:
                ba = online(nt).argmax(1, keepdim=True); nq = target(nt).gather(1, ba).squeeze(1)
            else:
                nq = target(nt).max(1).values
            tq = rt + (GAMMA**NSTEP) * nq * (1-dt)
        cq = online(ot).gather(1, at.unsqueeze(1)).squeeze(1)
        td = (cq-tq).detach().cpu().numpy()
        loss = (wt * nn.functional.huber_loss(cq, tq, reduction='none')).mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(online.parameters(),10.0); opt.step()
        buf.update_priorities(ib, td)
    if i % TARGUP == 0: polyak_update(online, target, TAU)
    if i % 30000 == 0:
        print(f"  [{i:7d}/{STEPS}] mean_rew={np.mean(epr) if epr else 0:8.2f} ({(time.time()-t0)/60:.1f} min)", flush=True)
if best <= -1e17:   # never best-checkpointed -> save final as fallback
    torch.save({'q_net_state':online.state_dict(),'obs_dim':OBS_DIM,'n_actions':N_ACTIONS,
                'net_arch':NET_ARCH,'algorithm':MODE.upper(),'n_steps':NSTEP,
                'training_mode':'nyu_ci_channel'}, OUT)
print(f"[save] {OUT} (best mean_rew={best:.2f}) ({(time.time()-t0)/60:.1f} min)", flush=True)
