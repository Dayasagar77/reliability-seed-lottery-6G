#!/usr/bin/env python3
"""
finetune_v6.py — adapt the Dn-DQN policy to the corrected sustained-RLF env (v6).

Warm-starts from the existing trained checkpoint and continues training under
Phase2Env_v6 (sustained RLF, graded floor penalty, terminal penalty on sustained
failure only). Foreground, single run, resumable via a step sidecar.
Produces dqn_phase2_dndqn_v6.pt.
"""
import os, sys, time, random, json
from collections import deque
import numpy as np, torch, torch.nn as nn, torch.optim as optim

from dndqn import build_dndqn_networks, NStepPERBuffer, polyak_update, OBS_DIM, NET_ARCH
from env_6g_v5 import N_ACTIONS
import train_phase2 as tp
from train_phase2_v6 import Phase2Env_v6

TARGET   = int(sys.argv[1]) if len(sys.argv) > 1 else 150_000
CHUNK    = int(sys.argv[2]) if len(sys.argv) > 2 else TARGET
OUT      = "dqn_phase2_dndqn_v6.pt"
SIDE     = "finetune_v6_step.json"
GAMMA, NSTEP, LR = 0.99, 3, 3e-4
BATCH, BUF, TRAINFREQ, TARGUP, TAU, LSTART = 256, 300_000, 4, 500, 0.01, 3_000

slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH); pool = slow + med
online, target = build_dndqn_networks(OBS_DIM, N_ACTIONS, NET_ARCH, 'cpu')

gstep = 0
if os.path.exists(OUT) and os.path.exists(SIDE):
    online.load_state_dict(torch.load(OUT, map_location='cpu')['q_net_state'])
    gstep = json.load(open(SIDE)).get('global_step', 0)
    print(f"[resume] {OUT} at global_step={gstep}")
else:
    online.load_state_dict(torch.load('dqn_phase2_dndqn.pt', map_location='cpu')['q_net_state'])
    print("[init] warm-started from dqn_phase2_dndqn.pt (v5)")
target.load_state_dict(online.state_dict())
opt = optim.Adam(online.parameters(), lr=LR)
buf = NStepPERBuffer(BUF, n_steps=NSTEP, gamma=GAMMA, alpha=0.6)
rng = np.random.default_rng(1234 + gstep)

def eps_at(g):   # low exploration for fine-tuning: 0.10 -> 0.05 across TARGET
    return max(0.05, 0.10 - 0.05 * g / max(TARGET, 1))

env = Phase2Env_v6(pool[rng.integers(len(pool))]); obs, _ = env.reset()
epr = deque(maxlen=30); er = 0.0; t0 = time.time()
n_this = min(CHUNK, TARGET - gstep)
print(f"[train] this run: {n_this:,} steps (target {TARGET:,})", flush=True)
for i in range(1, n_this + 1):
    g = gstep + i
    eps = eps_at(g); beta = min(1.0, 0.4 + 0.6 * g / TARGET)
    if random.random() < eps: a = random.randint(0, N_ACTIONS - 1)
    else:
        with torch.no_grad(): a = online(torch.FloatTensor(obs)).argmax().item()
    nobs, r, term, trunc, info = env.step(a); done = term or trunc; er += r
    buf.add(obs, a, r, nobs, term); obs = nobs
    if done:
        epr.append(er); er = 0.0
        env = Phase2Env_v6(pool[rng.integers(len(pool))]); obs, _ = env.reset()
    if i >= LSTART and i % TRAINFREQ == 0 and len(buf) >= BATCH:
        ib, wb, ob, ab, rb, nb, db = buf.sample(BATCH, beta)
        ot = torch.FloatTensor(ob); nt = torch.FloatTensor(nb); at = torch.LongTensor(ab)
        rt = torch.FloatTensor(rb); dt = torch.FloatTensor(db); wt = torch.FloatTensor(wb)
        with torch.no_grad():
            ba = online(nt).argmax(1, keepdim=True); nq = target(nt).gather(1, ba).squeeze(1)
            tq = rt + (GAMMA ** NSTEP) * nq * (1 - dt)
        cq = online(ot).gather(1, at.unsqueeze(1)).squeeze(1)
        td = (cq - tq).detach().cpu().numpy()
        loss = (wt * nn.functional.huber_loss(cq, tq, reduction='none')).mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(online.parameters(), 10.0); opt.step()
        buf.update_priorities(ib, td)
    if g % TARGUP == 0: polyak_update(online, target, TAU)
    if i % 10000 == 0:
        print(f"[{g:7d}/{TARGET}] mean_rew={np.mean(epr) if epr else 0:9.2f} eps={eps:.3f} "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)
torch.save({'q_net_state': online.state_dict(), 'obs_dim': OBS_DIM, 'n_actions': N_ACTIONS,
            'net_arch': NET_ARCH, 'algorithm': 'Dn-DQN', 'phase': 2,
            'w_mob': 0.60, 'w_pwr': 0.20, 'w_spec': 0.20, 'n_steps': NSTEP,
            'training_mode': 'v6_sustained_rlf_finetune', 'n310': 3}, OUT)
json.dump({'global_step': gstep + n_this}, open(SIDE, 'w'))
print(f"[save] {OUT} @ step {gstep + n_this} ({(time.time()-t0)/60:.1f} min)", flush=True)
