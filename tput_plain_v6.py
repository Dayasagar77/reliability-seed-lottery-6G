#!/usr/bin/env python3
"""tput_plain_v6.py — corrected HO/Power/Spec/Throughput/RLF for the two plain-net
architecture ablations (DDQN+PER, Standard DQN), identical eval_v6 throughput basis,
so Table IV rows are complete and consistent. 5 seeds x 150 ep."""
import math, numpy as np, torch, csv
from env_6g_v6 import SubTHzEnv6G_v6
from env_6g_v5 import (GNB_X, CELL_RADIUS, W_MOB, W_PWR, W_SPEC,
                       PT_MIN_DBM, PT_MAX_DBM, RB_MIN, RB_MAX)
from qnet_plain import load_plain
import multi_seed_eval as mse
SEEDS, N_EP = mse.SEEDS, mse.N_EPISODES
GNB=np.asarray(GNB_X,float); RB_BW_HZ=380.16e6/RB_MAX
slow,med=mse.load_trajectories(mse.CSV_PATH); pool=slow+med
def ev(path):
    net,_=load_plain(path); net.eval(); good=tot=sus=0; n=0; P=[];S=[];T=[]
    for seed in SEEDS:
        rng=np.random.default_rng(seed)
        for _ in range(N_EP):
            traj=pool[rng.integers(len(pool))]; Tlen=len(traj)
            env=SubTHzEnv6G_v6(traj,w_mob=W_MOB,w_pwr=W_PWR,w_spec=W_SPEC)
            obs,_=env.reset(); done=False; info=None; pts=[];rbs=[];tp=[]
            while not done:
                with torch.no_grad(): a=int(net(torch.FloatTensor(obs)).argmax().item())
                obs,_,term,trunc,info=env.step(a); done=term or trunc
                pts.append(info['pt_dbm']); rbs.append(info['rb_count'])
                sdb=obs[4]*100.0; tp.append(info['rb_count']*(RB_BW_HZ/1e6)*math.log2(1+10**(sdb/10)))
                if info.get('handover_done'):
                    tot+=1
                    if not info.get('too_early') and not info.get('too_late'): good+=1
            n+=1; P.append((PT_MAX_DBM-np.mean(pts))/(PT_MAX_DBM-PT_MIN_DBM)*100)
            S.append((RB_MAX-np.mean(rbs))/(RB_MAX-RB_MIN)*100); T.append(np.mean(tp))
            x=float(traj[min(int(info['step']),Tlen-1),0])
            if info.get('sustained_rlf') and float(np.min(np.abs(GNB-x)))<=CELL_RADIUS: sus+=1
    return (100*good/max(tot,1), np.mean(P), np.std(P), np.mean(S), np.std(S),
            np.mean(T), np.std(T), 100*sus/n)
for name,path in [('DDQN+PER','dqn_ddqnper_v6.pt'),('Standard DQN','dqn_stddqn_v6.pt')]:
    ho,p,ps,s,ss,t,ts,rlf=ev(path)
    print(f"{name:14} HO {ho:.1f}  Power {p:.1f}±{ps:.1f}  Spec {s:.1f}±{ss:.1f}  Tput {t:.1f}±{ts:.1f}  RLF {rlf:.1f}")
