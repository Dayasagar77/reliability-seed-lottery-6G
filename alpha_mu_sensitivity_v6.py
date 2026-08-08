#!/usr/bin/env python3
"""alpha_mu_sensitivity_v6.py — sensitivity of the corrected results to the alpha-mu fading
parameters. Overrides channel_model.ALPHA_LOS/MU_LOS across the literature-reported range and
re-evaluates the SAME trained policy. Shows whether RLF/HO/power/spectrum are robust to the
(indoor-placeholder) fading assumption. 3 seeds x 100 ep per point."""
import numpy as np, torch
import channel_model as cm
from dndqn import load_dndqn_checkpoint
import multi_seed_eval as mse
from env_6g_v5 import (GNB_X, CELL_RADIUS, W_MOB, W_PWR, W_SPEC,
                       PT_MIN_DBM, PT_MAX_DBM, RB_MIN, RB_MAX)
from env_6g_v6 import SubTHzEnv6G_v6
GNB=np.asarray(GNB_X,float)
slow,med=mse.load_trajectories(mse.CSV_PATH); pool=slow+med
net,_=load_dndqn_checkpoint('dqn_phase2_dndqn_v6.pt'); net.eval()
SEEDS=[0,1,2]; N_EP=100
BASE_A, BASE_MU = cm.ALPHA_LOS, cm.MU_LOS

def evaluate():
    good=tot=sus=0; n=0; P=[]; S=[]
    for seed in SEEDS:
        rng=np.random.default_rng(seed)
        for _ in range(N_EP):
            traj=pool[rng.integers(len(pool))]; T=len(traj)
            env=SubTHzEnv6G_v6(traj,w_mob=W_MOB,w_pwr=W_PWR,w_spec=W_SPEC)
            obs,_=env.reset(); done=False; info=None; pts=[]; rbs=[]
            while not done:
                with torch.no_grad(): a=int(net(torch.FloatTensor(obs)).argmax().item())
                obs,_,term,trunc,info=env.step(a); done=term or trunc
                pts.append(info['pt_dbm']); rbs.append(info['rb_count'])
                if info.get('handover_done'):
                    tot+=1
                    if not info.get('too_early') and not info.get('too_late'): good+=1
            n+=1; P.append((PT_MAX_DBM-np.mean(pts))/(PT_MAX_DBM-PT_MIN_DBM)*100)
            S.append((RB_MAX-np.mean(rbs))/(RB_MAX-RB_MIN)*100)
            x=float(traj[min(int(info['step']),T-1),0])
            if info.get('sustained_rlf') and float(np.min(np.abs(GNB-x)))<=CELL_RADIUS: sus+=1
    return 100*good/max(tot,1), np.mean(P), np.mean(S), 100*sus/n

rows=[]
def run(tag,a,mu):
    cm.ALPHA_LOS=a; cm.MU_LOS=mu
    ho,p,s,rlf=evaluate(); rows.append((tag,a,mu,ho,p,s,rlf))
    print(f"  {tag:16} a={a:.3f} mu={mu:.3f} | HO {ho:5.1f}  Power {p:5.1f}  Spec {s:5.1f}  RLF {rlf:5.2f}%",flush=True)

print("="*84)
print("  Alpha-mu fading sensitivity (LoS params), same trained policy | 3 seeds x 100 ep")
print("="*84)
run("baseline",BASE_A,BASE_MU)
print("  --- vary mu (cluster count) at alpha=2.0 ---")
for mu in [0.6,1.0,1.5,2.0,3.0]: run(f"mu={mu}",2.0,mu)
print("  --- vary alpha (nonlinearity) at mu=1.0 ---")
for a in [1.8,2.0,2.5,3.0,3.5]: run(f"alpha={a}",a,1.0)
cm.ALPHA_LOS,cm.MU_LOS=BASE_A,BASE_MU
rl=[r[6] for r in rows]; ho=[r[3] for r in rows]; pw=[r[4] for r in rows]; sp=[r[5] for r in rows]
print("\n  RANGE across all fading settings:")
print(f"    RLF     {min(rl):.2f}% .. {max(rl):.2f}%   (spread {max(rl)-min(rl):.2f} pp)")
print(f"    HO      {min(ho):.1f}% .. {max(ho):.1f}%")
print(f"    Power   {min(pw):.1f}% .. {max(pw):.1f}%")
print(f"    Spec    {min(sp):.1f}% .. {max(sp):.1f}%")
import csv
with open('results/alpha_mu_sensitivity_v6.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['setting','alpha','mu','HO','Power','Spec','RLF'])
    for r in rows: w.writerow([r[0],r[1],r[2],f"{r[3]:.2f}",f"{r[4]:.2f}",f"{r[5]:.2f}",f"{r[6]:.3f}"])
print("  -> results/alpha_mu_sensitivity_v6.csv")
