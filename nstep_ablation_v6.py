#!/usr/bin/env python3
"""nstep_ablation_v6.py — n-step ablation under the corrected channel (v6).
Evaluates n=1, n=3 (proposed), n=5, all retrained under the 380 MHz + sustained-RLF
env, on HO/power/spectrum/RLF. 5 seeds x 150 ep."""
import numpy as np, torch, csv
from env_6g_v6 import SubTHzEnv6G_v6
from env_6g_v5 import (RLF_THRESH_DBM, GNB_X, CELL_RADIUS, W_MOB, W_PWR, W_SPEC,
                       PT_MIN_DBM, PT_MAX_DBM, RB_MIN, RB_MAX)
from dndqn import load_dndqn_checkpoint
import multi_seed_eval as mse

SEEDS, N_EP = mse.SEEDS, mse.N_EPISODES
GNB = np.asarray(GNB_X, float)
slow, med = mse.load_trajectories(mse.CSV_PATH); pool = slow + med

def evaluate(path):
    net,_ = load_dndqn_checkpoint(path); net.eval()
    P=[];S=[]; good=totho=sus=0; n=0
    for seed in SEEDS:
        rng=np.random.default_rng(seed)
        for _ in range(N_EP):
            traj=pool[rng.integers(len(pool))]; T=len(traj)
            env=SubTHzEnv6G_v6(traj,w_mob=W_MOB,w_pwr=W_PWR,w_spec=W_SPEC)
            obs,_=env.reset(); done=False; info=None; pts=[];rbs=[]
            while not done:
                with torch.no_grad(): a=int(net(torch.FloatTensor(obs)).argmax().item())
                obs,_,term,trunc,info=env.step(a); done=term or trunc
                pts.append(info['pt_dbm']); rbs.append(info['rb_count'])
                if info.get('handover_done'):
                    totho+=1
                    if not info.get('too_early') and not info.get('too_late'): good+=1
            n+=1
            P.append((PT_MAX_DBM-np.mean(pts))/(PT_MAX_DBM-PT_MIN_DBM)*100)
            S.append((RB_MAX-np.mean(rbs))/(RB_MAX-RB_MIN)*100)
            x=float(traj[min(int(info['step']),T-1),0])
            if info.get('sustained_rlf') and float(np.min(np.abs(GNB-x)))<=CELL_RADIUS: sus+=1
    return dict(ho=100*good/max(totho,1), pwr=(np.mean(P),np.std(P)),
               spec=(np.mean(S),np.std(S)), rlf=100*sus/n)

configs=[('n=1','dqn_n1_v6.pt'),('n=3 (proposed)','dqn_phase2_dndqn_v6.pt'),('n=5','dqn_n5_v6.pt')]
print("="*70)
print("  N-step ablation under corrected channel (380 MHz, sustained RLF)")
print("="*70)
print(f"\n  {'config':>14} | {'HO%':>6} | {'Power%':>12} | {'Spec%':>12} | {'RLF%':>5}")
print("  "+"-"*58)
rows=[]
for name,path in configs:
    r=evaluate(path); rows.append((name,r))
    print(f"  {name:>14} | {r['ho']:6.1f} | {r['pwr'][0]:5.1f}±{r['pwr'][1]:<4.1f} | "
          f"{r['spec'][0]:5.1f}±{r['spec'][1]:<4.1f} | {r['rlf']:5.1f}")
n3=dict(rows)['n=3 (proposed)']
print(f"\n  spectrum: n=1 -> n=3 = {n3['spec'][0]-dict(rows)['n=1']['spec'][0]:+.1f} pp ; "
      f"n=3 -> n=5 = {dict(rows)['n=5']['spec'][0]-n3['spec'][0]:+.1f} pp")
with open('results/nstep_ablation_v6.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['config','HO','Power','Power_sd','Spec','Spec_sd','RLF'])
    for name,r in rows: w.writerow([name,f"{r['ho']:.2f}",f"{r['pwr'][0]:.2f}",f"{r['pwr'][1]:.2f}",
                                    f"{r['spec'][0]:.2f}",f"{r['spec'][1]:.2f}",f"{r['rlf']:.2f}"])
print("  -> results/nstep_ablation_v6.csv")
