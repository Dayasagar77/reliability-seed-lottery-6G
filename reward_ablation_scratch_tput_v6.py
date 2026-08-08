#!/usr/bin/env python3
"""reward_ablation_scratch_tput_v6.py — throughput side of the from-scratch reward ablation.
Uses the IDENTICAL throughput formula as eval_v6.py (RB_BW=380.16e6/RB_MAX, Shannon on obs[4]
SINR). Tells us whether the ablations' higher Spec% (more RB-down) is a real gain or a
reward-hack that sacrifices throughput. 5 seeds x 150 ep."""
import os, math, numpy as np, torch, csv
from env_6g_v6 import SubTHzEnv6G_v6
from env_6g_v5 import (GNB_X, CELL_RADIUS, W_MOB, W_PWR, W_SPEC, RB_DOWN, PWR_DOWN,
                       PT_MIN_DBM, PT_MAX_DBM, RB_MIN, RB_MAX)
from dndqn import load_dndqn_checkpoint
import multi_seed_eval as mse

SEEDS, N_EP = mse.SEEDS, mse.N_EPISODES
GNB = np.asarray(GNB_X, float)
RB_BW_HZ = 380.16e6 / RB_MAX                       # identical basis to eval_v6.py
slow, med = mse.load_trajectories(mse.CSV_PATH); pool = slow + med

def evaluate(path):
    net,_ = load_dndqn_checkpoint(path); net.eval()
    per=dict(ho=[],pwr=[],spec=[],tput=[],rb=[]); sus=0; n=0; acts=np.zeros(6); nsteps=0
    for seed in SEEDS:
        rng=np.random.default_rng(seed); good=totho=0; P=[];S=[];Tp=[];RB=[]
        for _ in range(N_EP):
            traj=pool[rng.integers(len(pool))]; T=len(traj)
            env=SubTHzEnv6G_v6(traj,w_mob=W_MOB,w_pwr=W_PWR,w_spec=W_SPEC)
            obs,_=env.reset(); done=False; info=None; pts=[];rbs=[];ep_t=[]
            while not done:
                with torch.no_grad(): a=int(net(torch.FloatTensor(obs)).argmax().item())
                acts[a]+=1; nsteps+=1
                obs,_,term,trunc,info=env.step(a); done=term or trunc
                pts.append(info['pt_dbm']); rbs.append(info['rb_count'])
                sdb=obs[4]*100.0; ep_t.append(info['rb_count']*(RB_BW_HZ/1e6)*math.log2(1+10**(sdb/10)))
                if info.get('handover_done'):
                    totho+=1
                    if not info.get('too_early') and not info.get('too_late'): good+=1
            n+=1
            P.append((PT_MAX_DBM-np.mean(pts))/(PT_MAX_DBM-PT_MIN_DBM)*100)
            S.append((RB_MAX-np.mean(rbs))/(RB_MAX-RB_MIN)*100)
            Tp.append(np.mean(ep_t)); RB.append(np.mean(rbs))
            x=float(traj[min(int(info['step']),T-1),0])
            if info.get('sustained_rlf') and float(np.min(np.abs(GNB-x)))<=CELL_RADIUS: sus+=1
        per['ho'].append(100*good/max(totho,1)); per['pwr'].append(np.mean(P))
        per['spec'].append(np.mean(S)); per['tput'].append(np.mean(Tp)); per['rb'].append(np.mean(RB))
    agg={k:(np.mean(v),np.std(v)) for k,v in per.items()}
    return agg, 100*sus/n, 100*acts[RB_DOWN]/nsteps

configs=[('Full from-scratch (C1+C3+C5)','dqn_scratch_none_v6.pt'),
         ('no C1 (power gate off)',       'dqn_scratch_c1_v6.pt'),
         ('no C3 (RLF penalty off)',      'dqn_scratch_c3_v6.pt'),
         ('no C5 (RB symmetry off)',      'dqn_scratch_c5_v6.pt')]
print("="*96)
print("  From-scratch reward ablation WITH throughput (corrected 380 MHz basis) | 5 seeds x 150 ep")
print("="*96)
print(f"\n  {'config':>28} | {'HO%':>5} {'Pwr%':>5} {'Spec%':>6} {'Tput(Mbps)':>12} {'meanRB':>7} {'RLF%':>5} {'RB_DOWN%':>8}")
print("  "+"-"*92)
rows=[]
for name,path in configs:
    if not os.path.exists(path): print(f"  {name:>28} | (missing {path})"); continue
    agg,rlf,rbd=evaluate(path); rows.append((name,agg,rlf,rbd))
    print(f"  {name:>28} | {agg['ho'][0]:5.1f} {agg['pwr'][0]:5.1f} {agg['spec'][0]:6.1f} "
          f"{agg['tput'][0]:8.1f}±{agg['tput'][1]:<3.0f} {agg['rb'][0]:7.1f} {rlf:5.1f} {rbd:8.1f}")
os.makedirs('results', exist_ok=True)
with open('results/reward_ablation_scratch_tput_v6.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['config','HO','Power','Spec','Tput_Mbps','Tput_sd','mean_RB','RLF','RB_DOWN_pct'])
    for name,agg,rlf,rbd in rows:
        w.writerow([name,f"{agg['ho'][0]:.2f}",f"{agg['pwr'][0]:.2f}",f"{agg['spec'][0]:.2f}",
                    f"{agg['tput'][0]:.2f}",f"{agg['tput'][1]:.2f}",f"{agg['rb'][0]:.2f}",
                    f"{rlf:.2f}",f"{rbd:.2f}"])
print("\n  KEY: if no-C5 shows higher Spec% but LOWER throughput than Full, C5 is validated as an")
print("       anti-reward-hack term (it stops the agent farming spectrum% by starving throughput).")
print("  -> results/reward_ablation_scratch_tput_v6.csv")
