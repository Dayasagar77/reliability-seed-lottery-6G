#!/usr/bin/env python3
"""handover_real_v6.py — HO metric redefined as an ACTUAL serving-cell change (not a
HANDOVER-action selection). Recomputes real HO rate + success for every method so Table IV
is fair. 5 seeds x 150 ep."""
import numpy as np, torch, csv
from env_6g_v6 import SubTHzEnv6G_v6
from env_6g_v5 import W_MOB, W_PWR, W_SPEC, HANDOVER, HOLD, A3_THRESH_DBM
from dndqn import load_dndqn_checkpoint
from qnet_plain import load_plain
from eval_pgrrm_baseline import pgrrm_action
import multi_seed_eval as mse
slow, med = mse.load_trajectories(mse.CSV_PATH); pool=slow+med
SEEDS, N_EP = mse.SEEDS, mse.N_EPISODES

def a3_action(last):
    if last is None: return HOLD
    return HANDOVER if last['rsrp_dbm'] < A3_THRESH_DBM else HOLD

def evaluate(kind, net=None):
    real=good=act=nep=0
    for seed in SEEDS:
        rng=np.random.default_rng(seed)
        for _ in range(N_EP):
            traj=pool[rng.integers(len(pool))]
            env=SubTHzEnv6G_v6(traj,w_mob=W_MOB,w_pwr=W_PWR,w_spec=W_SPEC)
            obs,_=env.reset(); done=False; last=None; nep+=1
            while not done:
                if kind=='net':
                    with torch.no_grad(): a=int(net(torch.FloatTensor(obs)).argmax().item())
                elif kind=='pgrrm': a=pgrrm_action(last)
                else: a=a3_action(last)
                if a==HANDOVER: act+=1
                pg=env.serving_gnb_idx
                obs,_,t,tr,info=env.step(a); last=info; done=t or tr
                if env.serving_gnb_idx!=pg:
                    real+=1
                    if not info.get('too_early') and not info.get('too_late'): good+=1
    return real/nep, 100*good/max(real,1), act/nep

configs=[('Proposed Dn-DQN','net',load_dndqn_checkpoint('dqn_phase2_dndqn_v6.pt')[0]),
         ('DDQN+PER','net',load_plain('dqn_ddqnper_v6.pt')[0]),
         ('Standard DQN','net',load_plain('dqn_stddqn_v6.pt')[0]),
         ('PG-RRM','pgrrm',None),
         ('3GPP A3','a3',None)]
print("="*74)
print("  HO redefined as ACTUAL serving-cell change | 5 seeds x 150 ep")
print("="*74)
print(f"  {'method':>18} | {'real HO/ep':>10} | {'HO-success':>10} | {'(HANDOVER acts/ep)':>18}")
print("  "+"-"*68)
rows=[]
for name,kind,net in configs:
    hoep,succ,actep=evaluate(kind,net); rows.append((name,hoep,succ,actep))
    print(f"  {name:>18} | {hoep:10.2f} | {succ:9.1f}% | {actep:18.1f}")
with open('results/handover_real_v6.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['method','real_HO_per_ep','HO_success_pct','HANDOVER_acts_per_ep'])
    for n,h,s,a in rows: w.writerow([n,f'{h:.3f}',f'{s:.2f}',f'{a:.2f}'])
print("  -> results/handover_real_v6.csv")
