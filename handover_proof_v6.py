#!/usr/bin/env python3
"""Shows, step by step, that the env flags a 'handover' on EVERY HANDOVER action even when the
serving cell does not change. This is why 'success over handover_done events' (95.3%) differs
from 'success over actual serving-cell changes' (93.6%). Run it yourself."""
import numpy as np, torch
from env_6g_v6 import SubTHzEnv6G_v6
from env_6g_v5 import W_MOB, W_PWR, W_SPEC, HANDOVER
from dndqn import load_dndqn_checkpoint
import multi_seed_eval as mse
slow, med = mse.load_trajectories(mse.CSV_PATH); pool=slow+med
net,_ = load_dndqn_checkpoint('dqn_phase2_dndqn_v6.pt'); net.eval()
ACT=['HOLD','PWR_UP','PWR_DOWN','RB_UP','RB_DOWN','HANDOVER']
env=SubTHzEnv6G_v6(pool[0],w_mob=W_MOB,w_pwr=W_PWR,w_spec=W_SPEC)
obs,_=env.reset(); done=False; step=0; ho_done=0; real=0
print(f"  {'step':>4} {'action':>9} {'env says handover?':>18} {'serving gNB':>18} {'REAL change?':>12}")
print("  "+"-"*66)
while not done and step<45:
    with torch.no_grad(): a=int(net(torch.FloatTensor(obs)).argmax().item())
    gb=env.serving_gnb_idx
    obs,_,t,tr,info=env.step(a); done=t or tr; step+=1
    ga=env.serving_gnb_idx; hd=bool(info.get('handover_done')); chg=(ga!=gb)
    if hd: ho_done+=1
    if chg: real+=1
    if a==HANDOVER or chg:
        print(f"  {step:>4} {ACT[a]:>9} {str(hd):>18} {f'{gb} -> {ga}':>18} {('YES' if chg else 'no'):>12}")
print("  "+"-"*66)
print(f"  In these {step} steps: env flagged a handover {ho_done} times; serving cell ACTUALLY changed {real} time(s).")
