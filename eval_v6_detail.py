#!/usr/bin/env python3
"""eval_v6_detail.py — episode-level bootstrap 95% CIs + velocity stratification
for the retrained Dn-DQN under env_6g_v6. Matches the paper's CI methodology
(episode-level, 10k resamples)."""
import sys, math, numpy as np, torch
from env_6g_v6 import SubTHzEnv6G_v6
from env_6g_v5 import (RLF_THRESH_DBM, GNB_X, CELL_RADIUS, W_MOB, W_PWR, W_SPEC,
                       PT_MIN_DBM, PT_MAX_DBM, RB_MIN, RB_MAX)
from dndqn import load_dndqn_checkpoint
import multi_seed_eval as mse

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'dqn_phase2_dndqn_v6.pt'
SEEDS, N_EP = mse.SEEDS, mse.N_EPISODES
GNB = np.asarray(GNB_X, float); RB_BW_HZ = 380.16e6 / RB_MAX
slow, med = mse.load_trajectories(mse.CSV_PATH)
net,_ = load_dndqn_checkpoint(MODEL); net.eval()
print(f"  MODEL = {MODEL}")
rngb = np.random.default_rng(7)

def ci(vals, n=10000):
    v=np.asarray(vals); idx=rngb.integers(0,len(v),size=(n,len(v)))
    means=v[idx].mean(1); return v.mean(), np.percentile(means,2.5), np.percentile(means,97.5)

def collect(pool, tag):
    P=[]; S=[]; Tp=[]; good=tot=0; sus=0; N=0
    for seed in SEEDS:
        rng=np.random.default_rng(seed)
        for _ in range(N_EP):
            traj=pool[rng.integers(len(pool))]; T=len(traj)
            env=SubTHzEnv6G_v6(traj,w_mob=W_MOB,w_pwr=W_PWR,w_spec=W_SPEC)
            obs,_=env.reset(); done=False; info=None; pts=[]; rbs=[]; tp=[]
            while not done:
                with torch.no_grad(): a=int(net(torch.FloatTensor(obs)).argmax().item())
                obs,_,term,trunc,info=env.step(a); done=term or trunc
                pts.append(info['pt_dbm']); rbs.append(info['rb_count'])
                sdb=obs[4]*100.0; tp.append(info['rb_count']*(RB_BW_HZ/1e6)*math.log2(1+10**(sdb/10)))
                if info.get('handover_done'):
                    tot+=1
                    if not info.get('too_early') and not info.get('too_late'): good+=1
            P.append((PT_MAX_DBM-np.mean(pts))/(PT_MAX_DBM-PT_MIN_DBM)*100)
            S.append((RB_MAX-np.mean(rbs))/(RB_MAX-RB_MIN)*100); Tp.append(np.mean(tp)); N+=1
            x=float(traj[min(int(info['step']),T-1),0])
            if info.get('sustained_rlf') and float(np.min(np.abs(GNB-x)))<=CELL_RADIUS: sus+=1
    ho=100*good/max(tot,1)
    print(f"\n  [{tag}]  N={N} episodes,  HO(per-event)={ho:.1f}%,  sustained in-cov RLF={sus} ({100*sus/N:.1f}%)")
    for nm,v in [('Power %',P),('Spectrum %',S),('Throughput Mbps',Tp)]:
        m,lo,hi=ci(v); print(f"     {nm:16}: {m:7.1f}   95% CI [{lo:6.1f}, {hi:6.1f}]")
    return ho, sus, N

print("="*70); print("  Dn-DQN v6 — bootstrap CIs + velocity stratification"); print("="*70)
collect(slow+med, "ALL 5-40 m/s")
collect(slow,     "SLOW 5-25 m/s")
collect(med,      "MEDIUM 25-40 m/s")
