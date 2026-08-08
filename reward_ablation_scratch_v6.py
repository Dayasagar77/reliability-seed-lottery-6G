#!/usr/bin/env python3
"""reward_ablation_scratch_v6.py — FROM-SCRATCH reward-component ablation (unbiased).
Full Dn-DQN vs no-C1 (RSRP-gated power) vs no-C3 (sustained-RLF penalty) vs no-C5
(symmetric RB), each trained from RANDOM INIT under the corrected channel (no warm-start
bias). 5 seeds x 150 ep. Reports RB_DOWN action share (C5) and sustained RLF (C1/C3).
Run only the configs whose checkpoints exist, so you can eval a partially-finished set."""
import os, numpy as np, torch, csv
from env_6g_v6 import SubTHzEnv6G_v6
from env_6g_v5 import (GNB_X, CELL_RADIUS, W_MOB, W_PWR, W_SPEC, RB_DOWN, PWR_DOWN,
                       PT_MIN_DBM, PT_MAX_DBM, RB_MIN, RB_MAX)
from dndqn import load_dndqn_checkpoint
import multi_seed_eval as mse

SEEDS, N_EP = mse.SEEDS, mse.N_EPISODES
GNB = np.asarray(GNB_X, float)
slow, med = mse.load_trajectories(mse.CSV_PATH); pool = slow + med

def evaluate(path):
    net,_ = load_dndqn_checkpoint(path); net.eval()
    good=totho=sus=0; n=0; acts=np.zeros(6); nsteps=0; P=[];S=[]
    for seed in SEEDS:
        rng=np.random.default_rng(seed)
        for _ in range(N_EP):
            traj=pool[rng.integers(len(pool))]; T=len(traj)
            env=SubTHzEnv6G_v6(traj,w_mob=W_MOB,w_pwr=W_PWR,w_spec=W_SPEC)
            obs,_=env.reset(); done=False; info=None; pts=[];rbs=[]
            while not done:
                with torch.no_grad(): a=int(net(torch.FloatTensor(obs)).argmax().item())
                acts[a]+=1; nsteps+=1
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
    return dict(ho=100*good/max(totho,1), pwr=np.mean(P), spec=np.mean(S),
                rlf=100*sus/n, rb_down=100*acts[RB_DOWN]/nsteps, pwr_down=100*acts[PWR_DOWN]/nsteps)

configs=[('Full from-scratch (C1+C3+C5)','dqn_scratch_none_v6.pt'),
         ('no C1 (power gate off)',       'dqn_scratch_c1_v6.pt'),
         ('no C3 (RLF penalty off)',      'dqn_scratch_c3_v6.pt'),
         ('no C5 (RB symmetry off)',      'dqn_scratch_c5_v6.pt')]
print("="*80)
print("  Reward-component ablation, FROM SCRATCH under corrected channel (unbiased effects)")
print("="*80)
print(f"\n  {'config':>28} | {'HO%':>5} {'Pwr%':>5} {'Spec%':>6} {'RLF%':>5} | "
      f"{'RB_DOWN%':>8} {'PWR_DOWN%':>9}")
print("  "+"-"*78)
rows=[]
for name,path in configs:
    if not os.path.exists(path):
        print(f"  {name:>28} | (checkpoint {path} not found — skipped)"); continue
    r=evaluate(path); rows.append((name,r))
    print(f"  {name:>28} | {r['ho']:5.1f} {r['pwr']:5.1f} {r['spec']:6.1f} {r['rlf']:5.1f} | "
          f"{r['rb_down']:8.1f} {r['pwr_down']:9.1f}")
print("\n  Reading: vs 'Full', no-C5 -> RB_DOWN share + spectrum% fall; no-C3 -> sustained RLF rises;")
print("           no-C1 -> power% / RLF shift. From-scratch = each component's TRUE contribution.")
if rows:
    os.makedirs('results', exist_ok=True)
    with open('results/reward_ablation_scratch_v6.csv','w',newline='') as f:
        w=csv.writer(f); w.writerow(['config','HO','Power','Spec','RLF','RB_DOWN_pct','PWR_DOWN_pct'])
        for name,r in rows: w.writerow([name,f"{r['ho']:.2f}",f"{r['pwr']:.2f}",f"{r['spec']:.2f}",
                                        f"{r['rlf']:.2f}",f"{r['rb_down']:.2f}",f"{r['pwr_down']:.2f}"])
    print("  -> results/reward_ablation_scratch_v6.csv")
