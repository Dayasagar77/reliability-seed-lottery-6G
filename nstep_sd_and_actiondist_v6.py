#!/usr/bin/env python3
"""nstep_sd_and_actiondist_v6.py — per-seed SDs for Table VI (n-step) and the v6 action
distribution + mean episode length for Fig 11. Consistent per-seed aggregation (like the
curriculum/arch tables)."""
import numpy as np, torch, json
from env_6g_v6 import SubTHzEnv6G_v6
from env_6g_v5 import W_MOB, W_PWR, W_SPEC, PT_MIN_DBM, PT_MAX_DBM, RB_MIN, RB_MAX
from dndqn import load_dndqn_checkpoint
import multi_seed_eval as mse
SEEDS, N_EP = mse.SEEDS, mse.N_EPISODES
slow, med = mse.load_trajectories(mse.CSV_PATH); pool=slow+med
ACT=['HOLD','PWR_UP','PWR_DOWN','RB_UP','RB_DOWN','HANDOVER']

def per_seed(path):
    net,_=load_dndqn_checkpoint(path); net.eval(); HO=[];PW=[];SP=[]
    for seed in SEEDS:
        rng=np.random.default_rng(seed); good=tot=0; P=[];S=[]
        for _ in range(N_EP):
            traj=pool[rng.integers(len(pool))]
            env=SubTHzEnv6G_v6(traj,w_mob=W_MOB,w_pwr=W_PWR,w_spec=W_SPEC)
            obs,_=env.reset(); done=False; info=None; pts=[];rbs=[]
            while not done:
                with torch.no_grad(): a=int(net(torch.FloatTensor(obs)).argmax().item())
                obs,_,t,tr,info=env.step(a); done=t or tr
                pts.append(info['pt_dbm']); rbs.append(info['rb_count'])
                if info.get('handover_done'):
                    tot+=1
                    if not info.get('too_early') and not info.get('too_late'): good+=1
            P.append((PT_MAX_DBM-np.mean(pts))/(PT_MAX_DBM-PT_MIN_DBM)*100)
            S.append((RB_MAX-np.mean(rbs))/(RB_MAX-RB_MIN)*100)
        HO.append(100*good/max(tot,1)); PW.append(np.mean(P)); SP.append(np.mean(S))
    return dict(ho=(np.mean(HO),np.std(HO)),pw=(np.mean(PW),np.std(PW)),sp=(np.mean(SP),np.std(SP)))

print("=== N-STEP per-seed SD (5 seeds x 150 ep) ===")
res={}
for name,path in [('n=1','dqn_n1_v6.pt'),('n=3','dqn_phase2_dndqn_v6.pt'),('n=5','dqn_n5_v6.pt')]:
    r=per_seed(path); res[name]=r
    print(f"{name}: HO {r['ho'][0]:.1f}+-{r['ho'][1]:.1f}  Power {r['pw'][0]:.1f}+-{r['pw'][1]:.1f}  Spec {r['sp'][0]:.1f}+-{r['sp'][1]:.1f}")

net,_=load_dndqn_checkpoint('dqn_phase2_dndqn_v6.pt'); net.eval()
acts=np.zeros(6); nsteps=0; eplens=[]
for seed in SEEDS:
    rng=np.random.default_rng(seed)
    for _ in range(N_EP):
        traj=pool[rng.integers(len(pool))]
        env=SubTHzEnv6G_v6(traj,w_mob=W_MOB,w_pwr=W_PWR,w_spec=W_SPEC)
        obs,_=env.reset(); done=False; L=0
        while not done:
            with torch.no_grad(): a=int(net(torch.FloatTensor(obs)).argmax().item())
            acts[a]+=1; nsteps+=1; L+=1
            obs,_,t,tr,info=env.step(a); done=t or tr
        eplens.append(L)
dist={ACT[i]:round(100*acts[i]/nsteps,1) for i in range(6)}
print("\n=== ACTION DISTRIBUTION (canonical model, v6) ===")
for k,v in dist.items(): print(f"  {k}: {v}%")
print(f"  mean episode length: {np.mean(eplens):.1f} steps  (median {np.median(eplens):.0f})")
json.dump({'nstep':{k:{'ho':v['ho'],'pw':v['pw'],'sp':v['sp']} for k,v in res.items()},
           'actions':dist,'eplen':round(float(np.mean(eplens)),1)},
          open('results/nstep_sd_actiondist_v6.json','w'))
print("-> results/nstep_sd_actiondist_v6.json")
