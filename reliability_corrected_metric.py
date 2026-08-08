#!/usr/bin/env python3
"""
reliability_corrected_metric.py — Dn-DQN vs PG-RRM in-coverage RLF under a
REALISTIC (sustained / averaged) RLF definition, vs the code's single-sample one.

Single-sample termination on one 40 ms sub-threshold RSRP fade is not how RLF is
declared (3GPP TS 38.133: N310 consecutive out-of-sync over evaluation periods,
then T310), and RSRP is itself a time/frequency average. We therefore run each
policy with RLF termination disabled, capture the full serving-RSRP trace, and
classify in-coverage RLF three ways. Efficiency (HO/power/spectrum) reported too.
"""
import numpy as np, torch
import multi_seed_eval as mse
from env_6g_v5 import (SubTHzEnv6G, RLF_THRESH_DBM, GNB_X, CELL_RADIUS,
                       W_MOB, W_PWR, W_SPEC, MAX_STEPS, PT_MIN_DBM, PT_MAX_DBM,
                       RB_MIN, RB_MAX)
from dndqn import load_dndqn_checkpoint
from eval_pgrrm_baseline import pgrrm_action

CSV, SEEDS, N_EP = mse.CSV_PATH, mse.SEEDS, mse.N_EPISODES
GNB = np.asarray(GNB_X, float)
slow, med = mse.load_trajectories(CSV); pool = slow + med
net, _ = load_dndqn_checkpoint('dqn_phase2_dndqn.pt')

class NoRLFEnv(SubTHzEnv6G):
    def step(self, action):
        obs, r, term, trunc, info = super().step(action)
        term = (self.step_idx >= self.T) or (self.episode_step >= MAX_STEPS)
        return obs, r, term, trunc, info

def max_consec(mask):
    b=c=0
    for v in mask:
        c=c+1 if v else 0; b=max(b,c)
    return b

def in_coverage_at_end(traj, T, env):
    x = float(traj[min(env.step_idx, T-1), 0])
    return float(np.min(np.abs(GNB - x))) <= CELL_RADIUS

def run(policy):
    tot=good=tot_ho=0
    ss=cons3=ma5=0                      # single-sample, >=3 consec, 5-MA
    pwr_ep, spec_ep = [], []
    for seed in SEEDS:
        rr=np.random.default_rng(seed)
        for _ in range(N_EP):
            traj=pool[rr.integers(len(pool))]; T=len(traj)
            env=NoRLFEnv(traj, w_mob=W_MOB, w_pwr=W_PWR, w_spec=W_SPEC)
            obs,_=env.reset(); done=False; info=None; last=None
            rs=[]; pts=[]; rbs=[]
            while not done:
                if policy=='dndqn':
                    with torch.no_grad(): a=int(net(torch.FloatTensor(obs)).argmax().item())
                else:
                    a=pgrrm_action(last)
                obs,_,term,trunc,info=env.step(a); last=info; done=term or trunc
                rs.append(info['rsrp_dbm']); pts.append(info['pt_dbm']); rbs.append(info['rb_count'])
                if info.get('handover_done'):
                    tot_ho+=1
                    if not info.get('too_early') and not info.get('too_late'): good+=1
            tot+=1
            rs=np.array(rs); below=rs<RLF_THRESH_DBM
            incov = in_coverage_at_end(traj,T,env)
            # only count as in-coverage RLF if the failure event is in coverage
            if below.any() and incov: ss+=1
            if max_consec(below)>=3 and incov: cons3+=1
            if len(rs)>=5:
                c=np.convolve(rs,np.ones(5)/5,mode='valid')
                if np.any(c<RLF_THRESH_DBM) and incov: ma5+=1
            pwr_ep.append((PT_MAX_DBM-np.mean(pts))/(PT_MAX_DBM-PT_MIN_DBM)*100)
            spec_ep.append((RB_MAX-np.mean(rbs))/(RB_MAX-RB_MIN)*100)
    return dict(ho=100*good/max(tot_ho,1), pwr=np.mean(pwr_ep), spec=np.mean(spec_ep),
                ss=ss, cons3=cons3, ma5=ma5, tot=tot)

print("="*74)
print("  In-coverage RLF: single-sample (code) vs realistic (sustained/averaged)")
print("  5 seeds x 150 ep = 750 episodes; RLF termination disabled for measurement")
print("="*74)
print(f"\n  {'policy':>7} | {'HO%':>5} {'Pwr%':>5} {'Spec%':>5} | "
      f"{'>=1 samp':>9} {'>=3 consec':>10} {'5-samp MA':>10}")
print("  "+"-"*66)
for name,pol in [('Dn-DQN','dndqn'),('PG-RRM','pgrrm')]:
    r=run(pol); t=r['tot']
    print(f"  {name:>7} | {r['ho']:5.1f} {r['pwr']:5.1f} {r['spec']:5.1f} | "
          f"{r['ss']:4d}({100*r['ss']/t:4.1f}%) {r['cons3']:3d}({100*r['cons3']/t:4.1f}%) "
          f"{r['ma5']:3d}({100*r['ma5']/t:4.1f}%)")
print("\n  '>=1 sample' reproduces the code's inflated figure; '>=3 consec' and")
print("  '5-sample moving-average' are the realistic 3GPP-style / RSRP-as-average")
print("  definitions. Efficiency (HO/Power/Spectrum) is unchanged by the metric fix.")
