#!/usr/bin/env python3
"""
eval_v6.py — consistent evaluation of all re-runnable policies under the corrected
sustained-RLF environment (env_6g_v6). 5 seeds x 150 ep = 750 episodes each.

Policies: Dn-DQN (retrained under v6), Dn-DQN (v5 model, consistency check),
          PG-RRM (rule-based), 3GPP A3 (rule-based, max power/RB, reactive HO).
Metrics : HO success (per-HO-event), Power/Spectrum saving %, Shannon throughput,
          in-coverage RLF under BOTH the old single-sample rule and the corrected
          sustained (N310=3) rule.
"""
import math, numpy as np, torch, csv
from env_6g_v6 import SubTHzEnv6G_v6
from env_6g_v5 import (RLF_THRESH_DBM, A3_THRESH_DBM, HO_PROACTIVE_THRESH, GNB_X,
                       CELL_RADIUS, W_MOB, W_PWR, W_SPEC, PT_MIN_DBM, PT_MAX_DBM,
                       RB_MIN, RB_MAX, HOLD, HANDOVER)
from dndqn import load_dndqn_checkpoint
from eval_pgrrm_baseline import pgrrm_action
import multi_seed_eval as mse

SEEDS, N_EP = mse.SEEDS, mse.N_EPISODES
GNB = np.asarray(GNB_X, float)
RB_BW_HZ = 380.16e6 / RB_MAX
slow, med = mse.load_trajectories(mse.CSV_PATH); pool = slow + med

def a3_action(last):
    if last is None: return HOLD
    return HANDOVER if last['rsrp_dbm'] < A3_THRESH_DBM else HOLD   # max pwr/RB, reactive HO

def get_net(path):
    net, _ = load_dndqn_checkpoint(path); net.eval(); return net

def eval_policy(kind, net=None):
    per_seed = {k: [] for k in ('ho','pwr','spec','tput')}
    ss_incov = sus_incov = 0                       # single-sample vs sustained in-cov RLF
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        good = totho = 0; pwr=[]; spec=[]; tput=[]
        for _ in range(N_EP):
            traj = pool[rng.integers(len(pool))]; T=len(traj)
            env = SubTHzEnv6G_v6(traj, w_mob=W_MOB, w_pwr=W_PWR, w_spec=W_SPEC)
            obs,_ = env.reset(); done=False; last=None; info=None
            pts=[]; rbs=[]; ep_t=[]; any_below=False
            while not done:
                if kind=='net':
                    with torch.no_grad(): a=int(net(torch.FloatTensor(obs)).argmax().item())
                elif kind=='pgrrm': a=pgrrm_action(last)
                else:               a=a3_action(last)
                obs,_,term,trunc,info = env.step(a); last=info; done=term or trunc
                pts.append(info['pt_dbm']); rbs.append(info['rb_count'])
                sdb=obs[4]*100.0; ep_t.append(info['rb_count']*(RB_BW_HZ/1e6)*math.log2(1+10**(sdb/10)))
                if info['rsrp_dbm']<RLF_THRESH_DBM: any_below=True
                if info.get('handover_done'):
                    totho+=1
                    if not info.get('too_early') and not info.get('too_late'): good+=1
            pwr.append((PT_MAX_DBM-np.mean(pts))/(PT_MAX_DBM-PT_MIN_DBM)*100)
            spec.append((RB_MAX-np.mean(rbs))/(RB_MAX-RB_MIN)*100)
            tput.append(np.mean(ep_t))
            x=float(traj[min(int(info['step']),T-1),0]); incov = float(np.min(np.abs(GNB-x)))<=CELL_RADIUS
            if any_below and incov: ss_incov+=1
            if info.get('sustained_rlf') and incov: sus_incov+=1
        per_seed['ho'].append(100*good/max(totho,1))
        per_seed['pwr'].append(np.mean(pwr)); per_seed['spec'].append(np.mean(spec))
        per_seed['tput'].append(np.mean(tput))
    agg = {k:(np.mean(v),np.std(v)) for k,v in per_seed.items()}
    N=len(SEEDS)*N_EP
    return agg, ss_incov, sus_incov, N

if __name__ == "__main__":
    rows=[]
    policies = [
        ('Dn-DQN (v6, retrained)', 'net', get_net('dqn_phase2_dndqn_v6.pt')),
        ('Dn-DQN (v5 model)',      'net', get_net('dqn_phase2_dndqn.pt')),
        ('PG-RRM',                 'pgrrm', None),
        ('3GPP A3',                'a3', None),
    ]
    print("="*94)
    print("  Evaluation under corrected sustained-RLF env (v6, N310=3) | 5 seeds x 150 = 750 ep")
    print("="*94)
    print(f"\n  {'policy':>22} | {'HO%':>10} {'Power%':>11} {'Spec%':>11} {'Tput(Mbps)':>12} | "
          f"{'RLF 1-samp':>10} {'RLF sustained':>13}")
    print("  "+"-"*90)
    for name,kind,net in policies:
        agg,ss,sus,N = eval_policy(kind, net)
        rows.append((name,agg,ss,sus,N))
        f=lambda k:f"{agg[k][0]:.1f}±{agg[k][1]:.1f}"
        print(f"  {name:>22} | {f('ho'):>10} {f('pwr'):>11} {f('spec'):>11} "
              f"{agg['tput'][0]:7.1f}±{agg['tput'][1]:<4.0f} | "
              f"{ss:4d}({100*ss/N:4.1f}%) {sus:4d}({100*sus/N:5.1f}%)")
    with open('results/eval_v6_summary.csv','w',newline='') as fh:
        w=csv.writer(fh); w.writerow(['policy','HO','HO_sd','Power','Power_sd','Spec','Spec_sd',
                                      'Tput','Tput_sd','RLF_1samp','RLF_1samp_pct','RLF_sustained','RLF_sustained_pct','N'])
        for name,agg,ss,sus,N in rows:
            w.writerow([name,*[f"{agg[k][i]:.3f}" for k in ('ho','pwr','spec','tput') for i in (0,1)],
                        ss,f"{100*ss/N:.2f}",sus,f"{100*sus/N:.2f}",N])
    print("\n  -> results/eval_v6_summary.csv")
    print("  'RLF 1-samp' = old single-sample artifact; 'RLF sustained' = corrected 3GPP-style.")
