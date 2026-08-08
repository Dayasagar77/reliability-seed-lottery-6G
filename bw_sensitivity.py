#!/usr/bin/env python3
"""bw_sensitivity.py — does switching noise bandwidth 2 GHz -> 380 MHz move the
MAIN results (HO/power/spectrum/sustained-RLF), or only throughput?

If HO/power/spectrum/RLF are stable, the 380 MHz fix is a throughput-only re-report
(no retrain). If they move, the policy's SINR observation shifted enough to need a
retrain under the corrected noise floor. Evaluated on the submitted model under the
corrected sustained-RLF env (env_6g_v6)."""
import math, numpy as np, torch
import channel_model as cm, env_6g_v5 as e5
from env_6g_v6 import SubTHzEnv6G_v6
from env_6g_v5 import (RLF_THRESH_DBM, GNB_X, CELL_RADIUS, W_MOB, W_PWR, W_SPEC,
                       PT_MIN_DBM, PT_MAX_DBM, RB_MIN, RB_MAX)
from dndqn import load_dndqn_checkpoint
import multi_seed_eval as mse

SEEDS, N_EP = mse.SEEDS, mse.N_EPISODES
GNB = np.asarray(GNB_X, float)
slow, med = mse.load_trajectories(mse.CSV_PATH); pool = slow + med
net, _ = load_dndqn_checkpoint('dqn_phase2_dndqn.pt'); net.eval()
KB, T0, NF = 1.38e-23, 290.0, 10**(10/10)

def set_bw(bw_hz):
    n0 = KB*T0*bw_hz*NF; n0dbm = 10*np.log10(n0/1e-3)
    cm.N0_WATTS = n0; cm.N0_DBM = n0dbm
    e5.N0_WATTS = n0; e5.N0_DBM = n0dbm
    return n0dbm

def evaluate(bw_hz):
    n0dbm = set_bw(bw_hz); rb_bw_mhz = (bw_hz/66)/1e6
    good=totho=sus=n=0; P=[];S=[];T=[];SI=[]
    for seed in SEEDS:
        rng=np.random.default_rng(seed)
        for _ in range(N_EP):
            traj=pool[rng.integers(len(pool))]; Tn=len(traj)
            env=SubTHzEnv6G_v6(traj,w_mob=W_MOB,w_pwr=W_PWR,w_spec=W_SPEC)
            obs,_=env.reset(); done=False; info=None; pts=[];rbs=[];tp=[];si=[]
            while not done:
                with torch.no_grad(): a=int(net(torch.FloatTensor(obs)).argmax().item())
                obs,_,term,trunc,info=env.step(a); done=term or trunc
                pts.append(info['pt_dbm']); rbs.append(info['rb_count'])
                sdb=obs[4]*100.0; si.append(sdb)
                tp.append(info['rb_count']*rb_bw_mhz*math.log2(1+10**(sdb/10)))
                if info.get('handover_done'):
                    totho+=1
                    if not info.get('too_early') and not info.get('too_late'): good+=1
            n+=1
            P.append((PT_MAX_DBM-np.mean(pts))/(PT_MAX_DBM-PT_MIN_DBM)*100)
            S.append((RB_MAX-np.mean(rbs))/(RB_MAX-RB_MIN)*100)
            T.append(np.mean(tp)); SI.append(np.mean(si))
            x=float(traj[min(int(info['step']),Tn-1),0])
            if info.get('sustained_rlf') and float(np.min(np.abs(GNB-x)))<=CELL_RADIUS: sus+=1
    return dict(n0=n0dbm, rb_bw=rb_bw_mhz, ho=100*good/max(totho,1), pwr=np.mean(P),
                spec=np.mean(S), rlf=100*sus/n, sinr=np.mean(SI), tput=np.mean(T))

print("="*84)
print("  Noise-bandwidth sensitivity of the MAIN results (submitted model, sustained-RLF env)")
print("="*84)
print(f"\n  {'bandwidth':>12} {'N0(dBm)':>8} | {'HO%':>5} {'Pwr%':>6} {'Spec%':>6} {'RLF%':>5} "
      f"{'SINR(dB)':>9} {'Tput(Mbps)':>11}")
print("  "+"-"*78)
for bw,label in [(2e9,'2 GHz (code)'),(380.16e6,'380 MHz (B)')]:
    r=evaluate(bw)
    print(f"  {label:>12} {r['n0']:8.2f} | {r['ho']:5.1f} {r['pwr']:6.1f} {r['spec']:6.1f} "
          f"{r['rlf']:5.1f} {r['sinr']:9.2f} {r['tput']:11.1f}")
print("\n  If HO/Pwr/Spec/RLF barely move: 380 MHz is a throughput-only re-report (no retrain).")
print("  If they move: the SINR observation shifted enough to require retraining under 380 MHz.")
