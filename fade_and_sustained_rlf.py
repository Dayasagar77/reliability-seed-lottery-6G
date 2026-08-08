#!/usr/bin/env python3
"""
fade_and_sustained_rlf.py — is the ~80% 'in-coverage RLF' a real reliability
figure, or an artifact of (a) single-sample RLF termination + (b) per-step
independent deep alpha-mu fades?

Part 1: alpha-mu fade-depth CCDF (LoS and NLoS) — how often is a single-sample
        fade deep enough (>~15-18 dB) to cross the -92 floor from a normal margin.
Part 2: re-evaluate reliability with RLF termination DISABLED, then classify each
        episode under progressively more realistic RLF definitions:
          - >=1 sub-threshold sample (current code definition)
          - >=K CONSECUTIVE sub-threshold samples (sustained; 3GPP-like)
          - W-sample moving-average RSRP < -92 (RSRP is an average, not a sample)
        Shows how the failure rate collapses as the definition becomes realistic.
"""
import numpy as np, torch
import multi_seed_eval as mse
from channel_model import alpha_mu_sample, ALPHA_LOS, MU_LOS, ALPHA_NLOS, MU_NLOS
from env_6g_v5 import (SubTHzEnv6G, RLF_THRESH_DBM, GNB_X, CELL_RADIUS,
                       W_MOB, W_PWR, W_SPEC, MAX_STEPS)
from dndqn import load_dndqn_checkpoint

# ---------- Part 1: fade-depth CCDF ----------
rng = np.random.default_rng(0)
print("="*70)
print("  Part 1 — single-sample alpha-mu fade depth (power, dB about mean)")
print("="*70)
for name, a, m in [("LoS ", ALPHA_LOS, MU_LOS), ("NLoS", ALPHA_NLOS, MU_NLOS)]:
    h = alpha_mu_sample(a, m, rms_val=1.0, size=400000, rng=rng)
    fade_db = 10*np.log10(h**2)          # <0 = fade below mean
    print(f"\n  {name} (alpha={a}, mu={m}): mean power {np.mean(h**2):.3f}")
    for thr in (-10,-12,-15,-18,-20,-25):
        p = float(np.mean(fade_db < thr))
        print(f"     P(fade < {thr:>4} dB) = {p:8.4f}   "
              f"(~1 in {1/max(p,1e-9):,.0f} samples)")

# ---------- Part 2: sustained-RLF re-evaluation ----------
CSV, SEEDS, N_EP = mse.CSV_PATH, mse.SEEDS, mse.N_EPISODES
GNB = np.asarray(GNB_X, float)
slow, med = mse.load_trajectories(CSV); pool = slow + med
net, _ = load_dndqn_checkpoint('dqn_phase2_dndqn.pt')

class NoRLFEnv(SubTHzEnv6G):
    """RLF does not terminate; episode ends only at trajectory end / MAX_STEPS.
    Lets us observe the full RSRP trace and apply a realistic RLF definition offline."""
    def step(self, action):
        obs, r, term, trunc, info = super().step(action)
        # strip the single-sample RLF cause; keep trajectory-end / max-step end
        term = (self.step_idx >= self.T) or (self.episode_step >= MAX_STEPS)
        return obs, r, term, trunc, info

def max_consecutive(mask):
    best = cur = 0
    for v in mask:
        cur = cur+1 if v else 0
        best = max(best, cur)
    return best

def movavg_below(rsrp, w, thr):
    if len(rsrp) < w: return False
    c = np.convolve(rsrp, np.ones(w)/w, mode='valid')
    return bool(np.any(c < thr))

ge1=ge2=ge3=ge5=0; ma3=ma5=ma10=0; incov_any=0; tot=0
subcount=[]
for seed in SEEDS:
    r = np.random.default_rng(seed)
    for _ in range(N_EP):
        traj = pool[r.integers(len(pool))]; T=len(traj)
        env = NoRLFEnv(traj, w_mob=W_MOB, w_pwr=W_PWR, w_spec=W_SPEC)
        obs,_ = env.reset(); done=False; rs=[]
        while not done:
            with torch.no_grad(): a=int(net(torch.FloatTensor(obs)).argmax().item())
            obs,_,term,trunc,info = env.step(a); done=term or trunc
            rs.append(info['rsrp_dbm'])
        rs=np.array(rs); below = rs < RLF_THRESH_DBM
        tot += 1
        nsub = int(below.sum()); subcount.append(nsub)
        mc = max_consecutive(below)
        if nsub>=1: ge1+=1
        if mc>=2: ge2+=1
        if mc>=3: ge3+=1
        if mc>=5: ge5+=1
        if movavg_below(rs,3,RLF_THRESH_DBM): ma3+=1
        if movavg_below(rs,5,RLF_THRESH_DBM): ma5+=1
        if movavg_below(rs,10,RLF_THRESH_DBM): ma10+=1

print("\n"+"="*70)
print(f"  Part 2 — reliability of baseline Dn-DQN under RLF definitions ({tot} ep)")
print("="*70)
p=lambda k:f"{k:4d}  ({100*k/tot:5.1f}%)"
print(f"  >=1 sub-threshold sample (CURRENT code def) : {p(ge1)}")
print(f"  >=2 CONSECUTIVE sub-threshold samples       : {p(ge2)}")
print(f"  >=3 CONSECUTIVE sub-threshold samples       : {p(ge3)}")
print(f"  >=5 CONSECUTIVE sub-threshold samples       : {p(ge5)}")
print(f"   3-sample moving-avg RSRP < -92 dBm         : {p(ma3)}")
print(f"   5-sample moving-avg RSRP < -92 dBm         : {p(ma5)}")
print(f"  10-sample moving-avg RSRP < -92 dBm         : {p(ma10)}")
print(f"\n  median sub-threshold samples per episode    : {int(np.median(subcount))}"
      f"  (mean {np.mean(subcount):.1f}) out of ~{int(np.median([len(pool[0])]))}+ steps")
print("\n  If the failure rate collapses from >=1 to >=3-consecutive / moving-avg,")
print("  the ~80% 'RLF' is a single-sample-fade artifact, not sustained link failure.")
