#!/usr/bin/env python3
"""attribution_v6.py — integrated-gradients feature attribution for Dn-DQN v6.
Confirms whether the corrected policy remains velocity-agnostic."""
import numpy as np, torch
from env_6g_v6 import SubTHzEnv6G_v6
from env_6g_v5 import W_MOB, W_PWR, W_SPEC
from dndqn import load_dndqn_checkpoint
import multi_seed_eval as mse

FEATURES=['RSRP','A3-margin','TxPower','RB','SINR','Velocity']
ACTIONS =['HOLD','PWR_UP','PWR_DOWN','RB_UP','RB_DOWN','HANDOVER']
slow, med = mse.load_trajectories(mse.CSV_PATH); pool=slow+med
net,_ = load_dndqn_checkpoint('dqn_phase2_dndqn_v6.pt'); net.eval()

# collect ~500 states from greedy rollouts under v6
states=[]; rng=np.random.default_rng(0)
for _ in range(25):
    env=SubTHzEnv6G_v6(pool[rng.integers(len(pool))],w_mob=W_MOB,w_pwr=W_PWR,w_spec=W_SPEC)
    obs,_=env.reset(); done=False
    while not done and len(states)<500:
        states.append(obs.copy())
        with torch.no_grad(): a=int(net(torch.FloatTensor(obs)).argmax().item())
        obs,_,term,trunc,_=env.step(a); done=term or trunc
X=torch.FloatTensor(np.array(states[:500]))
base=X.mean(0,keepdim=True)               # baseline = mean state

def ig(action, steps=32):
    attr=torch.zeros_like(X)
    for al in np.linspace(0,1,steps):
        inp=(base+al*(X-base)).clone().requires_grad_(True)
        q=net(inp)[:,action].sum()
        g,=torch.autograd.grad(q,inp)
        attr+=g/steps
    return ((X-base)*attr).abs().mean(0).detach().numpy()   # mean |attr| per feature

A=np.stack([ig(a) for a in range(6)])     # (actions, features)
per_action_pct = 100*A/A.sum(1,keepdims=True)
overall = A.mean(0); overall_pct = 100*overall/overall.sum()

print("="*60); print("  Dn-DQN v6 — integrated-gradients feature attribution"); print("="*60)
print(f"\n  {'feature':<12} {'overall %':>10}")
for f,p in sorted(zip(FEATURES,overall_pct),key=lambda z:-z[1]):
    print(f"  {f:<12} {p:9.1f}%")
print(f"\n  Velocity attribution: {overall_pct[5]:.1f}%  (min feature: "
      f"{FEATURES[int(np.argmin(overall_pct))]} {overall_pct.min():.1f}%)")
print(f"  SINR attribution    : {overall_pct[4]:.1f}%")
print("\n  Per-action velocity %:", " ".join(f"{ACTIONS[a]}={per_action_pct[a,5]:.0f}" for a in range(6)))
