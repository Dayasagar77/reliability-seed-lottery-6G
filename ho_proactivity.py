#!/usr/bin/env python3
"""ho_proactivity.py — does the joint controller hand over PROACTIVELY?
For every HANDOVER the policy executes, record the deterministic (mean-channel)
serving RSRP at the trigger and classify it vs the thresholds:
  proactive window (>= -75 dBm)      : fires early, before A3
  timely (-80 .. -75)                : fires in the proactive band, above A3
  at/after A3 (-92 .. -80)           : fires after A3 crossing, still above RLF
  late / failed (< -92)              : fires after the link already failed
Also reports the margin to the -92 dBm RLF floor at the moment of handover.
Submitted model; RSRP is bandwidth-independent so this is unaffected by the 380 MHz fix."""
import numpy as np, torch
from env_6g_v6 import SubTHzEnv6G_v6
from env_6g_v5 import (W_MOB, W_PWR, W_SPEC, HANDOVER, A3_THRESH_DBM,
                       HO_PROACTIVE_THRESH, RLF_THRESH_DBM)
from dndqn import load_dndqn_checkpoint
import multi_seed_eval as mse

net, _ = load_dndqn_checkpoint('dqn_phase2_dndqn.pt'); net.eval()
slow, med = mse.load_trajectories(mse.CSV_PATH); pool = slow + med
SEEDS = mse.SEEDS; N_EP = 100

trig = []          # deterministic serving RSRP at each handover
for seed in SEEDS:
    rng = np.random.default_rng(seed)
    for _ in range(N_EP):
        traj = pool[rng.integers(len(pool))]
        env = SubTHzEnv6G_v6(traj, w_mob=W_MOB, w_pwr=W_PWR, w_spec=W_SPEC)
        obs, _ = env.reset(); done = False
        while not done:
            with torch.no_grad():
                a = int(net(torch.FloatTensor(obs)).argmax().item())
            if a == HANDOVER:
                d = env._get_distance()                     # old serving cell, pre-switch
                det = env._compute_rsrp(d, apply_fading_flag=False)[0]
                trig.append(det)
            obs, _, term, trunc, info = env.step(a); done = term or trunc

trig = np.array(trig); n = len(trig)
proactive = np.mean(trig >= HO_PROACTIVE_THRESH) * 100          # >= -75
band      = np.mean((trig < HO_PROACTIVE_THRESH) & (trig >= A3_THRESH_DBM)) * 100  # -80..-75
above_rlf = np.mean((trig < A3_THRESH_DBM) & (trig >= RLF_THRESH_DBM)) * 100       # -92..-80
failed    = np.mean(trig < RLF_THRESH_DBM) * 100               # < -92

print("="*70)
print(f"  Handover-trigger analysis — {n:,} handovers, submitted model")
print("="*70)
print(f"  deterministic serving RSRP at handover:")
print(f"     median {np.median(trig):6.2f} dBm | mean {np.mean(trig):6.2f} | "
      f"10th pct {np.percentile(trig,10):6.2f} | 90th pct {np.percentile(trig,90):6.2f}")
print(f"  margin to RLF floor (-92 dBm) at handover: median {np.median(trig)-RLF_THRESH_DBM:5.1f} dB")
print(f"\n  WHERE handovers fire (thresholds: proactive -75, A3 -80, RLF -92):")
print(f"     >= -75 dBm  (early, before A3)      : {proactive:5.1f}%")
print(f"     -80..-75    (proactive band)        : {band:5.1f}%")
print(f"     -92..-80    (after A3, above RLF)   : {above_rlf:5.1f}%")
print(f"     <  -92 dBm  (after link failure)    : {failed:5.1f}%")
print(f"\n  fired ABOVE A3 (-80) i.e. before the reporting threshold: "
      f"{np.mean(trig>=A3_THRESH_DBM)*100:.1f}%")
print(f"  fired ABOVE RLF (-92) i.e. before link failure          : "
      f"{np.mean(trig>=RLF_THRESH_DBM)*100:.1f}%")
