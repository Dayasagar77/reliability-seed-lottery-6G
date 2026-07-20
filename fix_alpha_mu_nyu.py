#!/usr/bin/env python3
"""fix_alpha_mu_nyu.py — alpha-mu small-scale-fading sensitivity on the NYU 142 GHz CI channel
(channel-consistency fix 1b). Overrides channel_model.ALPHA_LOS / MU_LOS across the literature
range and re-evaluates the SAME trained NYU reliability policy (dqn_phase2_dndqn_nyu_rel.pt) with
the identical eval used for the headline results (run_multiseed_rel.evaluate, 5 seeds x 150 ep) —
so the baseline row lines up with the multi-seed numbers. -> results/alpha_mu_sensitivity_nyu.csv
Waits for n-step + reward-ablation to finish first (single-thread CPU)."""
import os, time, csv, subprocess
import numpy as np
import channel_model as cm
from dndqn import load_dndqn_checkpoint
import train_phase2 as tp
from run_multiseed_rel import evaluate

def waiting():
    try: return bool(subprocess.run(["pgrep", "-f", "fix_nstep|fix_reward"], capture_output=True, text=True).stdout.strip())
    except Exception: return False

def main():
    while waiting():
        print("[wait] nstep/reward running; sleep 120s", flush=True); time.sleep(120)
    print("Starting alpha-mu fading sensitivity on NYU (same trained policy re-evaluated)", flush=True)
    net = load_dndqn_checkpoint('dqn_phase2_dndqn_nyu_rel.pt')[0]; net.eval()
    slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH); pool = slow + med
    BASE_A, BASE_MU = cm.ALPHA_LOS, cm.MU_LOS
    rows = []
    def run(tag, a, mu):
        cm.ALPHA_LOS = a; cm.MU_LOS = mu
        m = evaluate(net, pool)
        rows.append((tag, a, mu, m['HO'], m['Power'], m['Spec'], m['RLF']))
        print(f"  {tag:12s} a={a:.3f} mu={mu:.3f} | HO {m['HO']:5.1f}  Power {m['Power']:5.1f}  Spec {m['Spec']:5.1f}  RLF {m['RLF']:5.2f}%", flush=True)
    run("baseline", BASE_A, BASE_MU)
    print("  --- vary mu (cluster count) at alpha=2.0 ---", flush=True)
    for mu in [0.6, 1.0, 1.5, 2.0, 3.0]: run(f"mu={mu}", 2.0, mu)
    print("  --- vary alpha (nonlinearity) at mu=1.0 ---", flush=True)
    for a in [1.8, 2.0, 2.5, 3.0, 3.5]: run(f"alpha={a}", a, 1.0)
    cm.ALPHA_LOS, cm.MU_LOS = BASE_A, BASE_MU
    rl = [r[6] for r in rows]; sp = [r[5] for r in rows]; ho = [r[3] for r in rows]; pw = [r[4] for r in rows]
    print(f"\n  RANGE across all fading settings: RLF {min(rl):.2f}..{max(rl):.2f}% (spread {max(rl)-min(rl):.2f}pp) | "
          f"Spec {min(sp):.1f}..{max(sp):.1f}% | HO {min(ho):.1f}..{max(ho):.1f}% | Power {min(pw):.1f}..{max(pw):.1f}%", flush=True)
    os.makedirs('results', exist_ok=True)
    with open('results/alpha_mu_sensitivity_nyu.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['setting', 'alpha', 'mu', 'HO', 'Power', 'Spec', 'RLF'])
        for r in rows: w.writerow([r[0], f"{r[1]:.3f}", f"{r[2]:.3f}", f"{r[3]:.2f}", f"{r[4]:.2f}", f"{r[5]:.2f}", f"{r[6]:.3f}"])
    print("  -> results/alpha_mu_sensitivity_nyu.csv", flush=True)

if __name__ == "__main__":
    main()
