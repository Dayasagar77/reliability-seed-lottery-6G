#!/usr/bin/env python3
"""fix_nstep_ablation_nyu.py — n-step ablation on the NYU 142 GHz CI channel (channel-consistency
fix 1b). Full dueling Double-DQN + PER under the reliability reward (Phase2Env_NYU_rel), n in {1,5}
x seeds {0,1,2}, 800k steps to MATCH the multi-seed n=3 run (ms_rel_s0..2). The n=3 rows come from
results/multiseed_rel_nyu.csv (seeds 0-2). Best-checkpointed, resumable via results/nstep_ablation_nyu.csv.
Waits for run_multiseed_rel to finish first (single-thread CPU — avoid contention)."""
import os, time, random, csv, subprocess
from collections import deque
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from train_phase2_nyu_rel import Phase2Env_NYU_rel
from dndqn import (build_dndqn_networks, NStepPERBuffer, polyak_update,
                   epsilon_schedule, beta_schedule, OBS_DIM, NET_ARCH, load_dndqn_checkpoint)
from env_6g_v5 import N_ACTIONS
import train_phase2 as tp
from run_multiseed_rel import evaluate, GAMMA          # identical eval + gamma as the headline run

STEPS, SEEDS, NSTEPS = 800_000, [0, 1, 2], [1, 5]
EPS_START, EPS_END, EPS_FRACTION, STAGE_SLOW_END = 1.0, 0.05, 0.30, 0.08
LEARNING_STARTS, TRAIN_FREQ, BATCH, TARGET_UPDATE, TAU = 10_000, 4, 256, 500, 0.01
PER_BETA0, PER_BETA1, ALPHA = 0.4, 1.0, 0.6

def set_seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s)

def train(seed, nstep, out, slow, med):
    set_seed(seed); pool = slow + med
    online, target = build_dndqn_networks(OBS_DIM, N_ACTIONS, NET_ARCH, 'cpu'); target.load_state_dict(online.state_dict())
    opt = optim.Adam(online.parameters(), lr=3e-4); buf = NStepPERBuffer(500_000, n_steps=nstep, gamma=GAMMA, alpha=ALPHA)
    rng = np.random.default_rng(seed)
    smp = lambda stg: (slow if stg == "SLOW" else pool)[rng.integers(len(slow if stg == "SLOW" else pool))]
    stage = "SLOW"; env = Phase2Env_NYU_rel(smp(stage)); obs, _ = env.reset()
    epr = deque(maxlen=20); er = 0.0; best = -1e18; saved = False
    def _save():
        torch.save({'q_net_state': online.state_dict(), 'obs_dim': OBS_DIM, 'n_actions': N_ACTIONS,
                    'net_arch': NET_ARCH, 'algorithm': 'Dn-DQN', 'n_steps': nstep, 'seed': seed}, out)
    for step in range(1, STEPS + 1):
        frac = step / STEPS; stage = "SLOW" if frac < STAGE_SLOW_END else "SLOW+MED"
        eps = epsilon_schedule(step, EPS_START, EPS_END, EPS_FRACTION, STEPS); beta = beta_schedule(step, PER_BETA0, PER_BETA1, STEPS)
        if random.random() < eps: a = random.randint(0, N_ACTIONS - 1)
        else:
            with torch.no_grad(): a = online(torch.FloatTensor(obs)).argmax().item()
        nobs, r, term, trunc, info = env.step(a); done = term or trunc; er += r
        buf.add(obs, a, r, nobs, term); obs = nobs
        if done:
            epr.append(er); er = 0.0; env = Phase2Env_NYU_rel(smp(stage)); obs, _ = env.reset()
            if frac > EPS_FRACTION and len(epr) == epr.maxlen:
                m = float(np.mean(epr))
                if m > best: best = m; _save(); saved = True
        if step >= LEARNING_STARTS and step % TRAIN_FREQ == 0 and len(buf) >= BATCH:
            ib, wb, ob, ab, rb, nb, db = buf.sample(BATCH, beta)
            ot = torch.FloatTensor(ob); nt = torch.FloatTensor(nb); at = torch.LongTensor(ab)
            rt = torch.FloatTensor(rb); dt = torch.FloatTensor(db); wt = torch.FloatTensor(wb)
            with torch.no_grad():
                ba = online(nt).argmax(1, keepdim=True); nq = target(nt).gather(1, ba).squeeze(1)
                tq = rt + (GAMMA ** nstep) * nq * (1 - dt)
            cq = online(ot).gather(1, at.unsqueeze(1)).squeeze(1); td = (cq - tq).detach().cpu().numpy()
            loss = (wt * nn.functional.huber_loss(cq, tq, reduction='none')).mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(online.parameters(), 10.0); opt.step()
            buf.update_priorities(ib, td)
        if step % TARGET_UPDATE == 0: polyak_update(online, target, TAU)
    if not saved: _save()

def multiseed_running():
    try:
        return bool(subprocess.run(["pgrep", "-f", "run_multiseed_rel"], capture_output=True, text=True).stdout.strip())
    except Exception:
        return False

def main():
    while multiseed_running():
        print("[wait] multi-seed still running; sleeping 120s ...", flush=True); time.sleep(120)
    print("MULTI-SEED done — starting n-step ablation on NYU (n in {1,5} x seeds {0,1,2}, 800k)", flush=True)
    slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH); pool = slow + med
    os.makedirs('results', exist_ok=True); path = 'results/nstep_ablation_nyu.csv'
    done = set(); rows = []
    if os.path.exists(path):
        with open(path) as f:
            for row in csv.DictReader(f): done.add((int(row['n']), int(row['seed']))); rows.append(row)
    for nstep in NSTEPS:
        for seed in SEEDS:
            if (nstep, seed) in done: print(f"[skip] n={nstep} seed={seed} (done)", flush=True); continue
            ck = f"nstep_n{nstep}_s{seed}.pt"; t0 = time.time()
            print(f"[train] n={nstep} seed={seed} -> {ck}", flush=True)
            train(seed, nstep, ck, slow, med)
            m = evaluate(load_dndqn_checkpoint(ck)[0], pool); mins = (time.time() - t0) / 60
            rows.append({'n': nstep, 'seed': seed, 'HO': f"{m['HO']:.2f}", 'Power': f"{m['Power']:.2f}",
                         'Spec': f"{m['Spec']:.2f}", 'RLF': f"{m['RLF']:.2f}", 'Tput': f"{m['Tput']:.1f}", 'train_min': f"{mins:.1f}"})
            with open(path, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=['n', 'seed', 'HO', 'Power', 'Spec', 'RLF', 'Tput', 'train_min'])
                w.writeheader(); w.writerows(rows)
            print(f"[done] n={nstep} seed={seed}: RLF {m['RLF']:.1f}% Spec {m['Spec']:.1f}% Pwr {m['Power']:.1f}% ({mins:.0f}m)", flush=True)
    print("DONE n-step ablation on NYU. (n=3 rows: multiseed_rel_nyu.csv seeds 0-2)", flush=True)

if __name__ == "__main__":
    main()
