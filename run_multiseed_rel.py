#!/usr/bin/env python3
"""
run_multiseed_rel.py [n_seeds] [steps] — multi-seed the RELIABILITY-TILTED proposed Dn-DQN
(reward W 0.8/0.1/0.1, RLF penalty -300) under the NYU 142 GHz CI channel, to confirm the
reliability-frontier point is robust across training seeds rather than seed-luck.

Single-seed result being checked: HO 95.3% / Power 7.3% / Spec 90.2% / RLF 4.0% / Tput 470.
Global-seeded (random+numpy+torch), best-checkpointed, resumable. Reports mean +/- std.

Usage:
    python -u run_multiseed_rel.py 5          # 5 seeds x 800k steps  (~3.5 hr) -> results/multiseed_rel_summary.csv
    python -u run_multiseed_rel.py 3          # 3 seeds               (~2 hr)
    python -u run_multiseed_rel.py 5 400000   # 5 seeds, 400k steps each (faster)
"""
import sys, os, time, random, math
from collections import deque
import numpy as np, torch, torch.nn as nn, torch.optim as optim, csv

from train_phase2_nyu_rel import Phase2Env_NYU_rel          # reliability-tilted env (W 0.8/0.1/0.1, RLF -300)
from dndqn import (build_dndqn_networks, NStepPERBuffer, polyak_update,
                   epsilon_schedule, beta_schedule, OBS_DIM, NET_ARCH, load_dndqn_checkpoint)
from env_6g_nyu import SubTHzEnv6G_NYU
from env_6g_v5 import N_ACTIONS, PT_MIN_DBM, PT_MAX_DBM, RB_MIN, RB_MAX, W_MOB, W_PWR, W_SPEC
import train_phase2 as tp
import multi_seed_eval as mse

N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
STEPS   = int(sys.argv[2]) if len(sys.argv) > 2 else 800_000
GAIN, RB_BW_HZ, GAMMA = 15.0, 380.16e6 / RB_MAX, 0.99
EVAL_SEEDS, EVAL_EP = mse.SEEDS, mse.N_EPISODES
EPS_START, EPS_END, EPS_FRACTION, STAGE_SLOW_END = 1.0, 0.05, 0.30, 0.08
LEARNING_STARTS, TRAIN_FREQ, BATCH, TARGET_UPDATE, TAU = 10_000, 4, 256, 500, 0.01
PER_BETA0, PER_BETA1, NSTEP, ALPHA = 0.4, 1.0, 3, 0.6

def set_seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s)

def train(seed, out, slow, med):
    set_seed(seed); pool = slow + med
    online, target = build_dndqn_networks(OBS_DIM, N_ACTIONS, NET_ARCH, 'cpu'); target.load_state_dict(online.state_dict())
    opt = optim.Adam(online.parameters(), lr=3e-4); buf = NStepPERBuffer(500_000, n_steps=NSTEP, gamma=GAMMA, alpha=ALPHA)
    rng = np.random.default_rng(seed)
    smp = lambda stg: (slow if stg == "SLOW" else pool)[rng.integers(len(slow if stg == "SLOW" else pool))]
    stage = "SLOW"; env = Phase2Env_NYU_rel(smp(stage)); obs, _ = env.reset()
    epr = deque(maxlen=20); er = 0.0; best = -1e18; saved = False; t0 = time.time()
    def _save():
        torch.save({'q_net_state': online.state_dict(), 'obs_dim': OBS_DIM, 'n_actions': N_ACTIONS,
                    'net_arch': NET_ARCH, 'algorithm': 'Dn-DQN', 'n_steps': NSTEP, 'seed': seed,
                    'training_mode': 'nyu_reliability_tilt'}, out)
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
                tq = rt + (GAMMA ** NSTEP) * nq * (1 - dt)
            cq = online(ot).gather(1, at.unsqueeze(1)).squeeze(1); td = (cq - tq).detach().cpu().numpy()
            loss = (wt * nn.functional.huber_loss(cq, tq, reduction='none')).mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(online.parameters(), 10.0); opt.step()
            buf.update_priorities(ib, td)
        if step % TARGET_UPDATE == 0: polyak_update(online, target, TAU)
    if not saved: _save()
    return (time.time() - t0) / 60.0

def evaluate(net, pool):
    net.eval(); su = []; pw = []; sp = []; tpv = []; rl = []
    for seed in EVAL_SEEDS:
        r = np.random.default_rng(seed); real = good = nep = rlf = 0; P = []; S = []; T = []
        for _ in range(EVAL_EP):
            traj = pool[r.integers(len(pool))]
            env = SubTHzEnv6G_NYU(traj, array_gain_dbi=GAIN, w_mob=W_MOB, w_pwr=W_PWR, w_spec=W_SPEC)
            obs, _ = env.reset(); done = False; nep += 1; pts = []; rbs = []; et = []
            while not done:
                with torch.no_grad(): a = int(net(torch.FloatTensor(obs)).argmax().item())
                pg = env.serving_gnb_idx; obs, _, t, tr, info = env.step(a); done = t or tr
                pts.append(info['pt_dbm']); rbs.append(info['rb_count']); sdb = obs[4] * 100.0
                et.append(info['rb_count'] * (RB_BW_HZ / 1e6) * math.log2(1 + 10 ** (sdb / 10)))
                if env.serving_gnb_idx != pg:
                    real += 1
                    if not info.get('too_early') and not info.get('too_late'): good += 1
            if info.get('sustained_rlf'): rlf += 1
            P.append((PT_MAX_DBM - np.mean(pts)) / (PT_MAX_DBM - PT_MIN_DBM) * 100)
            S.append((RB_MAX - np.mean(rbs)) / (RB_MAX - RB_MIN) * 100); T.append(np.mean(et))
        su.append(100 * good / max(real, 1)); pw.append(np.mean(P)); sp.append(np.mean(S)); tpv.append(np.mean(T)); rl.append(100 * rlf / nep)
    return dict(HO=float(np.mean(su)), Power=float(np.mean(pw)), Spec=float(np.mean(sp)),
                Tput=float(np.mean(tpv)), RLF=float(np.mean(rl)))

def main():
    print(f"MULTI-SEED reliability-tilted Dn-DQN | {N_SEEDS} seeds | {STEPS:,} steps | NYU CI channel", flush=True)
    slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH); pool = slow + med
    os.makedirs('results', exist_ok=True); rows_path = 'results/multiseed_rel_nyu.csv'
    done = set(); rows = []
    if os.path.exists(rows_path):
        with open(rows_path) as f:
            for row in csv.DictReader(f): done.add(int(row['seed'])); rows.append(row)
    for seed in range(N_SEEDS):
        if seed in done: print(f"[skip] seed {seed} (done)", flush=True); continue
        ckpt = f"ms_rel_s{seed}.pt"; print(f"[train] reliability-tilt seed {seed} -> {ckpt} ({STEPS:,} steps) ...", flush=True)
        mins = train(seed, ckpt, slow, med); m = evaluate(load_dndqn_checkpoint(ckpt)[0], pool)
        rows.append({'seed': seed, 'HO': f"{m['HO']:.2f}", 'Power': f"{m['Power']:.2f}", 'Spec': f"{m['Spec']:.2f}",
                     'RLF': f"{m['RLF']:.2f}", 'Tput': f"{m['Tput']:.1f}", 'train_min': f"{mins:.1f}"})
        with open(rows_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['seed', 'HO', 'Power', 'Spec', 'RLF', 'Tput', 'train_min']); w.writeheader(); w.writerows(rows)
        print(f"[done] seed {seed}: HO {m['HO']:.1f}% Pwr {m['Power']:.1f}% Spec {m['Spec']:.1f}% RLF {m['RLF']:.1f}% Tput {m['Tput']:.0f} ({mins:.1f}m)", flush=True)
    v = {k: [float(r[k]) for r in rows] for k in ['HO', 'Power', 'Spec', 'RLF', 'Tput']}
    g = lambda a: (float(np.mean(a)), float(np.std(a)))
    print("\n" + "=" * 80); print(f"  RELIABILITY-TILTED Dn-DQN — mean +/- std over {len(v['HO'])} seeds ({STEPS:,} steps each)"); print("=" * 80)
    for k in ['HO', 'Power', 'Spec', 'RLF', 'Tput']:
        mn, sd = g(v[k]); print(f"  {k:>6} : {mn:7.2f} +/- {sd:.2f}")
    print(f"\n  (single-seed: HO 95.3 / Power 7.3 / Spec 90.2 / RLF 4.0 ; A3: HO 96.2 / RLF 5.1 / 0 savings)")
    with open('results/multiseed_rel_summary.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['metric', 'mean', 'std'])
        for k in ['HO', 'Power', 'Spec', 'RLF', 'Tput']:
            mn, sd = g(v[k]); w.writerow([k, f"{mn:.2f}", f"{sd:.2f}"])
    print("  -> results/multiseed_rel_nyu.csv (per-seed) + results/multiseed_rel_summary.csv", flush=True)

if __name__ == "__main__":
    main()
