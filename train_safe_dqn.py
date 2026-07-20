#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_safe_dqn.py — Shielded Safe-DQN baseline (action masking), reviewer major comment 5.

Same dueling + n-step + PER Dn-DQN backbone, same flat 6-action space, same reliability-tilt reward
(Phase2Env_NYU_rel, fixed -300) as the matched-objective 'Proposed Dn-DQN' — the ONLY change is a
measured SAFETY SHIELD applied at action selection (a 'Safe-DQN variant with action masking'): when the
observed serving RSRP is below the same measurement-derived save line used by the proposed guard
(Theta_RLF + Delta_P + G = -92 + 7 + 9 = -76 dBm), the power-reducing action (PWR_DOWN) is masked.
This is the ACTION-level analogue of the paper's STATE guard, so it directly tests whether action
masking on the conventional action space matches the measurement-shaped state guard. Stable (standard
DQN training + a deterministic filter). Evaluated with the standard protocol (5x150). Zero fabrication.

Usage: python -u train_safe_dqn.py [n_seeds=3] [steps=400000] [mode=full|validate]
"""
import sys, os, time, random, math, csv
from collections import deque
import numpy as np, torch, torch.nn as nn, torch.optim as optim
torch.set_num_threads(1)

from train_phase2_nyu_rel import Phase2Env_NYU_rel
from env_6g_nyu import SubTHzEnv6G_NYU
from env_6g_v5 import (N_ACTIONS, PWR_DOWN, PT_MIN_DBM, PT_MAX_DBM, RB_MIN, RB_MAX,
                       RLF_THRESH_DBM, W_MOB, W_PWR, W_SPEC)
from dndqn import (build_dndqn_networks, NStepPERBuffer, polyak_update,
                   epsilon_schedule, beta_schedule, OBS_DIM, NET_ARCH, load_dndqn_checkpoint)
import train_phase2 as tp
import multi_seed_eval as mse

GAIN, RB_BW_HZ, GAMMA = 15.0, 380.16e6 / RB_MAX, 0.99
EVAL_SEEDS, EVAL_EP = mse.SEEDS, mse.N_EPISODES
EPS_START, EPS_END, EPS_FRACTION, STAGE_SLOW_END = 1.0, 0.05, 0.30, 0.08
LEARNING_STARTS, TRAIN_FREQ, BATCH, TARGET_UPDATE, TAU = 10_000, 4, 256, 500, 0.01
PER_BETA0, PER_BETA1, NSTEP, ALPHA = 0.4, 1.0, 3, 0.6
SAVE_LINE = RLF_THRESH_DBM + (PT_MAX_DBM - PT_MIN_DBM) + 9.0   # -92 + 7 + 9 = -76 dBm (measured guard line)


def shielded_action(qvals, obs, allow_random=None):
    """Mask PWR_DOWN when the observed serving RSRP is below the measured save line."""
    rsrp = float(obs[0]) * 100.0                 # obs[0] = rsrp_norm = rsrp_dbm / 100
    if allow_random is not None:                 # exploration: resample within allowed set
        if rsrp < SAVE_LINE and allow_random == PWR_DOWN:
            choices = [a for a in range(N_ACTIONS) if a != PWR_DOWN]
            return random.choice(choices)
        return allow_random
    q = qvals.clone().reshape(-1)
    if rsrp < SAVE_LINE:
        q[PWR_DOWN] = -1e9
    return int(q.argmax().item())


def set_seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s)


def train(seed, out, slow, med, steps):
    set_seed(seed); pool = slow + med
    online, target = build_dndqn_networks(OBS_DIM, N_ACTIONS, NET_ARCH, 'cpu'); target.load_state_dict(online.state_dict())
    opt = optim.Adam(online.parameters(), lr=3e-4); buf = NStepPERBuffer(500_000, n_steps=NSTEP, gamma=GAMMA, alpha=ALPHA)
    rng = np.random.default_rng(seed)
    smp = lambda stg: (slow if stg == "SLOW" else pool)[rng.integers(len(slow if stg == "SLOW" else pool))]
    stage = "SLOW"; env = Phase2Env_NYU_rel(smp(stage)); obs, _ = env.reset()
    epr = deque(maxlen=20); er = 0.0; best = -1e18; saved = False; t0 = time.time()
    def _save():
        torch.save({'q_net_state': online.state_dict(), 'obs_dim': OBS_DIM, 'n_actions': N_ACTIONS,
                    'net_arch': NET_ARCH, 'algorithm': 'Shielded-Safe-DQN', 'n_steps': NSTEP, 'seed': seed,
                    'save_line_dbm': SAVE_LINE, 'training_mode': 'nyu_ci_shielded'}, out)
    for step in range(1, steps + 1):
        frac = step / steps; stage = "SLOW" if frac < STAGE_SLOW_END else "SLOW+MED"
        eps = epsilon_schedule(step, EPS_START, EPS_END, EPS_FRACTION, steps); beta = beta_schedule(step, PER_BETA0, PER_BETA1, steps)
        if random.random() < eps:
            a = shielded_action(None, obs, allow_random=random.randint(0, N_ACTIONS - 1))
        else:
            with torch.no_grad(): q = online(torch.FloatTensor(obs))
            a = shielded_action(q, obs)
        nobs, r, term, trunc, info = env.step(a); done = term or trunc; er += r
        buf.add(obs, a, r, nobs, term); obs = nobs
        if done:
            epr.append(er); er = 0.0; env = Phase2Env_NYU_rel(smp(stage)); obs, _ = env.reset()
            if frac > EPS_FRACTION and len(epr) == epr.maxlen:
                m = float(np.mean(epr))
                if m > best: best = m; _save(); saved = True
        if step >= LEARNING_STARTS and step % TRAIN_FREQ == 0 and len(buf) >= BATCH:
            ib, wb, ob, ab, rb_, nb, db = buf.sample(BATCH, beta)
            ot = torch.FloatTensor(ob); nt = torch.FloatTensor(nb); at = torch.LongTensor(ab)
            rt = torch.FloatTensor(rb_); dt = torch.FloatTensor(db); wt = torch.FloatTensor(wb)
            with torch.no_grad():
                ba = online(nt).argmax(1, keepdim=True); nq = target(nt).gather(1, ba).squeeze(1)
                tq = rt + (GAMMA ** NSTEP) * nq * (1 - dt)
            cq = online(ot).gather(1, at.unsqueeze(1)).squeeze(1); td = (cq - tq).detach().cpu().numpy()
            loss = (wt * nn.functional.huber_loss(cq, tq, reduction='none')).mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(online.parameters(), 10.0); opt.step()
            buf.update_priorities(ib, td)
        if step % TARGET_UPDATE == 0: polyak_update(online, target, TAU)
        if step % 25_000 == 0:
            mr = float(np.mean(epr)) if epr else 0.0
            print(f"    [{step:7d}/{steps}] {stage:<9} mean_rew={mr:8.2f} eps={eps:.3f} best={best:8.1f}", flush=True)
    if not saved: _save()
    return (time.time() - t0) / 60.0


def evaluate(net, pool):
    net.eval(); su = []; pw = []; sp = []; tpv = []; rl = []
    for seed in EVAL_SEEDS:
        r = np.random.default_rng(seed); real = good = nep = rlf = 0; P = []; S = []; T = []
        for _ in range(EVAL_EP):
            traj = pool[r.integers(len(pool))]
            env = SubTHzEnv6G_NYU(traj, array_gain_dbi=GAIN, w_mob=W_MOB, w_pwr=W_PWR, w_spec=W_SPEC)
            obs, _ = env.reset(); done = False; nep += 1; pts = []; rbs = []; et = []; info = {}
            while not done:
                with torch.no_grad(): q = net(torch.FloatTensor(obs))
                a = shielded_action(q, obs)                       # shield applied at eval too
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
    return dict(HO=float(np.mean(su)), Power=float(np.mean(pw)), Spec=float(np.mean(sp)), Tput=float(np.mean(tpv)), RLF=float(np.mean(rl)))


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    steps   = int(sys.argv[2]) if len(sys.argv) > 2 else 400_000
    mode    = sys.argv[3] if len(sys.argv) > 3 else 'full'
    slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH); pool = slow + med
    print(f"SHIELDED Safe-DQN | {n_seeds} seeds x {steps:,} | save_line={SAVE_LINE} dBm | pool={len(pool)}", flush=True)
    os.makedirs('results', exist_ok=True)
    if mode == 'validate':
        mins = train(0, 'safe_validate_s0.pt', slow, med, steps)
        m = evaluate(load_dndqn_checkpoint('safe_validate_s0.pt')[0], pool)
        print(f"\n[VALIDATE] HO {m['HO']:.2f}  Power {m['Power']:.2f}  Spec {m['Spec']:.2f}  RLF {m['RLF']:.2f}  ({mins:.1f}m)", flush=True)
        return
    path = 'results/safe_dqn_nyu.csv'; done = set(); rows = []
    if os.path.exists(path):
        for row in csv.DictReader(open(path)): done.add(int(row['seed'])); rows.append(row)
    for seed in range(n_seeds):
        if seed in done: print(f"[skip] seed {seed}", flush=True); continue
        ck = f"ms_safe_s{seed}.pt"; print(f"[train] Safe-DQN seed {seed} -> {ck}", flush=True)
        mins = train(seed, ck, slow, med, steps); m = evaluate(load_dndqn_checkpoint(ck)[0], pool)
        rows.append({'seed': seed, 'HO': f"{m['HO']:.2f}", 'Power': f"{m['Power']:.2f}", 'Spec': f"{m['Spec']:.2f}",
                     'RLF': f"{m['RLF']:.2f}", 'Tput': f"{m['Tput']:.1f}", 'train_min': f"{mins:.1f}"})
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['seed', 'HO', 'Power', 'Spec', 'RLF', 'Tput', 'train_min']); w.writeheader(); w.writerows(rows)
        print(f"[done] seed {seed}: RLF {m['RLF']:.1f}% Pwr {m['Power']:.1f}% Spec {m['Spec']:.1f}% HO {m['HO']:.1f}% ({mins:.0f}m)", flush=True)
    v = {k: [float(r[k]) for r in rows] for k in ['HO', 'Power', 'Spec', 'RLF']}
    g = lambda a: (float(np.mean(a)), float(np.std(a)))
    print("\n" + "=" * 70)
    for k in ['HO', 'Power', 'Spec', 'RLF']:
        mn, sd = g(v[k]); print(f"  {k:>6} : {mn:7.2f} +/- {sd:.2f}")
    print("  (refs: Proposed flat 8.98 RLF | guarded branched 4.61 | StdDQN 19.29)")
    print("  -> results/safe_dqn_nyu.csv", flush=True)


if __name__ == "__main__":
    main()
