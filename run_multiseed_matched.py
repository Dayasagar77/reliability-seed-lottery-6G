#!/usr/bin/env python3
"""
run_multiseed_matched.py [n_seeds] [steps] — MATCHED-OBJECTIVE multi-seed comparison at the
RELIABLE operating point. Trains Proposed / DDQN+PER / Standard DQN N times each under the SAME
reliability-tilted reward (Phase2Env_NYU_rel: W 0.8/0.1/0.1, RLF penalty -300) on the NYU CI
channel, evaluates each, and reports mean +/- std per method.

Answers: at the reliable corner (~5% RLF, ~90% spectrum saved), does the proposed architecture
(dueling + n-step + PER) actually get there better than the ablated baselines, or do they all
reach the same corner (=> the frontier, not the architecture, is the contribution)? Same protocol
and budget for all methods; only the algorithm ingredients differ. Resumable.

Usage:
    python -u run_multiseed_matched.py 3            # 3 seeds x 3 methods, 400k steps  (~3 hr)
    python -u run_multiseed_matched.py 5 500000     # 5 seeds, 500k steps each
"""
import sys, os, time, random, math
from collections import deque
import numpy as np, torch, torch.nn as nn, torch.optim as optim, csv

from train_phase2_nyu_rel import Phase2Env_NYU_rel          # reliability reward (matched objective)
from dndqn import (build_dndqn_networks, NStepPERBuffer, polyak_update,
                   epsilon_schedule, beta_schedule, OBS_DIM, NET_ARCH, load_dndqn_checkpoint)
from qnet_plain import PlainQNet, load_plain
from env_6g_nyu import SubTHzEnv6G_NYU
from env_6g_v5 import N_ACTIONS, PT_MIN_DBM, PT_MAX_DBM, RB_MIN, RB_MAX, W_MOB, W_PWR, W_SPEC
import train_phase2 as tp
import multi_seed_eval as mse

N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
STEPS   = int(sys.argv[2]) if len(sys.argv) > 2 else 400_000
GAIN, RB_BW_HZ, GAMMA = 15.0, 380.16e6 / RB_MAX, 0.99
EVAL_SEEDS, EVAL_EP = mse.SEEDS, mse.N_EPISODES
EPS_START, EPS_END, EPS_FRACTION, STAGE_SLOW_END = 1.0, 0.05, 0.30, 0.08
LEARNING_STARTS, TRAIN_FREQ, BATCH, TARGET_UPDATE, TAU = 10_000, 4, 256, 500, 0.01
PER_BETA0, PER_BETA1 = 0.4, 1.0
METHODS = [('Proposed Dn-DQN', 'dndqn'), ('DDQN+PER', 'ddqn'), ('Standard DQN', 'std')]

def cfg(method):   # (dueling, double, n_step, per_alpha)
    return {'dndqn': (True, True, 3, 0.6), 'ddqn': (False, True, 1, 0.6),
            'std': (False, False, 1, 0.0)}[method]

def set_seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s)

def build(method):
    if cfg(method)[0]:
        o, t = build_dndqn_networks(OBS_DIM, N_ACTIONS, NET_ARCH, 'cpu')
    else:
        o, t = PlainQNet(OBS_DIM, N_ACTIONS, NET_ARCH), PlainQNet(OBS_DIM, N_ACTIONS, NET_ARCH)
    t.load_state_dict(o.state_dict()); return o, t

def train(method, seed, out, slow, med):
    set_seed(seed); _, double, nstep, alpha = cfg(method); pool = slow + med
    online, target = build(method)
    opt = optim.Adam(online.parameters(), lr=3e-4); buf = NStepPERBuffer(500_000, n_steps=nstep, gamma=GAMMA, alpha=alpha)
    rng = np.random.default_rng(seed)
    smp = lambda stg: (slow if stg == "SLOW" else pool)[rng.integers(len(slow if stg == "SLOW" else pool))]
    stage = "SLOW"; env = Phase2Env_NYU_rel(smp(stage)); obs, _ = env.reset()
    epr = deque(maxlen=20); er = 0.0; best = -1e18; saved = False
    def _save(): torch.save({'q_net_state': online.state_dict(), 'obs_dim': OBS_DIM, 'n_actions': N_ACTIONS,
                             'net_arch': NET_ARCH, 'algorithm': method, 'n_steps': nstep, 'seed': seed}, out)
    for step in range(1, STEPS + 1):
        frac = step / STEPS; stage = "SLOW" if frac < STAGE_SLOW_END else "SLOW+MED"
        eps = epsilon_schedule(step, EPS_START, EPS_END, EPS_FRACTION, STEPS); beta = beta_schedule(step, PER_BETA0, PER_BETA1, STEPS)
        if random.random() < eps: a = random.randint(0, N_ACTIONS - 1)
        else:
            with torch.no_grad(): a = online(torch.FloatTensor(obs)).argmax().item()
        nobs, r, t, tr, info = env.step(a); done = t or tr; er += r; buf.add(obs, a, r, nobs, t); obs = nobs
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
                if double:
                    ba = online(nt).argmax(1, keepdim=True); nq = target(nt).gather(1, ba).squeeze(1)
                else:
                    nq = target(nt).max(1).values
                tq = rt + (GAMMA ** nstep) * nq * (1 - dt)
            cq = online(ot).gather(1, at.unsqueeze(1)).squeeze(1); td = (cq - tq).detach().cpu().numpy()
            loss = (wt * nn.functional.huber_loss(cq, tq, reduction='none')).mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(online.parameters(), 10.0); opt.step()
            buf.update_priorities(ib, td)
        if step % TARGET_UPDATE == 0: polyak_update(online, target, TAU)
    if not saved: _save()

def load_net(method, path):
    return (load_dndqn_checkpoint(path)[0] if cfg(method)[0] else load_plain(path)[0])

def evaluate(net, pool):
    net.eval(); su = []; pw = []; sp = []; tpv = []; rl = []
    for seed in EVAL_SEEDS:
        r = np.random.default_rng(seed); real = good = nep = rlf = 0; P = []; S = []; T = []
        for _ in range(EVAL_EP):
            traj = pool[r.integers(len(pool))]; env = SubTHzEnv6G_NYU(traj, array_gain_dbi=GAIN, w_mob=W_MOB, w_pwr=W_PWR, w_spec=W_SPEC)
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
    return dict(HO=float(np.mean(su)), Power=float(np.mean(pw)), Spec=float(np.mean(sp)), Tput=float(np.mean(tpv)), RLF=float(np.mean(rl)))

def main():
    print(f"MATCHED comparison @ reliable corner | {N_SEEDS} seeds x {len(METHODS)} methods | {STEPS:,} steps | reliability reward", flush=True)
    slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH); pool = slow + med
    os.makedirs('results', exist_ok=True); rows_path = 'results/matched_rel_nyu.csv'
    done = set(); rows = []
    if os.path.exists(rows_path):
        with open(rows_path) as f:
            for row in csv.DictReader(f): done.add((row['method'], int(row['seed']))); rows.append(row)
    for label, method in METHODS:
        for seed in range(N_SEEDS):
            if (label, seed) in done: print(f"[skip] {label} seed {seed} (done)", flush=True); continue
            ck = f"mm_{method}_s{seed}.pt"; print(f"[train] {label} seed {seed} -> {ck} ({STEPS:,} steps) ...", flush=True)
            t0 = time.time(); train(method, seed, ck, slow, med); m = evaluate(load_net(method, ck), pool); mins = (time.time() - t0) / 60
            rows.append({'method': label, 'seed': seed, 'HO': f"{m['HO']:.2f}", 'Power': f"{m['Power']:.2f}",
                         'Spec': f"{m['Spec']:.2f}", 'RLF': f"{m['RLF']:.2f}", 'Tput': f"{m['Tput']:.1f}"})
            with open(rows_path, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=['method', 'seed', 'HO', 'Power', 'Spec', 'RLF', 'Tput']); w.writeheader(); w.writerows(rows)
            print(f"[done] {label} s{seed}: HO {m['HO']:.1f}% Pwr {m['Power']:.1f}% Spec {m['Spec']:.1f}% RLF {m['RLF']:.1f}% Tput {m['Tput']:.0f} ({mins:.1f}m)", flush=True)
    print("\n" + "=" * 96); print(f"  MATCHED @ RELIABLE CORNER — mean +/- std over {N_SEEDS} seeds ({STEPS:,} steps, reliability reward)"); print("=" * 96)
    print(f"  {'method':>16} | {'HOsucc%':>12} | {'Power%':>12} | {'Spec%':>12} | {'RLF%':>12} | {'Tput':>11}")
    print("  " + "-" * 88)
    agg = []
    for label, method in METHODS:
        v = {k: [float(r[k]) for r in rows if r['method'] == label] for k in ['HO', 'Power', 'Spec', 'RLF', 'Tput']}
        if not v['HO']: continue
        g = lambda a: (np.mean(a), np.std(a))
        ho, pw, sp, rl, tpv = g(v['HO']), g(v['Power']), g(v['Spec']), g(v['RLF']), g(v['Tput'])
        print(f"  {label:>16} | {ho[0]:5.1f}+/-{ho[1]:4.1f} | {pw[0]:5.1f}+/-{pw[1]:4.1f} | {sp[0]:5.1f}+/-{sp[1]:4.1f} | {rl[0]:5.1f}+/-{rl[1]:4.1f} | {tpv[0]:5.0f}+/-{tpv[1]:3.0f}")
        agg.append((label, ho, pw, sp, rl, tpv))
    print("\n  A3 reference: HO 96.2 / Power 0 / Spec 0 / RLF 5.1 / Tput 1909")
    with open('results/matched_rel_summary.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['method', 'HO_mean', 'HO_std', 'Power_mean', 'Power_std', 'Spec_mean', 'Spec_std', 'RLF_mean', 'RLF_std', 'Tput_mean', 'Tput_std'])
        for label, ho, pw, sp, rl, tpv in agg:
            w.writerow([label, f"{ho[0]:.2f}", f"{ho[1]:.2f}", f"{pw[0]:.2f}", f"{pw[1]:.2f}", f"{sp[0]:.2f}", f"{sp[1]:.2f}", f"{rl[0]:.2f}", f"{rl[1]:.2f}", f"{tpv[0]:.1f}", f"{tpv[1]:.1f}"])
    print("  -> results/matched_rel_nyu.csv + results/matched_rel_summary.csv", flush=True)

if __name__ == "__main__":
    main()
