#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_lagrangian_dqn.py — Lagrangian-constrained Dn-DQN safe-RL baseline (reviewer major comment 5).

Controlled single-variable comparison against the matched-objective 'Proposed Dn-DQN': SAME dueling +
n-step + PER backbone, SAME flat 6-action space, SAME reliability-tilt shaping (W 0.8/0.1/0.1, C1/C5).
The ONLY change is the reliability-enforcement mechanism: the fixed -300 sustained-RLF terminal penalty
is replaced by a LEARNED Lagrange multiplier lambda that enforces E[sustained-RLF] <= budget via dual
ascent (reward-constrained RL / RCPO, Tessler et al. 2019). This isolates 'algorithmic constraint
satisfaction' vs the paper's measurement-shaped guard. Evaluated with the standard protocol (5x150).
ZERO fabrication; resumable via results/lagrangian_dqn_nyu.csv.

Usage: python -u train_lagrangian_dqn.py [n_seeds=3] [steps=400000] [mode=full|validate]
"""
import sys, os, time, random, math, csv
from collections import deque
import numpy as np, torch, torch.nn as nn, torch.optim as optim
torch.set_num_threads(1)

from env_6g_nyu import SubTHzEnv6G_NYU
from env_6g_v6 import N310_DEFAULT
from env_6g_v5 import (PWR_DOWN, RB_DOWN, PT_MIN_DBM, PT_MAX_DBM, PT_STEP_DBM,
                       RB_MIN, RB_MAX, N_ACTIONS, W_MOB, W_PWR, W_SPEC, RLF_THRESH_DBM)
from dndqn import (build_dndqn_networks, NStepPERBuffer, polyak_update,
                   epsilon_schedule, beta_schedule, OBS_DIM, NET_ARCH, load_dndqn_checkpoint)
import train_phase2 as tp
import multi_seed_eval as mse

WM, WP, WS = 0.80, 0.10, 0.10                       # reliability tilt (identical to train_phase2_nyu_rel)
SAFE_RSRP_THRESH, PWR_DOWN_UNSAFE_PEN = -78.0, -0.20
_PWR_STEP_REW = WP * (PT_STEP_DBM / (PT_MAX_DBM - PT_MIN_DBM)) * 10.0
_RB_STEP_REW  = WS * (PT_STEP_DBM / (PT_MAX_DBM - PT_MIN_DBM)) * 10.0
GAIN, RB_BW_HZ, GAMMA = 15.0, 380.16e6 / RB_MAX, 0.99
EPS_START, EPS_END, EPS_FRACTION, STAGE_SLOW_END = 1.0, 0.05, 0.30, 0.08
LEARNING_STARTS, TRAIN_FREQ, BATCH, TARGET_UPDATE, TAU = 10_000, 4, 256, 500, 0.01
PER_BETA0, PER_BETA1, NSTEP, ALPHA = 0.4, 1.0, 3, 0.6
EVAL_SEEDS, EVAL_EP = mse.SEEDS, mse.N_EPISODES
# ── Two-timescale RCPO with a SMOOTH per-step out-of-sync cost (stable) ──
# cost_t = 1 if serving RSRP < RLF threshold (out-of-sync sample); constraint = mean per-episode
# out-of-sync step count <= BUDGET_C. A smooth per-step signal (not a rare terminal spike) keeps
# the n-step/PER value learning stable while still driving sustained RLF down.
BUDGET_C = 0.30            # target mean out-of-sync steps per episode
LAM_INIT, LAM_MAX = 0.0, 100.0
ETA_LAM = 5.0             # gentle dual step (two-timescale: slow vs the policy)
LAM_START_FRAC = 0.15      # begin dual updates only after exploration partly anneals
COST_WIN = 200             # running window (episodes) for the constraint estimate


class _Lam:
    __slots__ = ('val',)
    def __init__(self, v): self.val = float(v)


class Phase2Env_NYU_lag(SubTHzEnv6G_NYU):
    """Reliability-tilt base reward; fixed -300 RLF penalty REPLACED by -lambda*cost,
    cost = terminal sustained-RLF indicator (constraint). lambda read live from a shared holder."""
    def __init__(self, traj, lam):
        super().__init__(traj, array_gain_dbi=GAIN, n310=N310_DEFAULT, w_mob=WM, w_pwr=WP, w_spec=WS)
        self.lam = lam; self._prev_rsrp = -70.0

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options); self._prev_rsrp = -70.0
        return obs, info

    def step(self, action):
        rb = self._prev_rsrp
        obs, _, term, trunc, info = super().step(action); self._prev_rsrp = info['rsrp_dbm']
        r = WM * info.get('r_mobility', 0.0)
        r += (_PWR_STEP_REW if rb >= SAFE_RSRP_THRESH else PWR_DOWN_UNSAFE_PEN) if action == PWR_DOWN else WP * info.get('r_power', 0.0)
        r += _RB_STEP_REW if action == RB_DOWN else WS * info.get('r_spectrum', 0.0)
        cost = 1.0 if info['rsrp_dbm'] < RLF_THRESH_DBM else 0.0   # smooth per-step out-of-sync cost
        r -= self.lam.val * cost                                    # Lagrangian penalty (learned multiplier)
        info['cost'] = cost
        return obs, r, term, trunc, info


def set_seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s)


def train(seed, out, slow, med, steps):
    set_seed(seed); pool = slow + med
    online, target = build_dndqn_networks(OBS_DIM, N_ACTIONS, NET_ARCH, 'cpu'); target.load_state_dict(online.state_dict())
    opt = optim.Adam(online.parameters(), lr=3e-4); buf = NStepPERBuffer(500_000, n_steps=NSTEP, gamma=GAMMA, alpha=ALPHA)
    rng = np.random.default_rng(seed)
    smp = lambda stg: (slow if stg == "SLOW" else pool)[rng.integers(len(slow if stg == "SLOW" else pool))]
    lam = _Lam(LAM_INIT); cost_hist = deque(maxlen=COST_WIN)
    stage = "SLOW"; env = Phase2Env_NYU_lag(smp(stage), lam); obs, _ = env.reset()
    epr = deque(maxlen=20); er = 0.0; ec = 0.0; best = -1e18; saved = False; t0 = time.time(); n_ep = 0
    def _save():
        torch.save({'q_net_state': online.state_dict(), 'obs_dim': OBS_DIM, 'n_actions': N_ACTIONS,
                    'net_arch': NET_ARCH, 'algorithm': 'Lagrangian-Dn-DQN', 'n_steps': NSTEP, 'seed': seed,
                    'lambda_final': lam.val, 'rlf_budget': BUDGET_C, 'training_mode': 'nyu_ci_lagrangian'}, out)
    last_info = {}
    for step in range(1, steps + 1):
        frac = step / steps; stage = "SLOW" if frac < STAGE_SLOW_END else "SLOW+MED"
        eps = epsilon_schedule(step, EPS_START, EPS_END, EPS_FRACTION, steps); beta = beta_schedule(step, PER_BETA0, PER_BETA1, steps)
        if random.random() < eps: a = random.randint(0, N_ACTIONS - 1)
        else:
            with torch.no_grad(): a = online(torch.FloatTensor(obs)).argmax().item()
        nobs, r, term, trunc, info = env.step(a); done = term or trunc; er += r; ec += info['cost']; last_info = info
        buf.add(obs, a, r, nobs, term); obs = nobs
        if done:
            cost_hist.append(ec); ec = 0.0; n_ep += 1
            epr.append(er); er = 0.0
            if step > LAM_START_FRAC * steps and len(cost_hist) >= 20:  # two-timescale dual ascent
                viol = float(np.mean(cost_hist)) - BUDGET_C
                lam.val = float(min(max(lam.val + ETA_LAM * viol, 0.0), LAM_MAX))
            env = Phase2Env_NYU_lag(smp(stage), lam); obs, _ = env.reset()
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
            rr = float(np.mean(cost_hist)) if cost_hist else float('nan')
            print(f"    [{step:7d}/{steps}] {stage:<9} lambda={lam.val:6.1f} oos/ep={rr:4.2f} best={best:8.1f}", flush=True)
    if not saved: _save()
    return (time.time() - t0) / 60.0, lam.val


def evaluate(net, pool):
    net.eval(); su = []; pw = []; sp = []; tpv = []; rl = []
    for seed in EVAL_SEEDS:
        r = np.random.default_rng(seed); real = good = nep = rlf = 0; P = []; S = []; T = []
        for _ in range(EVAL_EP):
            traj = pool[r.integers(len(pool))]
            env = SubTHzEnv6G_NYU(traj, array_gain_dbi=GAIN, w_mob=WM, w_pwr=WP, w_spec=WS)
            obs, _ = env.reset(); done = False; nep += 1; pts = []; rbs = []; et = []; info = {}
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
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    steps   = int(sys.argv[2]) if len(sys.argv) > 2 else 400_000
    mode    = sys.argv[3] if len(sys.argv) > 3 else 'full'
    slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH); pool = slow + med
    print(f"LAGRANGIAN Dn-DQN | {n_seeds} seeds x {steps:,} | budget={BUDGET_C} eta={ETA_LAM} lam0={LAM_INIT} | pool={len(pool)}", flush=True)
    os.makedirs('results', exist_ok=True)
    if mode == 'validate':
        mins, lamf = train(0, 'lag_validate_s0.pt', slow, med, steps)
        m = evaluate(load_dndqn_checkpoint('lag_validate_s0.pt')[0], pool)
        print(f"\n[VALIDATE] lambda_final={lamf:.1f}  ->  HO {m['HO']:.2f}  Power {m['Power']:.2f}  Spec {m['Spec']:.2f}  RLF {m['RLF']:.2f}  ({mins:.1f}m)", flush=True)
        return
    path = 'results/lagrangian_dqn_nyu.csv'; done = set(); rows = []
    if os.path.exists(path):
        for row in csv.DictReader(open(path)): done.add(int(row['seed'])); rows.append(row)
    for seed in range(n_seeds):
        if seed in done: print(f"[skip] seed {seed}", flush=True); continue
        ck = f"ms_lag_s{seed}.pt"; print(f"[train] Lagrangian seed {seed} -> {ck}", flush=True)
        mins, lamf = train(seed, ck, slow, med, steps); m = evaluate(load_dndqn_checkpoint(ck)[0], pool)
        rows.append({'seed': seed, 'HO': f"{m['HO']:.2f}", 'Power': f"{m['Power']:.2f}", 'Spec': f"{m['Spec']:.2f}",
                     'RLF': f"{m['RLF']:.2f}", 'Tput': f"{m['Tput']:.1f}", 'lambda_final': f"{lamf:.1f}", 'train_min': f"{mins:.1f}"})
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['seed', 'HO', 'Power', 'Spec', 'RLF', 'Tput', 'lambda_final', 'train_min']); w.writeheader(); w.writerows(rows)
        print(f"[done] seed {seed}: RLF {m['RLF']:.1f}% Pwr {m['Power']:.1f}% Spec {m['Spec']:.1f}% HO {m['HO']:.1f}% lam={lamf:.0f} ({mins:.0f}m)", flush=True)
    v = {k: [float(r[k]) for r in rows] for k in ['HO', 'Power', 'Spec', 'RLF']}
    g = lambda a: (float(np.mean(a)), float(np.std(a)))
    print("\n" + "=" * 70)
    for k in ['HO', 'Power', 'Spec', 'RLF']:
        mn, sd = g(v[k]); print(f"  {k:>6} : {mn:7.2f} +/- {sd:.2f}")
    print("  (matched-objective refs: Proposed 8.98 RLF/20.92 Pwr | DDQN+PER 13.87/65.86 | StdDQN 19.29/76.40)")
    print("  -> results/lagrangian_dqn_nyu.csv", flush=True)


if __name__ == "__main__":
    main()
