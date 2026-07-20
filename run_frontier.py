#!/usr/bin/env python3
"""
run_frontier.py [steps] — trace the reliability-efficiency FRONTIER of the proposed Dn-DQN under
the NYU 142 GHz CI channel by sweeping the reward tilt from efficiency to reliability (1 seed each).

Answers the key question: is there a "knee" that saves meaningful power AND spectrum at an
acceptable RLF, or is power fundamentally a reliability resource at sub-THz (a cliff)?

Each config differs ONLY in the reward weights + RLF penalty; same channel, arch, schedule, budget.
Resumable (skips configs already in results/frontier_nyu.csv).

Usage:
    python -u run_frontier.py            # 5 configs x 500k steps each  (~2 hr) -> results/frontier_nyu.csv
    python -u run_frontier.py 800000     # 800k steps each (matches headline runs; ~3.3 hr)
"""
import sys, os, time, random, math
from collections import deque
import numpy as np, torch, torch.nn as nn, torch.optim as optim, csv

from env_6g_nyu import SubTHzEnv6G_NYU
from env_6g_v6 import N310_DEFAULT
from env_6g_v5 import (N_ACTIONS, PWR_DOWN, RB_DOWN, PT_MIN_DBM, PT_MAX_DBM, PT_STEP_DBM,
                       RB_MIN, RB_MAX, W_MOB as _WM, W_PWR as _WP, W_SPEC as _WS)
from dndqn import (build_dndqn_networks, NStepPERBuffer, polyak_update,
                   epsilon_schedule, beta_schedule, OBS_DIM, NET_ARCH, load_dndqn_checkpoint)
import train_phase2 as tp
import multi_seed_eval as mse

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 500_000
SEED = 0
GAIN, RB_BW_HZ, GAMMA = 15.0, 380.16e6 / RB_MAX, 0.99
EVAL_SEEDS, EVAL_EP = mse.SEEDS, mse.N_EPISODES
EPS_START, EPS_END, EPS_FRACTION, STAGE_SLOW_END = 1.0, 0.05, 0.30, 0.08
LEARNING_STARTS, TRAIN_FREQ, BATCH, TARGET_UPDATE, TAU = 10_000, 4, 256, 500, 0.01
PER_BETA0, PER_BETA1, NSTEP, ALPHA = 0.4, 1.0, 3, 0.6
SAFE_RSRP_THRESH, PWR_DOWN_UNSAFE_PEN = -78.0, -0.20

# (label, w_mob, w_pwr, w_spec, rlf_penalty) — efficiency -> reliability
CONFIGS = [
    ('C1_eff', 0.60, 0.200, 0.200, -100),
    ('C2',     0.65, 0.175, 0.175, -150),
    ('C3_mid', 0.70, 0.150, 0.150, -200),
    ('C4',     0.75, 0.125, 0.125, -250),
    ('C5_rel', 0.80, 0.100, 0.100, -300),
]

class FrontierEnv(SubTHzEnv6G_NYU):
    """NYU-channel env with parameterised reward (weights + RLF penalty); same shaping as Phase2Env_NYU."""
    def __init__(self, traj, wm, wp, ws, rlf_pen):
        super().__init__(traj, array_gain_dbi=GAIN, n310=N310_DEFAULT, w_mob=wm, w_pwr=wp, w_spec=ws)
        self.wm, self.wp, self.ws, self.rlf_pen = wm, wp, ws, rlf_pen
        self.pstep = wp * (PT_STEP_DBM / (PT_MAX_DBM - PT_MIN_DBM)) * 10.0
        self.rstep = ws * (PT_STEP_DBM / (PT_MAX_DBM - PT_MIN_DBM)) * 10.0
        self._prev = -70.0
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options); self._prev = -70.0; return obs, info
    def step(self, action):
        before = self._prev
        obs, _, term, trunc, info = super().step(action); self._prev = info['rsrp_dbm']
        rm = info.get('r_mobility', 0.0); rp = info.get('r_power', 0.0); rs = info.get('r_spectrum', 0.0)
        reward = self.wm * rm
        if action == PWR_DOWN: reward += self.pstep if before >= SAFE_RSRP_THRESH else PWR_DOWN_UNSAFE_PEN
        else: reward += self.wp * rp
        if action == RB_DOWN: reward += self.rstep
        else: reward += self.ws * rs
        if term and info.get('sustained_rlf'): reward += self.rlf_pen
        return obs, reward, term, trunc, info

def set_seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s)

def train(cfg, out, slow, med):
    label, wm, wp, ws, rlf = cfg; set_seed(SEED); pool = slow + med
    online, target = build_dndqn_networks(OBS_DIM, N_ACTIONS, NET_ARCH, 'cpu'); target.load_state_dict(online.state_dict())
    opt = optim.Adam(online.parameters(), lr=3e-4); buf = NStepPERBuffer(500_000, n_steps=NSTEP, gamma=GAMMA, alpha=ALPHA)
    rng = np.random.default_rng(SEED)
    smp = lambda stg: (slow if stg == "SLOW" else pool)[rng.integers(len(slow if stg == "SLOW" else pool))]
    stage = "SLOW"; env = FrontierEnv(smp(stage), wm, wp, ws, rlf); obs, _ = env.reset()
    epr = deque(maxlen=20); er = 0.0; best = -1e18; saved = False
    def _save():
        torch.save({'q_net_state': online.state_dict(), 'obs_dim': OBS_DIM, 'n_actions': N_ACTIONS,
                    'net_arch': NET_ARCH, 'algorithm': 'Dn-DQN', 'n_steps': NSTEP, 'config': label}, out)
    for step in range(1, STEPS + 1):
        frac = step / STEPS; stage = "SLOW" if frac < STAGE_SLOW_END else "SLOW+MED"
        eps = epsilon_schedule(step, EPS_START, EPS_END, EPS_FRACTION, STEPS); beta = beta_schedule(step, PER_BETA0, PER_BETA1, STEPS)
        if random.random() < eps: a = random.randint(0, N_ACTIONS - 1)
        else:
            with torch.no_grad(): a = online(torch.FloatTensor(obs)).argmax().item()
        nobs, r, t, tr, info = env.step(a); done = t or tr; er += r; buf.add(obs, a, r, nobs, t); obs = nobs
        if done:
            epr.append(er); er = 0.0; env = FrontierEnv(smp(stage), wm, wp, ws, rlf); obs, _ = env.reset()
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

def evaluate(net, pool):
    net.eval(); su = []; pw = []; sp = []; tpv = []; rl = []
    for seed in EVAL_SEEDS:
        r = np.random.default_rng(seed); real = good = nep = rlf = 0; P = []; S = []; T = []
        for _ in range(EVAL_EP):
            traj = pool[r.integers(len(pool))]; env = SubTHzEnv6G_NYU(traj, array_gain_dbi=GAIN, w_mob=_WM, w_pwr=_WP, w_spec=_WS)
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
    print(f"FRONTIER sweep | {len(CONFIGS)} reward tilts | {STEPS:,} steps each (seed {SEED}) | NYU CI channel", flush=True)
    slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH); pool = slow + med
    os.makedirs('results', exist_ok=True); path = 'results/frontier_nyu.csv'
    done = {}; rows = []
    if os.path.exists(path):
        with open(path) as f:
            for row in csv.DictReader(f): done[row['config']] = 1; rows.append(row)
    for cfg in CONFIGS:
        label = cfg[0]
        if label in done: print(f"[skip] {label} (done)", flush=True); continue
        ck = f"fr_{label}.pt"; print(f"[train] {label}  w={cfg[1]}/{cfg[2]}/{cfg[3]} rlf={cfg[4]} -> {ck} ...", flush=True)
        t0 = time.time(); train(cfg, ck, slow, med); m = evaluate(load_dndqn_checkpoint(ck)[0], pool); mins = (time.time() - t0) / 60
        rows.append({'config': label, 'w_mob': cfg[1], 'w_pwr': cfg[2], 'w_spec': cfg[3], 'rlf_pen': cfg[4],
                     'HO': f"{m['HO']:.2f}", 'Power': f"{m['Power']:.2f}", 'Spec': f"{m['Spec']:.2f}", 'RLF': f"{m['RLF']:.2f}", 'Tput': f"{m['Tput']:.1f}"})
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['config', 'w_mob', 'w_pwr', 'w_spec', 'rlf_pen', 'HO', 'Power', 'Spec', 'RLF', 'Tput']); w.writeheader(); w.writerows(rows)
        print(f"[done] {label}: HO {m['HO']:.1f}% Power {m['Power']:.1f}% Spec {m['Spec']:.1f}% RLF {m['RLF']:.1f}% Tput {m['Tput']:.0f} ({mins:.1f}m)", flush=True)
    print("\n" + "=" * 94); print(f"  FRONTIER (proposed Dn-DQN, seed {SEED}, {STEPS:,} steps) — efficiency -> reliability"); print("=" * 94)
    print(f"  {'config':>8} | {'w(mob/pwr/spec), rlf':>26} | {'HO%':>6} | {'Power%':>7} | {'Spec%':>6} | {'RLF%':>6} | {'Tput':>6}")
    print("  " + "-" * 84)
    for r in rows:
        print(f"  {r['config']:>8} | {r['w_mob']}/{r['w_pwr']}/{r['w_spec']}, {str(r['rlf_pen']):>5} | {float(r['HO']):6.1f} | {float(r['Power']):7.1f} | {float(r['Spec']):6.1f} | {float(r['RLF']):6.1f} | {float(r['Tput']):6.0f}")
    print("\n  A3 reference: HO 96.2 / Power 0 / Spec 0 / RLF 5.1 / Tput 1909")
    print("  -> results/frontier_nyu.csv", flush=True)

if __name__ == "__main__":
    main()
