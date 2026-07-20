#!/usr/bin/env python3
"""fix_reward_ablation_nyu.py — reward-component ablation on the NYU 142 GHz CI channel
(channel-consistency fix). Replicates train_reward_ablation_v6.py's toggles EXACTLY, on the NYU
reliability reward (Phase2Env_NYU_rel):
  C1 = RSRP-gated PWR_DOWN shaping | C3 = sustained-RLF terminal penalty (-300) | C5 = symmetric RB_DOWN reward
Warm-starts from the full reliability model (dqn_phase2_dndqn_nyu_rel.pt) and fine-tunes each
one-component-disabled variant (~300k steps), then evaluates over 5 eval seeds. Because variants are
fine-tuned FROM the full model (not trained cold), the measured effect of removing a component is a
LOWER BOUND — same caveat as the v6 ablation. -> results/reward_ablation_nyu.csv
Waits for multiseed + n-step to finish first (single-thread CPU). Resumable via the CSV."""
import os, time, random, csv, subprocess
from collections import deque
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from env_6g_nyu import SubTHzEnv6G_NYU
from env_6g_v6 import N310_DEFAULT
from env_6g_v5 import PWR_DOWN, RB_DOWN, PT_MIN_DBM, PT_MAX_DBM, PT_STEP_DBM, N_ACTIONS
from dndqn import (build_dndqn_networks, NStepPERBuffer, polyak_update, epsilon_schedule,
                   beta_schedule, OBS_DIM, NET_ARCH, load_dndqn_checkpoint)
import train_phase2 as tp
from run_multiseed_rel import evaluate

W_MOB, W_PWR, W_SPEC = 0.80, 0.10, 0.10
ARRAY_GAIN = 15.0
SAFE_RSRP_THRESH, PWR_DOWN_UNSAFE_PEN = -78.0, -0.20
_PWR_STEP_REW = W_PWR  * (PT_STEP_DBM / (PT_MAX_DBM - PT_MIN_DBM)) * 10.0
_RB_STEP_REW  = W_SPEC * (PT_STEP_DBM / (PT_MAX_DBM - PT_MIN_DBM)) * 10.0
RLF_TERMINAL_PENALTY = -300.0
FULL_CKPT = "dqn_phase2_dndqn_nyu_rel.pt"
STEPS, GAMMA, NSTEP, ALPHA = 300_000, 0.99, 3, 0.6
EPS_START, EPS_END, EPS_FRACTION, STAGE_SLOW_END = 0.20, 0.05, 0.30, 0.08   # warm-start fine-tune
LEARNING_STARTS, TRAIN_FREQ, BATCH, TARGET_UPDATE, TAU = 5_000, 4, 256, 500, 0.01
PER_BETA0, PER_BETA1 = 0.4, 1.0

class AblatEnv(SubTHzEnv6G_NYU):
    def __init__(self, traj, C1, C3, C5):
        super().__init__(traj, array_gain_dbi=ARRAY_GAIN, n310=N310_DEFAULT, w_mob=W_MOB, w_pwr=W_PWR, w_spec=W_SPEC)
        self.C1, self.C3, self.C5 = C1, C3, C5; self._prev_rsrp = -70.0
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options); self._prev_rsrp = -70.0; return obs, info
    def step(self, action):
        rb = self._prev_rsrp
        obs, _, term, trunc, info = super().step(action); self._prev_rsrp = info['rsrp_dbm']
        reward = W_MOB * info.get('r_mobility', 0.0)
        if action == PWR_DOWN:
            reward += (_PWR_STEP_REW if rb >= SAFE_RSRP_THRESH else PWR_DOWN_UNSAFE_PEN) if self.C1 else W_PWR * info.get('r_power', 0.0)
        else:
            reward += W_PWR * info.get('r_power', 0.0)
        if action == RB_DOWN:
            reward += _RB_STEP_REW if self.C5 else W_SPEC * info.get('r_spectrum', 0.0)
        else:
            reward += W_SPEC * info.get('r_spectrum', 0.0)
        if self.C3 and term and info.get('sustained_rlf'):
            reward += RLF_TERMINAL_PENALTY
        return obs, reward, term, trunc, info

def set_seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s)

def finetune(disable, out, slow, med, seed=0):
    set_seed(seed); pool = slow + med
    C1, C3, C5 = disable != 'c1', disable != 'c3', disable != 'c5'
    online, target = build_dndqn_networks(OBS_DIM, N_ACTIONS, NET_ARCH, 'cpu')
    online.load_state_dict(load_dndqn_checkpoint(FULL_CKPT)[0].state_dict())        # warm-start
    target.load_state_dict(online.state_dict())
    opt = optim.Adam(online.parameters(), lr=3e-4); buf = NStepPERBuffer(500_000, n_steps=NSTEP, gamma=GAMMA, alpha=ALPHA)
    rng = np.random.default_rng(seed)
    smp = lambda stg: (slow if stg == "SLOW" else pool)[rng.integers(len(slow if stg == "SLOW" else pool))]
    stage = "SLOW"; env = AblatEnv(smp(stage), C1, C3, C5); obs, _ = env.reset()
    epr = deque(maxlen=20); er = 0.0; best = -1e18; saved = False
    def _save(): torch.save({'q_net_state': online.state_dict(), 'obs_dim': OBS_DIM, 'n_actions': N_ACTIONS,
                             'net_arch': NET_ARCH, 'algorithm': 'Dn-DQN', 'n_steps': NSTEP, 'disable': disable}, out)
    for step in range(1, STEPS + 1):
        frac = step / STEPS; stage = "SLOW" if frac < STAGE_SLOW_END else "SLOW+MED"
        eps = epsilon_schedule(step, EPS_START, EPS_END, EPS_FRACTION, STEPS); beta = beta_schedule(step, PER_BETA0, PER_BETA1, STEPS)
        if random.random() < eps: a = random.randint(0, N_ACTIONS - 1)
        else:
            with torch.no_grad(): a = online(torch.FloatTensor(obs)).argmax().item()
        nobs, r, term, trunc, info = env.step(a); done = term or trunc; er += r
        buf.add(obs, a, r, nobs, term); obs = nobs
        if done:
            epr.append(er); er = 0.0; env = AblatEnv(smp(stage), C1, C3, C5); obs, _ = env.reset()
            if frac > EPS_FRACTION and len(epr) == epr.maxlen:
                m = float(np.mean(epr))
                if m > best: best = m; _save(); saved = True
        if step >= LEARNING_STARTS and step % TRAIN_FREQ == 0 and len(buf) >= BATCH:
            ib, wb, ob, ab, rb2, nb, db = buf.sample(BATCH, beta)
            ot = torch.FloatTensor(ob); nt = torch.FloatTensor(nb); at = torch.LongTensor(ab)
            rt = torch.FloatTensor(rb2); dt = torch.FloatTensor(db); wt = torch.FloatTensor(wb)
            with torch.no_grad():
                ba = online(nt).argmax(1, keepdim=True); nq = target(nt).gather(1, ba).squeeze(1)
                tq = rt + (GAMMA ** NSTEP) * nq * (1 - dt)
            cq = online(ot).gather(1, at.unsqueeze(1)).squeeze(1); td = (cq - tq).detach().cpu().numpy()
            loss = (wt * nn.functional.huber_loss(cq, tq, reduction='none')).mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(online.parameters(), 10.0); opt.step()
            buf.update_priorities(ib, td)
        if step % TARGET_UPDATE == 0: polyak_update(online, target, TAU)
    if not saved: _save()

def _write(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['config', 'HO', 'Power', 'Spec', 'RLF', 'Tput']); w.writeheader(); w.writerows(rows)

def waiting():
    try: return bool(subprocess.run(["pgrep", "-f", "run_multiseed_rel|fix_nstep"], capture_output=True, text=True).stdout.strip())
    except Exception: return False

def main():
    while waiting(): print("[wait] multiseed/nstep running; sleep 120s", flush=True); time.sleep(120)
    print("Starting reward-component ablation on NYU (warm-start fine-tune, effects = lower bound)", flush=True)
    slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH); pool = slow + med
    os.makedirs('results', exist_ok=True); path = 'results/reward_ablation_nyu.csv'
    done = set(); rows = []
    if os.path.exists(path):
        for r in csv.DictReader(open(path)): done.add(r['config']); rows.append(r)
    if 'Full (C1+C3+C5)' not in done:
        m = evaluate(load_dndqn_checkpoint(FULL_CKPT)[0], pool)
        rows.append({'config': 'Full (C1+C3+C5)', 'HO': f"{m['HO']:.2f}", 'Power': f"{m['Power']:.2f}", 'Spec': f"{m['Spec']:.2f}", 'RLF': f"{m['RLF']:.2f}", 'Tput': f"{m['Tput']:.1f}"})
        _write(path, rows); print(f"[done] Full: RLF {m['RLF']:.1f}% Spec {m['Spec']:.1f}%", flush=True)
    for dis, label in [('c1', 'no C1 (power gate off)'), ('c3', 'no C3 (RLF penalty off)'), ('c5', 'no C5 (RB symmetry off)')]:
        if label in done: print(f"[skip] {label}", flush=True); continue
        ck = f"rew_no{dis}_nyu.pt"; t0 = time.time()
        print(f"[train] {label} -> {ck} ({STEPS:,} steps warm-start)", flush=True)
        finetune(dis, ck, slow, med); m = evaluate(load_dndqn_checkpoint(ck)[0], pool)
        rows.append({'config': label, 'HO': f"{m['HO']:.2f}", 'Power': f"{m['Power']:.2f}", 'Spec': f"{m['Spec']:.2f}", 'RLF': f"{m['RLF']:.2f}", 'Tput': f"{m['Tput']:.1f}"})
        _write(path, rows); print(f"[done] {label}: RLF {m['RLF']:.1f}% Spec {m['Spec']:.1f}% Pwr {m['Power']:.1f}% ({(time.time()-t0)/60:.0f}m)", flush=True)
    print("DONE reward-component ablation on NYU", flush=True)

if __name__ == "__main__":
    main()
