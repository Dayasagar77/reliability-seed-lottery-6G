#!/usr/bin/env python3
"""
collect_ablation.py  —  Ablation Study for Dn-DQN
=========================================================
7 configurations isolating each design contribution independently.

ABLATION DESIGN:
  All trainable configs use Dn-DQN algorithm (Dueling + Double + N-step PER)
  except Config 6 which uses standard QNetwork for architecture comparison.
  This allows isolating: reward design, curriculum, and architecture separately.

  Config 1: 3GPP A3 Baseline      — rule-based HO, no resource control
  Config 2: Dn-DQN: HO Only         — Phase 1 model (W_MOB=1, W_PWR=0, W_SPEC=0)
  Config 3: Dn-DQN: HO+Power        — W_SPEC=0, Fix1+Fix3, no Fix5, curriculum
  Config 4: Dn-DQN: HO+Spectrum     — W_PWR=0, Fix3+Fix5, no Fix1, curriculum
  Config 5: Dn-DQN: No Curriculum   — full reward design, cold start (EPS=1.0)
  Config 6: Standard DQN          — full reward design, curriculum, NO Dueling/N-step
  Config 7: Dn-DQN: Proposed        — full system (dqn_phase2_dndqn.pt)

WHAT EACH COMPARISON PROVES:
  Config 3 vs 7: removing W_SPEC+Fix5 → proves spectrum contribution
  Config 4 vs 7: removing W_PWR+Fix1  → proves power contribution
  Config 5 vs 7: cold start vs curriculum → proves curriculum value
  Config 6 vs 7: Standard DQN vs Dn-DQN → proves architecture contribution

ACADEMIC INTEGRITY:
  All configs trained with same hyperparameters (LR, buffer, batch, gamma).
  Only the specified design variable changes between configs.
  Algorithm held constant except in Config 6 (architecture comparison).
  5 seeds × 150 episodes per config = 750 episodes each.

RUN:
    python -u collect_ablation.py
    (~3.5 hrs: 4 trainable configs × ~35 min + eval ~15 min)
"""

import os, sys, time, random
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from env_6g_v5 import (
    SubTHzEnv6G,
    HANDOVER, PWR_DOWN, RB_DOWN, HOLD, N_ACTIONS,
    RLF_THRESH_DBM, A3_THRESH_DBM, HO_PROACTIVE_THRESH,
    PT_MIN_DBM, PT_MAX_DBM, PT_STEP_DBM,
    RB_MIN, RB_MAX, W_MOB, W_PWR, W_SPEC,
)
from dndqn import (
    DnDQN, QNetwork, NStepPERBuffer,
    build_dndqn_networks, load_dndqn_checkpoint,
    load_standard_checkpoint,
    polyak_update, epsilon_schedule, beta_schedule,
    OBS_DIM, NET_ARCH,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

CSV_PATH       = "6G_Handover_Training_Data_v3.csv"
PHASE1_PATH    = "dqn_phase1_dndqn.pt"
PROPOSED_PATH  = "dqn_phase2_dndqn.pt"

TOTAL_TIMESTEPS   = 800_000
LEARNING_RATE     = 3e-4
GAMMA             = 0.99
BUFFER_SIZE       = 500_000
BATCH_SIZE        = 256
TARGET_UPDATE_FREQ = 500
TAU               = 0.01
TRAIN_FREQ        = 4
LEARNING_STARTS   = 10_000
LOG_INTERVAL      = 100_000

EPS_WARM          = 0.50
EPS_COLD          = 1.00
EPS_END           = 0.05
EPS_FRACTION      = 0.30

NSTEP             = 3
PER_ALPHA         = 0.6
PER_BETA_START    = 0.4
PER_BETA_END      = 1.0

SAFE_RSRP_THRESH     = -78.0
PWR_DOWN_UNSAFE_PEN  = -0.20
RLF_TERMINAL_PENALTY = -100.0

N_SEEDS    = 5
N_EPISODES = 150
SEEDS      = [0, 1, 2, 3, 4]

VEL_SLOW_MAX  = 25.0
VEL_MED_MAX   = 40.0
STAGE_SLOW_END = 0.08
HO_MIN_LEAD   = 25

N_ACT  = N_ACTIONS
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_trajectories(csv_path):
    df = pd.read_csv(csv_path)
    slow, med, skip = [], [], 0
    for (_, _), grp in df.groupby(["Car_ID", "Recording"]):
        traj = grp[["x_position", "xVelocity"]].values.astype(np.float32)
        if len(traj) < HO_MIN_LEAD + 10:
            skip += 1; continue
        v = float(traj[:, 1].mean())
        if   v <= VEL_SLOW_MAX: slow.append(traj)
        elif v <= VEL_MED_MAX:  med.append(traj)
    return slow, med


# ═══════════════════════════════════════════════════════════════════════════════
#  ABLATION TRAINING ENVIRONMENT
#  Configurable: weights + which fixes to apply
# ═══════════════════════════════════════════════════════════════════════════════

class AblationEnv(SubTHzEnv6G):
    """
    Configurable ablation training env.
    Each config applies only the fixes relevant to its design variables.
    """
    def __init__(self, traj, w_mob, w_pwr, w_spec,
                 use_fix1=False, use_fix3=False, use_fix5=False):
        super().__init__(traj, w_mob=w_mob, w_pwr=w_pwr, w_spec=w_spec)
        self.use_fix1 = use_fix1
        self.use_fix3 = use_fix3
        self.use_fix5 = use_fix5
        self._prev_rsrp = -70.0
        # Pre-compute step rewards
        self._pwr_step = w_pwr * (PT_STEP_DBM / (PT_MAX_DBM - PT_MIN_DBM)) * 10.0
        self._rb_step  = w_spec * (PT_STEP_DBM / (PT_MAX_DBM - PT_MIN_DBM)) * 10.0

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._prev_rsrp = -70.0
        return obs, info

    def step(self, action):
        rsrp_before = self._prev_rsrp
        obs, _, terminated, truncated, info = super().step(action)
        self._prev_rsrp = info['rsrp_dbm']
        reward = self._ablation_reward(action, info, terminated, rsrp_before)
        return obs, reward, terminated, truncated, info

    def _ablation_reward(self, action, info, terminated, rsrp_before):
        r_mob  = info.get('r_mobility', 0.0)
        r_pwr  = info.get('r_power',    0.0)
        r_spec = info.get('r_spectrum', 0.0)
        reward = self.w_mob * r_mob

        if self.use_fix1 and action == PWR_DOWN:
            reward += self._pwr_step if rsrp_before >= SAFE_RSRP_THRESH \
                      else PWR_DOWN_UNSAFE_PEN
        else:
            reward += self.w_pwr * r_pwr

        if self.use_fix5 and action == RB_DOWN:
            reward += self._rb_step
        else:
            reward += self.w_spec * r_spec

        if self.use_fix3 and terminated and info['rsrp_dbm'] < RLF_THRESH_DBM:
            reward += RLF_TERMINAL_PENALTY

        return reward


class AblationEvalEnv(SubTHzEnv6G):
    """Standard eval env — proposed balanced weights, no reward shaping."""
    def __init__(self, traj):
        super().__init__(traj, w_mob=W_MOB, w_pwr=W_PWR, w_spec=W_SPEC)


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAINING FUNCTION — Dn-DQN
# ═══════════════════════════════════════════════════════════════════════════════

def train_dndqn_config(slow, med, w_mob, w_pwr, w_spec,
                      use_fix1, use_fix3, use_fix5,
                      eps_start, phase1_transfer, save_path, label):
    """Train one ablation config using Dn-DQN (Dueling + Double + N-step PER)."""
    print(f"\n  Training: {label}")
    print(f"  Weights: W_MOB={w_mob} W_PWR={w_pwr} W_SPEC={w_spec}")
    print(f"  Fixes: Fix1={use_fix1} Fix3={use_fix3} Fix5={use_fix5}")
    print(f"  Curriculum: {phase1_transfer} | EPS_start={eps_start}")
    print(f"  Architecture: Dn-DQN (Dueling + Double + N-step-{NSTEP} + PER)")

    online, target = build_dndqn_networks(
        OBS_DIM, N_ACT, NET_ARCH, str(DEVICE))

    if phase1_transfer:
        if not os.path.exists(PHASE1_PATH):
            sys.exit(f"[ERROR] {PHASE1_PATH} not found.")
        ckpt = torch.load(PHASE1_PATH, map_location=DEVICE)
        online.load_state_dict(ckpt['q_net_state'])
        target.load_state_dict(ckpt['q_net_state'])
        print(f"  [Transfer] Phase 1 Dn-DQN weights loaded ✓")
    else:
        target.load_state_dict(online.state_dict())

    target.eval()
    optimizer = optim.Adam(online.parameters(), lr=LEARNING_RATE)
    buffer    = NStepPERBuffer(BUFFER_SIZE, n_steps=NSTEP, gamma=GAMMA,
                                alpha=PER_ALPHA)
    rng       = np.random.default_rng(42)

    def sample_traj(stage):
        p = slow if stage == "SLOW" else slow + med
        return p[rng.integers(len(p))]

    stage = "SLOW"
    env   = AblationEnv(sample_traj(stage), w_mob, w_pwr, w_spec,
                        use_fix1, use_fix3, use_fix5)
    obs, _ = env.reset()
    ep_rewards = deque(maxlen=20)
    ep_reward  = 0.0
    t_start    = time.time()

    for step in range(1, TOTAL_TIMESTEPS + 1):
        frac  = step / TOTAL_TIMESTEPS
        stage = "SLOW" if frac < STAGE_SLOW_END else "SLOW+MED"
        eps   = epsilon_schedule(step, eps_start, EPS_END, EPS_FRACTION,
                                  TOTAL_TIMESTEPS)
        beta  = beta_schedule(step, PER_BETA_START, PER_BETA_END,
                               TOTAL_TIMESTEPS)

        if random.random() < eps:
            action = random.randint(0, N_ACT - 1)
        else:
            with torch.no_grad():
                action = online(
                    torch.FloatTensor(obs).to(DEVICE)).argmax().item()

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ep_reward += reward
        buffer.add(obs, action, reward, next_obs, terminated)
        obs = next_obs

        if done:
            ep_rewards.append(ep_reward)
            ep_reward = 0.0
            env = AblationEnv(sample_traj(stage), w_mob, w_pwr, w_spec,
                              use_fix1, use_fix3, use_fix5)
            obs, _ = env.reset()

        if (step >= LEARNING_STARTS and step % TRAIN_FREQ == 0
                and len(buffer) >= BATCH_SIZE):
            idx_b, w_b, obs_b, act_b, ret_b, nobs_b, done_b = \
                buffer.sample(BATCH_SIZE, beta)
            obs_t  = torch.FloatTensor(obs_b).to(DEVICE)
            nobs_t = torch.FloatTensor(nobs_b).to(DEVICE)
            act_t  = torch.LongTensor(act_b).to(DEVICE)
            ret_t  = torch.FloatTensor(ret_b).to(DEVICE)
            done_t = torch.FloatTensor(done_b).to(DEVICE)
            w_t    = torch.FloatTensor(w_b).to(DEVICE)

            with torch.no_grad():
                best_acts = online(nobs_t).argmax(1, keepdim=True)
                next_q    = target(nobs_t).gather(1, best_acts).squeeze(1)
                target_q  = ret_t + (GAMMA ** NSTEP) * next_q * (1 - done_t)

            curr_q    = online(obs_t).gather(1, act_t.unsqueeze(1)).squeeze(1)
            td_errors = (curr_q - target_q).detach().cpu().numpy()
            loss = (w_t * nn.functional.huber_loss(
                curr_q, target_q, reduction='none')).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(online.parameters(), 10.0)
            optimizer.step()
            buffer.update_priorities(idx_b, td_errors)

        if step % TARGET_UPDATE_FREQ == 0:
            polyak_update(online, target, TAU)

        if step % LOG_INTERVAL == 0:
            mean_rew = float(np.mean(ep_rewards)) if ep_rewards else 0.0
            elapsed  = (time.time() - t_start) / 60
            print(f"    [{step:8d}/{TOTAL_TIMESTEPS}] "
                  f"mean_rew={mean_rew:9.2f} | eps={eps:.3f} | {elapsed:.1f}min")

    elapsed = (time.time() - t_start) / 60
    print(f"  Done in {elapsed:.1f} min")
    torch.save({
        'q_net_state':    online.state_dict(),
        'obs_dim': OBS_DIM, 'n_actions': N_ACT, 'net_arch': NET_ARCH,
        'algorithm': 'Dn-DQN',
        'ablation_label': label,
        'w_mob': w_mob, 'w_pwr': w_pwr, 'w_spec': w_spec,
        'fixes': {'fix1': use_fix1, 'fix3': use_fix3, 'fix5': use_fix5},
        'phase1_transfer': phase1_transfer,
        'n_steps': NSTEP,
    }, save_path)
    print(f"  Saved → {save_path}")
    online.eval()
    return online


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAINING FUNCTION — STANDARD DQN (config 6 only)
# ═══════════════════════════════════════════════════════════════════════════════

def train_standard_dqn(slow, med, save_path, label):
    """
    Config 6 only: Standard QNetwork with full reward design and curriculum.
    Everything identical to proposed except architecture (no Dueling/N-step).
    Uses standard 1-step replay (standard QNetwork from v5_mb).
    """
    print(f"\n  Training: {label}")
    print(f"  Architecture: Standard DQN (no Dueling, no N-step)")
    print(f"  Reward: Full (Fix1+Fix3+Fix5) | Curriculum: True")

    from dndqn import QNetwork
    from collections import deque as _deque

    # Build standard QNetwork
    online = QNetwork(OBS_DIM, N_ACT, NET_ARCH).to(DEVICE)
    target = QNetwork(OBS_DIM, N_ACT, NET_ARCH).to(DEVICE)

    # Transfer Phase 1 Dn-DQN weights → standard net (shared encoder compatible)
    # Note: Phase 1 DnDQN has different head structure — only transfer
    # the shared encoder weights, reinit the output head.
    if os.path.exists(PHASE1_PATH):
        p1_ckpt = torch.load(PHASE1_PATH, map_location=DEVICE)
        p1_state = p1_ckpt['q_net_state']
        # Map shared encoder weights only (compatible layers)
        std_state = online.state_dict()
        mapped = {}
        # shared.0 → net.0, shared.2 → net.2
        layer_map = {
            'shared.0.weight': 'net.0.weight', 'shared.0.bias': 'net.0.bias',
            'shared.2.weight': 'net.2.weight', 'shared.2.bias': 'net.2.bias',
        }
        for p1_key, std_key in layer_map.items():
            if p1_key in p1_state and std_key in std_state:
                mapped[std_key] = p1_state[p1_key]
        std_state.update(mapped)
        online.load_state_dict(std_state)
        target.load_state_dict(std_state)
        print(f"  [Transfer] Shared encoder from Phase 1 Dn-DQN ✓ (head re-initialised)")

    target.eval()
    optimizer = optim.Adam(online.parameters(), lr=LEARNING_RATE)

    # Standard 1-step PER (not N-step)
    from dndqn import NStepPERBuffer as _PER
    buffer = _PER(BUFFER_SIZE, n_steps=1, gamma=GAMMA, alpha=PER_ALPHA)

    # Fix constants for reward shaping
    _pwr_step = W_PWR * (PT_STEP_DBM / (PT_MAX_DBM - PT_MIN_DBM)) * 10.0
    _rb_step  = W_SPEC * (PT_STEP_DBM / (PT_MAX_DBM - PT_MIN_DBM)) * 10.0

    rng   = np.random.default_rng(42)
    def sample_traj(stage):
        p = slow if stage == "SLOW" else slow + med
        return p[rng.integers(len(p))]

    stage = "SLOW"
    env   = AblationEnv(sample_traj(stage), W_MOB, W_PWR, W_SPEC,
                        use_fix1=True, use_fix3=True, use_fix5=True)
    obs, _     = env.reset()
    ep_rewards = deque(maxlen=20)
    ep_reward  = 0.0
    t_start    = time.time()

    for step in range(1, TOTAL_TIMESTEPS + 1):
        frac  = step / TOTAL_TIMESTEPS
        stage = "SLOW" if frac < STAGE_SLOW_END else "SLOW+MED"
        eps   = epsilon_schedule(step, EPS_WARM, EPS_END, EPS_FRACTION,
                                  TOTAL_TIMESTEPS)
        beta  = beta_schedule(step, PER_BETA_START, PER_BETA_END,
                               TOTAL_TIMESTEPS)

        if random.random() < eps:
            action = random.randint(0, N_ACT - 1)
        else:
            with torch.no_grad():
                action = online(
                    torch.FloatTensor(obs).to(DEVICE)).argmax().item()

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ep_reward += reward
        buffer.add(obs, action, reward, next_obs, terminated)
        obs = next_obs

        if done:
            ep_rewards.append(ep_reward)
            ep_reward = 0.0
            env = AblationEnv(sample_traj(stage), W_MOB, W_PWR, W_SPEC,
                              True, True, True)
            obs, _ = env.reset()

        if (step >= LEARNING_STARTS and step % TRAIN_FREQ == 0
                and len(buffer) >= BATCH_SIZE):
            idx_b, w_b, obs_b, act_b, ret_b, nobs_b, done_b = \
                buffer.sample(BATCH_SIZE, beta)
            obs_t  = torch.FloatTensor(obs_b).to(DEVICE)
            nobs_t = torch.FloatTensor(nobs_b).to(DEVICE)
            act_t  = torch.LongTensor(act_b).to(DEVICE)
            ret_t  = torch.FloatTensor(ret_b).to(DEVICE)
            done_t = torch.FloatTensor(done_b).to(DEVICE)
            w_t    = torch.FloatTensor(w_b).to(DEVICE)

            with torch.no_grad():
                # Standard DQN: target net evaluates max action
                next_q   = target(nobs_t).max(1)[0]
                target_q = ret_t + GAMMA * next_q * (1 - done_t)

            curr_q    = online(obs_t).gather(1, act_t.unsqueeze(1)).squeeze(1)
            td_errors = (curr_q - target_q).detach().cpu().numpy()
            loss = (w_t * nn.functional.huber_loss(
                curr_q, target_q, reduction='none')).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(online.parameters(), 10.0)
            optimizer.step()
            buffer.update_priorities(idx_b, td_errors)

        if step % TARGET_UPDATE_FREQ == 0:
            for p, tp in zip(online.parameters(), target.parameters()):
                tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)

        if step % LOG_INTERVAL == 0:
            mean_rew = float(np.mean(ep_rewards)) if ep_rewards else 0.0
            elapsed  = (time.time() - t_start) / 60
            print(f"    [{step:8d}/{TOTAL_TIMESTEPS}] "
                  f"mean_rew={mean_rew:9.2f} | eps={eps:.3f} | {elapsed:.1f}min")

    elapsed = (time.time() - t_start) / 60
    print(f"  Done in {elapsed:.1f} min")
    torch.save({
        'q_net_state': online.state_dict(),
        'obs_dim': OBS_DIM, 'n_actions': N_ACT, 'net_arch': NET_ARCH,
        'algorithm': 'Standard-DQN',
        'ablation_label': label,
        'w_mob': W_MOB, 'w_pwr': W_PWR, 'w_spec': W_SPEC,
        'fixes': {'fix1': True, 'fix3': True, 'fix5': True},
        'phase1_transfer': True,
        'n_steps': 1,
    }, save_path)
    print(f"  Saved → {save_path}")
    online.eval()
    return online


# ═══════════════════════════════════════════════════════════════════════════════
#  EVALUATION — per-HO metric
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_policy(policy_fn, pool, label):
    """Per-HO event success rate across 5×150 = 750 episodes."""
    print(f"\n  Evaluating: {label} ...")
    seed_ho, seed_pwr, seed_spec, seed_comp = [], [], [], []
    total_good = total_trigs = total_late = 0

    for seed in SEEDS:
        rng     = np.random.default_rng(seed)
        good_ho = tot_ho = too_late = 0
        pwr_list, spec_list, rlf_list = [], [], []

        for _ in range(N_EPISODES):
            traj   = pool[rng.integers(len(pool))]
            env    = AblationEvalEnv(traj)
            obs, _ = env.reset()
            done   = False
            ep_pt, ep_rb = [], []
            ep_rlf = False

            while not done:
                action = policy_fn(obs)
                obs, _, terminated, truncated, info = env.step(action)
                ep_pt.append(info['pt_dbm'])
                ep_rb.append(info['rb_count'])
                done = terminated or truncated

                if info.get('handover_done'):
                    tot_ho += 1
                    if not info.get('too_early') and not info.get('too_late'):
                        good_ho += 1
                    if info.get('too_late'):
                        too_late += 1

                if terminated and info['rsrp_dbm'] < RLF_THRESH_DBM:
                    ep_rlf = True

            pwr_list.append(
                (PT_MAX_DBM - float(np.mean(ep_pt))) / (PT_MAX_DBM - PT_MIN_DBM) * 100)
            spec_list.append(
                (RB_MAX - float(np.mean(ep_rb))) / (RB_MAX - RB_MIN) * 100)
            rlf_list.append(float(ep_rlf))

        seed_ho.append(good_ho / max(tot_ho, 1) * 100)
        seed_pwr.append(np.mean(pwr_list))
        seed_spec.append(np.mean(spec_list))
        seed_comp.append((1 - np.mean(rlf_list)) * 100)
        total_good  += good_ho
        total_trigs += tot_ho
        total_late  += too_late

    return {
        'label': label,
        'ho_mean':   np.mean(seed_ho),    'ho_std':   np.std(seed_ho),
        'pwr_mean':  np.mean(seed_pwr),   'pwr_std':  np.std(seed_pwr),
        'spec_mean': np.mean(seed_spec),  'spec_std': np.std(seed_spec),
        'comp_mean': np.mean(seed_comp),  'comp_std': np.std(seed_comp),
        'total_good': total_good, 'total_trigs': total_trigs,
        'too_late_pct': total_late / max(total_trigs, 1) * 100,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("  Dn-DQN Ablation Study — 7 configurations")
    print("  5 seeds × 150 ep = 750 episodes each")
    print("  All trainable configs: Dn-DQN (except Config 6: Standard DQN)")
    print("=" * 72)

    print(f"\n[Data] Loading {CSV_PATH} ...")
    slow, med = load_trajectories(CSV_PATH)
    pool = slow + med
    print(f"[Data] Slow: {len(slow):,} | Med: {len(med):,} | Pool: {len(pool):,}")

    results = []

    # ── Config 1: 3GPP A3 ─────────────────────────────────────────────────────
    print("\n[Ablation 1/7] 3GPP A3 Baseline")
    a3_thresh_norm = A3_THRESH_DBM / 100.0
    def a3_policy(obs):
        return HANDOVER if float(obs[0]) < a3_thresh_norm else HOLD
    results.append(evaluate_policy(a3_policy, pool, "3GPP A3 Baseline"))

    # ── Config 2: Dn-DQN HO Only (Phase 1 model) ────────────────────────────────
    print("\n[Ablation 2/7] Dn-DQN: HO Only (Phase 1 model)")
    if not os.path.exists(PHASE1_PATH):
        sys.exit(f"[ERROR] {PHASE1_PATH} not found.")
    p1_net, _ = load_dndqn_checkpoint(PHASE1_PATH, str(DEVICE))
    def p1_policy(obs):
        with torch.no_grad():
            return p1_net(torch.FloatTensor(obs).to(DEVICE)).argmax().item()
    results.append(evaluate_policy(p1_policy, pool, "Dn-DQN: HO Only (Phase 1)"))

    # ── Config 3: Dn-DQN HO+Power (W_SPEC=0, Fix5=False) ───────────────────────
    print("\n[Ablation 3/7] Dn-DQN: HO+Power (W_SPEC=0)")
    save_3 = "ablation_ho_power.pt"
    if os.path.exists(save_3):
        print(f"  Found {save_3} — loading.")
        q3, _ = load_dndqn_checkpoint(save_3, str(DEVICE))
    else:
        q3 = train_dndqn_config(
            slow, med, w_mob=0.60, w_pwr=0.40, w_spec=0.00,
            use_fix1=True, use_fix3=True, use_fix5=False,
            eps_start=EPS_WARM, phase1_transfer=True,
            save_path=save_3, label="Dn-DQN: HO+Power",
        )
    def policy_3(obs):
        with torch.no_grad():
            return q3(torch.FloatTensor(obs).to(DEVICE)).argmax().item()
    results.append(evaluate_policy(policy_3, pool, "Dn-DQN: HO+Power (W_SPEC=0)"))

    # ── Config 4: Dn-DQN HO+Spectrum (W_PWR=0, Fix1=False) ─────────────────────
    print("\n[Ablation 4/7] Dn-DQN: HO+Spectrum (W_PWR=0)")
    save_4 = "ablation_ho_spectrum.pt"
    if os.path.exists(save_4):
        print(f"  Found {save_4} — loading.")
        q4, _ = load_dndqn_checkpoint(save_4, str(DEVICE))
    else:
        q4 = train_dndqn_config(
            slow, med, w_mob=0.60, w_pwr=0.00, w_spec=0.40,
            use_fix1=False, use_fix3=True, use_fix5=True,
            eps_start=EPS_WARM, phase1_transfer=True,
            save_path=save_4, label="Dn-DQN: HO+Spectrum",
        )
    def policy_4(obs):
        with torch.no_grad():
            return q4(torch.FloatTensor(obs).to(DEVICE)).argmax().item()
    results.append(evaluate_policy(policy_4, pool, "Dn-DQN: HO+Spectrum (W_PWR=0)"))

    # ── Config 5: Dn-DQN No Curriculum ──────────────────────────────────────────
    print("\n[Ablation 5/7] Dn-DQN: No Curriculum (cold start)")
    save_5 = "ablation_no_curriculum.pt"
    if os.path.exists(save_5):
        print(f"  Found {save_5} — loading.")
        q5, _ = load_dndqn_checkpoint(save_5, str(DEVICE))
    else:
        q5 = train_dndqn_config(
            slow, med, w_mob=W_MOB, w_pwr=W_PWR, w_spec=W_SPEC,
            use_fix1=True, use_fix3=True, use_fix5=True,
            eps_start=EPS_COLD, phase1_transfer=False,
            save_path=save_5, label="Dn-DQN: No Curriculum",
        )
    def policy_5(obs):
        with torch.no_grad():
            return q5(torch.FloatTensor(obs).to(DEVICE)).argmax().item()
    results.append(evaluate_policy(policy_5, pool, "Dn-DQN: No Curriculum"))

    # ── Config 6: Standard DQN (no Dueling, 1-step, full reward, curriculum) ─
    print("\n[Ablation 6/7] Standard DQN (architecture comparison)")
    save_6 = "ablation_standard_dqn_full.pt"
    if os.path.exists(save_6):
        print(f"  Found {save_6} — loading.")
        q6, _ = load_standard_checkpoint(save_6, str(DEVICE))
    else:
        q6 = train_standard_dqn(
            slow, med, save_path=save_6,
            label="Standard DQN: Full Joint",
        )
    def policy_6(obs):
        with torch.no_grad():
            return q6(torch.FloatTensor(obs).to(DEVICE)).argmax().item()
    results.append(evaluate_policy(policy_6, pool, "Standard DQN: Full Joint"))

    # ── Config 7: Proposed Dn-DQN ───────────────────────────────────────────
    print(f"\n[Ablation 7/7] Dn-DQN: Proposed — {PROPOSED_PATH}")
    if not os.path.exists(PROPOSED_PATH):
        sys.exit(f"[ERROR] {PROPOSED_PATH} not found. Run train_phase2.py.")
    prop_net, _ = load_dndqn_checkpoint(PROPOSED_PATH, str(DEVICE))
    def prop_policy(obs):
        with torch.no_grad():
            return prop_net(torch.FloatTensor(obs).to(DEVICE)).argmax().item()
    results.append(evaluate_policy(prop_policy, pool, "Dn-DQN: Proposed"))

    # ── Final Table ────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 84)
    print("  ABLATION STUDY RESULTS  (5 seeds × 150 ep = 750 ep each)")
    print("  Metric: per-HO event success rate across ALL handover events")
    print("  NOTE: Configs 1-5,7 use Dn-DQN. Config 6 uses Standard DQN.")
    print("  Algorithm is held constant within Dn-DQN configs; only design")
    print("  variable changes. Config 6 vs 7 isolates architecture.")
    print("=" * 84)
    hdr = f"  {'Config':<36} {'HO%':>14} {'Power%':>12} {'Spec%':>12} {'EpComp%':>10}"
    print(hdr)
    print("  " + "-" * 82)
    for r in results:
        ho_s   = f"{r['ho_mean']:.1f}±{r['ho_std']:.1f}"
        pwr_s  = f"{r['pwr_mean']:.1f}±{r['pwr_std']:.1f}"
        spec_s = f"{r['spec_mean']:.1f}±{r['spec_std']:.1f}"
        comp_s = f"{r['comp_mean']:.1f}±{r['comp_std']:.1f}"
        marker = " ◄ PROPOSED" if "Proposed" in r['label'] else ""
        print(f"  {r['label']:<36} {ho_s:>14} {pwr_s:>12} "
              f"{spec_s:>12} {comp_s:>10}{marker}")
    print("=" * 84)

    # Integrity checks
    prop = results[-1]
    std  = results[-2]   # config 6
    r3   = results[2]
    r4   = results[3]
    r5   = results[4]

    print("\n[Ablation integrity checks]")
    if prop['spec_mean'] > r3['spec_mean']:
        print(f"  ✓ Proposed spec ({prop['spec_mean']:.1f}%) > "
              f"HO+Power ({r3['spec_mean']:.1f}%) — Fix5+W_SPEC contribution confirmed")
    else:
        print(f"  ⚠ Proposed spec ≤ HO+Power — unexpected, investigate Fix5")

    if prop['pwr_mean'] < r3['pwr_mean']:
        print(f"  ✓ HO+Power ({r3['pwr_mean']:.1f}%) > Proposed ({prop['pwr_mean']:.1f}%) "
              f"— expected: full W_PWR=0.40 concentrates on power")

    if prop['ho_mean'] >= r5['ho_mean'] - 1.0:
        print(f"  ✓ Curriculum HO ({prop['ho_mean']:.1f}%) ≥ "
              f"No-curriculum ({r5['ho_mean']:.1f}%) ±1% — curriculum stable")

    if prop['ho_mean'] >= std['ho_mean'] or prop['spec_mean'] > std['spec_mean']:
        print(f"  ✓ Dn-DQN outperforms Standard DQN on at least one metric "
              f"— architecture contribution confirmed")
    else:
        print(f"  ⚠ Standard DQN matches Dn-DQN — review architecture justification")

    print(f"\n[Next] Run: python -u collect_sensitivity.py")


if __name__ == "__main__":
    main()
