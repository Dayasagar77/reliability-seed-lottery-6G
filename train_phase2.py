#!/usr/bin/env python3
"""
train_phase2.py  —  Phase 2 Dn-DQN Training (Joint HO+Power+Spectrum)
=================================================================================
Phase 2 of two-phase curriculum transfer learning.
Loads Phase 1 Dn-DQN weights and fine-tunes on the full joint objective.

CURRICULUM TRANSFER:
  Phase 1 trained a strong HO policy using DnDQN.
  Phase 2 loads those weights → the agent already knows WHEN to handover.
  Phase 2 then extends to jointly optimise HO + Power + Spectrum.
  Starting from a converged HO policy reduces catastrophic forgetting.
  EPS_START=0.5 (not 1.0) — agent retains Phase 1 knowledge.

REWARD DESIGN (physics-grounded, see dndqn.py for justification):
  W_MOB=0.60, W_PWR=0.20, W_SPEC=0.20 — equal resource weights
  Fix1: conditional PWR_DOWN bonus (prevents unsafe power reduction)
  Fix3: RLF terminal penalty (-100)
  Fix5: symmetric RB_DOWN bonus (corrects discrete action asymmetry)
  All three are documented contributions of this work.

MULTI-BOUNDARY TRAINING:
  RANDOM_START_FRAC=1.00 — agent starts at random valid positions.
  At 8.3 Hz sampling with MAX_STEPS=800, each episode spans ~1928m
  and crosses ~18.6 cell boundaries. Agent learns to chain handovers.
  This eliminates the training/eval distribution mismatch.

Dn-DQN COMPONENTS ACTIVE IN PHASE 2:
  ✓ Dueling V(s) + A(s,a) architecture
  ✓ Double Q-learning (online selects, target evaluates)
  ✓ N-step returns (n=3) with episode boundary handling
  ✓ Prioritized Experience Replay (PER-α=0.6)

SAVES: dqn_phase2_dndqn.pt

RUN:
    python -u train_phase2.py
    (~40 min on CPU)
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
    RLF_THRESH_DBM, PT_MIN_DBM, PT_MAX_DBM, PT_STEP_DBM,
    RB_MIN, RB_MAX, RB_STEP,
    HO_PROACTIVE_THRESH,
    E2_DELAY_STEPS, MAX_STEPS,
    VEL_MIN, VEL_MAX,
)
from dndqn import (
    DnDQN, NStepPERBuffer,
    build_dndqn_networks, load_dndqn_checkpoint,
    polyak_update, epsilon_schedule, beta_schedule,
    OBS_DIM, NET_ARCH,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

CSV_PATH         = "6G_Handover_Training_Data_v3.csv"
PHASE1_PATH      = "dqn_phase1_dndqn.pt"
MODEL_SAVE_PATH  = "dqn_phase2_dndqn.pt"

# Phase 2 balanced weights — physics-grounded for Sub-THz
W_MOB  = 0.60
W_PWR  = 0.20
W_SPEC = 0.20

TOTAL_TIMESTEPS   = 800_000
LEARNING_RATE     = 3e-4
GAMMA             = 0.99
BUFFER_SIZE       = 500_000
BATCH_SIZE        = 256
TARGET_UPDATE_FREQ = 500
TAU               = 0.01
TRAIN_FREQ        = 4
LEARNING_STARTS   = 10_000
LOG_INTERVAL      = 5_000

# Curriculum: warm start (Phase 1 already converged HO policy)
EPS_START         = 0.50   # reduced from 1.0 — retains Phase 1 knowledge
EPS_END           = 0.05
EPS_FRACTION      = 0.30

NSTEP             = 3
PER_ALPHA         = 0.6
PER_BETA_START    = 0.4
PER_BETA_END      = 1.0

# Fix 1 — conditional PWR_DOWN
SAFE_RSRP_THRESH      = -78.0
PWR_DOWN_UNSAFE_PEN   = -0.20
_PWR_STEP_REW = W_PWR * (PT_STEP_DBM / (PT_MAX_DBM - PT_MIN_DBM)) * 10.0

# Fix 3 — RLF terminal penalty
RLF_TERMINAL_PENALTY = -100.0

# Fix 5 — symmetric RB step reward
_RB_STEP_REW  = W_SPEC * (PT_STEP_DBM / (PT_MAX_DBM - PT_MIN_DBM)) * 10.0

VEL_SLOW_MAX     = 25.0
VEL_MED_MAX      = 40.0
STAGE_SLOW_END   = 0.08
HO_MIN_LEAD      = 25

N_ACT  = N_ACTIONS
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_trajectories(csv_path):
    df = pd.read_csv(csv_path)
    slow, med, fast, skip = [], [], [], 0
    for (_, _), grp in df.groupby(["Car_ID", "Recording"]):
        traj = grp[["x_position", "xVelocity"]].values.astype(np.float32)
        if len(traj) < HO_MIN_LEAD + 10:
            skip += 1; continue
        v = float(traj[:, 1].mean())
        if   v <= VEL_SLOW_MAX: slow.append(traj)
        elif v <= VEL_MED_MAX:  med.append(traj)
        else:                   fast.append(traj)
    return slow, med, fast, skip


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 2 TRAINING ENVIRONMENT
#  Applies Fix1 + Fix3 + Fix5 reward shaping on top of base env_6g_v5 reward
# ═══════════════════════════════════════════════════════════════════════════════

class Phase2Env(SubTHzEnv6G):
    """
    Phase 2 training env with Fix1 + Fix3 + Fix5 reward shaping.
    Multi-boundary: random valid-position starts (no forced boundary start).
    """
    def __init__(self, vehicle_trajectory):
        super().__init__(
            vehicle_trajectory=vehicle_trajectory,
            w_mob=W_MOB, w_pwr=W_PWR, w_spec=W_SPEC,
        )
        self._prev_rsrp = -70.0

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._prev_rsrp = -70.0
        return obs, info

    def step(self, action):
        rsrp_before = self._prev_rsrp
        obs, _, terminated, truncated, info = super().step(action)
        self._prev_rsrp = info['rsrp_dbm']
        reward = self._shaped_reward(action, info, terminated, rsrp_before)
        return obs, reward, terminated, truncated, info

    def _shaped_reward(self, action, info, terminated, rsrp_before_dbm):
        """
        Fix1 + Fix3 + Fix5 reward shaping — proposed system contributions.

        Fix1 (conditional PWR_DOWN): prevents unsafe power reduction.
          If RSRP < SAFE_RSRP_THRESH (-78 dBm), PWR_DOWN is penalised.
          Otherwise, provides _PWR_STEP_REW as incremental incentive.

        Fix3 (RLF terminal penalty): -100 on radio link failure.
          Provides strong gradient signal away from link failures.

        Fix5 (symmetric RB step reward): mirrors Fix1 for spectrum.
          _RB_STEP_REW = _PWR_STEP_REW — equal per-step incentive for
          RB_DOWN and PWR_DOWN, correcting the 3× structural asymmetry
          in the raw reward caused by differing action step sizes.
        """
        r_mob  = info.get('r_mobility', 0.0)
        r_pwr  = info.get('r_power',    0.0)
        r_spec = info.get('r_spectrum', 0.0)

        reward = W_MOB * r_mob

        # Fix1: conditional PWR_DOWN
        if action == PWR_DOWN:
            reward += _PWR_STEP_REW if rsrp_before_dbm >= SAFE_RSRP_THRESH \
                      else PWR_DOWN_UNSAFE_PEN
        else:
            reward += W_PWR * r_pwr

        # Fix5: symmetric RB step reward
        if action == RB_DOWN:
            reward += _RB_STEP_REW
        else:
            reward += W_SPEC * r_spec

        # Fix3: RLF terminal penalty
        if terminated and info['rsrp_dbm'] < RLF_THRESH_DBM:
            reward += RLF_TERMINAL_PENALTY

        return reward


# ═══════════════════════════════════════════════════════════════════════════════
#  EVALUATION (quick — used at end of training)
# ═══════════════════════════════════════════════════════════════════════════════

def quick_eval(net, pool, n_ep=150):
    """Quick 150-episode evaluation — per-episode binary HO metric."""
    net.eval()
    rng = np.random.default_rng(0)
    ho_list, pwr_list, spec_list = [], [], []
    n_rlf = n_too_late = n_ho_trig = 0

    for _ in range(n_ep):
        traj   = pool[rng.integers(len(pool))]
        env    = Phase2Env(traj)
        obs, _ = env.reset()
        done   = False
        ep_good_ho = False
        ep_pt, ep_rb = [], []

        while not done:
            with torch.no_grad():
                action = net(torch.FloatTensor(obs).to(DEVICE)).argmax().item()
            obs, _, terminated, truncated, info = env.step(action)
            ep_pt.append(info['pt_dbm'])
            ep_rb.append(info['rb_count'])
            done = terminated or truncated

            if info.get('handover_done'):
                n_ho_trig += 1
                if not info.get('too_early') and not info.get('too_late'):
                    ep_good_ho = True
                if info.get('too_late'):
                    n_too_late += 1

            if terminated and info['rsrp_dbm'] < RLF_THRESH_DBM:
                n_rlf += 1

        ho_list.append(float(ep_good_ho))
        pwr_list.append(
            (PT_MAX_DBM - float(np.mean(ep_pt))) / (PT_MAX_DBM - PT_MIN_DBM) * 100)
        spec_list.append(
            (RB_MAX - float(np.mean(ep_rb))) / (RB_MAX - RB_MIN) * 100)

    return {
        'ho':   np.mean(ho_list) * 100,
        'pwr':  np.mean(pwr_list),
        'spec': np.mean(spec_list),
        'n_ho_trig': n_ho_trig,
        'n_too_late': n_too_late,
        'n_rlf': n_rlf,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 68)
    print("  Phase 2 Dn-DQN Training — Joint HO + Power + Spectrum")
    print("  Curriculum transfer from Phase 1 | Multi-boundary")
    print(f"  N-step={NSTEP} | Fix1+Fix3+Fix5")
    print("=" * 68)

    print(f"\n[Data] Loading {CSV_PATH} ...")
    slow, med, fast, skip = load_trajectories(CSV_PATH)
    pool = slow + med
    print(f"[Data] Slow: {len(slow):,} | Medium: {len(med):,} | "
          f"Fast: {len(fast):,} | Skipped: {skip}")

    print(f"\n[Model] Building DnDQN [128, 128, 64] ...")
    online_net, target_net = build_dndqn_networks(
        obs_dim=OBS_DIM, n_actions=N_ACT,
        net_arch=NET_ARCH, device=str(DEVICE)
    )

    # Curriculum transfer from Phase 1
    if not os.path.exists(PHASE1_PATH):
        sys.exit(f"[ERROR] Phase 1 model not found: {PHASE1_PATH}\n"
                 f"        Run train_phase1.py first.")
    p1_ckpt = torch.load(PHASE1_PATH, map_location=DEVICE)
    online_net.load_state_dict(p1_ckpt['q_net_state'])
    target_net.load_state_dict(p1_ckpt['q_net_state'])
    print(f"[Transfer] Phase 1 Dn-DQN weights loaded ✓")
    print(f"[Transfer] Phase 1 was: W_MOB={p1_ckpt['w_mob']} "
          f"W_PWR={p1_ckpt['w_pwr']} W_SPEC={p1_ckpt['w_spec']}")

    print(f"\n[Config] env_6g_v5: molecular absorption + balanced reward weights")
    print(f"[Config] W_MOB={W_MOB} | W_PWR={W_PWR} | W_SPEC={W_SPEC}")
    print(f"[Config] Timesteps: {TOTAL_TIMESTEPS:,} | N-step: {NSTEP}")
    print(f"[Config] EPS: {EPS_START} (warm) → {EPS_END} over "
          f"{int(EPS_FRACTION*100)}% steps")
    print(f"[Config] Fix1=ConditionalPWR | Fix3=RLF_PEN({RLF_TERMINAL_PENALTY}) "
          f"| Fix5=SymRBReward")
    print(f"[Config] Multi-boundary: random starts, ~18.6 HO events/episode")
    print(f"[Config] Save → {MODEL_SAVE_PATH}")

    optimizer = optim.Adam(online_net.parameters(), lr=LEARNING_RATE)
    buffer    = NStepPERBuffer(BUFFER_SIZE, n_steps=NSTEP, gamma=GAMMA,
                                alpha=PER_ALPHA)
    rng       = np.random.default_rng(42)

    def sample_traj(stage):
        p = slow if stage == "SLOW" else slow + med
        return p[rng.integers(len(p))]

    stage       = "SLOW"
    current_env = Phase2Env(sample_traj(stage))
    obs, _      = current_env.reset()
    ep_rewards  = deque(maxlen=20)
    ep_reward   = 0.0
    t_start     = time.time()

    print(f"\n[Train] Starting ... ({TOTAL_TIMESTEPS:,} steps)")
    print(f"[Train] Curriculum: expect faster convergence than cold-start")

    for step in range(1, TOTAL_TIMESTEPS + 1):
        frac  = step / TOTAL_TIMESTEPS
        stage = "SLOW" if frac < STAGE_SLOW_END else "SLOW+MED"
        eps   = epsilon_schedule(step, EPS_START, EPS_END, EPS_FRACTION,
                                  TOTAL_TIMESTEPS)
        beta  = beta_schedule(step, PER_BETA_START, PER_BETA_END,
                               TOTAL_TIMESTEPS)

        # Action selection
        if random.random() < eps:
            action = random.randint(0, N_ACT - 1)
        else:
            with torch.no_grad():
                action = online_net(
                    torch.FloatTensor(obs).to(DEVICE)
                ).argmax().item()

        next_obs, reward, terminated, truncated, info = current_env.step(action)
        done = terminated or truncated
        ep_reward += reward
        buffer.add(obs, action, reward, next_obs, terminated)
        obs = next_obs

        if done:
            ep_rewards.append(ep_reward)
            ep_reward   = 0.0
            current_env = Phase2Env(sample_traj(stage))
            obs, _      = current_env.reset()

        # Gradient update
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
                # Double Q: online selects action, target evaluates
                best_acts = online_net(nobs_t).argmax(1, keepdim=True)
                next_q    = target_net(nobs_t).gather(1, best_acts).squeeze(1)
                target_q  = ret_t + (GAMMA ** NSTEP) * next_q * (1 - done_t)

            current_q = online_net(obs_t).gather(
                1, act_t.unsqueeze(1)).squeeze(1)
            td_errors = (current_q - target_q).detach().cpu().numpy()
            loss = (w_t * nn.functional.huber_loss(
                current_q, target_q, reduction='none')).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(online_net.parameters(), 10.0)
            optimizer.step()
            buffer.update_priorities(idx_b, td_errors)

        # Soft target update
        if step % TARGET_UPDATE_FREQ == 0:
            polyak_update(online_net, target_net, TAU)

        if step % LOG_INTERVAL == 0:
            mean_rew = float(np.mean(ep_rewards)) if ep_rewards else 0.0
            print(f"[{step:8d}/{TOTAL_TIMESTEPS}] stage={stage:<12} | "
                  f"mean_rew={mean_rew:9.4f} | eps={eps:.3f} | beta={beta:.3f}")

    elapsed = (time.time() - t_start) / 60
    print(f"\n[Train] Finished in {elapsed:.1f} min")

    # Save checkpoint
    torch.save({
        'q_net_state':    online_net.state_dict(),
        'obs_dim':        OBS_DIM,
        'n_actions':      N_ACT,
        'net_arch':       NET_ARCH,
        'algorithm':      'Dn-DQN',
        'phase':          2,
        'w_mob': W_MOB, 'w_pwr': W_PWR, 'w_spec': W_SPEC,
        'n_steps':        NSTEP,
        'per_alpha':      PER_ALPHA,
        'training_mode':  'curriculum_transfer_multi_boundary',
        'fixes':          'Fix1+Fix3+Fix5',
        'weight_config':  'balanced_0.60_0.20_0.20',
    }, MODEL_SAVE_PATH)
    print(f"[Save] Phase 2 Dn-DQN → {MODEL_SAVE_PATH}")

    # Quick eval
    print(f"\n[Eval] Running 150 deterministic episodes ...")
    r = quick_eval(online_net, slow + med)
    print(f"\n{'='*68}")
    print(f"  Phase 2 Dn-DQN — Quick Eval Results")
    print(f"{'='*68}")
    print(f"  HO Success Rate : {r['ho']:.1f}%  (per-episode binary)")
    print(f"  Power Saving    : {r['pwr']:.1f}%")
    print(f"  Spectrum Saving : {r['spec']:.1f}%")
    print(f"  Outcome: HO_trig={r['n_ho_trig']} | "
          f"TooLate={r['n_too_late']} | RLF={r['n_rlf']}")
    print(f"{'='*68}")
    print(f"\n[Next] Run: python -u multi_seed_eval.py")


if __name__ == "__main__":
    main()
