#!/usr/bin/env python3
"""
train_phase1.py  —  Phase 1 Dn-DQN Training (HO-only)
=============================================================
Phase 1 of two-phase curriculum transfer learning.
Trains the Dn-DQN architecture on the handover-only objective.
Saved weights serve as initialisation for Phase 2 joint optimisation.

WHY PHASE 1 MUST USE Dn-DQN ARCHITECTURE:
  Phase 2 loads Phase 1 weights into a DnDQN.
  If Phase 1 uses a standard QNetwork, weight transfer fails because the
  architecture (shared → value/advantage heads) is different.
  Both phases must use DnDQN for curriculum transfer to work.

PHASE 1 CONFIGURATION:
  W_MOB=1.0, W_PWR=0.0, W_SPEC=0.0
  The agent learns ONLY when and where to handover.
  Power and spectrum are not optimised — this is intentional.
  The dueling V(s) stream learns a strong mobility value function.
  The advantage stream learns which action best improves RSRP.

N-STEP RETURNS:
  n=3 with episode boundary handling (from dndqn.NStepPERBuffer).
  Propagates delayed handover consequences 3 steps back.

SAVES: dqn_phase1_dndqn.pt

RUN:
    python -u train_phase1.py
    (~25 min on CPU)
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
    HANDOVER, HOLD, N_ACTIONS,
    RLF_THRESH_DBM, PT_MIN_DBM, PT_MAX_DBM, RB_MIN, RB_MAX,
    HO_PROACTIVE_THRESH, A3_THRESH_DBM,
    E2_DELAY_STEPS, MAX_STEPS,
    VEL_MIN, VEL_MAX,
)
from dndqn import (
    DnDQN, NStepPERBuffer,
    build_dndqn_networks, polyak_update,
    epsilon_schedule, beta_schedule,
    OBS_DIM, N_ACTIONS as N_ACT, NET_ARCH,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

CSV_PATH        = "6G_Handover_Training_Data_v3.csv"
MODEL_SAVE_PATH = "dqn_phase1_dndqn.pt"

# Phase 1 weights — HO only
W_MOB_P1  = 1.0
W_PWR_P1  = 0.0
W_SPEC_P1 = 0.0

TOTAL_TIMESTEPS  = 800_000
LEARNING_RATE    = 3e-4
GAMMA            = 0.99
BUFFER_SIZE      = 500_000
BATCH_SIZE       = 256
TARGET_UPDATE_FREQ = 500
TAU              = 0.01
TRAIN_FREQ       = 4
LEARNING_STARTS  = 10_000
LOG_INTERVAL     = 5_000

EPS_START        = 1.0    # full exploration — cold start for Phase 1
EPS_END          = 0.05
EPS_FRACTION     = 0.30

NSTEP            = 3      # n-step returns
PER_ALPHA        = 0.6
PER_BETA_START   = 0.4
PER_BETA_END     = 1.0

VEL_SLOW_MAX     = 25.0
VEL_MED_MAX      = 40.0
STAGE_SLOW_END   = 0.08
HO_MIN_LEAD      = 25

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
#  PHASE 1 TRAINING ENVIRONMENT
#  W_PWR=0, W_SPEC=0 — only mobility reward active
# ═══════════════════════════════════════════════════════════════════════════════

class Phase1Env(SubTHzEnv6G):
    """Phase 1 env — HO-only reward, no power/spectrum optimisation."""
    def __init__(self, vehicle_trajectory):
        super().__init__(
            vehicle_trajectory=vehicle_trajectory,
            w_mob=W_MOB_P1, w_pwr=W_PWR_P1, w_spec=W_SPEC_P1,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 68)
    print("  Phase 1 Dn-DQN Training — HO Only")
    print("  Architecture: Dueling N-step DQN")
    print(f"  N-step={NSTEP} | PER-α={PER_ALPHA}")
    print("=" * 68)

    print(f"\n[Data] Loading {CSV_PATH} ...")
    slow, med, fast, skip = load_trajectories(CSV_PATH)
    print(f"[Data] Slow: {len(slow):,} | Medium: {len(med):,} | "
          f"Fast: {len(fast):,} | Skipped: {skip}")

    print(f"\n[Model] Building DnDQN [128, 128, 64] ...")
    online_net, target_net = build_dndqn_networks(
        obs_dim=OBS_DIM, n_actions=N_ACT,
        net_arch=NET_ARCH, device=str(DEVICE)
    )

    print(f"[Config] W_MOB={W_MOB_P1} | W_PWR={W_PWR_P1} | W_SPEC={W_SPEC_P1}")
    print(f"[Config] Timesteps: {TOTAL_TIMESTEPS:,} | N-step: {NSTEP}")
    print(f"[Config] EPS: {EPS_START} → {EPS_END} over {int(EPS_FRACTION*100)}% steps")
    print(f"[Config] Save → {MODEL_SAVE_PATH}")

    optimizer = optim.Adam(online_net.parameters(), lr=LEARNING_RATE)
    buffer    = NStepPERBuffer(BUFFER_SIZE, n_steps=NSTEP, gamma=GAMMA,
                                alpha=PER_ALPHA)
    rng       = np.random.default_rng(42)

    def sample_traj(stage):
        pool = slow if stage == "SLOW" else slow + med
        return pool[rng.integers(len(pool))]

    stage       = "SLOW"
    current_env = Phase1Env(sample_traj(stage))
    obs, _      = current_env.reset()
    ep_rewards  = deque(maxlen=20)
    ep_reward   = 0.0
    t_start     = time.time()

    # Training loop
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
                q_vals = online_net(
                    torch.FloatTensor(obs).to(DEVICE)
                )
                action = q_vals.argmax().item()

        next_obs, reward, terminated, truncated, info = current_env.step(action)
        done = terminated or truncated
        ep_reward += reward
        buffer.add(obs, action, reward, next_obs, terminated)
        obs = next_obs

        if done:
            ep_rewards.append(ep_reward)
            ep_reward   = 0.0
            current_env = Phase1Env(sample_traj(stage))
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
                # Double Q-learning: online selects, target evaluates
                best_actions = online_net(nobs_t).argmax(1, keepdim=True)
                next_q       = target_net(nobs_t).gather(1, best_actions).squeeze(1)
                # N-step target (gamma^n already in ret_b via n-step accumulation)
                target_q     = ret_t + (GAMMA ** NSTEP) * next_q * (1 - done_t)

            current_q = online_net(obs_t).gather(1, act_t.unsqueeze(1)).squeeze(1)
            td_errors  = (current_q - target_q).detach().cpu().numpy()
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

    # Save
    torch.save({
        'q_net_state':    online_net.state_dict(),
        'obs_dim':        OBS_DIM,
        'n_actions':      N_ACT,
        'net_arch':       NET_ARCH,
        'algorithm':      'Dn-DQN',
        'phase':          1,
        'w_mob': W_MOB_P1, 'w_pwr': W_PWR_P1, 'w_spec': W_SPEC_P1,
        'n_steps':        NSTEP,
        'per_alpha':      PER_ALPHA,
        'training_mode':  'cold_start_HO_only',
    }, MODEL_SAVE_PATH)
    print(f"[Save] Phase 1 Dn-DQN → {MODEL_SAVE_PATH}")

    # Quick eval
    print("\n[Eval] Running 150 deterministic episodes ...")
    online_net.eval()
    eval_rng = np.random.default_rng(0)
    pool     = slow + med
    ho_list  = []
    for ep in range(150):
        traj   = pool[eval_rng.integers(len(pool))]
        env    = Phase1Env(traj)
        obs_e, _ = env.reset()
        done_e   = False
        ep_good_ho = False
        while not done_e:
            with torch.no_grad():
                a = online_net(torch.FloatTensor(obs_e).to(DEVICE)).argmax().item()
            obs_e, _, terminated, truncated, info = env.step(a)
            done_e = terminated or truncated
            if info.get('handover_done'):
                if not info.get('too_early') and not info.get('too_late'):
                    ep_good_ho = True
        ho_list.append(float(ep_good_ho))
    ho_pct = np.mean(ho_list) * 100
    print(f"[Eval] HO Success: {ho_pct:.1f}% (target ≥85%)")
    if ho_pct >= 85.0:
        print(f"[Eval] PASS — Proceed to Phase 2.")
    else:
        print(f"[Eval] WARN — HO below 85%. Consider retraining or increasing steps.")

    print(f"\n[Next] Run: python -u train_phase2.py")


if __name__ == "__main__":
    main()
