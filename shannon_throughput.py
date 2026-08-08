#!/usr/bin/env python3
"""
shannon_throughput.py  —  Shannon Capacity Comparison
===========================================================
Computes per-UE Shannon capacity for all methods:
  Dn-DQN, Standard DQN, DDQN+PER, PPO, 3GPP A3

MOTIVATION:
  The primary claim for Sub-THz spectrum optimization is not "we save RBs"
  but "we improve system capacity." Shannon capacity translates RB usage
  and SINR into a concrete, universally understood metric.

  Dn-DQN uses fewer RBs (mean ~20 vs 66 max) but at what SINR?
  This script answers: what is the actual per-UE throughput?

FORMULA:
  C = Σ_per_step [ rb_count × RB_BW × log2(1 + 10^(SINR_dB/10)) ]
  RB_BW = 2000 MHz total / 66 RBs = 30.303 MHz per RB
  SINR_dB = obs[4] × 100.0 (from normalised state)

  Per-episode mean throughput (Mbps):
    C_ep = mean_over_steps [ rb_count × 30.303 × log2(1 + SINR_linear) ]

  Per-method mean ± std across 5 seeds × 150 episodes = 750 episodes.

WHY THIS MATTERS:
  If Dn-DQN achieves similar or higher throughput than baselines despite
  fewer RBs, it demonstrates:
  1. Freed RBs can serve additional UEs → system capacity improvement
  2. SINR efficiency is maintained despite power reduction
  3. Sub-THz spectrum optimization is the correct objective

  This is the paper's strongest physical argument.

MODELS EVALUATED:
  - Dn-DQN: dqn_phase2_dndqn.pt
  - Standard DQN: ablation_standard_dqn_full.pt
  - DDQN+PER: dqn_baseline_ddqn_v5_mb_correct.pt (from Updated Journal dir)
  - PPO: baseline_ppo_v5_mb_correct.zip (from Updated Journal dir)
  - 3GPP A3: rule-based (no model file)

  IMPORTANT: DDQN and PPO baselines are in the old directory.
  If not found, the script continues with available models.

RUN:
    python -u shannon_throughput.py
    (~20 min)
"""

import os, sys, time
import numpy as np
import pandas as pd
import torch

from env_6g_v5 import (
    SubTHzEnv6G,
    HANDOVER, HOLD, N_ACTIONS,
    RLF_THRESH_DBM, A3_THRESH_DBM,
    PT_MIN_DBM, PT_MAX_DBM, RB_MIN, RB_MAX,
    W_MOB, W_PWR, W_SPEC,
    VEL_MIN, VEL_MAX,
)
from dndqn import (
    load_dndqn_checkpoint, load_standard_checkpoint,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

CSV_PATH         = "6G_Handover_Training_Data_v3.csv"
DNDQN_PATH        = "dqn_phase2_dndqn.pt"
STD_DQN_PATH     = "ablation_standard_dqn_full.pt"

# Old directory — may or may not be present
OLD_DIR          = r"D:\DAYA PHD\PHD WORK\Updated Journal"
DDQN_PATH        = os.path.join(OLD_DIR, "dqn_baseline_ddqn_v5_mb_correct.pt")
PPO_PATH         = os.path.join(OLD_DIR, "baseline_ppo_v5_mb_correct.zip")

# Sub-THz RB parameters
TOTAL_BW_MHZ     = 2000.0        # total bandwidth at 140 GHz (MHz)
N_RB_MAX         = 66            # maximum RBs (RB_MAX from env)
RB_BW_MHZ        = TOTAL_BW_MHZ / N_RB_MAX   # 30.303 MHz per RB

VEL_SLOW_MAX     = 25.0
VEL_MED_MAX      = 40.0
HO_MIN_LEAD      = 25
N_SEEDS          = 5
N_EPISODES       = 150
SEEDS            = [0, 1, 2, 3, 4]
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


class EvalEnv(SubTHzEnv6G):
    def __init__(self, traj):
        super().__init__(traj, w_mob=W_MOB, w_pwr=W_PWR, w_spec=W_SPEC)


# ═══════════════════════════════════════════════════════════════════════════════
#  SHANNON CAPACITY COMPUTATION
#
#  At each step:
#    SINR_dB     = obs[4] × 100.0
#    SINR_linear = 10^(SINR_dB / 10)
#    rb_count    = info['rb_count']
#    C_step      = rb_count × RB_BW_MHZ × log2(1 + SINR_linear)  [Mbps]
#
#  Per-episode throughput = mean(C_step) across all steps
#  Per-method throughput  = mean ± std across 5 seeds × 150 episodes
# ═══════════════════════════════════════════════════════════════════════════════

def compute_shannon_throughput(policy_fn, pool, label,
                                n_seeds=N_SEEDS, n_episodes=N_EPISODES):
    """
    Evaluate a policy and compute per-step Shannon capacity.

    Returns dict with:
      throughput_mean, throughput_std (Mbps per UE, averaged over episode steps)
      rb_mean, rb_std (mean RB count per step)
      sinr_mean, sinr_std (mean SINR dB per step)
      Also HO and resource metrics for cross-check.
    """
    print(f"  Evaluating: {label} ...")

    seed_throughput = []
    seed_rb         = []
    seed_sinr       = []
    seed_ho         = []
    seed_pwr        = []
    seed_spec       = []
    total_good = total_trigs = 0

    for seed in SEEDS:
        rng     = np.random.default_rng(seed)
        ep_tp_list  = []
        ep_rb_list  = []
        ep_sinr_list = []
        ep_pt_list  = []
        ep_spec_list = []
        good_ho = tot_ho = 0

        for ep in range(n_episodes):
            traj   = pool[rng.integers(len(pool))]
            env    = EvalEnv(traj)
            obs, _ = env.reset()
            done   = False
            step_tp_list   = []
            step_rb_list   = []
            step_sinr_list = []
            ep_pt  = []
            ep_rb  = []

            while not done:
                # Extract SINR from observation before action
                sinr_db     = float(obs[4]) * 100.0    # obs[4] = sinr_norm = sinr_db/100
                sinr_linear = 10.0 ** (sinr_db / 10.0)
                sinr_linear = max(sinr_linear, 1e-10)  # numerical safety

                action = policy_fn(obs)
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                rb_count = info['rb_count']
                ep_pt.append(info['pt_dbm'])
                ep_rb.append(rb_count)

                # Shannon capacity at this step (Mbps)
                C_step = rb_count * RB_BW_MHZ * np.log2(1.0 + sinr_linear)
                step_tp_list.append(C_step)
                step_rb_list.append(rb_count)
                step_sinr_list.append(sinr_db)

                if info.get('handover_done'):
                    tot_ho += 1
                    if not info.get('too_early') and not info.get('too_late'):
                        good_ho += 1

            ep_tp_list.append(np.mean(step_tp_list))
            ep_rb_list.append(np.mean(step_rb_list))
            ep_sinr_list.append(np.mean(step_sinr_list))
            ep_pt_list.append(
                (PT_MAX_DBM - float(np.mean(ep_pt))) / (PT_MAX_DBM - PT_MIN_DBM) * 100)
            ep_spec_list.append(
                (RB_MAX - float(np.mean(ep_rb))) / (RB_MAX - RB_MIN) * 100)

        seed_throughput.append(np.mean(ep_tp_list))
        seed_rb.append(np.mean(ep_rb_list))
        seed_sinr.append(np.mean(ep_sinr_list))
        seed_ho.append(good_ho / max(tot_ho, 1) * 100)
        seed_pwr.append(np.mean(ep_pt_list))
        seed_spec.append(np.mean(ep_spec_list))
        total_good  += good_ho
        total_trigs += tot_ho

    return {
        'label':             label,
        'throughput_mean':   np.mean(seed_throughput),
        'throughput_std':    np.std(seed_throughput),
        'rb_mean':           np.mean(seed_rb),
        'rb_std':            np.std(seed_rb),
        'sinr_mean':         np.mean(seed_sinr),
        'sinr_std':          np.std(seed_sinr),
        'ho_mean':           np.mean(seed_ho),
        'ho_std':            np.std(seed_ho),
        'pwr_mean':          np.mean(seed_pwr),
        'spec_mean':         np.mean(seed_spec),
        'total_good':        total_good,
        'total_trigs':       total_trigs,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("  Shannon Capacity Comparison — All Methods")
    print(f"  RB_BW = {TOTAL_BW_MHZ:.0f} MHz / {N_RB_MAX} RBs = {RB_BW_MHZ:.3f} MHz/RB")
    print(f"  C = rb_count × {RB_BW_MHZ:.3f} × log2(1 + SINR_linear)  [Mbps/UE]")
    print("=" * 72)

    print(f"\n[Data] Loading {CSV_PATH} ...")
    slow, med = load_trajectories(CSV_PATH)
    pool = slow + med
    print(f"[Data] Pool: {len(pool):,}")

    results = []

    # ── 3GPP A3 ───────────────────────────────────────────────────────────────
    a3_norm = A3_THRESH_DBM / 100.0
    def a3_policy(obs):
        return HANDOVER if float(obs[0]) < a3_norm else HOLD
    r = compute_shannon_throughput(a3_policy, pool, "3GPP A3 Rule-based")
    results.append(r)

    # ── Standard DQN ──────────────────────────────────────────────────────────
    if os.path.exists(STD_DQN_PATH):
        std_net, _ = load_standard_checkpoint(STD_DQN_PATH, str(DEVICE))
        def std_policy(obs):
            with torch.no_grad():
                return std_net(
                    torch.FloatTensor(obs).to(DEVICE)).argmax().item()
        r = compute_shannon_throughput(std_policy, pool, "Standard DQN (Config 6)")
        results.append(r)
    else:
        print(f"[SKIP] {STD_DQN_PATH} not found")

    # ── DDQN+PER (from old directory) ─────────────────────────────────────────
    if os.path.exists(DDQN_PATH):
        ddqn_net, _ = load_standard_checkpoint(DDQN_PATH, str(DEVICE))
        def ddqn_policy(obs):
            with torch.no_grad():
                return ddqn_net(
                    torch.FloatTensor(obs).to(DEVICE)).argmax().item()
        r = compute_shannon_throughput(ddqn_policy, pool, "DDQN+PER (no curriculum)")
        results.append(r)
    else:
        print(f"[SKIP] DDQN model not found at {DDQN_PATH}")
        print(f"       Copy dqn_baseline_ddqn_v5_mb_correct.pt here to include.")

    # ── PPO ───────────────────────────────────────────────────────────────────
    if os.path.exists(PPO_PATH):
        try:
            from stable_baselines3 import PPO
            ppo_model = PPO.load(PPO_PATH.replace('.zip', ''), device=DEVICE)
            def ppo_policy(obs):
                action, _ = ppo_model.predict(obs, deterministic=True)
                return int(action)
            r = compute_shannon_throughput(ppo_policy, pool, "PPO (SB3)")
            results.append(r)
        except Exception as e:
            print(f"[SKIP] PPO load failed: {e}")
    else:
        print(f"[SKIP] PPO model not found at {PPO_PATH}")
        print(f"       Copy baseline_ppo_v5_mb_correct.zip here to include.")

    # ── Dn-DQN ──────────────────────────────────────────────────────────────
    if not os.path.exists(DNDQN_PATH):
        sys.exit(f"[ERROR] {DNDQN_PATH} not found.")
    dndqn_net, dndqn_ckpt = load_dndqn_checkpoint(DNDQN_PATH, str(DEVICE))
    def dndqn_policy(obs):
        with torch.no_grad():
            return dndqn_net(
                torch.FloatTensor(obs).to(DEVICE)).argmax().item()
    r = compute_shannon_throughput(dndqn_policy, pool, "Dn-DQN (proposed)")
    results.append(r)

    # ── Results Table ─────────────────────────────────────────────────────────
    print(f"\n{'='*84}")
    print(f"  SHANNON CAPACITY RESULTS")
    print(f"  Metric: per-UE mean throughput (Mbps) = mean over all episode steps")
    print(f"  5 seeds × 150 episodes = 750 episodes per method")
    print(f"{'='*84}")
    hdr = (f"  {'Method':<30} {'Throughput(Mbps)':>18} "
           f"{'RB_mean':>9} {'SINR_dB':>9} {'Spec%':>8} {'HO%':>8}")
    print(hdr)
    print("  " + "-" * 82)
    for r in results:
        tp_s    = f"{r['throughput_mean']:.1f}±{r['throughput_std']:.1f}"
        rb_s    = f"{r['rb_mean']:.1f}±{r['rb_std']:.1f}"
        sinr_s  = f"{r['sinr_mean']:.1f}±{r['sinr_std']:.1f}"
        marker  = " ◄" if "Dn-DQN" in r['label'] else ""
        print(f"  {r['label']:<30} {tp_s:>18} {rb_s:>9} "
              f"{sinr_s:>9} {r['spec_mean']:>7.1f}% {r['ho_mean']:>7.1f}%{marker}")

    print(f"\n  RB_BW = {RB_BW_MHZ:.3f} MHz/RB | "
          f"Max throughput if all RBs used at peak SINR:")
    print(f"  C_max_theoretical = 66 × {RB_BW_MHZ:.3f} × log2(1 + SINR_peak)")

    # ── Integrity & Analysis ──────────────────────────────────────────────────
    print(f"\n[Analysis]")

    dndqn_r = next((r for r in results if "Dn-DQN" in r['label']), None)
    a3_r   = next((r for r in results if "A3" in r['label']), None)
    std_r  = next((r for r in results if "Standard" in r['label']), None)

    if dndqn_r and a3_r:
        tp_gain_vs_a3 = dndqn_r['throughput_mean'] - a3_r['throughput_mean']
        rb_saved      = a3_r['rb_mean'] - dndqn_r['rb_mean']
        print(f"\n  Dn-DQN vs A3 Rule-based:")
        print(f"    Throughput: {a3_r['throughput_mean']:.1f} → "
              f"{dndqn_r['throughput_mean']:.1f} Mbps "
              f"({tp_gain_vs_a3:+.1f} Mbps)")
        print(f"    RBs used:   {a3_r['rb_mean']:.1f} → "
              f"{dndqn_r['rb_mean']:.1f} (saved {rb_saved:.1f} RBs/UE)")
        if rb_saved > 0:
            # Freed RBs can serve additional concurrent UEs
            additional_ue_capacity = rb_saved / dndqn_r['rb_mean']
            print(f"    Freed RBs support {additional_ue_capacity:.1f}× more "
                  f"concurrent UEs at same per-UE allocation")

    if dndqn_r and std_r:
        tp_gain_vs_std = dndqn_r['throughput_mean'] - std_r['throughput_mean']
        print(f"\n  Dn-DQN vs Standard DQN:")
        print(f"    Throughput: {std_r['throughput_mean']:.1f} → "
              f"{dndqn_r['throughput_mean']:.1f} Mbps ({tp_gain_vs_std:+.1f})")
        print(f"    Spectrum saving: {std_r['spec_mean']:.1f}% → "
              f"{dndqn_r['spec_mean']:.1f}%")
        if tp_gain_vs_std > 0:
            print(f"    ✓ Dn-DQN achieves higher throughput AND higher spectrum saving")
            print(f"    → Dn-DQN identifies Pareto-superior operating point")
        elif tp_gain_vs_std > -50:
            print(f"    ~ Dn-DQN has marginally lower per-UE throughput but frees")
            print(f"      {std_r['rb_mean']-dndqn_r['rb_mean']:.1f} RBs/UE for system reuse")
        else:
            print(f"    ⚠ Dn-DQN significantly reduces throughput — investigate SINR impact")

    # Save
    rows = [{
        'method': r['label'],
        'throughput_mean': round(r['throughput_mean'], 3),
        'throughput_std':  round(r['throughput_std'],  3),
        'rb_mean':         round(r['rb_mean'],         2),
        'sinr_mean':       round(r['sinr_mean'],       2),
        'spec_mean':       round(r['spec_mean'],       2),
        'ho_mean':         round(r['ho_mean'],         2),
    } for r in results]
    pd.DataFrame(rows).to_csv("shannon_throughput_results.csv", index=False)
    print(f"\n[Save] shannon_throughput_results.csv")
    print(f"[Next] Analyse results and decide paper target (OJVT vs TVT).")


if __name__ == "__main__":
    main()
