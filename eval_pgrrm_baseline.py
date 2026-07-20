#!/usr/bin/env python3
"""
eval_pgrrm_baseline.py  --  Proactive Greedy RRM (PG-RRM) Baseline
================================================================================
IEEE Access Major Revision -- Reviewer Issue M4 (Baseline Fairness).

PG-RRM Policy (priority-ordered, applied each MDP step):
  1. HANDOVER   if RSRP < HO_PROACTIVE_THRESH       -- same trigger as DQN
  2. RB_DOWN    if SINR > SINR_HIGH and rb > RB_MIN  -- reduce spectrum (strong link)
  3. RB_UP      if SINR < SINR_LOW  and rb < RB_MAX  -- increase spectrum (weak link)
  4. PWR_DOWN   if RSRP > RSRP_STRONG and pt > PT_MIN -- reduce power (strong RSRP)
  5. PWR_UP     if RSRP < RSRP_MARG  and pt < PT_MAX  -- increase power (near edge)
  6. HOLD       otherwise

Thresholds: physically motivated (3GPP CQI mapping, link budget), NOT tuned.

Throughput formula: IDENTICAL to shannon_throughput.py
  C = rb_count x (2000 MHz / 66) x log2(1 + 10^(obs[4]*100/10))
  i.e. RB_BW = 2000e6 / RB_MAX = 30.303 MHz per RB
       SINR_dB = obs[4] * 100.0  (matches paper formula exactly)

Evaluation setup: IDENTICAL to multi_seed_eval.py
  Same CSV, filtering, pool (slow+medium), 5 seeds x 150 episodes.
  Same per-HO-event metric, same power/spectrum saving formulas.

RUN:
    python -u eval_pgrrm_baseline.py
"""

import numpy as np
import pandas as pd
import math
import time

from env_6g_v5 import (
    SubTHzEnv6G,
    RLF_THRESH_DBM,
    PT_MIN_DBM, PT_MAX_DBM,
    RB_MIN, RB_MAX,
    W_MOB, W_PWR, W_SPEC,
    HO_PROACTIVE_THRESH,
)

# Action indices — must match env_6g_v5.py ordering
HOLD, PWR_UP, PWR_DOWN, RB_UP, RB_DOWN, HANDOVER = 0, 1, 2, 3, 4, 5

# ── PG-RRM thresholds (physically motivated, not optimised) ─────────────────────
SINR_HIGH     = 15.0   # dB — 3GPP CQI 12+: excess capacity, reduce RBs
SINR_LOW      =  5.0   # dB — 3GPP CQI <=5: degraded link, increase RBs
RSRP_STRONG   = -65.0  # dBm — 10 dBm above proactive trigger: safe to reduce power
RSRP_MARGINAL = -72.0  # dBm —  3 dBm above proactive trigger: increase power

# ── Evaluation parameters — identical to multi_seed_eval.py ─────────────
CSV_PATH     = "6G_Handover_Training_Data_v3.csv"
N_SEEDS      = 5
N_EPISODES   = 150
SEEDS        = [0, 1, 2, 3, 4]
VEL_SLOW_MAX = 25.0
VEL_MED_MAX  = 40.0
HO_MIN_LEAD  = 25

# ── Throughput formula — identical to shannon_throughput.py ──────────────
# RB_BW = total 2000 MHz / 66 RBs = 30.303 MHz per RB
# SINR_dB = obs[4] * 100.0  (normalised observation, same as paper formula)
RB_BW_HZ = 2000e6 / RB_MAX   # 30,303,030 Hz per RB


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


def pgrrm_action(last_info):
    """
    Rule-based policy using info from previous step (1-step lag,
    consistent with E2_DELAY_STEPS=1 in env_6g_v5.py).
    HANDOVER is idempotent in env — safe to reissue each step.
    """
    if last_info is None:
        return HOLD

    rsrp = last_info['rsrp_dbm']
    sinr = last_info['sinr_db']
    pt   = last_info['pt_dbm']
    rb   = last_info['rb_count']

    if rsrp < HO_PROACTIVE_THRESH:          return HANDOVER
    if sinr > SINR_HIGH and rb > RB_MIN:    return RB_DOWN
    if sinr < SINR_LOW  and rb < RB_MAX:    return RB_UP
    if rsrp > RSRP_STRONG and pt > PT_MIN_DBM: return PWR_DOWN
    if rsrp < RSRP_MARGINAL and pt < PT_MAX_DBM: return PWR_UP
    return HOLD


def eval_one_seed(pool, seed):
    rng = np.random.default_rng(seed)
    good_ho = tot_ho = too_late = too_early = 0
    pwr_list, spec_list, rlf_list, tput_list = [], [], [], []

    for _ in range(N_EPISODES):
        traj   = pool[rng.integers(len(pool))]
        env    = SubTHzEnv6G(traj, w_mob=W_MOB, w_pwr=W_PWR, w_spec=W_SPEC)
        obs, _ = env.reset()
        done      = False
        last_info = None
        ep_pt, ep_rb, ep_tput, ep_rlf = [], [], [], False

        while not done:
            action = pgrrm_action(last_info)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            last_info = info

            # ── per-step resource metrics (identical to DQN eval) ──
            ep_pt.append(info['pt_dbm'])
            ep_rb.append(info['rb_count'])

            # ── Shannon throughput — identical formula to shannon_throughput.py
            # obs[4] = sinr_norm; obs[4]*100.0 = SINR in dB (paper formula)
            sinr_db_tput  = obs[4] * 100.0
            sinr_lin_tput = 10.0 ** (sinr_db_tput / 10.0)
            tput_mbps     = info['rb_count'] * (RB_BW_HZ / 1e6) * math.log2(1.0 + sinr_lin_tput)
            ep_tput.append(tput_mbps)

            # ── HO counting — identical to multi_seed_eval.py ──
            if info.get('handover_done'):
                tot_ho += 1
                tl = info.get('too_late',  False)
                te = info.get('too_early', False)
                if not tl and not te: good_ho  += 1
                if tl: too_late  += 1
                if te: too_early += 1

            if terminated and info['rsrp_dbm'] < RLF_THRESH_DBM:
                ep_rlf = True

        pwr_list.append(
            (PT_MAX_DBM - float(np.mean(ep_pt))) / (PT_MAX_DBM - PT_MIN_DBM) * 100)
        spec_list.append(
            (RB_MAX - float(np.mean(ep_rb))) / (RB_MAX - RB_MIN) * 100)
        tput_list.append(float(np.mean(ep_tput)))
        rlf_list.append(float(ep_rlf))

    return {
        'ho_pct':    good_ho / max(tot_ho, 1) * 100,
        'pwr':       np.mean(pwr_list),
        'spec':      np.mean(spec_list),
        'tput':      np.mean(tput_list),
        'rlf_eps':   int(np.sum(rlf_list)),
        'good_ho':   good_ho,
        'tot_ho':    tot_ho,
        'too_late':  too_late,
        'too_early': too_early,
    }


def main():
    print("=" * 72)
    print("  PG-RRM Heuristic Baseline -- IEEE Access Revision M4")
    print(f"  {N_SEEDS} seeds x {N_EPISODES} episodes = {N_SEEDS*N_EPISODES} total")
    print(f"  HO trigger:      RSRP < {HO_PROACTIVE_THRESH} dBm  (same as DQN)")
    print(f"  SINR thresholds: LOW={SINR_LOW} dB, HIGH={SINR_HIGH} dB")
    print(f"  RSRP thresholds: strong={RSRP_STRONG} dBm, marginal={RSRP_MARGINAL} dBm")
    print(f"  Throughput:      RB_BW={RB_BW_HZ/1e6:.3f} MHz/RB, SINR=obs[4]*100 dB")
    print(f"                   (matches shannon_throughput.py exactly)")
    print("=" * 72)

    print("\nLoading trajectories...")
    slow, med = load_trajectories(CSV_PATH)
    pool = slow + med
    print(f"  slow={len(slow)}, medium={len(med)}, pool={len(pool)}")

    results = []
    t_total = time.time()
    for s in SEEDS:
        t0 = time.time()
        r  = eval_one_seed(pool, s)
        results.append(r)
        print(f"  Seed {s}: HO={r['ho_pct']:.1f}%  Pwr={r['pwr']:.1f}%  "
              f"Spec={r['spec']:.1f}%  Tput={r['tput']:.1f} Mbps  "
              f"RLF={r['rlf_eps']}  ({time.time()-t0:.1f}s)")

    ho   = np.array([r['ho_pct'] for r in results])
    pwr  = np.array([r['pwr']    for r in results])
    spec = np.array([r['spec']   for r in results])
    tput = np.array([r['tput']   for r in results])
    rlf  = sum(r['rlf_eps']  for r in results)
    g_ho = sum(r['good_ho']  for r in results)
    t_ho = sum(r['tot_ho']   for r in results)
    late = sum(r['too_late'] for r in results)
    earl = sum(r['too_early']for r in results)

    print()
    print("=" * 72)
    print("  PG-RRM RESULTS  (5 seeds x 150 ep = 750 total)")
    print("=" * 72)
    print(f"  HO success (per-HO event): {ho.mean():.1f} +/- {ho.std():.1f} %")
    print(f"  Power saving:              {pwr.mean():.1f} +/- {pwr.std():.1f} %")
    print(f"  Spectrum saving:           {spec.mean():.1f} +/- {spec.std():.1f} %")
    print(f"  Shannon throughput:        {tput.mean():.1f} +/- {tput.std():.1f} Mbps")
    print(f"  Total RLF episodes:        {rlf}")
    print(f"  HO breakdown:  good={g_ho}, late={late}, early={earl}, total={t_ho}")
    print(f"  Elapsed:       {time.time()-t_total:.1f}s")
    print()
    print("  --- TABLE IV ROW ---")
    print(f"  | PG-RRM | {ho.mean():.1f}+-{ho.std():.1f} | "
          f"{pwr.mean():.1f}+-{pwr.std():.1f} | "
          f"{spec.mean():.1f}+-{spec.std():.1f} | "
          f"{tput.mean():.1f}+-{tput.std():.1f} |")

if __name__ == "__main__":
    main()
