#!/usr/bin/env python3
"""
multi_seed_eval.py  —  Multi-Seed Evaluation for Dn-DQN
===============================================================
5 seeds × 150 episodes = 750 total episodes.
Uses per-HO event metric — standard in DRL handover literature.

RUN:
    python -u multi_seed_eval.py
    (~8 min)
"""

import os, sys, time
import numpy as np
import pandas as pd
import torch

from env_6g_v5 import (
    SubTHzEnv6G,
    RLF_THRESH_DBM, PT_MIN_DBM, PT_MAX_DBM, RB_MIN, RB_MAX,
    W_MOB, W_PWR, W_SPEC,
)
from dndqn import load_dndqn_checkpoint, OBS_DIM, NET_ARCH

MODEL_PATH  = "dqn_phase2_dndqn.pt"
CSV_PATH    = "6G_Handover_Training_Data_v3.csv"
N_SEEDS     = 5
N_EPISODES  = 150
SEEDS       = [0, 1, 2, 3, 4]
VEL_SLOW_MAX = 25.0
VEL_MED_MAX  = 40.0
HO_MIN_LEAD  = 25
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def eval_one_seed(net, pool, seed):
    net.eval()
    rng = np.random.default_rng(seed)
    good_ho = tot_ho = too_late = too_early = 0
    pwr_list, spec_list, rlf_list, lat_list = [], [], [], []

    for _ in range(N_EPISODES):
        traj   = pool[rng.integers(len(pool))]
        env    = EvalEnv(traj)
        obs, _ = env.reset()
        done   = False
        ep_pt, ep_rb = [], []
        ep_rlf = False

        while not done:
            t0 = time.perf_counter()
            with torch.no_grad():
                action = net(torch.FloatTensor(obs).to(DEVICE)).argmax().item()
            lat_list.append((time.perf_counter() - t0) * 1000)

            obs, _, terminated, truncated, info = env.step(action)
            ep_pt.append(info['pt_dbm'])
            ep_rb.append(info['rb_count'])
            done = terminated or truncated

            if info.get('handover_done'):
                tot_ho += 1
                tl = info.get('too_late', False)
                te = info.get('too_early', False)
                if not tl and not te:
                    good_ho += 1
                if tl: too_late  += 1
                if te: too_early += 1

            if terminated and info['rsrp_dbm'] < RLF_THRESH_DBM:
                ep_rlf = True

        pwr_list.append(
            (PT_MAX_DBM - float(np.mean(ep_pt))) / (PT_MAX_DBM - PT_MIN_DBM) * 100)
        spec_list.append(
            (RB_MAX - float(np.mean(ep_rb))) / (RB_MAX - RB_MIN) * 100)
        rlf_list.append(float(ep_rlf))

    lat = np.array(lat_list)
    return {
        'ho_pct':       good_ho / max(tot_ho, 1) * 100,
        'ep_comp':      (1 - np.mean(rlf_list)) * 100,
        'pwr':          np.mean(pwr_list),
        'spec':         np.mean(spec_list),
        'lat_mean':     float(np.mean(lat)),
        'lat_p99':      float(np.percentile(lat, 99)),
        'lat_max':      float(np.max(lat)),
        'good_ho':      good_ho,
        'tot_ho':       tot_ho,
        'too_late':     too_late,
        'too_early':    too_early,
        'rlf_eps':      int(np.sum(rlf_list)),
    }


def main():
    print("=" * 72)
    print("  Dn-DQN Multi-Seed Evaluation")
    print(f"  {N_SEEDS} seeds × {N_EPISODES} episodes = {N_SEEDS*N_EPISODES} total")
    print("  Metric: per-HO event success rate")
    print("=" * 72)

    if not os.path.exists(MODEL_PATH):
        sys.exit(f"[ERROR] {MODEL_PATH} not found. Run train_phase2.py first.")

    print(f"\n[Data] Loading {CSV_PATH} ...")
    slow, med = load_trajectories(CSV_PATH)
    pool = slow + med
    print(f"[Data] Pool: {len(pool):,} trajectories")

    print(f"\n[Model] Loading {MODEL_PATH} ...")
    net, ckpt = load_dndqn_checkpoint(MODEL_PATH, device=str(DEVICE))
    print(f"[Model] Algorithm: {ckpt.get('algorithm','?')} | "
          f"N-step={ckpt.get('n_steps','?')}")
    print(f"[Model] W_MOB={ckpt.get('w_mob')} W_PWR={ckpt.get('w_pwr')} "
          f"W_SPEC={ckpt.get('w_spec')}")

    print(f"\n[Eval] Running {N_SEEDS} seeds ...\n")
    hdr = (f"  {'Seed':<5} {'HO%(per-ev)':>12} {'EpComp%':>9} "
           f"{'Pwr%':>7} {'Spec%':>7} {'HO_trigs':>9} {'LatMax':>8} {'RLF':>6}")
    print(hdr)
    print("  " + "-" * 68)

    results = []
    for seed in SEEDS:
        r = eval_one_seed(net, pool, seed)
        results.append(r)
        print(f"  {seed:<5} {r['ho_pct']:>11.1f}% {r['ep_comp']:>8.1f}% "
              f"{r['pwr']:>6.1f}% {r['spec']:>6.1f}% "
              f"{r['tot_ho']:>9} {r['lat_max']:>7.3f}ms {r['rlf_eps']:>6}")

    ho_arr   = np.array([r['ho_pct']  for r in results])
    pwr_arr  = np.array([r['pwr']     for r in results])
    spec_arr = np.array([r['spec']    for r in results])
    comp_arr = np.array([r['ep_comp'] for r in results])
    lat_max  = float(np.max([r['lat_max'] for r in results]))
    lat_mean = float(np.mean([r['lat_mean'] for r in results]))
    lat_p99  = float(np.mean([r['lat_p99']  for r in results]))

    tot_good  = sum(r['good_ho']   for r in results)
    tot_trigs = sum(r['tot_ho']    for r in results)
    tot_late  = sum(r['too_late']  for r in results)
    tot_rlf   = sum(r['rlf_eps']   for r in results)
    avg_ho_ep = tot_trigs / (N_SEEDS * N_EPISODES)

    print(f"\n{'='*72}")
    print(f"  FINAL RESULTS  (5 seeds × 150 ep = 750 episodes)")
    print(f"{'='*72}")
    print(f"\n  ── PRIMARY METRICS (Table III) ──")
    print(f"  HO Success Rate  : {ho_arr.mean():.1f}% ± {ho_arr.std():.1f}%")
    print(f"    (per-HO: {tot_good}/{tot_trigs} triggers, "
          f"{avg_ho_ep:.1f} avg/episode)")
    print(f"  Power Saving     : {pwr_arr.mean():.1f}% ± {pwr_arr.std():.1f}%")
    print(f"  Spectrum Saving  : {spec_arr.mean():.1f}% ± {spec_arr.std():.1f}%")
    print(f"\n  ── LATENCY (O-RAN Near-RT RIC) ──")
    print(f"  Mean             : {lat_mean:.3f} ms")
    print(f"  p99              : {lat_p99:.3f} ms")
    print(f"  Max              : {lat_max:.3f} ms")
    print(f"  Near-RT RIC      : "
          f"{'PASS ✓ (max < 10 ms)' if lat_max < 10 else 'FAIL ✗'}")
    print(f"\n  ── DISCLOSURE METRICS ──")
    print(f"  Avg HO triggers/episode : {avg_ho_ep:.1f}")
    print(f"  Too-late HOs     : {tot_late} ({tot_late/max(tot_trigs,1)*100:.1f}%)")
    print(f"  Episode completion: {comp_arr.mean():.1f}% ± {comp_arr.std():.1f}%")
    print(f"  RLF episodes     : {tot_rlf}/{N_SEEDS*N_EPISODES} "
          f"({tot_rlf/(N_SEEDS*N_EPISODES)*100:.1f}%)")
    print(f"{'='*72}")

    print(f"\n[Integrity checks]")
    print(f"  {'✓' if ho_arr.mean()>=90 else '⚠'} HO = {ho_arr.mean():.1f}% "
          f"{'≥90% PASS' if ho_arr.mean()>=90 else '< 90% — investigate'}")
    print(f"  {'✓' if ho_arr.std()<=3 else '⚠'} HO std = {ho_arr.std():.1f}%")
    print(f"  {'✓' if tot_late/max(tot_trigs,1)*100<=5 else '⚠'} "
          f"Too-late = {tot_late/max(tot_trigs,1)*100:.1f}%")
    print(f"\n[Next] Run: python -u collect_ablation.py")


if __name__ == "__main__":
    main()
