#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""guard_noise_exp.py — robustness of the measured state guard to RSRP measurement noise and
reporting delay (answers reviewer major comment 4). ZERO fabrication: reuses the exact
evaluate_branched protocol (5 eval seeds x 150 episodes) and the 5 trained guarded models
(br_gst_s0..4). Physics (sustained-RLF declaration, throughput, HO outcomes) always uses the
TRUE rsrp; only the rsrp the xApp *observes* (NN state + guard threshold, Eq. 12) is corrupted.

Usage: python -u guard_noise_exp.py validate      # reproduce paper baseline (unseeded, as-published)
       python -u guard_noise_exp.py sweep [nep]    # paired seeded noise/delay sweep
"""
import sys, math, csv, numpy as np, torch
torch.set_num_threads(1)
_ARGV = list(sys.argv); sys.argv = [sys.argv[0]]   # neutralise argv for modules that parse it at import
import train_phase2 as tp
from dndqn_branched import load_branched, R_STAY_PROTECT
from dndqn import OBS_DIM
from env_6g_branched import BranchedEnv6G
from env_6g_v5 import W_MOB, W_PWR, W_SPEC, PT_MIN_DBM, PT_MAX_DBM, RB_MIN, RB_MAX
from train_branched_nyu import evaluate_branched   # the EXACT published protocol
import multi_seed_eval as mse
sys.argv = _ARGV

GAIN, RB_BW_HZ = 15.0, 380.16e6 / RB_MAX
MODELS = [f'br_gst_s{i}.pt' for i in range(5)]
PAPER = dict(HO=95.18, Power=63.61, Spec=80.63, RLF=4.61)


def ms(a):
    return float(np.mean(a)), float(np.std(a))


# ─────────────────────────────────────────────────────────────────────────────
class NoisyMeasBranchedEnv(BranchedEnv6G):
    """E2 measurement imperfection on the observed RSRP only. sigma_db = Gaussian
    measurement-noise std; extra_delay = reporting-latency steps ADDED to the base
    env's inherent 1-step guard delay (Eq.12 already uses RSRP_{t-1})."""
    def __init__(self, *a, meas_sigma_db=0.0, extra_delay=0, meas_seed=0, **kw):
        super().__init__(*a, **kw)
        self.meas_sigma_db = float(meas_sigma_db)
        self.extra_delay = int(extra_delay)
        self._mrng = np.random.default_rng(meas_seed)
        self._meas_hist = []
        self._cur_noise = 0.0

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._meas_hist = []
        self._cur_noise = 0.0
        return obs, info

    def _build_obs(self, rsrp_dbm, sinr_db, velocity_ms):
        # NN sees the noisy measured rsrp (same realization the guard will use)
        return super()._build_obs(rsrp_dbm + self._cur_noise, sinr_db, velocity_ms)

    def step(self, action):
        self._cur_noise = self._mrng.normal(0.0, self.meas_sigma_db) if self.meas_sigma_db > 0 else 0.0
        obs, reward, term, trunc, info = super().step(action)
        measured = info['rsrp_dbm'] + self._cur_noise            # true physics rsrp + measurement noise
        self._meas_hist.append(measured)
        idx = max(0, len(self._meas_hist) - 1 - self.extra_delay)
        self._prev_rsrp = self._meas_hist[idx]                   # guard consumes delayed noisy measurement
        return obs, reward, term, trunc, info


def eval_noisy(net, pool, sigma_db, extra_delay):
    """Paired, SEEDED copy of evaluate_branched using NoisyMeasBranchedEnv (fading + noise
    reproducible; identical trajectories/fading across conditions -> clean delta)."""
    su, pw, sp, rl = [], [], [], []
    for seed in mse.SEEDS:
        r = np.random.default_rng(seed); real = good = nep = rlf = 0; P = []; S = []
        for ep in range(mse.N_EPISODES):
            traj = pool[r.integers(len(pool))]
            env = NoisyMeasBranchedEnv(traj, binary_power=True, guard_db=9.0, guard_mode='state',
                                       array_gain_dbi=GAIN, w_mob=W_MOB, w_pwr=W_PWR, w_spec=W_SPEC,
                                       meas_sigma_db=sigma_db, extra_delay=extra_delay,
                                       meas_seed=1000 * seed + ep)
            obs, _ = env.reset(seed=10_000 * seed + ep); done = False; nep += 1; pts = []; rbs = []; info = {}
            while not done:
                a = net.act(obs); pg = env.serving_gnb_idx
                obs, _, t, tr, info = env.step(a); done = t or tr
                pts.append(info['pt_dbm']); rbs.append(info['rb_count'])
                if env.serving_gnb_idx != pg:
                    real += 1
                    if not info.get('too_early') and not info.get('too_late'): good += 1
            if info.get('sustained_rlf'): rlf += 1
            P.append((PT_MAX_DBM - np.mean(pts)) / (PT_MAX_DBM - PT_MIN_DBM) * 100)
            S.append((RB_MAX - np.mean(rbs)) / (RB_MAX - RB_MIN) * 100)
        su.append(100 * good / max(real, 1)); pw.append(np.mean(P)); sp.append(np.mean(S)); rl.append(100 * rlf / nep)
    return dict(HO=float(np.mean(su)), Power=float(np.mean(pw)), Spec=float(np.mean(sp)), RLF=float(np.mean(rl)))


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'validate'
    slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH); pool = slow + med
    print(f"pool={len(pool)} (slow {len(slow)}/med {len(med)}) | models={len(MODELS)} | {mode}", flush=True)

    if mode == 'validate':
        print("\n== VALIDATION: exact published protocol (unseeded fading), guarded arm ==")
        per = {k: [] for k in ['HO', 'Power', 'Spec', 'RLF']}
        for m in MODELS:
            net = load_branched(m, OBS_DIM)[0]
            r = evaluate_branched(net, pool, binary_power=True, guard_db=9.0, guard_mode='state')
            for k in per: per[k].append(r[k])
            print(f"  {m}: RLF {r['RLF']:5.2f}  Power {r['Power']:5.2f}  Spec {r['Spec']:5.2f}  HO {r['HO']:5.2f}", flush=True)
        print("  " + "-" * 60)
        for k in ['RLF', 'Power', 'Spec', 'HO']:
            mu, sd = ms(per[k])
            print(f"  {k:6s} {mu:6.2f} ± {sd:5.2f}   (paper {PAPER[k]})", flush=True)

    else:
        nep = int(sys.argv[2]) if len(sys.argv) > 2 else mse.N_EPISODES
        mse.N_EPISODES = nep
        conds = [('baseline (σ=0)', 0.0, 0), ('σ=1 dB', 1.0, 0), ('σ=2 dB', 2.0, 0),
                 ('σ=3 dB', 3.0, 0), ('delay +1 step (80 ms)', 0.0, 1), ('σ=2 dB + delay', 2.0, 1)]
        nets = [load_branched(m, OBS_DIM)[0] for m in MODELS]
        rows = []
        for name, sig, dly in conds:
            per = {k: [] for k in ['HO', 'Power', 'Spec', 'RLF']}
            for net in nets:
                r = eval_noisy(net, pool, sig, dly)
                for k in per: per[k].append(r[k])
            agg = {k: ms(per[k]) for k in per}
            rows.append((name, agg, per['RLF']))
            print(f"  {name:22s} RLF {agg['RLF'][0]:5.2f}±{agg['RLF'][1]:4.2f}  "
                  f"Power {agg['Power'][0]:5.2f}±{agg['Power'][1]:4.2f}  "
                  f"Spec {agg['Spec'][0]:5.2f}±{agg['Spec'][1]:4.2f}  HO {agg['HO'][0]:5.2f}±{agg['HO'][1]:4.2f}", flush=True)
        base = rows[0][1]
        print("\n  Δ vs baseline (pp):")
        for name, agg, _ in rows[1:]:
            print(f"    {name:22s} ΔRLF {agg['RLF'][0]-base['RLF'][0]:+5.2f}  ΔPower {agg['Power'][0]-base['Power'][0]:+5.2f}", flush=True)
        with open('results/guard_noise_sensitivity.csv', 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['condition', 'RLF_mean', 'RLF_std', 'Power_mean', 'Power_std', 'Spec_mean', 'Spec_std', 'HO_mean', 'HO_std'])
            for name, agg, _ in rows:
                w.writerow([name] + [f"{agg[k][j]:.3f}" for k in ['RLF', 'Power', 'Spec', 'HO'] for j in (0, 1)])
        print("  -> results/guard_noise_sensitivity.csv", flush=True)
