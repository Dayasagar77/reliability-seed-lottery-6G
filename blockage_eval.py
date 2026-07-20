#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""blockage_eval.py — evaluate the FROZEN guarded controllers under real HighD truck blockage
(reviewer major comment 2). Two stages, both real:

  validate: no-blockage evaluation on the re-derived 10 Hz pool. GATE: the frozen br_gst models
            must reproduce their published operating point (~4.6-4.9% RLF) before any blockage
            claim is made from this pool.
  sweep:    same pool, same frozen models, truck-blockage attenuation applied to the serving link
            whenever the real per-step blocked flag is 1, swept across the vehicle-blockage loss
            range reported in mmWave/sub-THz measurements (10/15/20/25/30 dB).

Blockage is applied INSIDE _compute_rsrp (serving link), so it propagates to the observed state,
the guard input, the N310 RLF counter, and HO outcomes — the full pipeline sees it.
Usage: python -u blockage_eval.py validate | sweep [nep]
-> results/blockage_guard_stress.csv
"""
import sys, csv, math
import numpy as np, pandas as pd, torch
torch.set_num_threads(1)
_ARGV = list(sys.argv); sys.argv = [sys.argv[0]]
from dndqn_branched import load_branched
from dndqn import OBS_DIM
from env_6g_branched import BranchedEnv6G
from env_6g_v5 import W_MOB, W_PWR, W_SPEC, PT_MIN_DBM, PT_MAX_DBM, RB_MIN, RB_MAX
import multi_seed_eval as mse
sys.argv = _ARGV

GAIN = 15.0
MODELS = [f'br_gst_s{i}.pt' for i in range(5)]
FLAGS = ['/mnt/user-data/uploads/startfromscratch/rederived_blockage_%s.csv' % r for r in ('35', '40', '50')]
VEL_SLOW_MAX, VEL_MED_MAX, MIN_LEN = 25.0, 40.0, 35


def load_pool():
    """Re-derived trajectories as (T,3) [x, v, blocked]; slow+medium bands, len>=35 (same filters)."""
    slow, med, nfast = [], [], 0
    for p in FLAGS:
        df = pd.read_csv(p)
        for cid, g in df.groupby('Car_ID'):
            g = g.sort_values('step')
            a = g[['x', 'v_abs', 'blocked']].values.astype(np.float32)
            if len(a) < MIN_LEN:
                continue
            v = float(a[:, 1].mean())
            if v <= VEL_SLOW_MAX:
                slow.append(a)
            elif v <= VEL_MED_MAX:
                med.append(a)
            else:
                nfast += 1
    return slow, med, nfast


class BlockedBranchedEnv(BranchedEnv6G):
    """BranchedEnv6G whose serving link suffers `block_db` extra loss on real-blocked steps."""
    def __init__(self, traj, block_db=0.0, **kw):
        self._blk = traj[:, 2]
        super().__init__(traj[:, :2], **kw)
        self.block_db = float(block_db)

    def _compute_rsrp(self, d_m, apply_fading_flag=True):
        rsrp_dbm, is_los = super()._compute_rsrp(d_m, apply_fading_flag)
        i = min(self.step_idx, len(self._blk) - 1)
        if self.block_db > 0.0 and self._blk[i] > 0:
            rsrp_dbm -= self.block_db
        return rsrp_dbm, is_los


def eval_cond(net, pool, block_db, nep):
    su, pw, sp, rl = [], [], [], []
    for seed in mse.SEEDS:
        r = np.random.default_rng(seed); real = good = n = rlf = 0; P = []; S = []
        for ep in range(nep):
            traj = pool[r.integers(len(pool))]
            env = BlockedBranchedEnv(traj, block_db=block_db, binary_power=True, guard_db=9.0,
                                     guard_mode='state', array_gain_dbi=GAIN,
                                     w_mob=W_MOB, w_pwr=W_PWR, w_spec=W_SPEC)
            obs, _ = env.reset(seed=10_000 * seed + ep); done = False; n += 1; pts = []; rbs = []; info = {}
            while not done:
                a = net.act(obs); pg = env.serving_gnb_idx
                obs, _, t, tr_, info = env.step(a); done = t or tr_
                pts.append(info['pt_dbm']); rbs.append(info['rb_count'])
                if env.serving_gnb_idx != pg:
                    real += 1
                    if not info.get('too_early') and not info.get('too_late'): good += 1
            if info.get('sustained_rlf'): rlf += 1
            P.append((PT_MAX_DBM - np.mean(pts)) / (PT_MAX_DBM - PT_MIN_DBM) * 100)
            S.append((RB_MAX - np.mean(rbs)) / (RB_MAX - RB_MIN) * 100)
        su.append(100 * good / max(real, 1)); pw.append(np.mean(P)); sp.append(np.mean(S)); rl.append(100 * rlf / n)
    return dict(HO=float(np.mean(su)), Power=float(np.mean(pw)), Spec=float(np.mean(sp)), RLF=float(np.mean(rl)))


def run(conds, pool, nep, tag):
    nets = [load_branched(m, OBS_DIM)[0] for m in MODELS]
    out = []
    for name, bdb in conds:
        per = {k: [] for k in ('HO', 'Power', 'Spec', 'RLF')}
        for net in nets:
            m = eval_cond(net, pool, bdb, nep)
            for k in per: per[k].append(m[k])
        agg = {k: (float(np.mean(v)), float(np.std(v))) for k, v in per.items()}
        out.append((name, bdb, agg))
        print(f"  {name:26s} RLF {agg['RLF'][0]:5.2f}±{agg['RLF'][1]:4.2f}  Power {agg['Power'][0]:5.2f}±{agg['Power'][1]:4.2f}  "
              f"Spec {agg['Spec'][0]:5.2f}±{agg['Spec'][1]:4.2f}  HO {agg['HO'][0]:5.2f}±{agg['HO'][1]:4.2f}", flush=True)
    return out


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'validate'
    nep = int(sys.argv[2]) if len(sys.argv) > 2 else mse.N_EPISODES
    slow, med, nfast = load_pool()
    pool = slow + med
    print(f"re-derived pool: slow={len(slow)} med={len(med)} (fast excluded={nfast}) total={len(pool)}", flush=True)
    if mode == 'validate':
        run([('re-derived, no blockage', 0.0)], pool, nep, 'validate')
        print("  (published pool reference: RLF 4.61±0.92 paper / 4.85±1.05 unseeded rerun)")
    else:
        conds = [('no blockage (baseline)', 0.0)] + [(f'truck blockage {b} dB', float(b)) for b in (10, 15, 20, 25, 30)]
        out = run(conds, pool, nep, 'sweep')
        with open('results/blockage_guard_stress.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['condition', 'block_db', 'RLF_mean', 'RLF_std', 'Power_mean', 'Power_std',
                        'Spec_mean', 'Spec_std', 'HO_mean', 'HO_std'])
            for name, bdb, agg in out:
                w.writerow([name, bdb] + [f"{agg[k][j]:.3f}" for k in ('RLF', 'Power', 'Spec', 'HO') for j in (0, 1)])
        print("  -> results/blockage_guard_stress.csv", flush=True)
