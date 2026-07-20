#!/usr/bin/env python3
"""
fix_paper1_evals.py  —  Paper 1 validity fixes 1a + 1c, EVAL-ONLY (no training), all on the
NYU 142 GHz CI channel with the SAME 5-seed x 150-ep protocol as the headline results.

Fix 1a (train/eval leakage): evaluate frozen policies on the HELD-OUT `fast` pool (v > 40 m/s),
which was never used in training or eval, and compare to the in-distribution slow+med pool.
    -> results/generalization_fast_nyu.csv

Fix 1c (framing): one comparative table at MATCHED reliability — Proposed at its reliability
corner (fr_C4 / fr_C5_rel, RLF ~4-5%) next to the learned + rule baselines, same protocol.
    -> results/comparative_matched_nyu.csv
"""
import math, csv, os, numpy as np, torch
import train_phase2 as tp                                   # load_trajectories -> slow, med, fast, skip
import multi_seed_eval as mse                               # SEEDS, N_EPISODES
from env_6g_nyu import SubTHzEnv6G_NYU
from env_6g_v5 import (W_MOB, W_PWR, W_SPEC, PT_MIN_DBM, PT_MAX_DBM, RB_MIN, RB_MAX,
                       HANDOVER, HOLD, A3_THRESH_DBM)
from dndqn import load_dndqn_checkpoint
from qnet_plain import load_plain
from eval_pgrrm_baseline import pgrrm_action                # identical PG-RRM rule used in eval_nyu

SEEDS, N_EP = mse.SEEDS, mse.N_EPISODES
GAIN, RB_BW_HZ = 15.0, 380.16e6 / RB_MAX

def a3_action(last):                                        # 3GPP A3 event rule (same as eval_nyu.py)
    return HANDOVER if last['rsrp_dbm'] < A3_THRESH_DBM else HOLD

def load_net(path, dueling):
    return (load_dndqn_checkpoint(path)[0] if dueling else load_plain(path)[0])

def choose(net, obs, last, rule):
    if net is not None:
        with torch.no_grad():
            return int(net(torch.FloatTensor(obs)).argmax().item())
    return HOLD if last is None else rule(last)             # rules need a warm-up step

def eval_method(net, rule, pool):
    """Return per-metric (mean, std) across SEEDS on `pool`. Metric code == eval_one_nyu.py."""
    HOs, SUC, PWR, SPEC, TP, RLFR = [], [], [], [], [], []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        real = good = nep = rlf = 0; P = []; S = []; Tp = []
        for _ in range(N_EP):
            traj = pool[rng.integers(len(pool))]
            env = SubTHzEnv6G_NYU(traj, array_gain_dbi=GAIN, w_mob=W_MOB, w_pwr=W_PWR, w_spec=W_SPEC)
            obs, _ = env.reset(); done = False; nep += 1; pts = []; rbs = []; ep_t = []; last = None
            while not done:
                a = choose(net, obs, last, rule)
                pg = env.serving_gnb_idx
                obs, _, t, tr, info = env.step(a); done = t or tr; last = info
                pts.append(info['pt_dbm']); rbs.append(info['rb_count'])
                sdb = obs[4] * 100.0
                ep_t.append(info['rb_count'] * (RB_BW_HZ / 1e6) * math.log2(1 + 10 ** (sdb / 10)))
                if env.serving_gnb_idx != pg:
                    real += 1
                    if not info.get('too_early') and not info.get('too_late'): good += 1
            if info.get('sustained_rlf'): rlf += 1
            P.append((PT_MAX_DBM - np.mean(pts)) / (PT_MAX_DBM - PT_MIN_DBM) * 100)
            S.append((RB_MAX - np.mean(rbs)) / (RB_MAX - RB_MIN) * 100)
            Tp.append(np.mean(ep_t))
        HOs.append(100 * good / max(real, 1)); PWR.append(np.mean(P)); SPEC.append(np.mean(S))
        TP.append(np.mean(Tp)); RLFR.append(100 * rlf / nep)
    g = lambda a: (float(np.mean(a)), float(np.std(a)))
    return dict(HO=g(HOs), Power=g(PWR), Spec=g(SPEC), Tput=g(TP), RLF=g(RLFR))

def main():
    os.makedirs('results', exist_ok=True)
    slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH)
    indist = slow + med
    print(f"[data] slow={len(slow)} med={len(med)} FAST(held-out)={len(fast)} skipped={skip}", flush=True)

    # ---------- Fix 1a: held-out generalization ----------
    gen_methods = [
        ('Proposed Dn-DQN (rel)', 'dqn_phase2_dndqn_nyu_rel.pt', True),
        ('Standard DQN',          'dqn_stddqn_nyu.pt',           False),
    ]
    rows = []
    if len(fast) == 0:
        print("[warn] FAST pool empty — cannot run held-out generalization", flush=True)
    else:
        for label, path, dz in gen_methods:
            if not os.path.exists(path):
                print(f"[skip] {label}: {path} missing", flush=True); continue
            net = load_net(path, dz)
            for pool_name, pool in [('in-dist(slow+med)', indist), ('held-out(fast)', fast)]:
                m = eval_method(net, None, pool)
                rows.append({'method': label, 'pool': pool_name,
                             'HO': f"{m['HO'][0]:.2f}", 'HO_sd': f"{m['HO'][1]:.2f}",
                             'Power': f"{m['Power'][0]:.2f}", 'Spec': f"{m['Spec'][0]:.2f}",
                             'RLF': f"{m['RLF'][0]:.2f}", 'RLF_sd': f"{m['RLF'][1]:.2f}",
                             'Tput': f"{m['Tput'][0]:.1f}"})
                print(f"[gen] {label:24s} {pool_name:18s} HO {m['HO'][0]:.1f}% Pwr {m['Power'][0]:.1f}% "
                      f"Spec {m['Spec'][0]:.1f}% RLF {m['RLF'][0]:.1f}% Tput {m['Tput'][0]:.0f}", flush=True)
        with open('results/generalization_fast_nyu.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['method','pool','HO','HO_sd','Power','Spec','RLF','RLF_sd','Tput'])
            w.writeheader(); w.writerows(rows)
        print("  -> results/generalization_fast_nyu.csv", flush=True)

    # ---------- Fix 1c: comparative table at matched reliability ----------
    comp_methods = [
        ('Proposed Dn-DQN @C4', 'fr_C4.pt',            True,  None),
        ('Proposed Dn-DQN @C5', 'fr_C5_rel.pt',        True,  None),
        ('DDQN+PER',            'dqn_ddqnper_nyu.pt',  False, None),
        ('Standard DQN',        'dqn_stddqn_nyu.pt',   False, None),
        ('PG-RRM',              None,                   None,  pgrrm_action),
        ('3GPP A3',             None,                   None,  a3_action),
    ]
    crows = []
    for label, path, dz, rule in comp_methods:
        if path is not None and not os.path.exists(path):
            print(f"[skip] {label}: {path} missing", flush=True); continue
        net = load_net(path, dz) if path is not None else None
        m = eval_method(net, rule, indist)
        crows.append({'method': label,
                      'HO': f"{m['HO'][0]:.2f}", 'HO_sd': f"{m['HO'][1]:.2f}",
                      'Power': f"{m['Power'][0]:.2f}", 'Power_sd': f"{m['Power'][1]:.2f}",
                      'Spec': f"{m['Spec'][0]:.2f}", 'Spec_sd': f"{m['Spec'][1]:.2f}",
                      'RLF': f"{m['RLF'][0]:.2f}", 'RLF_sd': f"{m['RLF'][1]:.2f}",
                      'Tput': f"{m['Tput'][0]:.1f}"})
        print(f"[comp] {label:22s} HO {m['HO'][0]:.1f}% Pwr {m['Power'][0]:.1f}% "
              f"Spec {m['Spec'][0]:.1f}% RLF {m['RLF'][0]:.1f}% Tput {m['Tput'][0]:.0f}", flush=True)
    with open('results/comparative_matched_nyu.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['method','HO','HO_sd','Power','Power_sd','Spec','Spec_sd','RLF','RLF_sd','Tput'])
        w.writeheader(); w.writerows(crows)
    print("  -> results/comparative_matched_nyu.csv", flush=True)
    print("DONE fix_paper1_evals", flush=True)

if __name__ == "__main__":
    main()
