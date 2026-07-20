#!/usr/bin/env python3
"""eval_nyu.py — fair re-evaluation of ALL methods under the NYU 142 GHz CI channel.
Same metrics as the paper's Table IV, but every method is run in SubTHzEnv6G_NYU (array
gain 15 dBi): REAL handover rate (actual serving-cell changes), HO success %, power saving,
spectrum saving, sustained-RLF %, and mean throughput. 5 seeds x 150 ep.
Requires the NYU-retrained checkpoints; rule baselines (A3, PG-RRM) need no training."""
import os, math, numpy as np, torch, csv
from env_6g_nyu import SubTHzEnv6G_NYU
from env_6g_v5 import (W_MOB, W_PWR, W_SPEC, HANDOVER, HOLD, A3_THRESH_DBM,
                       PT_MIN_DBM, PT_MAX_DBM, RB_MIN, RB_MAX)
from dndqn import load_dndqn_checkpoint
from qnet_plain import load_plain
from eval_pgrrm_baseline import pgrrm_action
import multi_seed_eval as mse

SEEDS, N_EP = mse.SEEDS, mse.N_EPISODES
RB_BW_HZ = 380.16e6 / RB_MAX
GAIN = 15.0
slow, med = mse.load_trajectories(mse.CSV_PATH); pool = slow + med


def a3_action(last):
    if last is None: return HOLD
    return HANDOVER if last['rsrp_dbm'] < A3_THRESH_DBM else HOLD


def evaluate(kind, net=None):
    HO, SUC, PWR, SPEC, TP, RLFR = [], [], [], [], [], []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        real = good = nep = rlf = 0; P = []; S = []; Tp = []
        for _ in range(N_EP):
            traj = pool[rng.integers(len(pool))]
            env = SubTHzEnv6G_NYU(traj, array_gain_dbi=GAIN, w_mob=W_MOB, w_pwr=W_PWR, w_spec=W_SPEC)
            obs, _ = env.reset(); done = False; last = None; nep += 1
            pts = []; rbs = []; ep_t = []
            while not done:
                if   kind == 'net':   a = int(net(torch.FloatTensor(obs)).argmax().item()) if True else 0
                elif kind == 'pgrrm': a = pgrrm_action(last)
                else:                 a = a3_action(last)
                pg = env.serving_gnb_idx
                obs, _, t, tr, info = env.step(a); last = info; done = t or tr
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
        HO.append(real / nep); SUC.append(100 * good / max(real, 1))
        PWR.append(np.mean(P)); SPEC.append(np.mean(S)); TP.append(np.mean(Tp)); RLFR.append(100 * rlf / nep)
    m = lambda a: (float(np.mean(a)), float(np.std(a)))
    return dict(hoep=m(HO), suc=m(SUC), pwr=m(PWR), spec=m(SPEC), tput=m(TP), rlf=m(RLFR))


CONFIGS = [
    ('Proposed Dn-DQN', 'net',   'dqn_phase2_dndqn_nyu.pt', 'dndqn'),
    ('DDQN+PER',        'net',   'dqn_ddqnper_nyu.pt',      'plain'),
    ('Standard DQN',    'net',   'dqn_stddqn_nyu.pt',       'plain'),
    ('PG-RRM',          'pgrrm', None,                      None),
    ('3GPP A3',         'a3',    None,                      None),
]

print("=" * 100)
print(f"  Table IV under the NYU 142 GHz CI channel (array gain {GAIN:.0f} dBi) | 5 seeds x {N_EP} ep")
print("=" * 100)
print(f"  {'method':>16} | {'realHO/ep':>9} | {'HOsucc%':>8} | {'Power%':>7} | {'Spec%':>7} | {'Tput(Mbps)':>11} | {'RLF%':>6}")
print("  " + "-" * 92)
rows = []
for name, kind, path, loader in CONFIGS:
    net = None
    if kind == 'net':
        if not path or not os.path.exists(path):
            print(f"  {name:>16} | (checkpoint {path} not found — retrain first)"); continue
        net = load_dndqn_checkpoint(path)[0] if loader == 'dndqn' else load_plain(path)[0]
        net.eval()
    r = evaluate(kind, net); rows.append((name, r))
    print(f"  {name:>16} | {r['hoep'][0]:9.2f} | {r['suc'][0]:7.1f}% | {r['pwr'][0]:6.1f}% | "
          f"{r['spec'][0]:6.1f}% | {r['tput'][0]:7.1f}+/-{r['tput'][1]:<3.0f} | {r['rlf'][0]:5.1f}%")

os.makedirs('results', exist_ok=True)
with open('results/eval_nyu.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['method', 'realHO_per_ep', 'HO_success', 'Power', 'Spec', 'Tput_Mbps', 'Tput_sd', 'RLF_pct'])
    for name, r in rows:
        w.writerow([name, f"{r['hoep'][0]:.3f}", f"{r['suc'][0]:.2f}", f"{r['pwr'][0]:.2f}",
                    f"{r['spec'][0]:.2f}", f"{r['tput'][0]:.2f}", f"{r['tput'][1]:.2f}", f"{r['rlf'][0]:.2f}"])
print("  -> results/eval_nyu.csv")
