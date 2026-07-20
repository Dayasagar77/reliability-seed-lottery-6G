#!/usr/bin/env python3
"""train_branched_nyu.py — trains the cliff-aware branched JOINT controller (Paper 1 novelty) on the
NYU 142 GHz CI channel, reliability reward. Two branched configs x seeds:
    binary_power=True  -> cliff-aware (PROPOSED)      -> ckpt br_bin_s{seed}.pt
    binary_power=False -> vanilla-BDQ (incremental)   -> ckpt br_inc_s{seed}.pt
Compare against the existing FLAT Dn-DQN multi-seed (Spec 88.4/RLF 11.5) — the 3-way ablation
(flat / branched-incremental / branched-binary) that proves the binarisation earns its place.
Best-checkpointed, resumable via results/branched_ablation_nyu.csv. Waits for the rigor batch (n-step/
reward/alpha-mu) to free the CPU first.
Usage: python -u train_branched_nyu.py [n_seeds=3] [steps=800000]
"""
import sys, os, time, random, math, csv, subprocess
from collections import deque
import numpy as np, torch, torch.nn as nn, torch.optim as optim

from env_6g_branched import BranchedEnv6G
from dndqn_branched import build_branched_networks, branched_td_loss, save_branched, load_branched
from dndqn import NStepPERBuffer, polyak_update, epsilon_schedule, beta_schedule, OBS_DIM
from env_6g_v5 import W_MOB, W_PWR, W_SPEC, PT_MIN_DBM, PT_MAX_DBM, RB_MIN, RB_MAX
import train_phase2 as tp
import multi_seed_eval as mse

N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
STEPS   = int(sys.argv[2]) if len(sys.argv) > 2 else 800_000
EVAL_SEEDS, EVAL_EP = mse.SEEDS, mse.N_EPISODES
GAIN, RB_BW_HZ, GAMMA = 15.0, 380.16e6 / RB_MAX, 0.99
EPS_START, EPS_END, EPS_FRACTION, STAGE_SLOW_END = 1.0, 0.05, 0.30, 0.08
LEARNING_STARTS, TRAIN_FREQ, BATCH, TARGET_UPDATE, TAU = 10_000, 4, 256, 500, 0.01
PER_BETA0, PER_BETA1, NSTEP, ALPHA = 0.4, 1.0, 3, 0.6

def set_seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s)

def train(seed, binary_power, out, slow, med, steps=STEPS, guard_db=None, guard_mode='state'):
    set_seed(seed); pool = slow + med
    online, target = build_branched_networks(OBS_DIM)
    opt = optim.Adam(online.parameters(), lr=3e-4)
    buf = NStepPERBuffer(500_000, n_steps=NSTEP, gamma=GAMMA, alpha=ALPHA)
    rng = np.random.default_rng(seed)
    smp = lambda stg: (slow if stg == "SLOW" else pool)[rng.integers(len(slow if stg == "SLOW" else pool))]
    stage = "SLOW"; env = BranchedEnv6G(smp(stage), binary_power=binary_power, guard_db=guard_db, guard_mode=guard_mode); obs, _ = env.reset()
    epr = deque(maxlen=20); er = 0.0; best = -1e18; saved = False
    def _save(): save_branched(online, out, {'seed': seed, 'binary_power': binary_power, 'training_mode': 'nyu_ci_reliability_tilt'})
    for step in range(1, steps + 1):
        frac = step / steps; stage = "SLOW" if frac < STAGE_SLOW_END else "SLOW+MED"
        eps = epsilon_schedule(step, EPS_START, EPS_END, EPS_FRACTION, steps); beta = beta_schedule(step, PER_BETA0, PER_BETA1, steps)
        if random.random() < eps:
            a = [random.randint(0, 2), random.randint(0, 2)]
        else:
            a = online.act(obs)
        nobs, r, term, trunc, info = env.step(a); done = term or trunc; er += r
        buf.add(obs, a, r, nobs, term); obs = nobs
        if done:
            epr.append(er); er = 0.0; env = BranchedEnv6G(smp(stage), binary_power=binary_power, guard_db=guard_db, guard_mode=guard_mode); obs, _ = env.reset()
            if frac > EPS_FRACTION and len(epr) == epr.maxlen:
                m = float(np.mean(epr))
                if m > best: best = m; _save(); saved = True
        if step >= LEARNING_STARTS and step % TRAIN_FREQ == 0 and len(buf) >= BATCH:
            ib, wb, ob, ab, rb, nb, db = buf.sample(BATCH, beta)
            loss, td = branched_td_loss(online, target, ob, ab, rb, nb, db, wb, gamma_n=GAMMA ** NSTEP)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(online.parameters(), 10.0); opt.step()
            buf.update_priorities(ib, td)
        if step % TARGET_UPDATE == 0: polyak_update(online, target, TAU)
    if not saved: _save()

def evaluate_branched(net, pool, binary_power, guard_db=None, guard_mode='state'):
    su = []; pw = []; sp = []; tpv = []; rl = []
    for seed in EVAL_SEEDS:
        r = np.random.default_rng(seed); real = good = nep = rlf = 0; P = []; S = []; T = []
        for _ in range(EVAL_EP):
            traj = pool[r.integers(len(pool))]
            env = BranchedEnv6G(traj, binary_power=binary_power, guard_db=guard_db, guard_mode=guard_mode, array_gain_dbi=GAIN, w_mob=W_MOB, w_pwr=W_PWR, w_spec=W_SPEC)
            obs, _ = env.reset(); done = False; nep += 1; pts = []; rbs = []; et = []; info = {}
            while not done:
                a = net.act(obs); pg = env.serving_gnb_idx
                obs, _, t, tr, info = env.step(a); done = t or tr
                pts.append(info['pt_dbm']); rbs.append(info['rb_count']); sdb = obs[4] * 100.0
                et.append(info['rb_count'] * (RB_BW_HZ / 1e6) * math.log2(1 + 10 ** (sdb / 10)))
                if env.serving_gnb_idx != pg:
                    real += 1
                    if not info.get('too_early') and not info.get('too_late'): good += 1
            if info.get('sustained_rlf'): rlf += 1
            P.append((PT_MAX_DBM - np.mean(pts)) / (PT_MAX_DBM - PT_MIN_DBM) * 100)
            S.append((RB_MAX - np.mean(rbs)) / (RB_MAX - RB_MIN) * 100); T.append(np.mean(et))
        su.append(100 * good / max(real, 1)); pw.append(np.mean(P)); sp.append(np.mean(S)); tpv.append(np.mean(T)); rl.append(100 * rlf / nep)
    return dict(HO=float(np.mean(su)), Power=float(np.mean(pw)), Spec=float(np.mean(sp)), Tput=float(np.mean(tpv)), RLF=float(np.mean(rl)))

def _write(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['config', 'seed', 'HO', 'Power', 'Spec', 'RLF', 'Tput', 'train_min']); w.writeheader(); w.writerows(rows)

def waiting():
    try: return bool(subprocess.run(["pgrep", "-f", "fix_nstep|fix_reward|fix_alpha"], capture_output=True, text=True).stdout.strip())
    except Exception: return False

def main():
    while waiting(): print("[wait] rigor batch (nstep/reward/alpha) running; sleep 120s", flush=True); time.sleep(120)
    print(f"BRANCHED ablation | {N_SEEDS} seeds | {STEPS:,} steps | NYU CI | binary(PROPOSED) + incremental(vanilla-BDQ)", flush=True)
    slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH); pool = slow + med
    os.makedirs('results', exist_ok=True); path = 'results/branched_ablation_nyu.csv'
    done = set(); rows = []
    if os.path.exists(path):
        for r in csv.DictReader(open(path)): done.add((r['config'], int(r['seed']))); rows.append(r)
    for label, bp, tag, grd in [('branched-binary (proposed)', True, 'bin', None), ('branched-guarded-state (proposed 9dB)', True, 'gst', 9.0), ('branched-incremental (BDQ)', False, 'inc', None)]:
        for seed in range(N_SEEDS):
            if (label, seed) in done: print(f"[skip] {label} seed {seed}", flush=True); continue
            ck = f"br_{tag}_s{seed}.pt"; t0 = time.time()
            print(f"[train] {label} seed {seed} -> {ck}", flush=True)
            train(seed, bp, ck, slow, med, guard_db=grd)
            m = evaluate_branched(load_branched(ck, OBS_DIM)[0], pool, bp, guard_db=grd); mins = (time.time() - t0) / 60
            rows.append({'config': label, 'seed': seed, 'HO': f"{m['HO']:.2f}", 'Power': f"{m['Power']:.2f}",
                         'Spec': f"{m['Spec']:.2f}", 'RLF': f"{m['RLF']:.2f}", 'Tput': f"{m['Tput']:.1f}", 'train_min': f"{mins:.1f}"})
            _write(path, rows)
            print(f"[done] {label} seed {seed}: RLF {m['RLF']:.1f}% Spec {m['Spec']:.1f}% Pwr {m['Power']:.1f}% HO {m['HO']:.1f}% ({mins:.0f}m)", flush=True)
    # summary
    print("\n" + "=" * 70)
    for label in ['branched-binary (proposed)', 'branched-guarded-state (proposed 9dB)', 'branched-incremental (BDQ)']:
        v = {k: [float(x[k]) for x in rows if x['config'] == label] for k in ['HO', 'Power', 'Spec', 'RLF']}
        if v['RLF']:
            g = lambda a: (np.mean(a), np.std(a))
            print(f"  {label}: RLF {g(v['RLF'])[0]:.2f}±{g(v['RLF'])[1]:.2f}  Spec {g(v['Spec'])[0]:.2f}±{g(v['Spec'])[1]:.2f}  Power {g(v['Power'])[0]:.2f}±{g(v['Power'])[1]:.2f}")
    print("  (flat Dn-DQN multi-seed: RLF 11.47±4.18  Spec 88.36±0.91)")
    print("  -> results/branched_ablation_nyu.csv", flush=True)

if __name__ == "__main__":
    main()
