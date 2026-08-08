#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_srdndqn.py — SR-Dn-DQN: measurement-SHIELDED BACKUP for the branched joint controller.

NEW ALGORITHM (Paper 1 revision, answers Reviewer 4 + supervisor: a genuine learning mechanism).
Element B implemented here: the measured physical guard is folded into the BELLMAN BACKUP, not applied
post-hoc. In the Double-DQN target for the reliability branch, the unsafe action R_STAY_SAVE is masked
whenever the NEXT-state observed serving RSRP is below the measured save line (-76 dBm). The agent thus
LEARNS values inside the physically-safe region and (hypothesis H1) becomes reliable WITHOUT any runtime
guard. This is distinct from:
  * the post-hoc STATE guard (env_6g_branched guard_mode='state' — overrides power after the fact), and
  * action masking at selection (train_safe_dqn.py — changes behaviour only, not the learned value).

Arms (pre-registered ablation ladder A0/A1/A2):
  unguarded : no env guard,           backup shield OFF   (A0 floor)
  postguard : env state guard 9 dB,   backup shield OFF   (A1 = current RiE proposed baseline)
  shielded  : no env guard,           backup shield ON    (A2 = element B, the new mechanism)

Element A (cross-seed reliability-variance constraint) is added in a follow-up once A2 is validated.

Usage: python -u train_srdndqn.py <arm> <n_seeds> <steps>   e.g.  python -u train_srdndqn.py shielded 3 800000
       python -u train_srdndqn.py smoke   # quick self-check (all three arms, tiny)
Zero fabrication: every reported number traces to results/srdndqn_<arm>.csv.
"""
import sys, os, time, random, math, csv
from collections import deque
import numpy as np, torch, torch.nn as nn, torch.optim as optim
torch.set_num_threads(1)

from env_6g_branched import BranchedEnv6G
from dndqn_branched import (build_branched_networks, save_branched, load_branched,
                            R_STAY_SAVE)
from dndqn import NStepPERBuffer, polyak_update, epsilon_schedule, beta_schedule, OBS_DIM
from env_6g_v5 import W_MOB, W_PWR, W_SPEC, PT_MIN_DBM, PT_MAX_DBM, RB_MIN, RB_MAX, RLF_THRESH_DBM
import train_phase2 as tp
import multi_seed_eval as mse

GAIN, RB_BW_HZ, GAMMA = 15.0, 380.16e6 / RB_MAX, 0.99
EVAL_SEEDS, EVAL_EP = mse.SEEDS, mse.N_EPISODES
EPS_START, EPS_END, EPS_FRACTION, STAGE_SLOW_END = 1.0, 0.05, 0.30, 0.08
LEARNING_STARTS, TRAIN_FREQ, BATCH, TARGET_UPDATE, TAU = 10_000, 4, 256, 500, 0.01
PER_BETA0, PER_BETA1, NSTEP, ALPHA = 0.4, 1.0, 3, 0.6
GUARD_DB = 9.0
SAVE_LINE = RLF_THRESH_DBM + (PT_MAX_DBM - PT_MIN_DBM) + GUARD_DB   # -92 + 7 + 9 = -76 dBm (measured line)

ARMS = {  # arm -> (env_guard_db, backup_shield)
    'unguarded': (None, False),
    'postguard': (GUARD_DB, False),
    'shielded':  (None, True),
}


def set_seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s)


def shielded_branched_td_loss(online, target, ob, ab, rb, nb, db, wb, gamma_n, shield):
    """Branched Double-DQN loss. If shield=True, mask R_STAY_SAVE in the reliability-branch TARGET
    for next-states whose observed serving RSRP (nb[:,0]*100) is below SAVE_LINE. Element B."""
    ot = torch.FloatTensor(ob); nt = torch.FloatTensor(nb)
    at = torch.LongTensor(ab);  rt = torch.FloatTensor(rb)
    dt = torch.FloatTensor(db); wt = torch.FloatTensor(wb)
    q_online = online(ot)
    with torch.no_grad():
        q_next_online = online(nt)
        q_next_target = target(nt)
    # next-state RSRP in dBm for the shield mask
    next_rsrp = nt[:, 0] * 100.0                        # (B,)
    unsafe = (next_rsrp < SAVE_LINE)                    # (B,) bool — where saving is physically unsafe
    loss = 0.0
    td_accum = np.zeros(len(rb), dtype=np.float64)
    n_branches = len(q_online)
    for bi in range(n_branches):
        cq = q_online[bi].gather(1, at[:, bi:bi + 1]).squeeze(1)
        with torch.no_grad():
            qno = q_next_online[bi].clone()
            qnt = q_next_target[bi].clone()
            if shield and bi == 0 and unsafe.any():     # reliability branch only
                qno[unsafe, R_STAY_SAVE] = -1e9         # forbid unsafe SAVE in the online argmax ...
                qnt[unsafe, R_STAY_SAVE] = -1e9         # ... and in the target evaluation
            na = qno.argmax(1, keepdim=True)            # Double-DQN: (shielded) online picks
            nq = qnt.gather(1, na).squeeze(1)           # target evaluates
            tq = rt + gamma_n * nq * (1.0 - dt)
        loss = loss + (wt * nn.functional.huber_loss(cq, tq, reduction='none')).mean()
        td_accum += (cq - tq).detach().cpu().numpy()
    return loss / n_branches, np.abs(td_accum / n_branches)


CKPT_EVERY = 25_000       # save a resume checkpoint this often (insurance against container reclaim)


def train(seed, arm, out, slow, med, steps, wall=None):
    """Resumable, optionally wall-bounded training. Returns (done, minutes).
      done=True  -> reached `steps`, best-checkpoint saved, resume file removed.
      done=False -> hit the `wall` time budget; resume checkpoint saved for the next chunk.
    A <out>.resume file holds {online,target,opt,step,best,saved} so a killed/paused run continues
    mid-seed instead of restarting from zero."""
    env_guard_db, shield = ARMS[arm]
    set_seed(seed); pool = slow + med
    online, target = build_branched_networks(OBS_DIM)
    opt = optim.Adam(online.parameters(), lr=3e-4)
    buf = NStepPERBuffer(500_000, n_steps=NSTEP, gamma=GAMMA, alpha=ALPHA)  # buffer warm-restarts each chunk
    rng = np.random.default_rng(seed)
    resume = out + '.resume'; start_step = 1; best = -1e18; saved = False
    if os.path.exists(resume):
        ck = torch.load(resume, map_location='cpu')
        online.load_state_dict(ck['online']); target.load_state_dict(ck['target']); opt.load_state_dict(ck['opt'])
        start_step = int(ck['step']) + 1; best = float(ck['best']); saved = bool(ck['saved'])
        print(f"    [{arm} s{seed}] RESUME from step {start_step-1}", flush=True)
    def mk(stg):
        traj = (slow if stg == "SLOW" else pool)[rng.integers(len(slow if stg == "SLOW" else pool))]
        return BranchedEnv6G(traj, binary_power=True, guard_db=env_guard_db, guard_mode='state')
    stage = "SLOW"; env = mk(stage); obs, _ = env.reset()
    epr = deque(maxlen=20); er = 0.0; t0 = time.time()
    def _save(): save_branched(online, out, {'seed': seed, 'binary_power': True, 'arm': arm,
                                             'backup_shield': shield, 'save_line_dbm': SAVE_LINE,
                                             'algorithm': 'SR-Dn-DQN'})
    def _ckpt(step):
        torch.save({'online': online.state_dict(), 'target': target.state_dict(), 'opt': opt.state_dict(),
                    'step': step, 'best': best, 'saved': saved}, resume)
    for step in range(start_step, steps + 1):
        frac = step / steps; stage = "SLOW" if frac < STAGE_SLOW_END else "SLOW+MED"
        eps = epsilon_schedule(step, EPS_START, EPS_END, EPS_FRACTION, steps)
        beta = beta_schedule(step, PER_BETA0, PER_BETA1, steps)
        a = [random.randint(0, 2), random.randint(0, 2)] if random.random() < eps else online.act(obs)
        nobs, r, term, trunc, info = env.step(a); done = term or trunc; er += r
        buf.add(obs, a, r, nobs, term); obs = nobs
        if done:
            epr.append(er); er = 0.0; env = mk(stage); obs, _ = env.reset()
            if frac > EPS_FRACTION and len(epr) == epr.maxlen:
                m = float(np.mean(epr))
                if m > best: best = m; _save(); saved = True
        if step >= LEARNING_STARTS and step % TRAIN_FREQ == 0 and len(buf) >= BATCH:
            ib, wb, ob, ab, rb, nb, db = buf.sample(BATCH, beta)
            loss, td = shielded_branched_td_loss(online, target, ob, ab, rb, nb, db, wb,
                                                 gamma_n=GAMMA ** NSTEP, shield=shield)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(online.parameters(), 10.0); opt.step()
            buf.update_priorities(ib, td)
        if step % TARGET_UPDATE == 0: polyak_update(online, target, TAU)
        if step % CKPT_EVERY == 0:
            _ckpt(step)
            mr = float(np.mean(epr)) if epr else 0.0
            print(f"    [{arm} s{seed}] [{step:7d}/{steps}] {stage:<9} mean_rew={mr:8.2f} eps={eps:.3f} best={best:8.1f}", flush=True)
        if wall is not None and (step % 2000 == 0) and (time.time() - t0) > wall:
            _ckpt(step)
            print(f"    [{arm} s{seed}] WALL hit at step {step} -> checkpoint saved", flush=True)
            return False, (time.time() - t0) / 60.0
    if not saved: _save()
    if os.path.exists(resume): os.remove(resume)   # seed complete: clear resume marker
    return True, (time.time() - t0) / 60.0


def evaluate(net, pool, arm):
    """Eval the RAW learned policy. For the shielded arm NO runtime guard is applied — the whole point of
    element B is that the shield lives in training, so deployment is guard-free. postguard keeps its env
    state guard (it is part of that arm's controller)."""
    env_guard_db, _ = ARMS[arm]
    su = []; pw = []; sp = []; tpv = []; rl = []
    for s in EVAL_SEEDS:
        r = np.random.default_rng(s); real = good = nep = rlf = 0; P = []; S = []; T = []
        for _ in range(EVAL_EP):
            traj = pool[r.integers(len(pool))]
            env = BranchedEnv6G(traj, binary_power=True, guard_db=env_guard_db, guard_mode='state',
                                array_gain_dbi=GAIN, w_mob=W_MOB, w_pwr=W_PWR, w_spec=W_SPEC)
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
    return dict(HO=float(np.mean(su)), Power=float(np.mean(pw)), Spec=float(np.mean(sp)),
                Tput=float(np.mean(tpv)), RLF=float(np.mean(rl)), RLF_std=float(np.std(rl)))


ORDER = ['shielded', 'postguard', 'unguarded']   # arm priority within a seed


def _done_rows(arm):
    path = f'results/srdndqn_{arm}.csv'; done = set(); rows = []
    if os.path.exists(path):
        for row in csv.DictReader(open(path)): done.add(int(row['seed'])); rows.append(row)
    return path, done, rows


def _finalize_seed(arm, seed, ck, mins, pool):
    """Eval the completed seed and append its row to the arm CSV."""
    m = evaluate(load_branched(ck, OBS_DIM)[0], pool, arm)
    path, done, rows = _done_rows(arm)
    rows = [r for r in rows if int(r['seed']) != seed]
    rows.append({'seed': seed, 'HO': f"{m['HO']:.2f}", 'Power': f"{m['Power']:.2f}", 'Spec': f"{m['Spec']:.2f}",
                 'RLF': f"{m['RLF']:.2f}", 'Tput': f"{m['Tput']:.1f}", 'train_min': f"{mins:.1f}"})
    rows.sort(key=lambda r: int(r['seed']))
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['seed', 'HO', 'Power', 'Spec', 'RLF', 'Tput', 'train_min'])
        w.writeheader(); w.writerows(rows)
    print(f"[done] {arm} s{seed}: RLF {m['RLF']:.1f}% Pwr {m['Power']:.1f}% Spec {m['Spec']:.1f}% HO {m['HO']:.1f}% ({mins:.0f}m)", flush=True)


def run_arm(arm, n_seeds, steps, slow, med, pool):
    """Full synchronous run of one arm (used on the laptop, no wall)."""
    _, done, _ = _done_rows(arm)
    for seed in range(n_seeds):
        if seed in done: print(f"[skip] {arm} seed {seed}", flush=True); continue
        ck = f"srdndqn_{arm}_s{seed}.pt"; print(f"[train] {arm} seed {seed} -> {ck}", flush=True)
        _, mins = train(seed, arm, ck, slow, med, steps)
        _finalize_seed(arm, seed, ck, mins, pool)


def run_chunk(wall, n_seeds, steps, slow, med, pool):
    """Advance the seed-major work list by up to `wall` seconds (one seed per call). Cloud-safe:
    resumes mid-seed from the .resume checkpoint. Prints ALL_DONE when the whole ablation is finished."""
    os.makedirs('results', exist_ok=True)
    dones = {a: _done_rows(a)[1] for a in ORDER}
    for seed in range(n_seeds):                       # seed-major: s0 of every arm first
        for arm in ORDER:
            if seed in dones[arm]:
                continue
            ck = f"srdndqn_{arm}_s{seed}.pt"
            print(f"[chunk] advancing {arm} seed {seed} (wall {wall}s)", flush=True)
            finished, mins = train(seed, arm, ck, slow, med, steps, wall=wall)
            if finished:
                _finalize_seed(arm, seed, ck, mins, pool)
            else:
                print(f"[chunk] {arm} s{seed} paused (checkpointed); rerun to continue", flush=True)
            return   # one seed advanced per chunk; heartbeat calls us again
    print("ALL_DONE", flush=True)
    for arm in ORDER:
        _, _, rows = _done_rows(arm); rlfs = [float(r['RLF']) for r in rows]
        if rlfs:
            print(f"  {arm}: RLF {np.mean(rlfs):.2f} ± {np.std(rlfs):.2f}  Pwr {np.mean([float(r['Power']) for r in rows]):.1f}%  (n={len(rlfs)})", flush=True)
    print("  refs: flat 11.47±4.18 | branched-guarded 4.61 | StdDQN 19.29", flush=True)


def main():
    slow = med = pool = None
    def _load():
        nonlocal slow, med, pool
        slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH); pool = slow + med
    if len(sys.argv) > 1 and sys.argv[1] == 'smoke':
        _load(); print(f"SMOKE | pool={len(pool)} | SAVE_LINE={SAVE_LINE} dBm", flush=True)
        for arm in ['unguarded', 'postguard', 'shielded']:
            t0 = time.time(); train(0, arm, f'smoke_{arm}.pt', slow, med, steps=6000)
            m = evaluate(load_branched(f'smoke_{arm}.pt', OBS_DIM)[0], pool, arm)
            print(f"  [{arm}] RLF {m['RLF']:.1f}% Pwr {m['Power']:.1f}% HO {m['HO']:.1f}%  ({(time.time()-t0):.0f}s)", flush=True)
        print("SMOKE OK", flush=True); return
    if len(sys.argv) > 1 and sys.argv[1] == 'chunk':
        wall = int(sys.argv[2]) if len(sys.argv) > 2 else 540
        n_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 3
        steps = int(sys.argv[4]) if len(sys.argv) > 4 else 800_000
        _load(); print(f"CHUNK | wall={wall}s | target {n_seeds}x{steps:,} | pool={len(pool)}", flush=True)
        run_chunk(wall, n_seeds, steps, slow, med, pool); return
    arm = sys.argv[1] if len(sys.argv) > 1 else 'shielded'
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 800_000
    _load(); print(f"SR-Dn-DQN | arm={arm} | {n_seeds}x{steps:,} | SAVE_LINE={SAVE_LINE} dBm | pool={len(pool)}", flush=True)
    run_arm(arm, n_seeds, steps, slow, med, pool)


if __name__ == "__main__":
    main()
