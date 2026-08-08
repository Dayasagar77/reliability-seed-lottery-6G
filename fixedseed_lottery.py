#!/usr/bin/env python3
"""
fixedseed_lottery.py -- is it a SEED lottery, or a RETRAINING lottery?

THE FINDING THAT PROMPTED THIS
------------------------------
env_6g_v5.py:132 builds the channel RNG as np.random.default_rng(seed). Both trainers construct
their environment WITHOUT a seed -- run_multiseed_rel.py:43 `Phase2Env_NYU_rel(smp(stage))` and
train_branched_nyu.py:39 `BranchedEnv6G(smp(stage), ...)` -- so seed is None and every training
episode draws its shadow fading, its small-scale fading and its trajectory start point from fresh
OS entropy. set_seed(s) controls python-random (the epsilon-greedy draw), the legacy numpy global,
torch (network init), and `rng = default_rng(seed)` controls WHICH trajectory is sampled. It does
not control the CHANNEL THE AGENT TRAINS AGAINST.

Measured consequence, already confirmed: two trainings of the guarded arm at seed 0 in this
container, same config, same budget, produce DIFFERENT WEIGHTS (18 tensors, differing md5).

WHY THAT MATTERS FOR THE MANUSCRIPT
-----------------------------------
The abstract says the 4.32-14.55% spread is "produced by nothing but the seed", and Section VI
says "only the random seed changes". On the evidence above, neither is true as written: the seed
changes AND every channel realisation the agent ever sees is an independent unseeded draw.

This does NOT threaten the paper's thesis. sigma_train is estimated across five INDEPENDENT
TRAININGS and eight CRN evaluation blocks; the decomposition is unaffected by which of the
training-time entropy sources drives the dispersion. What is wrong is the CAUSAL LABEL. And if
the seed is a label rather than a cause, the practitioner-facing message gets STRONGER, not
weaker: you cannot buy reproducibility by fixing the seed.

THE EXPERIMENT
--------------
Train the flat controller K times at THE SAME SEED (0), identical config, identical 8e5 budget.
Evaluate each under the published R = 8 CRN protocol at the identical channel seeds
(BASE = 20260801), reusing the published evaluator unmodified via tablevii_r8's patch.

    If sigma_between(fixed seed) is comparable to the published sigma_train(varying seed),
    the seed is a label and the lottery is a RETRAINING lottery.

    If sigma_between(fixed seed) is near zero, the seed really is the cause and the
    manuscript's wording stands.

Either way the answer is reportable and the experiment is cheap. Nothing here is assumed.

VENUE. This container is cloud hardware running the LAPTOP toolchain (torch 2.5.1, numpy 2.4.4).
Evaluation here is already proven bit-exact against the laptop grid: 40/40 cells, guarded seed 0.
Training reproducibility is exactly what is in question, so nothing in this run is compared
against a laptop-trained checkpoint -- every arm of the comparison is trained HERE.
"""
import csv
import hashlib
import os
import subprocess
import sys
import time

K = int(os.environ.get('FS_K', 5))
STEPS = int(os.environ.get('FS_STEPS', 800_000))
R = int(os.environ.get('FS_R', 8))
BASE = 20260801
SEED = 0
VENUE = 'cloud-hw-torch2.5.1'
OUT = 'results/fixedseed'
METRICS = ['HO', 'Power', 'Spec', 'RLF', 'Tput']


def log(m):
    print(time.strftime('%Y-%m-%d %H:%M:%S') + ' | ' + m, flush=True)


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def ck_for(k):
    return 'fs_flat_r%d.pt' % k


# ------------------------------------------------------------------ worker: train one replicate
def train_one(k):
    saved = sys.argv
    sys.argv = [saved[0], '1', str(STEPS)]
    import torch
    torch.set_num_threads(1)
    import run_multiseed_rel as F
    import train_phase2 as tp
    sys.argv = saved
    slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH)
    out = ck_for(k)
    log('[train r%d] seed %d (FIXED) | %s steps | pool %d -> %s'
        % (k, SEED, format(STEPS, ','), len(slow) + len(med), out))
    t0 = time.time()
    F.train(SEED, out, slow, med)
    log('[train r%d] done in %.1f min | md5 %s' % (k, (time.time() - t0) / 60, md5(out)))
    return 0


# ------------------------------------------------------------------ worker: evaluate one
def eval_one(k):
    saved = sys.argv
    sys.argv = [saved[0]]
    import torch
    torch.set_num_threads(1)
    import numpy as np
    import env_6g_v5
    import train_phase2 as tp
    import run_multiseed_rel as F
    from dndqn import load_dndqn_checkpoint
    import tablevii_r8 as T7
    sys.argv = saved

    T7.install_patch(env_6g_v5)
    ck = ck_for(k)
    net = load_dndqn_checkpoint(ck)[0]
    slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH)
    pool = slow + med
    os.makedirs(OUT, exist_ok=True)
    rows = []
    log('[eval r%d] %s (md5 %s) | R=%d | %d eval seeds x %d episodes | pool %d'
        % (k, ck, md5(ck), R, len(F.EVAL_SEEDS), F.EVAL_EP, len(pool)))
    for r in range(R):
        t0 = time.time()
        with T7.seeded(r):
            m = F.evaluate(net, pool)
        row = {'replicate': k, 'seed_declared': SEED, 'ckpt': ck, 'ckpt_md5': md5(ck),
               'repeat': r, 'chan_seed_first': BASE + 100000 * r,
               'secs': round(time.time() - t0, 1), 'venue': VENUE,
               'python': sys.version.split()[0], 'torch': torch.__version__,
               'numpy': np.__version__, 'steps': STEPS,
               'env_v5_md5': md5('env_6g_v5.py'), 'driver_md5': md5('run_multiseed_rel.py'),
               'measured_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
        for kk in METRICS:
            row[kk] = '%.4f' % float(m[kk])
        rows.append(row)
        log('[eval r%d] rep%d | ' % (k, r) + ' '.join('%s %8.3f' % (x, float(row[x]))
                                                      for x in METRICS) + ' | %.0fs' % row['secs'])
        p = os.path.join(OUT, 'unit_fs_r%d.csv' % k)
        with open(p + '.tmp', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        os.replace(p + '.tmp', p)
    return 0


# ------------------------------------------------------------------ orchestrator
def main():
    if len(sys.argv) > 2 and sys.argv[1] == '--train-one':
        return train_one(int(sys.argv[2]))
    if len(sys.argv) > 2 and sys.argv[1] == '--eval-one':
        return eval_one(int(sys.argv[2]))

    log('=' * 96)
    log('FIXED-SEED REPLICATION | K=%d trainings, ALL at seed %d | %s steps | R=%d'
        % (K, SEED, format(STEPS, ','), R))
    log('venue %s | this asks whether the seed is a CAUSE or a LABEL' % VENUE)
    log('=' * 96)

    slots = 2
    todo = [k for k in range(K) if not os.path.exists(ck_for(k))]
    log('training %d replicate(s) %s at %d slot(s)' % (len(todo), todo, slots))
    running = []
    while todo or running:
        while todo and len(running) < slots:
            k = todo.pop(0)
            f = open('fs_train_r%d.log' % k, 'w')
            p = subprocess.Popen([sys.executable, '-u', __file__, '--train-one', str(k)],
                                 stdout=f, stderr=subprocess.STDOUT)
            running.append((k, p, f))
            log('  launched replicate %d (pid %d)' % (k, p.pid))
        time.sleep(10)
        for item in list(running):
            k, p, f = item
            if p.poll() is not None:
                f.close()
                running.remove(item)
                log('  replicate %d exited rc=%d %s'
                    % (k, p.returncode, 'md5 ' + md5(ck_for(k)) if os.path.exists(ck_for(k))
                       else 'NO CHECKPOINT'))

    sigs = {k: md5(ck_for(k)) for k in range(K) if os.path.exists(ck_for(k))}
    log('checkpoint md5s: %s' % sigs)
    log('distinct checkpoints: %d of %d' % (len(set(sigs.values())), len(sigs)))

    for k in range(K):
        if os.path.exists(os.path.join(OUT, 'unit_fs_r%d.csv' % k)):
            log('[eval r%d] already present — skipping' % k)
            continue
        rc = subprocess.call([sys.executable, '-u', __file__, '--eval-one', str(k)])
        if rc:
            log('eval replicate %d FAILED rc=%d' % (k, rc))
            return rc

    merged = os.path.join(OUT, 'fixedseed_flat_%s.csv' % VENUE)
    allrows = []
    for k in range(K):
        p = os.path.join(OUT, 'unit_fs_r%d.csv' % k)
        if os.path.exists(p):
            allrows += list(csv.DictReader(open(p)))
    if allrows:
        with open(merged, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(allrows[0].keys()))
            w.writeheader()
            w.writerows(allrows)
        log('merged %d rows -> %s' % (len(allrows), merged))
    log('DONE')
    return 0


if __name__ == '__main__':
    sys.exit(main())
