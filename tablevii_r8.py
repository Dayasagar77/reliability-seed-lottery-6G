#!/usr/bin/env python3
"""
tablevii_r8.py — Architecture 2: re-measure ALL FOUR learned arms of Table VII at R = 8 common
random numbers, ON THE LAPTOP, so that the whole paper rests on one venue and one protocol.

WHY THIS EXISTS
---------------
Two separate defects converge on this one run.

DEFECT 1 — the evaluation channel RNG is unseeded. env_6g_v5.py:132 builds
np.random.default_rng(seed) with seed=None, so every evaluation draws fresh OS entropy. The
consequence is that the +- printed in Table VII is not sigma_train; it is

    sigma_observed^2 = sigma_train^2 + sigma_eval^2

and any re-evaluation of the same frozen checkpoint is a fresh draw. Averaging R = 8 seeded
repeats per checkpoint before taking sigma across training seeds strips sigma_eval out of the
error bar, which is precisely what the Associate Editor's criticism (b) demands:

    "some baselines are represented by a single trained model and others by evaluation-seed
     variability, making the comparisons methodologically inconsistent"

DEFECT 2 — and this is the one that turns an 80-evaluation job into a 160-evaluation job. Every
R = 8 measurement in the project so far was produced in the CLOUD under torch 2.13.0+cu130
(_equiv/out/vardecomp.log line 1, _equiv/out/seeded.log line 1). Every number in Table VII and
Table X was produced on the LAPTOP under torch 2.5.1+cu121. TRAINING is already proven
non-portable between those venues: the same seed, config and 800k budget give handover 93.00 vs
91.04 and power 18.96 vs 24.52 on seed 0 of the flat arm.

_equiv/venue_portability.py asked whether EVALUATION of an already-frozen checkpoint is portable.
It passed: 50 of 50 cells inside their prediction intervals, no metric-level offset surviving
Holm. But it passed at n = 5 checkpoints per arm, which cannot resolve a small offset, and the
flat arm leans the same way on three metrics at raw p < 0.08 (Tput +7.75 Mbit/s, RLF +1.26,
Spec -0.28). Passing a test that is underpowered against the thing you are worried about is not
the same as the thing not being there.

The governing asymmetry: A VENUE OFFSET CANCELS OUT OF A STANDARD DEVIATION AND DOES NOT CANCEL
OUT OF A MEAN. A constant added to all five training seeds leaves sigma exactly unchanged.
Section IX-B reports sigma, ratios of sigma, and bounds on ratios of sigma — structurally immune.
Table VII reports MEANS — not immune. So the existing cloud R = 8 data cannot supply Table VII
rows 1 and 4 even though it exists, and all four arms must be re-measured here.

    4 learned arms x 5 training seeds x 8 CRN repeats = 160 evaluations, one venue, one protocol.

WHAT THIS RUN IS NOT
--------------------
It is not a re-training. Every checkpoint it reads was trained on this laptop in July under the
published budget and is unmodified. It is not a new evaluation loop either: the loops that
compute HO / Power / Spec / RLF / Tput are the published ones, called unmodified —
train_branched_nyu.evaluate_branched for the three branched arms and run_multiseed_rel.evaluate
for the flat arm. The ONLY behavioural difference from the published protocol is that reset()
now receives a channel seed it previously did not.

THE SEED SCHEME (publishable verbatim)
--------------------------------------
    chan_seed = BASE + 100000 * repeat + episode_counter
    BASE = 20260801, episode_counter = 0, 1, 2, ... in evaluation order within one repeat.

Trajectory selection is untouched: every evaluator still draws trajectories with
np.random.default_rng(s) over EVAL_SEEDS exactly as published. Repeat r is therefore reproducible
bit-for-bit, and different r give independent channel realisations. Because repeat r uses the same
channel seeds for EVERY arm and EVERY training seed, the design is CROSSED (common random
numbers), not nested — which is why the analysis below reports a two-way decomposition and why a
naive paired t-test across training seeds would be invalid on this data.

The patch block below is character-identical to the one in _equiv/seeded_eval.py that produced the
cloud data, and --verify-patch asserts that mechanically whenever that file is present. The two
runs must differ only in venue; if they differ in harness as well, nothing can be concluded from
comparing them.

VENUE GUARD
-----------
Refuses to run on a torch major.minor other than EXPECT_TORCH unless --allow-toolchain-mismatch is
passed, so a stray cloud invocation cannot quietly put two toolchains inside one table. The whole
point of this run is the venue.

SEQUENCING
----------
Waits for BOTH:
  * laptop_campaign.status  -> every one of the 40 revision-campaign trainings finished (RAM), and
  * bdq_row2_repair.status  -> phase=COMPLETE, because that run PRODUCES three of the sixteen
    checkpoints this one needs (br_inc_regen_s{2,3,4}.pt).
Then preflights all 20 checkpoints and refuses to launch anything unless every one is on disk. A
partial table is worse than no table.

OUTPUTS
-------
    results/tablevii_r8/unit_{arm}_s{seed}_{venue}.csv   R rows, written by its own worker
    results/tablevii_r8_{venue}.csv                      all 160 rows merged
    results/tablevii_r8_summary_{venue}.json             the four rows + the decomposition
    tablevii_r8.status                                   heartbeat, overwritten each poll
    tablevii_r8.log                                      append-only event log
    logs/t7r8_{arm}_s{seed}.log                          per-worker stdout

REHEARSAL KNOBS
---------------
T7_TAG renames every output path, so a rehearsal physically cannot write into a real result file.
T7_R and T7_EVAL_EP shrink the run. The real run sets none of them and asserts so.

USAGE (PowerShell, from the folder holding the drivers)
    python -u tablevii_r8.py                 # wait for campaign + BDQ, then all 160
    python -u tablevii_r8.py --dry-run       # print the plan, launch nothing
    python -u tablevii_r8.py --no-wait       # start immediately
    python -u tablevii_r8.py --verify-patch  # prove the CRN patch matches the cloud harness
    python -u tablevii_r8.py --table         # recompute the rows from whatever units exist
Resume: rerun the same command. A unit is done when its CSV holds R rows.
"""
import os, sys

# One thread per process, set before torch is imported anywhere in this interpreter.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import argparse, csv, hashlib, json, math, subprocess, time

# VENUE is a LABEL that ends up in every filename and every CSV row, so it must be derived from
# evidence, not asserted. The env var is only a default; check_toolchain() overrides it from the
# observed torch version, because the toolchain is the thing that actually differs between the two
# machines and it cannot be got wrong by forgetting to export a variable.
#   torch 2.5.x  -> laptop  (Python 3.11.9,  torch 2.5.1+cu121,  20 cores)  <- the reporting venue
#   torch 2.13.x -> cloud   (Python 3.11.15, torch 2.13.0+cu130, 2 cores)   <- the replication arm
VENUE = os.environ.get('T7_VENUE', 'laptop')
TORCH_VENUE = {'2.5': 'laptop', '2.13': 'cloud'}
BASE = 20260801
EXPECT_TORCH = '2.5'
R_REAL = 8
SEEDS = [0, 1, 2, 3, 4]
METRICS = ['HO', 'Power', 'Spec', 'RLF', 'Tput']

# ---- REHEARSAL KNOBS -------------------------------------------------------------------------
TAG = os.environ.get('T7_TAG', '')                        # '' for the real run
R = int(os.environ.get('T7_R', R_REAL))                   # 8 for the real run
EVAL_EP_OVERRIDE = int(os.environ.get('T7_EVAL_EP', 0))   # 0 = use the published EVAL_EP
REHEARSAL = bool(TAG) or R != R_REAL or bool(EVAL_EP_OVERRIDE)
_SFX = f'_{TAG}' if TAG else ''

# ---- the four learned arms of Table VII -------------------------------------------------------
# key -> (manuscript row label, binary_power, guard_db, guard_mode, evaluator family)
ARMS = {
    'flat':        ('Flat Dn-DQN (6-action)',                None, None, None,    'flat'),
    'incremental': ('Branched incremental (BDQ-style)',      False, None, 'state', 'branched'),
    'binary':      ('Branched binary, unguarded',            True,  None, 'state', 'branched'),
    'guarded':     ('Branched binary + state guard (proposed)', True, 9.0, 'state', 'branched'),
}
ARM_ORDER = ['flat', 'incremental', 'binary', 'guarded']

# published Table VII, transcribed from the manuscript (mean, pop sd, decimals) for comparison only
PUBLISHED = {
    'flat':        {'HO': (92.40, 2.02, 2), 'Power': (27.37, 18.28, 2), 'Spec': (88.36, 0.91, 2),
                    'RLF': (11.47, 4.18, 2), 'Tput': (479.5, 16.5, 1)},
    'incremental': {'HO': (93.15, 3.30, 2), 'Power': (26.88, 15.20, 2), 'Spec': (54.99, 29.94, 2),
                    'RLF': (9.04, 4.23, 2), 'Tput': (981.6, 445.5, 1)},
    # CORRECTED 2026-08-01. The five values previously on these two lines
    #   HO 94.29 +- 1.15 | Power 61.35 +- 6.02 | Spec 78.71 +- 18.32 | RLF 7.32 +- 3.55 | Tput 600.3 +- 260.5
    # were a drafting error, not a transcription of anything. A grep across every manuscript file in
    # /root/work/v13 (DnDQN_manuscript_v13.md, springer.md, springer_r1.md, v13.md, v13_raw.md,
    # v13r1.md, v13r1_raw.md, v13r1_ai.md) finds NONE of them anywhere -- they are not this row, not
    # another row, and not another table. The published row, identical in all eight files, is below.
    # Impact was display-only: PUBLISHED feeds `published_mean`/`published_sd` and the printed
    # "shift" column only (see the summary dict near line 630). Every measured quantity -- mu, sd,
    # se, per_seed -- derives from the merged CSV and was never touched by it.
    'binary':      {'HO': (89.11, 3.88, 2), 'Power': (68.14, 19.43, 2), 'Spec': (72.36, 28.09, 2),
                    'RLF': (18.59, 6.06, 2), 'Tput': (599.6, 325.2, 1)},
    'guarded':     {'HO': (95.18, 0.45, 2), 'Power': (63.61, 4.38, 2), 'Spec': (80.63, 17.02, 2),
                    'RLF': (4.61, 0.92, 2), 'Tput': (505.9, 223.5, 1)},
}

FIELDS = ['arm', 'label', 'seed', 'ckpt', 'ckpt_origin', 'ckpt_md5', 'repeat', 'chan_seed_first',
          'secs'] + METRICS + ['venue', 'python', 'torch', 'numpy', 'binary_power', 'guard_db',
          'guard_mode', 'eval_seeds', 'eval_ep', 'env_v5_md5', 'env_branched_md5', 'driver_md5',
          'measured_utc']

OUTDIR = f'results/tablevii_r8{_SFX}'


# Any path carrying the VENUE label must be built at CALL time, never at import time. VENUE starts
# from an env var and is CORRECTED by check_toolchain() from the observed torch version, so a
# module-level f-string freezes the unverified default into the filename. That is exactly how a
# table of cloud measurements lands in a file named '..._laptop.csv' while its own unit CSVs are
# correctly named '..._cloud.csv' — the precise venue mislabel this architecture exists to prevent.
# unit_path() was already a function and was therefore correct; these two were not.
def merged_path():
    return f'results/tablevii_r8{_SFX}_{VENUE}.csv'


def summary_path():
    return f'results/tablevii_r8{_SFX}_summary_{VENUE}.json'


STATUS = f'tablevii_r8{_SFX}.status'
LOG = f'tablevii_r8{_SFX}.log'


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
    print(line, flush=True)
    try:
        with open(LOG, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    except OSError:
        pass


def md5(path):
    if not path or not os.path.exists(path):
        return ''
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def ckpt_for(arm, seed):
    """The incremental arm is split: seeds 0 and 1 are the surviving 2026-07-12 originals, seeds
    2-4 were destroyed and are regenerated by bdq_row2_repair.py under a DIFFERENT filename so the
    two provenances stay distinguishable in the release archive. Prefer the original; fall back."""
    if arm == 'flat':
        return f'ms_rel_s{seed}.pt', 'original-20260711'
    if arm == 'binary':
        return f'br_bin_s{seed}.pt', 'original-20260712'
    if arm == 'guarded':
        return f'br_gst_s{seed}.pt', 'original-20260712'
    orig = f'br_inc_s{seed}.pt'
    if os.path.exists(orig):
        return orig, 'original-20260712'
    return f'br_inc_regen_s{seed}.pt', 'regenerated-20260801'


def unit_path(arm, seed):
    return os.path.join(OUTDIR, f'unit_{arm}_s{seed}_{VENUE}.csv')


def unit_done(arm, seed):
    p = unit_path(arm, seed)
    if not os.path.exists(p):
        return False
    try:
        return sum(1 for _ in csv.DictReader(open(p, newline=''))) >= R
    except OSError:
        return False


def avail_mb():
    """Windows via GlobalMemoryStatusEx, Linux via /proc/meminfo, so the orchestration can be
    rehearsed in the cloud container before it is trusted with the real run."""
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes

        class MS(ctypes.Structure):
            _fields_ = [('dwLength', wintypes.DWORD), ('dwMemoryLoad', wintypes.DWORD),
                        ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                        ('ullTotalPageFile', ctypes.c_ulonglong),
                        ('ullAvailPageFile', ctypes.c_ulonglong),
                        ('ullTotalVirtual', ctypes.c_ulonglong),
                        ('ullAvailVirtual', ctypes.c_ulonglong),
                        ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
        st = MS()
        st.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        return st.ullAvailPhys // (1024 * 1024)
    try:
        for line in open('/proc/meminfo'):
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 1 << 30


def status_kv(path):
    """Parse the first line of a 'k=v k=v' status file. None when absent or unparseable."""
    if not os.path.exists(path):
        return None
    try:
        head = open(path, encoding='utf-8').readline()
    except OSError:
        return None
    kv = {'_raw': head.strip(), '_mtime': os.path.getmtime(path)}
    for tok in head.split():
        if '=' in tok:
            k, v = tok.split('=', 1)
            kv[k] = v
    return kv


# --------------------------------------------------------------------------------------------
# the CRN patch  (character-identical to _equiv/seeded_eval.py; see --verify-patch)
# --------------------------------------------------------------------------------------------
def import_stack():
    """Every driver module parses sys.argv AT IMPORT TIME (run_multiseed_rel:27,
    train_branched_nyu:22). Importing them with this script's argv would crash on int('--unit') or
    silently reset a step budget. Neutralise argv across the import, then restore it."""
    saved = sys.argv
    sys.argv = [saved[0]]
    try:
        import torch
        torch.set_num_threads(1)
        import numpy as np
        import env_6g_v5
        import train_phase2 as tp
        from dndqn import OBS_DIM, load_dndqn_checkpoint
        from dndqn_branched import load_branched
        import train_branched_nyu as B
        import run_multiseed_rel as F
        return dict(torch=torch, np=np, env_6g_v5=env_6g_v5, tp=tp, OBS_DIM=OBS_DIM,
                    load_dndqn_checkpoint=load_dndqn_checkpoint, load_branched=load_branched,
                    B=B, F=F)
    finally:
        sys.argv = saved


# ----------------------------------------------------------------- the patch
_BASE_CLS = None
_ORIG_RESET = None
_FEED = {'on': False, 'repeat': 0, 'n': 0}


def install_patch(env_6g_v5):
    global _BASE_CLS, _ORIG_RESET
    if _BASE_CLS is not None:
        return
    _BASE_CLS = env_6g_v5.SubTHzEnv6G
    _ORIG_RESET = _BASE_CLS.reset

    def _patched_reset(self, seed=None, options=None):
        if seed is None and _FEED['on']:
            seed = BASE + 100000 * _FEED['repeat'] + _FEED['n']
            _FEED['n'] += 1
        return _ORIG_RESET(self, seed=seed, options=options)

    _BASE_CLS.reset = _patched_reset


class seeded:
    """context manager: inside it, every unseeded reset() gets a deterministic seed."""
    def __init__(self, repeat):
        self.repeat = repeat

    def __enter__(self):
        _FEED.update(on=True, repeat=self.repeat, n=0)
        return self

    def __exit__(self, *a):
        _FEED.update(on=False, n=0)
        return False
# ----------------------------------------------------------------- end of patch


PATCH_SEMANTICS = "seed = BASE + 100000 * repeat + n, applied only when reset() is called with seed=None"


def verify_patch(path='_equiv/seeded_eval.py'):
    """Prove mechanically that this file's CRN rule is the same rule that produced the cloud data.

    The two files cannot be byte-compared (this one installs the patch inside a function so the
    worker controls import order; seeded_eval.py patches at module scope). What must match is the
    ARITHMETIC and the CONDITION, so those two lines are compared after whitespace normalisation,
    together with BASE. Anything else differing is a structural difference that does not change a
    single channel seed."""
    if not os.path.exists(path):
        log(f"verify-patch | {path} not present on this venue — skipping "
            f"(it is the cloud-side harness; the check is meaningful only where both files exist)")
        return None
    src = open(path, encoding='utf-8').read()
    want = [
        "seed = BASE + 100000 * _FEED['repeat'] + _FEED['n']",
        "if seed is None and _FEED['on']:",
        "_FEED['n'] += 1",
        "return _ORIG_RESET(self, seed=seed, options=options)",
    ]
    mine = open(os.path.abspath(__file__), encoding='utf-8').read()
    ok = True
    for w in want:
        a, b = w in src, w in mine
        log(f"verify-patch | {'MATCH  ' if (a and b) else 'MISMATCH'} | seeded_eval={a} this={b} | {w}")
        ok = ok and a and b
    import re
    m = re.search(r'^BASE\s*=\s*(\d+)', src, re.M)
    base_ok = bool(m) and int(m.group(1)) == BASE
    log(f"verify-patch | {'MATCH  ' if base_ok else 'MISMATCH'} | BASE seeded_eval="
        f"{m.group(1) if m else '?'} this={BASE}")
    ok = ok and base_ok
    log(f"verify-patch | {'PASS — same CRN rule as the cloud run' if ok else 'FAIL — the two harnesses differ'}")
    return ok


# --------------------------------------------------------------------------------------------
# worker: one (arm, seed) unit = R seeded repeats of one checkpoint
# --------------------------------------------------------------------------------------------
def run_unit(arm, seed):
    S = import_stack()
    install_patch(S['env_6g_v5'])
    torch, B, F, tp = S['torch'], S['B'], S['F'], S['tp']
    label, bp, gdb, gmode, family = ARMS[arm]
    ck, origin = ckpt_for(arm, seed)
    if not os.path.exists(ck):
        log(f"[{arm} s{seed}] MISSING CHECKPOINT {ck} — refusing")
        return 3
    if EVAL_EP_OVERRIDE:
        B.EVAL_EP = EVAL_EP_OVERRIDE          # rehearsal only
        F.N_EPISODES = EVAL_EP_OVERRIDE
    slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH)
    pool = slow + med
    tc = dict(python=sys.version.split()[0], torch=torch.__version__, numpy=S['np'].__version__)

    if family == 'flat':
        net = S['load_dndqn_checkpoint'](ck)[0]

        def evaluate():
            return F.evaluate(net, pool)
        n_eval_seeds, n_ep = len(B.EVAL_SEEDS), B.EVAL_EP
    else:
        net = S['load_branched'](ck, S['OBS_DIM'])[0]

        def evaluate():
            return B.evaluate_branched(net, pool, bp, guard_db=gdb, guard_mode=gmode)
        n_eval_seeds, n_ep = len(B.EVAL_SEEDS), B.EVAL_EP

    log(f"[{arm} s{seed}] start | {ck} ({origin}, md5 {md5(ck)}) | R={R} repeats "
        f"| {n_eval_seeds} eval seeds x {n_ep} episodes | torch {tc['torch']} "
        f"| threads {torch.get_num_threads()} | pool {len(pool)}")

    os.makedirs(OUTDIR, exist_ok=True)
    rows = []
    for r in range(R):
        t0 = time.time()
        with seeded(r):
            m = evaluate()
        row = {'arm': arm, 'label': label, 'seed': seed, 'ckpt': ck, 'ckpt_origin': origin,
               'ckpt_md5': md5(ck), 'repeat': r, 'chan_seed_first': BASE + 100000 * r,
               'secs': round(time.time() - t0, 1), 'venue': VENUE,
               'python': tc['python'], 'torch': tc['torch'], 'numpy': tc['numpy'],
               'binary_power': bp, 'guard_db': gdb, 'guard_mode': gmode,
               'eval_seeds': n_eval_seeds, 'eval_ep': n_ep,
               'env_v5_md5': md5('env_6g_v5.py'), 'env_branched_md5': md5('env_6g_branched.py'),
               'driver_md5': md5('train_branched_nyu.py' if family != 'flat' else 'run_multiseed_rel.py'),
               'measured_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
        for k in METRICS:
            row[k] = f"{float(m[k]):.4f}"
        rows.append(row)
        log(f"[{arm} s{seed}] r{r} | " + " ".join(f"{k} {float(row[k]):8.3f}" for k in METRICS)
            + f" | {row['secs']:.0f}s")
        tmp = unit_path(arm, seed) + '.tmp'
        with open(tmp, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, unit_path(arm, seed))
    log(f"[{arm} s{seed}] done | {R} repeats -> {unit_path(arm, seed)}")
    return 0


# --------------------------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------------------------
def write_status(phase, done, total, running, t0, extra=''):
    txt = (f"{time.strftime('%Y-%m-%d %H:%M:%S')} phase={phase} done={done}/{total} "
           f"running={len(running)} free_mb={avail_mb()} "
           f"elapsed_min={(time.time() - t0) / 60:.1f}{(' ' + extra) if extra else ''}")
    for name, rec in running.items():
        p, started = rec[0], rec[1]
        txt += f"\n  {name} pid {p.pid} {(time.time() - started) / 60:.1f}m"
    try:
        with open(STATUS, 'w', encoding='utf-8') as f:
            f.write(txt + '\n')
    except OSError:
        pass


def run_pool(units, slots, floor_mb, t0, poll=20, extra_argv=()):
    env = dict(os.environ)
    env['OMP_NUM_THREADS'] = env['MKL_NUM_THREADS'] = '1'
    env['PYTHONPATH'] = os.getcwd() + os.pathsep + env.get('PYTHONPATH', '')
    env['T7_VENUE'] = VENUE
    for k in ('T7_TAG', 'T7_R', 'T7_EVAL_EP'):
        if os.environ.get(k):
            env[k] = os.environ[k]
    os.makedirs('logs', exist_ok=True)
    queue = list(units)
    running, failed, done = {}, [], 0
    total = len(queue)
    while queue or running:
        for name in list(running):
            p, started, fh, (arm, seed) = running[name]
            if p.poll() is None:
                continue
            fh.close()
            mins = (time.time() - started) / 60
            ok = p.returncode == 0 and unit_done(arm, seed)
            if ok:
                done += 1
                log(f"DONE  {name} rc={p.returncode} in {mins:.1f}m ({done}/{total})")
            else:
                failed.append((arm, seed))
                why = (f"exit code {p.returncode}" if p.returncode != 0 else
                       f"exited 0 but {unit_path(arm, seed)} is absent — the worker wrote its "
                       f"result somewhere else, which almost always means the two processes "
                       f"disagree about the venue label")
                log(f"FAIL  {name} after {mins:.1f}m: {why} — see logs/{name}.log")
            del running[name]
        while queue and len(running) < slots:
            free = avail_mb()
            if free < floor_mb:
                log(f"HOLD  {free} MB free < floor {floor_mb} MB "
                    f"({len(running)} running, {len(queue)} queued)")
                break
            arm, seed = queue.pop(0)
            name = f't7r8_{arm}_s{seed}'
            fh = open(os.path.join('logs', name + '.log'), 'a', encoding='utf-8')
            cmd = ([sys.executable, '-u', os.path.abspath(__file__), '--unit', arm, str(seed)]
                   + list(extra_argv))
            p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=os.getcwd())
            running[name] = (p, time.time(), fh, (arm, seed))
            log(f"START {name} pid {p.pid} ({free} MB free)")
        write_status('evaluate', done, total, running, t0)
        if queue or running:
            time.sleep(poll)
    return failed


def wait_for(t0, poll=60, max_hours=14, probe=False):
    """Hold until the 40-unit campaign has drained AND the BDQ repair reports COMPLETE. The BDQ
    wait is not politeness: that run produces br_inc_regen_s{2,3,4}.pt, three of the sixteen
    checkpoints this one reads.

    probe=True evaluates the gate exactly once, logs what it sees, writes no status file and never
    sleeps. That is what --dry-run needs: a dry run that blocks for an hour behind the very job it
    is supposed to be reporting on is not a dry run, and it also silently occupies the wait slot."""
    deadline = time.time() + max_hours * 3600
    while time.time() < deadline:
        camp = status_kv('laptop_campaign.status')
        bdq = status_kv('bdq_row2_repair.status')
        camp_ok = True
        camp_txt = 'absent'
        if camp and 'done' in camp:
            d, tot = camp['done'].split('/')
            camp_ok = int(d) >= int(tot) and int(camp.get('running', 0)) == 0
            camp_txt = f"{camp['done']} running={camp.get('running', '?')}"
            if not camp_ok and (time.time() - camp['_mtime']) / 60 > 30:
                log("wait | campaign status is >30 min stale — treating the campaign as gone")
                camp_ok = True
        bdq_ok = True
        bdq_txt = 'absent'
        if bdq:
            bdq_txt = bdq.get('phase', '?')
            bdq_ok = bdq.get('phase') == 'COMPLETE'
            if not bdq_ok and (time.time() - bdq['_mtime']) / 60 > 45:
                log("wait | BDQ status is >45 min stale — NOT proceeding; three checkpoints depend "
                    "on it. Investigate before this run can be trusted.")
        # a merged BDQ csv with five rows is equally good evidence
        merged_bdq = f'results/bdq_row2_repair_{VENUE}.csv'
        if not bdq_ok and os.path.exists(merged_bdq):
            try:
                if sum(1 for _ in csv.DictReader(open(merged_bdq, newline=''))) >= len(SEEDS):
                    log(f"wait | {merged_bdq} holds all {len(SEEDS)} rows — treating BDQ as done")
                    bdq_ok = True
            except OSError:
                pass
        if camp_ok and bdq_ok:
            log(f"wait | clear to proceed | campaign={camp_txt} bdq={bdq_txt}")
            return True
        if probe:
            log(f"wait | PROBE ONLY | campaign={camp_txt} bdq={bdq_txt} — the real run would hold "
                f"here; this dry run does not")
            return True
        try:
            with open(STATUS, 'w', encoding='utf-8') as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} phase=wait campaign={camp_txt} "
                        f"bdq={bdq_txt} free_mb={avail_mb()} "
                        f"elapsed_min={(time.time() - t0) / 60:.1f}\n")
        except OSError:
            pass
        time.sleep(poll)
    log("wait | max wait exceeded — NOT proceeding, because a missing checkpoint would silently "
        "produce a partial table")
    return False


def merge():
    rows = []
    for arm in ARM_ORDER:
        for s in SEEDS:
            p = unit_path(arm, s)
            if os.path.exists(p):
                rows += list(csv.DictReader(open(p, newline='')))
    if rows:
        with open(merged_path(), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
    return rows


# --------------------------------------------------------------------------------------------
# the table
# --------------------------------------------------------------------------------------------
def _pop_sd(v):
    m = sum(v) / len(v)
    return m, math.sqrt(sum((x - m) ** 2 for x in v) / len(v))


def table():
    import numpy as np
    rows = merge()
    print("=" * 126)
    print("  TABLE VII — ALL FOUR LEARNED ARMS, R = %d CRN REPEATS, VENUE = %s" % (R, VENUE))
    print(f"  source : {os.path.abspath(merged_path())}  ({len(rows)} evaluations)")
    print("  protocol: mean over R repeats per checkpoint, THEN population sigma across "
          f"{len(SEEDS)} training seeds")
    print("=" * 126)
    if not rows:
        print("  no units complete yet")
        return 1
    tors = sorted({r['torch'] for r in rows})
    print(f"  toolchain(s) present in this file: {', '.join(tors)}"
          + ("   <-- MIXED VENUE, DO NOT REPORT" if len(tors) > 1 else ""))

    summary = {'venue': VENUE, 'R': R, 'BASE': BASE, 'n_rows': len(rows), 'torch': tors, 'arms': {}}
    for arm in ARM_ORDER:
        sub = [r for r in rows if r['arm'] == arm]
        if not sub:
            continue
        label = ARMS[arm][0]
        seeds_here = sorted({int(r['seed']) for r in sub})
        print(f"\n  ---- {label}  ({len(sub)} evaluations, seeds {seeds_here}) ----")
        print(f"    {'metric':>6} | {'mean':>9} | {'sigma_train':>11} | {'sigma_eval':>10} | "
              f"{'published':>19} | shift")
        print("    " + "-" * 92)
        arm_out = {}
        for m in METRICS:
            per = np.array([np.mean([float(r[m]) for r in sub if int(r['seed']) == s])
                            for s in seeds_here])
            within = [np.std([float(r[m]) for r in sub if int(r['seed']) == s], ddof=1)
                      for s in seeds_here]
            mu, sd = per.mean(), per.std()               # population sigma, the paper's convention
            se = float(np.sqrt(np.mean(np.square(within))))   # pooled within-checkpoint sd
            pm, ps, dp = PUBLISHED[arm][m]
            arm_out[m] = dict(mean=float(mu), sigma_train_pop=float(sd), sigma_eval_pooled=se,
                              per_seed=[float(x) for x in per],
                              published_mean=pm, published_sd=ps)
            print(f"    {m:>6} | {mu:9.{dp}f} | {sd:11.{dp}f} | {se:10.{dp}f} | "
                  f"{pm:>8.{dp}f} +-{ps:<8.{dp}f} | {mu - pm:+.{dp}f} mean, {sd - ps:+.{dp}f} sd")
        cells = " | ".join(f"{arm_out[m]['mean']:.{PUBLISHED[arm][m][2]}f} $\\pm$ "
                           f"{arm_out[m]['sigma_train_pop']:.{PUBLISHED[arm][m][2]}f}"
                           for m in METRICS)
        print(f"\n    ROW: | {label} | {cells} |")
        summary['arms'][arm] = dict(label=label, seeds=seeds_here, metrics=arm_out)

    print("\n" + "=" * 126)
    print("  WHAT THE ERROR BAR LOST: sigma_observed (1 draw) vs sigma_train (R-averaged)")
    print("  A published +- of s_obs contains sigma_train^2 + sigma_eval^2. Averaging R repeats")
    print("  shrinks the second term by R, so a bar that moves DOWN here was partly evaluation noise.")
    print("=" * 126)
    print(f"    {'arm':>12} {'metric':>6} | {'published sd':>12} | {'sigma_train':>11} | "
          f"{'sigma_eval':>10} | {'implied noise share of published var':>36}")
    print("    " + "-" * 100)
    for arm in ARM_ORDER:
        if arm not in summary['arms']:
            continue
        for m in METRICS:
            a = summary['arms'][arm]['metrics'][m]
            ps = a['published_sd']
            share = (1 - (a['sigma_train_pop'] ** 2) / (ps ** 2)) * 100 if ps > 0 else float('nan')
            print(f"    {arm:>12} {m:>6} | {ps:12.3f} | {a['sigma_train_pop']:11.3f} | "
                  f"{a['sigma_eval_pooled']:10.3f} | {share:34.1f}%")

    os.makedirs('results', exist_ok=True)
    json.dump(summary, open(summary_path(), 'w'), indent=2)
    print(f"\n  wrote {merged_path()}")
    print(f"  wrote {summary_path()}")
    print("=" * 126)
    expected = len(ARM_ORDER) * len(SEEDS) * R
    if len(rows) != expected:
        print(f"  INCOMPLETE: {len(rows)} of {expected} evaluations present — this is not a "
              f"reportable table yet.")
        return 1
    return 0


def check_toolchain(allow):
    saved = sys.argv
    sys.argv = [saved[0]]
    try:
        import torch
    finally:
        sys.argv = saved
    mm = '.'.join(torch.__version__.split('+')[0].split('.')[:2])

    # Derive the venue label from the toolchain and overwrite whatever was declared. A file called
    # ..._laptop.csv whose rows were measured under torch 2.13 is exactly the kind of quiet
    # mislabelling this whole exercise exists to remove, and it must not be possible to produce one
    # by forgetting an env var.
    global VENUE
    observed = TORCH_VENUE.get(mm, f'unknown-torch{mm}')
    if observed != VENUE:
        log(f"venue | declared {VENUE!r} but torch {torch.__version__} says {observed!r} — "
            f"relabelling to {observed!r}; the toolchain is the evidence, the env var is not")
        VENUE = observed
        os.environ['T7_VENUE'] = observed

    if mm != EXPECT_TORCH and not allow:
        log(f"REFUSING: torch {torch.__version__} but every published Table VII number was produced "
            f"on torch {EXPECT_TORCH}.x. The entire purpose of this run is that it happens on the "
            f"reporting venue; running it here would reproduce the defect it exists to remove. "
            f"Pass --allow-toolchain-mismatch only for a rehearsal.")
        sys.exit(2)
    return torch.__version__


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--unit', nargs=2, metavar=('ARM', 'SEED'), default=None)
    ap.add_argument('--table', action='store_true')
    ap.add_argument('--verify-patch', action='store_true')
    ap.add_argument('--slots', type=int, default=4)
    ap.add_argument('--floor-mb', type=int, default=1500)
    ap.add_argument('--no-wait', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--allow-toolchain-mismatch', action='store_true')
    ap.add_argument('--arms', default='', help='comma-separated subset of arms to run; '
                                               'default = all four. Restricts preflight too, so a '
                                               'subset run is possible where only some checkpoints '
                                               'exist. A table built from a subset is NOT the paper '
                                               'table and says so.')
    ap.add_argument('--trust-partial-ckpt', action='store_true',
                    help='evaluate regenerated checkpoints even though the BDQ repair has not '
                         'reported COMPLETE. Only correct if you have separately established that '
                         'those files are finished; otherwise it silently admits a half-trained '
                         'policy into Table VII row 2.')
    a = ap.parse_args()

    if a.verify_patch:
        ok = verify_patch()
        return 0 if ok in (True, None) else 1
    if a.unit:
        check_toolchain(a.allow_toolchain_mismatch)
        arm, seed = a.unit[0], int(a.unit[1])
        if arm not in ARMS:
            log(f"unknown arm {arm!r}; expected one of {ARM_ORDER}")
            return 2
        return run_unit(arm, seed)
    if a.table:
        check_toolchain(True)      # label-only: resolve VENUE from the toolchain, never refuse
        return table()

    t0 = time.time()
    log("=" * 110)
    tv = check_toolchain(a.allow_toolchain_mismatch)   # first, so VENUE is truthful in every log line
    if REHEARSAL:
        log(f"*** REHEARSAL *** tag={TAG!r} R={R} eval_ep_override={EVAL_EP_OVERRIDE} — this run "
            f"writes ONLY tagged/short files and is not a result")
    arms = ARM_ORDER
    if a.arms:
        want = [x.strip() for x in a.arms.split(',') if x.strip()]
        bad = [x for x in want if x not in ARMS]
        if bad:
            log(f"unknown arm(s) {bad}; expected a subset of {ARM_ORDER}")
            return 2
        arms = [x for x in ARM_ORDER if x in want]
        log(f"*** ARM SUBSET *** {arms} — this is NOT the full Table VII run; the table it "
            f"produces covers only these rows")
    units = [(arm, s) for arm in arms for s in SEEDS]
    log(f"TABLE VII R={R} CRN | venue={VENUE} | {len(arms)} arms x {len(SEEDS)} seeds x "
        f"R={R} = {len(units) * R} evaluations | BASE={BASE}")
    log(f"toolchain | python {sys.version.split()[0]} | torch {tv} | cwd {os.getcwd()}")
    verify_patch()
    log(f"CRN rule | {PATCH_SEMANTICS}")

    if not a.no_wait:
        if not wait_for(t0, probe=a.dry_run):
            return 1

    log("preflight | checkpoint inventory")
    missing = []
    for arm, s in units:
        ck, origin = ckpt_for(arm, s)
        here = os.path.exists(ck)
        if not here:
            missing.append((arm, s, ck))
        log(f"  {arm:>12} s{s} | {ck:<24} {origin:<22} "
            f"{'present ' + str(os.path.getsize(ck)) + ' B' if here else 'ABSENT'}"
            f"{'  [unit already complete]' if unit_done(arm, s) else ''}")
    if missing:
        log(f"preflight | REFUSING to launch: {len(missing)} checkpoint(s) absent — "
            + ", ".join(f"{a_}/s{s}:{c}" for a_, s, c in missing))
        log("preflight | a partial table is worse than no table. Nothing was launched.")
        return 1

    # PRESENCE IS NOT COMPLETENESS. bdq_row2_repair.py checkpoints periodically while it trains, so
    # br_inc_regen_s{2,3,4}.pt appear on disk — full size, readable, loadable — long before their
    # 800,000 steps are done. Observed directly: at 21:14 all three were 'present 160912 B' while
    # the repair still reported phase=train done=0/3. Evaluating one of those would put a
    # half-trained policy into Table VII row 2 with no visible symptom, which is a falsification in
    # everything but intent. The wait gate normally prevents it; --no-wait would walk straight
    # past it. So refuse here too, independently of how we got here.
    regen = [(arm, s, ck) for arm, s in units
             for ck, origin in [ckpt_for(arm, s)] if origin.startswith('regenerated')]
    if regen and not a.trust_partial_ckpt:
        bdq = status_kv('bdq_row2_repair.status')
        phase = (bdq or {}).get('phase')
        if phase != 'COMPLETE':
            log(f"preflight | REFUSING to launch: {len(regen)} checkpoint(s) come from the BDQ "
                f"repair, which reports phase={phase!r}, not 'COMPLETE' — "
                + ", ".join(f"{a_}/s{s}:{c}" for a_, s, c in regen))
            log("preflight | those files exist on disk because the trainer checkpoints as it goes; "
                "existing is not the same as being finished. Nothing was launched.")
            return 1
        log(f"preflight | {len(regen)} regenerated checkpoint(s) accepted — BDQ repair reports "
            f"COMPLETE")

    todo = [(arm, s) for arm, s in units if not unit_done(arm, s)]
    log(f"plan | {len(todo)} units to run, {len(units) - len(todo)} already complete | "
        f"slots={a.slots} floor={a.floor_mb} MB")
    if a.dry_run:
        log("DRY RUN — nothing launched")
        return 0
    if not todo:
        log("nothing to do; recomputing the table from existing units")
        return table()

    passthru = ['--allow-toolchain-mismatch'] if a.allow_toolchain_mismatch else []
    failed = run_pool(todo, a.slots, a.floor_mb, t0, extra_argv=passthru)
    if failed:
        log(f"FAILED units: {failed} — not writing a table from a partial run")
        write_status('FAILED', len(units) - len(failed), len(units), {}, t0)
        return 1

    log(f"all units complete in {(time.time() - t0) / 60:.1f} min — building the table")
    rc = table()
    write_status('COMPLETE', len(units), len(units), {}, t0)
    log("COMPLETE")
    return rc


if __name__ == '__main__':
    sys.exit(main())
