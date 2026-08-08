#!/usr/bin/env python3
"""
bdq_row2_repair.py — regenerate Table VII row 2 (Branched incremental, BDQ-style) end to end.

WHY THIS EXISTS
---------------
The provenance audit (tablevii_provenance_audit.py) showed that 14 of Table VII's 15 learned cells
regenerate exactly from results/branched_ablation_nyu.csv under the manuscript's population-sigma
convention. One cell does not:

    Branched incremental (BDQ-style)   Tput   981.6 +- 445.5      seeds 1,2,3,4 absent from the file

The archived per-seed file holds a throughput for seed 0 only. The published mean and sd are
self-consistent with five values (mean 981.6, pop sd 445.5, n=5, seed 0 = 1316.4 implies the other
four have mean 897.9 and residual SS 880,260 — a coherent set), and the number appears nowhere
except that one table cell, so this is a number that was almost certainly measured and then not
archived. It is NOT fabricated. But "almost certainly measured" is not a provenance, and a cell that
cannot be regenerated from a file has no business in a paper that is being resubmitted specifically
to answer a reviewer's methodological-consistency objection.

Two of the five checkpoints survive (br_inc_s0.pt 2026-07-12 22:26, br_inc_s1.pt 2026-07-12 22:55,
laptop only). Three do not. So the repair is:

    1. retrain br_inc seeds 2, 3, 4 at the published budget (800,000 steps, binary_power=False,
       guard_db=None) on the venue and with the environment file that produced the originals;
    2. re-evaluate ALL FIVE seeds in one campaign, under one protocol, writing every metric
       including Tput;
    3. replace Table VII row 2 in full, and recompute the Finding-3 numbers that depend on it.

Replacing the whole row rather than patching one cell is the point. Splicing a freshly measured
throughput next to four-week-old HO/Power/Spec/RLF values would mix two draws of an unseeded
evaluation channel inside a single row — a miniature of the exact defect the paper is being
rewritten to remove.

WHAT IS AND IS NOT BEING CLAIMED ABOUT THE RETRAINED SEEDS
----------------------------------------------------------
The retrained policies for seeds 2-4 are NOT the original policies recovered. They are fresh draws
from the same training distribution: same driver (train_branched_nyu.py, md5
a4991bd7a89ddb1e1581d347d7a15127, identical on both venues), same seed, same budget, same
environment file. Because the evaluation channel RNG is unseeded (env_6g_v5.py:132 builds
np.random.default_rng(None)), even re-evaluating the two SURVIVING checkpoints gives new numbers.
That is why all five cells of row 2 will move, and why the row must be reported as regenerated on
this date rather than presented as the old row with one gap filled.

VENUE
-----
This must run on the laptop. train_branched_nyu.py is byte-identical across venues, but
env_6g_branched.py is not:

    laptop  md5 6a998fae24fb55e3b645d66606b2a72a   8051 bytes   2026-07-12 14:47
    cloud   md5 443386f642d19bb8eeffbc9c74b2d853   8317 bytes   2026-07-29 00:15

The laptop copy predates br_inc_s0.pt (2026-07-12 22:26) and is the one that trained every row of
Table VII. The laptop also runs torch 2.5.1, the toolchain of every published number; the cloud runs
torch 2.13. The script refuses to run its training phase on a torch major/minor other than the one
recorded in EXPECT_TORCH unless --allow-toolchain-mismatch is passed, so a stray cloud invocation
cannot quietly put two toolchains inside one table row.

CONCURRENCY
-----------
Three trainings at ~743 MB resident each. The 40-unit revision campaign is using the machine, so by
default this waits until laptop_campaign.status reports every unit finished and no unit running,
then holds until --floor-mb of RAM is free. Every worker is pinned to one thread (OMP/MKL=1 plus
torch.set_num_threads(1)), which is the paper's stated protocol.

OUTPUTS
-------
    results/bdq_row2_repair_s{seed}_{venue}.csv   one row per seed, written by its own worker
    results/bdq_row2_repair_{venue}.csv           the five rows merged, with checkpoint provenance
    bdq_row2_repair.status                        heartbeat, one line, overwritten each poll
    bdq_row2_repair.log                           append-only event log
    logs/bdq_row2_{train,eval}_s{seed}.log        per-worker stdout
    br_inc_regen_s{2,3,4}.pt                      the regenerated checkpoints, deliberately NOT
                                                  named br_inc_s{2,3,4}.pt so that the two
                                                  originals and the three regenerations stay
                                                  distinguishable in the release archive

USAGE (PowerShell, from the folder holding the drivers)
    python -u bdq_row2_repair.py                      # wait for the campaign, then do everything
    python -u bdq_row2_repair.py --no-wait            # start immediately
    python -u bdq_row2_repair.py --dry-run            # print the plan, launch nothing
    python -u bdq_row2_repair.py --table              # recompute the row from existing per-seed files
Resume: rerun the same command. A seed is done when its per-seed CSV holds its row.
"""
import os, sys

# One thread per process, set before torch is imported anywhere in this interpreter.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import argparse, csv, hashlib, json, math, subprocess, time

VENUE = os.environ.get('BDQ_VENUE', 'laptop')
LABEL = 'branched-incremental (BDQ)'
SEEDS = [0, 1, 2, 3, 4]
RETRAIN = [2, 3, 4]                      # the three whose checkpoints are gone
BINARY_POWER = False                     # incremental (vanilla-BDQ) power head
GUARD_DB = None                          # unguarded arm
EXPECT_TORCH = '2.5'                     # major.minor of the toolchain that produced Table VII

# ---- REHEARSAL KNOBS -----------------------------------------------------------------------
# These exist only so the whole pipeline (train -> checkpoint -> evaluate -> row -> merge ->
# table) can be smoke-tested in minutes before it is trusted with a 70-minute real run. BDQ_TAG
# renames every checkpoint and every output file, so a rehearsal physically cannot write into a
# real result path. The real run sets none of them and is asserted below.
TAG = os.environ.get('BDQ_TAG', '')                       # '' for the real run
STEPS = int(os.environ.get('BDQ_STEPS', 800_000))         # 800,000 for the real run
EVAL_EP_OVERRIDE = int(os.environ.get('BDQ_EVAL_EP', 0))  # 0 = use the driver's 150
REHEARSAL = bool(TAG) or STEPS != 800_000 or EVAL_EP_OVERRIDE
_SFX = f'_{TAG}' if TAG else ''

if TAG:
    RETRAIN = list(SEEDS)                # a rehearsal owns none of the real checkpoints

# published Table VII row 2, transcribed from the manuscript (mean, pop sd, decimals)
PUBLISHED_ROW2 = {'HO': (93.15, 3.30, 2), 'Power': (26.88, 15.20, 2), 'Spec': (54.99, 29.94, 2),
                  'RLF': (9.04, 4.23, 2), 'Tput': (981.6, 445.5, 1)}
METRICS = ['HO', 'Power', 'Spec', 'RLF', 'Tput']

FIELDS = ['config', 'seed', 'HO', 'Power', 'Spec', 'RLF', 'Tput', 'phase_min',
          'ckpt', 'ckpt_origin', 'ckpt_mtime', 'ckpt_md5', 'venue', 'python', 'torch', 'numpy',
          'steps', 'binary_power', 'guard_db', 'env_branched_md5', 'driver_md5', 'measured_utc']


def ckpt_for(seed):
    """Originals keep their 2026-07-12 filenames; regenerations get a name of their own."""
    if TAG:
        return f'br_inc_{TAG}_s{seed}.pt'
    return f'br_inc_s{seed}.pt' if seed not in RETRAIN else f'br_inc_regen_s{seed}.pt'


def origin_for(seed):
    if TAG:
        return f'REHEARSAL-{TAG}'
    return 'original-20260712' if seed not in RETRAIN else 'regenerated-20260801'


def rows_path(seed):
    return f'results/bdq_row2_repair{_SFX}_s{seed}_{VENUE}.csv'


MERGED = f'results/bdq_row2_repair{_SFX}_{VENUE}.csv'
STATUS = f'bdq_row2_repair{_SFX}.status'
LOG = f'bdq_row2_repair{_SFX}.log'


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
    print(line, flush=True)
    try:
        with open(LOG, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    except OSError:
        pass


def md5(path):
    if not os.path.exists(path):
        return ''
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def avail_mb():
    """Windows via GlobalMemoryStatusEx, Linux via /proc/meminfo, so this can be rehearsed in the
    cloud container before it is trusted with the real run."""
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


def campaign_state(path='laptop_campaign.status'):
    """Parse 'done=30/40 running=6 queued=4 slots=6 free_mb=1761 elapsed_min=230.0'.
    Returns None when the file is absent (no campaign to wait for)."""
    if not os.path.exists(path):
        return None
    try:
        head = open(path, encoding='utf-8').readline()
    except OSError:
        return None
    kv = {}
    for tok in head.split():
        if '=' in tok:
            k, v = tok.split('=', 1)
            kv[k] = v
    if 'done' not in kv:
        return None
    d, t = kv['done'].split('/')
    return dict(done=int(d), total=int(t), running=int(kv.get('running', 0)),
                queued=int(kv.get('queued', 0)), mtime=os.path.getmtime(path), raw=head.strip())


def import_driver():
    """train_branched_nyu.py reads sys.argv AT IMPORT TIME:
           N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
           STEPS   = int(sys.argv[2]) if len(sys.argv) > 2 else 800_000
       Importing it with this script's argv would either crash on int('--train-one') or silently
       reset the step budget. Neutralise argv across the import, then restore it. Every call below
       passes steps explicitly anyway, so the module-level STEPS is never relied on."""
    saved = sys.argv
    sys.argv = [saved[0]]
    try:
        import train_branched_nyu as T
        import train_phase2 as tp
        import torch
        torch.set_num_threads(1)
        return T, tp, torch
    finally:
        sys.argv = saved


def toolchain():
    import torch, numpy
    return dict(python=sys.version.split()[0], torch=torch.__version__, numpy=numpy.__version__)


def check_toolchain(allow):
    tc = toolchain()
    mm = '.'.join(tc['torch'].split('+')[0].split('.')[:2])
    if mm != EXPECT_TORCH and not allow:
        log(f"REFUSING: torch {tc['torch']} but Table VII was produced on torch {EXPECT_TORCH}.x. "
            f"Retraining here would put two toolchains inside one table row. Run this on the laptop, "
            f"or pass --allow-toolchain-mismatch if you know what you are doing.")
        sys.exit(2)
    return tc


def write_row(seed, metrics, phase_min, tc):
    os.makedirs('results', exist_ok=True)
    ck = ckpt_for(seed)
    row = {'config': LABEL, 'seed': seed,
           'HO': f"{metrics['HO']:.2f}", 'Power': f"{metrics['Power']:.2f}",
           'Spec': f"{metrics['Spec']:.2f}", 'RLF': f"{metrics['RLF']:.2f}",
           'Tput': f"{metrics['Tput']:.1f}", 'phase_min': f"{phase_min:.1f}",
           'ckpt': ck, 'ckpt_origin': origin_for(seed),
           'ckpt_mtime': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(ck))),
           'ckpt_md5': md5(ck), 'venue': VENUE,
           'python': tc['python'], 'torch': tc['torch'], 'numpy': tc['numpy'],
           'steps': STEPS, 'binary_power': BINARY_POWER, 'guard_db': GUARD_DB,
           'env_branched_md5': md5('env_6g_branched.py'), 'driver_md5': md5('train_branched_nyu.py'),
           'measured_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    with open(rows_path(seed), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerow(row)
    return row


# --------------------------------------------------------------------------------------------
# workers
# --------------------------------------------------------------------------------------------
def train_one(seed):
    T, tp, torch = import_driver()
    ck = ckpt_for(seed)
    if os.path.exists(ck):
        log(f"[train s{seed}] {ck} already present ({os.path.getsize(ck)} B) — not retraining")
        return 0
    log(f"[train s{seed}] start | {STEPS:,} steps | binary_power={BINARY_POWER} guard_db={GUARD_DB} "
        f"| torch {torch.__version__} | threads {torch.get_num_threads()}")
    slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH)
    t0 = time.time()
    T.train(seed, BINARY_POWER, ck, slow, med, steps=STEPS, guard_db=GUARD_DB)
    mins = (time.time() - t0) / 60
    log(f"[train s{seed}] done in {mins:.1f}m -> {ck} ({os.path.getsize(ck)} B, md5 {md5(ck)})")
    return 0


def eval_one(seed):
    T, tp, torch = import_driver()
    from dndqn import OBS_DIM
    from dndqn_branched import load_branched
    ck = ckpt_for(seed)
    if not os.path.exists(ck):
        log(f"[eval s{seed}] MISSING CHECKPOINT {ck} — cannot evaluate")
        return 3
    tc = toolchain()
    if EVAL_EP_OVERRIDE:
        T.EVAL_EP = EVAL_EP_OVERRIDE     # rehearsal only; see REHEARSAL KNOBS
    log(f"[eval s{seed}] start | {ck} ({origin_for(seed)}, md5 {md5(ck)}) | "
        f"{len(T.EVAL_SEEDS)} eval seeds x {T.EVAL_EP} episodes")
    slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH)
    pool = slow + med
    net = load_branched(ck, OBS_DIM)[0]
    t0 = time.time()
    m = T.evaluate_branched(net, pool, BINARY_POWER, guard_db=GUARD_DB)
    mins = (time.time() - t0) / 60
    write_row(seed, m, mins, tc)
    log(f"[eval s{seed}] done in {mins:.1f}m | HO {m['HO']:.2f} Power {m['Power']:.2f} "
        f"Spec {m['Spec']:.2f} RLF {m['RLF']:.2f} Tput {m['Tput']:.1f} -> {rows_path(seed)}")
    return 0


# --------------------------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------------------------
def have_row(seed):
    p = rows_path(seed)
    if not os.path.exists(p):
        return False
    try:
        return sum(1 for _ in csv.DictReader(open(p, newline=''))) >= 1
    except OSError:
        return False


def status_line(phase, done, total, running, t0):
    txt = (f"{time.strftime('%Y-%m-%d %H:%M:%S')} phase={phase} done={done}/{total} "
           f"running={len(running)} free_mb={avail_mb()} elapsed_min={(time.time() - t0) / 60:.1f}")
    for name, (p, started, fh) in running.items():
        txt += f"\n  {name} pid {p.pid} {(time.time() - started) / 60:.1f}m"
    try:
        with open(STATUS, 'w', encoding='utf-8') as f:
            f.write(txt + '\n')
    except OSError:
        pass


def run_pool(kind, seeds, slots, floor_mb, t0, poll=20, extra_argv=()):
    """Launch `python -u <this> --{kind}-one SEED` for each seed, `slots` at a time, holding when
    free RAM is under the floor. Returns the list of seeds whose worker exited non-zero."""
    env = dict(os.environ)
    env['OMP_NUM_THREADS'] = env['MKL_NUM_THREADS'] = '1'
    env['PYTHONPATH'] = os.getcwd() + os.pathsep + env.get('PYTHONPATH', '')
    env['BDQ_VENUE'] = VENUE
    os.makedirs('logs', exist_ok=True)
    queue = list(seeds)
    running, failed, done = {}, [], 0
    total = len(queue)
    while queue or running:
        for name in list(running):
            p, started, fh = running[name]
            if p.poll() is None:
                continue
            fh.close()
            mins = (time.time() - started) / 60
            seed = int(name.split('s')[-1])
            ok = p.returncode == 0 and (kind == 'train' or have_row(seed))
            if ok:
                done += 1
                log(f"DONE  {name} rc={p.returncode} in {mins:.1f}m")
            else:
                failed.append(seed)
                log(f"FAIL  {name} rc={p.returncode} after {mins:.1f}m — see logs/{name}.log")
            del running[name]
        while queue and len(running) < slots:
            free = avail_mb()
            if free < floor_mb:
                log(f"HOLD  {free} MB free < floor {floor_mb} MB "
                    f"({len(running)} running, {len(queue)} queued)")
                break
            seed = queue.pop(0)
            name = f'bdq_row2_{kind}_s{seed}'
            fh = open(os.path.join('logs', name + '.log'), 'a', encoding='utf-8')
            cmd = ([sys.executable, '-u', os.path.abspath(__file__), f'--{kind}-one', str(seed)]
                   + list(extra_argv))
            p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env,
                                 cwd=os.getcwd())
            running[name] = (p, time.time(), fh)
            log(f"START {name} pid {p.pid} ({free} MB free) | {' '.join(cmd)}")
        status_line(kind, done, total, running, t0)
        if queue or running:
            time.sleep(poll)
    return failed


def wait_for_campaign(t0, poll=60, max_hours=12):
    st = campaign_state()
    if st is None:
        log("wait | no laptop_campaign.status found — nothing to wait for")
        return
    log(f"wait | campaign at {st['done']}/{st['total']}, {st['running']} running — "
        f"holding until every unit is finished")
    deadline = time.time() + max_hours * 3600
    while time.time() < deadline:
        st = campaign_state()
        if st is None:
            return
        stale_min = (time.time() - st['mtime']) / 60
        if st['done'] >= st['total'] and st['running'] == 0:
            log(f"wait | campaign complete ({st['raw']}) — proceeding")
            return
        if stale_min > 30:
            log(f"wait | campaign status file is {stale_min:.0f} min stale ({st['raw']}) — "
                f"treating the campaign as gone and proceeding")
            return
        try:
            with open(STATUS, 'w', encoding='utf-8') as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} phase=wait-campaign "
                        f"campaign={st['done']}/{st['total']} running={st['running']} "
                        f"free_mb={avail_mb()} elapsed_min={(time.time() - t0) / 60:.1f}\n")
        except OSError:
            pass
        time.sleep(poll)
    log("wait | max wait exceeded — proceeding anyway")


def merge():
    rows = []
    for s in SEEDS:
        p = rows_path(s)
        if os.path.exists(p):
            rows += list(csv.DictReader(open(p, newline='')))
    rows.sort(key=lambda r: int(r['seed']))
    if rows:
        with open(MERGED, 'w', newline='') as f:
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


def _samp_sd(v):
    m = sum(v) / len(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else float('nan')


def _t_tests(a, b):
    """Three conventions, all printed, because the manuscript's choice has to be identified by
    reproducing a published test from archived data rather than assumed."""
    import numpy as np
    from scipy import stats
    a, b = np.asarray(a, float), np.asarray(b, float)
    out = {}
    tw, pw = stats.ttest_ind(a, b, equal_var=False)
    ts, ps = stats.ttest_ind(a, b, equal_var=True)
    out['welch'] = (float(tw), float(pw))
    out['student'] = (float(ts), float(ps))
    if len(a) == len(b):
        tp_, pp = stats.ttest_rel(a, b)
        out['paired'] = (float(tp_), float(pp))
    sp = math.sqrt(((len(a) - 1) * a.std(ddof=1) ** 2 + (len(b) - 1) * b.std(ddof=1) ** 2)
                   / (len(a) + len(b) - 2))
    out['cohen_d'] = float((a.mean() - b.mean()) / sp) if sp > 0 else float('nan')
    return out


def table(archive='results/branched_ablation_nyu.csv'):
    rows = merge()
    print("=" * 112)
    print("  TABLE VII ROW 2 — REGENERATED")
    print(f"  source : {os.path.abspath(MERGED)}  ({len(rows)} seeds)")
    print("  sd     : population form (ddof=0), the manuscript's convention")
    print("=" * 112)
    if not rows:
        print("  no per-seed rows yet")
        return 1
    print(f"\n  {'seed':>4} | {'origin':>20} | {'HO':>7} | {'Power':>7} | {'Spec':>7} | "
          f"{'RLF':>7} | {'Tput':>8} | ckpt")
    print("  " + "-" * 104)
    for r in rows:
        print(f"  {r['seed']:>4} | {r['ckpt_origin']:>20} | {float(r['HO']):7.2f} | "
              f"{float(r['Power']):7.2f} | {float(r['Spec']):7.2f} | {float(r['RLF']):7.2f} | "
              f"{float(r['Tput']):8.1f} | {r['ckpt']}")
    new = {}
    print(f"\n  {'metric':>6} | {'n':>2} | {'new mean':>9} | {'new sd0':>8} | {'published':>18} | shift")
    print("  " + "-" * 78)
    for m in METRICS:
        v = [float(r[m]) for r in rows]
        mu, sd = _pop_sd(v)
        new[m] = (mu, sd, _samp_sd(v))
        pm, ps, dp = PUBLISHED_ROW2[m]
        print(f"  {m:>6} | {len(v):>2} | {mu:9.{dp}f} | {sd:8.{dp}f} | "
              f"{pm:>8.{dp}f} +-{ps:<7.{dp}f} | {mu - pm:+.{dp}f} mean, {sd - ps:+.{dp}f} sd")
    print("\n  REPLACEMENT ROW (paste into Table VII):")
    cells = " | ".join(f"{new[m][0]:.{PUBLISHED_ROW2[m][2]}f} $\\pm$ {new[m][1]:.{PUBLISHED_ROW2[m][2]}f}"
                       for m in ['HO', 'Power', 'Spec', 'RLF', 'Tput'])
    print(f"    | Branched incremental (BDQ-style) | {cells} |")

    # comparisons against the two archived arms, for the Finding-3 recomputation
    if os.path.exists(archive):
        arch = {}
        for r in csv.DictReader(open(archive, newline='')):
            arch.setdefault(r['config'], {}).setdefault(int(r['seed']), r)
        for other in ['branched-binary (proposed)', 'branched-guarded-state (proposed 9dB)']:
            if other not in arch:
                continue
            print(f"\n  BDQ vs {other}  (seeds 0-4, same-seed pairs)")
            print(f"    {'metric':>6} | {'BDQ mean+-sd0':>20} | {'other mean+-sd0':>20} | "
                  f"{'paired t':>9} {'p':>8} | {'Welch t':>8} {'p':>8} | {'d':>6}")
            print("    " + "-" * 104)
            for m in METRICS:
                try:
                    b = [float(arch[other][s][m]) for s in SEEDS if arch[other].get(s)
                         and (arch[other][s].get(m) or '').strip()]
                except (KeyError, ValueError):
                    b = []
                a = [float(r[m]) for r in rows]
                if len(b) != len(a):
                    print(f"    {m:>6} | (other arm has {len(b)} values, skipped)")
                    continue
                t = _t_tests(a, b)
                am, asd = _pop_sd(a)
                bm, bsd = _pop_sd(b)
                pr = t.get('paired', (float('nan'), float('nan')))
                print(f"    {m:>6} | {am:9.2f} +-{asd:<8.2f} | {bm:9.2f} +-{bsd:<8.2f} | "
                      f"{pr[0]:9.2f} {pr[1]:8.4f} | {t['welch'][0]:8.2f} {t['welch'][1]:8.4f} | "
                      f"{t['cohen_d']:6.2f}")
    out = os.path.join('results', f'bdq_row2_summary_{VENUE}.json')
    json.dump({'n': len(rows), 'venue': VENUE,
               'new': {m: {'mean': new[m][0], 'sd_pop': new[m][1], 'sd_samp': new[m][2]} for m in METRICS},
               'published': {m: {'mean': PUBLISHED_ROW2[m][0], 'sd_pop': PUBLISHED_ROW2[m][1]} for m in METRICS},
               'seeds': [{k: r[k] for k in ('seed', 'ckpt', 'ckpt_origin', 'ckpt_md5', 'torch',
                                            'measured_utc')} for r in rows]},
              open(out, 'w'), indent=2)
    print(f"\n  wrote {MERGED}")
    print(f"  wrote {out}")
    print("=" * 112)
    return 0 if len(rows) == len(SEEDS) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-one', type=int, default=None)
    ap.add_argument('--eval-one', type=int, default=None)
    ap.add_argument('--table', action='store_true')
    ap.add_argument('--slots', type=int, default=3)
    ap.add_argument('--floor-mb', type=int, default=1400)
    ap.add_argument('--no-wait', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--allow-toolchain-mismatch', action='store_true')
    a = ap.parse_args()

    if a.train_one is not None:
        check_toolchain(a.allow_toolchain_mismatch)
        return train_one(a.train_one)
    if a.eval_one is not None:
        check_toolchain(a.allow_toolchain_mismatch)
        return eval_one(a.eval_one)
    if a.table:
        return table()

    t0 = time.time()
    log("=" * 96)
    if REHEARSAL:
        log(f"*** REHEARSAL *** tag={TAG!r} steps={STEPS:,} eval_ep_override={EVAL_EP_OVERRIDE} — "
            f"this run writes ONLY tagged files and is not a result")
    log(f"BDQ ROW 2 REPAIR | venue={VENUE} | retrain seeds {RETRAIN} | evaluate seeds {SEEDS} | "
        f"{STEPS:,} steps")
    tc = check_toolchain(a.allow_toolchain_mismatch)
    log(f"toolchain | python {tc['python']} | torch {tc['torch']} | numpy {tc['numpy']} | "
        f"cwd {os.getcwd()}")
    log(f"files | env_6g_branched.py md5 {md5('env_6g_branched.py')} | "
        f"train_branched_nyu.py md5 {md5('train_branched_nyu.py')}")
    for s in SEEDS:
        log(f"  seed {s}: ckpt {ckpt_for(s):<22} {origin_for(s):<20} "
            f"{'present' if os.path.exists(ckpt_for(s)) else 'ABSENT — will train'}"
            f"{'  [row already written]' if have_row(s) else ''}")
    if a.dry_run:
        log("DRY RUN — nothing launched")
        return 0

    if not a.no_wait:
        wait_for_campaign(t0)

    # workers re-check the toolchain themselves, so the operator's decision must reach them
    passthru = ['--allow-toolchain-mismatch'] if a.allow_toolchain_mismatch else []

    need_train = [s for s in RETRAIN if not os.path.exists(ckpt_for(s))]
    if need_train:
        log(f"phase 1 | training seeds {need_train}, {a.slots} at a time")
        failed = run_pool('train', need_train, a.slots, a.floor_mb, t0, extra_argv=passthru)
        if failed:
            log(f"phase 1 | FAILED seeds {failed} — stopping before evaluation so that a partial "
                f"row cannot be reported")
            return 1
    else:
        log("phase 1 | all checkpoints present, nothing to train")

    missing = [s for s in SEEDS if not os.path.exists(ckpt_for(s))]
    if missing:
        log(f"phase 2 | REFUSING to evaluate: checkpoints absent for seeds {missing}")
        return 1

    need_eval = [s for s in SEEDS if not have_row(s)]
    if need_eval:
        log(f"phase 2 | evaluating seeds {need_eval} in one campaign, {a.slots} at a time")
        failed = run_pool('eval', need_eval, a.slots, a.floor_mb, t0, extra_argv=passthru)
        if failed:
            log(f"phase 2 | FAILED seeds {failed}")
            return 1
    else:
        log("phase 2 | all five rows already written")

    log(f"phase 3 | recomputing row 2 ({(time.time() - t0) / 60:.1f} min total)")
    rc = table()
    try:
        with open(STATUS, 'w', encoding='utf-8') as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} phase=COMPLETE done={len(SEEDS)}/{len(SEEDS)} "
                    f"running=0 free_mb={avail_mb()} elapsed_min={(time.time() - t0) / 60:.1f}\n")
    except OSError:
        pass
    log("COMPLETE")
    return rc


if __name__ == '__main__':
    sys.exit(main())
