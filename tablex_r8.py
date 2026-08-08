#!/usr/bin/env python3
"""
tablex_r8.py — extend the R=8 common-random-number evaluation grid to the four learned Table X
rows that tablevii_r8.py does not cover.

WHY THIS EXISTS
---------------
Table VII now reports every learned row on a seeded R=8 grid: five training seeds, each checkpoint
re-evaluated eight times under common random numbers, so the printed +/- is training-seed sigma with
the evaluation-noise component averaged down by sqrt(8) rather than confounded into it.

Table X's flat row is COUPLED to Table VII row 1 -- same five trainings out of the same file -- so
it moves to the seeded basis with it (Table X correction spec section 4.1: "Do not apply the seeded
values to one table alone"). But Table X's other learned rows come from the matched campaign
(Proposed Dn-DQN / DDQN+PER / Standard DQN) and the Safe-DQN campaign, which are FIVE TRAINING
SEEDS x ONE UNSEEDED EVALUATION PASS each.

Printing an R=8 row and a single-pass row side by side in one comparison table would reintroduce the
Associate Editor's criticism (b) in a subtler form:

    "some baselines are represented by a single trained model and others by evaluation-seed
     variability, making the comparisons methodologically inconsistent"

-- because the two sigmas would still mean different things (one is sigma_train with the evaluation
component suppressed, the other is sigma_train and sigma_eval fully confounded). This driver removes
that asymmetry by putting every learned Table X row on the identical grid.

    4 arms x 5 seeds x 8 repeats = 160 evaluations.

WHAT IS AND IS NOT SHARED WITH tablevii_r8.py
---------------------------------------------
tablevii_r8.py is FROZEN at md5 5fd895bf1560f2fe94715b8557b544a9. It is the released Table VII
harness, its md5 is recorded in the correction spec and the data-availability statement, and nothing
here modifies it. This driver IMPORTS it and calls its patch, so the channel seeds for Table X are
produced by literally the same code objects that produced Table VII's -- not by a copy that could
drift:

    T7.install_patch(env_6g_v5)      # patches env_6g_v5.SubTHzEnv6G.reset
    with T7.seeded(r): ...           # seed = T7.BASE + 100000*r + n  when reset(seed=None)

Everything that must NOT be shared -- output directory, merged CSV path, summary JSON path, status
file, log -- is defined here with its own names, so a Table X run can never overwrite, extend or be
mistaken for the Table VII artefact.

THE EVALUATORS, AND WHY THEY ARE THE RIGHT ONES
-----------------------------------------------
Each arm is evaluated by the same evaluate() OBJECT its own campaign driver used, imported from the
unmodified published module -- never reimplemented here:

    m5_dndqn / m5_ddqn / m5_std   ->  run_multiseed_matched.evaluate   (campaign_matched.py imports it)
    sd5_safe                      ->  train_safe_dqn.evaluate          (campaign_safedqn.py imports it)

run_multiseed_matched.evaluate and run_multiseed_rel.evaluate (the one tablevii_r8.py's flat family
uses for Table VII row 1) are AST-IDENTICAL -- same environment, same GAIN, same EVAL_SEEDS, same
EVAL_EP, differing only in where the source lines are broken. verify_evaluators() proves this at
run time and REFUSES if it ever stops being true, because if those two functions diverged, Table VII
row 1 and Table X's matched Dn-DQN row would silently stop being comparable.

train_safe_dqn.evaluate deliberately differs: the shield is applied at evaluation as well as at
training, which is what the published Safe-DQN baseline is. Same environment, same GAIN, same
EVAL_SEEDS, same EVAL_EP.

CHECKPOINT LOADERS
------------------
    m5_dndqn   cfg('dndqn') = (dueling=True, ...)   -> load_dndqn_checkpoint
    m5_ddqn    cfg('ddqn')  = (dueling=False, ...)  -> load_plain   (PlainQNet)
    m5_std     cfg('std')   = (dueling=False, ...)  -> load_plain   (PlainQNet)
    sd5_safe   dueling                              -> load_dndqn_checkpoint

taken from run_multiseed_matched.cfg()/load_net() and campaign_safedqn.py respectively, not guessed.

USAGE (PowerShell, from the folder holding the drivers)
    python -u tablex_r8.py                  # all 160
    python -u tablex_r8.py --dry-run        # print the plan, launch nothing
    python -u tablex_r8.py --table          # recompute the rows from whatever units exist
    python -u tablex_r8.py --verify         # patch + evaluator identity checks only, no compute
Resume: rerun the same command. A unit is done when its CSV holds R rows.
"""
import os, sys

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import argparse, ast, csv, hashlib, inspect, json, math, subprocess, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tablevii_r8 as T7                       # frozen; imported, never modified

# VENUE is a LABEL derived from evidence, exactly as in T7: the env var is only a default and
# check_toolchain() overwrites it from the observed torch version.
VENUE = os.environ.get('T7_VENUE', 'laptop')
BASE = T7.BASE                                  # 20260801 -- the same constant, not a copy of it
EXPECT_TORCH = T7.EXPECT_TORCH                  # '2.5' -> the reporting venue
R_REAL = T7.R_REAL                              # 8
SEEDS = list(T7.SEEDS)                          # [0,1,2,3,4]
METRICS = list(T7.METRICS)                      # HO Power Spec RLF Tput

TAG = os.environ.get('TX_TAG', '')
R = int(os.environ.get('TX_R', R_REAL))
EVAL_EP_OVERRIDE = int(os.environ.get('TX_EVAL_EP', 0))
REHEARSAL = bool(TAG) or R != R_REAL or bool(EVAL_EP_OVERRIDE)
_SFX = f'_{TAG}' if TAG else ''

# ---- the four learned Table X arms this driver covers ----------------------------------------
# key -> (manuscript row label, checkpoint pattern, loader, evaluator module key, training driver)
ARMS = {
    'm5_dndqn': ('Proposed Dn-DQN (matched objective)', 'm5_dndqn_s{}.pt', 'dueling', 'matched',
                 'run_multiseed_matched.py', 'campaign_matched.py'),
    'm5_ddqn':  ('DDQN+PER',                            'm5_ddqn_s{}.pt',  'plain',   'matched',
                 'run_multiseed_matched.py', 'campaign_matched.py'),
    'm5_std':   ('Standard DQN',                        'm5_std_s{}.pt',   'plain',   'matched',
                 'run_multiseed_matched.py', 'campaign_matched.py'),
    'sd5_safe': ('Shielded Safe-DQN',                   'sd5_safe_s{}.pt', 'dueling', 'safe',
                 'train_safe_dqn.py',        'campaign_safedqn.py'),
}
ARM_ORDER = ['m5_dndqn', 'm5_ddqn', 'm5_std', 'sd5_safe']

# The 5-training-seed SINGLE-PASS laptop campaign values these R=8 rows refine. Recomputed
# population form (np.std, ddof=0) on 1 Aug 2026 from
#   results/matched_rel_5seed_800k_laptop.csv   (Proposed Dn-DQN, DDQN+PER, Standard DQN)
#   results/safedqn_5seed_800k_laptop.csv       (Shielded Safe-DQN)
# They are the COMPARISON BASELINE for the shift column, not an input to any measured quantity.
# Their sigma confounds training-seed and evaluation-seed variance; the whole point of the R=8 grid
# is to separate those two, so a large shift in sigma here is the expected result, not an error.
SINGLEPASS = {
    'm5_dndqn': {'HO': (92.93, 3.10, 2), 'Power': (21.90, 23.15, 2), 'Spec': (87.10, 4.71, 2),
                 'RLF': (8.35, 5.94, 2), 'Tput': (505.40, 94.28, 1)},
    'm5_ddqn':  {'HO': (91.02, 2.72, 2), 'Power': (54.95, 20.28, 2), 'Spec': (58.56, 20.10, 2),
                 'RLF': (12.24, 4.98, 2), 'Tput': (813.98, 207.29, 1)},
    'm5_std':   {'HO': (86.30, 1.59, 2), 'Power': (81.78, 4.04, 2), 'Spec': (76.94, 4.00, 2),
                 'RLF': (18.59, 2.62, 2), 'Tput': (543.60, 47.78, 1)},
    'sd5_safe': {'HO': (92.86, 1.47, 2), 'Power': (27.28, 7.31, 2), 'Spec': (84.81, 4.77, 2),
                 'RLF': (10.45, 3.67, 2), 'Tput': (527.12, 65.00, 1)},
}

FIELDS = ['arm', 'label', 'seed', 'ckpt', 'ckpt_md5', 'repeat', 'chan_seed_first', 'secs'] \
    + METRICS + ['venue', 'python', 'torch', 'numpy', 'loader', 'evaluator', 'eval_ast_md5',
                 'eval_seeds', 'eval_ep', 'env_v5_md5', 'env_nyu_md5', 'train_driver_md5',
                 'campaign_md5', 't7_md5', 'measured_utc']

OUTDIR = f'results/tablex_r8{_SFX}'
STATUS = f'tablex_r8{_SFX}.status'
LOG = f'tablex_r8{_SFX}.log'

md5 = T7.md5
avail_mb = T7.avail_mb


def merged_path():
    return f'results/tablex_r8{_SFX}_{VENUE}.csv'


def summary_path():
    return f'results/tablex_r8{_SFX}_summary_{VENUE}.json'


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
    print(line, flush=True)
    try:
        with open(LOG, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    except OSError:
        pass


def ckpt_for(arm, seed):
    return ARMS[arm][1].format(seed)


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


# --------------------------------------------------------------------------------------------
# imports
# --------------------------------------------------------------------------------------------
def import_stack():
    """Every driver module parses sys.argv AT IMPORT TIME (run_multiseed_matched:30,
    run_multiseed_rel:27). Neutralise argv across the import, then restore it. The N_SEEDS/STEPS
    those modules compute from argv govern only their own main(); evaluate() does not read them."""
    saved = sys.argv
    sys.argv = [saved[0]]
    try:
        import torch
        torch.set_num_threads(1)
        import numpy as np
        import env_6g_v5
        import env_6g_nyu
        import train_phase2 as tp
        from dndqn import load_dndqn_checkpoint
        from qnet_plain import load_plain
        import run_multiseed_matched as M
        import run_multiseed_rel as F
        import train_safe_dqn as SD
        return dict(torch=torch, np=np, env_6g_v5=env_6g_v5, env_6g_nyu=env_6g_nyu, tp=tp,
                    load_dndqn_checkpoint=load_dndqn_checkpoint, load_plain=load_plain,
                    M=M, F=F, SD=SD)
    finally:
        sys.argv = saved


def _fn_ast_md5(fn):
    """md5 of the function's ABSTRACT SYNTAX TREE, so formatting differences cannot mask or fake a
    difference in behaviour. Two functions with the same value here compute the same thing."""
    src = inspect.getsource(fn)
    node = ast.parse(src).body[0]
    return hashlib.md5(ast.dump(node, annotate_fields=True,
                                include_attributes=False).encode()).hexdigest()


def verify_evaluators(S):
    """Three checks, all refusable.

    1. run_multiseed_matched.evaluate must be AST-identical to run_multiseed_rel.evaluate. If it is
       not, the matched Dn-DQN row of Table X and row 1 of Table VII stop being measured by the same
       instrument and must not appear in one comparison.
    2. Both must agree with train_safe_dqn on the evaluation protocol constants (GAIN, EVAL_SEEDS,
       EVAL_EP). A shielded evaluator that ran a different number of episodes would not be
       comparable no matter how the sigma is computed.
    3. The environment the CRN patch targets must actually be in the MRO of the environment the
       evaluators construct, and must not be shadowed by an override.
    """
    ok = True
    m, f = _fn_ast_md5(S['M'].evaluate), _fn_ast_md5(S['F'].evaluate)
    same = (m == f)
    log(f"verify | evaluate() AST | run_multiseed_matched={m} run_multiseed_rel={f} | "
        f"{'MATCH — Table X matched rows and Table VII row 1 share one instrument' if same else 'MISMATCH'}")
    ok = ok and same
    sd = _fn_ast_md5(S['SD'].evaluate)
    log(f"verify | evaluate() AST | train_safe_dqn={sd} | expected to DIFFER (shield applied at "
        f"evaluation is what the published Safe-DQN baseline is)")

    consts = {}
    for name, mod in (('matched', S['M']), ('rel', S['F']), ('safe', S['SD'])):
        consts[name] = (mod.GAIN, tuple(mod.EVAL_SEEDS), mod.EVAL_EP)
        log(f"verify | protocol {name:8s} | GAIN={mod.GAIN} EVAL_SEEDS={list(mod.EVAL_SEEDS)} "
            f"EVAL_EP={mod.EVAL_EP}")
    agree = len(set(consts.values())) == 1
    log(f"verify | protocol constants | {'MATCH across all three modules' if agree else 'MISMATCH'}")
    ok = ok and agree

    NYU = S['env_6g_nyu'].SubTHzEnv6G_NYU
    base_in_mro = S['env_6g_v5'].SubTHzEnv6G in NYU.__mro__
    overrides = 'reset' in NYU.__dict__
    log(f"verify | CRN reach | SubTHzEnv6G_NYU MRO = {[c.__name__ for c in NYU.__mro__]}")
    log(f"verify | CRN reach | patched base in MRO={base_in_mro} | NYU overrides reset="
        f"{overrides} | {'PASS' if (base_in_mro and not overrides) else 'FAIL'}")
    ok = ok and base_in_mro and not overrides

    log(f"verify | {'PASS' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------------------------------
# worker: one (arm, seed) unit = R seeded repeats of one checkpoint
# --------------------------------------------------------------------------------------------
def run_unit(arm, seed):
    S = import_stack()
    T7.install_patch(S['env_6g_v5'])            # the SAME patch object Table VII used
    torch = S['torch']
    label, pat, loader, evk, drv, camp = ARMS[arm]
    ck = ckpt_for(arm, seed)
    if not os.path.exists(ck):
        log(f"[{arm} s{seed}] MISSING CHECKPOINT {ck} — refusing")
        return 3
    if not verify_evaluators(S):
        log(f"[{arm} s{seed}] REFUSING: evaluator identity check failed")
        return 4

    mod = S['M'] if evk == 'matched' else S['SD']
    if EVAL_EP_OVERRIDE:
        mod.EVAL_EP = EVAL_EP_OVERRIDE          # rehearsal only
    slow, med, fast, skip = S['tp'].load_trajectories(S['tp'].CSV_PATH)
    pool = slow + med
    tc = dict(python=sys.version.split()[0], torch=torch.__version__, numpy=S['np'].__version__)

    net = (S['load_dndqn_checkpoint'](ck)[0] if loader == 'dueling' else S['load_plain'](ck)[0])
    eval_md5 = _fn_ast_md5(mod.evaluate)

    def evaluate():
        return mod.evaluate(net, pool)

    n_eval_seeds, n_ep = len(mod.EVAL_SEEDS), mod.EVAL_EP
    log(f"[{arm} s{seed}] start | {ck} (md5 {md5(ck)}) | loader={loader} evaluator={evk} "
        f"| R={R} repeats | {n_eval_seeds} eval seeds x {n_ep} episodes | torch {tc['torch']} "
        f"| threads {torch.get_num_threads()} | pool {len(pool)}")

    os.makedirs(OUTDIR, exist_ok=True)
    rows = []
    for r in range(R):
        t0 = time.time()
        with T7.seeded(r):
            m = evaluate()
        row = {'arm': arm, 'label': label, 'seed': seed, 'ckpt': ck, 'ckpt_md5': md5(ck),
               'repeat': r, 'chan_seed_first': BASE + 100000 * r,
               'secs': round(time.time() - t0, 1), 'venue': VENUE,
               'python': tc['python'], 'torch': tc['torch'], 'numpy': tc['numpy'],
               'loader': loader, 'evaluator': evk, 'eval_ast_md5': eval_md5,
               'eval_seeds': n_eval_seeds, 'eval_ep': n_ep,
               'env_v5_md5': md5('env_6g_v5.py'), 'env_nyu_md5': md5('env_6g_nyu.py'),
               'train_driver_md5': md5(drv), 'campaign_md5': md5(camp),
               't7_md5': md5(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          'tablevii_r8.py')),
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


def run_pool(units, slots, floor_mb, t0, poll=20):
    env = dict(os.environ)
    env['OMP_NUM_THREADS'] = env['MKL_NUM_THREADS'] = '1'
    env['PYTHONPATH'] = os.getcwd() + os.pathsep + env.get('PYTHONPATH', '')
    env['T7_VENUE'] = VENUE
    for k in ('TX_TAG', 'TX_R', 'TX_EVAL_EP'):
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
            name = f'tx_r8_{arm}_s{seed}'
            fh = open(os.path.join('logs', name + '.log'), 'a', encoding='utf-8')
            cmd = [sys.executable, '-u', os.path.abspath(__file__), '--unit', arm, str(seed)]
            p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=os.getcwd())
            running[name] = (p, time.time(), fh, (arm, seed))
            log(f"START {name} pid {p.pid} ({free} MB free)")
        write_status('evaluate', done, total, running, t0)
        if queue or running:
            time.sleep(poll)
    return failed


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
def table():
    import numpy as np
    rows = merge()
    if not rows:
        print("no units present yet")
        return 1
    venues = {r['venue'] for r in rows}
    torches = {r['torch'] for r in rows}
    if len(venues) > 1:
        print(f"REFUSING: rows from MIXED VENUES {sorted(venues)} — a reported +/- must come from "
              f"one toolchain. Separate the files and rerun.")
        return 2
    print("=" * 124)
    print(f"  TABLE X — learned rows on the R={R} CRN grid | venue {sorted(venues)[0]} | "
          f"torch {sorted(torches)} | {len(rows)} evaluations | BASE={BASE}")
    print(f"  mean over {len(SEEDS)} training seeds; +/- is POPULATION sigma across TRAINING seeds "
          f"(ddof=0), each seed's value already averaged over R={R} CRN repeats")
    print("=" * 124)
    summary = {'venue': sorted(venues)[0], 'R': R, 'BASE': BASE, 'seeds': SEEDS,
               'n_rows': len(rows), 'torch': sorted(torches), 'arms': {},
               'generated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    lines = []
    for arm in ARM_ORDER:
        sub = [r for r in rows if r['arm'] == arm]
        if not sub:
            continue
        seeds_here = sorted({int(r['seed']) for r in sub})
        label = ARMS[arm][0]
        arm_out = {'label': label, 'seeds': seeds_here,
                   'repeats_per_seed': {str(s): sum(1 for r in sub if int(r['seed']) == s)
                                        for s in seeds_here}}
        cells = []
        print(f"\n  {label}   [{len(seeds_here)} seeds x "
              f"{len(sub) // max(len(seeds_here), 1)} repeats]")
        print(f"    {'metric':>6} | {'mean':>9} | {'sd_train':>9} | {'sd_eval':>9} | "
              f"{'1-pass mean':>11} | {'1-pass sd':>9} | {'d_mean':>8} | {'d_sd':>8}")
        for m in METRICS:
            per = np.array([np.mean([float(r[m]) for r in sub if int(r['seed']) == s])
                            for s in seeds_here])
            within = [np.std([float(r[m]) for r in sub if int(r['seed']) == s], ddof=1)
                      for s in seeds_here]
            mu, sd = per.mean(), per.std()                     # population sigma, paper convention
            se = float(np.sqrt(np.mean(np.square(within))))    # pooled within-checkpoint sd
            pm, ps, dp = SINGLEPASS[arm][m]
            print(f"    {m:>6} | {mu:9.4f} | {sd:9.4f} | {se:9.4f} | {pm:11.2f} | {ps:9.2f} | "
                  f"{mu - pm:+8.2f} | {sd - ps:+8.2f}")
            arm_out[m] = dict(mean=float(mu), sigma_train_pop=float(sd), sigma_eval_pooled=se,
                              per_seed=[float(x) for x in per],
                              singlepass_mean=pm, singlepass_sd=ps)
            cells.append(f"{mu:.{dp}f} $\\pm$ {sd:.{dp}f}")
        summary['arms'][arm] = arm_out
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    print("\n" + "=" * 124)
    print("  MANUSCRIPT ROWS (paste order: HO | Power | Spec | RLF | Tput)")
    print("=" * 124)
    for ln in lines:
        print("ROW: " + ln)

    os.makedirs('results', exist_ok=True)
    json.dump(summary, open(summary_path(), 'w'), indent=2)
    print(f"\n  summary -> {summary_path()}")
    print(f"  merged  -> {merged_path()}")

    expected = len(ARM_ORDER) * len(SEEDS) * R
    if len(rows) != expected:
        print(f"\n  INCOMPLETE: {len(rows)} of {expected} evaluations present — the rows above are "
              f"a partial table and are NOT the paper table.")
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
    global VENUE
    observed = T7.TORCH_VENUE.get(mm, f'unknown-torch{mm}')
    if observed != VENUE:
        log(f"venue | declared {VENUE!r} but torch {torch.__version__} says {observed!r} — "
            f"relabelling to {observed!r}; the toolchain is the evidence, the env var is not")
        VENUE = observed
        os.environ['T7_VENUE'] = observed
    if mm != EXPECT_TORCH and not allow:
        log(f"REFUSING: torch {torch.__version__} but every reported v15 number was produced on "
            f"torch {EXPECT_TORCH}.x. Table X must be measured on the reporting venue. "
            f"Pass --allow-toolchain-mismatch only for a rehearsal.")
        sys.exit(2)
    return torch.__version__


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--unit', nargs=2, metavar=('ARM', 'SEED'), default=None)
    ap.add_argument('--table', action='store_true')
    ap.add_argument('--verify', action='store_true')
    ap.add_argument('--slots', type=int, default=5)
    ap.add_argument('--floor-mb', type=int, default=1500)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--allow-toolchain-mismatch', action='store_true')
    ap.add_argument('--arms', default='', help='comma-separated subset of arms to run; '
                                               'default = all four. A table built from a subset '
                                               'is NOT the paper table and says so.')
    a = ap.parse_args()

    if a.verify:
        check_toolchain(True)
        S = import_stack()
        T7.install_patch(S['env_6g_v5'])
        T7.verify_patch()
        log(f"CRN rule | {T7.PATCH_SEMANTICS} | BASE={BASE} (imported from tablevii_r8, md5 "
            f"{md5(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tablevii_r8.py'))})")
        return 0 if verify_evaluators(S) else 1
    if a.unit:
        check_toolchain(a.allow_toolchain_mismatch)
        arm, seed = a.unit[0], int(a.unit[1])
        if arm not in ARMS:
            log(f"unknown arm {arm!r}; expected one of {ARM_ORDER}")
            return 2
        return run_unit(arm, seed)
    if a.table:
        check_toolchain(True)          # label-only: resolve VENUE from the toolchain, never refuse
        return table()

    t0 = time.time()
    log("=" * 118)
    tv = check_toolchain(a.allow_toolchain_mismatch)
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
        log(f"*** ARM SUBSET *** {arms} — NOT the full Table X run")
    units = [(arm, s) for arm in arms for s in SEEDS]
    log(f"TABLE X R={R} CRN | venue={VENUE} | {len(arms)} arms x {len(SEEDS)} seeds x R={R} "
        f"= {len(units) * R} evaluations | BASE={BASE}")
    log(f"toolchain | python {sys.version.split()[0]} | torch {tv} | cwd {os.getcwd()}")

    S = import_stack()
    T7.install_patch(S['env_6g_v5'])
    T7.verify_patch()
    if not verify_evaluators(S):
        log("REFUSING: evaluator identity check failed — see the lines above")
        return 1

    log("preflight | checkpoint inventory")
    missing = []
    for arm, s in units:
        ck = ckpt_for(arm, s)
        here = os.path.exists(ck)
        log(f"  {'OK     ' if here else 'MISSING'} {arm:9s} s{s} | {ck:20s} "
            f"| {md5(ck) if here else '-':32s} | {os.path.getsize(ck) if here else 0} B")
        if not here:
            missing.append((arm, s))
    if missing:
        log(f"REFUSING: {len(missing)} checkpoint(s) absent: {missing}. A partial grid would "
            f"silently print a table over fewer seeds than it claims.")
        return 1

    todo = [(arm, s) for arm, s in units if not unit_done(arm, s)]
    log(f"plan | {len(units) - len(todo)} unit(s) already complete, {len(todo)} to run "
        f"| slots={a.slots} floor={a.floor_mb} MB | free now {avail_mb()} MB")
    if a.dry_run:
        log("DRY RUN — launching nothing")
        for arm, s in todo:
            log(f"  would run {arm} s{s} -> {unit_path(arm, s)}")
        return 0

    failed = run_pool(todo, a.slots, a.floor_mb, t0)
    if failed:
        log(f"FAILED units: {failed} — the table below is partial")
        write_status('FAILED', len(units) - len(failed), len(units), {}, t0)
    else:
        write_status('COMPLETE', len(units), len(units), {}, t0)
    log(f"all units finished in {(time.time() - t0) / 60:.1f} min — building the table")
    return table()


if __name__ == '__main__':
    sys.exit(main())
