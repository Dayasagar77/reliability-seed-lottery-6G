#!/usr/bin/env python3
"""
srdndqn_r8.py — put Section IX-K's learned-shield ablation (rigor manuscript Table XVII) on the
SAME R=8 common-random-number evaluation grid that Tables VII and X use.

WHY THIS EXISTS
---------------
Table XVII as printed in DnDQN_manuscript_rigor.docx cannot be carried into the revision, for two
independent reasons found by re-deriving all nineteen of its cells from results/srdndqn_*.csv:

  1. DENOMINATOR.  All 19 published dispersion values match the BIASED (n) denominator; 0 of 19
     match the unbiased (k-1) denominator the locked reporting basis mandates.  Its own paired
     tests, meanwhile, use ddof=1 (that is how t=22.1336 and d_z=9.8985 reproduce exactly).  One
     section, two denominators.

  2. PROTOCOL.  Every srdndqn_*.csv row is ONE unseeded evaluation of one checkpoint.  Its spread
     therefore confounds training-seed variance with evaluation-seed variance — the exact defect
     the Associate Editor named:

         "some baselines are represented by a single trained model and others by evaluation-seed
          variability, making the comparisons methodologically inconsistent"

     Pasting Table XVII into the manuscript written to FIX that objection would reintroduce it.

This driver removes both hazards without retraining anything:

    4 arms x 5 seeds x 8 CRN repeats = 160 evaluations of the twenty EXISTING srdndqn_*.pt
    checkpoints.

After it, IX-K reports mean +/- sigma_train on the identical basis as every other learned row in
the paper, sigma_train = sqrt(max((MS_A - MS_E)/R, 0)), and the section becomes an asset rather
than an inconsistency.

VENUE — READ THIS BEFORE CHANGING EXPECT_TORCH
----------------------------------------------
Tables VII and X are LAPTOP-trained, so tablevii_r8.py and lottery_r8.py refuse to measure on any
other toolchain.  The twenty srdndqn checkpoints are CLOUD-trained: chunk_run.log records
"CHUNK | wall=400s | target 5x800,000", and the wall/resume path is the cloud path
(train_srdndqn.run_arm's docstring reserves the no-wall path for the laptop).  There is no
laptop-trained counterpart and manufacturing one would cost 20 x 800k steps.

So this grid's reporting venue is CLOUD, declared, and the refusal is inverted accordingly.  That
is defensible precisely because IX-K is SELF-CONTAINED: all four arms share one venue, one
toolchain and one evaluator, and the section's claims are internal paired contrasts between those
four arms.  What it must NOT do is cross-compare an IX-K number against a Table VII number; the
manuscript text has to say so, because per env_6g_v5.py:132 the trainer's channel is unseeded and
a same-named seed on another venue is a different network.

Evaluation itself is venue-portable and that is measured, not assumed: venue_diff.py matched
400/400 metric cells across torch 2.5.1+cu121 and 2.13.0+cu130.  A cloud-measured evaluation of a
cloud-trained checkpoint is therefore exactly what a laptop would have produced from the same
checkpoint.

WHAT IS SHARED WITH THE FROZEN INSTRUMENT
-----------------------------------------
tablevii_r8.py is FROZEN at md5 5fd895bf1560f2fe94715b8557b544a9.  Nothing here modifies it.  This
driver IMPORTS from it: the CRN patch object, seeded(), BASE, R_REAL, METRICS, md5(), avail_mb().
The evaluator is train_srdndqn.evaluate — the same function lottery_r8.py uses for its guarded arm
and the same one that produced results/srdndqn_*.csv, so this grid and the single-pass ledger it
refines are one instrument, not two.

Usage:
    python -u srdndqn_r8.py --verify          # provenance + CRN-reach checks, no measurement
    python -u srdndqn_r8.py --slots 1         # run the grid (resumable)
    python -u srdndqn_r8.py --table           # recompute rows from whatever units exist
    python -u srdndqn_r8.py --unit shielded 3 # one unit
Resume: rerun the same command.  A unit is done when its CSV holds R rows.
"""
import os, sys

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import argparse, ast, csv, hashlib, inspect, json, math, subprocess, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tablevii_r8 as T7                       # frozen; imported, never modified

BASE = T7.BASE
R_REAL = T7.R_REAL
# Table XVII's columns are RLF, guard-fire, Power saving, Spectrum saving, HO success.  'Override'
# IS the guard-fire rate, so it is a reported metric here even though Tables VII/X do not carry it.
METRICS = list(T7.METRICS) + ['Override']

# CLOUD is the reporting venue for THIS grid — see the VENUE note above.
EXPECT_TORCH = '2.13'
VENUE = os.environ.get('SR8_VENUE', T7.TORCH_VENUE.get(EXPECT_TORCH, 'cloud'))

SEEDS = list(range(5))

TAG = os.environ.get('SR8_TAG', '')
R = int(os.environ.get('SR8_R', R_REAL))
EVAL_EP_OVERRIDE = int(os.environ.get('SR8_EVAL_EP', 0))
REHEARSAL = bool(TAG) or R != R_REAL or bool(EVAL_EP_OVERRIDE)
_SFX = f'_{TAG}' if TAG else ''

# ---- the four IX-K arms -------------------------------------------------------------------
# key -> (manuscript label, ckpt pattern, ckpt origin tag)
# Loader is 'branched' and evaluator is train_srdndqn.evaluate for ALL FOUR: one instrument.
# The per-arm guard/shield configuration is NOT duplicated here — it is read at run time out of
# train_srdndqn.ARMS, so this file cannot drift from the code that trained the checkpoints.
ARMS = {
    'postguard':    ('Measured state guard (proposed, VI-B)',      'srdndqn_postguard_s{}.pt'),
    'shielded':     ('Shielded backup only (guard removed)',       'srdndqn_shielded_s{}.pt'),
    'shield_guard': ('Shielded backup + guard',                    'srdndqn_shield_guard_s{}.pt'),
    'lagrangian':   ('Lagrangian constraint + guard',              'srdndqn_lagrangian_s{}.pt'),
}
ARM_ORDER = ['postguard', 'shielded', 'shield_guard', 'lagrangian']
CKPT_ORIGIN = 'srdndqn-ablation-20260728'

# The SINGLE-PASS ledger these R=8 rows refine, read from its own file at table() time rather than
# transcribed.  A hand-copied baseline that drifts from its source is a fabrication risk.
SINGLEPASS_CSV = {a: f'results/srdndqn_{a}.csv' for a in ARMS}

# Arms whose guard is structurally disabled: Override is an exact zero by construction, not a
# measurement, and must be printed as a dash rather than as 0.00 +/- 0.00.
GUARD_DISABLED = {'shielded'}

# ---------------------------------------------------------------------------------------------
# CODE IDENTITY.  These are the CLOUD bytes -- the exact files that trained the twenty
# srdndqn_*.pt checkpoints on 28-29 Jul and that will now evaluate them.
#
# They are NOT the values pinned in lottery_r8.py.  That file pins the LAPTOP artefact, because
# the grid it produces joins Table VII, which was measured on the laptop.  Two files differ:
#
#     train_srdndqn.py      laptop 9a490505efa6d2b31a0999264973e86a
#                            cloud c3fb4393b1bc148e441c15a85400c1d1
#     env_6g_branched.py    laptop 6a998fae24fb55e3b645d66606b2a72a
#                            cloud 443386f642d19bb8eeffbc9c74b2d853
#
# The cloud files are a strict SUPERSET of the laptop files: 29 Jul added the shield_guard and
# lagrangian arms, the Lagrangian multiplier `rlf_pen_scale` (default 1.0, so the reward is
# unchanged for every other arm), the info key 'guard_override', and the Override counters in
# evaluate() (which consume no random numbers).  Pinning the laptop values here would refuse a
# grid measured with the only code that can produce it, and pinning them WITHOUT saying so would
# be worse.  The superset claim is not asserted from reading the diff -- it is measured by
# _equiv/equiv_eval_crn.py, which runs the same checkpoint under both versions at identical CRN
# seeds; that measurement, not this comment, is the warrant for these pins.
# ---------------------------------------------------------------------------------------------
PIN = {
    'train_srdndqn.py':   'c3fb4393b1bc148e441c15a85400c1d1',
    'env_6g_v5.py':       'fc31bce91a3da89b735bcefdfc326d32',
    'env_6g_branched.py': '443386f642d19bb8eeffbc9c74b2d853',
    'dndqn_branched.py':  'd023c86cb486ff9b62b530c8568f8cbf',
    'multi_seed_eval.py': 'fce9a282713385c68fd1455c9b3076b8',
    'tablevii_r8.py':     '5fd895bf1560f2fe94715b8557b544a9',
}

# The laptop predecessor of each pinned file that has one, logged beside the live value so an
# auditor sees the divergence in the run log rather than having to go looking for it.
PIN_LAPTOP = {
    'train_srdndqn.py':   '9a490505efa6d2b31a0999264973e86a',
    'env_6g_branched.py': '6a998fae24fb55e3b645d66606b2a72a',
}

# Set by --equiv-proof <path>: the recorded result of the laptop/cloud equivalence measurement.
EQUIV_PROOF = 'results/equiv_crn_verdict.json'

FIELDS = ['arm', 'label', 'seed', 'ckpt', 'ckpt_origin', 'ckpt_md5', 'repeat', 'chan_seed_first',
          'secs'] + METRICS + \
    ['RLF_std', 'venue', 'python', 'torch', 'numpy', 'binary_power', 'guard_db', 'backup_shield',
     'guard_mode', 'loader', 'evaluator', 'eval_ast_md5', 'eval_seeds', 'eval_ep', 'env_v5_md5',
     'env_branched_md5', 'driver_md5', 't7_md5', 'measured_utc']

OUTDIR = f'results/srdndqn_r8{_SFX}'
STATUS = f'srdndqn_r8{_SFX}.status'
LOG = f'srdndqn_r8{_SFX}.log'

md5 = T7.md5
avail_mb = T7.avail_mb


def merged_path():
    """Built at CALL time.  A module-level f-string would freeze the UNVERIFIED default venue into
    the filename, which is how a table of cloud measurements ends up named '..._laptop.csv'."""
    return f'results/srdndqn_r8{_SFX}_{VENUE}.csv'


def summary_path():
    return f'results/srdndqn_r8{_SFX}_summary_{VENUE}.json'


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


def _here(name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)


def import_stack():
    """train_srdndqn.py touches sys.argv only inside main(), but neutralising argv costs nothing
    and protects against an import-time argv parser being added later."""
    saved = sys.argv
    sys.argv = [saved[0]]
    try:
        import torch
        torch.set_num_threads(1)
        import numpy as np
        import env_6g_v5
        import env_6g_branched
        import train_phase2 as tp
        from dndqn import OBS_DIM
        from dndqn_branched import load_branched
        import train_srdndqn as S
        return dict(torch=torch, np=np, env_6g_v5=env_6g_v5, env_6g_branched=env_6g_branched,
                    tp=tp, OBS_DIM=OBS_DIM, load_branched=load_branched, S=S)
    finally:
        sys.argv = saved


def _fn_ast_md5(fn):
    """md5 of the function's ABSTRACT SYNTAX TREE, so formatting differences cannot mask or fake a
    difference in behaviour."""
    src = inspect.getsource(fn)
    node = ast.parse(src).body[0]
    return hashlib.md5(ast.dump(node, annotate_fields=True,
                                include_attributes=False).encode()).hexdigest()


def _crn_reach(cls, base):
    """Walk the MRO from the concrete environment class up to the patched base class.  Every class
    on the way that defines its own reset() must forward the seed to super(), or the CRN feed is
    silently swallowed and all R repeats become identical -- which would print a grid with zero
    evaluation variance and look like a spectacularly precise result."""
    if base not in cls.__mro__:
        return False, f"patched base {base.__name__} NOT in MRO of {cls.__name__}"
    notes = []
    for c in cls.__mro__:
        if c is base:
            break
        if 'reset' in c.__dict__:
            try:
                src = inspect.getsource(c.__dict__['reset'])
            except (OSError, TypeError):
                return False, f"{c.__name__}.reset overrides and its source is unreadable"
            fwd = ('super().reset(' in src) and ('seed=seed' in src)
            notes.append(f"{c.__name__}.reset overrides -> forwards seed={fwd}")
            if not fwd:
                return False, "; ".join(notes) + "  <-- BREAKS THE CRN CHAIN"
    return True, ("; ".join(notes) if notes else "no intervening reset override")


def verify(S, deep=True):
    ok = True

    # 1  code identity
    for name, want in PIN.items():
        live = md5(_here(name) if os.path.exists(_here(name)) else name)
        good = (live == want)
        lap = PIN_LAPTOP.get(name)
        note = f" | laptop predecessor {lap} (superset; see equivalence proof)" if lap else ''
        log(f"verify | pin {name:22s} {live} {'OK' if good else 'DRIFT expected ' + want}{note}")
        ok = ok and good

    # 1b  the equivalence proof that licenses the two cloud-side pins must EXIST and have PASSED.
    #     Without it the pins above are a claim about the 29-Jul diff, and this campaign exists
    #     because a claim was once accepted where a measurement was required.
    if PIN_LAPTOP:
        p = _here(EQUIV_PROOF) if os.path.exists(_here(EQUIV_PROOF)) else EQUIV_PROOF
        if not os.path.exists(p):
            log(f"verify | equivalence proof | {EQUIV_PROOF} ABSENT — the cloud pins for "
                f"{sorted(PIN_LAPTOP)} are unwarranted until _equiv/equiv_eval_crn.py has been "
                f"run under BOTH code versions and its verdict recorded")
            ok = False
        else:
            try:
                v = json.load(open(p, encoding='utf-8'))
            except (OSError, ValueError) as e:
                log(f"verify | equivalence proof | {EQUIV_PROOF} unreadable: {e}")
                v = {}
                ok = False
            good = (v.get('equivalence') == 'PASS') and (v.get('sensitivity') == 'PASS')
            log(f"verify | equivalence proof | {EQUIV_PROOF} | equivalence="
                f"{v.get('equivalence')} sensitivity={v.get('sensitivity')} "
                f"worst|d|={v.get('worst_abs_diff')} over {v.get('metrics_compared')} metrics x "
                f"{v.get('repeats')} repeats | {'OK' if good else 'FAIL'}")
            ok = ok and good

    # 2  every checkpoint present, and no two arms pointing at one file
    seen = {}
    for arm in ARM_ORDER:
        for s in SEEDS:
            ck = ckpt_for(arm, s)
            if not os.path.exists(ck):
                log(f"verify | ckpt MISSING {ck}")
                ok = False
                continue
            h = md5(ck)
            if h in seen:
                log(f"verify | ckpt COLLISION {ck} has the same md5 as {seen[h]} — two arms "
                    f"cannot be the same network")
                ok = False
            seen[h] = ck
    log(f"verify | checkpoints | {len(seen)} distinct md5s over "
        f"{len(ARM_ORDER) * len(SEEDS)} expected")
    ok = ok and (len(seen) == len(ARM_ORDER) * len(SEEDS))

    # 3  the arm table in THIS file agrees with the arm table in the training code
    for arm in ARM_ORDER:
        if arm not in S['S'].ARMS:
            log(f"verify | arm {arm!r} is not in train_srdndqn.ARMS — cannot resolve its guard")
            ok = False
            continue
        gdb, shield = S['S'].ARMS[arm]
        disabled_here = arm in GUARD_DISABLED
        disabled_code = (gdb is None)
        agree = (disabled_here == disabled_code)
        log(f"verify | arm {arm:13s} guard_db={gdb} backup_shield={shield} | "
            f"guard-disabled: this file={disabled_here} code={disabled_code} "
            f"{'OK' if agree else 'DISAGREE'}")
        ok = ok and agree

    # 4  evaluation protocol constants
    log(f"verify | protocol | GAIN={S['S'].GAIN} EVAL_SEEDS={list(S['S'].EVAL_SEEDS)} "
        f"EVAL_EP={S['S'].EVAL_EP} SAVE_LINE={S['S'].SAVE_LINE} GUARD_DB={S['S'].GUARD_DB}")

    # 5  the CRN patch actually reaches the environment the evaluator builds
    good, why = _crn_reach(S['env_6g_branched'].BranchedEnv6G, S['env_6g_v5'].SubTHzEnv6G)
    log(f"verify | CRN reach | BranchedEnv6G MRO="
        f"{[c.__name__ for c in S['env_6g_branched'].BranchedEnv6G.__mro__[:4]]} | {why} "
        f"{'OK' if good else 'FAIL'}")
    ok = ok and good

    log(f"verify | evaluate() AST md5 = {_fn_ast_md5(S['S'].evaluate)} "
        f"(train_srdndqn.evaluate — the same function that wrote results/srdndqn_*.csv)")

    # 6  deep: the CRN feed actually moves the answer.  Two different repeats of one checkpoint
    #    must differ; two identical repeats must match to the bit.  This is the check that
    #    catches a swallowed seed, which would otherwise print a grid of zero evaluation variance.
    if deep:
        ck = ckpt_for('postguard', 0)
        if os.path.exists(ck):
            t0 = time.time()
            slow, med, _f, _s = S['tp'].load_trajectories(S['tp'].CSV_PATH)
            pool = slow + med
            net = S['load_branched'](ck, S['OBS_DIM'])[0]
            with T7.seeded(0):
                a0 = S['S'].evaluate(net, pool, 'postguard')
            with T7.seeded(0):
                a0b = S['S'].evaluate(net, pool, 'postguard')
            with T7.seeded(1):
                a1 = S['S'].evaluate(net, pool, 'postguard')
            same = max(abs(float(a0[k]) - float(a0b[k])) for k in METRICS)
            diff = max(abs(float(a0[k]) - float(a1[k])) for k in METRICS)
            det = same < 1e-12
            moves = diff > 1e-9
            log(f"verify | CRN determinism | repeat 0 twice, worst |d| = {same:.3e} "
                f"{'PASS (bit-identical)' if det else 'FAIL (not reproducible)'}")
            log(f"verify | CRN sensitivity | repeat 0 vs 1, worst |d| = {diff:.3e} "
                f"{'PASS (the seed reaches the channel)' if moves else 'FAIL (seed swallowed)'}")
            log(f"verify | deep pass took {time.time() - t0:.0f}s")
            ok = ok and det and moves
        else:
            log(f"verify | deep skipped: {ck} absent")
            ok = False

    log(f"verify | {'PASS' if ok else 'FAIL'}")
    return ok


def run_unit(arm, seed):
    S = import_stack()
    T7.install_patch(S['env_6g_v5'])            # the SAME patch object Tables VII and X used
    torch = S['torch']
    label, pat = ARMS[arm]
    ck = ckpt_for(arm, seed)
    if not os.path.exists(ck):
        log(f"[{arm} s{seed}] MISSING CHECKPOINT {ck} — refusing")
        return 3
    if not verify(S, deep=False):               # cheap checks in the worker; parent did the deep one
        log(f"[{arm} s{seed}] REFUSING: verification failed")
        return 4

    if EVAL_EP_OVERRIDE:                        # rehearsal only
        S['S'].EVAL_EP = EVAL_EP_OVERRIDE
    slow, med, fast, skip = S['tp'].load_trajectories(S['tp'].CSV_PATH)
    pool = slow + med
    tc = dict(python=sys.version.split()[0], torch=torch.__version__, numpy=S['np'].__version__)

    net = S['load_branched'](ck, S['OBS_DIM'])[0]
    gdb, shield = S['S'].ARMS[arm]              # resolved from the training code, not duplicated
    eval_md5 = _fn_ast_md5(S['S'].evaluate)
    ctrl = {'binary_power': True, 'guard_db': ('' if gdb is None else gdb),
            'backup_shield': shield, 'guard_mode': ('' if gdb is None else 'state')}

    n_eval_seeds, n_ep = len(S['S'].EVAL_SEEDS), S['S'].EVAL_EP
    log(f"[{arm} s{seed}] start | {ck} (md5 {md5(ck)}) | guard_db={gdb} shield={shield} "
        f"| R={R} repeats | {n_eval_seeds} eval seeds x {n_ep} episodes | torch {tc['torch']} "
        f"| threads {torch.get_num_threads()} | pool {len(pool)}")

    os.makedirs(OUTDIR, exist_ok=True)
    rows = []
    for r in range(R):
        t0 = time.time()
        with T7.seeded(r):
            m = S['S'].evaluate(net, pool, arm)
        row = {'arm': arm, 'label': label, 'seed': seed, 'ckpt': ck,
               'ckpt_origin': CKPT_ORIGIN, 'ckpt_md5': md5(ck), 'repeat': r,
               'chan_seed_first': BASE + 100000 * r, 'secs': round(time.time() - t0, 1),
               'RLF_std': (f"{float(m['RLF_std']):.4f}" if 'RLF_std' in m else ''),
               'venue': VENUE, 'python': tc['python'], 'torch': tc['torch'], 'numpy': tc['numpy'],
               'loader': 'branched', 'evaluator': 'srdndqn', 'eval_ast_md5': eval_md5,
               'eval_seeds': n_eval_seeds, 'eval_ep': n_ep,
               'env_v5_md5': md5('env_6g_v5.py'),
               'env_branched_md5': md5('env_6g_branched.py'),
               'driver_md5': md5('train_srdndqn.py'),
               't7_md5': md5(_here('tablevii_r8.py')),
               'measured_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
        row.update(ctrl)
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
    env['SR8_VENUE'] = VENUE
    for k in ('SR8_TAG', 'SR8_R', 'SR8_EVAL_EP'):
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
            name = f'sr8_{arm}_s{seed}'
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


def read_singlepass(arm):
    """The 5-seed single-pass ledger this arm's R=8 grid refines, read from its own file.  Its sd
    is reported alongside for comparison ONLY — it is the quantity being replaced, not a result."""
    path = SINGLEPASS_CSV[arm]
    if not os.path.exists(path):
        return None
    out = {m: [] for m in METRICS}
    try:
        for r in csv.DictReader(open(path, newline='')):
            for m in METRICS:
                v = r.get(m, '')
                out[m].append(float(v) if v not in ('', None) else float('nan'))
    except (OSError, ValueError):
        return None
    return out if all(len(v) == len(SEEDS) for v in out.values()) else None


def decompose(X):
    """Two-way crossed decomposition, seeds x CRN repeats, no replication within a cell.
    Identical arithmetic to tablevii_decompose.decompose() and lottery_r8.decompose(), reproduced
    here so the laptop needs no scipy.  The exact F pivot, one-sided bounds and resolvability
    verdict are computed downstream by that module, so all tables are analysed by one code path."""
    import numpy as np
    X = np.asarray(X, dtype=float)
    k, r = X.shape
    gm = X.mean()
    a, b = X.mean(axis=1), X.mean(axis=0)
    ss_a = r * float(np.sum((a - gm) ** 2))
    ss_b = k * float(np.sum((b - gm) ** 2))
    ss_e = float(np.sum((X - a[:, None] - b[None, :] + gm) ** 2))
    df_a, df_b, df_e = k - 1, r - 1, (k - 1) * (r - 1)
    ms_a, ms_b, ms_e = ss_a / df_a, ss_b / df_b, ss_e / max(df_e, 1)
    v_train = (ms_a - ms_e) / r
    v_block = (ms_b - ms_e) / k
    return dict(k=k, r=r, mean=float(gm), seed_means=[float(x) for x in a],
                ms_a=ms_a, ms_b=ms_b, ms_e=ms_e, df_a=df_a, df_b=df_b, df_e=df_e,
                v_train=v_train, sd_train=math.sqrt(max(v_train, 0.0)),
                v_block=v_block, sd_block=math.sqrt(max(v_block, 0.0)),
                sd_resid=math.sqrt(max(ms_e, 0.0)),
                F=(ms_a / ms_e if ms_e > 0 else float('inf')))


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
    print("=" * 134)
    print(f"  SECTION IX-K LEARNED-SHIELD ABLATION on the R={R} CRN grid | venue "
          f"{sorted(venues)[0]} | torch {sorted(torches)} | {len(rows)} evaluations | BASE={BASE}")
    print(f"  mean over {len(SEEDS)} training seeds. sd_train = sqrt(max((MS_A - MS_E)/R, 0)) — the")
    print(f"  sample sd with the evaluation-noise contribution subtracted. This REPLACES the")
    print(f"  published Table XVII dispersion, which used the biased (n) denominator on a single")
    print(f"  unseeded pass per seed.")
    print(f"  1-pass columns are the SAME twenty checkpoints measured once, unseeded")
    print(f"  (results/srdndqn_<arm>.csv): shown for comparison, not for reporting.")
    print("=" * 134)
    summary = {'venue': sorted(venues)[0], 'R': R, 'BASE': BASE, 'seeds': SEEDS,
               'n_rows': len(rows), 'torch': sorted(torches), 'arms': {},
               'guard_disabled_arms': sorted(GUARD_DISABLED),
               'generated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    lines = []
    for arm in ARM_ORDER:
        sub = [r for r in rows if r['arm'] == arm]
        if not sub:
            continue
        seeds_here = sorted({int(r['seed']) for r in sub})
        reps = sorted({int(r['repeat']) for r in sub})
        label = ARMS[arm][0]
        sp = read_singlepass(arm)
        arm_out = {'label': label, 'seeds': seeds_here, 'repeats': reps,
                   'ckpts': sorted({r['ckpt'] for r in sub}),
                   'singlepass_csv': SINGLEPASS_CSV[arm]}
        cells = {}
        print(f"\n  {label}   [{len(seeds_here)} seeds x {len(reps)} repeats]")
        print(f"    {'metric':>8} | {'mean':>9} | {'sd_train':>9} | {'sd_ddof1':>9} | "
              f"{'sd_resid':>9} | {'F(A/E)':>10} | {'1p mean':>9} | {'1p sd(n)':>9} | "
              f"{'1p sd(n-1)':>10}")
        for m in METRICS:
            X = [[float(r[m]) for r in sorted((q for q in sub if int(q['seed']) == s),
                                              key=lambda q: int(q['repeat']))]
                 for s in seeds_here]
            flat = [v for rowx in X for v in rowx]
            structural_zero = (m == 'Override' and arm in GUARD_DISABLED
                               and max(abs(v) for v in flat) == 0.0)
            d = decompose(X)
            sd1 = float(np.std(d['seed_means'], ddof=1))
            pm = psn = ps1 = float('nan')
            if sp and not all(math.isnan(v) for v in sp[m]):
                vals = [v for v in sp[m] if not math.isnan(v)]
                pm = float(np.mean(vals))
                psn = float(np.std(vals, ddof=0))
                ps1 = float(np.std(vals, ddof=1)) if len(vals) > 1 else float('nan')
            print(f"    {m:>8} | {d['mean']:9.4f} | {d['sd_train']:9.4f} | {sd1:9.4f} | "
                  f"{d['sd_resid']:9.4f} | {d['F']:10.2f} | {pm:9.3f} | {psn:9.3f} | {ps1:10.3f}"
                  + ("   <- structural zero (guard disabled): report as an em-dash"
                     if structural_zero else ""))
            arm_out[m] = dict(mean=d['mean'], sd_train=d['sd_train'], sd_ddof1=sd1,
                              sd_resid=d['sd_resid'], sd_block=d['sd_block'], F=d['F'],
                              df=(d['df_a'], d['df_b'], d['df_e']),
                              ms_a=d['ms_a'], ms_b=d['ms_b'], ms_e=d['ms_e'],
                              v_train=d['v_train'], seed_means=d['seed_means'],
                              structural_zero=structural_zero,
                              singlepass_mean=pm, singlepass_sd_ddof0=psn,
                              singlepass_sd_ddof1=ps1,
                              grid=[[float(v) for v in rowx] for rowx in X])
            cells[m] = ('---' if structural_zero
                        else f"{d['mean']:.2f} $\\pm$ {d['sd_train']:.2f}")
        summary['arms'][arm] = arm_out
        # Table XVII column order: RLF | guard-fire | Power saving | Spectrum saving | HO success
        order = ['RLF', 'Override', 'Power', 'Spec', 'HO']
        lines.append(f"| {label} | " + " | ".join(cells.get(m, 'n/a') for m in order) + " |")

    print("\n" + "=" * 134)
    print("  TABLE XVII REPLACEMENT ROWS, sd_train basis")
    print("  (column order: RLF | guard-fire | Power saving | Spectrum saving | HO success)")
    print("=" * 134)
    for ln in lines:
        print("ROW: " + ln)

    os.makedirs('results', exist_ok=True)
    json.dump(summary, open(summary_path(), 'w'), indent=2)
    print(f"\n  summary -> {summary_path()}")
    print(f"  merged  -> {merged_path()}")
    print("\n  NOTE: the paired necessity/sufficiency contrasts are computed downstream from the")
    print("        per-seed means in this summary, NOT from these row cells.")

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
        os.environ['SR8_VENUE'] = observed
    if mm != EXPECT_TORCH and not allow:
        log(f"REFUSING: torch {torch.__version__}, but the twenty srdndqn checkpoints were "
            f"TRAINED on torch {EXPECT_TORCH}.x (cloud) and this grid's reporting venue is "
            f"therefore cloud. Measuring them elsewhere is legitimate — evaluation is "
            f"venue-portable — but the venue label must then be set deliberately, not by "
            f"accident. Pass --allow-toolchain-mismatch if that is what you intend.")
        sys.exit(2)
    return torch.__version__


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--unit', nargs=2, metavar=('ARM', 'SEED'), default=None)
    ap.add_argument('--table', action='store_true')
    ap.add_argument('--verify', action='store_true')
    ap.add_argument('--slots', type=int, default=1)
    ap.add_argument('--floor-mb', type=int, default=1500)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--allow-toolchain-mismatch', action='store_true')
    ap.add_argument('--arms', default='', help='comma-separated subset; default = all four. A '
                                               'table built from a subset is NOT the paper table '
                                               'and says so.')
    a = ap.parse_args()

    if a.verify:
        check_toolchain(True)
        S = import_stack()
        T7.install_patch(S['env_6g_v5'])
        T7.verify_patch()
        log(f"CRN rule | {T7.PATCH_SEMANTICS} | BASE={BASE} (imported from tablevii_r8, md5 "
            f"{md5(_here('tablevii_r8.py'))})")
        return 0 if verify(S, deep=True) else 1

    if a.unit:
        check_toolchain(a.allow_toolchain_mismatch)
        arm, seed = a.unit[0], int(a.unit[1])
        if arm not in ARMS:
            log(f"unknown arm {arm!r}; expected one of {ARM_ORDER}")
            return 2
        return run_unit(arm, seed)

    if a.table:
        return table()

    tv = check_toolchain(a.allow_toolchain_mismatch)
    arms = [x for x in (a.arms.split(',') if a.arms else ARM_ORDER) if x]
    bad = [x for x in arms if x not in ARMS]
    if bad:
        log(f"unknown arm(s) {bad}; expected a subset of {ARM_ORDER}")
        return 2
    units = [(arm, s) for arm in arms for s in SEEDS if not unit_done(arm, s)]
    already = len(arms) * len(SEEDS) - len(units)
    log(f"grid | {len(arms)} arms x {len(SEEDS)} seeds x R={R} = "
        f"{len(arms) * len(SEEDS) * R} evaluations | {already} units already complete | "
        f"{len(units)} to run | venue {VENUE} | torch {tv}"
        + ("  [REHEARSAL — NOT the paper table]" if REHEARSAL else ""))
    if a.dry_run:
        for arm, s in units:
            log(f"  would run {arm} s{s} -> {unit_path(arm, s)} (ckpt {ckpt_for(arm, s)})")
        return 0

    S = import_stack()
    T7.install_patch(S['env_6g_v5'])
    T7.verify_patch()
    if not verify(S, deep=True):
        log("REFUSING to measure: verification failed")
        return 1
    del S

    t0 = time.time()
    failed = run_pool(units, a.slots, a.floor_mb, t0)
    if failed:
        log(f"FAILED units: {failed}")
    log(f"grid complete in {(time.time() - t0) / 60:.1f}m")
    return table()


if __name__ == '__main__':
    sys.exit(main())
