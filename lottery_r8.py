#!/usr/bin/env python3
"""
lottery_r8.py — put the TEN-SEED seed-lottery experiment on the same R=8 common-random-number
evaluation grid that Tables VII and X now use.

WHY THIS EXISTS: "the two different tens"
-----------------------------------------
Two separate quantities in this paper are both called "ten trainings of the flat C5 controller",
and until this driver runs they are NOT the same ten models:

  (i)  the POOLED n=10 row of Table X is  ms_rel_s0..s4  (Table VII arm 'flat')
                                        U m5_dndqn_s0..s4 (Table X arm 'm5_dndqn'),
       both measured on the R=8 CRN grid, so its +/- is sigma_train with the evaluation
       component averaged down by sqrt(8);

  (ii) the seed-lottery variance claim (Brown-Forsythe W=7.9512, p=0.0114, ratio 9.4x) is
       lot_flat_s0..s9  vs  gd_s0..s9, each measured ONCE, unseeded, single pass, so its
       spread confounds training-seed variance with evaluation-seed variance.

Those two sets share NO checkpoints and were measured by different protocols. Printing one
table whose flat row means (i) and a results subsection whose variance claim means (ii), both
described to the reader as "ten independent trainings", is exactly the inconsistency the
Associate Editor objected to:

    "some baselines are represented by a single trained model and others by evaluation-seed
     variability, making the comparisons methodologically inconsistent"

This driver removes the hazard by measuring the lottery's twenty checkpoints on the identical
R=8 CRN grid:

    2 arms x 10 seeds x 8 repeats = 160 evaluations.

After it, the lottery arms carry sigma_train on the same basis as every other learned row, the
variance claim and the pooled row can be reported without either one borrowing the other's
protocol, and k on the headline ratio doubles from 5 to 10.

WHAT IS AND IS NOT SHARED WITH tablevii_r8.py
---------------------------------------------
tablevii_r8.py is FROZEN at md5 5fd895bf1560f2fe94715b8557b544a9. Nothing here modifies it.
This driver IMPORTS it and calls its patch, so the channel seeds are produced by literally the
same code objects that produced Table VII's and Table X's -- never by a copy that could drift:

    T7.install_patch(env_6g_v5)      # patches env_6g_v5.SubTHzEnv6G.reset
    with T7.seeded(r): ...           # seed = T7.BASE + 100000*r + n  when reset(seed=None)

Everything that must NOT be shared -- output directory, merged CSV, summary JSON, status file,
log -- has its own name here, so a lottery run can never overwrite or be mistaken for the
Table VII or Table X artefact.

THE EVALUATORS, AND WHY THEY ARE THE RIGHT ONES
-----------------------------------------------
Each arm is evaluated by the same evaluate() OBJECT its own campaign driver used, imported from
the unmodified published module -- never reimplemented here:

    lot_flat   ->  run_multiseed_rel.evaluate            (campaign_lottery.py: `import run_multiseed_rel as R`
                                                          ... `met = R.evaluate(...)`)
    lot_guard  ->  train_srdndqn.evaluate(., ., 'postguard')
                                                         (campaign_guard.py: `import train_srdndqn as S`
                                                          ... `met = S.evaluate(net, pool, ARM)`)

run_multiseed_rel.evaluate is the SAME OBJECT tablevii_r8.py uses for Table VII row 1, so the
lottery's flat arm and the published flat row are measured by one instrument. verify() proves
this from the released artefact: it reads driver_md5 out of results/tablevii_r8_laptop.csv for
arm 'flat' and requires the live run_multiseed_rel.py to hash to the same value.

train_srdndqn.evaluate(., ., 'postguard') constructs
    BranchedEnv6G(traj, binary_power=True, guard_db=9.0, guard_mode='state', array_gain_dbi=GAIN,
                  w_mob=W_MOB, w_pwr=W_PWR, w_spec=W_SPEC)
which is exactly what train_branched_nyu.evaluate_branched(net, pool, True, guard_db=9.0,
guard_mode='state') -- the instrument behind Table VII's 'guarded' row -- constructs, and the
episode loops are line-for-line identical (srdndqn returns one extra key, RLF_std). Their ASTs
therefore DIFFER while their behaviour does not, so verify() does not compare ASTs here: it
proves the equivalence EMPIRICALLY by running both evaluators on gd_s0.pt inside one CRN repeat
and requiring every metric to agree to < 1e-9. An assertion read off the source is not evidence;
an agreement to 1e-9 on 750 episodes is.

CHECKPOINT LOADERS
------------------
    lot_flat   lot_flat_s{}.pt   flat dueling DQN  -> dndqn.load_dndqn_checkpoint
    lot_guard  gd_s{}.pt         branched dueling  -> dndqn_branched.load_branched(ck, OBS_DIM)

taken from campaign_lottery.py:`from dndqn import load_dndqn_checkpoint` and
campaign_guard.py:`from dndqn_branched import load_branched` / `from dndqn import OBS_DIM`,
not guessed. Note the guarded arm's checkpoints are named gd_s*.pt, NOT postguard_*.pt: the arm
LABEL and the file NAME differ, and campaign_guard.py:124 is the only authority on which is which
(`ck = f"gd_s{seed}.pt"   # own namespace: never touches srdndqn_*`).

WHAT THIS DRIVER DOES NOT DO
----------------------------
It does not compute p-values or confidence bounds. sigma_train and the F ratio need no special
functions and are computed here; the exact F pivot, the one-sided bounds and the resolvability
verdict are produced downstream by the SAME tablevii_decompose.decompose() that produced them for
Tables VII, VIII and X, so all four tables are analysed by one piece of code rather than by four
re-derivations that could disagree.

USAGE (PowerShell, from the folder holding the drivers)
    python -u lottery_r8.py                 # all 160
    python -u lottery_r8.py --verify        # patch + provenance + evaluator equivalence, no grid
    python -u lottery_r8.py --dry-run       # print the plan, launch nothing
    python -u lottery_r8.py --table         # recompute rows from whatever units exist
Resume: rerun the same command. A unit is done when its CSV holds R rows.
"""
import os, sys

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import argparse, ast, csv, hashlib, inspect, json, math, subprocess, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tablevii_r8 as T7                       # frozen; imported, never modified

# VENUE is a LABEL derived from evidence: the env var is only a default and check_toolchain()
# overwrites it from the observed torch version.
VENUE = os.environ.get('T7_VENUE', 'laptop')
BASE = T7.BASE                                  # 20260801 -- the same constant, not a copy of it
EXPECT_TORCH = T7.EXPECT_TORCH                  # '2.5' -> the reporting venue
R_REAL = T7.R_REAL                              # 8
METRICS = list(T7.METRICS)                      # HO Power Spec RLF Tput

SEEDS = list(range(10))                         # the lottery is TEN trainings, not five

TAG = os.environ.get('LR8_TAG', '')
R = int(os.environ.get('LR8_R', R_REAL))
EVAL_EP_OVERRIDE = int(os.environ.get('LR8_EVAL_EP', 0))
REHEARSAL = bool(TAG) or R != R_REAL or bool(EVAL_EP_OVERRIDE)
_SFX = f'_{TAG}' if TAG else ''

# ---- the two lottery arms ---------------------------------------------------------------------
# key -> (manuscript label, ckpt pattern, loader, evaluator key, training driver, campaign driver,
#         ckpt origin tag)
ARMS = {
    'lot_flat':  ('Flat Dn-DQN (6-action), reliability tilt C5 — 10-seed lottery',
                  'lot_flat_s{}.pt', 'flat',     'rel',
                  'run_multiseed_rel.py', 'campaign_lottery.py', 'lottery-20260801'),
    'lot_guard': ('Branched binary + 9 dB state guard (proposed) — 10-seed lottery',
                  'gd_s{}.pt',       'branched', 'srdndqn',
                  'train_srdndqn.py',     'campaign_guard.py',   'lottery-20260801'),
}
ARM_ORDER = ['lot_flat', 'lot_guard']

# The SINGLE-PASS ledgers these R=8 rows refine. Read at table() time from the CSVs below rather
# than transcribed, so no hand-copied number can drift from its source file.
SINGLEPASS_CSV = {
    'lot_flat':  ('results/lottery_flat_10seed_800k_laptop.csv',  None),
    'lot_guard': ('results/lottery_guard_10seed_800k_laptop.csv', ('arm', 'postguard')),
}

# Provenance the released Table VII artefact must agree with. Checked against the CSV at run time
# where the artefact is present; these literals are the fallback and the human-readable record.
T7_CSV = 'results/tablevii_r8_laptop.csv'
PIN = {
    'run_multiseed_rel.py':  '6aca670e1badcff15d9ada8aec10ee82',
    'train_branched_nyu.py': 'a4991bd7a89ddb1e1581d347d7a15127',
    'train_srdndqn.py':      '9a490505efa6d2b31a0999264973e86a',
    'env_6g_v5.py':          'fc31bce91a3da89b735bcefdfc326d32',
    'env_6g_branched.py':    '6a998fae24fb55e3b645d66606b2a72a',
    'env_6g_nyu.py':         '62803a7e0adc16bd7a19bf2b4cee68ab',
    'dndqn_branched.py':     'd023c86cb486ff9b62b530c8568f8cbf',
    'multi_seed_eval.py':    'fce9a282713385c68fd1455c9b3076b8',
    'campaign_lottery.py':   '1d80b3a0cbc1a2dc23fc8167761894b1',
    'campaign_guard.py':     '53ad2b25065010fa10fb1e1952e01f94',
    'tablevii_r8.py':        '5fd895bf1560f2fe94715b8557b544a9',
}

FIELDS = ['arm', 'label', 'seed', 'ckpt', 'ckpt_origin', 'ckpt_md5', 'repeat', 'chan_seed_first',
          'secs'] + METRICS + \
    ['RLF_std', 'venue', 'python', 'torch', 'numpy', 'binary_power', 'guard_db', 'guard_mode',
     'loader', 'evaluator', 'eval_ast_md5', 'eval_seeds', 'eval_ep', 'env_v5_md5', 'env_nyu_md5',
     'env_branched_md5', 'driver_md5', 'campaign_md5', 't7_md5', 'measured_utc']

OUTDIR = f'results/lottery_r8{_SFX}'
STATUS = f'lottery_r8{_SFX}.status'
LOG = f'lottery_r8{_SFX}.log'

md5 = T7.md5
avail_mb = T7.avail_mb


def merged_path():
    """Built at CALL time, never at import time. A module-level f-string would freeze the
    UNVERIFIED default venue into the filename, which is precisely how a table of cloud
    measurements ends up in a file named '..._laptop.csv'."""
    return f'results/lottery_r8{_SFX}_{VENUE}.csv'


def summary_path():
    return f'results/lottery_r8{_SFX}_summary_{VENUE}.json'


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
    """run_multiseed_rel.py:27-28 and train_branched_nyu.py:23-24 parse sys.argv AT MODULE LEVEL.
    Importing them with this script's argv would crash on int('--unit') or silently reset a step
    budget. train_srdndqn.py touches argv only inside main() and is safe under any argv, but the
    neutralisation covers it anyway. The N_SEEDS/STEPS those modules compute govern only their own
    main(); evaluate() does not read them."""
    saved = sys.argv
    sys.argv = [saved[0]]
    try:
        import torch
        torch.set_num_threads(1)
        import numpy as np
        import env_6g_v5
        import env_6g_nyu
        import env_6g_branched
        import train_phase2 as tp
        from dndqn import OBS_DIM, load_dndqn_checkpoint
        from dndqn_branched import load_branched
        import run_multiseed_rel as F           # lot_flat evaluator  (== Table VII row 1's)
        import train_srdndqn as S               # lot_guard evaluator (== campaign_guard's)
        import train_branched_nyu as B          # Table VII's guarded evaluator, for the equivalence proof
        return dict(torch=torch, np=np, env_6g_v5=env_6g_v5, env_6g_nyu=env_6g_nyu,
                    env_6g_branched=env_6g_branched, tp=tp, OBS_DIM=OBS_DIM,
                    load_dndqn_checkpoint=load_dndqn_checkpoint, load_branched=load_branched,
                    F=F, S=S, B=B)
    finally:
        sys.argv = saved


def _fn_ast_md5(fn):
    """md5 of the function's ABSTRACT SYNTAX TREE, so formatting differences cannot mask or fake a
    difference in behaviour."""
    src = inspect.getsource(fn)
    node = ast.parse(src).body[0]
    return hashlib.md5(ast.dump(node, annotate_fields=True,
                                include_attributes=False).encode()).hexdigest()


def _here(name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)


def _t7_recorded():
    """Read the released Table VII artefact and return {arm: {field: value}} for the provenance
    fields. Returns {} when the artefact is not on this venue."""
    p = _here(T7_CSV) if not os.path.exists(T7_CSV) else T7_CSV
    if not os.path.exists(p):
        return {}
    out = {}
    try:
        for r in csv.DictReader(open(p, newline='')):
            out.setdefault(r['arm'], {k: r[k] for k in
                                      ('env_v5_md5', 'env_branched_md5', 'driver_md5',
                                       'eval_seeds', 'eval_ep', 'venue')})
    except (OSError, KeyError):
        return {}
    return out


def _crn_reach(cls, base):
    """Walk the MRO from the concrete environment class up to the patched base class. Every class
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
    """Seven refusable checks. Any FAIL means the grid this driver would produce is not
    comparable to the tables it is meant to join, so it must not be produced at all."""
    ok = True
    rec = _t7_recorded()

    # ---- 1/2/3  file provenance -----------------------------------------------------------
    def _pin(name, from_t7=None):
        nonlocal ok
        live = md5(_here(name) if os.path.exists(_here(name)) else name)
        want = rec.get(from_t7[0], {}).get(from_t7[1]) if (from_t7 and rec) else None
        src = f"tablevii_r8_laptop.csv[{from_t7[0]}].{from_t7[1]}" if want else "PIN literal"
        want = want or PIN.get(name)
        good = bool(live) and live == want
        log(f"verify | provenance | {name:24s} live={live or '(absent)'} want={want} "
            f"[{src}] | {'MATCH' if good else 'MISMATCH'}")
        ok = ok and good

    _pin('run_multiseed_rel.py', ('flat', 'driver_md5'))        # lot_flat's instrument == Table VII row 1's
    _pin('env_6g_v5.py', ('flat', 'env_v5_md5'))                # the class the CRN patch targets
    _pin('env_6g_branched.py', ('guarded', 'env_branched_md5'))  # the env lot_guard runs in
    _pin('train_branched_nyu.py', ('guarded', 'driver_md5'))    # Table VII guarded's instrument
    for n in ('train_srdndqn.py', 'env_6g_nyu.py', 'dndqn_branched.py', 'multi_seed_eval.py',
              'campaign_lottery.py', 'campaign_guard.py', 'tablevii_r8.py'):
        _pin(n)

    # ---- 4  protocol constants agree across the three modules -------------------------------
    consts = {}
    for name, mod in (('rel(lot_flat)', S['F']), ('srdndqn(lot_guard)', S['S']),
                      ('branched(TableVII)', S['B'])):
        consts[name] = (mod.GAIN, tuple(mod.EVAL_SEEDS), mod.EVAL_EP)
        log(f"verify | protocol {name:20s} | GAIN={mod.GAIN} EVAL_SEEDS={list(mod.EVAL_SEEDS)} "
            f"EVAL_EP={mod.EVAL_EP}")
    agree = len(set(consts.values())) == 1
    log(f"verify | protocol constants | {'MATCH across all three modules' if agree else 'MISMATCH'}")
    ok = ok and agree
    if rec:
        want = (str(len(S['F'].EVAL_SEEDS)), str(S['F'].EVAL_EP))
        got = (rec['flat']['eval_seeds'], rec['flat']['eval_ep'])
        same = want == got
        log(f"verify | protocol vs Table VII artefact | live={want} recorded={got} | "
            f"{'MATCH' if same else 'MISMATCH'}")
        ok = ok and same

    # ---- 5  the guard arm's controller definition -------------------------------------------
    gd = getattr(S['S'], 'GUARD_DB', None)
    arm_def = S['S'].ARMS.get('postguard')
    good = (gd == 9.0) and (arm_def == (9.0, False))
    log(f"verify | guard definition | GUARD_DB={gd} ARMS['postguard']={arm_def} | "
        f"{'MATCH — 9 dB state guard, backup shield OFF' if good else 'MISMATCH'}")
    ok = ok and good

    # ---- 6  the CRN patch actually reaches both evaluators' environments ---------------------
    base = S['env_6g_v5'].SubTHzEnv6G
    for who, cls in (('lot_flat  SubTHzEnv6G_NYU', S['env_6g_nyu'].SubTHzEnv6G_NYU),
                     ('lot_guard BranchedEnv6G', S['env_6g_branched'].BranchedEnv6G)):
        good, why = _crn_reach(cls, base)
        log(f"verify | CRN reach | {who:28s} MRO={[c.__name__ for c in cls.__mro__[:4]]} | {why} "
            f"| {'PASS' if good else 'FAIL'}")
        ok = ok and good

    # ---- 7  EMPIRICAL evaluator equivalence for the guard arm --------------------------------
    # The two guard evaluators are line-for-line identical apart from srdndqn's extra RLF_std key,
    # so their ASTs DIFFER and an AST comparison would wrongly refuse. Prove behaviour instead.
    a, b = _fn_ast_md5(S['S'].evaluate), _fn_ast_md5(S['B'].evaluate_branched)
    log(f"verify | evaluate() AST | train_srdndqn={a} train_branched_nyu={b} | expected to DIFFER "
        f"(srdndqn returns one extra key, RLF_std) — behaviour is proven empirically below")
    log(f"verify | evaluate() AST | run_multiseed_rel.evaluate={_fn_ast_md5(S['F'].evaluate)} "
        f"(the object tablevii_r8.py uses for arm 'flat')")
    if deep:
        ck = ckpt_for('lot_guard', 0)
        if not os.path.exists(ck):
            log(f"verify | equivalence | {ck} absent — CANNOT prove the two guard evaluators agree")
            ok = False
        else:
            t0 = time.time()
            slow, med, _f, _s = S['tp'].load_trajectories(S['tp'].CSV_PATH)
            pool = slow + med
            net = S['load_branched'](ck, S['OBS_DIM'])[0]
            with T7.seeded(0):
                m1 = S['S'].evaluate(net, pool, 'postguard')
            with T7.seeded(0):
                m2 = S['B'].evaluate_branched(net, pool, True, guard_db=9.0, guard_mode='state')
            worst = 0.0
            for k in METRICS:
                d = abs(float(m1[k]) - float(m2[k]))
                worst = max(worst, d)
                log(f"verify | equivalence | {k:5s} srdndqn={float(m1[k]):12.8f} "
                    f"branched={float(m2[k]):12.8f} |d|={d:.3e}")
            good = worst < 1e-9
            log(f"verify | equivalence | worst |d| = {worst:.3e} over {len(METRICS)} metrics on "
                f"{ck} ({len(S['S'].EVAL_SEEDS)}x{S['S'].EVAL_EP} episodes, CRN repeat 0) in "
                f"{time.time() - t0:.0f}s | {'PASS — one instrument' if good else 'FAIL'}")
            ok = ok and good

    log(f"verify | {'PASS' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------------------------------
# worker: one (arm, seed) unit = R seeded repeats of one checkpoint
# --------------------------------------------------------------------------------------------
def run_unit(arm, seed):
    S = import_stack()
    T7.install_patch(S['env_6g_v5'])            # the SAME patch object Tables VII and X used
    torch = S['torch']
    label, pat, loader, evk, drv, camp, origin = ARMS[arm]
    ck = ckpt_for(arm, seed)
    if not os.path.exists(ck):
        log(f"[{arm} s{seed}] MISSING CHECKPOINT {ck} — refusing")
        return 3
    # The worker re-runs the cheap provenance/patch checks but not the ~1 min empirical
    # equivalence pass, which the parent already proved once for the whole grid.
    if not verify(S, deep=False):
        log(f"[{arm} s{seed}] REFUSING: verification failed")
        return 4

    if EVAL_EP_OVERRIDE:                        # rehearsal only
        S['F'].EVAL_EP = S['S'].EVAL_EP = S['B'].EVAL_EP = EVAL_EP_OVERRIDE
    slow, med, fast, skip = S['tp'].load_trajectories(S['tp'].CSV_PATH)
    pool = slow + med
    tc = dict(python=sys.version.split()[0], torch=torch.__version__, numpy=S['np'].__version__)

    if arm == 'lot_flat':
        net = S['load_dndqn_checkpoint'](ck)[0]
        mod, fn = S['F'], (lambda: S['F'].evaluate(net, pool))
        eval_md5 = _fn_ast_md5(S['F'].evaluate)
        ctrl = {'binary_power': '', 'guard_db': '', 'guard_mode': ''}
    else:
        net = S['load_branched'](ck, S['OBS_DIM'])[0]
        mod, fn = S['S'], (lambda: S['S'].evaluate(net, pool, 'postguard'))
        eval_md5 = _fn_ast_md5(S['S'].evaluate)
        ctrl = {'binary_power': True, 'guard_db': S['S'].GUARD_DB, 'guard_mode': 'state'}

    n_eval_seeds, n_ep = len(mod.EVAL_SEEDS), mod.EVAL_EP
    log(f"[{arm} s{seed}] start | {ck} (md5 {md5(ck)}) | loader={loader} evaluator={evk} "
        f"| R={R} repeats | {n_eval_seeds} eval seeds x {n_ep} episodes | torch {tc['torch']} "
        f"| threads {torch.get_num_threads()} | pool {len(pool)}")

    os.makedirs(OUTDIR, exist_ok=True)
    rows = []
    for r in range(R):
        t0 = time.time()
        with T7.seeded(r):
            m = fn()
        row = {'arm': arm, 'label': label, 'seed': seed, 'ckpt': ck, 'ckpt_origin': origin,
               'ckpt_md5': md5(ck), 'repeat': r, 'chan_seed_first': BASE + 100000 * r,
               'secs': round(time.time() - t0, 1),
               'RLF_std': (f"{float(m['RLF_std']):.4f}" if 'RLF_std' in m else ''),
               'venue': VENUE, 'python': tc['python'], 'torch': tc['torch'], 'numpy': tc['numpy'],
               'loader': loader, 'evaluator': evk, 'eval_ast_md5': eval_md5,
               'eval_seeds': n_eval_seeds, 'eval_ep': n_ep,
               'env_v5_md5': md5('env_6g_v5.py'), 'env_nyu_md5': md5('env_6g_nyu.py'),
               'env_branched_md5': md5('env_6g_branched.py'),
               'driver_md5': md5(drv), 'campaign_md5': md5(camp),
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
    for k in ('LR8_TAG', 'LR8_R', 'LR8_EVAL_EP'):
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
            name = f'lr8_{arm}_s{seed}'
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
    """The 10-seed single-pass ledger this arm's R=8 grid refines, read from its own file. Never
    transcribed: a hand-copied baseline that drifts from its source is a fabrication risk."""
    path, filt = SINGLEPASS_CSV[arm]
    if not os.path.exists(path):
        return None
    out = {m: [] for m in METRICS}
    try:
        for r in csv.DictReader(open(path, newline='')):
            if filt and r.get(filt[0]) != filt[1]:
                continue
            for m in METRICS:
                out[m].append(float(r[m]))
    except (OSError, KeyError, ValueError):
        return None
    return out if all(len(v) == len(SEEDS) for v in out.values()) else None


def decompose(X):
    """Two-way crossed decomposition, seeds x CRN repeats, no replication within a cell.
    Identical arithmetic to tablevii_decompose.decompose(), reproduced here so the laptop needs
    no scipy. The exact F pivot, one-sided bounds and resolvability verdict are computed
    downstream by that module, so all four tables are analysed by one piece of code."""
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
                sd_resid=math.sqrt(max(ms_e, 0.0)), F=(ms_a / ms_e if ms_e > 0 else float('inf')))


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
    print("=" * 132)
    print(f"  TEN-SEED LOTTERY on the R={R} CRN grid | venue {sorted(venues)[0]} | "
          f"torch {sorted(torches)} | {len(rows)} evaluations | BASE={BASE}")
    print(f"  mean over {len(SEEDS)} training seeds. sd_train = sqrt(max((MS_A - MS_E)/R, 0)) — the "
          f"sample sd with the evaluation-noise contribution subtracted.")
    print(f"  1-pass columns are the SAME twenty checkpoints measured once, unseeded "
          f"(lottery_flat/guard_10seed_800k_laptop.csv): their sd confounds training and "
          f"evaluation noise.")
    print("=" * 132)
    summary = {'venue': sorted(venues)[0], 'R': R, 'BASE': BASE, 'seeds': SEEDS,
               'n_rows': len(rows), 'torch': sorted(torches), 'arms': {},
               'generated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    lines, ratios = [], {}
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
                   'singlepass_csv': SINGLEPASS_CSV[arm][0]}
        cells = []
        print(f"\n  {label}   [{len(seeds_here)} seeds x {len(reps)} repeats]")
        print(f"    {'metric':>6} | {'mean':>9} | {'sd_train':>9} | {'sd_ddof1':>9} | "
              f"{'sd_resid':>9} | {'F(A/E)':>10} | {'1p mean':>9} | {'1p sd':>9}")
        for m in METRICS:
            X = [[float(r[m]) for r in sorted((q for q in sub if int(q['seed']) == s),
                                              key=lambda q: int(q['repeat']))]
                 for s in seeds_here]
            d = decompose(X)
            sd1 = float(np.std(d['seed_means'], ddof=1))
            pm = ps = float('nan')
            if sp:
                pm, ps = float(np.mean(sp[m])), float(np.std(sp[m], ddof=1))
            print(f"    {m:>6} | {d['mean']:9.4f} | {d['sd_train']:9.4f} | {sd1:9.4f} | "
                  f"{d['sd_resid']:9.4f} | {d['F']:10.2f} | {pm:9.3f} | {ps:9.3f}")
            arm_out[m] = dict(mean=d['mean'], sd_train=d['sd_train'], sd_ddof1=sd1,
                              sd_resid=d['sd_resid'], sd_block=d['sd_block'], F=d['F'],
                              df=(d['df_a'], d['df_b'], d['df_e']),
                              ms_a=d['ms_a'], ms_b=d['ms_b'], ms_e=d['ms_e'],
                              v_train=d['v_train'], seed_means=d['seed_means'],
                              singlepass_mean=pm, singlepass_sd_ddof1=ps,
                              grid=[[float(v) for v in row] for row in X])
            dp = 1 if m == 'Tput' else 2
            cells.append(f"{d['mean']:.{dp}f} $\\pm$ {d['sd_train']:.{dp}f}")
        summary['arms'][arm] = arm_out
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    if all(a in summary['arms'] for a in ARM_ORDER):
        print("\n" + "=" * 132)
        print("  THE RELOCATION, on the R=8 basis — sd_train(flat) / sd_train(guard) per metric")
        print("  A ratio > 1 means the guard REDUCES training-seed spread on that axis; < 1 means")
        print("  it moves the spread onto that axis. This is the 'lottery relocates' claim, now")
        print("  measured with the evaluation-noise component removed rather than folded in.")
        print("=" * 132)
        print(f"    {'metric':>6} | {'sd_train flat':>13} | {'sd_train guard':>14} | "
              f"{'ratio':>9} | {'1-pass ratio':>12}")
        for m in METRICS:
            f_, g_ = summary['arms']['lot_flat'][m], summary['arms']['lot_guard'][m]
            rt = (f_['sd_train'] / g_['sd_train']) if g_['sd_train'] > 0 else float('inf')
            r1 = (f_['singlepass_sd_ddof1'] / g_['singlepass_sd_ddof1']
                  if g_['singlepass_sd_ddof1'] > 0 else float('inf'))
            ratios[m] = dict(sd_train_flat=f_['sd_train'], sd_train_guard=g_['sd_train'],
                             ratio=rt, singlepass_ratio=r1)
            print(f"    {m:>6} | {f_['sd_train']:13.4f} | {g_['sd_train']:14.4f} | "
                  f"{rt:9.3f} | {r1:12.3f}")
        summary['relocation'] = ratios
        print("\n  NOTE: these point ratios are DESCRIPTIVE. The reportable claim is the bounded "
              "form\n        computed downstream from the F pivot; do not quote a point ratio in "
              "the manuscript.")

    print("\n" + "=" * 132)
    print("  MANUSCRIPT ROWS, sd_train basis (paste order: HO | Power | Spec | RLF | Tput)")
    print("=" * 132)
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
        log(f"REFUSING: torch {torch.__version__} but every reported number in this revision was "
            f"produced on torch {EXPECT_TORCH}.x. The lottery grid must be measured on the "
            f"reporting venue. Pass --allow-toolchain-mismatch only for a rehearsal.")
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
    ap.add_argument('--arms', default='', help='comma-separated subset; default = both. A table '
                                               'built from a subset is NOT the paper table and '
                                               'says so.')
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
        log(f"*** ARM SUBSET *** {arms} — NOT the full lottery run")
    units = [(arm, s) for arm in arms for s in SEEDS]
    log(f"LOTTERY R={R} CRN | venue={VENUE} | {len(arms)} arms x {len(SEEDS)} seeds x R={R} "
        f"= {len(units) * R} evaluations | BASE={BASE}")
    log(f"toolchain | python {sys.version.split()[0]} | torch {tv} | cwd {os.getcwd()}")

    S = import_stack()
    T7.install_patch(S['env_6g_v5'])
    T7.verify_patch()
    if not verify(S, deep=True):
        log("REFUSING: verification failed — see the lines above")
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
