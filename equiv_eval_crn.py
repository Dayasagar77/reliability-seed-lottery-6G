#!/usr/bin/env python3
"""equiv_eval_crn.py — the measurement equiv_eval.py set out to make but could not.

WHY THIS FILE REPLACES equiv_eval.py
------------------------------------
equiv_eval.py states: "evaluate() is deterministic given EVAL_SEEDS ... HO / Power / Spec / RLF /
Tput must agree to full float precision -- not approximately, exactly."

That premise is FALSE, and it is why that harness must not be run as written.

In train_srdndqn.evaluate(), EVAL_SEEDS seeds only the TRAJECTORY SAMPLER:

    r = np.random.default_rng(s)          # picks which HighD trajectory each episode uses
    ...
    obs, _ = env.reset()                  # <-- NO seed argument

The channel is seeded inside SubTHzEnv6G.reset() by `self.rng = np.random.default_rng(seed)`.
With seed=None that draws from OS entropy, so two runs of the SAME code on the SAME checkpoint
already disagree. Running equiv_eval.py in laptopcode/ and cloudcode/ would therefore compare two
independent noise draws, and any difference it printed would be unattributable: it could be the
29-Jul code change, or it could be the channel. A harness that cannot tell those apart cannot
falsify anything, which is the one thing its docstring promised it would do.

WHAT THIS FILE DOES INSTEAD
---------------------------
It installs the SAME common-random-number patch that produced the cloud R=8 grid -- imported by
path from ../tablevii_r8.py (md5 5fd895bf1560f2fe94715b8557b544a9), not reimplemented here, so the
CRN rule cannot drift between the two runs -- and evaluates the same frozen checkpoint under
identical channel seeds.

Two repeats are measured, and BOTH claims are required:

  SENSITIVITY  repeat 0 vs repeat 1 within one code version MUST differ.
               Otherwise the patch never reached the environment, every comparison below is a
               comparison of two identical constants, and an exact match would mean nothing.

  EQUIVALENCE  repeat r under laptopcode vs repeat r under cloudcode MUST agree to < 1e-12 on
               HO, Power, Spec, RLF, Tput, RLF_std.

Nothing is trained. The checkpoint is a fixed set of weights and is byte-identical in both dirs.
'Override' is compared only when both versions report it; the laptop version has no such key,
because that metric is one of the things 29 Jul added.

Usage:  cd _equiv/<version>code && python -u ../equiv_eval_crn.py [ckpt] [repeats]
"""
import importlib.util
import json
import os
import sys
import time

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # /root/work/exp

# this code version's directory must win every import
sys.path.insert(0, os.getcwd())

# import the frozen CRN instrument BY PATH, so both runs get the identical rule
_T7_PATH = os.path.join(EXP, 'tablevii_r8.py')
_spec = importlib.util.spec_from_file_location('tablevii_r8_frozen', _T7_PATH)
T7 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T7)

import hashlib

import torch
torch.set_num_threads(1)                       # paper protocol: single-threaded CPU

import numpy as np
import env_6g_v5
import env_6g_branched
import train_srdndqn as S
import train_phase2 as tp
from dndqn import OBS_DIM
from dndqn_branched import load_branched


def md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest() if os.path.exists(p) else None


CKPT = sys.argv[1] if len(sys.argv) > 1 else 'srdndqn_postguard_s0.pt'
REPEATS = [int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [0, 1]
ARM = 'postguard'
SHARED = ['HO', 'Power', 'Spec', 'RLF', 'Tput', 'RLF_std']

T7.install_patch(env_6g_v5)

# the patch targets SubTHzEnv6G; prove it actually reaches the class evaluate() instantiates
import inspect
chain = []
base = env_6g_v5.SubTHzEnv6G
cls = env_6g_branched.BranchedEnv6G
assert base in cls.__mro__, 'patched base is not in BranchedEnv6G.__mro__'
for c in cls.__mro__:
    if c is base:
        break
    if 'reset' in c.__dict__:
        src = inspect.getsource(c.__dict__['reset'])
        chain.append({'cls': c.__name__,
                      'forwards_seed': ('super().reset(' in src) and ('seed=seed' in src)})

slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH)
pool = slow + med
net = load_branched(CKPT, OBS_DIM)[0]

runs = {}
for rep in REPEATS:
    t0 = time.time()
    with T7.seeded(rep):
        met = S.evaluate(net, pool, ARM)
    runs[rep] = {'seconds': round(time.time() - t0, 1),
                 'metrics': {k: float(v) for k, v in sorted(met.items())}}

out = {
    'ckpt': CKPT,
    'ckpt_md5': md5(CKPT),
    'cwd': os.path.basename(os.getcwd()),
    'torch': torch.__version__,
    'numpy': np.__version__,
    'python': sys.version.split()[0],
    'pool': len(pool),
    'arm': ARM,
    'crn': {'rule': T7.PATCH_SEMANTICS, 'BASE': T7.BASE, 't7_md5': md5(_T7_PATH)},
    'crn_reach': chain,
    'code': {'train_srdndqn.py': md5('train_srdndqn.py'),
             'env_6g_branched.py': md5('env_6g_branched.py'),
             'env_6g_v5.py': md5('env_6g_v5.py'),
             'dndqn_branched.py': md5('dndqn_branched.py')},
    'eval_seeds': list(S.EVAL_SEEDS), 'eval_ep': S.EVAL_EP, 'gain': S.GAIN,
    'runs': runs,
}

# sensitivity: within this version, the repeats must not be identical
if len(REPEATS) >= 2:
    a, b = runs[REPEATS[0]]['metrics'], runs[REPEATS[1]]['metrics']
    worst = max(abs(a[k] - b[k]) for k in SHARED if k in a and k in b)
    out['sensitivity'] = {'repeats': REPEATS[:2], 'max_abs_diff': worst,
                          'verdict': 'PASS — CRN reached the env, repeats differ' if worst > 1e-9
                                     else 'FAIL — repeats identical, patch never took effect'}

print('EQUIV_CRN_JSON ' + json.dumps(out))
