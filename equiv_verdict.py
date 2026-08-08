#!/usr/bin/env python3
"""equiv_verdict.py — compare the two equiv_eval_crn.py runs and record the verdict.

The verdict file is COMPUTED from the two run logs, never hand-written, so it cannot say PASS
about a comparison that was not made. srdndqn_r8.py refuses to produce a grid unless this file
exists and both verdicts are PASS.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
SHARED = ['HO', 'Power', 'Spec', 'RLF', 'Tput', 'RLF_std']
TAG = 'EQUIV_CRN_JSON '


def load(path):
    for line in open(path, encoding='utf-8'):
        if line.startswith(TAG):
            return json.loads(line[len(TAG):])
    raise SystemExit(f'no {TAG.strip()} line in {path}')


lap = load(os.path.join(HERE, 'out', 'equiv_crn_laptop.log'))
cld = load(os.path.join(HERE, 'out', 'equiv_crn_cloud.log'))

if lap['ckpt_md5'] != cld['ckpt_md5']:
    raise SystemExit('checkpoints differ — the comparison would be meaningless')

reps = sorted(set(lap['runs']) & set(cld['runs']))
cells = []
worst = 0.0
for r in reps:
    a, b = lap['runs'][r]['metrics'], cld['runs'][r]['metrics']
    for k in SHARED:
        if k in a and k in b:
            d = abs(a[k] - b[k])
            worst = max(worst, d)
            cells.append({'repeat': int(r), 'metric': k, 'laptop': a[k], 'cloud': b[k], 'abs_diff': d})

sens = min(lap.get('sensitivity', {}).get('max_abs_diff', 0.0),
           cld.get('sensitivity', {}).get('max_abs_diff', 0.0))

out = {
    'what': 'Does the 29-Jul cloud code change any metric this paper reports, for the arms that '
            'existed before it? Measured, not argued: the same frozen checkpoint evaluated under '
            'both code versions at identical common-random-number channel seeds.',
    'ckpt': lap['ckpt'], 'ckpt_md5': lap['ckpt_md5'], 'arm': lap['arm'],
    'eval_seeds': lap['eval_seeds'], 'eval_ep': lap['eval_ep'], 'pool': lap['pool'],
    'crn': lap['crn'],
    'code_laptop': lap['code'], 'code_cloud': cld['code'],
    'toolchain': {'python': lap['python'], 'torch': lap['torch'], 'numpy': lap['numpy']},
    'repeats': [int(r) for r in reps],
    'metrics_compared': SHARED,
    'cells': cells,
    'worst_abs_diff': worst,
    'equivalence': 'PASS' if worst < 1e-12 else 'FAIL',
    'sensitivity_min_abs_diff': sens,
    'sensitivity': 'PASS' if sens > 1e-9 else 'FAIL',
    'note': "'Override' is excluded from the comparison because it does not exist in the laptop "
            "version -- it is one of the things 29 Jul added. Every metric that exists in both is "
            "compared, at full float precision, for every repeat measured.",
}

p = os.path.join(EXP, 'results', 'equiv_crn_verdict.json')
os.makedirs(os.path.dirname(p), exist_ok=True)
with open(p, 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2)

print(f"{len(cells)} cells compared over repeats {out['repeats']}")
print(f"worst |laptop - cloud| = {worst:.3e}   equivalence {out['equivalence']}")
print(f"within-version repeat spread = {sens:.3e}   sensitivity {out['sensitivity']}")
print(f"written {p}")
sys.exit(0 if out['equivalence'] == 'PASS' and out['sensitivity'] == 'PASS' else 1)
