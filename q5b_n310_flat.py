#!/usr/bin/env python3
"""
q5b_n310_flat.py -- the half of Q5 the guarded sweep did not answer: does the LOTTERY survive?

Reviewer 2 asks whether "the cliff shape and the lottery structure" are robust to N310. The
lottery is a FLAT-controller phenomenon, so the guarded sweep answered only half.

FINDING THE RIGHT PATCH TARGET MATTERED. The guarded sweep patched
env_6g_branched.N310_DEFAULT, which works because BranchedEnv6G reads it inside __init__. The flat
path does not go anywhere near that: tablevii_r8 evaluates the flat arm through
run_multiseed_rel.evaluate, which builds SubTHzEnv6G_NYU directly. Patching the branched constant
would have left all three columns identical -- robustness by inaction. SubTHzEnv6G_NYU takes **kw
and forwards it, and BranchedEnv6G is what proves `n310` is an accepted keyword, so the flat path
is swept by injecting a subclass that supplies it. No published file is modified.

GATES
  1. N310 = 3 must reproduce the published flat lottery span, 4.32-14.55 %.
  2. The three settings must differ, or the injection did not take.
"""
import csv
import os
import sys
import time

SEEDS = [0, 1, 2, 3, 4]
LEVELS = [2, 3, 5]
R = 8
METRICS = ['HO', 'Power', 'Spec', 'RLF', 'Tput']
OUT = os.path.join('results', 'n310')
PUB_SPAN = (4.32, 14.55)


def log(m):
    print(time.strftime('%H:%M:%S') + ' | ' + m, flush=True)


def main():
    sys.argv = sys.argv[:1]
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import env_6g_v5
    import train_phase2 as tp
    import run_multiseed_rel as RM
    from dndqn import load_dndqn_checkpoint
    import tablevii_r8 as T7
    import tablevii_decompose as D
    D.CSV = 'results/tablevii_r8_laptop.csv'
    T7.install_patch(env_6g_v5)

    BASE = RM.SubTHzEnv6G_NYU
    state = {'n310': None}

    class N310Env(BASE):
        def __init__(self, *a, **kw):
            if state['n310'] is not None:
                kw.setdefault('n310', state['n310'])
            super().__init__(*a, **kw)

    RM.SubTHzEnv6G_NYU = N310Env
    log('injected N310Env into run_multiseed_rel (base %s)' % BASE.__name__)

    slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH)
    pool = slow + med
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, 'n310_flat.csv')
    rows = list(csv.DictReader(open(path))) if os.path.exists(path) else []
    have = {(int(r['n310']), int(r['seed']), int(r['repeat'])) for r in rows}

    try:
        for n in LEVELS:
            state['n310'] = n
            for s in SEEDS:
                ck = 'ms_rel_s%d.pt' % s
                if not os.path.exists(ck):
                    log('missing %s' % ck)
                    return 3
                if all((n, s, r) in have for r in range(R)):
                    continue
                res = load_dndqn_checkpoint(ck)
                net = res[0] if isinstance(res, tuple) else res
                net.eval()
                for r in range(R):
                    if (n, s, r) in have:
                        continue
                    with T7.seeded(r):
                        m = RM.evaluate(net, pool)
                    row = {'n310': n, 'seed': s, 'repeat': r, 'ckpt': ck, 'venue': 'laptop',
                           'measured_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
                    for k in METRICS:
                        row[k] = '%.4f' % float(m[k])
                    rows.append(row)
                with open(path, 'w', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)
                log('  flat N310=%d s%d | RLF %.2f' % (n, s, float(rows[-1]['RLF'])))
    finally:
        RM.SubTHzEnv6G_NYU = BASE

    cells = {(int(r['n310']), int(r['seed']), int(r['repeat'])): r for r in rows}
    print()
    print('=' * 100)
    print('Q5b -- DOES THE LOTTERY SURVIVE A CHANGE OF N310?  (flat controller, reliability tilt)')
    print('=' * 100)
    print('  %-14s %12s %9s %9s %9s %12s %12s'
          % ('N310', 'RLF mean', 'min', 'max', 'span', 'span/mean', 'sigma_train'))
    out = {}
    for n in LEVELS:
        X = np.array([[float(cells[(n, s, r)]['RLF']) for r in range(R)] for s in SEEDS])
        per = X.mean(axis=1)
        d = D.decompose(X)
        out[n] = (d, per)
        span = per.max() - per.min()
        print('  %-14s %12.2f %9.2f %9.2f %9.2f %12.2f %12.2f'
              % ('%d%s' % (n, ' (published)' if n == 3 else ''), d['grand'], per.min(), per.max(),
                 span, span / d['grand'] if d['grand'] else float('nan'), d['sd_train']))

    print()
    ok = True
    lo, hi = out[3][1].min(), out[3][1].max()
    g1 = abs(lo - PUB_SPAN[0]) < 0.15 and abs(hi - PUB_SPAN[1]) < 0.15
    print('  GATE 1  N310=3 reproduces the published span: %.2f-%.2f vs %.2f-%.2f  %s'
          % (lo, hi, PUB_SPAN[0], PUB_SPAN[1], 'PASS' if g1 else '*** FAIL ***'))
    ok &= g1
    means = [out[n][0]['grand'] for n in LEVELS]
    g2 = (max(means) - min(means)) > 1e-6
    print('  GATE 2  the settings differ (injection took): mean spread %.3f  %s'
          % (max(means) - min(means), 'PASS' if g2 else '*** FAIL - identical columns ***'))
    ok &= g2
    print()
    print('  The question is not whether the LEVEL moves -- it will, as it did for the guarded')
    print('  arm. It is whether the SPREAD survives, i.e. whether the lottery is a property of')
    print('  the controller or an artefact of how failure is counted.')
    print('=' * 100)
    print('  VERDICT: %s' % ('PASS' if ok else '*** FAIL ***'))
    return 0 if ok else 4


if __name__ == '__main__':
    sys.exit(main())
