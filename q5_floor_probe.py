#!/usr/bin/env python3
"""
q5_floor_probe.py -- measure the per-step throughput distribution BEFORE choosing Q5's floor.

Q5, verbatim: "The spectrum axis is described as 'unanchored' and its anchoring as future work.
Has a simple throughput-floor reward term been piloted? Even a negative result (e.g., it
re-introduces reliability sensitivity) would be informative and would bound the design space."

A throughput-floor reward needs two constants: the floor T_floor and its weight. Picking a round
number for T_floor would make the pilot's result an artefact of that number -- set it too low and
the term never fires, so the answer is "no effect"; set it too high and it fires on every step, so
it degenerates into a constant offset plus a push to allocate every RB, and the answer is "it
destroys spectrum saving". Either way the reviewer learns nothing about the design space, which is
what he actually asked to have bounded.

So the floor is derived from the distribution the PUBLISHED controller already produces. This
script measures that distribution on the published guarded arm (br_gst_s*.pt, G = 9 dB, state
mode) and reports percentiles, so the floor can be stated as "the p-th percentile of the
unanchored controller's own per-step throughput" -- a number that traces to a measurement, in
keeping with the paper's measurement-first design rule.

THROUGHPUT IS COMPUTED WITH THE PUBLISHED FORMULA, NOT A NEW ONE. train_branched_nyu.py:76-77 is

    sdb = obs[4] * 100.0
    et.append(info['rb_count'] * (RB_BW_HZ / 1e6) * math.log2(1 + 10 ** (sdb / 10)))

and that expression is imported and reused here, so the floor is a floor on exactly the quantity
the paper reports as Tput -- not on a lookalike.

Instrumentation only: the wrapper records, it never changes an action, a power, or a transition.
"""
import csv
import math
import os
import sys
import time

import numpy as np

SEEDS = [0, 1, 2, 3, 4]
R = int(os.environ.get('Q5_R', 4))
G = 9.0
OUT = 'results/q5floor'
PCTS = [1, 5, 10, 15, 20, 25, 33, 50, 66, 75, 90, 95, 99]


def log(m):
    print(time.strftime('%Y-%m-%d %H:%M:%S') + ' | ' + m, flush=True)


REC = []


def main():
    saved = sys.argv
    sys.argv = [saved[0]]
    import torch
    torch.set_num_threads(1)
    import env_6g_v5
    import env_6g_branched as EB
    import train_phase2 as tp
    import train_branched_nyu as B
    from dndqn_branched import load_branched
    from dndqn import OBS_DIM
    import tablevii_r8 as T7
    sys.argv = saved
    T7.install_patch(env_6g_v5)

    RB_BW_MHZ = B.RB_BW_HZ / 1e6

    if not hasattr(EB.BranchedEnv6G, '_q5_orig_step'):
        EB.BranchedEnv6G._q5_orig_step = EB.BranchedEnv6G.step

        def _step(self, action):
            out = self._q5_orig_step(action)
            obs, info = out[0], out[4]
            sdb = float(obs[4]) * 100.0
            t = float(info['rb_count']) * RB_BW_MHZ * math.log2(1.0 + 10.0 ** (sdb / 10.0))
            REC.append((t, float(info['rb_count']), sdb))
            return out

        EB.BranchedEnv6G.step = _step

    slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH)
    pool = slow + med
    os.makedirs(OUT, exist_ok=True)
    bank = os.path.join(OUT, 'q5_tput_steps.csv')
    have = set()
    if os.path.exists(bank):
        for row in csv.DictReader(open(bank)):
            have.add((int(row['seed']), int(row['repeat'])))
        log('resuming: %d of %d pairs banked' % (len(have), len(SEEDS) * R))
    log('Q5 floor probe | published guarded arm, G = %.1f dB | %d seeds x R=%d | pool %d'
        % (G, len(SEEDS), R, len(pool)))

    allv = []
    for s in SEEDS:
        net = load_branched('br_gst_s%d.pt' % s, OBS_DIM)[0]
        for r in range(R):
            REC.clear()
            with T7.seeded(r):
                B.evaluate_branched(net, pool, True, guard_db=G, guard_mode='state')
            a = np.array(REC, dtype=float)
            allv.append(a[:, 0])
            new = not os.path.exists(bank)
            with open(bank, 'a', newline='') as fh:
                w = csv.writer(fh)
                if new:
                    w.writerow(['seed', 'repeat', 'steps', 'mean_mbps', 'std_mbps']
                               + ['p%d' % p for p in PCTS])
                w.writerow([s, r, len(a), '%.4f' % a[:, 0].mean(), '%.4f' % a[:, 0].std()]
                           + ['%.4f' % v for v in np.percentile(a[:, 0], PCTS)])
            log('  seed %d repeat %d | %s steps | mean %.1f Mbps | p25 %.1f | p50 %.1f'
                % (s, r, format(len(a), ','), a[:, 0].mean(),
                   np.percentile(a[:, 0], 25), np.percentile(a[:, 0], 50)))

    v = np.concatenate(allv)
    print('\n' + '=' * 92)
    print('Q5 -- PER-STEP THROUGHPUT OF THE UNANCHORED CONTROLLER (published guarded arm, G=9 dB)')
    print('=' * 92)
    print('  steps pooled : %s over %d seeds x R=%d' % (format(len(v), ','), len(SEEDS), R))
    print('  mean         : %8.1f Mbps      std : %8.1f Mbps' % (v.mean(), v.std()))
    print('  zero-tput steps : %s (%.2f%%)'
          % (format(int((v <= 0).sum()), ','), 100.0 * (v <= 0).mean()))
    print('\n  %-10s %14s' % ('percentile', 'Mbps'))
    for p in PCTS:
        print('  %-10s %14.1f' % ('p%d' % p, np.percentile(v, p)))
    print('\n  A floor set at percentile p fires on p%% of steps by construction. The pilot wants a')
    print('  term that BITES without DOMINATING; p25 is the natural first choice and is what the')
    print('  campaign uses unless overridden. p10 would barely fire; p50 would fire half the time')
    print('  and become a standing push to allocate RBs rather than an anchor.')
    print('\n  proposed T_floor = p25 = %.1f Mbps' % np.percentile(v, 25))
    print('=' * 92)
    with open(os.path.join(OUT, 'q5_floor_choice.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['quantity', 'value', 'unit', 'basis'])
        w.writerow(['T_floor', '%.4f' % np.percentile(v, 25), 'Mbps',
                    'p25 of pooled per-step throughput, published guarded arm, %d seeds x R=%d'
                    % (len(SEEDS), R)])
        for p in PCTS:
            w.writerow(['p%d' % p, '%.4f' % np.percentile(v, p), 'Mbps', 'pooled per-step'])
        w.writerow(['mean', '%.4f' % v.mean(), 'Mbps', 'pooled per-step'])
        w.writerow(['steps', len(v), 'count', 'pooled'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
