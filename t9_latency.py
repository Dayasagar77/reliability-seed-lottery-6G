#!/usr/bin/env python3
"""
t9_latency.py -- T9: measure the BRANCHED controller on the i9, completing Table XXI's 2x2.

METHOD, TAKEN FROM THE PUBLISHED CODE, NOT INVENTED. multi_seed_eval.py times a whole DECISION:

    t0 = time.perf_counter()
    action = net(torch.FloatTensor(obs).to(DEVICE)).argmax().item()
    lat = (time.perf_counter() - t0) * 1000

That includes tensor construction, device placement, forward, argmax and the .item() sync. A
first version of this script timed only a bare forward on a pre-built tensor and measured 0.059 ms
against the published 0.109 -- with the MAX matching almost exactly (0.630 vs 0.633), which is the
signature of a method difference rather than a hardware one. The control caught it. That is what
the control is for.

The branched equivalent is dndqn_branched's own action rule:
    [int(q.argmax(dim=-1).item()) for q in qs]
one argmax per branch, so the branched decision genuinely does more work than the flat one and the
difference is an architecture cost rather than an artefact of timing different things.

CONTROL FIRST. The flat controller is re-measured here and must reproduce the published i9 row
(0.1093 / 0.2142 / 0.6332 ms, results/fig_latency_summary.csv) before any new number is printed.
"""
import statistics
import sys
import time

import numpy as np

sys.argv = sys.argv[:1]
import torch                                                            # noqa: E402
torch.set_num_threads(1)

from dndqn import OBS_DIM, load_dndqn_checkpoint                        # noqa: E402
from dndqn_branched import load_branched                                # noqa: E402

N = 10_000
WARMUP = 500
PUB = dict(mean=0.1093, p99=0.2142, mx=0.6332)     # results/fig_latency_summary.csv
TOL = 0.20
FLAT_CKPT = 'dqn_phase2_dndqn_nyu_rel_final.pt'
BR_CKPT = 'br_gst_s0.pt'


def bench(decide, obs_stream):
    for i in range(WARMUP):
        decide(obs_stream[i % len(obs_stream)])
    ts = []
    for i in range(N):
        o = obs_stream[i % len(obs_stream)]
        t0 = time.perf_counter()
        decide(o)
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    return dict(mean=statistics.fmean(ts), median=ts[len(ts) // 2],
                p95=ts[int(0.95 * len(ts))], p99=ts[int(0.99 * len(ts))], mx=ts[-1])


print('=' * 96)
print('T9 -- per-decision inference latency, i9-12900H, single-threaded, %s decisions' % f'{N:,}')
print('=' * 96)
print('  torch %s | threads %d' % (torch.__version__, torch.get_num_threads()))

# realistic observation stream rather than one frozen vector
rng = np.random.default_rng(0)
stream = [rng.normal(0.0, 1.0, size=OBS_DIM).astype(np.float32) for _ in range(512)]

r = load_dndqn_checkpoint(FLAT_CKPT)
fnet = r[0] if isinstance(r, tuple) else r
fnet.eval()
bnet, meta = load_branched(BR_CKPT, OBS_DIM)
bnet.eval()

with torch.no_grad():
    fr = bench(lambda o: fnet(torch.FloatTensor(o).to('cpu')).argmax().item(), stream)

print('\n  CONTROL  flat Dn-DQN (%s)' % FLAT_CKPT)
print('    measured  mean %.4f  p99 %.4f  max %.4f ms' % (fr['mean'], fr['p99'], fr['mx']))
print('    published mean %.4f  p99 %.4f  max %.4f ms' % (PUB['mean'], PUB['p99'], PUB['mx']))
rel = abs(fr['mean'] - PUB['mean']) / PUB['mean']
comparable = rel <= TOL
print('    control %s: mean is %.0f%% off published (tolerance %.0f%%)'
      % ('PASS' if comparable else 'FAIL', 100 * rel, 100 * TOL))
if not comparable:
    print('    The published figures were collected DURING a live rollout, where environment work')
    print('    between decisions disturbs cache state; an isolated loop keeps everything hot and')
    print('    will always run faster. So these absolutes are not interchangeable with Table XXI,')
    print('    and NO ROW MAY BE ADDED TO IT from this run. The like-for-like pair below is')
    print('    measured under one method and answers what T9 actually asks -- whether the')
    print('    architecture cost is real on fast hardware -- so it is reported separately.')

# The branched dueling head does a.mean(dim=1), so it requires an explicit batch dimension --
# the flat net accepts a bare (obs_dim,) vector, the branched one does not. Adding unsqueeze(0)
# is part of the branched decision, so it is inside the timed region rather than hoisted out.
with torch.no_grad():
    br = bench(lambda o: [int(q.argmax(dim=-1).item())
                          for q in bnet(torch.FloatTensor(o).to('cpu').unsqueeze(0))], stream)

nf = sum(p.numel() for p in fnet.parameters())
nb = sum(p.numel() for p in bnet.parameters())
print('\n  NEW      branched guarded (%s, branch_sizes=%s)' % (BR_CKPT, meta.get('branch_sizes')))
print('    measured  mean %.4f  median %.4f  p95 %.4f  p99 %.4f  max %.4f ms'
      % (br['mean'], br['median'], br['p95'], br['p99'], br['mx']))

print('\n' + '-' * 96)
print('  SAME-HARDWARE, SAME-METHOD PAIR ON THE i9-12900H%s'
      % ('' if comparable else '   [ISOLATED LOOP -- NOT a Table XXI row]'))
print('    flat      %6d params   %.4f ms mean   %.4f p99   %.4f max' % (nf, fr['mean'], fr['p99'], fr['mx']))
print('    branched  %6d params   %.4f ms mean   %.4f p99   %.4f max' % (nb, br['mean'], br['p99'], br['mx']))
print('    architecture cost here : %+.4f ms mean (%+.1f%%), %+d parameters (%+.1f%%)'
      % (br['mean'] - fr['mean'], 100 * (br['mean'] - fr['mean']) / fr['mean'],
         nb - nf, 100.0 * (nb - nf) / nf))
print('    published Xeon pair    : +0.0407 ms mean (0.0782 -> 0.1189)')
print('    both are far inside the >= 10 ms Near-RT RIC loop.')
print('=' * 96)
