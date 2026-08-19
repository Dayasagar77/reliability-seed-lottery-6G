#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""q4_latency.py - H4 of the risk-asymmetry pre-registration: does the 51-quantile head stay
inside the Near-RT RIC budget?

    H4  Per-decision latency of the quantile network remains under the 10 ms control interval on
        the same hardware as Supplementary Note 17.
        "A 51-quantile head is larger than the published one. If latency degrades materially it is
        reported as a cost, not omitted."

METHOD — TAKEN FROM t9_latency.py, WHICH TOOK IT FROM multi_seed_eval.py. Not invented here.
The timed region is a whole DECISION, not a bare forward pass: tensor construction, device
placement, forward, the per-branch argmax and the .item() synchronisation. t9_latency.py records
why that matters — an earlier version of it timed only a forward on a pre-built tensor and measured
0.059 ms against a published 0.109, with the MAX matching almost exactly, which is the signature of
a method difference rather than a hardware one. Its control caught that. The same control runs here.

WHAT IS COMPARED. Three networks, one loop, one method, same hardware:

    published branched   br_gst_s0.pt              scalar Q, 2 argmaxes          <- the CONTROL
    qr_sym               q4_qr_sym_s0.pt           51 quantiles, mean/mean       <- head cost
    qr_asym              q4_qr_asym_s0.pt          51 quantiles, CVaR/mean       <- + sort cost

Ordering the three this way separates the two costs the change introduces: the wider head, and the
risk operator itself. CVaR(0.25) sorts the quantile axis before taking the lowest ceil(0.25*51)=13;
the mean reduction does not sort. Whether that shows up at all is the question.

ABSOLUTES ARE NOT TABLE XXI ROWS. The published figures in Supplementary Note 17 were collected on
a deliberately weak 2-core cloud CPU and DURING a live rollout, where environment work between
decisions disturbs cache state. An isolated loop on the i9 keeps everything hot and will always run
faster. So the number that answers H4 is the SAME-METHOD PAIR below, not the absolute.

USAGE
    python -u q4_latency.py
"""
import statistics
import sys
import time

import numpy as np

sys.argv = sys.argv[:1]
import torch                                                          # noqa: E402
torch.set_num_threads(1)

from dndqn import OBS_DIM                                             # noqa: E402
from dndqn_branched import load_branched                              # noqa: E402
import q4_qr_branched as Q                                            # noqa: E402

N = 10_000
WARMUP = 500
NEAR_RT_MS = 10.0
PUB_BRANCHED = 'br_gst_s0.pt'
QR_SYM = 'results/q4_riskasym/q4_qr_sym_s0.pt'
QR_ASYM = 'results/q4_riskasym/q4_qr_asym_s0.pt'


def bench(decide, stream):
    for i in range(WARMUP):
        decide(stream[i % len(stream)])
    ts = []
    for i in range(N):
        o = stream[i % len(stream)]
        t0 = time.perf_counter()
        decide(o)
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    return dict(mean=statistics.fmean(ts), median=ts[len(ts) // 2],
                p95=ts[int(0.95 * len(ts))], p99=ts[int(0.99 * len(ts))], mx=ts[-1])


def params(net):
    return sum(p.numel() for p in net.parameters())


print('=' * 100)
print('H4 - per-decision inference latency, single-threaded, %s decisions per network' % f'{N:,}')
print('=' * 100)
print('  torch %s | threads %d | Near-RT control interval %.0f ms'
      % (torch.__version__, torch.get_num_threads(), NEAR_RT_MS))

rng = np.random.default_rng(0)
stream = [rng.normal(0.0, 1.0, size=OBS_DIM).astype(np.float32) for _ in range(512)]

bnet, bmeta = load_branched(PUB_BRANCHED, OBS_DIM)
bnet.eval()
qsym, _ = Q.load_qr(QR_SYM, OBS_DIM)
qsym.eval()
qasym, qmeta = Q.load_qr(QR_ASYM, OBS_DIM)
qasym.eval()

print('  published branched : %s  branch_sizes=%s' % (PUB_BRANCHED, bmeta.get('branch_sizes')))
print('  qr_sym             : risk=%s  N=%d' % (qsym.risk, qsym.n_quantiles))
print('  qr_asym            : risk=%s  N=%d  alpha=%.2f  -> CVaR keeps the lowest %d quantiles'
      % (qasym.risk, qasym.n_quantiles, qasym.alpha,
         max(1, int(np.ceil(qasym.alpha * qasym.n_quantiles)))))

# The branched dueling head does a.mean(dim=1) and so needs an explicit batch dimension; the
# unsqueeze is part of the decision and stays inside the timed region. BranchedQRNet.act() does its
# own tensor construction and unsqueeze internally, so it is timed whole.
with torch.no_grad():
    r_pub = bench(lambda o: [int(q.argmax(dim=-1).item())
                             for q in bnet(torch.FloatTensor(o).to('cpu').unsqueeze(0))], stream)
r_sym = bench(lambda o: qsym.act(o), stream)      # act() is already @torch.no_grad()
r_asym = bench(lambda o: qasym.act(o), stream)

rows = [('published branched (control)', bnet, r_pub),
        ('qr_sym   51 quantiles, mean', qsym, r_sym),
        ('qr_asym  51 quantiles, CVaR', qasym, r_asym)]

print('\n  %-30s %8s %9s %8s %8s %8s %8s'
      % ('network', 'params', 'mean ms', 'median', 'p95', 'p99', 'max'))
print('  ' + '-' * 96)
for name, net, r in rows:
    print('  %-30s %8d %9.4f %8.4f %8.4f %8.4f %8.4f'
          % (name, params(net), r['mean'], r['median'], r['p95'], r['p99'], r['mx']))

d_head = r_sym['mean'] - r_pub['mean']
d_risk = r_asym['mean'] - r_sym['mean']
d_all = r_asym['mean'] - r_pub['mean']
print('\n  COST DECOMPOSITION, same hardware and one method')
print('    quantile head (published -> qr_sym) : %+.4f ms mean (%+.1f%%), %+d params (%+.1f%%)'
      % (d_head, 100 * d_head / r_pub['mean'],
         params(qsym) - params(bnet), 100.0 * (params(qsym) - params(bnet)) / params(bnet)))
print('    risk operator (qr_sym -> qr_asym)   : %+.4f ms mean (%+.1f%%), %+d params'
      % (d_risk, 100 * d_risk / r_sym['mean'], params(qasym) - params(qsym)))
print('    total          (published -> qr_asym): %+.4f ms mean (%+.1f%%), %+d params (%+.1f%%)'
      % (d_all, 100 * d_all / r_pub['mean'],
         params(qasym) - params(bnet), 100.0 * (params(qasym) - params(bnet)) / params(bnet)))

worst = max(r['mx'] for _, _, r in rows)
h4 = r_asym['mx'] < NEAR_RT_MS
print('\n' + '=' * 100)
print('  H4  worst-case quantile decision %.4f ms against the %.0f ms Near-RT interval  -> %s'
      % (r_asym['mx'], NEAR_RT_MS, 'HOLDS' if h4 else 'FAILS'))
print('      headroom factor %.0fx on the worst case, %.0fx on the mean'
      % (NEAR_RT_MS / r_asym['mx'], NEAR_RT_MS / r_asym['mean']))
print('      worst case across all three networks: %.4f ms' % worst)
print()
print('  These absolutes are an isolated loop on this machine and are NOT interchangeable with')
print('  Supplementary Note 17, which was measured on a deliberately weak 2-core cloud CPU during')
print('  a live rollout. The decomposition above is the like-for-like statement: it is measured')
print('  under one method on one machine, so the deltas are architecture costs rather than')
print('  artefacts of timing different things on different hardware.')
print('=' * 100)
