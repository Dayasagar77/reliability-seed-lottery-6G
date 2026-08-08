#!/usr/bin/env python3
"""
t6_positional.py -- reviewer 2, T6: does appending MID at index 3 bias the output layer?

    "the intermediate level is appended rather than inserted... the paper should confirm that the
     appended position does not introduce a positional bias in the network's output layer."

THE BEHAVIOURAL EVIDENCE COMES FIRST, because it is the stronger argument and it already exists.
A positional bias in the output layer is a property of the ARCHITECTURE, so it would push every
training the same way. Measured MID usage across five identically configured trainings is
0.0, 0.1, 19.4, 48.2 and 62.3 % of steps. A systematic bias cannot produce a 62-point spread; that
is the retraining lottery, not a wiring artefact.

This script tests the mechanism directly anyway, because "it can't be that" is an argument and
this project reports measurements:

  1. output-layer bias terms per action index, across all five nets
  2. mean advantage per action index over a common observation sample
  3. whether index 3 is an outlier in either, CONSISTENTLY across nets

A real positional bias shows up as index 3 sitting apart in the same direction in every net. Rank
disagreement across nets is the signature of ordinary training variation.
"""
import sys

import numpy as np

sys.argv = sys.argv[:1]
import torch                                                            # noqa: E402
torch.set_num_threads(1)
from dndqn import OBS_DIM                                               # noqa: E402
from dndqn_branched import load_branched                                # noqa: E402

CKPTS = ['q2_3l_s%d.pt' % s for s in range(5)]
LABEL = ['PROTECT', 'SAVE', 'HANDOVER', 'MID']
USAGE = [11.4, 11.5, 58.3, 15.4, 54.1], [43.5, 0.0, 24.3, 13.9, 28.2], \
        [25.7, 26.2, 17.3, 22.4, 17.7], [19.4, 62.3, 0.1, 48.2, 0.0]

print('=' * 92)
print('T6 -- POSITIONAL BIAS OF THE APPENDED MID ACTION')
print('=' * 92)
print('  behavioural evidence, from the Q2 action distribution:')
print('    MID usage across five identical trainings: %s'
      % ', '.join('%.1f%%' % v for v in USAGE[3]))
print('    spread %.1f points. An output-layer bias is architectural and would push every'
      % (max(USAGE[3]) - min(USAGE[3])))
print('    training the same way; it cannot produce a spread of this size.')

rng = np.random.default_rng(0)
sample = torch.tensor(rng.normal(0, 1, size=(4096, OBS_DIM)).astype(np.float32))

print('\n  MECHANISM CHECK -- reliability-branch output layer, per action index')
print('  %-6s %-10s %10s %10s %10s %10s' % ('net', 'quantity', *LABEL))
bias_rows, adv_rows = [], []
for i, ck in enumerate(CKPTS):
    net, meta = load_branched(ck, OBS_DIM)
    net.eval()
    # last Linear of the reliability branch (branch 0)
    lin = None
    for mod in net.modules():
        if isinstance(mod, torch.nn.Linear) and mod.out_features == 4:
            lin = mod
    if lin is None:
        print('    s%d: no 4-output Linear found; branch_sizes=%s' % (i, meta.get('branch_sizes')))
        continue
    b = lin.bias.detach().numpy()
    with torch.no_grad():
        q = net(sample)[0].numpy()
    adv = q.mean(axis=0) - q.mean()
    bias_rows.append(b)
    adv_rows.append(adv)
    print('  s%-5d %-10s %10.4f %10.4f %10.4f %10.4f' % (i, 'bias', *b))
    print('  %-6s %-10s %10.4f %10.4f %10.4f %10.4f' % ('', 'mean adv', *adv))

if bias_rows:
    B = np.array(bias_rows)
    A = np.array(adv_rows)
    print('\n  is index 3 (MID) consistently apart?')
    for name, M in (('bias', B), ('mean advantage', A)):
        ranks = [int(np.argsort(np.argsort(-row))[3]) + 1 for row in M]   # 1 = highest
        print('    %-15s MID rank per net: %s   %s'
              % (name, ranks,
                 'SAME RANK EVERY NET -- investigate' if len(set(ranks)) == 1
                 else 'ranks disagree across nets -> no systematic positional effect'))
    print('\n  correlation between a net\'s MID advantage and its MID usage: %.3f'
          % np.corrcoef(A[:, 3], np.array(USAGE[3]))[0, 1])
    print('    (high correlation means usage follows what the net learned, not where the action')
    print('     sits in the output vector)')
print('=' * 92)
