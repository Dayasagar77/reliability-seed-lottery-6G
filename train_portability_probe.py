#!/usr/bin/env python3
"""
train_portability_probe.py -- does TRAINING reproduce in this container?

WHY THIS EXISTS. Evaluation is now proven bit-exact here (40/40 cells, guarded seed 0 vs the
laptop grid). But tablevii_r8.py's own header records that TRAINING is NON-portable between
venues: the same seed, config and 800k budget gave HO 93.00 vs 91.04 and Power 18.96 vs 24.52
on seed 0 of the flat arm. That comparison was laptop/torch-2.5.1 against cloud/torch-2.13.0.
This container is a THIRD configuration -- cloud hardware, laptop toolchain -- so the old
finding does not settle the question.

It matters because Q2 (three-level actuator) and Q5 (throughput-floor reward) require NEW
TRAINING. A row trained in a venue whose training does not reproduce the laptop's cannot be
placed in Table VII beside laptop-trained rows: that is precisely the defect the R=8 campaign
was built to remove.

TWO THINGS ARE MEASURED
  1. WITHIN-VENUE DETERMINISM. Train the same config twice here and compare weights bit-for-bit.
     Necessary but not sufficient for cross-venue reproduction -- if it fails here, stop.
  2. STEP RATE, so the full 5-seed x 8e5-step campaigns can be sized honestly rather than guessed.

Short budget on purpose. This is a probe, not a result; nothing it produces goes in the paper.
"""
import hashlib
import sys
import time

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 20_000
sys.argv = [sys.argv[0]]

import numpy as np                                                   # noqa: E402
import torch                                                         # noqa: E402
torch.set_num_threads(1)
import train_branched_nyu as B                                       # noqa: E402
import train_phase2 as tp                                            # noqa: E402


def wsig(path):
    """md5 over every tensor in the checkpoint, in sorted key order.

    The first version of this looked for ck['model'] and, finding nothing, hashed zero bytes --
    so both runs returned md5(''), and the determinism check "passed" without ever comparing a
    weight. This project has now been bitten three times by a gate that passes because its
    detector never fired. So: walk the checkpoint for tensors WHEREVER they are, and refuse to
    return a signature at all if fewer than MIN_T tensors were hashed."""
    MIN_T = 8
    ck = torch.load(path, map_location='cpu', weights_only=False)
    h = hashlib.md5()
    n = [0]

    def walk(obj, prefix):
        if hasattr(obj, 'detach') and hasattr(obj, 'cpu'):
            h.update(prefix.encode())
            h.update(np.ascontiguousarray(obj.detach().cpu().numpy()).tobytes())
            n[0] += 1
        elif isinstance(obj, dict):
            for k in sorted(obj.keys(), key=str):
                walk(obj[k], prefix + '/' + str(k))
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                walk(v, prefix + '/%d' % i)

    walk(ck, '')
    if n[0] < MIN_T:
        raise SystemExit('wsig found only %d tensors in %s -- the signature would be vacuous. '
                         'Refusing to report a determinism verdict.' % (n[0], path))
    return '%s (%d tensors)' % (h.hexdigest(), n[0])


print('=' * 96)
print('TRAINING PORTABILITY PROBE  |  guarded arm, seed 0, %s steps' % format(STEPS, ','))
print('  torch %s | numpy %s | threads %d' % (torch.__version__, np.__version__,
                                              torch.get_num_threads()))
print('=' * 96)

slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH)
print('  pool: slow %d + med %d = %d (fast %d held out, %d skipped)'
      % (len(slow), len(med), len(slow) + len(med), len(fast), skip))

times = []
for run in (1, 2):
    out = 'probe_run%d.pt' % run
    t0 = time.time()
    B.train(0, True, out, slow, med, steps=STEPS, guard_db=9.0, guard_mode='state')
    dt = time.time() - t0
    times.append(dt)
    print('  run %d: %.1fs  (%.0f steps/s)  sig %s' % (run, dt, STEPS / dt, wsig(out)))

a, b = wsig('probe_run1.pt'), wsig('probe_run2.pt')
print()
print('  WITHIN-VENUE DETERMINISM: %s' % ('BIT-IDENTICAL' if a == b else '*** DIVERGED ***'))
rate = STEPS / (sum(times) / len(times))
full = 800_000 / rate
print('  step rate %.0f/s  ->  one 8e5-step training = %.1f min' % (rate, full / 60))
print('  5-seed campaign = %.1f h serial;  two campaigns = %.1f h serial' % (5 * full / 3600,
                                                                            10 * full / 3600))
print('=' * 96)
