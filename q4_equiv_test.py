#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""q4_equiv_test.py - the gate for q4_resumable.py.

WHY THE FIRST VERSION OF THIS FILE WAS WRONG
--------------------------------------------
It compared a published run against a re-expressed run and demanded bit-identical weights. That
premise is false in this codebase, and known to be false: env_6g_v5.SubTHzEnv6G_NYU.__init__ builds
its channel generator as np.random.default_rng(seed) with seed defaulting to None, and neither
train_branched_nyu nor run_multiseed_rel passes one. Every environment therefore draws its shadow
fading, its small-scale fading and its episode start from OS entropy. set_seed() fixes network
initialisation, the epsilon-greedy draws and which trajectory is sampled - not the channel the
agent trains against. Two runs at the same nominal seed diverge. This is already recorded in
DnDQN_Paper1_TwoRowsOneSystem_ReplicationResult_20260801.md ("No seed produced the same network
twice") and is disclosed in the paper's reproducibility statement.

So the original gate could never pass, however faithful the loop was. It measured the environment's
entropy, not the thing under test.

WHAT THIS VERSION DOES
----------------------
It makes the system deterministic FOR THE TEST ONLY, by seeding each environment's channel
generator from the legacy NumPy global - a stream that set_seed() fixes and that q4_resumable
captures and restores. Under that harness the whole pipeline is reproducible, so "identical
weights" becomes a meaningful bar. The campaign itself runs unharnessed, exactly as published:
nothing here changes how a real unit trains.

  TEST 0a  CONTROL, no harness. Published run vs published run, same seed.
           EXPECTED TO DIFFER. If these matched, the harness would be unnecessary and the whole
           premise of this file would be wrong.

           This control MUST run past LEARNING_STARTS (10,000 steps). Below it the learn block
           never fires, opt.step() is never called, and both runs save a network that is still at
           its torch.manual_seed(0) initialisation - identical, and identical for a reason that
           has nothing to do with the channel. An earlier version of this file ran the control at
           4,000 steps and reported a spurious FAIL for exactly that reason.

  TEST 0b  HARNESS VALIDATION. Published vs published, harnessed.
           Must be identical, or the harness does not actually pin the entropy and nothing below
           can be trusted.

  TEST A   FIDELITY. Published train() vs train_resumable(), harnessed.
           Must be identical: the loop was re-expressed, not reimplemented.

  TEST B   TRANSPARENCY. Straight run vs a run genuinely killed and resumed, harnessed. The kill is
           real - the child calls os._exit() the instant the snapshot reaches disk, no cleanup, no
           flush. Miss one Adam moment, one sum-tree priority, one RNG stream or the environment's
           fading cache and the weights diverge.

  TEST C   TEST B again through the QR-DQN rebinding, the path a real q4 unit takes.

USAGE
    python -u q4_equiv_test.py
    python -u q4_equiv_test.py --steps 8000     # faster, less coverage
    python -u q4_equiv_test.py --scalar-only    # skip the QR path
"""
import os
import sys
import time
import argparse
import subprocess

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

WORK = 'results/q4_equiv'


def install_harness(B):
    """Seed every environment's channel RNG from the legacy NumPy global.

    The legacy global is fixed by B.set_seed() and is captured and restored by q4_resumable, so
    under this harness the channel realisation is reproducible AND survives a resume. Drawing from
    it (rather than from a private stream) is what makes TEST B meaningful: a private stream would
    restart at resume and the comparison would fail for a reason that has nothing to do with the
    snapshot.

    BranchedEnv6G does not accept or forward a seed, so the generator is replaced after
    construction. __init__ consumes no randomness before that point - _reset_state() only clears
    the shadow cache - and reset() is called afterwards by the training loop, so it picks up the
    seeded generator.
    """
    import numpy as np
    original = B.BranchedEnv6G

    def seeded(traj, *a, **kw):
        e = original(traj, *a, **kw)
        e.rng = np.random.default_rng(int(np.random.randint(0, 2 ** 31 - 1)))
        return e

    B.BranchedEnv6G = seeded


def child(mode, steps, crash_at, out, snap, harness):
    import torch
    torch.set_num_threads(1)
    saved = sys.argv[:]
    sys.argv = ['train_branched_nyu.py', '1', str(steps)]
    try:
        import train_branched_nyu as B
    finally:
        sys.argv = saved
    import train_phase2 as tp
    import q4_resumable as R

    if mode.startswith('qr'):
        import q4_qr_branched as Q
        B.build_branched_networks = lambda obs_dim, *a, **k: Q.build_qr_networks(
            obs_dim, risk=Q.RISK_ASYM)
        B.branched_td_loss = Q.qr_branched_td_loss
        B.save_branched = Q.save_qr

    if harness:
        install_harness(B)

    slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH)

    if crash_at:
        real = R._atomic_save

        def killer(obj, path, Bmod):
            real(obj, path, Bmod)
            print('  [child] snapshot written at step %d - killing process' % obj['step'],
                  flush=True)
            sys.stdout.flush()
            os._exit(9)
        R._atomic_save = killer

    if mode.endswith('published'):
        B.train(0, True, out, slow, med, steps=steps, guard_db=9.0, guard_mode='state')
    else:
        R.train_resumable(B, 0, True, out, slow, med, steps=steps,
                          guard_db=9.0, guard_mode='state',
                          snap_path=snap, snap_every=(crash_at - 1) if crash_at else 10 ** 9,
                          log=lambda m: print('  [child] ' + m, flush=True))
    return 0


def run(tag, mode, steps, out, snap=None, crash_at=0, harness=True):
    cmd = [sys.executable, '-u', os.path.abspath(__file__), '--child', mode,
           '--steps', str(steps), '--out', out, '--crash-at', str(crash_at)]
    if snap:
        cmd += ['--snap', snap]
    if harness:
        cmd += ['--harness']
    t0 = time.time()
    p = subprocess.run(cmd, cwd=os.getcwd())
    print('    %-36s rc=%-3s %.0fs' % (tag, p.returncode, time.time() - t0), flush=True)
    return p.returncode


def learning_starts(path='train_branched_nyu.py'):
    """Read LEARNING_STARTS from the published trainer rather than hardcoding it here.

    The control below is only capable of detecting a divergence if it runs long enough for a
    gradient step to happen; below this threshold every run saves its initialisation unchanged.
    """
    import re
    try:
        m = re.search(r'LEARNING_STARTS[^=\n]*=\s*([\d_]+)', open(path, encoding='utf-8').read())
        return int(m.group(1).replace('_', '')) if m else 10_000
    except OSError:
        return 10_000


def same(a, b):
    import torch
    ca = torch.load(a, map_location='cpu', weights_only=False)
    cb = torch.load(b, map_location='cpu', weights_only=False)
    sa, sb = ca['q_net_state'], cb['q_net_state']
    if set(sa) != set(sb):
        return False, 'different parameter sets'
    bad = [k for k in sa if not torch.equal(sa[k], sb[k])]
    if bad:
        worst = max(float((sa[k].double() - sb[k].double()).abs().max()) for k in bad)
        return False, '%d/%d tensors differ, max abs delta %.3e' % (len(bad), len(sa), worst)
    return True, '%d/%d tensors identical' % (len(sa), len(sa))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--child', default='')
    ap.add_argument('--steps', type=int, default=14000)
    ap.add_argument('--out', default='')
    ap.add_argument('--snap', default='')
    ap.add_argument('--crash-at', type=int, default=0)
    ap.add_argument('--harness', action='store_true')
    ap.add_argument('--scalar-only', action='store_true')
    ap.add_argument('--control-only', action='store_true',
                    help='run TEST 0a alone (the unseeded-channel control)')
    a = ap.parse_args()

    if a.child:
        return child(a.child, a.steps, a.crash_at, a.out, a.snap or None, a.harness)

    steps, crash = a.steps, int(a.steps * 0.8)
    ls = learning_starts()
    if steps <= ls:
        print('REFUSING: --steps %d is at or below LEARNING_STARTS (%d), so opt.step() never '
              'fires\n          and every run saves an untrained network. Use --steps %d or more.'
              % (steps, ls, ls + 2000))
        return 2
    ctrl = steps          # the control must clear LEARNING_STARTS to be capable of divergence
    os.makedirs(WORK, exist_ok=True)
    for f in os.listdir(WORK):
        os.remove(os.path.join(WORK, f))
    W = lambda n: os.path.join(WORK, n)
    results = []

    print('=' * 86)
    print('  q4 RESUME EQUIVALENCE GATE')
    print('  %s steps per run, simulated kill at step %s, seed 0, guard 9.0 state'
          % ('{:,}'.format(steps), '{:,}'.format(crash)))
    print('=' * 86)

    print('\nTEST 0a - CONTROL: published vs published, NO harness (%s steps, LEARNING_STARTS=%s)'
          % ('{:,}'.format(ctrl), '{:,}'.format(ls)))
    print('    the channel RNG is unseeded by design, so these SHOULD differ')
    run('published run 1', 'published', ctrl, W('z1.pt'), harness=False)
    run('published run 2', 'published', ctrl, W('z2.pt'), harness=False)
    ok, why = same(W('z1.pt'), W('z2.pt'))
    results.append(('0a control: unseeded channel confirmed', not ok,
                    ('differ as expected - %s' % why) if not ok else
                    'IDENTICAL - unexpected; the premise of this gate is wrong'))
    print('    -> %s  %s' % ('PASS' if not ok else 'FAIL', results[-1][2]))
    if a.control_only:
        print('\n  control only - stopping here.')
        return 0 if not ok else 1

    print('\nTEST 0b - HARNESS VALIDATION: published vs published, harnessed')
    run('published harnessed run 1', 'published', steps, W('h1.pt'))
    run('published harnessed run 2', 'published', steps, W('h2.pt'))
    ok, why = same(W('h1.pt'), W('h2.pt'))
    results.append(('0b harness pins the entropy', ok, why))
    print('    -> %s  %s' % ('PASS' if ok else 'FAIL', why))
    if not ok:
        print('\n  The harness does not pin the channel. Nothing below is interpretable.')
        return 1

    print('\nTEST A - loop fidelity: published vs train_resumable, harnessed')
    run('train_resumable, no snapshots', 'resumable', steps, W('a_res.pt'))
    ok, why = same(W('h1.pt'), W('a_res.pt'))
    results.append(('A  loop fidelity', ok, why))
    print('    -> %s  %s' % ('PASS' if ok else 'FAIL', why))

    print('\nTEST B - resume transparency: straight vs killed-and-resumed, harnessed')
    rc = run('killed at snapshot', 'resumable', steps, W('b.pt'), W('b.snap'), crash)
    if rc != 9:
        results.append(('B  resume transparency', False, 'child rc=%s, expected 9' % rc))
    else:
        print('    snapshot %.1f MB' % (os.path.getsize(W('b.snap')) / 1e6))
        run('resumed from snapshot', 'resumable', steps, W('b.pt'), W('b.snap'))
        ok, why = same(W('a_res.pt'), W('b.pt'))
        results.append(('B  resume transparency', ok, why))
        print('    -> %s  %s' % ('PASS' if ok else 'FAIL', why))

    if not a.scalar_only:
        print('\nTEST C - the real q4 path: QR-DQN, straight vs killed-and-resumed, harnessed')
        run('qr straight', 'qr_resumable', steps, W('c1.pt'))
        rc = run('qr killed at snapshot', 'qr_resumable', steps, W('c2.pt'), W('c.snap'), crash)
        if rc != 9:
            results.append(('C  resume transparency, QR-DQN', False,
                            'child rc=%s, expected 9' % rc))
        else:
            print('    snapshot %.1f MB' % (os.path.getsize(W('c.snap')) / 1e6))
            run('qr resumed from snapshot', 'qr_resumable', steps, W('c2.pt'), W('c.snap'))
            ok, why = same(W('c1.pt'), W('c2.pt'))
            results.append(('C  resume transparency, QR-DQN', ok, why))
            print('    -> %s  %s' % ('PASS' if ok else 'FAIL', why))

    print('\n' + '=' * 86)
    print('  VERDICT')
    print('=' * 86)
    for name, ok, why in results:
        print('  %-38s %-6s %s' % (name, 'PASS' if ok else 'FAIL', why))
    allok = all(ok for _, ok, _ in results)
    print()
    if allok:
        print('  ALL PASS. train_resumable reproduces the published loop exactly under a pinned')
        print('  channel, and a hard kill is invisible in the resulting weights.')
        print('  The campaign runs UNHARNESSED - the channel stays unseeded, exactly as published.')
    else:
        print('  FAILED - do NOT launch. Send me this output.')
    return 0 if allok else 1


if __name__ == '__main__':
    sys.exit(main())
