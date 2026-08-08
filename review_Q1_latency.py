#!/usr/bin/env python3
"""
review_Q1_latency.py -- Q1: how often does the guard fire one interval late, and does it depend
on velocity and on the shadow transition?

THE MECHANISM, FROM THE SOURCE. env_6g_branched.py:61 sets rsrp_before = self._prev_rsrp, and the
state-mode guard at :85 tests that STALE value against the save line; only after the transition
does :95 refresh self._prev_rsrp. So the guard is structurally one control interval behind. The
manuscript already says this ("the guard consumes the reported RSRP from the previous control
interval, so by construction it already works on delayed measurements") and Section IX-H bounds
the effect of adding ONE MORE interval. What is not measured is the baseline: how often the
built-in one-interval lag matters, and whether it concentrates at high speed, which is exactly
where the reviewer expects it (4 m per 100 ms step at 40 m/s).

WHAT IS COUNTED, per step of every evaluated episode:
    fired_stale  = the guard fired on the previous interval's RSRP  (what actually happens)
    fired_fresh  = it would have fired on THIS interval's RSRP      (a clairvoyant guard)
    LATE         = fired_fresh and not fired_stale -- the link is already below the line and the
                   guard does not yet know: protection arrives one interval late
    SPURIOUS     = fired_stale and not fired_fresh -- the link has recovered but the guard is
                   still acting on the old sample: one interval of saving is lost

Both are then binned by instantaneous velocity and split by whether the step crosses LOS->NLOS.
A clairvoyant guard is not implementable -- the comparison bounds the cost of the lag, it does
not propose removing it.

Instrumentation only: the wrapper records, it never changes an action, a power, or a transition.
"""
import csv
import os
import sys
import time
from collections import defaultdict

import numpy as np

SEEDS = [0, 1, 2, 3, 4]
R = int(os.environ.get('Q1_R', 4))
G = 9.0
VBANDS = [(0, 15), (15, 20), (20, 25), (25, 30), (30, 45)]
OUT = 'results/q1latency'


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

    PT_MAX = EB.PT_MAX_DBM
    SAVE_LINE = -92.0 + (EB.PT_MAX_DBM - EB.PT_MIN_DBM) + G
    log('save line = %.2f dBm at G = %.1f dB (PT_MAX %.0f, PT_MIN %.0f)'
        % (SAVE_LINE, G, EB.PT_MAX_DBM, EB.PT_MIN_DBM))

    if not hasattr(EB.BranchedEnv6G, '_q1_orig_step'):
        EB.BranchedEnv6G._q1_orig_step = EB.BranchedEnv6G.step

        def _step(self, action):
            before = self._prev_rsrp
            out = self._q1_orig_step(action)
            info = out[4]
            now = info['rsrp_dbm']
            REC.append((float(before), float(now), float(info['xVelocity_ms']),
                        bool(info['is_los']), float(info['pt_dbm'])))
            return out

        EB.BranchedEnv6G.step = _step

    slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH)
    pool = slow + med
    os.makedirs(OUT, exist_ok=True)
    log('Q1 guard-latency instrumentation | %d seeds x R=%d | pool %d' % (len(SEEDS), R, len(pool)))

    # RESUMABLE. The container is being reclaimed every few minutes; anything held only in
    # memory until the end is lost each time. Counts are banked per (seed, repeat) and pairs
    # already banked are skipped, so a reclaim costs at most one ~19 s evaluation.
    bank = os.path.join(OUT, 'q1_raw.csv')
    have = set()
    if os.path.exists(bank):
        for row in csv.DictReader(open(bank)):
            have.add((int(row['seed']), int(row['repeat'])))
        log('resuming: %d of %d (seed, repeat) pairs already banked'
            % (len(have), len(SEEDS) * R))
    for s in SEEDS:
        if all((s, r) in have for r in range(R)):
            continue
        net = load_branched('br_gst_s%d.pt' % s, OBS_DIM)[0]
        for r in range(R):
            if (s, r) in have:
                continue
            REC.clear()
            with T7.seeded(r):
                B.evaluate_branched(net, pool, True, guard_db=G, guard_mode='state')
            a = np.array(REC, dtype=float)
            bef, now_, vel_, los_ = a[:, 0], a[:, 1], a[:, 2], a[:, 3].astype(bool)
            fs_, ff_ = bef < SAVE_LINE, now_ < SAVE_LINE
            lt_, sp_ = ff_ & ~fs_, fs_ & ~ff_
            pl = np.concatenate([[los_[0]], los_[:-1]])
            cr = pl & ~los_
            new = not os.path.exists(bank)
            with open(bank, 'a', newline='') as fh:
                w = csv.writer(fh)
                if new:
                    w.writerow(['seed', 'repeat', 'steps', 'fired', 'would', 'late', 'spurious',
                                'cross_steps', 'cross_late', 'fast_steps', 'fast_late']
                               + ['b%d_%s' % (i, k) for i in range(len(VBANDS))
                                  for k in ('n', 'fired', 'late', 'spur')])
                hi_ = vel_ >= 30
                row = [s, r, len(a), int(fs_.sum()), int(ff_.sum()), int(lt_.sum()),
                       int(sp_.sum()), int(cr.sum()), int(lt_[cr].sum()),
                       int(hi_.sum()), int(lt_[hi_].sum())]
                for lo, hi2 in VBANDS:
                    m = (vel_ >= lo) & (vel_ < hi2)
                    row += [int(m.sum()), int(fs_[m].sum()), int(lt_[m].sum()), int(sp_[m].sum())]
                w.writerow(row)
            log('  seed %d repeat %d banked | %s steps | late %d spurious %d'
                % (s, r, format(len(a), ','), int(lt_.sum()), int(sp_.sum())))
    return report(bank)


def report(bank):
    rows = list(csv.DictReader(open(bank)))
    tot = lambda k: sum(int(x[k]) for x in rows)
    n = tot('steps')
    print('\n' + '=' * 92)
    print('Q1 -- GUARD FIRING LATENCY (structural one-interval lag), guarded arm, G = 9 dB')
    print('=' * 92)
    print('  (seed, repeat) pairs banked : %d' % len(rows))
    print('  steps analysed              : %s' % format(n, ','))
    print('  guard active (stale RSRP)   : %6.2f%%' % (100.0 * tot('fired') / n))
    print('  guard would be active (now) : %6.2f%%' % (100.0 * tot('would') / n))
    print('  LATE     (protection one interval behind)  : %6.3f%%' % (100.0 * tot('late') / n))
    print('  SPURIOUS (still protecting after recovery) : %6.3f%%' % (100.0 * tot('spurious') / n))
    print('\n  %-16s %12s %12s %12s' % ('velocity (m/s)', 'steps', 'guard on', 'LATE'))
    for i, (lo, hi) in enumerate(VBANDS):
        nb = sum(int(x['b%d_n' % i]) for x in rows)
        if not nb:
            continue
        print('  %-16s %12s %11.2f%% %11.3f%%'
              % ('%d-%d' % (lo, hi), format(nb, ','),
                 100.0 * sum(int(x['b%d_fired' % i]) for x in rows) / nb,
                 100.0 * sum(int(x['b%d_late' % i]) for x in rows) / nb))
    cs, cl = tot('cross_steps'), tot('cross_late')
    fs2, fl = tot('fast_steps'), tot('fast_late')
    if cs:
        print('\n  LOS->NLOS crossing steps : %s | LATE %6.3f%%' % (format(cs, ','), 100.0 * cl / cs))
    if fs2:
        print('  steps at >= 30 m/s       : %s | LATE %6.3f%%' % (format(fs2, ','), 100.0 * fl / fs2))
    print('=' * 92)
    return 0


def _unused(A):
    before, now, vel, is_los, pt = A[:, 0], A[:, 1], A[:, 2], A[:, 3].astype(bool), A[:, 4]
    fired_stale = before < SAVE_LINE
    fired_fresh = now < SAVE_LINE
    late = fired_fresh & ~fired_stale
    spurious = fired_stale & ~fired_fresh
    # LOS -> NLOS crossing within an episode: is_los true on the previous sample, false now.
    prev_los = np.concatenate([[is_los[0]], is_los[:-1]])
    crossing = prev_los & ~is_los

    print('\n' + '=' * 96)
    print('Q1 -- GUARD FIRING LATENCY (structural one-interval lag), guarded arm, G = 9 dB')
    print('=' * 96)
    print('  steps analysed              : %s' % format(len(A), ','))
    print('  guard active (stale RSRP)   : %6.2f%% of steps' % (100 * fired_stale.mean()))
    print('  guard would be active (now) : %6.2f%% of steps' % (100 * fired_fresh.mean()))
    print('  LATE      (protection one interval behind) : %6.3f%% of steps' % (100 * late.mean()))
    print('  SPURIOUS  (still protecting after recovery): %6.3f%% of steps'
          % (100 * spurious.mean()))
    print('\n  %-16s %10s %12s %12s %12s' % ('velocity (m/s)', 'steps', 'guard on', 'LATE',
                                             'SPURIOUS'))
    for lo, hi in VBANDS:
        m = (vel >= lo) & (vel < hi)
        if m.sum() == 0:
            continue
        print('  %-16s %10s %11.2f%% %11.3f%% %11.3f%%'
              % ('%d-%d' % (lo, hi), format(int(m.sum()), ','), 100 * fired_stale[m].mean(),
                 100 * late[m].mean(), 100 * spurious[m].mean()))
    print('\n  conditioned on the shadow transition:')
    for name, m in (('LOS->NLOS crossing steps', crossing), ('all other steps', ~crossing)):
        if m.sum():
            print('    %-26s %10s steps | LATE %6.3f%% | SPURIOUS %6.3f%%'
                  % (name, format(int(m.sum()), ','), 100 * late[m].mean(),
                     100 * spurious[m].mean()))
    hi = vel >= 30
    print('\n  at >= 30 m/s (the reviewer\'s regime): LATE %6.3f%% vs %6.3f%% below 30 m/s'
          % (100 * late[hi].mean() if hi.sum() else float('nan'),
             100 * late[~hi].mean() if (~hi).sum() else float('nan')))
    print('=' * 96)

    with open(os.path.join(OUT, 'q1_summary.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['band_lo', 'band_hi', 'steps', 'guard_on_pct', 'late_pct', 'spurious_pct'])
        for lo, hi_ in VBANDS:
            m = (vel >= lo) & (vel < hi_)
            if m.sum():
                w.writerow([lo, hi_, int(m.sum()), 100 * fired_stale[m].mean(),
                            100 * late[m].mean(), 100 * spurious[m].mean()])
    return 0


if __name__ == '__main__':
    sys.exit(main())
