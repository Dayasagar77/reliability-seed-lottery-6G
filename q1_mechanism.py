#!/usr/bin/env python3
"""
q1_mechanism.py -- decompose Q1's LATE and SPURIOUS counts by MECHANISM before either number is
written into the paper.

WHY THIS EXISTS. review_Q1_latency.py measures how often the guard is one control interval behind
and reports LATE = 8.307 %, SPURIOUS = 8.222 %. Those are correct counts of the event
"stale and fresh disagree about the save line". They are NOT yet evidence for a single mechanism,
and the paper would have attributed them to one: the structural one-interval lag.

Two things in the banked summary said the attribution had to be checked first.

  (a) env_6g_branched.py:52,56 initialise self._prev_rsrp = -70.0 on construction AND on every
      reset. -70.0 sits ABOVE the save line (-76 dBm), so on the first step of every episode the
      guard cannot fire no matter how bad the link is. If the fresh RSRP is already below the
      line there, the step is counted LATE -- correctly, the guard really is not protecting -- but
      the cause is a COLD START, not a one-interval lag. Cold starts cannot be SPURIOUS at all,
      since the sentinel is above the line.

  (b) The 0-15 m/s band reported guard-on 50.62 % with SPURIOUS 50.43 % and LATE 0.161 %. A
      symmetric one-interval lag on a stationary process produces roughly equal LATE and SPURIOUS
      (globally it does: 8.307 vs 8.222). A band that is almost all SPURIOUS and almost no LATE is
      not a lag signature; it is a systematic upward level shift between consecutive samples.
      Something other than fading was moving RSRP in that band, and it had to be named.

The first pass answered it: the serving cell changes on 25.6 % of all steps, and on a handover
step `before` and `now` are measurements of TWO DIFFERENT CELLS. Comparing them does not measure a
delay at all. So every LATE and every SPURIOUS step is assigned to exactly one mechanism, in
priority order:

  coldstart   the first step of the episode -- the stale value is the -70.0 sentinel, not a
              measurement.
  handover    info['handover_done'] -- the serving cell changed this step, so the two RSRP samples
              come from different cells. The disagreement is a discontinuity by construction.
              Note what a handover-coincident SPURIOUS step actually is: the old cell was below the
              line, the new cell is above it, and the guard spends one interval at full power while
              the handover completes. That is the guard behaving sensibly, not a fault.
  los_nlos    the shadow transition on an UNCHANGED serving cell (LOS->NLOS for LATE, NLOS->LOS
              for SPURIOUS): geometry moved the level of the same link within one interval.
  residual    none of the above -- fading/pathloss drift across one interval on an unchanged link.
              THIS is the pure structural-lag term, and it is the only one the paper may call that.

The partition is exact and exhaustive by construction, and the script asserts it.

TRANSPARENCY GATE. The richer recorder must reproduce the already-banked q1_raw.csv counts for
every (seed, repeat) row EXACTLY -- same steps, fired, would, late, spurious. If the extra
instrumentation perturbed anything, the decomposition would be describing a different run than the
one the paper reports, so the script stops instead.

BASIS. Full banked grid: 5 seeds x R=4, the same CRN blocks as q1_raw.csv, so the residual rate
carries a between-training dispersion computed with the same pinned estimator
(tablevii_decompose) and the same frozen resolvability rule as every other number in the campaign.
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
BANK = os.path.join(OUT, 'q1_raw.csv')
MECH = os.path.join(OUT, 'q1_mechanism_raw.csv')

MECHS = ('coldstart', 'handover', 'los_nlos', 'residual')
FIELDS = (['seed', 'repeat', 'steps', 'fired', 'would', 'late', 'spurious',
           'cold_steps', 'ho_steps', 'to_nlos', 'to_los',
           'cross_steps', 'cross_late', 'cross_nonho', 'cross_nonho_late',
           'noncross_nonho', 'noncross_nonho_late']
          + ['late_' + m for m in MECHS] + ['spur_' + m for m in MECHS]
          + ['b%d_%s' % (i, k) for i in range(len(VBANDS))
             for k in ('n', 'late', 'spur', 'late_residual', 'spur_residual',
                       'ho', 'nonho_n', 'nonho_late')])

REC = []


def log(m):
    print(time.strftime('%Y-%m-%d %H:%M:%S') + ' | ' + m, flush=True)


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

    SAVE_LINE = -92.0 + (EB.PT_MAX_DBM - EB.PT_MIN_DBM) + G
    SENTINEL = -70.0
    log('save line = %.2f dBm | reset sentinel _prev_rsrp = %.1f dBm (above the line: %s)'
        % (SAVE_LINE, SENTINEL, SENTINEL >= SAVE_LINE))

    if not hasattr(EB.BranchedEnv6G, '_qm_orig_step'):
        EB.BranchedEnv6G._qm_orig_step = EB.BranchedEnv6G.step

        def _step(self, action):
            before = self._prev_rsrp
            out = self._qm_orig_step(action)
            info = out[4]
            REC.append((float(before), float(info['rsrp_dbm']), float(info['xVelocity_ms']),
                        bool(info['is_los']), int(info['episode_step']),
                        bool(info['handover_done'])))
            return out

        EB.BranchedEnv6G.step = _step

    slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH)
    pool = slow + med

    banked = {(int(r['seed']), int(r['repeat'])): r for r in csv.DictReader(open(BANK))}

    have = set()
    if os.path.exists(MECH):
        for r in csv.DictReader(open(MECH)):
            have.add((int(r['seed']), int(r['repeat'])))
        log('resuming: %d of %d (seed, repeat) pairs already decomposed' % (len(have), len(SEEDS) * R))

    for s in SEEDS:
        if all((s, r) in have for r in range(R)):
            continue
        net = load_branched('br_gst_s%d.pt' % s, OBS_DIM)[0]
        for rep in range(R):
            if (s, rep) in have:
                continue
            REC.clear()
            t0 = time.time()
            with T7.seeded(rep):
                B.evaluate_branched(net, pool, True, guard_db=G, guard_mode='state')
            a = np.array(REC, dtype=float)
            bef, now, vel = a[:, 0], a[:, 1], a[:, 2]
            los = a[:, 3].astype(bool)
            estep = a[:, 4].astype(int)
            ho = a[:, 5].astype(bool)

            fired, would = bef < SAVE_LINE, now < SAVE_LINE
            late, spur = would & ~fired, fired & ~would

            # ---- transparency gate: must reproduce the banked row exactly ----
            b = banked[(s, rep)]
            want = (int(b['steps']), int(b['fired']), int(b['would']), int(b['late']),
                    int(b['spurious']))
            got = (len(a), int(fired.sum()), int(would.sum()), int(late.sum()), int(spur.sum()))
            if want != got:
                log('  TRANSPARENCY FAIL s%d r%d: banked %s vs re-run %s' % (s, rep, want, got))
                print('\nSTOPPING -- the richer recorder does not reproduce the banked run, so this'
                      ' decomposition would describe a different experiment than the one reported.')
                return 3

            cold = estep == estep.min()
            assert np.all(np.abs(bef[cold] - SENTINEL) < 1e-9), 'cold-start stale value is not -70.0'
            prev_los = np.concatenate([[los[0]], los[:-1]])
            to_nlos = prev_los & ~los
            to_los = ~prev_los & los

            base = ~cold & ~ho                      # unchanged serving cell, real prior measurement
            m_late = {'coldstart': late & cold, 'handover': late & ~cold & ho,
                      'los_nlos': late & base & to_nlos, 'residual': late & base & ~to_nlos}
            m_spur = {'coldstart': spur & cold, 'handover': spur & ~cold & ho,
                      'los_nlos': spur & base & to_los, 'residual': spur & base & ~to_los}
            assert sum(int(v.sum()) for v in m_late.values()) == int(late.sum())
            assert sum(int(v.sum()) for v in m_spur.values()) == int(spur.sum())

            row = {'seed': s, 'repeat': rep, 'steps': len(a), 'fired': int(fired.sum()),
                   'would': int(would.sum()), 'late': int(late.sum()), 'spurious': int(spur.sum()),
                   'cold_steps': int(cold.sum()), 'ho_steps': int(ho.sum()),
                   'to_nlos': int(to_nlos.sum()), 'to_los': int(to_los.sum()),
                   'cross_steps': int(to_nlos.sum()), 'cross_late': int(late[to_nlos].sum()),
                   'cross_nonho': int((to_nlos & base).sum()),
                   'cross_nonho_late': int((late & to_nlos & base).sum()),
                   'noncross_nonho': int((~to_nlos & base).sum()),
                   'noncross_nonho_late': int((late & ~to_nlos & base).sum())}
            for m in MECHS:
                row['late_' + m] = int(m_late[m].sum())
                row['spur_' + m] = int(m_spur[m].sum())
            for i, (lo, hi) in enumerate(VBANDS):
                mb = (vel >= lo) & (vel < hi)
                row['b%d_n' % i] = int(mb.sum())
                row['b%d_late' % i] = int((mb & late).sum())
                row['b%d_spur' % i] = int((mb & spur).sum())
                row['b%d_late_residual' % i] = int((mb & m_late['residual']).sum())
                row['b%d_spur_residual' % i] = int((mb & m_spur['residual']).sum())
                row['b%d_ho' % i] = int((mb & ho).sum())
                row['b%d_nonho_n' % i] = int((mb & base).sum())
                row['b%d_nonho_late' % i] = int((mb & base & late).sum())

            new = not os.path.exists(MECH)
            with open(MECH, 'a', newline='') as fh:
                w = csv.DictWriter(fh, fieldnames=list(FIELDS))
                if new:
                    w.writeheader()
                w.writerow(row)
            log('  s%d r%d ok (%.0fs) | late %d (residual %d) | spurious %d (residual %d)'
                % (s, rep, time.time() - t0, row['late'], row['late_residual'],
                   row['spurious'], row['spur_residual']))
    return report()


def report():
    rows = list(csv.DictReader(open(MECH)))
    T = lambda k: sum(int(x[k]) for x in rows)
    n = T('steps')
    W = 98
    print('\n' + '=' * W)
    print('Q1 MECHANISM DECOMPOSITION -- what is actually producing LATE and SPURIOUS')
    print('  basis: %d (seed, repeat) pairs, %s steps, CRN blocks identical to q1_raw.csv'
          % (len(rows), format(n, ',')))
    print('=' * W)
    print('  TRANSPARENCY: every re-run reproduced its banked (steps, fired, would, late, '
          'spurious) exactly.')
    print('\n  episode starts   : %s (%.3f%% of steps)  -- stale value is the -70.0 sentinel'
          % (format(T('cold_steps'), ','), 100.0 * T('cold_steps') / n))
    print('  serving-cell changes : %s (%.2f%% of steps) -- `before` and `now` are DIFFERENT cells'
          % (format(T('ho_steps'), ','), 100.0 * T('ho_steps') / n))
    print('  LOS->NLOS steps  : %s      NLOS->LOS steps : %s'
          % (format(T('to_nlos'), ','), format(T('to_los'), ',')))

    for name, tot_key, pref in (('LATE', 'late', 'late_'), ('SPURIOUS', 'spurious', 'spur_')):
        t = T(tot_key)
        print('\n  %-9s total %s steps = %.3f%% of all steps' % (name, format(t, ','), 100.0 * t / n))
        print('  %-12s %12s %10s %14s' % ('mechanism', 'steps', 'share', 'as % of all'))
        for k in MECHS:
            c = T(pref + k)
            print('  %-12s %12s %9.1f%% %13.3f%%'
                  % (k, format(c, ','), 100.0 * c / max(t, 1), 100.0 * c / n))

    print('\n  ' + '-' * (W - 4))
    print('  THE REVIEWER\'S QUESTION, WITH THE CONFOUND REMOVED')
    print('  (unchanged serving cell only: no handover step, no episode start)')
    print('  ' + '-' * (W - 4))
    print('  %-12s %12s %12s %12s %12s' % ('band (m/s)', 'steps', 'LATE raw%', 'unchanged-cell',
                                           'LATE there%'))
    for i, (lo, hi) in enumerate(VBANDS):
        nb = sum(int(x['b%d_n' % i]) for x in rows)
        if not nb:
            continue
        nh = sum(int(x['b%d_nonho_n' % i]) for x in rows)
        nl = sum(int(x['b%d_nonho_late' % i]) for x in rows)
        print('  %-12s %12s %11.3f%% %12s %11.3f%%'
              % ('%d-%d' % (lo, hi), format(nb, ','),
                 100.0 * sum(int(x['b%d_late' % i]) for x in rows) / nb,
                 format(nh, ','), 100.0 * nl / max(nh, 1)))

    cs, cl = T('cross_nonho'), T('cross_nonho_late')
    ns, nl2 = T('noncross_nonho'), T('noncross_nonho_late')
    raw_c, raw_cl = T('cross_steps'), T('cross_late')
    print('\n  shadow transition, RAW (as first reported):')
    print('    LATE %6.3f%% on %s LOS->NLOS steps vs %6.3f%% elsewhere -- ratio %.2fx'
          % (100.0 * raw_cl / raw_c, format(raw_c, ','),
             100.0 * (T('late') - raw_cl) / (n - raw_c),
             (raw_cl / raw_c) / ((T('late') - raw_cl) / (n - raw_c))))
    print('  shadow transition, UNCHANGED SERVING CELL ONLY:')
    print('    LATE %6.3f%% on %s steps vs %6.3f%% on %s steps -- ratio %.2fx'
          % (100.0 * cl / max(cs, 1), format(cs, ','), 100.0 * nl2 / max(ns, 1),
             format(ns, ','), (cl / max(cs, 1)) / max(nl2 / max(ns, 1), 1e-12)))
    print('    of the %s raw LOS->NLOS LATE steps, %s (%.1f%%) coincide with a handover'
          % (format(raw_cl, ','), format(raw_cl - cl, ','), 100.0 * (raw_cl - cl) / max(raw_cl, 1)))

    # ---- dispersion of the residual (pure-lag) LATE rate, pinned estimator ----
    sys.path.insert(0, '/home/claude/work')
    import tablevii_decompose as D
    X = np.array([[100.0 * int(next(x for x in rows if int(x['seed']) == s
                                    and int(x['repeat']) == rr)['late_residual'])
                   / int(next(x for x in rows if int(x['seed']) == s
                              and int(x['repeat']) == rr)['steps'])
                   for rr in range(R)] for s in SEEDS])
    d = D.decompose(X)
    resolvable = bool(d['v_train'] > 0 and d['p'] < 0.05)
    print('\n  ' + '-' * (W - 4))
    print('  DISPERSION of the residual (pure one-interval lag) LATE rate')
    print('  ' + '-' * (W - 4))
    print('  grid           : %d seeds x R=%d' % (len(SEEDS), R))
    print('  mean           : %.4f%%' % d['grand'])
    print('  per-training   : ' + ', '.join('%.4f' % v for v in d['seed_means']))
    print('  v_train        : %.7g   sd_train : %.4f' % (d['v_train'], d['sd_train']))
    print('  F(%d,%d)        : %.4f   p = %.4g' % (len(SEEDS) - 1,
                                                   (len(SEEDS) - 1) * (R - 1), d['f_obs'], d['p']))
    print('  RESOLVABLE     : %s' % resolvable)
    if resolvable:
        print('  -> report as   : %.3f%% +- %.3f%% (between-training)' % (d['grand'], d['sd_train']))
    else:
        print('  -> report as   : %.3f%% (sigma_train < %.3f%%)' % (d['grand'], d['sd_train_hi']))
    print('=' * W)
    return 0


if __name__ == '__main__':
    sys.exit(main())
