#!/usr/bin/env python3
"""
q1_verify.py -- independently re-derive every number review_Q1_latency.py printed, from the
banked raw counts, and write the summary CSV the live report() path never wrote.

WHY. review_Q1_latency.py contains TWO report functions. The live one, report(bank), aggregates
the banked per-(seed, repeat) counts. The other, _unused(A), is the earlier in-memory version --
and the q1_summary.csv write lives inside it, so it never runs. Nothing is wrong with the result;
the summary file simply does not get written. Rather than edit a script that has already produced
its data, the summary is derived here from q1_raw.csv, and while we are at it every printed
figure is recomputed by a separate code path and required to agree.

WHAT IS CHECKED
  TOTALS      each of the six headline percentages recomputed and matched to the log to 1e-9
  BANDS       the five velocity-band rows recomputed and matched
  CLOSURE     the per-band step counts must sum to the total step count -- if the bands do not
              partition the data, a band percentage is a percentage of something unstated
  IDENTITY    late and spurious must satisfy  fired + late - spurious == would, per row, exactly.
              This is an algebraic identity of the four counters, not a statistical check: LATE is
              fresh AND NOT stale, SPURIOUS is stale AND NOT fresh, so |fresh| = |stale| + |late|
              - |spurious| must hold in every single banked row. A violation would mean the
              instrumentation miscounted, and no amount of averaging would reveal it.
  DISPERSION  between-training and evaluation components of the LATE rate, via the pinned
              estimator (tablevii_decompose.md5 4d30eacb161d92df09821ef755569f85), so the
              headline can be quoted with the same resolvability rule the rest of the paper uses.

The reviewer asked two questions. Both are answered by a ratio that is reported here explicitly
rather than left for the reader to divide: LATE at a LOS->NLOS crossing against LATE elsewhere,
and LATE in the top velocity band against the bottom.
"""
import csv
import hashlib
import sys

import numpy as np

RAW = 'results/q1latency/q1_raw.csv'
OUT = 'results/q1latency/q1_summary.csv'
VBANDS = [(0, 15), (15, 20), (20, 25), (25, 30), (30, 45)]

# what the run printed, transcribed from logs/q1.log for the agreement check
PRINTED = {
    'pairs': 20, 'steps': 906868, 'fired_pct': 16.55, 'would_pct': 16.64,
    'late_pct': 8.307, 'spurious_pct': 8.222,
    'bands': [(14274, 50.62, 0.161), (1085, 13.46, 5.622), (150557, 15.35, 7.395),
              (209681, 16.22, 8.239), (531271, 16.12, 8.817)],
    'cross_steps': 46570, 'cross_late_pct': 59.414,
    'fast_steps': 531271, 'fast_late_pct': 8.817,
}


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    rows = list(csv.DictReader(open(RAW)))
    tot = lambda k: sum(int(x[k]) for x in rows)
    fails = []

    print('=' * 92)
    print('Q1 VERIFICATION -- independent re-derivation from %s' % RAW)
    print('=' * 92)
    print('  banked rows            : %d   (expected %d)' % (len(rows), PRINTED['pairs']))
    if len(rows) != PRINTED['pairs']:
        fails.append('row count %d != %d' % (len(rows), PRINTED['pairs']))

    n = tot('steps')
    print('  steps analysed         : %s   (printed %s)' % (format(n, ','),
                                                            format(PRINTED['steps'], ',')))
    if n != PRINTED['steps']:
        fails.append('step total %d != printed %d' % (n, PRINTED['steps']))

    # ------------------------------------------------------------------ IDENTITY
    bad_id = []
    for x in rows:
        lhs = int(x['fired']) + int(x['late']) - int(x['spurious'])
        if lhs != int(x['would']):
            bad_id.append((x['seed'], x['repeat'], lhs, int(x['would'])))
    print('  IDENTITY  fired+late-spurious == would : %s (%d/%d rows)'
          % ('PASS' if not bad_id else 'FAIL', len(rows) - len(bad_id), len(rows)))
    if bad_id:
        fails.append('counter identity violated in %d row(s): %s' % (len(bad_id), bad_id[:3]))

    # ------------------------------------------------------------------ CLOSURE
    band_n = [sum(int(x['b%d_n' % i]) for x in rows) for i in range(len(VBANDS))]
    print('  CLOSURE   band steps sum to total       : %s (%s vs %s)'
          % ('PASS' if sum(band_n) == n else 'FAIL', format(sum(band_n), ','), format(n, ',')))
    if sum(band_n) != n:
        fails.append('velocity bands do not partition the steps: %d vs %d' % (sum(band_n), n))

    # ------------------------------------------------------------------ TOTALS
    calc = {
        'fired_pct': 100.0 * tot('fired') / n,
        'would_pct': 100.0 * tot('would') / n,
        'late_pct': 100.0 * tot('late') / n,
        'spurious_pct': 100.0 * tot('spurious') / n,
    }
    print('\n  %-14s %14s %14s %10s' % ('quantity', 'recomputed', 'printed', 'agree'))
    for k, dp in (('fired_pct', 2), ('would_pct', 2), ('late_pct', 3), ('spurious_pct', 3)):
        agree = abs(round(calc[k], dp) - PRINTED[k]) < 1e-9
        print('  %-14s %13.4f%% %13.*f%% %10s' % (k, calc[k], dp, PRINTED[k],
                                                  'yes' if agree else 'NO'))
        if not agree:
            fails.append('%s recomputed %.4f, printed %.3f' % (k, calc[k], PRINTED[k]))

    # ------------------------------------------------------------------ BANDS
    print('\n  %-12s %12s %12s %12s %10s' % ('band (m/s)', 'steps', 'guard on', 'LATE', 'agree'))
    band_late = []
    for i, (lo, hi) in enumerate(VBANDS):
        nb = band_n[i]
        if not nb:
            continue
        gp = 100.0 * sum(int(x['b%d_fired' % i]) for x in rows) / nb
        lp = 100.0 * sum(int(x['b%d_late' % i]) for x in rows) / nb
        band_late.append(lp)
        pn, pg, pl = PRINTED['bands'][i]
        agree = nb == pn and abs(round(gp, 2) - pg) < 1e-9 and abs(round(lp, 3) - pl) < 1e-9
        print('  %-12s %12s %11.2f%% %11.3f%% %10s'
              % ('%d-%d' % (lo, hi), format(nb, ','), gp, lp, 'yes' if agree else 'NO'))
        if not agree:
            fails.append('band %d-%d disagrees with the printed row' % (lo, hi))

    # ------------------------------------------------------------------ the two answers
    cs, cl = tot('cross_steps'), tot('cross_late')
    cross_pct = 100.0 * cl / cs
    other_steps, other_late = n - cs, tot('late') - cl
    other_pct = 100.0 * other_late / other_steps
    fs, fl = tot('fast_steps'), tot('fast_late')
    fast_pct = 100.0 * fl / fs
    slow_steps, slow_late = n - fs, tot('late') - fl
    slow_pct = 100.0 * slow_late / slow_steps

    print('\n  ' + '-' * 88)
    print('  THE TWO THINGS THE REVIEWER ASKED')
    print('  ' + '-' * 88)
    print('  shadow transition : LATE %6.3f%% on %s crossing steps'
          % (cross_pct, format(cs, ',')))
    print('                      LATE %6.3f%% on %s other steps'
          % (other_pct, format(other_steps, ',')))
    print('                      ratio %.2fx' % (cross_pct / other_pct))
    print('  velocity          : LATE %6.3f%% at >= 30 m/s (%s steps)'
          % (fast_pct, format(fs, ',')))
    print('                      LATE %6.3f%% below 30 m/s (%s steps)'
          % (slow_pct, format(slow_steps, ',')))
    print('                      ratio %.2fx' % (fast_pct / slow_pct))
    print('                      band monotone in velocity: %s'
          % all(band_late[i] <= band_late[i + 1] for i in range(len(band_late) - 1)))
    if abs(round(cross_pct, 3) - PRINTED['cross_late_pct']) > 1e-9:
        fails.append('crossing LATE recomputed %.3f, printed %.3f'
                     % (cross_pct, PRINTED['cross_late_pct']))
    if abs(round(fast_pct, 3) - PRINTED['fast_late_pct']) > 1e-9:
        fails.append('fast LATE recomputed %.3f, printed %.3f'
                     % (fast_pct, PRINTED['fast_late_pct']))

    # ------------------------------------------------------------------ DISPERSION
    print('\n  ' + '-' * 88)
    print('  DISPERSION of the LATE rate (pinned estimator, resolvability rule)')
    print('  ' + '-' * 88)
    sys.path.insert(0, '/home/claude/work')
    import tablevii_decompose as D
    print('  estimator md5 : %s' % md5('tablevii_decompose.py'))
    seeds = sorted({int(x['seed']) for x in rows})
    reps = sorted({int(x['repeat']) for x in rows})
    X = np.array([[100.0 * int(next(x for x in rows if int(x['seed']) == s
                                    and int(x['repeat']) == r)['late'])
                   / int(next(x for x in rows if int(x['seed']) == s
                              and int(x['repeat']) == r)['steps'])
                   for r in reps] for s in seeds])
    d = D.decompose(X)
    resolvable = bool(d['v_train'] > 0 and d['p'] < 0.05)
    print('  grid           : %d seeds x R=%d' % (len(seeds), len(reps)))
    print('  mean LATE      : %.4f%%' % d['grand'])
    print('  per-training   : %s' % ', '.join('%.4f' % v for v in d['seed_means']))
    print('  v_train        : %.6g   (moment estimator; negative means unresolvable outright)'
          % d['v_train'])
    print('  sd_train       : %s' % ('%.4f' % d['sd_train'] if d['v_train'] > 0 else 'n/a'))
    print('  F(%d,%d)        : %.4f   p = %.4g' % (len(seeds) - 1,
                                                   (len(seeds) - 1) * (len(reps) - 1),
                                                   d['f_obs'], d['p']))
    print('  sd_e (eval)    : %.4f' % d['sd_e'])
    print('  RESOLVABLE     : %s' % resolvable)
    if resolvable:
        print('  -> report as   : %.3f%% +- %.3f%% (between-training)' % (d['grand'], d['sd_train']))
    else:
        # The frozen rule: never render an unresolvable component as "+- 0.00". State the ceiling.
        ceil = d['sd_train'] if d['v_train'] > 0 else 0.0
        print('  -> report as   : %.3f%% (sigma_train < %.2f%%, not resolvable at k=%d, R=%d)'
              % (d['grand'], max(ceil, 0.01), len(seeds), len(reps)))

    # ------------------------------------------------------------------ write summary
    with open(OUT, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['scope', 'steps', 'guard_on_pct', 'late_pct', 'spurious_pct'])
        w.writerow(['all', n, calc['fired_pct'], calc['late_pct'], calc['spurious_pct']])
        for i, (lo, hi) in enumerate(VBANDS):
            nb = band_n[i]
            if not nb:
                continue
            w.writerow(['vel_%d_%d' % (lo, hi), nb,
                        100.0 * sum(int(x['b%d_fired' % i]) for x in rows) / nb,
                        100.0 * sum(int(x['b%d_late' % i]) for x in rows) / nb,
                        100.0 * sum(int(x['b%d_spur' % i]) for x in rows) / nb])
        w.writerow(['los_to_nlos_crossing', cs, '', cross_pct, ''])
        w.writerow(['non_crossing', other_steps, '', other_pct, ''])
        w.writerow(['vel_ge_30', fs, '', fast_pct, ''])
        w.writerow(['vel_lt_30', slow_steps, '', slow_pct, ''])
    print('\n  wrote %s' % OUT)

    print('\n' + '=' * 92)
    if fails:
        print('VERIFICATION: FAIL')
        for f_ in fails:
            print('  - ' + f_)
        return 3
    print('VERIFICATION: PASS -- every printed Q1 figure re-derives from the banked counts,')
    print('              the counter identity holds in all 20 rows, and the velocity bands')
    print('              partition the 906,868 analysed steps exactly.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
